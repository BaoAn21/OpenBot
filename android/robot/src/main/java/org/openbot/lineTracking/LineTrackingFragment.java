package org.openbot.lineTracking; // Make sure this package is correct

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

import org.openbot.R;
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentLineTrackingBinding;
import org.openbot.utils.Constants;
import org.openbot.vehicle.Control;

// OpenCV Imports - ASSUMES OPENCV IS IN YOUR PROJECT
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Locale;

public class LineTrackingFragment extends CameraFragment {
    private static final String TAG = "LineTrackingFragment";

    private FragmentLineTrackingBinding binding;
    private boolean isAutoMode = false;

    // --- Processing and Drawing Variables ---
    private int roiWidthPercent = 50; // 50% of screen width
    private int angleTolerance = 20;  // ±20 degrees from vertical

    private Bitmap overlayBitmap;
    private Canvas overlayCanvas;

    private Paint roiPaint;
    private Paint robotDirPaint;
    private Paint detectedLinePaint;
    private Paint detectedLinePaintError;

    // OpenCV Mats - allocated once to be efficient
    private Mat mat;
    private Mat matGray;
    private Mat matCanny;
    private Mat houghLines;


    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        binding = FragmentLineTrackingBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Initialize Paint objects
        roiPaint = new Paint();
        roiPaint.setColor(Color.BLUE);
        roiPaint.setStyle(Paint.Style.STROKE);
        roiPaint.setStrokeWidth(3);
        roiPaint.setAlpha(100); // Semi-transparent

        robotDirPaint = new Paint();
        robotDirPaint.setColor(Color.GREEN);
        robotDirPaint.setStyle(Paint.Style.STROKE);
        robotDirPaint.setStrokeWidth(5);
        robotDirPaint.setPathEffect(new android.graphics.DashPathEffect(new float[]{10, 10}, 0));

        detectedLinePaint = new Paint();
        detectedLinePaint.setColor(Color.YELLOW);
        detectedLinePaint.setStyle(Paint.Style.STROKE);
        detectedLinePaint.setStrokeWidth(8);

        detectedLinePaintError = new Paint();
        detectedLinePaintError.setColor(Color.RED);
        detectedLinePaintError.setStyle(Paint.Style.STROKE);
        detectedLinePaintError.setStrokeWidth(8);

        // Initialize OpenCV Mats
        mat = new Mat();
        matGray = new Mat();
        matCanny = new Mat();

        // Autopilot Switch
        binding.autoSwitch.setChecked(isAutoMode);
        binding.autoSwitch.setOnClickListener(v -> setAutoMode(binding.autoSwitch.isChecked()));

        // ROI Width Slider
        binding.roiWidthText.setText(roiWidthPercent + "%");
        binding.roiWidthSeekbar.setProgress(roiWidthPercent);
        binding.roiWidthSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    roiWidthPercent = progress;
                    binding.roiWidthText.setText(progress + "%");
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Angle Tolerance Slider
        binding.angleToleranceText.setText("±" + angleTolerance + "°");
        binding.angleToleranceSeekbar.setProgress(angleTolerance);
        binding.angleToleranceSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    angleTolerance = progress;
                    binding.angleToleranceText.setText("±" + progress + "°");
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
    }

    private void setAutoMode(boolean isEnabled) {
        isAutoMode = isEnabled;
        binding.autoSwitch.setChecked(isEnabled);
        if (!isAutoMode) {
            vehicle.setControl(new Control(0.0f, 0.0f));
            // Reset UI when turning off
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    if (binding != null) {
                        binding.errorText.setText("");
                        binding.controllerContainer.controlInfo.setText("0,0");
                        binding.overlayImageView.setImageBitmap(null);
                    }
                });
            }
        }
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        if (binding == null || !isAutoMode) {
            return;
        }

        // --- 1. Initialize Overlay Canvas ---
        // We re-create this each time to match the input image size
        if (overlayBitmap == null || overlayBitmap.getWidth() != image.getWidth() || overlayBitmap.getHeight() != image.getHeight()) {
            overlayBitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
            overlayCanvas = new Canvas(overlayBitmap);
        }
        // Clear the canvas for this frame
        overlayCanvas.drawColor(Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR);

        // --- 2. Define ROI ---
        int w = image.getWidth();
        int h = image.getHeight();
        int roiWidth = w * roiWidthPercent / 100;
        int roiLeft = (w - roiWidth) / 2;
        Rect roiRect = new Rect(roiLeft, 0, roiLeft + roiWidth, h);

        // Draw ROI lines on overlay
        overlayCanvas.drawRect(roiRect, roiPaint);

        // Draw Robot Direction line on overlay
        float centerX = w / 2.0f;
        overlayCanvas.drawLine(centerX, 0, centerX, h, robotDirPaint);

        // --- 3. OpenCV Processing ---
        Utils.bitmapToMat(image, mat); // Convert bitmap to Mat

        // Crop to Region of Interest
        Mat roiMat = new Mat(mat, new org.opencv.core.Rect(roiRect.left, roiRect.top, roiRect.width(), roiRect.height()));

        // Convert to Grayscale
        Imgproc.cvtColor(roiMat, matGray, Imgproc.COLOR_RGB2GRAY);

        // Canny Edge Detection
        Imgproc.Canny(matGray, matCanny, 50, 150);

        // Hough Line Transform
        houghLines = new Mat();
        Imgproc.HoughLinesP(matCanny, houghLines, 1, Math.PI / 180, 50, 50, 10);

        // --- 4. Filter Lines & Calculate Average ---
        double avgLineX = 0;
        int lineCount = 0;
        double minAngle = 90.0 - angleTolerance;
        double maxAngle = 90.0 + angleTolerance;

        for (int i = 0; i < houghLines.rows(); i++) {
            double[] line = houghLines.get(i, 0);
            double x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];

            // Calculate angle
            double angle = Math.abs(Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI);

            // Filter by angle (keep "vertical-ish" lines)
            if (angle > minAngle && angle < maxAngle) {
                lineCount++;
                avgLineX += (x1 + x2) / 2.0;

                // Draw the line on the overlay (relative to ROI)
                overlayCanvas.drawLine((float)(roiRect.left + x1), (float)y1, (float)(roiRect.left + x2), (float)y2, detectedLinePaint);
            }
        }

        // --- 5. Calculate Error and Steer ---
        Control driveCommand;
        double error = 0;

        if (lineCount > 0) {
            // Find the average X position of the "good" lines
            double finalAvgX = avgLineX / lineCount;
            double detectedLineScreenX = roiRect.left + finalAvgX;

            // Calculate error: difference between detected line and robot's center
            error = detectedLineScreenX - centerX;

            // Draw the final "average" detected line
            overlayCanvas.drawLine((float)detectedLineScreenX, 0, (float)detectedLineScreenX, h, detectedLinePaintError);

            // Steering Logic (P-Controller)
            final double STEERING_SENSITIVITY = 0.5; // Adjust this to make steering more/less aggressive
            float turn = (float)(error / (w / 2.0)) * (float)STEERING_SENSITIVITY;

            // Simple P-Controller: Go forward, and add/subtract "turn"
            float speed = 1.0f; // Autonomous speed scale
            float left = speed - turn;
            float right = speed + turn;

            // Clamp values between -1.0 and 1.0
            left = Math.max(-1.0f, Math.min(1.0f, left));
            right = Math.max(-1.0f, Math.min(1.0f, right));

            driveCommand = new Control(left, right);

        } else {
            // No lines detected: STOP
            driveCommand = new Control(0.0f, 0.0f);
        }

        // --- 6. Update UI and Send Command ---
        vehicle.setControl(driveCommand);

        if (getActivity() != null) {
            float left = vehicle.getLeftSpeed();
            float right = vehicle.getRightSpeed();
            final double finalError = error;

            getActivity().runOnUiThread(() -> {
                if (binding != null) {
                    // Update error text
                    binding.errorText.setText(String.format(Locale.US, "Error: %.0f", finalError));
                    // Update motor command text
                    binding.controllerContainer.controlInfo.setText(
                            String.format(Locale.US, "%.0f,%.0f", left, right));
                    // Set the new overlay bitmap
                    binding.overlayImageView.setImageBitmap(overlayBitmap);
                }
            });
        }

        // Release Mats to prevent memory leaks
        roiMat.release();
        houghLines.release();
    }

    @Override
    protected void processUSBData(String data) {
        if (binding != null && binding.controllerContainer != null) {
            binding.controllerContainer.speedInfo.setText(
                    getString(
                            R.string.speedInfo,
                            String.format(
                                    Locale.US, "%3.0f,%3.0f", vehicle.getLeftWheelRpm(), vehicle.getRightWheelRpm())));
        }
    }

    @Override
    protected void processControllerKeyData(String command) {
        if (command == null || binding == null) return;
        switch (command) {
            case Constants.CMD_NETWORK: // Toggle autopilot
                setAutoMode(!binding.autoSwitch.isChecked());
                break;
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Release OpenCV Mats when view is destroyed
        if (mat != null) mat.release();
        if (matGray != null) matGray.release();
        if (matCanny != null) matCanny.release();

        binding = null; // Clean up binding
    }
}