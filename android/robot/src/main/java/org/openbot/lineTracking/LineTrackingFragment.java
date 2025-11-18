package org.openbot.lineTracking;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
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

// OpenCV Imports
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Locale;

public class LineTrackingFragment extends CameraFragment {
    private static final String TAG = "LineTrackingFragment";

    private FragmentLineTrackingBinding binding;
    private boolean isAutoMode = false;

    // --- Processing and Drawing Variables ---
    private Bitmap overlayBitmap;
    private Canvas overlayCanvas;
    private Paint roiPaint;
    private Paint robotDirPaint;
    private Paint detectedLinePaintError;

    // --- Color Tracking Variables ---
    private int scanY = 480;
    private int scanHeight = 20;

    // TUNE THIS: HSV Color Thresholds for the line (e.g., yellow)
    private Scalar colorThrLow = new Scalar(20, 100, 100);  // Yellow
    private Scalar colorThrHi = new Scalar(30, 255, 255); // Yellow

    private int targetPixel = 0;
    private final double CONFIDENCE_THRESHOLD = 500;

    // --- PID VARIABLES REMOVED ---
    // No Kp, Ki, Kd, previousError, or integralError needed.

    // --- OpenCV Mats ---
    private Mat mat;
    private Mat matGray;
    private Mat matHsv;
    private Mat matMask;
    private Mat matSlice;
    private Mat matHist;

    private Matrix rotationMatrix;
    private Bitmap rotatedBitmap;


    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentLineTrackingBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        Log.d(TAG, String.valueOf(getRotationDegrees()));

        // Initialize Paint objects
        roiPaint = new Paint();
        roiPaint.setColor(Color.BLUE);
        roiPaint.setStyle(Paint.Style.STROKE);
        roiPaint.setStrokeWidth(3);
        roiPaint.setAlpha(100);

        robotDirPaint = new Paint();
        robotDirPaint.setColor(Color.GREEN);
        robotDirPaint.setStyle(Paint.Style.STROKE);
        robotDirPaint.setStrokeWidth(5);
        robotDirPaint.setPathEffect(new android.graphics.DashPathEffect(new float[]{10, 10}, 0));

        detectedLinePaintError = new Paint();
        detectedLinePaintError.setColor(Color.RED);
        detectedLinePaintError.setStyle(Paint.Style.STROKE);
        detectedLinePaintError.setStrokeWidth(8);

        // Initialize OpenCV Mats
        mat = new Mat();
        matGray = new Mat();
        matHsv = new Mat();
        matMask = new Mat();
        matSlice = new Mat();
        matHist = new Mat();

        // Autopilot Switch
        binding.autoSwitch.setChecked(isAutoMode);
        binding.autoSwitch.setOnClickListener(v -> setAutoMode(binding.autoSwitch.isChecked()));

        // --- Link Scan Y Slider ---
        binding.scanYText.setText(String.valueOf(scanY));
        binding.scanYSeekbar.setProgress(scanY);
        binding.scanYSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    scanY = progress;
                    binding.scanYText.setText(String.valueOf(progress));
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // --- Link Scan Height Slider ---
        binding.scanHeightText.setText(String.valueOf(scanHeight));
        binding.scanHeightSeekbar.setProgress(scanHeight);
        binding.scanHeightSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    scanHeight = progress;
                    binding.scanHeightText.setText(String.valueOf(progress));
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });


        rotationMatrix = new Matrix();
        rotationMatrix.postRotate(90);
    }

    private void setAutoMode(boolean isEnabled) {
        isAutoMode = isEnabled;
        binding.autoSwitch.setChecked(isEnabled);
        if (!isAutoMode) {
            vehicle.setControl(new Control(0.0f, 0.0f));
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

        // --- 1. ROTATE THE BITMAP ---
        rotatedBitmap = Bitmap.createBitmap(image, 0, 0, image.getWidth(), image.getHeight(), rotationMatrix, true);

        // --- 2. Initialize Overlay Canvas ---
        if (overlayBitmap == null || overlayBitmap.getWidth() != rotatedBitmap.getWidth() || overlayBitmap.getHeight() != rotatedBitmap.getHeight()) {
            overlayBitmap = Bitmap.createBitmap(rotatedBitmap.getWidth(), rotatedBitmap.getHeight(), Bitmap.Config.ARGB_8888);
            overlayCanvas = new Canvas(overlayBitmap);
        }
        overlayCanvas.drawColor(Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR);

        // --- 3. Define Scan Area (on the rotated 480x640 image) ---
        int w = rotatedBitmap.getWidth();  // e.g., 480
        int h = rotatedBitmap.getHeight(); // e.g., 640

        // targetPixel is no longer needed, we just use the center (w / 2)
        float centerX = w / 2.0f;

        if (scanY + scanHeight > h) scanY = h - scanHeight;
        if (scanY < 0) scanY = 0;

        Rect roiRect = new Rect(0, scanY, w, scanY + scanHeight);
        overlayCanvas.drawRect(roiRect, roiPaint);
        overlayCanvas.drawLine(centerX, 0, centerX, h, robotDirPaint);

        // --- 4. OpenCV Processing (Color Tracking) ---
        Utils.bitmapToMat(rotatedBitmap, mat);
        matSlice = new Mat(mat, new org.opencv.core.Rect(roiRect.left, roiRect.top, roiRect.width(), roiRect.height()));
        Imgproc.cvtColor(matSlice, matHsv, Imgproc.COLOR_RGB2HSV);
        Core.inRange(matHsv, colorThrLow, colorThrHi, matMask);
        Core.reduce(matMask, matHist, 0, Core.REDUCE_SUM, CvType.CV_32S);

        // --- 5. Find Line Center ---
        Core.MinMaxLocResult mmr = Core.minMaxLoc(matHist);
        int maxYellowIdx = (int) mmr.maxLoc.x; // X-coordinate of the line center
        double maxYellowVal = mmr.maxVal; // This is our "confidence"

        Control driveCommand;
        float leftControl = 0.0f;
        float rightControl = 0.0f;

        if (maxYellowVal > CONFIDENCE_THRESHOLD) {
            // We see the line.

            // 1. Normalize the line's position to a -1.0 to +1.0 scale
            float x_pos_norm = 1.0f - 2.0f * maxYellowIdx / w;

            // --- NEW: NON-LINEAR STEERING ---
            // Apply an exponent to the error.
            // A higher number = more sensitive at the edges, less sensitive at the center.
            // (Must be an odd number to keep the sign correct, e.g., 3, 5)
            float sensitivity = 3.0f; // TUNE THIS. 1.0 is linear. 3.0 is very curved.
            float x_pos_scaled = (float) (Math.signum(x_pos_norm) * Math.pow(Math.abs(x_pos_norm), sensitivity));
            // --- END NEW LOGIC ---


            // 2. Apply "tank steer" logic using the NEW scaled value
            if (x_pos_scaled < 0) {
                // Line is to the RIGHT (negative norm)
                leftControl = 1.0f;
                rightControl = 1.0f + x_pos_scaled; // x_pos_scaled is negative
            } else {
                // Line is to the LEFT (positive norm)
                leftControl = 1.0f - x_pos_scaled; // x_pos_scaled is positive
                rightControl = 1.0f;
            }

            // (Optional) Apply dynamic speed
            float speed = 1.0f;
            leftControl *= speed;
            rightControl *= speed;

            // Draw the detected line center on our overlay
            overlayCanvas.drawLine((float) maxYellowIdx, (float) scanY, (float) maxYellowIdx, (float) (scanY + scanHeight), detectedLinePaintError);

            // Set the final command
            driveCommand = new Control(leftControl, rightControl);

            // Update error text for debugging (show the scaled error)
            if (getActivity() != null) {
                final float finalError = x_pos_scaled;
                getActivity().runOnUiThread(() -> {
                    if (binding != null) {
                        binding.errorText.setText(String.format(Locale.US, "Error: %.2f", finalError));
                    }
                });
            }

        } else {
            // --- LINE LOST! ---
            // (Same as before)
            driveCommand = new Control(0.0f, 0.0f);
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    if (binding != null) {
                        binding.errorText.setText("Line Lost");
                    }
                });
            }
        }

        // --- 7. Update UI and Send Command ---
        vehicle.setControl(driveCommand);

        if (getActivity() != null) {
            // Get the actual speeds sent to the vehicle
            float left = vehicle.getLeftSpeed();
            float right = vehicle.getRightSpeed();

            getActivity().runOnUiThread(() -> {
                if (binding != null) {
                    binding.controllerContainer.controlInfo.setText(
                            String.format(Locale.US, "%.0f,%.0f", left, right));
                    binding.overlayImageView.setImageBitmap(overlayBitmap);
                }
            });
        }
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
        if (mat != null) mat.release();
        if (matGray != null) matGray.release();
        if (matHsv != null) matHsv.release();
        if (matMask != null) matMask.release();
        if (matSlice != null) matSlice.release();
        if (matHist != null) matHist.release();
        binding = null;
    }
}