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
    private Paint roiBorderPaint;
    private Paint robotDirPaint;
    private Paint detectedLinePaint;

    private Bitmap debugRoiBitmap;

    // --- Configuration ---
    private int scanY = 300;
    private int scanHeight = 30;
    private boolean isMirrored = false; // Controlled by checkbox

    // --- Thresholds ---
    private Scalar colorThrLow = new Scalar(20, 100, 100);
    private Scalar colorThrHi = new Scalar(30, 255, 255);
    private final double CONFIDENCE_THRESHOLD = 500;

    private Mat mat;
    private Mat matHsv;
    private Mat matMask;
    private Mat matSlice;
    private Mat matHist;

    private Matrix transformMatrix;
    private Bitmap processedBitmap;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentLineTrackingBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Initialize Paints
        roiBorderPaint = new Paint();
        roiBorderPaint.setColor(Color.YELLOW);
        roiBorderPaint.setStyle(Paint.Style.STROKE);
        roiBorderPaint.setStrokeWidth(5);

        robotDirPaint = new Paint();
        robotDirPaint.setColor(Color.GREEN);
        robotDirPaint.setStyle(Paint.Style.STROKE);
        robotDirPaint.setStrokeWidth(5);

        detectedLinePaint = new Paint();
        detectedLinePaint.setColor(Color.RED);
        detectedLinePaint.setStyle(Paint.Style.FILL);

        // Initialize OpenCV Mats
        mat = new Mat();
        matHsv = new Mat();
        matMask = new Mat();
        matSlice = new Mat();
        matHist = new Mat();

        // --- UI CONTROLS ---

        binding.autoSwitch.setChecked(isAutoMode);
        binding.autoSwitch.setOnClickListener(v -> setAutoMode(binding.autoSwitch.isChecked()));

        // Color Selector
        binding.colorSelectorGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.rb_yellow) {
                colorThrLow = new Scalar(20, 100, 100);
                colorThrHi = new Scalar(30, 255, 255);
            } else if (checkedId == R.id.rb_white) {
                colorThrLow = new Scalar(0, 0, 200);
                colorThrHi = new Scalar(180, 50, 255);
            } else if (checkedId == R.id.rb_black) {
                colorThrLow = new Scalar(0, 0, 0);
                colorThrHi = new Scalar(180, 255, 50);
            }
        });

        // Camera Toggle
        binding.cameraToggle.setOnClickListener(v -> {
            toggleCamera();
            transformMatrix = null; // Force matrix rebuild
        });

        // Mirror/Flip Toggle
        binding.mirrorControl.setOnClickListener(v -> {
            isMirrored = binding.mirrorControl.isChecked();
            transformMatrix = null; // Force matrix rebuild with new flip state
        });

        // Sliders
        binding.scanYText.setText(String.valueOf(scanY));
        binding.scanYSeekbar.setProgress(scanY);
        binding.scanYSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                scanY = progress;
                binding.scanYText.setText(String.valueOf(progress));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        binding.scanHeightText.setText(String.valueOf(scanHeight));
        binding.scanHeightSeekbar.setProgress(scanHeight);
        binding.scanHeightSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                scanHeight = progress;
                binding.scanHeightText.setText(String.valueOf(progress));
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
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    if (binding != null) {
                        binding.errorText.setText("Stopped");
                        binding.controllerContainer.controlInfo.setText("0,0");
                        binding.overlayImageView.setImageBitmap(null);
                    }
                });
            }
        }
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        if (binding == null || !isAutoMode) return;

        // 1. ROTATION & MIRRORING
        if (transformMatrix == null) {
            transformMatrix = new Matrix();
            // Rotate based on camera sensor
            transformMatrix.postRotate(getRotationDegrees());

            // Apply Mirror if checkbox is checked
            if (isMirrored) {
                transformMatrix.postScale(-1, 1);
            }
        }

        // Create the processed bitmap (Rotated + Mirrored)
        // This image is exactly what we will draw to screen AND process with OpenCV
        processedBitmap = Bitmap.createBitmap(image, 0, 0, image.getWidth(), image.getHeight(), transformMatrix, true);

        // 2. PREPARE OVERLAY CANVAS
        if (overlayBitmap == null || overlayBitmap.getWidth() != processedBitmap.getWidth() || overlayBitmap.getHeight() != processedBitmap.getHeight()) {
            overlayBitmap = Bitmap.createBitmap(processedBitmap.getWidth(), processedBitmap.getHeight(), Bitmap.Config.ARGB_8888);
            overlayCanvas = new Canvas(overlayBitmap);
        }
        overlayCanvas.drawColor(Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR);

        // 3. DEFINE ROI
        int w = processedBitmap.getWidth();
        int h = processedBitmap.getHeight();

        if (scanY + scanHeight > h) scanY = h - scanHeight;
        if (scanY < 0) scanY = 0;

        Rect roiRect = new Rect(0, scanY, w, scanY + scanHeight);

        // 4. OPENCV PROCESSING
        Utils.bitmapToMat(processedBitmap, mat);
        matSlice = new Mat(mat, new org.opencv.core.Rect(roiRect.left, roiRect.top, roiRect.width(), roiRect.height()));
        Imgproc.cvtColor(matSlice, matHsv, Imgproc.COLOR_RGB2HSV);
        Core.inRange(matHsv, colorThrLow, colorThrHi, matMask);

        // --- VISUAL DEBUGGING ---
        if (debugRoiBitmap == null || debugRoiBitmap.getWidth() != matMask.width() || debugRoiBitmap.getHeight() != matMask.height()) {
            debugRoiBitmap = Bitmap.createBitmap(matMask.width(), matMask.height(), Bitmap.Config.ARGB_8888);
        }
        Utils.matToBitmap(matMask, debugRoiBitmap);
        overlayCanvas.drawBitmap(debugRoiBitmap, null, roiRect, null);
        overlayCanvas.drawRect(roiRect, roiBorderPaint);

        // 5. CALCULATE LINE POSITION
        Core.reduce(matMask, matHist, 0, Core.REDUCE_SUM, CvType.CV_32S);
        Core.MinMaxLocResult mmr = Core.minMaxLoc(matHist);
        double maxVal = mmr.maxVal;
        int maxIdx = (int) mmr.maxLoc.x;

        Control driveCommand;
        float error = 0.0f;

        if (maxVal > CONFIDENCE_THRESHOLD) {
            // LINE FOUND
            float lineCenterX = (float) maxIdx;
            float lineCenterY = scanY + (scanHeight / 2.0f);
            overlayCanvas.drawCircle(lineCenterX, lineCenterY, 15, detectedLinePaint);

            float centerX = w / 2.0f;
            error = (lineCenterX - centerX) / (w / 2.0f);

            float turn = error * 1.5f;
            float leftSpeed = 0.8f + turn;
            float rightSpeed = 0.8f - turn;

            leftSpeed = Math.max(-1.0f, Math.min(1.0f, leftSpeed));
            rightSpeed = Math.max(-1.0f, Math.min(1.0f, rightSpeed));

            driveCommand = new Control(leftSpeed, rightSpeed);

            robotDirPaint.setColor(Color.GREEN);
            overlayCanvas.drawLine(centerX, lineCenterY, lineCenterX, lineCenterY, robotDirPaint);

        } else {
            // LINE LOST
            driveCommand = new Control(0.0f, 0.0f);
            robotDirPaint.setColor(Color.RED);
            overlayCanvas.drawLine(w/2f, scanY, w/2f, scanY+scanHeight, robotDirPaint);
        }

        vehicle.setControl(driveCommand);

        if (getActivity() != null) {
            final float finalError = error;
            float l = vehicle.getLeftSpeed();
            float r = vehicle.getRightSpeed();

            getActivity().runOnUiThread(() -> {
                if (binding != null) {
                    binding.overlayImageView.setImageBitmap(overlayBitmap);
                    binding.errorText.setText(String.format(Locale.US, "Err: %.2f", finalError));
                    binding.controllerContainer.controlInfo.setText(String.format(Locale.US, "%.1f,%.1f", l, r));
                }
            });
        }
    }

    @Override
    protected void processControllerKeyData(String command) {
        if (Constants.CMD_NETWORK.equals(command)) {
            setAutoMode(!binding.autoSwitch.isChecked());
        }
    }

    @Override
    protected void processUSBData(String data) {}

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (mat != null) mat.release();
        if (matHsv != null) matHsv.release();
        if (matMask != null) matMask.release();
        if (matSlice != null) matSlice.release();
        if (matHist != null) matHist.release();
        binding = null;
    }
}