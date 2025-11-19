package org.openbot.mlkit.subjectSegmentation;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageProxy;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenter;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions;

import org.openbot.R;
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentSubjectSegmentationBinding;
import org.openbot.utils.Constants;
import org.openbot.utils.Enums;
import org.openbot.vehicle.Control;

import java.util.Locale;
import java.util.Random;

public class SubjectSegmentationFragment extends CameraFragment {
    private String TAG = "SubjectSegmentationFragment";

    private SubjectSegmenter segmenter;
    private boolean isProcessing = false;
    private FragmentSubjectSegmentationBinding binding;
    private SubjectSegmenterOptions segmenterOptions;

    private int closenessThreshold = 20;
    private boolean isAutoMode = false;
    private Enums.SpeedMode currentSpeedMode;
    private Random random = new Random();
    private int currentZone = 0;
    private boolean isTurningRight = false;

    // --- OVERLAY & CONTROL VARIABLES ---
    private Paint paint;
    private Paint borderPaint;
    private Bitmap rotatedMask = null;
    private Matrix frameToViewTransform = null;
    private RectF maskDestinationRect = null;
    private int maskColor = 0;

    // Manual Control Flags
    private boolean isMirrored = false;

    private static final float AUTONOMOUS_SPEED_SCALE = 1.0f;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentSubjectSegmentationBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // 1. Setup Paints
        paint = new Paint();
        paint.setAlpha(160);
        paint.setFilterBitmap(true);

        borderPaint = new Paint();
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(5f);
        borderPaint.setColor(Color.YELLOW);

        segmenterOptions = new SubjectSegmenterOptions.Builder()
                .enableForegroundBitmap()
                .build();

        // --- CONTROLS SETUP ---

        // Autopilot Switch
        binding.autoSwitch.setChecked(isAutoMode);
        binding.autoSwitch.setOnClickListener(v -> setAutoMode(binding.autoSwitch.isChecked()));

        // Camera Toggle
        binding.cameraToggle.setOnClickListener(v -> {
            toggleCamera(); // Helper from CameraFragment base class
            // Reset matrix when camera changes
            frameToViewTransform = null;
        });

        isMirrored = binding.mirrorControl.isChecked();
        binding.mirrorControl.setOnClickListener(v -> {
            isMirrored = binding.mirrorControl.isChecked();
            // Reset matrix to trigger recalculation with new flip setting
            frameToViewTransform = null;
        });

        // Speed Button
        currentSpeedMode = Enums.SpeedMode.getByID(preferencesManager.getSpeedMode());
        setSpeedModeUI(currentSpeedMode);
        binding.speedMode.setOnClickListener(v -> {
            currentSpeedMode = Enums.toggleSpeed(Enums.Direction.CYCLIC.getValue(), currentSpeedMode);
            setSpeedModeUI(currentSpeedMode);
            preferencesManager.setSpeedMode(currentSpeedMode.getValue());
            vehicle.setSpeedMultiplier(currentSpeedMode.getValue());
        });

        // Threshold Seekbar
        binding.thresholdValueText.setText(closenessThreshold + "%");
        binding.thresholdSeekbar.setProgress(closenessThreshold);
        binding.thresholdSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    closenessThreshold = progress;
                    binding.thresholdValueText.setText(progress + "%");
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // --- OVERLAY DRAWING CALLBACK ---
        binding.maskOverlay.addCallback(
                canvas -> {
                    if (rotatedMask != null && frameToViewTransform != null) {
                        // Draw Mask
                        paint.setColorFilter(new PorterDuffColorFilter(maskColor, PorterDuff.Mode.SRC_IN));
                        canvas.drawBitmap(rotatedMask, frameToViewTransform, paint);

                        // Draw Debug Border (The "Square Box") to see alignment
                        if (maskDestinationRect != null) {
                            canvas.drawRect(maskDestinationRect, borderPaint);
                        }
                    }
                });
    }

    private void setAutoMode(boolean isEnabled) {
        isAutoMode = isEnabled;
        binding.autoSwitch.setChecked(isEnabled);
        if (!isAutoMode) {
            vehicle.setControl(new Control(0.0f, 0.0f));
            rotatedMask = null;
            maskDestinationRect = null;
            binding.maskOverlay.postInvalidate();
        }
    }

    private void setSpeedModeUI(Enums.SpeedMode speedMode) {
        if (speedMode == null) return;
        switch (speedMode) {
            case SLOW: binding.speedMode.setImageResource(R.drawable.ic_speed_low); break;
            case NORMAL: binding.speedMode.setImageResource(R.drawable.ic_speed_medium); break;
            case FAST: binding.speedMode.setImageResource(R.drawable.ic_speed_high); break;
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (segmenter == null && segmenterOptions != null) {
            segmenter = SubjectSegmentation.getClient(segmenterOptions);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        vehicle.setControl(new Control(0.0f, 0.0f));
        if (segmenter != null) {
            segmenter.close();
            segmenter = null;
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (segmenter != null) {
            segmenter.close();
            segmenter = null;
        }
        binding = null;
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        if (segmenter == null || image == null || binding == null) return;
        if (!isAutoMode) return;
        if (isProcessing) return;

        isProcessing = true;
        final long startTime = SystemClock.elapsedRealtime();

        // Get correct rotation so ML Kit gives us an upright image
        int rotation = imageProxy.getImageInfo().getRotationDegrees();
        InputImage inputImage = InputImage.fromBitmap(image, rotation);

        segmenter.process(inputImage)
                .addOnSuccessListener(result -> {
                    isProcessing = false;

                    if (binding == null || getActivity() == null || !isAutoMode) return;

                    long inferenceTime = SystemClock.elapsedRealtime() - startTime;
                    Bitmap foregroundMask = result.getForegroundBitmap();

                    if (foregroundMask == null) return;

                    // --- AUTOPILOT LOGIC ---
                    int width = foregroundMask.getWidth();
                    int height = foregroundMask.getHeight();

                    // Simple pixel counting
                    long foregroundPixels = 0;
                    int[] pixels = new int[width * height];
                    foregroundMask.getPixels(pixels, 0, width, 0, 0, width, height);
                    for (int i = 0; i < pixels.length; i+=10) {
                        if (android.graphics.Color.alpha(pixels[i]) > 0) foregroundPixels++;
                    }
                    double areaPercentage = (foregroundPixels * 10.0 / (width * height)) * 100.0;

                    Control driveCommand;
                    int newZone = 0;
                    if (areaPercentage >= closenessThreshold) newZone = 1;
                    if (newZone != currentZone && newZone == 1) isTurningRight = random.nextBoolean();
                    currentZone = newZone;

                    if (currentZone == 1) { // STOP/ROTATE
                        driveCommand = isTurningRight ?
                                new Control(AUTONOMOUS_SPEED_SCALE, -AUTONOMOUS_SPEED_SCALE) :
                                new Control(-AUTONOMOUS_SPEED_SCALE, AUTONOMOUS_SPEED_SCALE);
                        this.maskColor = android.graphics.Color.argb(200, 255, 255, 0); // Yellow
                    } else { // FORWARD
                        driveCommand = new Control(AUTONOMOUS_SPEED_SCALE * 0.5f, AUTONOMOUS_SPEED_SCALE * 0.5f);
                        this.maskColor = android.graphics.Color.argb(200, 255, 0, 0); // Red
                    }
                    vehicle.setControl(driveCommand);

                    // --- DISPLAY LOGIC ---
                    if (getActivity() != null && binding.maskOverlay.getWidth() > 0) {

                        float left = vehicle.getLeftSpeed();
                        float right = vehicle.getRightSpeed();
                        getActivity().runOnUiThread(() -> {
                            if (binding != null && binding.controllerContainer != null) {
                                binding.controllerContainer.controlInfo.setText(
                                        String.format(Locale.US, "%.0f,%.0f", left, right));
                            }
                        });

                        // --- MANUAL MATRIX ---
                        if (frameToViewTransform == null || rotatedMask != foregroundMask) {

                            frameToViewTransform = new Matrix();
                            float maskW = foregroundMask.getWidth();
                            float maskH = foregroundMask.getHeight();
                            float viewW = binding.maskOverlay.getWidth();
                            float viewH = binding.maskOverlay.getHeight();

                            // 1. DEFINE RECTANGLES
                            RectF src = new RectF(0, 0, maskW, maskH);
                            RectF dst = new RectF(0, 0, viewW, viewH);

                            // 2. MAP SOURCE TO DESTINATION
                            // Matrix.ScaleToFit.CENTER matches your CameraFragment's FIT_CENTER logic.
                            // This sets the base scale.
                            frameToViewTransform.setRectToRect(src, dst, Matrix.ScaleToFit.CENTER);

                            // 3. MANUAL MIRRORING (Applied AFTER the mapping)
                            // This ensures the flip happens on top of the scaling, not replaced by it.
                            if (isMirrored) {
                                // Flip horizontally (-1 scale) around the CENTER of the VIEW (viewW / 2)
                                frameToViewTransform.postScale(-1, 1, viewW / 2f, viewH / 2f);
                            }

                            // Update Debug Box
                            // We map the src rect through the matrix to see where it lands
                            maskDestinationRect = new RectF(src);
                            frameToViewTransform.mapRect(maskDestinationRect);
                        }

                        this.rotatedMask = foregroundMask;

                        getActivity().runOnUiThread(() -> {
                            if (binding != null) {
                                binding.maskOverlay.postInvalidate();
                                binding.inferenceInfo.setText(inferenceTime + " ms (" + (int)areaPercentage + "%)");
                            }
                        });
                    }
                })
                .addOnFailureListener(e -> {
                    isProcessing = false;
                    Log.e(TAG, "Segmentation failed", e);
                });
    }

    @Override
    protected void processControllerKeyData(String command) {
        if (command != null && command.equals(Constants.CMD_NETWORK)) {
            setAutoMode(!binding.autoSwitch.isChecked());
        }
    }

    @Override
    protected void processUSBData(String data) {}
}