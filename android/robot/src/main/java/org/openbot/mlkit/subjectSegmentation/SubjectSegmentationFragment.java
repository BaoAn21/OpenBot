package org.openbot.mlkit.subjectSegmentation;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar; // IMPORT THIS

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenter;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions;

import org.openbot.R; // IMPORT THIS
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentSubjectSegmentationBinding;
import org.openbot.utils.Constants;
import org.openbot.utils.Enums; // IMPORT THIS
import org.openbot.vehicle.Control; // IMPORT THIS

public class SubjectSegmentationFragment extends CameraFragment {
    private String TAG = "SubjectSegmentationFragment";

    private SubjectSegmenter segmenter;
    private FragmentSubjectSegmentationBinding binding;
    private SubjectSegmenterOptions segmenterOptions; // Make this a class variable

    // --- NEW VARIABLES ---
    private int closenessThreshold = 20;
    private boolean isAutoMode = false;
    private Enums.SpeedMode currentSpeedMode;
    // ---------------------

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentSubjectSegmentationBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Build the options, but don't create the client yet
        segmenterOptions = new SubjectSegmenterOptions.Builder()
                .enableForegroundBitmap()
                .build();

        // --- SETUP NEW BUTTONS ---

        // Autopilot Switch
        binding.autoSwitch.setChecked(isAutoMode);
        binding.autoSwitch.setOnClickListener(v -> {
            setAutoMode(binding.autoSwitch.isChecked());
        });

        // Speed Mode Button
        currentSpeedMode = Enums.SpeedMode.getByID(preferencesManager.getSpeedMode());
        setSpeedModeUI(currentSpeedMode);
        binding.speedMode.setOnClickListener(v -> {
            currentSpeedMode = Enums.toggleSpeed(Enums.Direction.CYCLIC.getValue(), currentSpeedMode);
            setSpeedModeUI(currentSpeedMode);
            // Save the new speed setting
            preferencesManager.setSpeedMode(currentSpeedMode.getValue());
            vehicle.setSpeedMultiplier(currentSpeedMode.getValue());
        });

        // --- SEEKBAR LOGIC ---
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
        // ---------------------
    }

    // --- NEW HELPER METHODS ---

    private void setAutoMode(boolean isEnabled) {
        isAutoMode = isEnabled;
        binding.autoSwitch.setChecked(isEnabled);

        // If we are turning off auto-mode, stop the vehicle
        if (!isAutoMode) {
            vehicle.setControl(new Control(0.0f, 0.0f));
        }
    }

    private void setSpeedModeUI(Enums.SpeedMode speedMode) {
        if (speedMode == null) return;
        switch (speedMode) {
            case SLOW:
                binding.speedMode.setImageResource(R.drawable.ic_speed_low);
                break;
            case NORMAL:
                binding.speedMode.setImageResource(R.drawable.ic_speed_medium);
                break;
            case FAST:
                binding.speedMode.setImageResource(R.drawable.ic_speed_high);
                break;
        }
    }
    // -------------------------


    // --- UPDATED LIFECYCLE METHODS ---

    @Override
    public void onResume() {
        super.onResume();
        // Create the segmenter client when the fragment is active
        if (segmenter == null && segmenterOptions != null) {
            segmenter = SubjectSegmentation.getClient(segmenterOptions);
            Log.d(TAG, "Segmenter created.");
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        // Stop the vehicle when pausing
        vehicle.setControl(new Control(0.0f, 0.0f));

        // Close the segmenter to stop all processing
        if (segmenter != null) {
            segmenter.close();
            segmenter = null;
            Log.d(TAG, "Segmenter closed.");
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Safety check to close segmenter
        if (segmenter != null) {
            segmenter.close();
            segmenter = null;
        }
        binding = null; // Clean up binding
    }
    // -------------------------------


    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        // --- ADDED NULL CHECKS AND AUTOPILOT CHECK ---
        if (segmenter == null || image == null || binding == null) {
            return;
        }

        // If Autopilot is OFF, do nothing.
        // This allows manual controller drive to work.
        if (!isAutoMode) {
            // We should also clear the mask
            if (binding.maskImageView.getDrawable() != null) {
                getActivity().runOnUiThread(() -> binding.maskImageView.setImageBitmap(null));
            }
            return;
        }
        // ---------------------------------------------

        int rotation = getRotationDegrees();
        InputImage inputImage = InputImage.fromBitmap(image, rotation);

        segmenter.process(inputImage)
                .addOnSuccessListener(
                        result -> {
                            // Check if binding is null (fragment is gone) or if auto-mode was turned off
                            if (binding == null || getActivity() == null || !isAutoMode) {
                                return;
                            }

                            Bitmap foregroundMask = result.getForegroundBitmap();
                            if (foregroundMask == null) {
                                // No object detected, so GO FORWARD
                                vehicle.setControl(new Control(1.0f, 1.0f));
                                // Clear the mask
                                getActivity().runOnUiThread(() -> binding.maskImageView.setImageBitmap(null));
                                return;
                            }

                            // --- 1. AREA CALCULATION ---
                            int width = foregroundMask.getWidth();
                            int height = foregroundMask.getHeight();
                            long totalCameraArea = (long) width * height;
                            long foregroundAreaInPixels = 0;

                            int[] pixels = new int[width * height];
                            foregroundMask.getPixels(pixels, 0, width, 0, 0, width, height);

                            for (int pixel : pixels) {
                                if (android.graphics.Color.alpha(pixel) > 0) {
                                    foregroundAreaInPixels++;
                                }
                            }
                            double areaPercentage = 0.0;
                            if (totalCameraArea > 0) {
                                areaPercentage = (foregroundAreaInPixels / (double) totalCameraArea) * 100.0;
                            }
                            Log.d(TAG, String.format("Object covers: %.2f%% (Threshold: %d%%)", areaPercentage, closenessThreshold));

                            // --- 2. AUTOPILOT LOGIC & COLOR ---
                            final int maskColor;
                            if (areaPercentage >= closenessThreshold) {
                                // Object is "close enough" - STOP
                                vehicle.setControl(new Control(0.0f, 0.0f));
                                maskColor = android.graphics.Color.argb(150, 0, 255, 0); // GREEN
                            } else {
                                // Object is "too far" - GO FORWARD
                                vehicle.setControl(new Control(1.0f, 1.0f));
                                maskColor = android.graphics.Color.argb(150, 255, 0, 0); // RED
                            }
                            // ----------------------------------

                            // --- 3. ROTATE FOR DISPLAY ---
                            Matrix matrix = new Matrix();
                            matrix.postRotate(90);
                            Bitmap rotatedMask = Bitmap.createBitmap(
                                    foregroundMask, 0, 0,
                                    foregroundMask.getWidth(), foregroundMask.getHeight(),
                                    matrix, true
                            );

                            // --- 4. UPDATE UI ---
                            getActivity().runOnUiThread(() -> {
                                if (binding != null) {
                                    binding.maskImageView.setImageBitmap(rotatedMask);
                                    binding.maskImageView.setColorFilter(
                                            maskColor,
                                            android.graphics.PorterDuff.Mode.SRC_IN
                                    );
                                }
                            });
                        })
                .addOnFailureListener(
                        e -> {
                            if (binding != null) {
                                Log.e(TAG, "Segmentation failed", e);
                            }
                        });
    }

    @Override
    protected void processControllerKeyData(String command) {
        // Not used in this fragment
        if (command == null) return;
        switch (command) {
            case Constants.CMD_NETWORK: // Assuming "NETWORK" is the command for your toggle button
                // Toggle the autopilot state
                setAutoMode(!binding.autoSwitch.isChecked());
                break;

            // Add cases for other controller commands if needed in this fragment
            // case Constants.CMD_SPEED_UP:
            //    break;
        }
    }

    @Override
    protected void processUSBData(String data) {
        // Not used in this fragment
    }
}