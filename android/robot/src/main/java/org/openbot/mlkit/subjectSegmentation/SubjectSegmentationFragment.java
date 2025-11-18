package org.openbot.mlkit.subjectSegmentation;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
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

    // --- NEW VARIABLES ---
    private int closenessThreshold = 20;
    private boolean isAutoMode = false;
    private Enums.SpeedMode currentSpeedMode;
    private Random random = new Random();
    private int currentZone = 0;
    // This will store our "latched" turn direction
    private boolean isTurningRight = false;
    // ---------------------

    // private long lastProcessingTimeMs = 0; // Removed

    // ---- AUTONOMOUS SPEED --- //
    // Set the speed for all autonomous actions (forward, turn, reverse)
    private static final float AUTONOMOUS_SPEED_SCALE = 1.0f; // 1.0f = 100% speed
    // --- ////

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentSubjectSegmentationBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        Log.d(TAG, String.valueOf(getRotationDegrees()));
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
        // 1. Basic Null Checks
        if (segmenter == null || image == null || binding == null) {
            return;
        }

        // 2. Auto Mode Check
        if (!isAutoMode) {
            if (binding.maskImageView.getDrawable() != null) {
                getActivity().runOnUiThread(() -> binding.maskImageView.setImageBitmap(null));
            }
            return;
        }

        // 3. [CRITICAL FIX] Busy Check
        // If the AI is still busy with the previous frame, DROP this one completely.
        // This prevents the "queue" from building up and causing the time to add up.
        if (isProcessing) {
            return;
        }

        // 4. Lock the flag and start timer
        isProcessing = true;
        final long startTime = SystemClock.elapsedRealtime();

        int rotation = getRotationDegrees();
        InputImage inputImage = InputImage.fromBitmap(image, rotation);

        segmenter.process(inputImage)
                .addOnSuccessListener(result -> {
                    // 5. [FIX] Unlock the flag immediately
                    isProcessing = false;

                    if (binding == null || getActivity() == null || !isAutoMode) {
                        return;
                    }

                    // Calculate PURE inference time (no waiting time)
                    long inferenceTime = SystemClock.elapsedRealtime() - startTime;

                    Bitmap foregroundMask = result.getForegroundBitmap();

                    if (foregroundMask == null) {
                        vehicle.setControl(new Control(1.0f, 1.0f));
                        getActivity().runOnUiThread(() -> {
                            if(binding != null) {
                                binding.maskImageView.setImageBitmap(null);
                                // Update text even if no object found
                                binding.inferenceInfo.setText(inferenceTime + " ms");
                            }
                        });
                        return;
                    }

                    // --- YOUR EXISTING LOGIC STARTS HERE ---

                    // ... (Area Calculation) ...
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
                    Log.d(TAG, String.format("Object covers: %.2f%%", areaPercentage));

                    // ... (Autopilot Logic) ...
                    int maskColor;
                    Control driveCommand;
                    final float STOP_THRESHOLD = closenessThreshold;

                    int newZone = 0; // 0=FAR
                    if (areaPercentage >= STOP_THRESHOLD) {
                        newZone = 1; // ROTATE
                    }

                    if (newZone != currentZone &&  newZone == 1) {
                        isTurningRight = random.nextBoolean();
                    }
                    currentZone = newZone;

                    switch (currentZone) {
                        case 1: // ROTATE
                            if (isTurningRight) {
                                driveCommand = new Control(AUTONOMOUS_SPEED_SCALE, -AUTONOMOUS_SPEED_SCALE);
                            } else {
                                driveCommand = new Control(-AUTONOMOUS_SPEED_SCALE, AUTONOMOUS_SPEED_SCALE);
                            }
                            maskColor = android.graphics.Color.argb(150, 255, 255, 0);
                            break;
                        case 0: // FAR
                        default:
                            driveCommand = new Control(AUTONOMOUS_SPEED_SCALE* 0.5f, AUTONOMOUS_SPEED_SCALE* 0.5f);
                            maskColor = android.graphics.Color.argb(150, 255, 0, 0);
                            break;
                    }
                    vehicle.setControl(driveCommand);

                    // Update Speed Info on UI
                    if (getActivity() != null) {
                        float left = vehicle.getLeftSpeed();
                        float right = vehicle.getRightSpeed();
                        getActivity().runOnUiThread(() -> {
                            if (binding != null && binding.controllerContainer != null) {
                                binding.controllerContainer.controlInfo.setText(
                                        String.format(Locale.US, "%.0f,%.0f", left, right));
                            }
                        });
                    }

                    // Rotate Mask for Display

                    Matrix matrix = new Matrix();
//                    if (getRotationDegrees() != 0) {
//                        matrix.postRotate(90);
//                    } else  {
//                        matrix.postRotate(0);
//                    }
                    matrix.postRotate(90);
                    Bitmap rotatedMask = Bitmap.createBitmap(
                            foregroundMask, 0, 0,
                            foregroundMask.getWidth(), foregroundMask.getHeight(),
                            matrix, true
                    );

                    // Update Mask Image AND Time
                    getActivity().runOnUiThread(() -> {
                        if (binding != null) {
                            binding.maskImageView.setImageBitmap(rotatedMask);
                            binding.maskImageView.setColorFilter(maskColor, android.graphics.PorterDuff.Mode.SRC_IN);
                            binding.inferenceInfo.setText(inferenceTime + " ms");
                        }
                    });
                })
                .addOnFailureListener(e -> {
                    // 6. [FIX] VERY IMPORTANT: Unlock flag on failure too
                    isProcessing = false;
                    Log.e(TAG, "Segmentation failed", e);
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
        // Add this check to avoid a crash if the binding is null
//        if (binding == null || binding.controllerContainer == null) {
//            return;
//        }
//
//        binding.controllerContainer.speedInfo.setText(
//                getString(
//                        R.string.string.speedInfo,
//                        String.format(
//                                Locale.US, "%3.0f,%3.0f", vehicle.getLeftWheelRpm(), vehicle.getRightWheelRpm())));
    }
}