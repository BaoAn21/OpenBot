package org.openbot.depthDetection;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;
import org.openbot.R;
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentDepthDetectionBinding;
import org.openbot.utils.Constants;
import org.openbot.utils.Enums;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public class DepthDetectionFragment extends CameraFragment {

    private static final String TAG = "DepthDetectionFragment";
    private static final int DEPTH_IMAGE_DIM = 256;
    private static final float FORWARD_SPEED = 0.25f;
    private float closenessThreshold = 100.0f;
    private MidasNetSmall midasNet;
    private FragmentDepthDetectionBinding binding;
    private Enums.SpeedMode currentSpeedMode;
    private boolean isAutopilotActive = true;
    private Bitmap lastDepthBitmap = null;

    private final AtomicBoolean isProcessingFrame = new AtomicBoolean(false);

    private ModelType currentModelType = ModelType.QUANTIZED; // Default to FLOAT model

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentDepthDetectionBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        updateThresholdDisplay(); // Update the TextView with the initial value
        // ----------------------------------------------------
        updateAutopilotStatusUI();

        NetworkUtils.initializeWebSocket(); // Initialize (safe to call multiple times)
        NetworkUtils.connectWebSocket();

        // --- SETUP THRESHOLD BUTTON LISTENERS ---
        binding.plusThresholdButton.setOnClickListener(v -> {
            // Increase threshold, max 255
            closenessThreshold = Math.min(closenessThreshold + 5.0f, 255.0f);
            updateThresholdDisplay();
            // You might want to save the new value to SharedPreferences here
        });
        binding.minusThresholdButton.setOnClickListener(v -> {
            // Decrease threshold, min 0
            closenessThreshold = Math.max(closenessThreshold - 5.0f, 0.0f);
            updateThresholdDisplay();
            // You might want to save the new value to SharedPreferences here
        });

        binding.cameraToggleButton.setOnClickListener(v -> toggleCamera());
        binding.speedModeButton.setOnClickListener(v -> {
            currentSpeedMode = Enums.toggleSpeed(Enums.Direction.CYCLIC.getValue(), currentSpeedMode);
            setSpeedMode(currentSpeedMode);
        });
        binding.modelSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            // isChecked will be true for Quantized, false for Float
            ModelType newModel = isChecked ? ModelType.QUANTIZED : ModelType.FLOAT;
            reinitializeModel(newModel);
        });

        currentSpeedMode = Enums.SpeedMode.getByID(preferencesManager.getSpeedMode());
        setSpeedMode(currentSpeedMode);

        try {
            midasNet = new MidasNetSmall(requireActivity(), MapType.DEPTHVIEW_GRAYSCALE, ModelType.FLOAT);
            Log.d(TAG, "MiDAS model initialized successfully.");
            isProcessingFrame.set(true);
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize MiDAS model", e);
        }
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        // --- GRAB LOCAL REFERENCE AFTER FLAG CHECK ---
        // Capture the current midasNet instance *after* checking the flag.
        long inferenceTime = 0;
        String currentStatus = "MANUAL";
        Bitmap depthBitmapToShow = null;
        if (isAutopilotActive) {
        MidasNetSmall currentMidasNet = this.midasNet;

        // Check the flag AND the local reference
        if (!isProcessingFrame.get() || vehicle == null || currentMidasNet == null || image == null) {
            if (imageProxy != null) {
                imageProxy.close();
            }
            return;
        }
        // ---------------------------------------------

        // --- Use the local reference 'currentMidasNet' from now on ---
        Matrix matrix = new Matrix();
        matrix.postRotate(90f);
        Bitmap rotatedFrame = Bitmap.createBitmap(
                image, 0, 0,
                image.getWidth(), image.getHeight(),
                matrix, true
        );
        // Use the local reference
        float[] depthValues = currentMidasNet.getDepthMapFloatArray(rotatedFrame);
        // Send debug data
            // --- Send depth data via WebSocket ---
            if (depthValues != null) {
                NetworkUtils.sendDepthData(depthValues); // Send the data
            }
            // -------------------------------------
        inferenceTime = currentMidasNet.getLastInferenceTimeMs();
            // +++ Generate the grayscale bitmap +++
            if (depthValues != null && depthValues.length > 0) {
                depthBitmapToShow = ImageUtils.toGrayscaleBitmap(depthValues, DEPTH_IMAGE_DIM);
            }
            // ++++++++++++++++++++++++++++++++++++

        Log.i(TAG, "Inference Time: " + inferenceTime + " ms");
//        debugLogCenterValues(depthValues);

        // --- Only run analysis and control IF autopilot is active ---
        // Default status when autopilot is off

            boolean isObjectClose = analyzeDepthData(depthValues);

            if (isObjectClose) {
                vehicle.setControl(0, 0); // Autopilot stops
                currentStatus = "STOPPED (Obstacle)";
            } else {
                vehicle.setControl(FORWARD_SPEED, FORWARD_SPEED); // Autopilot moves
                currentStatus = "MOVING FORWARD (Auto)";
            }
        } else {
            // If autopilot is OFF, DO NOT send any control commands here.
            // Control comes from the gamepad via the parent fragment.
            // We just set the status text.
        }
        // ------------------------------------------------------------

        // Update UI
        final String statusToDisplay = currentStatus;
        final long timeToDisplay = inferenceTime;
        final Bitmap finalDepthBitmap = depthBitmapToShow; // Use final variable for lambda
        if (getActivity() != null) {
            getActivity().runOnUiThread(() -> {
                if (binding != null) {
                    binding.inferenceTimeTextview.setText(
                            String.format(Locale.US, "Inference: %d ms", timeToDisplay)
                    );
                    binding.statusTextview.setText("Status: " + statusToDisplay);
                    // +++ Update the ImageView +++
                    if (finalDepthBitmap != null) {
                        binding.depthImageView.setImageBitmap(finalDepthBitmap);
                        // Store the bitmap if needed for later recycling
                        recycleLastBitmap(); // Recycle the previous one
                        lastDepthBitmap = finalDepthBitmap;
                    }
                    // +++++++++++++++++++++++++++
                }
            });
        }
    }
    // +++ Add method to recycle bitmap +++
    private void recycleLastBitmap() {
        if (lastDepthBitmap != null && !lastDepthBitmap.isRecycled()) {
            lastDepthBitmap.recycle();
            lastDepthBitmap = null;
        }
    }
    // ++++++++++++++++++++++++++++++++++++

    // --- ADD HELPER TO UPDATE AUTOPILOT UI ---
    private void updateAutopilotStatusUI() {
        if (binding != null) {
            if (isAutopilotActive) {
                binding.autopilotStatusTextview.setText("Autopilot: ON");
                binding.autopilotStatusTextview.setTextColor(Color.GREEN);
            } else {
                binding.autopilotStatusTextview.setText("Autopilot: OFF");
                binding.autopilotStatusTextview.setTextColor(Color.RED);
            }
        }
    }
    // -----------------------------------------

    private void reinitializeModel(ModelType newModelType) {
        isProcessingFrame.set(false);
        runOnBackgroundThread(() -> {
            if (midasNet != null) {
                midasNet.close();
                midasNet = null;
                Log.d(TAG, "Previous MiDAS model closed.");
            }

            currentModelType = newModelType;

            try {
                midasNet = new MidasNetSmall(requireActivity(), MapType.DEPTHVIEW_GRAYSCALE, currentModelType);
                Log.d(TAG, "MiDAS model re-initialized successfully: " + currentModelType + " on thread: " + Thread.currentThread().getName());

                isProcessingFrame.set(true);
                Log.d(TAG, "Re-enabled frame processing."); // Add log
                if (getActivity() != null) {
                    getActivity().runOnUiThread(() -> {
                        if (binding != null) {
                            binding.modelTypeTextview.setText(currentModelType.toString());
                        }
                    });
                }

            } catch (IOException e) {
                Log.e(TAG, "Failed to re-initialize MiDAS model", e);
            } catch (IllegalStateException e) {
                Log.e(TAG, "IllegalStateException during model re-initialization", e);
            } catch (Exception e) { // Catch any other potential native errors
                Log.e(TAG, "Unexpected error during model re-initialization", e);
            }
        });
        // No 'else' block needed, as the helper method handles the null check internally.
    }

    // --- ADD THIS HELPER METHOD ---
    private void updateThresholdDisplay() {
        if (binding != null) {
            // Update the TextView to show the current threshold value
            binding.thresholdValueTextview.setText(String.format(Locale.US, "%.0f", closenessThreshold));
        }
    }
    // ----------------------------

    private boolean analyzeDepthData(float[] depthValues) {
        int zoneSize = 20;
        int startX = (DEPTH_IMAGE_DIM / 2) - (zoneSize / 2);
        int startY = (DEPTH_IMAGE_DIM / 2) - (zoneSize / 2);
        float sumOfValues = 0;
        int pixelCount = 0;
        for (int y = startY; y < startY + zoneSize; y++) {
            for (int x = startX; x < startX + zoneSize; x++) {
                int index = y * DEPTH_IMAGE_DIM + x;
                sumOfValues += depthValues[index];
                pixelCount++;
            }
        }
        float averageDistanceValue = sumOfValues / pixelCount;
        boolean isClose = averageDistanceValue > closenessThreshold;
        if (isClose) {
            Log.d(TAG, "DANGER: Object is very close! Average value: " + averageDistanceValue);
        }
        return isClose;
        // ----------------------------------------------------------
//        if (depthValues == null || depthValues.length == 0) {
//            Log.w(TAG, "analyzeDepthData: depthValues array is null or empty!");
//            return false; // Cannot analyze if there's no data
//        }
//
//        float sumOfValues = 0;
//        int pixelCount = depthValues.length; // Total number of pixels
//
//        // Iterate through ALL depth values
//        for (float value : depthValues) {
//            sumOfValues += value;
//        }
//
//        // Calculate the average depth value for the entire frame
//        float averageDistanceValue = sumOfValues / pixelCount;
//
//        // Check if the overall average depth indicates closeness
//        boolean isClose = averageDistanceValue > closenessThreshold;
//
//        if (isClose) {
//            Log.d(TAG, String.format(Locale.US,
//                    "DANGER: Average depth (%.2f) exceeds threshold (%.2f). Object potentially close.",
//                    averageDistanceValue, closenessThreshold));
//        } else {
//            Log.d(TAG, String.format(Locale.US,
//                    "INFO: Average depth (%.2f) is below threshold (%.2f). Path appears clear.",
//                    averageDistanceValue, closenessThreshold));
//        }
//
//        return isClose;
    }

    private void debugLogCenterValues(float[] depthValues) {
        int zoneSize = 10;
        int startX = (DEPTH_IMAGE_DIM / 2) - (zoneSize / 2);
        int startY = (DEPTH_IMAGE_DIM / 2) - (zoneSize / 2);

        StringBuilder sb = new StringBuilder();
        sb.append("Center 10x10 Depth Values:\n");

        for (int y = startY; y < startY + zoneSize; y++) {
            for (int x = startX; x < startX + zoneSize; x++) {
                int index = y * DEPTH_IMAGE_DIM + x;
                int intValue = (int) depthValues[index];
                sb.append(String.format("%5d", intValue));
            }
            sb.append("\n");
        }
        Log.d(TAG, sb.toString());
    }

    private void setSpeedMode(Enums.SpeedMode speedMode) {
        if (speedMode == null) return;
        switch (speedMode) {
            case SLOW:
                binding.speedModeButton.setImageResource(R.drawable.ic_speed_low);
                break;
            case NORMAL:
                binding.speedModeButton.setImageResource(R.drawable.ic_speed_medium);
                break;
            case FAST:
                binding.speedModeButton.setImageResource(R.drawable.ic_speed_high);
                break;
        }
        preferencesManager.setSpeedMode(speedMode.getValue());
        if (vehicle != null) {
            vehicle.setSpeedMultiplier(speedMode.getValue());
        }
    }

    @Override
    public void onDestroyView() {
        Log.d(TAG, "onDestroyView: Starting cleanup...");

        // 1. Immediately signal processing to stop
        isProcessingFrame.set(false);
        Log.d(TAG, "onDestroyView: Processing flag set to false.");

        // 2. Capture the current instance if it exists
        MidasNetSmall netToClose = midasNet;

        // 3. Set the field to null *after* capturing the reference to close.
        // This slightly reduces the chance of the race condition, but the
        // primary safeguard is the isProcessingFrame check + local var in processFrame.
        midasNet = null;
        Log.d(TAG, "onDestroyView: midasNet field set to null.");

        // 4. Schedule the actual closure on the background thread
        if (netToClose != null) {
            runOnBackgroundThread(() -> {
                try {
                    Log.d(TAG, "onDestroyView: Closing MidasNet instance on background thread...");
                    netToClose.close(); // Close the captured instance
                    Log.d(TAG, "onDestroyView: MidasNet instance closed successfully.");
                } catch (Exception e) {
                    Log.e(TAG, "onDestroyView: Error closing MidasNet on background thread", e);
                }
            });
        }

        // 5. Call super (which will eventually shut down the executor)
        super.onDestroyView();

        // 6. Nullify binding
        binding = null;
        Log.d(TAG, "onDestroyView: Cleanup initiated.");
    }

    @Override
    protected void processUSBData(String data) {
        if (binding != null && vehicle != null) {
            binding.speedInfoTextview.setText(
                    String.format(Locale.US, "RPM: %3.0f,%3.0f", vehicle.getLeftWheelRpm(), vehicle.getRightWheelRpm())
            );
        }
    }

    @Override
    protected void processControllerKeyData(String command) {
        if (command == null) return;

        switch (command) {
            case Constants.CMD_NETWORK: // Assuming "NETWORK" is the command for your toggle button
                // Toggle the autopilot state
                isAutopilotActive = !isAutopilotActive;
                Log.d(TAG, "Autopilot toggled: " + isAutopilotActive);

                // IMPORTANT: Stop the robot immediately if autopilot is turned OFF
                if (!isAutopilotActive && vehicle != null) {
                    vehicle.setControl(0, 0);
                    Log.d(TAG, "Autopilot OFF. Sending STOP command.");
                }

                // Update the UI text
                updateAutopilotStatusUI();
                break;

            // Add cases for other controller commands if needed in this fragment
            // case Constants.CMD_SPEED_UP:
            //    break;
        }
    }
}