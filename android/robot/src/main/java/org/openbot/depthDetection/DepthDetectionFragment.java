package org.openbot.depthDetection;

import android.graphics.Bitmap;
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
import org.openbot.utils.Enums;
import java.io.IOException;
import java.util.Locale;

public class DepthDetectionFragment extends CameraFragment {

    private static final String TAG = "DepthDetectionFragment";
    private static final int DEPTH_IMAGE_DIM = 256;
    private static final float FORWARD_SPEED = 0.25f;
    private static final float CLOSENESS_THRESHOLD = 100.0f;

    private MidasNetSmall midasNet;
    private FragmentDepthDetectionBinding binding;
    private Enums.SpeedMode currentSpeedMode;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentDepthDetectionBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.cameraToggleButton.setOnClickListener(v -> toggleCamera());
        binding.speedModeButton.setOnClickListener(v -> {
            currentSpeedMode = Enums.toggleSpeed(Enums.Direction.CYCLIC.getValue(), currentSpeedMode);
            setSpeedMode(currentSpeedMode);
        });

        currentSpeedMode = Enums.SpeedMode.getByID(preferencesManager.getSpeedMode());
        setSpeedMode(currentSpeedMode);

        try {
            midasNet = new MidasNetSmall(requireActivity(), MapType.DEPTHVIEW_GRAYSCALE, ModelType.FLOAT);
            Log.d(TAG, "MiDAS model initialized successfully.");
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize MiDAS model", e);
        }
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        if (vehicle == null || midasNet == null || image == null) {
            return;
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(90f);
        Bitmap rotatedFrame = Bitmap.createBitmap(
                image, 0, 0,
                image.getWidth(), image.getHeight(),
                matrix, true
        );

        float[] depthValues = midasNet.getDepthMapFloatArray(rotatedFrame);

        debugLogCenterValues(depthValues);

        boolean isObjectClose = analyzeDepthData(depthValues);

        if (isObjectClose) {
            vehicle.setControl(0, 0);
        } else {
            vehicle.setControl(FORWARD_SPEED, FORWARD_SPEED);
        }
    }

    private boolean analyzeDepthData(float[] depthValues) {
        int zoneSize = 10;
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
        boolean isClose = averageDistanceValue > CLOSENESS_THRESHOLD;
        if (isClose) {
            Log.d(TAG, "DANGER: Object is very close! Average value: " + averageDistanceValue);
        }
        return isClose;
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
        super.onDestroyView();
        binding = null;
        if (midasNet != null) {
            midasNet.close();
        }
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
        // Not used
    }
}