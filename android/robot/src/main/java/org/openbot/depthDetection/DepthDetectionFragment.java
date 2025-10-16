package org.openbot.depthDetection;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;
import androidx.viewbinding.ViewBinding;

import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentDepthDetectionBinding; // Import the generated binding class

import java.io.IOException;

public class DepthDetectionFragment extends CameraFragment {

    private static final String TAG = "DepthDetectionFragment";
    private MidasNetSmall midasNet;
    private ImageView depthMapView;

    // Declare the binding object
    private FragmentDepthDetectionBinding binding;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        // *** THE CORRECT PATTERN FROM ObjectNavFragment ***
        // 1. Inflate the layout using the binding class
        binding = FragmentDepthDetectionBinding.inflate(inflater, container, false);

        // 2. Pass the root of the binding to the parent's special method
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        ImageView cameraToggleButton = binding.cameraToggleButton;

        // Set a click listener
        cameraToggleButton.setOnClickListener(v -> {
            // Call the public toggleCamera() method from the parent CameraFragment
            toggleCamera();
        });

        // Access the ImageView via the binding object, not findViewById
        depthMapView = binding.depthMapView;

        try {
            midasNet = new MidasNetSmall(requireActivity(), MapType.DEPTHVIEW_GRAYSCALE);
            Log.d(TAG, "MiDAS model initialized successfully.");
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize MiDAS model", e);
        }
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        // This log message should now appear
        Log.d(TAG, "processFrame is receiving data!");

        if (midasNet != null && image != null) {
            Bitmap depthMap = midasNet.getDepthMap(image);
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    if (depthMapView != null) {
                        depthMapView.setImageBitmap(depthMap);
                    }
                });
            }
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Nullify the binding object to prevent memory leaks
        binding = null;
        if (midasNet != null) {
            midasNet.close();
        }
    }

    // Unused methods
    @Override
    protected void processControllerKeyData(String command) {}

    @Override
    protected void processUSBData(String data) {}
}