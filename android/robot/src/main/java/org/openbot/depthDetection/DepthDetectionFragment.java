package org.openbot.depthDetection;

import android.graphics.Bitmap;
import android.graphics.Matrix;
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

    private Bitmap rotatedBitmap;
    private int sensorOrientation;

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

        sensorOrientation = 90 - ImageUtils.getScreenOrientation(requireActivity());

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
        if (midasNet == null || image == null) {
            return;
        }

        // 1. Get the original depth map from the model. It will be incorrectly rotated.
        Bitmap originalDepthMap = midasNet.getDepthMap(image);

        // 2. Create a transformation matrix to fix the rotation.
        // A 90-degree clockwise rotation is needed to make it upright.
        Matrix matrix = new Matrix();
        matrix.postRotate(90f);

        // 3. Create a new, correctly oriented bitmap from the original depth map.
        Bitmap rotatedDepthMap = Bitmap.createBitmap(
                originalDepthMap,
                0,                                // Start at x=0
                0,                                // Start at y=0
                originalDepthMap.getWidth(),      // Use the full width
                originalDepthMap.getHeight(),     // Use the full height
                matrix,
                true
        );

        // 4. Display the final, corrected bitmap on the screen.
        if (getActivity() != null) {
            getActivity().runOnUiThread(() -> {
                if (depthMapView != null) {
                    depthMapView.setImageBitmap(rotatedDepthMap);
                }
            });
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