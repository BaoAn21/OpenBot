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
    // This is our threshold. Lower values mean closer.
    // 25 is a good starting point. You can adjust it by testing.
    private static final float CLOSENESS_THRESHOLD = 25.0f;
    private static final int DEPTH_IMAGE_DIM = 256;

    // Declare the binding object
    private FragmentDepthDetectionBinding binding;
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {

        binding = FragmentDepthDetectionBinding.inflate(inflater, container, false);

        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        ImageView cameraToggleButton = binding.cameraToggleButton;


        // Set a click listener
        cameraToggleButton.setOnClickListener(v -> {
            toggleCamera();
        });

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

        // 1. Create a matrix to rotate the incoming camera frame.
        Matrix matrix = new Matrix();
        matrix.postRotate(90f);

        // 2. Create a new, correctly oriented bitmap from the original camera frame.
        Bitmap rotatedFrame = Bitmap.createBitmap(
                image,
                0,
                0,
                image.getWidth(),
                image.getHeight(),
                matrix,
                true
        );

        // 3. Run inference on the CORRECTLY ROTATED frame.
        // The output from the model will now be upright.
        Bitmap finalDepthMap = midasNet.getDepthMap(rotatedFrame);

        // 4. Display the result directly. No second rotation is needed.
        if (getActivity() != null) {
            getActivity().runOnUiThread(() -> {
                if (depthMapView != null) {
                    depthMapView.setImageBitmap(finalDepthMap);
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