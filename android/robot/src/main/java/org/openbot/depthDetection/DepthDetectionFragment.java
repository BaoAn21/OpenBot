package org.openbot.depthDetection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.openbot.R;
import org.openbot.common.ControlsFragment;
import org.openbot.utils.YuvToRgbConverter;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DepthDetectionFragment extends ControlsFragment {

    private static final String TAG = "DepthDetectionFragment";

    private MidasNetSmall midasNet;
    private ImageView depthMapView;
    private PreviewView previewView;

    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    private YuvToRgbConverter yuvToRgbConverter;
    private Bitmap bitmapBuffer;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        // Inflate our new layout that contains the PreviewView and ImageView
        return inflater.inflate(R.layout.fragment_depth_detection, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Find our UI elements
        previewView = view.findViewById(R.id.camera_preview);
        depthMapView = view.findViewById(R.id.depth_map_view);

        // Hide the camera preview, we only want to see the depth map
        previewView.setVisibility(View.INVISIBLE);

        // Initialize the model and camera components
        cameraExecutor = Executors.newSingleThreadExecutor();
        yuvToRgbConverter = new YuvToRgbConverter(requireContext());

        try {
            midasNet = new MidasNetSmall(requireActivity(), MapType.DEPTHVIEW_GRAYSCALE);
            Log.d(TAG, "MiDAS model initialized.");
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize MiDAS model", e);
        }

        // Check for camera permissions and start the camera
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext());
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Log.e(TAG, "Camera provider setup failed", e);
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    @SuppressLint("UnsafeOptInUsageError")
    private void bindCameraUseCases() {
        if (cameraProvider == null) {
            return;
        }

        cameraProvider.unbindAll();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, image -> {
            if (midasNet == null) {
                image.close();
                return;
            }

            if (bitmapBuffer == null) {
                bitmapBuffer = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
            }

            yuvToRgbConverter.yuvToRgb(image.getImage(), bitmapBuffer);
            image.close();

            // Run inference and get the depth map
            Bitmap depthMap = midasNet.getDepthMap(bitmapBuffer);

            // Update the UI on the main thread
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    if (depthMapView != null) {
                        depthMapView.setImageBitmap(depthMap);
                    }
                });
            }
        });

        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
        } catch (Exception e) {
            Log.e(TAG, "Use case binding failed", e);
        }
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                } else {
                    Toast.makeText(requireContext(), "Camera permission is required.", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        cameraExecutor.shutdown();
        if (midasNet != null) {
            midasNet.close();
        }
    }

    // Unused methods from ControlsFragment
    @Override
    protected void processControllerKeyData(String command) {}

    @Override
    protected void processUSBData(String data) {}
}