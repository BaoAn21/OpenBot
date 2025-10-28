package org.openbot.lineTracking;

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

import org.opencv.android.Utils; // OpenCV utility
import org.opencv.core.Core;
import org.opencv.core.Mat; // OpenCV Matrix
import org.opencv.core.Size; // For blur size
import org.opencv.imgproc.Imgproc; // Image processing functions

import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentLineTrackingBinding;
import org.openbot.vehicle.Control; // Import Control class

public class LineTrackingFragment extends CameraFragment {
    private static final String TAG = "LineTrackingFragment";

    private FragmentLineTrackingBinding binding;

    // --- OpenCV Mats ---
    private Mat matFrame;
    private Mat matGray;
    private Mat matCanny;
    // -------------------

    // --- Canny Thresholds ---
    // You can tune these values
    private static final double CANNY_THRESHOLD_1 = 50;
    private static final double CANNY_THRESHOLD_2 = 150;
    // ------------------------

    private ImageView processedImageView; // Add variable for ImageView


    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentLineTrackingBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        processedImageView = binding.processedImageView;

        // Initialize Mats once
        matFrame = new Mat();
        matGray = new Mat();
        matCanny = new Mat();

        Log.i(TAG, "LineTrackingFragment onViewCreated (Canny Mode)");
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        // Check if all required objects are initialized
        if (vehicle == null || image == null || matFrame == null || matGray == null || matCanny == null) {
            Log.e(TAG, "processFrame aborted: null objects");
            if (imageProxy != null) imageProxy.close(); // Make sure proxy is closed if we abort early
            return;
        }

        // 1. Convert Bitmap to Mat
        Utils.bitmapToMat(image, matFrame);

        Core.rotate(matFrame, matFrame, Core.ROTATE_90_CLOCKWISE);

        // 2. Convert to Grayscale
        Imgproc.cvtColor(matFrame, matGray, Imgproc.COLOR_RGB2GRAY);

        // 3. Apply Gaussian Blur to reduce noise
        Imgproc.GaussianBlur(matGray, matGray, new Size(5, 5), 0);

        // 4. Apply Canny Edge Detection
        Imgproc.Canny(matGray, matCanny, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);

        // 5. Convert Canny output (which is 1-channel) to a 4-channel BGRA Mat
        // This is necessary so we can convert it to an ARGB_8888 Bitmap for display
        Imgproc.cvtColor(matCanny, matFrame, Imgproc.COLOR_GRAY2BGRA);

        // 6. Convert the resulting Mat back to a Bitmap
        Bitmap displayBitmap = Bitmap.createBitmap(matFrame.cols(), matFrame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matFrame, displayBitmap);

        // 7. Stop the robot
        vehicle.setControl(new Control(0, 0));

        // === UPDATE IMAGEVIEW ===
        final Bitmap finalDisplayBitmap = displayBitmap;
        if (getActivity() != null) {
            getActivity().runOnUiThread(() -> {
                if (processedImageView != null) {
                    processedImageView.setImageBitmap(finalDisplayBitmap);
                }
            });
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Release Mats to free memory
        if (matFrame != null) matFrame.release();
        if (matGray != null) matGray.release();
        if (matCanny != null) matCanny.release();

        binding = null; // Release binding
    }


    @Override
    protected void processControllerKeyData(String command) {
        // Not used
    }

    @Override
    protected void processUSBData(String data) {
        // Not used
    }
}