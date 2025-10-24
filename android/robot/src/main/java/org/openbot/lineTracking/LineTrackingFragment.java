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
import org.opencv.core.Core; // Core OpenCV functions
import org.opencv.core.Mat; // OpenCV Matrix
import org.opencv.core.MatOfPoint; // For contours
import org.opencv.core.Point; // For points
import org.opencv.core.Scalar; // For colors
import org.opencv.core.Size; // For blur size
import org.opencv.imgproc.Imgproc; // Image processing functions

import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentLineTrackingBinding;
import org.openbot.vehicle.Control; // Import Control class
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class LineTrackingFragment extends CameraFragment {
    private static final String TAG = "LineTrackingFragment";

    private FragmentLineTrackingBinding binding;

    // --- OpenCV Mats ---
    private Mat matFrame;
    private Mat matHsv;
    private Mat matMask;
    private Mat matHierarchy;
    // -------------------

    // --- Line Color Thresholds (Example for Black Line) ---
    // These values need tuning! Use an HSV color picker tool.
    private Scalar lowerBlack = new Scalar(0, 0, 0);
    private Scalar upperBlack = new Scalar(180, 255, 50); // Adjust Value (brightness) threshold
    // ----------------------------------------------------

    // --- Robot Control ---
    private static final float FORWARD_SPEED = 0.25f; // Slow speed
    private static final float TURN_SPEED = 0.35f;   // Adjust turn intensity
    private ImageView processedImageView; // Add variable for ImageView
    // ---------------------


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
        matHsv = new Mat();
        matMask = new Mat();
        matHierarchy = new Mat();
        Log.i(TAG, "LineTrackingFragment onViewCreated");
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        if (vehicle == null || image == null || matFrame == null || matHsv == null || matMask == null || matHierarchy == null) {
            Log.e(TAG, "processFrame aborted: null objects");
            if (imageProxy != null) imageProxy.close(); // Make sure proxy is closed if we abort early
            return;
        }

        // 1. Convert Bitmap to Mat
        Utils.bitmapToMat(image, matFrame);

        // Optional: Rotation if needed
        // Core.rotate(matFrame, matFrame, Core.ROTATE_90_CLOCKWISE);

        // --- Keep a clean copy FOR PROCESSING ---
        Mat processingMat = matFrame.clone();

        // 2. Convert processing copy to HSV
        Imgproc.cvtColor(processingMat, matHsv, Imgproc.COLOR_RGB2HSV);

        // 3. Create Binary Mask
        Core.inRange(matHsv, lowerBlack, upperBlack, matMask);

        // Optional Morphological Ops on matMask...

        // 4. Find Contours (using the mask)
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(matMask, contours, matHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = -1;
        int maxAreaIdx = -1;
        for (int i = 0; i < contours.size(); i++) {
            double area = Imgproc.contourArea(contours.get(i));
            if (area > maxArea) {
                maxArea = area;
                maxAreaIdx = i;
            }
        }
        // --- ADD LOGGING HERE ---
        Log.d(TAG, "Number of contours found: " + contours.size() + ", Max Area: " + maxArea);
        // ------------------------

        Control controlSignal = new Control(0, 0);
        Bitmap displayBitmap = image; // Default to showing original image if no line found

        // 5 & 6. Calculate Centroid and Steering Logic
        if (maxAreaIdx != -1 && maxArea > 10) { // Add a minimum area threshold (e.g., 10 pixels)
            Moments moments = Imgproc.moments(contours.get(maxAreaIdx));
            // --- REMOVE THE m00 CHECK FOR NOW to match original behavior closer ---
            // Calculate centroid coordinates
            Point centroid = new Point();
            centroid.x = moments.get_m10() / moments.get_m00(); // Might produce NaN/Infinity if m00 is 0
            centroid.y = moments.get_m01() / moments.get_m00(); // Might produce NaN/Infinity if m00 is 0

            // Check if centroid calculation was valid (not NaN or Infinity)
            if (!Double.isNaN(centroid.x) && !Double.isInfinite(centroid.x)) {
                int frameCenterX = matFrame.cols() / 2; // Use original frame width for center calc
                double error = centroid.x - frameCenterX;
                float turn = (float) (error * (TURN_SPEED / frameCenterX));
                controlSignal = new Control(FORWARD_SPEED + turn, FORWARD_SPEED - turn);

                Log.d(TAG, String.format(Locale.US,"Line Centroid: (%.1f, %.1f), Error: %.1f, Turn: %.2f", centroid.x, centroid.y, error, turn));

                // --- 7. DRAW VISUALIZATION on the original matFrame ---
                Imgproc.drawContours(matFrame, contours, maxAreaIdx, new Scalar(0, 255, 0), 3); // Green contour
                Imgproc.circle(matFrame, centroid, 10, new Scalar(255, 0, 0), -1);           // Red centroid dot
                Imgproc.line(matFrame, new Point(frameCenterX, 0), new Point(frameCenterX, matFrame.rows()), new Scalar(0, 0, 255), 2); // Blue center line

                // Convert the matFrame (with drawings) back to Bitmap
                displayBitmap = Bitmap.createBitmap(matFrame.cols(), matFrame.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(matFrame, displayBitmap);
                // ----------------------------------------------------
            } else {
                Log.w(TAG, "Centroid calculation resulted in NaN/Infinity. m00 might be zero.");
                controlSignal = new Control(0, 0); // Stop if calculation invalid
            }
        } else {
            Log.d(TAG, "No significant line contour detected.");
            controlSignal = new Control(0, 0);
        }

        vehicle.setControl(controlSignal);

        // === UPDATE IMAGEVIEW ===
        final Bitmap finalDisplayBitmap = displayBitmap; // Use final variable for lambda
        if (getActivity() != null) {
            getActivity().runOnUiThread(() -> {
                if (processedImageView != null) {
                    processedImageView.setImageBitmap(finalDisplayBitmap);
                }
            });
        }
        // Release the cloned Mat if you used one earlier
        processingMat.release();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Release Mats to free memory
        if (matFrame != null) matFrame.release();
        if (matHsv != null) matHsv.release();
        if (matMask != null) matMask.release();
        if (matHierarchy != null) matHierarchy.release();
        binding = null; // Release binding
    }


    @Override
    protected void processControllerKeyData(String command) {
        // Not used in this basic implementation
    }

    @Override
    protected void processUSBData(String data) {
        // Could update RPM display if you add it to the layout
    }
}