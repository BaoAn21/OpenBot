//package org.openbot.lineTracking;
//
//import android.graphics.Bitmap;
//import android.os.Bundle;
//import android.util.Log;
//import android.view.LayoutInflater;
//import android.view.View;
//import android.view.ViewGroup;
//import android.widget.ImageView;
//
//import androidx.annotation.NonNull;
//import androidx.annotation.Nullable;
//import androidx.camera.core.ImageProxy;
//
//import org.opencv.android.Utils; // OpenCV utility
//import org.opencv.core.Core;
//import org.opencv.core.Mat; // OpenCV Matrix
//import org.opencv.core.Size; // For blur size
//import org.opencv.imgproc.Imgproc; // Image processing functions
//
//import org.openbot.common.CameraFragment;
//import org.openbot.databinding.FragmentLineTrackingBinding;
//import org.openbot.vehicle.Control; // Import Control class
//
//public class LineTrackingFragment extends CameraFragment {
//    private static final String TAG = "LineTrackingFragment";
//
//    private FragmentLineTrackingBinding binding;
//
//    // --- OpenCV Mats ---
//    private Mat matFrame;
//    private Mat matGray;
//    private Mat matCanny;
//    // -------------------
//
//    // --- Canny Thresholds ---
//    // You can tune these values
//    private static final double CANNY_THRESHOLD_1 = 50;
//    private static final double CANNY_THRESHOLD_2 = 150;
//    // ------------------------
//
//    private ImageView processedImageView; // Add variable for ImageView
//
//
//    @Override
//    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
//        binding = FragmentLineTrackingBinding.inflate(inflater, container, false);
//        return inflateFragment(binding, inflater, container);
//    }
//
//    @Override
//    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
//        super.onViewCreated(view, savedInstanceState);
//        processedImageView = binding.processedImageView;
//
//        // Initialize Mats once
//        matFrame = new Mat();
//        matGray = new Mat();
//        matCanny = new Mat();
//
//        Log.i(TAG, "LineTrackingFragment onViewCreated (Canny Mode)");
//    }
//
//    @Override
//    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
//        // Check if all required objects are initialized
//        if (vehicle == null || image == null || matFrame == null || matGray == null || matCanny == null) {
//            Log.e(TAG, "processFrame aborted: null objects");
//            if (imageProxy != null) imageProxy.close(); // Make sure proxy is closed if we abort early
//            return;
//        }
//
//        // 1. Convert Bitmap to Mat
//        Utils.bitmapToMat(image, matFrame);
//
//        Core.rotate(matFrame, matFrame, Core.ROTATE_90_CLOCKWISE);
//
//        // 2. Convert to Grayscale
//        Imgproc.cvtColor(matFrame, matGray, Imgproc.COLOR_RGB2GRAY);
//
//        // 3. Apply Gaussian Blur to reduce noise
//        Imgproc.GaussianBlur(matGray, matGray, new Size(5, 5), 0);
//
//        // 4. Apply Canny Edge Detection
//        Imgproc.Canny(matGray, matCanny, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);
//
//        // 5. Convert Canny output (which is 1-channel) to a 4-channel BGRA Mat
//        // This is necessary so we can convert it to an ARGB_8888 Bitmap for display
//        Imgproc.cvtColor(matCanny, matFrame, Imgproc.COLOR_GRAY2BGRA);
//
//        // 6. Convert the resulting Mat back to a Bitmap
//        Bitmap displayBitmap = Bitmap.createBitmap(matFrame.cols(), matFrame.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(matFrame, displayBitmap);
//
//        // 7. Stop the robot
//        vehicle.setControl(new Control(0, 0));
//
//        // === UPDATE IMAGEVIEW ===
//        final Bitmap finalDisplayBitmap = displayBitmap;
//        if (getActivity() != null) {
//            getActivity().runOnUiThread(() -> {
//                if (processedImageView != null) {
//                    processedImageView.setImageBitmap(finalDisplayBitmap);
//                }
//            });
//        }
//    }
//
//    @Override
//    public void onDestroyView() {
//        super.onDestroyView();
//        // Release Mats to free memory
//        if (matFrame != null) matFrame.release();
//        if (matGray != null) matGray.release();
//        if (matCanny != null) matCanny.release();
//
//        binding = null; // Release binding
//    }
//
//
//    @Override
//    protected void processControllerKeyData(String command) {
//        // Not used
//    }
//
//    @Override
//    protected void processUSBData(String data) {
//        // Not used
//    }
//}

package org.openbot.lineTracking;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import org.openbot.R;
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentLineTrackingBinding;
import org.openbot.vehicle.Control;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class LineTrackingFragment extends CameraFragment {
    private static final String TAG = "LineTrackingFragment";

    private FragmentLineTrackingBinding binding;

    // --- OpenCV Mats ---
    private Mat matFrame;
    private Mat matGray;
    private Mat matBlur;
    private Mat matCanny;
    private Mat matKernel;
    private Mat matDilated;
    private Mat matClosed;
    private Mat matHierarchy;
    private Mat matDisplay;
    private List<MatOfPoint> contoursList;
    // -------------------

    // --- Tunable Parameters ---
    // These are the defaults, can be changed from UI
    private int cannyLow = 30;
    private int cannyHigh = 100;
    private int kernelSize = 7;
    private int dilateIterations = 3;
    private int erodeIterations = 2;
    private double minObstacleArea = 50000;
    // ------------------------

    // --- Driving logic ---
    private static final double STOP_AREA_PERCENT = 0.3; // 30% of the screen
    private static final float BASE_SPEED = 0.4f;
    private static final float TURN_GAIN = 0.6f;
    // ------------------------

    // --- Control State ---
    private boolean isAutopilotOn = false;
    private long lastProcessingTimeMs;
    // ---------------------

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentLineTrackingBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Initialize Mats once
        matFrame = new Mat();
        matGray = new Mat();
        matBlur = new Mat();
        matCanny = new Mat();
        // Create the kernel for the first time
        matKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelSize, kernelSize));
        matDilated = new Mat();
        matClosed = new Mat();
        matHierarchy = new Mat();
        matDisplay = new Mat();
        contoursList = new ArrayList<>();

        // --- Setup UI Listeners ---
        binding.cameraToggleButton.setOnClickListener(v -> toggleCamera());

        // Autopilot Start/Stop Toggle
        binding.autopilotToggleButton.setOnClickListener(v -> {
            isAutopilotOn = !isAutopilotOn;
            if (isAutopilotOn) {
                binding.autopilotToggleButton.setText("Autopilot: ON (Tap to stop)");
                binding.autopilotToggleButton.setTextColor(Color.GREEN);
            } else {
                binding.autopilotToggleButton.setText("Autopilot: OFF (Tap to start)");
                binding.autopilotToggleButton.setTextColor(Color.RED);
                // Stop the robot immediately
                vehicle.setControl(new Control(0, 0));
                binding.statusTextview.setText("Status: STOPPED");
            }
        });

        // --- Canny Low Controls ---
        binding.cannyLowValueTextview.setText(String.valueOf(cannyLow));
        binding.plusCannyLowButton.setOnClickListener(v -> updateCannyLow(5));
        binding.minusCannyLowButton.setOnClickListener(v -> updateCannyLow(-5));

        // --- Canny High Controls ---
        binding.cannyHighValueTextview.setText(String.valueOf(cannyHigh));
        binding.plusCannyHighButton.setOnClickListener(v -> updateCannyHigh(5));
        binding.minusCannyHighButton.setOnClickListener(v -> updateCannyHigh(-5));

        // --- NEW: Kernel Size Controls ---
        binding.kernelValueTextview.setText(String.valueOf(kernelSize));
        binding.plusKernelButton.setOnClickListener(v -> updateKernel(2));
        binding.minusKernelButton.setOnClickListener(v -> updateKernel(-2));

        // --- NEW: Dilate Iterations Controls ---
        binding.dilateValueTextview.setText(String.valueOf(dilateIterations));
        binding.plusDilateButton.setOnClickListener(v -> updateDilate(1));
        binding.minusDilateButton.setOnClickListener(v -> updateDilate(-1));

        // --- NEW: Erode Iterations Controls ---
        binding.erodeValueTextview.setText(String.valueOf(erodeIterations));
        binding.plusErodeButton.setOnClickListener(v -> updateErode(1));
        binding.minusErodeButton.setOnClickListener(v -> updateErode(-1));

        // --- NEW: Min Area Controls ---
        binding.minAreaValueTextview.setText(String.valueOf((int)minObstacleArea));
        binding.plusMinAreaButton.setOnClickListener(v -> updateMinArea(1000));
        binding.minusMinAreaButton.setOnClickListener(v -> updateMinArea(-1000));

        Log.i(TAG, "LineTrackingFragment onViewCreated (Canny Blob Mode)");
    }

    // --- UI Helper Methods ---

    private void updateCannyLow(int delta) {
        cannyLow += delta;
        if (cannyLow > 255) cannyLow = 255;
        if (cannyLow < 0) cannyLow = 0;
        binding.cannyLowValueTextview.setText(String.valueOf(cannyLow));
    }

    private void updateCannyHigh(int delta) {
        cannyHigh += delta;
        if (cannyHigh > 255) cannyHigh = 255;
        if (cannyHigh < 0) cannyHigh = 0;
        binding.cannyHighValueTextview.setText(String.valueOf(cannyHigh));
    }

    private void updateKernel(int delta) {
        kernelSize += delta;
        if (kernelSize < 3) kernelSize = 3; // Kernel must be at least 3x3 and odd
        if (kernelSize % 2 == 0) kernelSize++; // Ensure it's odd
        binding.kernelValueTextview.setText(String.valueOf(kernelSize));
        // Re-create the kernel mat with the new size
        matKernel.release(); // Release old mat
        matKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelSize, kernelSize));
    }

    private void updateDilate(int delta) {
        dilateIterations += delta;
        if (dilateIterations < 1) dilateIterations = 1; // Must be at least 1
        binding.dilateValueTextview.setText(String.valueOf(dilateIterations));
    }

    private void updateErode(int delta) {
        erodeIterations += delta;
        if (erodeIterations < 1) erodeIterations = 1; // Must be at least 1
        binding.erodeValueTextview.setText(String.valueOf(erodeIterations));
    }

    private void updateMinArea(int delta) {
        minObstacleArea += delta;
        if (minObstacleArea < 0) minObstacleArea = 0;
        binding.minAreaValueTextview.setText(String.valueOf((int)minObstacleArea));
    }


    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
//        Log.d(TAG, "Rotation" + getRotationDegrees());
        // Check if all required objects are initialized
        if (vehicle == null || image == null || matFrame == null) {
            Log.e(TAG, "processFrame aborted: null objects");
            return;
        }

        // Default control: stop. This is a safety measure.
        Control control = new Control(0, 0);
        String status = "STOPPED";

        try {
            final long startTime = SystemClock.elapsedRealtime();

            // 1. Convert Bitmap to Mat (Bitmap is ARGB, matFrame becomes 4-channel RGBA)
            Utils.bitmapToMat(image, matFrame);
            Core.rotate(matFrame, matFrame, Core.ROTATE_90_CLOCKWISE);

            // 2. Convert RGBA to BGR (3-channel) and clone for display
            Imgproc.cvtColor(matFrame, matDisplay, Imgproc.COLOR_RGBA2BGR);

            // 3. Grayscale and Blur
            Imgproc.cvtColor(matDisplay, matGray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(matGray, matBlur, new Size(5, 5), 0);

            // 4. Canny Edge Detection (use values from UI)
            Imgproc.Canny(matBlur, matCanny, cannyLow, cannyHigh);

            // 5. "Closing" Gaps
            // Use new Point(-1,-1) for default anchor
            Imgproc.dilate(matCanny, matDilated, matKernel, new Point(-1,-1), dilateIterations);
            Imgproc.erode(matDilated, matClosed, matKernel, new Point(-1,-1), erodeIterations);

            // 6. Find Contours (on the 'matClosed' image)
            contoursList.clear(); // Clear list from previous frame
            Imgproc.findContours(matClosed, contoursList, matHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // 7. Find the largest contour
            double maxArea = 0;
            MatOfPoint bestContour = null;
            for (MatOfPoint contour : contoursList) {
                double area = Imgproc.contourArea(contour);
                if (area > maxArea) {
                    maxArea = area;
                    bestContour = contour;
                }
            }

            // 8. --- Main Driving Logic ---
            double stopArea = matFrame.rows() * matFrame.cols() * STOP_AREA_PERCENT;

            if (isAutopilotOn) {
                if (maxArea > minObstacleArea) {
                    // We see an obstacle...
                    status = "OBSTACLE DETECTED";
                    if (maxArea > stopArea) {
                        // Obstacle is too close! STOP.
                        status = "!! STOPPING !!";
                        control = new Control(0, 0);
                    } else {
                        // Obstacle detected, but not too close. Steer away from it.
                        status = "AVOIDING";
                        Moments moments = Imgproc.moments(bestContour);
                        double cX = moments.get_m10() / moments.get_m00();
                        double imageCenterX = matFrame.cols() / 2.0;

                        // Calculate error
                        double error = cX - imageCenterX;

                        // Proportional turn.
                        float turn = (float) (error / imageCenterX) * TURN_GAIN;
                        float left = BASE_SPEED - turn;
                        float right = BASE_SPEED + turn;
                        control = new Control(left, right);
                    }

                    // --- Draw Contour ---
                    // Draw the biggest contour on the display mat in green
                    if (bestContour != null) {
                        Imgproc.drawContours(matDisplay, contoursList, contoursList.indexOf(bestContour), new Scalar(0, 255, 0), 3);
                    }

                } else {
                    // No significant obstacle detected. Go straight.
                    status = "PATH CLEAR";
                    control = new Control(BASE_SPEED, BASE_SPEED);
                }
            } // else: autopilot is off, control remains (0,0) and status remains "STOPPED"

            // --- End Driving Logic ---

            // 10. Send command to vehicle
            vehicle.setControl(control);

            // 11. Convert Mats back to Bitmaps for display
            // --- Main Display ---
            // Convert BGR matDisplay to RGBA for Bitmap
            Imgproc.cvtColor(matDisplay, matDisplay, Imgproc.COLOR_BGR2RGBA);
            Bitmap displayBitmap = Bitmap.createBitmap(matDisplay.cols(), matDisplay.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matDisplay, displayBitmap);

            // --- Debug "Blob" Display ---
            // This shows the Canny + Closed image
            // Convert 1-channel matClosed to 4-channel RGBA
            Imgproc.cvtColor(matClosed, matFrame, Imgproc.COLOR_GRAY2RGBA);
            Bitmap debugBitmap = Bitmap.createBitmap(matFrame.cols(), matFrame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matFrame, debugBitmap);


            lastProcessingTimeMs = SystemClock.elapsedRealtime() - startTime;

            // 12. Update the UI on the UI thread
            final String finalStatus = status;
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    // Check if binding is still valid (in case view was destroyed)
                    if (binding != null) {
                        binding.processedImageView.setImageBitmap(displayBitmap);
                        binding.debugBlobView.setImageBitmap(debugBitmap);
                        binding.statusTextview.setText("Status: " + finalStatus);
                        binding.inferenceTimeTextview.setText(
                                String.format(Locale.US, "Inference: %d ms", lastProcessingTimeMs)
                        );
                    }
                });
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in processFrame", e);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Release Mats to free memory
        if (matFrame != null) matFrame.release();
        if (matGray != null) matGray.release();
        if (matBlur != null) matBlur.release();
        if (matCanny != null) matCanny.release();
        if (matKernel != null) matKernel.release();
        if (matDilated != null) matDilated.release();
        if (matClosed != null) matClosed.release();
        if (matHierarchy != null) matHierarchy.release();
        if (matDisplay != null) matDisplay.release();

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