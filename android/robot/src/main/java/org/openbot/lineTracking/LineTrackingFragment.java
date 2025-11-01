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
    private Mat matThresh;
    private Mat matHierarchy;
    private Mat matDisplay;
    private List<MatOfPoint> contoursList;
    // -------------------

    // --- Tunable Parameters ---
    private static final double MIN_OBSTACLE_AREA = 2500;
    private static final double STOP_AREA_PERCENT = 0.3; // 30% of the screen
    private static final float BASE_SPEED = 0.4f;
    private static final float TURN_GAIN = 0.6f;
    private final Size BLUR_STRENGTH = new Size(7, 7);
    // ------------------------

    // --- Control State ---
    private boolean isAutopilotOn = false;
    private int thresholdValue = 127;
    private int thresholdMode = Imgproc.THRESH_BINARY_INV; // For dark objects on light background
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
        matThresh = new Mat();
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

        // Threshold Value Controls
        binding.thresholdValueTextview.setText(String.valueOf(thresholdValue));
        binding.plusThresholdButton.setOnClickListener(v -> updateThreshold(5));
        binding.minusThresholdButton.setOnClickListener(v -> updateThreshold(-5));

        // Threshold Mode Switch (BINARY vs BINARY_INV)
        binding.thresholdModeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // For BRIGHT objects on DARK background
                thresholdMode = Imgproc.THRESH_BINARY;
                binding.thresholdModeTextview.setText("Bright Object");
            } else {
                // For DARK objects on LIGHT background
                thresholdMode = Imgproc.THRESH_BINARY_INV;
                binding.thresholdModeTextview.setText("Dark Object (INV)");
            }
        });

        Log.i(TAG, "LineTrackingFragment onViewCreated (Threshold Blob Mode)");
    }

    private void updateThreshold(int delta) {
        thresholdValue += delta;
        if (thresholdValue > 255) thresholdValue = 255;
        if (thresholdValue < 0) thresholdValue = 0;
        binding.thresholdValueTextview.setText(String.valueOf(thresholdValue));
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
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

            // 2. Rotate the frame
            Core.rotate(matFrame, matFrame, Core.ROTATE_90_CLOCKWISE);

            // 3. Convert RGBA to BGR (3-channel) and clone for display
            Imgproc.cvtColor(matFrame, matDisplay, Imgproc.COLOR_RGBA2BGR);

            // 4. Grayscale and Blur
            Imgproc.cvtColor(matDisplay, matGray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(matGray, matBlur, BLUR_STRENGTH, 0);

            // 5. Threshold Image (Blobs)
            // We use the manual thresholdValue from the UI
            Imgproc.threshold(matBlur, matThresh, thresholdValue, 255, thresholdMode);

            // 6. Find Contours
            contoursList.clear(); // Clear list from previous frame
            Imgproc.findContours(matThresh, contoursList, matHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

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
                if (maxArea > MIN_OBSTACLE_AREA) {
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

                        // Calculate error:
                        // positive error: obstacle is on the right
                        // negative error: obstacle is on the left
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
            // Convert 1-channel matThresh to 4-channel RGBA
            Imgproc.cvtColor(matThresh, matFrame, Imgproc.COLOR_GRAY2RGBA);
            Bitmap debugBitmap = Bitmap.createBitmap(matFrame.cols(), matFrame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matFrame, debugBitmap);


            lastProcessingTimeMs = SystemClock.elapsedRealtime() - startTime;

            // 12. Update the UI on the UI thread
            final String finalStatus = status;
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    binding.processedImageView.setImageBitmap(displayBitmap);
                    binding.debugBlobView.setImageBitmap(debugBitmap);
                    binding.statusTextview.setText("Status: " + finalStatus);
                    binding.inferenceTimeTextview.setText(
                            String.format(Locale.US, "Inference: %d ms", lastProcessingTimeMs)
                    );
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
        if (matThresh != null) matThresh.release();
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