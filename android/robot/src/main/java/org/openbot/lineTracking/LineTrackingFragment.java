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

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentLineTrackingBinding;
import org.openbot.vehicle.Control;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class LineTrackingFragment extends CameraFragment {
    private static final String TAG = "LineTrackingFragment";

    private FragmentLineTrackingBinding binding;

    // --- OpenCV Mats ---
    private Mat matFrame;
    private Mat matGray;
    private Mat matCanny;
    private Mat matRoi; // Region of Interest
    private Mat matHoughLines;
    // -------------------

    // --- Canny Thresholds ---
    private static final double CANNY_THRESHOLD_1 = 50;
    private static final double CANNY_THRESHOLD_2 = 150;
    // ------------------------

    // --- Robot Control ---
    private static final float FORWARD_SPEED = 0.2f; // Keep it slow
    private static final float Kp = 0.5f; // Proportional gain (TUNE THIS)
    // ---------------------

    private ImageView processedImageView;


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
        matRoi = new Mat();
        matHoughLines = new Mat();

        Log.i(TAG, "LineTrackingFragment onViewCreated (Edge Following Mode)");
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        if (vehicle == null || image == null || matFrame == null || matGray == null || matCanny == null) {
            Log.e(TAG, "processFrame aborted: null objects");
            if (imageProxy != null) imageProxy.close();
            return;
        }

        // 1. Convert Bitmap to Mat and Rotate
        Utils.bitmapToMat(image, matFrame);
        Core.rotate(matFrame, matFrame, Core.ROTATE_90_CLOCKWISE);

        // 2. Define Region of Interest (ROI) - Bottom half of the image
        Rect roiRect = new Rect(0, matFrame.rows() / 2, matFrame.cols(), matFrame.rows() / 2);
        matRoi = new Mat(matFrame, roiRect);

        // 3. Process the ROI
        Imgproc.cvtColor(matRoi, matGray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(matGray, matGray, new Size(5, 5), 0);
        Imgproc.Canny(matGray, matCanny, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);

        // 4. Find Lines with Hough Transform
        matHoughLines = new Mat();
        Imgproc.HoughLinesP(matCanny, matHoughLines, 1, Math.PI / 180, 50, 20, 10);

        // 5. Filter and Average Lines
        List<Double> leftSlopes = new ArrayList<>();
        List<Double> leftIntercepts = new ArrayList<>();
        List<Double> rightSlopes = new ArrayList<>();
        List<Double> rightIntercepts = new ArrayList<>();

        for (int i = 0; i < matHoughLines.rows(); i++) {
            double[] line = matHoughLines.get(i, 0);
            double x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];

            // Calculate slope
            double slope = (x1 == x2) ? 999 : (y2 - y1) / (x2 - x1);
            double intercept = y1 - (slope * x1);
            double angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

            // Filter by angle/slope
            if (Math.abs(angle) > 20 && Math.abs(angle) < 160) { // Ignore horizontal lines
                if (slope < -0.5) { // Left line
                    leftSlopes.add(slope);
                    leftIntercepts.add(intercept);
                    Imgproc.line(matRoi, new Point(x1, y1), new Point(x2, y2), new Scalar(255, 0, 0), 2); // Blue
                } else if (slope > 0.5) { // Right line
                    rightSlopes.add(slope);
                    rightIntercepts.add(intercept);
                    Imgproc.line(matRoi, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255), 2); // Red
                }
            }
        }

        // 6. Calculate Average Lines and Find Lane Center
        float error = 0;
        float turn = 0;
        int roiCenterX = matRoi.cols() / 2;
        int horizonY = matRoi.rows() / 2; // "Look-ahead" point

        double leftX = 0;
        if (!leftSlopes.isEmpty()) {
            double avgLeftSlope = average(leftSlopes);
            double avgLeftIntercept = average(leftIntercepts);
            leftX = (horizonY - avgLeftIntercept) / avgLeftSlope;
            Imgproc.line(matRoi, new Point(leftX, horizonY), new Point(leftX, matRoi.rows()), new Scalar(255, 255, 0), 3); // Cyan
        }

        double rightX = matRoi.cols();
        if (!rightSlopes.isEmpty()) {
            double avgRightSlope = average(rightSlopes);
            double avgRightIntercept = average(rightIntercepts);
            rightX = (horizonY - avgRightIntercept) / avgRightSlope;
            Imgproc.line(matRoi, new Point(rightX, horizonY), new Point(rightX, matRoi.rows()), new Scalar(255, 0, 255), 3); // Magenta
        }

        // 7. Calculate Control Signal
        if (!leftSlopes.isEmpty() || !rightSlopes.isEmpty()) {
            // If we only see one line, just try to stay a fixed distance from it
            if (leftSlopes.isEmpty()) {
                leftX = rightX - (matRoi.cols() * 0.8); // Estimate left line position
            } else if (rightSlopes.isEmpty()) {
                rightX = leftX + (matRoi.cols() * 0.8); // Estimate right line position
            }

            double laneCenterX = (leftX + rightX) / 2;
            error = (float) (laneCenterX - roiCenterX);
            turn = Kp * (error / roiCenterX); // Normalize error

            // Draw the center point and error
            Imgproc.circle(matRoi, new Point(laneCenterX, horizonY), 5, new Scalar(0, 255, 0), -1); // Green
            Imgproc.line(matRoi, new Point(roiCenterX, horizonY), new Point(laneCenterX, horizonY), new Scalar(0, 255, 255), 2); // Yellow Error Line

            Log.d(TAG, String.format(Locale.US, "Error: %.2f, Turn: %.2f", error, turn));
            vehicle.setControl(new Control(FORWARD_SPEED + turn, FORWARD_SPEED - turn));

        } else {
            // No lines detected, stop
            Log.d(TAG, "No lines detected.");
            vehicle.setControl(new Control(0, 0));
        }

        // === UPDATE IMAGEVIEW ===
        // Copy the processed ROI back onto the main frame for display
        matRoi.copyTo(new Mat(matFrame, roiRect));

        final Bitmap finalDisplayBitmap = Bitmap.createBitmap(matFrame.cols(), matFrame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matFrame, finalDisplayBitmap);

        if (getActivity() != null) {
            getActivity().runOnUiThread(() -> {
                if (processedImageView != null) {
                    processedImageView.setImageBitmap(finalDisplayBitmap);
                }
            });
        }
    }

    // Helper function to average a list of doubles
    private double average(List<Double> list) {
        if (list.isEmpty()) return 0;
        double sum = 0;
        for (double d : list) {
            sum += d;
        }
        return sum / list.size();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Release Mats
        if (matFrame != null) matFrame.release();
        if (matGray != null) matGray.release();
        if (matCanny != null) matCanny.release();
        if (matRoi != null) matRoi.release();
        if (matHoughLines != null) matHoughLines.release();

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