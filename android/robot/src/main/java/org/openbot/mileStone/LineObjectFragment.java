package org.openbot.mileStone;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenter;
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions;

import org.openbot.R;
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentLineObjectBinding;
import org.openbot.utils.Constants;
import org.openbot.vehicle.Control;

// OpenCV
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Locale;

public class LineObjectFragment extends CameraFragment {
    private FragmentLineObjectBinding binding;
    private boolean isAutoMode = false;
    private boolean isProcessing = false;

    // --- AI / Safety Variables ---
    private SubjectSegmenter segmenter;
    private int stopThreshold = 20; // % of screen filled to trigger STOP
    private boolean isBlocked = false;
    private double currentObjectPercentage = 0.0;

    // --- Line Tracking Variables ---
    private int scanY = 300;
    private int scanHeight = 30; // Default thickness
    private Scalar colorThrLow = new Scalar(20, 100, 100); // Default Yellow
    private Scalar colorThrHi = new Scalar(30, 255, 255);
    private final double LINE_CONFIDENCE = 500;

    // --- Matrix & Bitmaps ---
    private Matrix inputTransformMatrix; // Rotates camera to upright
    private Matrix maskTransformMatrix;  // Scales mask to PIP view
    private Bitmap processedBitmap;      // The upright image used for OpenCV
    private Bitmap rotatedMask;          // The mask from ML Kit

    // --- Visualization (Main Line View) ---
    private Bitmap mainOverlayBitmap;
    private Canvas mainOverlayCanvas;
    private Paint paintRoiBorder;
    private Paint paintLineVector;
    private Bitmap debugRoiBitmap; // The black/white strip

    // --- Visualization (PIP Safety View) ---
    private Paint paintMask;
    private boolean isMirrored = false;

    // --- OpenCV Mats ---
    private Mat mat, matHsv, matMask, matSlice, matHist;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentLineObjectBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // 1. Init ML Kit
        SubjectSegmenterOptions options = new SubjectSegmenterOptions.Builder()
                .enableForegroundBitmap()
                .build();
        segmenter = SubjectSegmentation.getClient(options);

        // 2. Init OpenCV Mats
        mat = new Mat(); matHsv = new Mat(); matMask = new Mat(); matSlice = new Mat(); matHist = new Mat();

        // 3. Init Paints
        paintMask = new Paint();
        paintMask.setAlpha(150); // Semi-transparent mask
        paintMask.setFilterBitmap(true);

        paintRoiBorder = new Paint();
        paintRoiBorder.setColor(Color.YELLOW);
        paintRoiBorder.setStyle(Paint.Style.STROKE);
        paintRoiBorder.setStrokeWidth(5);

        paintLineVector = new Paint();
        paintLineVector.setColor(Color.GREEN);
        paintLineVector.setStyle(Paint.Style.STROKE);
        paintLineVector.setStrokeWidth(8);

        // --- UI LISTENERS ---

        // Autopilot
        binding.autoSwitch.setOnClickListener(v -> setAutoMode(binding.autoSwitch.isChecked()));

        // Camera & Mirror
        binding.cameraToggle.setOnClickListener(v -> { toggleCamera(); inputTransformMatrix = null; });
        binding.mirrorControl.setOnClickListener(v -> {
            isMirrored = binding.mirrorControl.isChecked();
            inputTransformMatrix = null;
        });

        // Sliders
        binding.stopThresholdSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                stopThreshold = progress;
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        binding.scanYSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                scanY = progress;
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        binding.scanHeightSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                scanHeight = progress;
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Color Selector
        binding.colorSelectorGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.rb_yellow) { colorThrLow = new Scalar(20, 100, 100); colorThrHi = new Scalar(30, 255, 255); }
            else if (checkedId == R.id.rb_white) { colorThrLow = new Scalar(0, 0, 200); colorThrHi = new Scalar(180, 50, 255); }
            else if (checkedId == R.id.rb_black) { colorThrLow = new Scalar(0, 0, 0); colorThrHi = new Scalar(180, 255, 50); }
        });

        // --- MASK DRAWING CALLBACK (PIP View) ---
        binding.safetyPipView.addCallback(canvas -> {
            if (rotatedMask != null && maskTransformMatrix != null) {
                // Draw Red Mask if blocked, Green if clear? Or just Red.
                int color = isBlocked ? Color.RED : Color.YELLOW;
                paintMask.setColorFilter(new PorterDuffColorFilter(color, PorterDuff.Mode.SRC_IN));
                canvas.drawBitmap(rotatedMask, maskTransformMatrix, paintMask);
            }
        });
    }

    private void setAutoMode(boolean enabled) {
        isAutoMode = enabled;
        binding.autoSwitch.setChecked(enabled);
        if (!enabled) {
            vehicle.setControl(0, 0);
            binding.statusText.setText("PAUSED");
            binding.statusText.setTextColor(Color.GRAY);
            binding.safetyPipView.postInvalidate(); // Clear mask
            binding.lineTrackingMainView.setImageBitmap(null); // Clear line view
        }
    }

    @Override
    protected void processFrame(Bitmap image, ImageProxy imageProxy) {
        if (!isAutoMode || isProcessing) return;
        isProcessing = true;

        // --- 1. PREPARE INPUT IMAGES ---

        // A. For OpenCV (Line): Rotate & Mirror input to match screen
        if (inputTransformMatrix == null) {
            inputTransformMatrix = new Matrix();
            inputTransformMatrix.postRotate(getRotationDegrees());
            if (isMirrored) inputTransformMatrix.postScale(-1, 1);
        }
        processedBitmap = Bitmap.createBitmap(image, 0, 0, image.getWidth(), image.getHeight(), inputTransformMatrix, true);

        // B. For ML Kit (Safety): Use live rotation from proxy
        int mlKitRotation = imageProxy.getImageInfo().getRotationDegrees();
        InputImage mlImage = InputImage.fromBitmap(image, mlKitRotation);

        // --- 2. RUN SAFETY CHECK (ML KIT) ---
        segmenter.process(mlImage)
                .addOnSuccessListener(result -> {
                    Bitmap mask = result.getForegroundBitmap();
                    if (mask == null) {
                        isBlocked = false;
                        currentObjectPercentage = 0.0;
                    } else {
                        // Check blockage size (Sample every 10th pixel)
                        int w = mask.getWidth();
                        int h = mask.getHeight();
                        int[] pixels = new int[w * h];
                        mask.getPixels(pixels, 0, w, 0, 0, w, h);

                        long filledPixels = 0;
                        for (int i = 0; i < pixels.length; i+=10) {
                            if (Color.alpha(pixels[i]) > 0) filledPixels++;
                        }

                        currentObjectPercentage = (filledPixels * 10.0 / (w * h)) * 100.0;
                        isBlocked = currentObjectPercentage > stopThreshold;

                        // Save for PIP display
                        this.rotatedMask = mask;

                        // Calculate Matrix to fit mask into PIP View
                        if (maskTransformMatrix == null && binding.safetyPipView.getWidth() > 0) {
                            maskTransformMatrix = new Matrix();
                            float pipW = binding.safetyPipView.getWidth();
                            float pipH = binding.safetyPipView.getHeight();

                            // 1. Mirror (Match OpenCV)
                            if (isMirrored) maskTransformMatrix.postScale(-1, 1, w/2f, h/2f);

                            // 2. Scale to Fit PIP
                            RectF src = new RectF(0, 0, w, h);
                            RectF dst = new RectF(0, 0, pipW, pipH);
                            maskTransformMatrix.setRectToRect(src, dst, Matrix.ScaleToFit.CENTER);

                            // Re-apply mirror
                            if (isMirrored) maskTransformMatrix.postScale(-1, 1, pipW/2f, pipH/2f);
                        }
                    }

                    // --- 3. DECIDE & DRIVE ---

                    if (isBlocked) {
                        // SAFETY STOP
                        vehicle.setControl(0, 0);
                        updateUI("BLOCKED", Color.RED);
                    } else {
                        // LINE TRACKING (OpenCV)
                        runLineTrackingLogic();
                    }

                    isProcessing = false;
                })
                .addOnFailureListener(e -> {
                    isProcessing = false;
                });
    }

    private void runLineTrackingLogic() {
        // Logic is applied on 'processedBitmap' (the upright, mirrored image)
        int w = processedBitmap.getWidth();
        int h = processedBitmap.getHeight();

        // Prepare Main Visualization Canvas
        if (mainOverlayBitmap == null || mainOverlayBitmap.getWidth() != w || mainOverlayBitmap.getHeight() != h) {
            mainOverlayBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            mainOverlayCanvas = new Canvas(mainOverlayBitmap);
        }
        mainOverlayCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

        // Safety clamp
        if (scanY + scanHeight > h) scanY = h - scanHeight;
        if (scanY < 0) scanY = 0;

        Rect roiRect = new Rect(0, scanY, w, scanY + scanHeight);

        // OpenCV
        Utils.bitmapToMat(processedBitmap, mat);
        matSlice = new Mat(mat, new org.opencv.core.Rect(roiRect.left, roiRect.top, roiRect.width(), roiRect.height()));
        Imgproc.cvtColor(matSlice, matHsv, Imgproc.COLOR_RGB2HSV);
        Core.inRange(matHsv, colorThrLow, colorThrHi, matMask);

        // --- Visual Debug: Draw ROI Strip onto Main Canvas ---
        if (debugRoiBitmap == null || debugRoiBitmap.getWidth() != matMask.width() || debugRoiBitmap.getHeight() != matMask.height()) {
            debugRoiBitmap = Bitmap.createBitmap(matMask.width(), matMask.height(), Bitmap.Config.ARGB_8888);
        }
        Utils.matToBitmap(matMask, debugRoiBitmap);

        // Draw the B/W strip into the main view
        mainOverlayCanvas.drawBitmap(debugRoiBitmap, null, roiRect, null);
        mainOverlayCanvas.drawRect(roiRect, paintRoiBorder);

        // Calculate Line Center
        Core.reduce(matMask, matHist, 0, Core.REDUCE_SUM, CvType.CV_32S);
        Core.MinMaxLocResult mmr = Core.minMaxLoc(matHist);

        Control driveCommand;

        if (mmr.maxVal > LINE_CONFIDENCE) {
            // Line Found
            float lineX = (float) mmr.maxLoc.x;
            float centerX = w / 2.0f;
            float error = (lineX - centerX) / (w / 2.0f);

            // Draw Vector Line (Center to Line)
            mainOverlayCanvas.drawLine(centerX, (float)(scanY + scanHeight/2), lineX, (float)(scanY + scanHeight/2), paintLineVector);
            mainOverlayCanvas.drawCircle(lineX, (float)(scanY + scanHeight/2), 10, paintLineVector);

            float turn = error * 1.5f; // Gain
            float left = 0.6f + turn;
            float right = 0.6f - turn;

            // Clamp
            left = Math.max(-1f, Math.min(1f, left));
            right = Math.max(-1f, Math.min(1f, right));

            driveCommand = new Control(left, right);
            updateUI("TRACKING", Color.GREEN);
        } else {
            // Line Lost
            driveCommand = new Control(0, 0);
            paintLineVector.setColor(Color.RED);
            mainOverlayCanvas.drawLine(w/2f, scanY, w/2f, scanY+scanHeight, paintLineVector);
            paintLineVector.setColor(Color.GREEN); // Reset for next frame
            updateUI("LOST LINE", Color.YELLOW);
        }

        vehicle.setControl(driveCommand);
    }

    private void updateUI(String status, int color) {
        if (getActivity() == null) return;
        getActivity().runOnUiThread(() -> {
            if (binding == null) return;

            // Update Text
            binding.statusText.setText(status);
            binding.statusText.setTextColor(color);

            // Update Object Percentage (NEW)
            binding.objectPercentText.setText(String.format(Locale.US, "Obj Area: %.0f%%", currentObjectPercentage));

            // Update Speed Info (Bottom Bar - Standard OpenBot UI)
            binding.controllerContainer.controlInfo.setText(
                    String.format(Locale.US, "%.2f,%.2f", vehicle.getLeftSpeed(), vehicle.getRightSpeed()));

            // Update PIP Mask
            binding.safetyPipView.postInvalidate();

            // Update Main Line View
            binding.lineTrackingMainView.setImageBitmap(mainOverlayBitmap);
        });
    }

    @Override
    protected void processControllerKeyData(String command) {
        if (Constants.CMD_NETWORK.equals(command)) setAutoMode(!isAutoMode);
    }

    @Override
    protected void processUSBData(String data) {
        // Handle incoming USB data (Speed from wheel encoders)
        if (binding != null && binding.controllerContainer != null) {
            binding.controllerContainer.speedInfo.setText(
                    getString(
                            R.string.speedInfo,
                            String.format(
                                    Locale.US, "%3.0f,%3.0f", vehicle.getLeftWheelRpm(), vehicle.getRightWheelRpm())));
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (mat != null) mat.release();
        // Release other Mats...
        if (segmenter != null) segmenter.close();
        binding = null;
    }
}