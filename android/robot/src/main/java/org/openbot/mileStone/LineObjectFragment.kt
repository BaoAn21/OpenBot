package org.openbot.mileStone

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffColorFilter
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenter
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions
import org.openbot.R
import org.openbot.common.CameraFragment
import org.openbot.databinding.FragmentLineObjectBinding
import org.openbot.utils.Constants
import org.openbot.vehicle.Control
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class LineObjectFragment : CameraFragment() {

    // --- View Binding ---
    private var _binding: FragmentLineObjectBinding? = null
    private val binding get() = _binding!!

    // --- State Variables ---
    private var isAutoMode = false
    private var isProcessing = false
    private var isMirrored = false

    // --- ML Kit (Object Detection) ---
    private var segmenter: SubjectSegmenter? = null
    private var stopThreshold = 20 // % of screen filled to trigger STOP
    private var isBlocked = false
    private var currentObjectPercentage = 0.0
    private var rotatedMask: Bitmap? = null
    private var maskTransformMatrix: Matrix? = null

    // --- OpenCV & Line Tracking Config (MATCHING JAVA CODE) ---
    private var scanY = 300
    private var scanHeight = 30
    private var colorThrLow = Scalar(20.0, 100.0, 100.0)
    private var colorThrHi = Scalar(30.0, 255.0, 255.0)
    private val CONFIDENCE_THRESHOLD = 500.0

    // --- Bitmaps & Matrices ---
    private var processedBitmap: Bitmap? = null
    private var inputTransformMatrix: Matrix? = null

    // --- Visualization ---
    private var overlayBitmap: Bitmap? = null
    private var overlayCanvas: Canvas? = null
    private var debugRoiBitmap: Bitmap? = null

    // --- Paints ---
    private lateinit var roiBorderPaint: Paint
    private lateinit var robotDirPaint: Paint
    private lateinit var detectedLinePaint: Paint
    private lateinit var paintMask: Paint // For the object mask

    // --- OpenCV Objects ---
    private var mat: Mat? = null
    private var matHsv: Mat? = null
    private var matMask: Mat? = null
    private var matSlice: Mat? = null
    private var matHist: Mat? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentLineObjectBinding.inflate(inflater, container, false)
        // Use 'binding' (the object), NOT binding.root
        return inflateFragment(binding, inflater, container)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // 1. Init ML Kit
        val options = SubjectSegmenterOptions.Builder().enableForegroundBitmap().build()
        segmenter = SubjectSegmentation.getClient(options)

        // 2. Init OpenCV
        mat = Mat(); matHsv = Mat(); matMask = Mat(); matSlice = Mat(); matHist = Mat()

        // 3. Init Paints (Styles copied from Java)
        roiBorderPaint = Paint().apply {
            color = Color.YELLOW
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }
        robotDirPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }
        detectedLinePaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
        }
        paintMask = Paint().apply {
            alpha = 150
            isFilterBitmap = true
        }

        // 4. UI Listeners
        binding.autoSwitch.setOnClickListener { setAutoMode(binding.autoSwitch.isChecked) }

        binding.cameraToggle.setOnClickListener {
            toggleCamera()
            inputTransformMatrix = null
        }

        binding.mirrorControl.setOnClickListener {
            isMirrored = binding.mirrorControl.isChecked
            inputTransformMatrix = null
        }

        // Sliders & Color Selectors
        setupSlidersAndColors()

        // PIP Callback (Draws the Object Mask)
        binding.safetyPipView.addCallback { canvas ->
            if (rotatedMask != null && maskTransformMatrix != null) {
                val color = if (isBlocked) Color.RED else Color.YELLOW
                paintMask.colorFilter = PorterDuffColorFilter(color, PorterDuff.Mode.SRC_IN)
                canvas.drawBitmap(rotatedMask!!, maskTransformMatrix!!, paintMask)
            }
        }
    }

    private fun setAutoMode(enabled: Boolean) {
        isAutoMode = enabled
        binding.autoSwitch.isChecked = enabled
        if (!enabled) {
            vehicle.setControl(0f, 0f)
            updateUI("PAUSED", Color.GRAY, 0f)
            binding.lineTrackingMainView.setImageBitmap(null)
            binding.safetyPipView.postInvalidate()
        }
    }

    override fun processFrame(image: Bitmap, imageProxy: ImageProxy) {
        if (!isAutoMode || isProcessing) return
        isProcessing = true

        // 1. PREPARE IMAGE (Rotate & Mirror)
        if (inputTransformMatrix == null) {
            inputTransformMatrix = Matrix().apply {
                postRotate(rotationDegrees.toFloat())
                if (isMirrored) postScale(-1f, 1f)
            }
        }

        // Create the bitmap that matches the screen orientation
        processedBitmap = Bitmap.createBitmap(
            image, 0, 0, image.width, image.height, inputTransformMatrix, true
        )

        // 2. OBJECT DETECTION (ML Kit)
        // ML Kit needs rotation info separately
        val mlImage = InputImage.fromBitmap(image, imageProxy.imageInfo.rotationDegrees)

        segmenter?.process(mlImage)
            ?.addOnSuccessListener { result ->
                // --- A. PROCESS SAFETY MASK ---
                val mask = result.foregroundBitmap
                processSafetyMask(mask)

                // --- B. DRIVE CONTROL ---
                if (isBlocked) {
                    // EMERGENCY STOP
                    vehicle.setControl(0f, 0f)
                    updateUI("BLOCKED", Color.RED, 0f)

                    // Still draw the camera feed so screen isn't black
                    drawBlockedOverlay()
                } else {
                    // CLEAR -> RUN LINE TRACKING (Your Java Logic)
                    runLineTrackingLogic()
                }

                isProcessing = false
            }
            ?.addOnFailureListener {
                isProcessing = false
            }
    }

    /**
     * This function mimics your Java 'LineTrackingFragment.java' logic exactly
     */
    private fun runLineTrackingLogic() {
        val bitmap = processedBitmap ?: return
        val w = bitmap.width
        val h = bitmap.height

        // 1. Prepare Canvas
        if (overlayBitmap == null || overlayBitmap!!.width != w || overlayBitmap!!.height != h) {
            overlayBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            overlayCanvas = Canvas(overlayBitmap!!)
        }
        overlayCanvas?.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        // 2. Clamp Scan Area
        if (scanY + scanHeight > h) scanY = h - scanHeight
        if (scanY < 0) scanY = 0

        val roiRect = Rect(0, scanY, w, scanY + scanHeight)

        // 3. OpenCV Processing
        Utils.bitmapToMat(bitmap, mat)

        // Crop
        val openCVRect = org.opencv.core.Rect(roiRect.left, roiRect.top, roiRect.width(), roiRect.height())
        matSlice = Mat(mat, openCVRect)

        // HSV & Threshold
        Imgproc.cvtColor(matSlice, matHsv, Imgproc.COLOR_RGB2HSV)
        Core.inRange(matHsv, colorThrLow, colorThrHi, matMask)

        // 4. Draw Debug Strip (Black/White) onto Screen
        if (debugRoiBitmap == null || debugRoiBitmap!!.width != matMask!!.width() || debugRoiBitmap!!.height != matMask!!.height()) {
            debugRoiBitmap = Bitmap.createBitmap(matMask!!.width(), matMask!!.height(), Bitmap.Config.ARGB_8888)
        }
        Utils.matToBitmap(matMask, debugRoiBitmap)

        overlayCanvas?.drawBitmap(debugRoiBitmap!!, null, roiRect, null)
        overlayCanvas?.drawRect(roiRect, roiBorderPaint)

        // 5. Find Center
        Core.reduce(matMask, matHist, 0, Core.REDUCE_SUM, CvType.CV_32S)
        val mmr = Core.minMaxLoc(matHist)
        val maxVal = mmr.maxVal
        val maxIdx = mmr.maxLoc.x.toFloat()

        val driveCommand: Control
        var error = 0f

        // --- STEERING LOGIC (COPIED FROM JAVA) ---
        if (maxVal > CONFIDENCE_THRESHOLD) {
            // LINE FOUND
            val lineCenterX = maxIdx
            val lineCenterY = (scanY + scanHeight / 2).toFloat()

            // Draw Red Dot
            overlayCanvas?.drawCircle(lineCenterX, lineCenterY, 15f, detectedLinePaint)

            val centerX = w / 2.0f
            error = (lineCenterX - centerX) / (w / 2.0f)

            // The Steering Math
            val turn = error * 1.5f
            var leftSpeed = 0.8f + turn   // <--- UPDATED TO 0.8
            var rightSpeed = 0.8f - turn  // <--- UPDATED TO 0.8

            // Clamping
            leftSpeed = max(-1.0f, min(1.0f, leftSpeed))
            rightSpeed = max(-1.0f, min(1.0f, rightSpeed))

            driveCommand = Control(leftSpeed, rightSpeed)

            // Draw Green Direction Line
            robotDirPaint.color = Color.GREEN
            overlayCanvas?.drawLine(centerX, lineCenterY, lineCenterX, lineCenterY, robotDirPaint)

            updateUI("TRACKING", Color.GREEN, error)
        } else {
            // LINE LOST
            driveCommand = Control(0f, 0f)

            robotDirPaint.color = Color.RED
            overlayCanvas?.drawLine(w / 2f, scanY.toFloat(), w / 2f, (scanY + scanHeight).toFloat(), robotDirPaint)

            updateUI("LOST LINE", Color.YELLOW, 0f)
        }

        vehicle.setControl(driveCommand)
    }

    private fun processSafetyMask(mask: Bitmap?) {
        if (mask == null) {
            isBlocked = false
            currentObjectPercentage = 0.0
            return
        }

        // Calculate blockage percentage
        val w = mask.width
        val h = mask.height
        val pixels = IntArray(w * h)
        mask.getPixels(pixels, 0, w, 0, 0, w, h)

        var filledPixels: Long = 0
        for (i in pixels.indices step 10) {
            if (Color.alpha(pixels[i]) > 0) filledPixels++
        }

        currentObjectPercentage = (filledPixels * 10.0 / (w * h)) * 100.0
        isBlocked = currentObjectPercentage > stopThreshold

        // Prepare Mask for PIP View
        this.rotatedMask = mask
        if (maskTransformMatrix == null && binding.safetyPipView.width > 0) {
            maskTransformMatrix = Matrix()
            val pipW = binding.safetyPipView.width.toFloat()
            val pipH = binding.safetyPipView.height.toFloat()

            // Match mirroring
            if (isMirrored) maskTransformMatrix?.postScale(-1f, 1f, w/2f, h/2f)

            val src = RectF(0f, 0f, w.toFloat(), h.toFloat())
            val dst = RectF(0f, 0f, pipW, pipH)
            maskTransformMatrix?.setRectToRect(src, dst, Matrix.ScaleToFit.CENTER)

            if (isMirrored) maskTransformMatrix?.postScale(-1f, 1f, pipW/2f, pipH/2f)
        }
    }

    private fun drawBlockedOverlay() {
        // If blocked, we still want to see the camera, maybe draw a big "STOP" box?
        // For now, we just refresh the view so it isn't stuck
        if(overlayBitmap != null) {
            binding.lineTrackingMainView.setImageBitmap(overlayBitmap)
        }
    }

    private fun updateUI(status: String, color: Int, error: Float) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread

            binding.statusText.text = status
            binding.statusText.setTextColor(color)
            binding.objectPercentText.text = String.format(Locale.US, "Obj: %.0f%%", currentObjectPercentage)

            binding.controllerContainer.controlInfo.text = String.format(
                Locale.US, "%.1f,%.1f", vehicle.leftSpeed, vehicle.rightSpeed
            )

            binding.safetyPipView.postInvalidate()
            binding.lineTrackingMainView.setImageBitmap(overlayBitmap)
        }
    }

    private fun setupSlidersAndColors() {
        // Setup SeekBars exactly like Java
        binding.scanYSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                scanY = progress
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        binding.scanHeightSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                scanHeight = progress
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        binding.stopThresholdSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                stopThreshold = progress
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        binding.colorSelectorGroup.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.rb_yellow -> {
                    colorThrLow = Scalar(20.0, 100.0, 100.0)
                    colorThrHi = Scalar(30.0, 255.0, 255.0)
                }
                R.id.rb_white -> {
                    colorThrLow = Scalar(0.0, 0.0, 200.0)
                    colorThrHi = Scalar(180.0, 50.0, 255.0)
                }
                R.id.rb_black -> {
                    colorThrLow = Scalar(0.0, 0.0, 0.0)
                    colorThrHi = Scalar(180.0, 255.0, 50.0)
                }
            }
        }
    }

    override fun processControllerKeyData(command: String) {
        if (Constants.CMD_NETWORK == command) setAutoMode(!isAutoMode)
    }

    override fun processUSBData(data: String) {
        activity?.runOnUiThread {
            if (_binding != null) {
                binding.controllerContainer.speedInfo.text = getString(
                    R.string.speedInfo,
                    String.format(Locale.US, "%3.0f,%3.0f", vehicle.leftWheelRpm, vehicle.rightWheelRpm)
                )
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        mat?.release()
        matHsv?.release()
        matMask?.release()
        matSlice?.release()
        matHist?.release()
        segmenter?.close()
        _binding = null
    }
}