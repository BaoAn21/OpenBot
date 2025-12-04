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
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenter
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.openbot.common.CameraFragment
import org.openbot.databinding.FragmentLineObjectTrackingBinding
import org.openbot.utils.Constants
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt
import com.google.android.gms.tasks.Tasks

class LineObjectTrackingFragment : CameraFragment() {
    private val TAG = "LineObjectTrackingFragment"
    private var _binding: FragmentLineObjectTrackingBinding? = null
    private val binding get() = _binding!!

    private var segmenter: SubjectSegmenter? = null

    // --- THREADING & STATE ---
    // Executor for the Slow ML Task
    private val mlExecutor = Executors.newSingleThreadExecutor()
    // Atomic Flag to prevent ML Queue pile-up
    private val isMLProcessing = AtomicBoolean(false)

    // Volatile State Variables (Shared between Threads)
    @Volatile private var latestSteeringLeft = 0f
    @Volatile private var latestSteeringRight = 0f
    @Volatile private var isBlockedByObstacle = false
    @Volatile private var isLineLost = false
    @Volatile private var isAligning = false

    // Config
    private var stopThresholdPercent = 40
    private var isConfigExpanded = false
    private var isAutoMode = false
    private var isMirrored = false

    // Pixel-Based ROI Config
    private var roiWidthPixel = 100  // Width in pixels
    private var roiStartPixel = 0    // Start Position (Left Edge) in pixels
    private var isResolutionInit = false // Flag to init sliders once

    // Driving Parameters
    private var alignmentThresholdDeg = 5.0
    private val PIVOT_SPEED = 1f
    private val DRIVE_SPEED = 0.4f

    // Matrix
    private var frameToViewTransform: Matrix? = null
    private var lastMaskWidth = 0
    private var lastMaskHeight = 0
    private var lastRotation = -1

    // Buffers
    private var grayMat: Mat? = null
    private var cannyMat: Mat? = null
    private var houghLines: Mat? = null
    private var colorMat: Mat? = null
    private var edgeBitmap: Bitmap? = null

    // Paints
    private val borderPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 10f
    }
    private val roiPaint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val robotViewPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(20f, 10f), 0f)
    }

    data class LineResult(
        val bitmap: Bitmap?,
        val angleError: Double?,
        val hasLine: Boolean
    )

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentLineObjectTrackingBinding.inflate(inflater, container, false)
        return inflateFragment(binding, inflater, container)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        if (OpenCVLoader.initDebug()) Log.d(TAG, "OpenCV loaded")

        val options = SubjectSegmenterOptions.Builder()
            .enableForegroundBitmap()
            .build()
        segmenter = SubjectSegmentation.getClient(options)

        setupUI()
    }

    private fun setupUI() {
        binding.autoSwitch.setOnCheckedChangeListener { _, isChecked ->
            isAutoMode = isChecked
            if (!isChecked) {
                vehicle.setControl(0f, 0f)
                binding.stopWarning.visibility = View.GONE
                binding.motorText.text = "Manual Mode"
            }
        }

        binding.mirrorSwitch.setOnCheckedChangeListener { _, isChecked ->
            isMirrored = isChecked
            frameToViewTransform = null
        }

        binding.cameraToggleBtn.setOnClickListener {
            toggleCamera()
            frameToViewTransform = null
            isResolutionInit = false // Reset slider limits for new camera
        }

        binding.thresholdSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                stopThresholdPercent = progress
                binding.thresholdText.text = "Stop if Area > $progress%"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // ROI WIDTH (Pixels)
        binding.roiWidthSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                roiWidthPixel = progress
                binding.roiWidthText.text = "ROI Width: ${progress}px"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // ROI START POS (Left Edge in Pixels)
        binding.roiPosSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                roiStartPixel = progress
                binding.roiPosText.text = "ROI Start: ${progress}px"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        binding.alignAngleSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                alignmentThresholdDeg = progress.toDouble()
                binding.alignAngleText.text = "Align Error: $progress°"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        binding.configHeader.setOnClickListener { toggleConfig(!isConfigExpanded) }
        binding.mainContainer.setOnClickListener { if (isConfigExpanded) toggleConfig(false) }
    }

    private fun toggleConfig(expand: Boolean) {
        isConfigExpanded = expand
        if (isConfigExpanded) {
            binding.configContent.visibility = View.VISIBLE
            binding.configTitle.visibility = View.VISIBLE
            binding.configCard.setCardBackgroundColor(Color.WHITE)
        } else {
            binding.configContent.visibility = View.GONE
            binding.configTitle.visibility = View.GONE
            binding.configCard.setCardBackgroundColor(Color.parseColor("#DDFFFFFF"))
        }
    }

    private fun initSliderLimits(camW: Int, camH: Int, rotation: Int) {
        if (isResolutionInit) return

        val isRotated = rotation % 180 == 90
        val scanDimension = if (isRotated) camH else camW

        activity?.runOnUiThread {
            binding.roiWidthSeekbar.max = scanDimension
            binding.roiPosSeekbar.max = scanDimension

            if (roiWidthPixel == 0) roiWidthPixel = scanDimension / 5
            if (roiStartPixel == 0) roiStartPixel = (scanDimension / 2) - (roiWidthPixel / 2)

            binding.roiWidthSeekbar.progress = roiWidthPixel
            binding.roiPosSeekbar.progress = roiStartPixel
            binding.roiPosText.text = "ROI Start: ${roiStartPixel}px"
            binding.roiWidthText.text = "ROI Width: ${roiWidthPixel}px"
        }
        isResolutionInit = true
    }

    override fun processFrame(image: Bitmap?, imageProxy: ImageProxy?) {
        if (image == null || imageProxy == null || segmenter == null) return

        val sensorRotation = imageProxy.imageInfo.rotationDegrees

        // Init Sliders once
        initSliderLimits(image.width, image.height, sensorRotation)

        // 1. FAST PATH: LINE TRACKING (Runs EVERY Frame)
        val lineResult = processCannyAndHough(image, sensorRotation)
        calculateDriveLogic(lineResult)
        updateVehicleControl()

        activity?.runOnUiThread {
            if(_binding != null) {
                drawROI(lineResult.bitmap, sensorRotation, image.width, image.height)
            }
        }

        // 2. SLOW PATH: ML KIT (Blocking Background Task)
        if (!isMLProcessing.get()) {
            isMLProcessing.set(true)

            // Copy is essential
            val mlBitmap = image.copy(image.config, false)

            mlExecutor.execute {
                try {
                    val inputImage = InputImage.fromBitmap(mlBitmap, sensorRotation)

                    // FIX: Use Tasks.await to force this thread to STOP and WAIT for ML Kit
                    // This ensures we don't release the flag until calculations are actually done.
                    val result = Tasks.await(segmenter!!.process(inputImage))

                    // If we get here, we have the result. Process it immediately.
                    val mask = result.foregroundBitmap
                    if (mask != null) {
                        val w = mask.width
                        val h = mask.height
                        var objectPixels = 0
                        val pixels = IntArray(w * h)
                        mask.getPixels(pixels, 0, w, 0, 0, w, h)

                        for (i in pixels.indices step 10) {
                            if (Color.alpha(pixels[i]) > 0) objectPixels++
                        }
                        val percentage = (objectPixels * 10f / pixels.size.toFloat()) * 100f

                        // Update Shared State
                        isBlockedByObstacle = percentage > stopThresholdPercent

                        // Update UI
                        activity?.runOnUiThread {
                            if (_binding != null) {
                                displayMask(mask, sensorRotation)
                                drawMaskBorder(mask)
                                binding.orientationText.text = "Rot: $sensorRotation"
                                binding.dimensionText.text = "Area: %.1f%%".format(percentage)
                            }
                        }
                        updateVehicleControl()
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "ML Error", e)
                } finally {
                    // NOW this is safe. We only set false after await() returns.
                    isMLProcessing.set(false)
                }
            }
        }
    }

    // --- LOGIC: Calculate Steering State (Fast) ---
    private fun calculateDriveLogic(lineResult: LineResult) {
        val rawAngleError = lineResult.angleError ?: 0.0
        val finalAngleError = if (isMirrored) -rawAngleError else rawAngleError
        val isAligned = lineResult.hasLine && abs(finalAngleError) < alignmentThresholdDeg

        if (!lineResult.hasLine) {
            isLineLost = true
            isAligning = false
            latestSteeringLeft = 0f
            latestSteeringRight = 0f
        } else if (!isAligned) {
            isLineLost = false
            isAligning = true
            // Calculate Pivot
            val direction = if (finalAngleError > 0) 1f else -1f
            latestSteeringLeft = direction * PIVOT_SPEED
            latestSteeringRight = -direction * PIVOT_SPEED

            // Optional: Update text for angle (Need to be on UI thread or shared var, keeping simple here)
            activity?.runOnUiThread {
                if(_binding != null) binding.dimensionText.text = "Err: %.1f°".format(finalAngleError)
            }
        } else {
            isLineLost = false
            isAligning = false
            // Go Forward
            latestSteeringLeft = DRIVE_SPEED
            latestSteeringRight = DRIVE_SPEED
        }
    }

    // --- LOGIC: Process ML Kit (Slow Background) ---
    private fun processMLKit(image: Bitmap, sensorRotation: Int) {
        val inputImage = InputImage.fromBitmap(image, sensorRotation)

        // Note: process() is synchronous here because we are already inside a background Executor
        // But Google ML Kit API is Task based, so we block-wait or use listeners.
        // Since we are in an executor, listeners are fine.

        segmenter?.process(inputImage)
            ?.addOnSuccessListener { result ->
                val mask = result.foregroundBitmap
                if (mask != null) {
                    val w = mask.width
                    val h = mask.height
                    var objectPixels = 0
                    val pixels = IntArray(w * h)
                    mask.getPixels(pixels, 0, w, 0, 0, w, h)

                    for (i in pixels.indices step 10) {
                        if (Color.alpha(pixels[i]) > 0) objectPixels++
                    }
                    val percentage = (objectPixels * 10f / pixels.size.toFloat()) * 100f

                    // Update Shared State
                    isBlockedByObstacle = percentage > stopThresholdPercent

                    // Update UI for ML
                    activity?.runOnUiThread {
                        if (_binding != null) {
                            displayMask(mask, sensorRotation)
                            drawMaskBorder(mask)
                            binding.orientationText.text = "Rot: $sensorRotation"
                            binding.dimensionText.text = "Area: %.1f%%".format(percentage)
                        }
                    }

                    // Trigger a control update in case obstacle status changed while we were processing
                    updateVehicleControl()
                }
            }
            ?.addOnFailureListener { e ->
                Log.e(TAG, "Segmentation failed", e)
            }
        // We do not recycle mlBitmap here as ML Kit might still be using internal refs
        // until the task completes, but usually safe to let GC handle it.
    }

    // --- CENTRAL CONTROL HUB ---
    private fun updateVehicleControl() {
        if (!isAutoMode) return

        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread

            // Priority 1: Obstacle (Safety)
            if (isBlockedByObstacle) {
                vehicle.setControl(0f, 0f)
                binding.stopWarning.visibility = View.VISIBLE
                binding.stopWarning.text = "BLOCKED"
                binding.stopWarning.setTextColor(Color.RED)
                binding.motorText.text = "L: 0.0  R: 0.0"
            }
            // Priority 2: Line Lost
            else if (isLineLost) {
                vehicle.setControl(0f, 0f)
                binding.stopWarning.visibility = View.VISIBLE
                binding.stopWarning.text = "NO LINE"
                binding.stopWarning.setTextColor(Color.RED)
                binding.motorText.text = "L: 0.0  R: 0.0"
            }
            // Priority 3: Aligning
            else if (isAligning) {
                vehicle.setControl(latestSteeringLeft, latestSteeringRight)
                binding.stopWarning.visibility = View.VISIBLE
                binding.stopWarning.text = "ALIGNING"
                binding.stopWarning.setTextColor(Color.YELLOW)
                binding.motorText.text = "L: %.2f  R: %.2f".format(latestSteeringLeft, latestSteeringRight)
            }
            // Priority 4: Drive
            else {
                vehicle.setControl(latestSteeringLeft, latestSteeringRight)
                binding.stopWarning.visibility = View.GONE
                binding.motorText.text = "L: %.2f  R: %.2f".format(latestSteeringLeft, latestSteeringRight)
            }
        }
    }

    private fun processCannyAndHough(image: Bitmap, rotation: Int): LineResult {
        try {
            val camW = image.width
            val camH = image.height
            val isRotated = rotation % 180 == 90

            val scanDimension = if (isRotated) camH else camW
            val maxStart = (scanDimension - roiWidthPixel).coerceAtLeast(0)
            val validStart = roiStartPixel.coerceIn(0, maxStart)

            val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
            val shouldMirror = isFront || isMirrored

            val effectiveStart = if (shouldMirror) {
                scanDimension - validStart - roiWidthPixel
            } else {
                validStart
            }

            val cropRect: Rect
            if (isRotated) {
                val top = effectiveStart.coerceIn(0, camH)
                val bottom = (effectiveStart + roiWidthPixel).coerceIn(0, camH)
                cropRect = Rect(0, top, camW, bottom)
            } else {
                val left = effectiveStart.coerceIn(0, camW)
                val right = (effectiveStart + roiWidthPixel).coerceIn(0, camW)
                cropRect = Rect(left, 0, right, camH)
            }

            if (cropRect.width() <= 0 || cropRect.height() <= 0)
                return LineResult(null, null, false)

            val croppedBmp = Bitmap.createBitmap(image, cropRect.left, cropRect.top, cropRect.width(), cropRect.height())

            if (grayMat == null) grayMat = Mat()
            if (cannyMat == null) cannyMat = Mat()
            if (houghLines == null) houghLines = Mat()
            if (colorMat == null) colorMat = Mat()

            Utils.bitmapToMat(croppedBmp, grayMat)
            Imgproc.cvtColor(grayMat, grayMat, Imgproc.COLOR_RGB2GRAY)
            Imgproc.GaussianBlur(grayMat, grayMat, org.opencv.core.Size(5.0, 5.0), 0.0)
            Imgproc.Canny(grayMat, cannyMat, 50.0, 150.0)

            Imgproc.HoughLinesP(cannyMat, houghLines, 1.0, Math.PI / 180, 50, 50.0, 10.0)
            Imgproc.cvtColor(cannyMat, colorMat, Imgproc.COLOR_GRAY2RGB)

            var maxLen = 0.0
            var bestLine: IntArray? = null
            var angleError = 0.0
            val targetAngle = if (isRotated) 0.0 else 90.0

            for (i in 0 until houghLines!!.rows()) {
                val l = houghLines!!.get(i, 0)
                val x1 = l[0]; val y1 = l[1]; val x2 = l[2]; val y2 = l[3]
                val dx = (x2 - x1).toDouble()
                val dy = (y2 - y1).toDouble()
                val len = sqrt(dx * dx + dy * dy)

                var angle = Math.toDegrees(atan2(dy, dx))
                if (angle < 0) angle += 180.0

                var diff = abs(angle - targetAngle)
                if (diff > 90) diff = abs(diff - 180)

                if (diff < 30) {
                    if (len > maxLen) {
                        maxLen = len
                        bestLine = intArrayOf(x1.toInt(), y1.toInt(), x2.toInt(), y2.toInt())
                        angleError = if (isRotated) {
                            if (angle > 90) angle - 180 else angle
                        } else {
                            angle - 90
                        }
                    }
                    Imgproc.line(colorMat, Point(x1.toDouble(), y1.toDouble()), Point(x2.toDouble(), y2.toDouble()), Scalar(100.0, 100.0, 100.0), 2)
                }
            }

            if (bestLine != null) {
                Imgproc.line(colorMat, Point(bestLine[0].toDouble(), bestLine[1].toDouble()), Point(bestLine[2].toDouble(), bestLine[3].toDouble()), Scalar(255.0, 0.0, 0.0), 8)
            }

            if (edgeBitmap == null || edgeBitmap?.width != colorMat?.cols() || edgeBitmap?.height != colorMat?.rows()) {
                edgeBitmap = Bitmap.createBitmap(colorMat!!.cols(), colorMat!!.rows(), Bitmap.Config.ARGB_8888)
            }
            Utils.matToBitmap(colorMat, edgeBitmap)

            return LineResult(edgeBitmap, angleError, bestLine != null)
        } catch (e: Exception) {
            return LineResult(null, 0.0, false)
        }
    }

    private fun drawROI(cannyEdgeBitmap: Bitmap?, rotation: Int, camW: Int, camH: Int) {
        if (_binding == null) return
        val viewW = binding.roiOverlay.width
        val viewH = binding.roiOverlay.height
        if (viewW == 0 || viewH == 0) return

        val overlayBitmap = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(overlayBitmap)

        val isRotated = rotation % 180 == 90
        val scanDimension = if (isRotated) camH else camW
        val screenDimension = viewW.toFloat()
        val scale = screenDimension / scanDimension.toFloat()

        val roiW_Screen = roiWidthPixel * scale
        val maxStart = (scanDimension - roiWidthPixel).coerceAtLeast(0)
        val validStart = roiStartPixel.coerceIn(0, maxStart)
        val roiStart_Screen = validStart * scale

        val left = roiStart_Screen
        val right = roiStart_Screen + roiW_Screen
        val destRect = RectF(left, 0f, right, viewH.toFloat())

        if (cannyEdgeBitmap != null) {
            val m = Matrix()
            val bmpW = cannyEdgeBitmap.width.toFloat()
            val bmpH = cannyEdgeBitmap.height.toFloat()

            m.postTranslate(-bmpW / 2f, -bmpH / 2f)
            m.postRotate(rotation.toFloat())

            val rotatedW = if (isRotated) bmpH else bmpW
            val rotatedH = if (isRotated) bmpW else bmpH

            val scaleX = destRect.width() / rotatedW
            val scaleY = destRect.height() / rotatedH
            m.postScale(scaleX, scaleY)

            val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
            if (isFront || isMirrored) {
                m.postScale(-1f, 1f)
            }

            m.postTranslate(destRect.centerX(), destRect.centerY())
            canvas.drawBitmap(cannyEdgeBitmap, m, null)
        }

        canvas.drawRect(destRect, roiPaint)
        val refX = destRect.centerX()
        canvas.drawLine(refX, destRect.top, refX, destRect.bottom, robotViewPaint)
        binding.roiOverlay.setImageBitmap(overlayBitmap)
    }

    private fun displayMask(mask: Bitmap, sensorRotation: Int) {
        if (_binding == null) return
        val viewW = binding.maskOverlay.width.toFloat()
        val viewH = binding.maskOverlay.height.toFloat()
        if (viewW == 0f) return

        if (frameToViewTransform == null ||
            lastMaskWidth != mask.width ||
            lastMaskHeight != mask.height ||
            lastRotation != sensorRotation) {

            frameToViewTransform = Matrix()
            val maskW = mask.width.toFloat()
            val maskH = mask.height.toFloat()

            frameToViewTransform?.postTranslate(-maskW / 2f, -maskH / 2f)
            frameToViewTransform?.postRotate(sensorRotation.toFloat())

            val isRotated = sensorRotation % 180 == 90
            val rotatedW = if (isRotated) maskH else maskW
            val rotatedH = if (isRotated) maskW else maskH

            val scaleX = viewW / rotatedW
            val scaleY = viewH / rotatedH
            val scale = kotlin.math.max(scaleX, scaleY)

            frameToViewTransform?.postScale(scale, scale)
            frameToViewTransform?.postTranslate(viewW / 2f, viewH / 2f)

            val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
            if (isFront || isMirrored) {
                frameToViewTransform?.postScale(-1f, 1f, viewW / 2f, viewH / 2f)
            }

            lastMaskWidth = mask.width
            lastMaskHeight = mask.height
            lastRotation = sensorRotation
        }

        binding.maskOverlay.imageMatrix = frameToViewTransform
        binding.maskOverlay.colorFilter = PorterDuffColorFilter(Color.MAGENTA, PorterDuff.Mode.SRC_IN)
        binding.maskOverlay.setImageBitmap(mask)
    }

    private fun drawMaskBorder(mask: Bitmap) {
        if (_binding == null || frameToViewTransform == null) return
        val viewW = binding.rectOverlay.width
        val viewH = binding.rectOverlay.height
        if (viewW == 0 || viewH == 0) return

        val overlayBitmap = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(overlayBitmap)
        val maskRect = RectF(0f, 0f, mask.width.toFloat(), mask.height.toFloat())
        frameToViewTransform?.mapRect(maskRect)
        canvas.drawRect(maskRect, borderPaint)
        binding.rectOverlay.setImageBitmap(overlayBitmap)
    }

    override fun processControllerKeyData(command: String?) {
        if (Constants.CMD_NETWORK == command) {
            activity?.runOnUiThread {
                binding.autoSwitch.isChecked = !binding.autoSwitch.isChecked
            }
        }
    }

    override fun processUSBData(data: String?) { }

    override fun onDestroyView() {
        super.onDestroyView()
        mlExecutor.shutdown()
        segmenter?.close()
        segmenter = null
        _binding = null
    }
}