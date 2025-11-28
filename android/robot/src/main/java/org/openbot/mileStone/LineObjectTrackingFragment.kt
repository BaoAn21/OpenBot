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
import org.opencv.core.Scalar
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import org.openbot.common.CameraFragment
import org.openbot.databinding.FragmentLineObjectTrackingBinding
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

class LineObjectTrackingFragment : CameraFragment() {
    private val TAG = "LineObjectTrackingFragment"
    private var _binding: FragmentLineObjectTrackingBinding? = null
    private val binding get() = _binding!!

    private var segmenter: SubjectSegmenter? = null
    private var isProcessing = false

    // Config
    private var stopThresholdPercent = 40
    private var isConfigExpanded = false
    private var isAutoMode = false
    private var roiWidthPercent = 20
    private var roiCenterPercent = 50

    // Matrix
    private var frameToViewTransform: Matrix? = null
    private var lastMaskWidth = 0
    private var lastMaskHeight = 0
    private var lastRotation = -1

    // Buffers
    private var grayMat: Mat? = null
    private var cannyMat: Mat? = null
    private var houghLines: Mat? = null
    private var colorMat: Mat? = null // To draw colored lines on OpenCV
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
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(20f, 10f), 0f) // Dashed line
    }

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
            }
        }

        binding.cameraToggleBtn.setOnClickListener {
            toggleCamera()
            frameToViewTransform = null
        }

        binding.thresholdSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                stopThresholdPercent = progress
                binding.thresholdText.text = "Stop if Area > $progress%"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        binding.roiWidthSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                roiWidthPercent = progress
                binding.roiWidthText.text = "ROI Width: $progress%"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        binding.roiPosSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                roiCenterPercent = progress
                binding.roiPosText.text = "ROI Center: $progress%"
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

    override fun processFrame(image: Bitmap?, imageProxy: ImageProxy?) {
        if (image == null || imageProxy == null || segmenter == null || isProcessing) return

        isProcessing = true
        val sensorRotation = imageProxy.imageInfo.rotationDegrees

        // 1. Process Line Detection (Hough)
        // Returns a Bitmap with the Best Line drawn in RED
        val cannyResult = processCannyAndHough(image, sensorRotation)

        // 2. Process Object Detection (ML Kit)
        val inputImage = InputImage.fromBitmap(image, sensorRotation)
        segmenter?.process(inputImage)
            ?.addOnSuccessListener { result ->
                isProcessing = false
                val mask = result.foregroundBitmap

                if (mask != null && _binding != null) {

                    val w = mask.width
                    val h = mask.height
                    var objectPixels = 0
                    val pixels = IntArray(w * h)
                    mask.getPixels(pixels, 0, w, 0, 0, w, h)

                    for (i in pixels.indices step 10) {
                        if (Color.alpha(pixels[i]) > 0) objectPixels++
                    }
                    val percentage = (objectPixels * 10f / pixels.size.toFloat()) * 100f

                    activity?.runOnUiThread {
                        if (_binding == null) return@runOnUiThread

                        // Driving Logic
                        if (isAutoMode) {
                            if (percentage > stopThresholdPercent) {
                                vehicle.setControl(0f, 0f)
                                binding.stopWarning.visibility = View.VISIBLE
                                binding.stopWarning.text = "BLOCKED"
                            } else {
                                // TODO: Use the Angle from Hough to steer here!
                                vehicle.setControl(0.5f, 0.5f)
                                binding.stopWarning.visibility = View.GONE
                            }
                        } else {
                            binding.stopWarning.visibility = View.GONE
                        }

                        // Drawing
                        displayMask(mask, sensorRotation)
                        drawMaskBorder(mask)
                        drawROI(cannyResult, sensorRotation)

                        binding.orientationText.text = "Rot: $sensorRotation"
                        binding.dimensionText.text = "Area: %.1f%%".format(percentage)
                    }
                }
            }
            ?.addOnFailureListener { e ->
                isProcessing = false
                Log.e(TAG, "Segmentation failed", e)
            }
    }

    private fun processCannyAndHough(image: Bitmap, rotation: Int): Bitmap? {
        try {
            val camW = image.width
            val camH = image.height
            val isRotated = rotation % 180 == 90

            val cropRect: Rect

            if (isRotated) {
                // Portrait: Crop Horizontal Strip
                val stripThickness = camH * (roiWidthPercent / 100f)
                val centerPos = camH * (roiCenterPercent / 100f)
                val top = (centerPos - stripThickness / 2).toInt().coerceIn(0, camH)
                val bottom = (centerPos + stripThickness / 2).toInt().coerceIn(0, camH)
                cropRect = Rect(0, top, camW, bottom)
            } else {
                // Landscape: Crop Vertical Strip
                val stripThickness = camW * (roiWidthPercent / 100f)
                val centerPos = camW * (roiCenterPercent / 100f)
                val left = (centerPos - stripThickness / 2).toInt().coerceIn(0, camW)
                val right = (centerPos + stripThickness / 2).toInt().coerceIn(0, camW)
                cropRect = Rect(left, 0, right, camH)
            }

            if (cropRect.width() <= 0 || cropRect.height() <= 0) return null
            val croppedBmp = Bitmap.createBitmap(image, cropRect.left, cropRect.top, cropRect.width(), cropRect.height())

            // Initialize Mats
            if (grayMat == null) grayMat = Mat()
            if (cannyMat == null) cannyMat = Mat()
            if (houghLines == null) houghLines = Mat()
            if (colorMat == null) colorMat = Mat()

            // 1. Preprocessing
            Utils.bitmapToMat(croppedBmp, grayMat)
            Imgproc.cvtColor(grayMat, grayMat, Imgproc.COLOR_RGB2GRAY)
            Imgproc.GaussianBlur(grayMat, grayMat, org.opencv.core.Size(5.0, 5.0), 0.0)

            // 2. Canny Edge Detection
            Imgproc.Canny(grayMat, cannyMat, 50.0, 150.0)

            // 3. Hough Lines
            // threshold=50, minLineLength=50, maxLineGap=10
            Imgproc.HoughLinesP(cannyMat, houghLines, 1.0, Math.PI / 180, 50, 50.0, 10.0)

            // 4. Find Best Line
            var maxLen = 0.0
            var bestLine: IntArray? = null

            for (i in 0 until houghLines!!.rows()) {
                val l = houghLines!!.get(i, 0) // returns [x1, y1, x2, y2]
                val x1 = l[0]
                val y1 = l[1]
                val x2 = l[2]
                val y2 = l[3]

                val dx = x2 - x1
                val dy = y2 - y1
                val len = sqrt(dx * dx + dy * dy)

                // 5. Filter Slope based on Orientation
                var isLineGood = false
                if (isRotated) {
                    // Portrait: Sensor is Sideways.
                    // "Straight Up" in world = "Left-Right" in image.
                    // We want Horizontal lines (dy is small, dx is big)
                    if (abs(dx) > abs(dy)) isLineGood = true
                } else {
                    // Landscape: Sensor is Upright.
                    // "Straight Up" in world = "Up-Down" in image.
                    // We want Vertical lines (dy is big, dx is small)
                    if (abs(dy) > abs(dx)) isLineGood = true
                }

                if (isLineGood && len > maxLen) {
                    maxLen = len
                    bestLine = intArrayOf(x1.toInt(), y1.toInt(), x2.toInt(), y2.toInt())
                }
            }

            // 6. Draw on Color Mat (So we can see Red lines)
            // Convert grayscale/canny back to RGB so we can draw colored lines
            Imgproc.cvtColor(cannyMat, colorMat, Imgproc.COLOR_GRAY2RGB)

            if (bestLine != null) {
                // Draw Best Line in RED (Thick)
                Imgproc.line(
                    colorMat,
                    Point(bestLine[0].toDouble(), bestLine[1].toDouble()),
                    Point(bestLine[2].toDouble(), bestLine[3].toDouble()),
                    Scalar(255.0, 0.0, 0.0), 5
                )
            }

            // 7. Output to Bitmap
            if (edgeBitmap == null || edgeBitmap?.width != colorMat?.cols() || edgeBitmap?.height != colorMat?.rows()) {
                edgeBitmap = Bitmap.createBitmap(colorMat!!.cols(), colorMat!!.rows(), Bitmap.Config.ARGB_8888)
            }
            Utils.matToBitmap(colorMat, edgeBitmap)
            return edgeBitmap

        } catch (e: Exception) { return null }
    }

    private fun drawROI(cannyEdgeBitmap: Bitmap?, rotation: Int) {
        if (_binding == null) return
        val viewW = binding.roiOverlay.width
        val viewH = binding.roiOverlay.height
        if (viewW == 0 || viewH == 0) return

        val overlayBitmap = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(overlayBitmap)

        val roiW = viewW * (roiWidthPercent / 100f)
        val centerX = viewW * (roiCenterPercent / 100f)
        val left = centerX - (roiW / 2)
        val right = centerX + (roiW / 2)
        val destRect = RectF(left, 0f, right, viewH.toFloat())

        // 1. Draw Canny Image (with Red Line inside)
        if (cannyEdgeBitmap != null) {
            val m = Matrix()
            val bmpW = cannyEdgeBitmap.width.toFloat()
            val bmpH = cannyEdgeBitmap.height.toFloat()

            m.postTranslate(-bmpW / 2f, -bmpH / 2f)
            m.postRotate(rotation.toFloat())

            val isRotated = rotation % 180 == 90
            val rotatedW = if (isRotated) bmpH else bmpW
            val rotatedH = if (isRotated) bmpW else bmpH

            val scaleX = destRect.width() / rotatedW
            val scaleY = destRect.height() / rotatedH
            m.postScale(scaleX, scaleY)

            val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
            if (isFront) {
                m.postScale(-1f, 1f)
            }

            m.postTranslate(destRect.centerX(), destRect.centerY())
            canvas.drawBitmap(cannyEdgeBitmap, m, null)
        }

        // 2. Draw Blue Border
        canvas.drawRect(destRect, roiPaint)

        // 3. Draw "Robot View" Green Line (Reference)
        // Always vertical in the center of the ROI
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

            if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
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
        if (org.openbot.utils.Constants.CMD_NETWORK == command) {
            activity?.runOnUiThread {
                binding.autoSwitch.isChecked = !binding.autoSwitch.isChecked
            }
        }
    }

    override fun processUSBData(data: String?) { }

    override fun onDestroyView() {
        super.onDestroyView()
        segmenter?.close()
        segmenter = null
        _binding = null
    }
}