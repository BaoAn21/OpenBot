package org.openbot.mileStone

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffColorFilter
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
import org.openbot.common.CameraFragment
import org.openbot.databinding.FragmentLineObjectTrackingBinding
import org.openbot.utils.Constants

class LineObjectTrackingFragment : CameraFragment() {
    private val TAG = "LineObjectTrackingFragment"
    private var _binding: FragmentLineObjectTrackingBinding? = null
    private val binding get() = _binding!!

    private var segmenter: SubjectSegmenter? = null
    private var isProcessing = false

    // Configuration
    private var stopThresholdPercent = 40 // Default 40%
    private var isConfigExpanded = false
    private var isAutoMode = false

    // Matrix
    private var frameToViewTransform: Matrix? = null
    private var lastMaskWidth = 0
    private var lastMaskHeight = 0
    private var lastRotation = -1

    // Paint
    private val borderPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 10f
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentLineObjectTrackingBinding.inflate(inflater, container, false)
        return inflateFragment(binding, inflater, container)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val options = SubjectSegmenterOptions.Builder()
            .enableForegroundBitmap()
            .build()
        segmenter = SubjectSegmentation.getClient(options)

        setupUI()
    }

    private fun setupUI() {
        // 1. Auto Pilot Switch
        binding.autoSwitch.setOnCheckedChangeListener { _, isChecked ->
            isAutoMode = isChecked
            if (!isChecked) {
                // If turned OFF, stop immediately
                vehicle.setControl(0f, 0f)
                binding.stopWarning.visibility = View.GONE
            }
        }

        // 2. Camera Switch
        binding.cameraToggleBtn.setOnClickListener {
            toggleCamera()
            frameToViewTransform = null
        }

        // 3. Threshold Slider
        binding.thresholdSeekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                stopThresholdPercent = progress
                binding.thresholdText.text = "Stop if Area > $progress%"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 4. Expand/Collapse
        binding.configHeader.setOnClickListener { toggleConfig(!isConfigExpanded) }
        binding.mainContainer.setOnClickListener { if (isConfigExpanded) toggleConfig(false) }
    }

    private fun toggleConfig(expand: Boolean) {
        isConfigExpanded = expand
        // Animation handled by XML android:animateLayoutChanges="true"

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
        val inputImage = InputImage.fromBitmap(image, sensorRotation)

        segmenter?.process(inputImage)
            ?.addOnSuccessListener { result ->
                isProcessing = false
                val mask = result.foregroundBitmap

                if (mask != null && _binding != null) {

                    // --- 1. CALCULATE PERCENTAGE (Background Thread) ---
                    val w = mask.width
                    val h = mask.height
                    val totalPixels = w * h
                    var objectPixels = 0

                    val pixels = IntArray(totalPixels)
                    mask.getPixels(pixels, 0, w, 0, 0, w, h)

                    for (i in pixels.indices step 10) {
                        if (Color.alpha(pixels[i]) > 0) objectPixels++
                    }

                    val estimatedObjectPixels = objectPixels * 10
                    val percentage = (estimatedObjectPixels.toFloat() / totalPixels.toFloat()) * 100f

                    activity?.runOnUiThread {
                        if (_binding == null) return@runOnUiThread

                        // --- 2. DRIVING LOGIC ---
                        if (isAutoMode) {
                            if (percentage > stopThresholdPercent) {
                                // BLOCKED: Stop
                                vehicle.setControl(0f, 0f)
                                binding.stopWarning.visibility = View.VISIBLE
                                binding.stopWarning.text = "BLOCKED"
                            } else {
                                // CLEAR: Drive Forward
                                // Using 0.5f as a safe default speed
                                vehicle.setControl(0.5f, 0.5f)
                                binding.stopWarning.visibility = View.GONE
                            }
                        } else {
                            // AUTO OFF: Do nothing (Allow manual control)
                            // We do NOT set 0,0 here, otherwise joystick won't work
                            binding.stopWarning.visibility = View.GONE
                        }

                        // --- 3. UI UPDATES (Always run) ---
                        displayMask(mask, sensorRotation)
                        drawMaskBorder(mask)

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

            // MAX -> Center Crop
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
        if (Constants.CMD_NETWORK == command) {

            // Must run on UI thread to update the View
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