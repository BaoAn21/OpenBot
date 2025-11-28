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
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenter
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions
import org.openbot.common.CameraFragment
import org.openbot.databinding.FragmentLineObjectTrackingBinding

class LineObjectTrackingFragment : CameraFragment() {
    private val TAG = "LineObjectTrackingFragment"
    private var _binding: FragmentLineObjectTrackingBinding? = null
    private val binding get() = _binding!!

    private var segmenter: SubjectSegmenter? = null
    private var isProcessing = false

    // Matrix for aligning the mask
    private var frameToViewTransform: Matrix? = null
    private var lastMaskWidth = 0
    private var lastMaskHeight = 0
    private var lastRotation = -1

    // Paint for the Boundary Box
    private val borderPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 10f // Thick line so it's easy to see
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

        binding.cameraToggle.setOnClickListener {
            toggleCamera()
            frameToViewTransform = null
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
                    activity?.runOnUiThread {
                        // 1. Draw Mask (Magenta Silhouette)
                        displayMask(mask, sensorRotation)

                        // 2. Draw the Boundary of the Mask (Yellow Box)
                        drawMaskBorder(mask)

                        binding.orientationText.text = "Rot: $sensorRotation"
                        binding.dimensionText.text = "Cam: ${image.width}x${image.height}"
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

            // Center -> Rotate -> Scale -> Center -> Mirror
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

        // Create a canvas to draw the box on
        val overlayBitmap = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(overlayBitmap)

        // 1. Create a rectangle representing the FULL SIZE of the mask
        // e.g. Rect(0, 0, 640, 480)
        val maskRect = RectF(0f, 0f, mask.width.toFloat(), mask.height.toFloat())

        // 2. Transform this rectangle using the same matrix as the image
        // This will rotate, scale, and move the rect to match the visual mask exactly.
        frameToViewTransform?.mapRect(maskRect)

        // 3. Draw it
        canvas.drawRect(maskRect, borderPaint)

        binding.rectOverlay.setImageBitmap(overlayBitmap)
    }

    override fun processControllerKeyData(command: String?) { }
    override fun processUSBData(data: String?) { }

    override fun onDestroyView() {
        super.onDestroyView()
        segmenter?.close()
        segmenter = null
        _binding = null
    }
}