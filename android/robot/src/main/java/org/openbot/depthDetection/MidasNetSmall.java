package org.openbot.depthDetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log; // Make sure Log is imported
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;
import java.io.IOException;
import java.nio.ByteBuffer;

enum ModelType {
    QUANTIZED,
    FLOAT
}

public class MidasNetSmall {
    private static final String MODEL_FLOAT_NAME = "lite-model_midas_v2_1_small_1_lite_1.tflite";
    private static final String MODEL_QUANTIZED_NAME = "Midas-V2_w8a8.tflite";
    private static final int INPUT_IMAGE_DIM = 256;
    private static final int NUM_THREADS = 4;
    private static final float[] NORM_MEAN = {123.675f, 116.28f, 103.53f};
    private static final float[] NORM_STD = {58.395f, 57.12f, 57.375f};

    private final ModelType modelType;
    private Interpreter interpreter;
    private final ImageProcessor inputTensorProcessor;
    private final TensorProcessor outputTensorProcessor;
    private final MapType mapType;
    private long lastInferenceTimeMs = 0;

    // --- ADD A MEMBER VARIABLE FOR THE DELEGATE ---
    private GpuDelegate gpuDelegate = null;
    // ---------------------------------------------

    public MidasNetSmall(Context context, MapType mapType, ModelType modelType) throws IOException {
        this.mapType = mapType;
        this.modelType = modelType;
        String modelName;

        ImageProcessor.Builder inputBuilder = new ImageProcessor.Builder()
                .add(new ResizeOp(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, ResizeOp.ResizeMethod.BILINEAR));

        if (modelType == ModelType.FLOAT) {
            modelName = MODEL_FLOAT_NAME;
            inputBuilder.add(new NormalizeOp(NORM_MEAN, NORM_STD));
            outputTensorProcessor = new TensorProcessor.Builder().add(new DepthScalingOp()).build();
        } else {
            modelName = MODEL_QUANTIZED_NAME;
            outputTensorProcessor = null;
        }
        inputTensorProcessor = inputBuilder.build();

        Interpreter.Options options = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();

        // --- MODIFY DELEGATE CREATION TO STORE IT ---
        if (compatList.isDelegateSupportedOnThisDevice()) {
            // Create and store the delegate
            gpuDelegate = new GpuDelegate(compatList.getBestOptionsForThisDevice());
            options.addDelegate(gpuDelegate);
            Log.d("MidasNetSmall", "GPU delegate added.");
        } else {
            options.setNumThreads(NUM_THREADS);
            Log.d("MidasNetSmall", "GPU not supported, using CPU threads: " + NUM_THREADS);
        }
        // --------------------------------------------

        interpreter = new Interpreter(FileUtil.loadMappedFile(context, modelName), options);
    }

    public float[] getDepthMapFloatArray(Bitmap inputImage) {
        TensorImage inputTensor = TensorImage.fromBitmap(inputImage);
        inputTensor = inputTensorProcessor.process(inputTensor);

        long startTime = SystemClock.uptimeMillis();

        if (modelType == ModelType.FLOAT) {
            TensorBuffer outputTensor = TensorBufferFloat.createFixedSize(
                    new int[]{1, INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, 1}, DataType.FLOAT32
            );
            interpreter.run(inputTensor.getBuffer(), outputTensor.getBuffer());
            lastInferenceTimeMs = SystemClock.uptimeMillis() - startTime;
            outputTensor = outputTensorProcessor.process(outputTensor);
            return outputTensor.getFloatArray();
        } else {
            TensorBuffer outputTensor = TensorBuffer.createFixedSize(
                    new int[]{1, INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, 1}, DataType.UINT8
            );
            interpreter.run(inputTensor.getBuffer(), outputTensor.getBuffer());
            lastInferenceTimeMs = SystemClock.uptimeMillis() - startTime;
            ByteBuffer outputBuffer = outputTensor.getBuffer();
            outputBuffer.rewind();
            float[] floatArray = new float[outputBuffer.remaining()];
            for (int i = 0; i < floatArray.length; i++) {
                floatArray[i] = (float) (outputBuffer.get() & 0xFF);
            }
            return floatArray;
        }
    }

    public Bitmap getDepthMap(Bitmap inputImage) {
        float[] floatArray = getDepthMapFloatArray(inputImage);
        if (mapType == MapType.DEPTHVIEW_GRAYSCALE) {
            return ImageUtils.toGrayscaleBitmap(floatArray, INPUT_IMAGE_DIM);
        } else {
            return ImageUtils.toHeatMapBitmap(floatArray, INPUT_IMAGE_DIM);
        }
    }

    public long getLastInferenceTimeMs() {
        return lastInferenceTimeMs;
    }

    public void close() {
        // --- ADD DELEGATE CLOSURE ---
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
            Log.d("MidasNetSmall", "GPU delegate closed.");
        }
        // --------------------------
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
            Log.d("MidasNetSmall", "Interpreter closed.");
        }
    }

    private static class DepthScalingOp implements TensorOperator {
        // ... (remains the same) ...
        @Override
        public TensorBuffer apply(TensorBuffer input) {
            float[] values = input.getFloatArray();
            if (values.length == 0) return input;
            float max = values[0], min = values[0];
            for (int i = 1; i < values.length; i++) {
                if (values[i] > max) max = values[i];
                if (values[i] < min) min = values[i];
            }
            float range = max - min;
            if (range > 1e-6f) {
                for (int i = 0; i < values.length; i++) {
                    values[i] = ((values[i] - min) / range) * 255.0f;
                }
            } else {
                java.util.Arrays.fill(values, 0.0f);
            }
            TensorBuffer output = TensorBufferFloat.createFrom(input, DataType.FLOAT32);
            output.loadArray(values, input.getShape());
            return output;
        }
    }
}