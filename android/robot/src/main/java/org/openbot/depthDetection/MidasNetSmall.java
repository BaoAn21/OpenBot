// File: org/openbot/depthDetection/MidasNetSmall.java

package org.openbot.depthDetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
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

public class MidasNetSmall {
    private static final String MODEL_NAME = "lite-model_midas_v2_1_small_1_lite_1.tflite";
    private static final int INPUT_IMAGE_DIM = 256;
    private static final int NUM_THREADS = 4;
    private static final float[] NORM_MEAN = {123.675f, 116.28f, 103.53f};
    private static final float[] NORM_STD = {58.395f, 57.12f, 57.375f};

    private Interpreter interpreter;
    private final ImageProcessor inputTensorProcessor;
    private final TensorProcessor outputTensorProcessor;
    private MapType mapType;

    public MidasNetSmall(Context context, MapType mapType) throws IOException {
        this.mapType = mapType;

        inputTensorProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(NORM_MEAN, NORM_STD))
                .build();

        outputTensorProcessor = new TensorProcessor.Builder()
                .add(new DepthScalingOp())
                .build();

        Interpreter.Options options = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();
        if (compatList.isDelegateSupportedOnThisDevice()) {
            options.addDelegate(new GpuDelegate(compatList.getBestOptionsForThisDevice()));
        } else {
            options.setNumThreads(NUM_THREADS);
        }

        interpreter = new Interpreter(FileUtil.loadMappedFile(context, MODEL_NAME), options);
    }

    public Bitmap getDepthMap(Bitmap inputImage) {
        TensorImage inputTensor = TensorImage.fromBitmap(inputImage);
        inputTensor = inputTensorProcessor.process(inputTensor);

        TensorBuffer outputTensor = TensorBufferFloat.createFixedSize(
                new int[]{INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, 1},
                DataType.FLOAT32
        );

        interpreter.run(inputTensor.getBuffer(), outputTensor.getBuffer());
        outputTensor = outputTensorProcessor.process(outputTensor);

        if (mapType == MapType.DEPTHVIEW_GRAYSCALE) {
            return ImageUtils.toGrayscaleBitmap(outputTensor.getFloatArray(), INPUT_IMAGE_DIM);
        } else {
            return ImageUtils.toHeatMapBitmap(outputTensor.getFloatArray(), INPUT_IMAGE_DIM);
        }
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }

    private static class DepthScalingOp implements TensorOperator {
        @Override
        public TensorBuffer apply(TensorBuffer input) {
            float[] values = input.getFloatArray();
            if (values.length == 0) return input;

            float max = values[0];
            float min = values[0];
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