package org.openbot.tflite;

import android.app.Activity;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.openbot.vehicle.Control;
import timber.log.Timber;

/**
 * Autopilot over a stack of frames, as produced by the {@code pilot_net_seq} policy.
 *
 * <p>The network is the single-frame autopilot with a deeper first convolution: instead of one RGB
 * frame it takes {@code seqLen} frames concatenated along the channel axis, which is what lets it
 * see apparent motion and therefore tell forward driving from reverse. Everything else - the
 * normalization, the crop, the command input, the two output values - is identical to {@link
 * Autopilot}.
 *
 * <p>The channel layout is the one {@code tf.concat(frames, axis=-1)} produces on channels-last
 * tensors: for every pixel, the three values of the oldest frame, then the next, ending with the
 * current frame. Getting that order wrong does not fail loudly - the model still emits plausible
 * numbers - so the packing below is the one place worth being careful.
 */
public class AutopilotSeq extends Network {

  /** A ByteBuffer to hold data, to be feed into Tensorflow Lite as inputs. */
  protected ByteBuffer cmdBuffer;

  private int cmdIndex;
  private int imgIndex;

  /** Number of frames the stack holds, read off the model rather than configured. */
  private final int seqLen;

  /** Additional normalization of the used input. */
  private static final float IMAGE_MEAN = 0.0f;

  private static final float IMAGE_STD = 255.0f;

  public AutopilotSeq(Activity activity, Model model, Device device, int numThreads)
      throws IOException {
    super(activity, model, device, numThreads);
    try {
      cmdIndex = tflite.getInputIndex("serving_default_cmd_input:0");
      imgIndex = tflite.getInputIndex("serving_default_img_input:0");
    } catch (IllegalArgumentException e) {
      cmdIndex = tflite.getInputIndex("cmd_input");
      imgIndex = tflite.getInputIndex("img_input");
    }

    final int[] shape = tflite.getInputTensor(imgIndex).shape();
    if (shape.length != 4 || shape[1] != getImageSizeY() || shape[2] != getImageSizeX()) {
      throw new IllegalArgumentException(
          "Invalid tensor dimensions: model wants "
              + Arrays.toString(shape)
              + ", model config says "
              + getImageSizeX()
              + "x"
              + getImageSizeY());
    }
    if (shape[3] % 3 != 0) {
      throw new IllegalArgumentException("Input depth " + shape[3] + " is not a multiple of 3");
    }
    seqLen = shape[3] / 3;
    if (seqLen < 2) {
      throw new IllegalArgumentException("Single-frame model - load it as Autopilot instead");
    }

    // Network sized imgData for one 3-channel frame, which is all the single-frame models need.
    // A stack needs seqLen times that, so replace the buffer here. It cannot be done in Network
    // itself because seqLen is only known after the interpreter exists, i.e. after super().
    imgData =
        ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE
                * getImageSizeX()
                * getImageSizeY()
                * shape[3]
                * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());

    cmdBuffer = ByteBuffer.allocateDirect(4);
    cmdBuffer.order(ByteOrder.nativeOrder());

    Timber.d("Created a Tensorflow Lite AutopilotSeq over %d frames.", seqLen);
  }

  /** How many frames this model expects per inference. */
  public int getSeqLen() {
    return seqLen;
  }

  private void convertIndicatorToByteBuffer(int indicator) {
    if (cmdBuffer == null) {
      return;
    }
    cmdBuffer.rewind();
    cmdBuffer.putFloat(indicator);
  }

  /**
   * Packs the stack, interleaving the frames per pixel to match the channels-last concatenation
   * used at training time.
   *
   * @param frames one ARGB_8888 pixel array per frame, ordered oldest first, each of them
   *     getImageSizeX() * getImageSizeY() pixels in row-major order
   */
  private void convertFramesToByteBuffer(int[][] frames) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    final int numPixels = getImageSizeX() * getImageSizeY();
    final long startTime = SystemClock.elapsedRealtime();
    for (int p = 0; p < numPixels; ++p) {
      for (int f = 0; f < seqLen; ++f) {
        final int pixelValue = frames[f][p];
        imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
      }
    }
    LOGGER.v("Timecost to put values into ByteBuffer: " + (SystemClock.elapsedRealtime() - startTime));
  }

  /**
   * Runs the policy on a stack of frames.
   *
   * @param frames pixel arrays ordered oldest first, the last one being the current frame
   * @param indicator the driving command
   */
  public Control recognizeImage(final int[][] frames, final int indicator) {
    if (frames.length != seqLen) {
      throw new IllegalArgumentException(
          "Model wants " + seqLen + " frames, got " + frames.length);
    }

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");
    Trace.beginSection("preprocessFrames");
    convertFramesToByteBuffer(frames);
    convertIndicatorToByteBuffer(indicator);
    Trace.endSection(); // preprocessFrames

    // Run the inference call.
    Trace.beginSection("runInference");
    long startTime = SystemClock.elapsedRealtime();
    Object[] inputArray;
    if (cmdIndex == 0) {
      inputArray = new Object[] {cmdBuffer, imgData};
    } else {
      inputArray = new Object[] {imgData, cmdBuffer};
    }

    float[][] predicted_ctrl = new float[1][2];
    outputMap.put(0, predicted_ctrl);
    tflite.runForMultipleInputsOutputs(inputArray, outputMap);
    long endTime = SystemClock.elapsedRealtime();
    Trace.endSection();
    Timber.v("Timecost to run model inference: %s", (endTime - startTime));

    Trace.endSection(); // "recognizeImage"
    // Model output is (steering, throttle) directly, not a differential-drive (left, right)
    // pair, so it maps straight onto Control's raw-unit constructor instead of fromLeftRight.
    return new Control(predicted_ctrl[0][0] * Control.MAX, predicted_ctrl[0][1] * Control.MAX);
  }

  @Override
  public boolean getMaintainAspect() {
    return true;
  }

  @Override
  public RectF getCropRect() {
    return new RectF(0.0f, 240.0f / 720.0f, 0.0f, 0.0f);
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    // Deliberately unsupported: Network's single-bitmap path would write one frame contiguously,
    // which is not the interleaved layout this model reads. Use recognizeImage(int[][], int).
    throw new UnsupportedOperationException("AutopilotSeq packs whole stacks, not single frames");
  }
}
