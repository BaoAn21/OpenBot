package org.openbot.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
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
 * Autopilot for the {@code donkey_memory} policy, donkeycar's "memory" pilot: one camera frame plus
 * the last {@code memLen} (steering, throttle) pairs the model itself produced, and - for a model
 * trained with {@code --mem_cmd} - the indicator as a cmd input.
 *
 * <p>The memory is laid out the way training built it: {@code [s(t-N), t(t-N), ..., s(t-1),
 * t(t-1)]}, oldest first, in the model's own normalized units (raw device units / 255). Keeping
 * that history is the caller's job, see {@link org.openbot.autopilot.AutopilotMemFragment}.
 *
 * <p>Inputs are found by name rather than position, since a model has two or three of them
 * depending on how it was trained.
 */
public class AutopilotMem extends Network {

  protected ByteBuffer memBuffer;
  protected ByteBuffer cmdBuffer;

  private int imgIndex = -1;
  private int memIndex = -1;
  /** -1 when the model was trained without a cmd input. */
  private int cmdIndex = -1;

  /** Number of (steering, throttle) pairs in the memory, read off the model. */
  private final int memLen;

  /** Additional normalization of the used input. */
  private static final float IMAGE_MEAN = 0.0f;

  private static final float IMAGE_STD = 255.0f;

  public AutopilotMem(Activity activity, Model model, Device device, int numThreads)
      throws IOException {
    super(activity, model, device, numThreads);
    // Names are "serving_default_<name>:0" or plain "<name>" depending on the converter.
    for (int i = 0; i < tflite.getInputTensorCount(); ++i) {
      final String name = tflite.getInputTensor(i).name();
      if (name.contains("img_input")) imgIndex = i;
      else if (name.contains("mem_input")) memIndex = i;
      else if (name.contains("cmd_input")) cmdIndex = i;
    }
    if (imgIndex < 0 || memIndex < 0) {
      throw new IllegalArgumentException(
          "Not a memory model: needs img_input and mem_input inputs");
    }
    if (!Arrays.equals(
        tflite.getInputTensor(imgIndex).shape(),
        new int[] {1, getImageSizeY(), getImageSizeX(), 3}))
      throw new IllegalArgumentException("Invalid tensor dimensions");

    final int[] memShape = tflite.getInputTensor(memIndex).shape();
    if (memShape.length != 2 || memShape[1] % 2 != 0) {
      throw new IllegalArgumentException(
          "Invalid memory dimensions " + Arrays.toString(memShape) + ", expected [1, 2 * N]");
    }
    memLen = memShape[1] / 2;

    memBuffer = ByteBuffer.allocateDirect(4 * 2 * memLen);
    memBuffer.order(ByteOrder.nativeOrder());
    cmdBuffer = ByteBuffer.allocateDirect(4);
    cmdBuffer.order(ByteOrder.nativeOrder());

    Timber.d(
        "Created a Tensorflow Lite AutopilotMem with a memory of %d controls, %s cmd input.",
        memLen, hasCmdInput() ? "with" : "without");
  }

  /** How many previous (steering, throttle) pairs this model expects. */
  public int getMemLen() {
    return memLen;
  }

  /** Whether the model was trained with the indicator as a cmd input ({@code --mem_cmd}). */
  public boolean hasCmdInput() {
    return cmdIndex >= 0;
  }

  /**
   * Runs the policy on one frame.
   *
   * @param bitmap the cropped camera frame
   * @param memory the model's previous outputs, {@code 2 * getMemLen()} values oldest first
   * @param indicator the driving command; ignored by a model without a cmd input
   */
  public Control recognizeImage(final Bitmap bitmap, final float[] memory, final int indicator) {
    if (memory.length != 2 * memLen) {
      throw new IllegalArgumentException(
          "Model wants " + 2 * memLen + " memory values, got " + memory.length);
    }

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");
    Trace.beginSection("preprocessBitmap");
    convertBitmapToByteBuffer(bitmap);
    memBuffer.rewind();
    for (float value : memory) memBuffer.putFloat(value);
    cmdBuffer.rewind();
    cmdBuffer.putFloat(indicator);
    Trace.endSection(); // preprocessBitmap

    // Run the inference call.
    Trace.beginSection("runInference");
    long startTime = SystemClock.elapsedRealtime();
    final Object[] inputArray = new Object[hasCmdInput() ? 3 : 2];
    inputArray[imgIndex] = imgData;
    inputArray[memIndex] = memBuffer;
    if (hasCmdInput()) inputArray[cmdIndex] = cmdBuffer;

    float[][] predicted_ctrl = new float[1][2];
    outputMap.put(0, predicted_ctrl);
    tflite.runForMultipleInputsOutputs(inputArray, outputMap);
    long endTime = SystemClock.elapsedRealtime();
    Trace.endSection();
    Timber.v("Timecost to run model inference: %s", (endTime - startTime));

    Trace.endSection(); // "recognizeImage"
    // Model output is (steering, throttle) directly: tanh steering in [-1, 1] and sigmoid
    // throttle in [0, 1], mapped onto Control's raw units like the other autopilots.
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
    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
  }
}
