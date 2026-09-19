package org.openbot.autopilot;

import android.graphics.Bitmap;

/**
 * A short, time-stamped ring of recently seen frames, so a sequence policy can be fed a stack whose
 * frames are spaced the way they were at training time.
 *
 * <p>Indexing by arrival order would not do that. Frames arrive at the camera rate but inference
 * runs only when the previous one finished, so "the last five frames I ran on" are spaced by
 * inference latency - typically several times the training spacing, which makes the apparent motion
 * in the stack correspondingly wrong. Every preview frame is recorded here instead, and the stack is
 * assembled by looking up the frame nearest each target timestamp.
 *
 * <p>Writes come from the camera analyzer thread and reads from the inference thread, hence the
 * synchronization; reads copy out, so the caller can never be handed a slot that is about to be
 * overwritten.
 */
class FrameHistory {

  private final long[] timestamps;
  private final int[][] pixels;
  private final int capacity;

  /** Next slot to write; the ring holds the newest {@code size} frames ending just before it. */
  private int next = 0;

  private int size = 0;

  FrameHistory(int capacity, int numPixels) {
    this.capacity = capacity;
    this.timestamps = new long[capacity];
    this.pixels = new int[capacity][numPixels];
  }

  int getNumPixels() {
    return pixels[0].length;
  }

  synchronized void clear() {
    next = 0;
    size = 0;
  }

  synchronized boolean isEmpty() {
    return size == 0;
  }

  /** Records one frame. The bitmap is read out immediately and is not retained. */
  synchronized void add(long timestampMs, Bitmap frame) {
    frame.getPixels(
        pixels[next], 0, frame.getWidth(), 0, 0, frame.getWidth(), frame.getHeight());
    timestamps[next] = timestampMs;
    next = (next + 1) % capacity;
    if (size < capacity) ++size;
  }

  /**
   * Fills {@code out} with the frames closest to the requested ages, oldest first.
   *
   * <p>Ages are measured back from the newest recorded frame, in milliseconds, and are expected
   * newest-first (0 for the current frame) - the same convention the training offsets use. A stack
   * reaching further back than the ring goes repeats the oldest frame available, which is what the
   * window builder does at the start of a session, so a cold start looks to the model like the
   * motionless start of a recording rather than something it has never seen.
   *
   * @param agesMs age of each requested frame, newest first
   * @param out destination, {@code out[0]} receiving the oldest frame
   * @return the span in milliseconds between the oldest and newest frame actually selected, or -1
   *     if there is nothing recorded yet
   */
  synchronized long fill(float[] agesMs, int[][] out) {
    if (size == 0) return -1;

    final long now = timestamps[(next - 1 + capacity) % capacity];
    long oldest = now;
    for (int i = 0; i < agesMs.length; ++i) {
      final int slot = nearest(now - (long) agesMs[i]);
      // agesMs is newest first, the stack is oldest first.
      System.arraycopy(pixels[slot], 0, out[agesMs.length - 1 - i], 0, pixels[slot].length);
      oldest = Math.min(oldest, timestamps[slot]);
    }
    return now - oldest;
  }

  /** Slot whose timestamp is closest to the target. */
  private int nearest(long targetMs) {
    int best = (next - 1 + capacity) % capacity;
    long bestDistance = Long.MAX_VALUE;
    for (int i = 0; i < size; ++i) {
      final int slot = (next - 1 - i + capacity) % capacity;
      final long distance = Math.abs(timestamps[slot] - targetMs);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = slot;
      } else {
        // Timestamps decrease as we walk back, so once we start moving away we are past it.
        break;
      }
    }
    return best;
  }
}
