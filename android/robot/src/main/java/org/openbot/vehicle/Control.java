package org.openbot.vehicle;

/**
 * Ackermann-style control command: a steering value and a throttle value, both in raw device units
 * of [-255, 255]. Positive steering turns right, positive throttle drives forward.
 */
public class Control {
  public static final float MAX = 255.f;

  private final float steering;
  private final float throttle;

  public Control(float steering, float throttle) {
    this.steering = Math.max(-MAX, Math.min(MAX, steering));
    this.throttle = Math.max(-MAX, Math.min(MAX, throttle));
  }

  /**
   * Builds a control from a normalized differential-drive pair in [-1, 1], as produced by the
   * TFLite models, the object tracker and the phone/web controller.
   */
  public static Control fromLeftRight(float left, float right) {
    return new Control((left - right) / 2 * MAX, (left + right) / 2 * MAX);
  }

  public float getSteering() {
    return steering;
  }

  public float getThrottle() {
    return throttle;
  }

  public Control mirror() {
    return new Control(-this.steering, this.throttle);
  }
}
