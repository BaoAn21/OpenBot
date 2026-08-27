// Created by Matthias Mueller - Intel Intelligent Systems Lab - 2020

package org.openbot.env;

import android.util.Pair;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import org.openbot.utils.Enums.DriveMode;
import org.openbot.vehicle.Control;

public class GameController {
  private DriveMode driveMode;

  public GameController(DriveMode driveMode) {
    this.driveMode = driveMode;
  }

  public void setDriveMode(DriveMode mode) {
    driveMode = mode;
  }

  public DriveMode getDriveMode() {
    return driveMode;
  }

  private static float getCenteredAxis(MotionEvent event, int axis, int historyPos) {

    if (event == null || event.getDevice() == null) return 0;
    final InputDevice.MotionRange range = event.getDevice().getMotionRange(axis, event.getSource());

    // A joystick at rest does not always report an absolute position of
    // (0,0). Use the getFlat() method to determine the range of values
    // bounding the joystick axis center.
    if (range != null) {
      final float flat = range.getFlat();
      final float value =
          historyPos < 0
              ? event.getAxisValue(axis)
              : event.getHistoricalAxisValue(axis, historyPos);

      // Ignore axis values that are within the 'flat' region of the
      // joystick axis center.
      if (Math.abs(value) > flat) {
        return value;
      }
    }
    return 0;
  }

  public Control processButtonInput(KeyEvent event) {
    float steering = 0;
    float throttle = 0;
    switch (event.getKeyCode()) {
      case KeyEvent.KEYCODE_BUTTON_A:
        //        Toast.makeText(OpenBotApplication.getContext(), "A recognized",
        // Toast.LENGTH_SHORT).show();
        break;
      case KeyEvent.KEYCODE_BUTTON_B:
        //        Toast.makeText(OpenBotApplication.getContext(), "B recognized",
        // Toast.LENGTH_SHORT).show();
        break;
      case KeyEvent.KEYCODE_BUTTON_Y:
        //        Toast.makeText(OpenBotApplication.getContext(), "Y recognized",
        // Toast.LENGTH_SHORT).show();
        break;
      case KeyEvent.KEYCODE_BUTTON_X:
        //        Toast.makeText(OpenBotApplication.getContext(), "X recognized",
        // Toast.LENGTH_SHORT).show();
        break;
      case KeyEvent.KEYCODE_BUTTON_L1:
        //        Toast.makeText(OpenBotApplication.getContext(), "L1 recognized",
        // Toast.LENGTH_SHORT).show();
        break;
      case KeyEvent.KEYCODE_BUTTON_R1:
        //        Toast.makeText(OpenBotApplication.getContext(), "R1 recognized",
        // Toast.LENGTH_SHORT).show();
        break;
      case KeyEvent.KEYCODE_DPAD_UP:
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
          throttle = Control.MAX;
        }
        break;
      case KeyEvent.KEYCODE_DPAD_RIGHT:
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
          steering = Control.MAX;
        }
        break;
      case KeyEvent.KEYCODE_DPAD_DOWN:
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
          throttle = -Control.MAX;
        }
        break;
      case KeyEvent.KEYCODE_DPAD_LEFT:
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
          steering = -Control.MAX;
        }
        break;
      default:
        //        Toast.makeText(
        //                OpenBotApplication.getContext(),
        //                "Key " + event.getKeyCode() + " not recognized",
        //                Toast.LENGTH_SHORT)
        //            .show();
        break;
    }
    return new Control(steering, throttle);
  }

  public Control processJoystickInput(MotionEvent event, int historyPos) {

    switch (driveMode) {
      case DUAL:
        // Left stick vertical drives, right stick horizontal steers.
        float leftStickY = getCenteredAxis(event, MotionEvent.AXIS_Y, historyPos);
        float rightStickX = getCenteredAxis(event, MotionEvent.AXIS_Z, historyPos);

        return convertDualToControl(leftStickY, rightStickX);

      case GAME:
        float rightTrigger = getCenteredAxis(event, MotionEvent.AXIS_GAS, historyPos);
        if (rightTrigger == 0) {
          rightTrigger = getCenteredAxis(event, MotionEvent.AXIS_RTRIGGER, historyPos);
        }

        float leftTrigger = getCenteredAxis(event, MotionEvent.AXIS_BRAKE, historyPos);
        if (leftTrigger == 0) {
          leftTrigger = getCenteredAxis(event, MotionEvent.AXIS_LTRIGGER, historyPos);
        }

        // Steering comes from the left control stick only, so that no other axis can
        // inject a steering command while the stick is centered.
        float steeringOffset = getCenteredAxis(event, MotionEvent.AXIS_X, historyPos);

        return convertGameToControl(leftTrigger, rightTrigger, steeringOffset);

      case JOYSTICK:
        // Calculate the vertical distance to move by
        // using the input value from one of these physical controls:
        // the left control stick, hat switch, or the right control stick.
        float yAxis = getCenteredAxis(event, MotionEvent.AXIS_Y, historyPos);
        if (yAxis == 0) {
          yAxis = getCenteredAxis(event, MotionEvent.AXIS_HAT_Y, historyPos);
        }
        if (yAxis == 0) {
          yAxis = getCenteredAxis(event, MotionEvent.AXIS_RZ, historyPos);
        }

        // Calculate the horizontal distance to move by
        // using the input value from one of these physical controls:
        // the left control stick, hat axis, or the right control stick.
        float xAxis = getCenteredAxis(event, MotionEvent.AXIS_X, historyPos);
        if (xAxis == 0) {
          xAxis = getCenteredAxis(event, MotionEvent.AXIS_HAT_X, historyPos);
        }
        if (xAxis == 0) {
          xAxis = getCenteredAxis(event, MotionEvent.AXIS_Z, historyPos);
        }

        return convertJoystickToControl(xAxis, yAxis);

      default:
        return new Control(0, 0);
    }
  }

  public Control convertDualToControl(float leftStickY, float rightStickX) {
    // Android reports the stick's up direction as negative, so the throttle axis is inverted.
    return new Control(rightStickX * Control.MAX, -leftStickY * Control.MAX);
  }

  public Control convertGameToControl(float leftTrigger, float rightTrigger, float steeringOffset) {
    return new Control(
        steeringOffset * Control.MAX, (rightTrigger - leftTrigger) * Control.MAX);
  }

  public Control convertJoystickToControl(float xAxis, float yAxis) {
    return new Control(xAxis * Control.MAX, -yAxis * Control.MAX);
  }

  public static Pair<Float, Float> processJoystickInputLeft(MotionEvent event, int historyPos) {

    // Calculate the horizontal distance to move by
    // using the input value from one of these physical controls:
    // the left control stick, hat axis, or the right control stick.
    float x = getCenteredAxis(event, MotionEvent.AXIS_X, historyPos);

    // Calculate the vertical distance to move by
    // using the input value from one of these physical controls:
    // the left control stick, hat switch, or the right control stick.
    float y = getCenteredAxis(event, MotionEvent.AXIS_Y, historyPos);

    return new Pair<>(x, y);
  }

  public static Pair<Float, Float> processJoystickInputRight(MotionEvent event, int historyPos) {

    // Calculate the horizontal distance to move by
    // using the input value from one of these physical controls:
    // the left control stick, hat axis, or the right control stick.
    float x = getCenteredAxis(event, MotionEvent.AXIS_Z, historyPos);

    // Calculate the vertical distance to move by
    // using the input value from one of these physical controls:
    // the left control stick, hat switch, or the right control stick.
    float y = getCenteredAxis(event, MotionEvent.AXIS_RZ, historyPos);

    return new Pair<>(x, y);
  }
}
