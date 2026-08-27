package org.openbot.env;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.openbot.utils.Enums;
import org.openbot.vehicle.Control;

public class GameControllerTest {
  @Test
  public void convertDualToControl_test() {
    GameController gameController = new GameController(Enums.DriveMode.DUAL);
    assertEquals(Enums.DriveMode.DUAL, gameController.getDriveMode());
    // Left stick pushed down (positive Y) reverses, right stick pushed left steers left.
    Control control = gameController.convertDualToControl(0.5f, -0.5f);
    assertEquals(-127.5f, control.getSteering(), 0.0f);
    assertEquals(-127.5f, control.getThrottle(), 0.0f);
  }

  @Test
  public void convertGameToControl_test() {
    GameController gameController = new GameController(Enums.DriveMode.GAME);
    assertEquals(Enums.DriveMode.GAME, gameController.getDriveMode());
    Control control;
    control = gameController.convertGameToControl(0.0f, 0.5f, 1.0f);
    assertEquals(255.0f, control.getSteering(), 0.0f);
    assertEquals(127.5f, control.getThrottle(), 0.0f);

    control = gameController.convertGameToControl(0.0f, 0.5f, -1.0f);
    assertEquals(-255.0f, control.getSteering(), 0.0f);
    assertEquals(127.5f, control.getThrottle(), 0.0f);

    // Brake only: reverse, straight ahead.
    control = gameController.convertGameToControl(0.5f, 0.0f, 0.0f);
    assertEquals(0.0f, control.getSteering(), 0.0f);
    assertEquals(-127.5f, control.getThrottle(), 0.0f);
  }

  @Test
  public void convertJoystickToControl_test() {
    GameController gameController = new GameController(Enums.DriveMode.JOYSTICK);
    assertEquals(Enums.DriveMode.JOYSTICK, gameController.getDriveMode());
    Control control;
    control = gameController.convertJoystickToControl(0.5f, -0.5f);
    assertEquals(127.5f, control.getSteering(), 0.0f);
    assertEquals(127.5f, control.getThrottle(), 0.0f);

    control = gameController.convertJoystickToControl(-0.5f, -0.5f);
    assertEquals(-127.5f, control.getSteering(), 0.0f);
    assertEquals(127.5f, control.getThrottle(), 0.0f);
  }

  @Test
  public void controlIsClampedToDeviceRange() {
    Control control = new Control(400.f, -1000.f);
    assertEquals(255.0f, control.getSteering(), 0.0f);
    assertEquals(-255.0f, control.getThrottle(), 0.0f);
  }

  @Test
  public void fromLeftRight_test() {
    // Both wheels forward: straight ahead at full throttle.
    Control control = Control.fromLeftRight(1.0f, 1.0f);
    assertEquals(0.0f, control.getSteering(), 0.0f);
    assertEquals(255.0f, control.getThrottle(), 0.0f);

    // Left wheel forward, right wheel back: full right steering, no throttle.
    control = Control.fromLeftRight(1.0f, -1.0f);
    assertEquals(255.0f, control.getSteering(), 0.0f);
    assertEquals(0.0f, control.getThrottle(), 0.0f);
  }
}
