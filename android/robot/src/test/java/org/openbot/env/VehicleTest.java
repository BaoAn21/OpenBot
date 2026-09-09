package org.openbot.env;

import static org.junit.Assert.assertEquals;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.openbot.utils.Enums;
import org.openbot.vehicle.Control;
import org.openbot.vehicle.Vehicle;

@RunWith(AndroidJUnit4.class)
public class VehicleTest {

  private Vehicle vehicle;

  @Before
  public void setupVehicle() {
    vehicle = new Vehicle(ApplicationProvider.getApplicationContext(), 115200);
  }

  @Test
  public void getRotation() {
    assertEquals(0, vehicle.getRotation(), 0.0);

    vehicle.setControl(new Control(Control.MAX / 2, 0));
    assertEquals(90, vehicle.getRotation(), 0.0);

    vehicle.setControl(new Control(-Control.MAX, 0));
    assertEquals(-180, vehicle.getRotation(), 0.0);
  }

  @Test
  public void getThrottleIsScaledBySpeedMode() {
    vehicle.setSpeedMultiplier(Enums.SpeedMode.SLOW.getValue());
    vehicle.setControl(new Control(0, -Control.MAX));
    assertEquals(-128, vehicle.getThrottle(), 0.0);

    vehicle.setSpeedMultiplier(Enums.SpeedMode.NORMAL.getValue());
    vehicle.setControl(new Control(0, -Control.MAX));
    assertEquals(-192, vehicle.getThrottle(), 0.0);

    vehicle.setSpeedMultiplier(Enums.SpeedMode.FAST.getValue());
    vehicle.setControl(new Control(0, Control.MAX));
    assertEquals(255, vehicle.getThrottle(), 0.0);
  }

  @Test
  public void getSteeringIgnoresSpeedMode() {
    vehicle.setSpeedMultiplier(Enums.SpeedMode.SLOW.getValue());
    vehicle.setControl(new Control(Control.MAX, 0));
    assertEquals(255, vehicle.getSteering(), 0.0);
  }

}
