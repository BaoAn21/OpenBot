package org.openbot.main;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;
import org.openbot.model.SubCategory;
import org.openbot.vehicle.Vehicle;

import java.util.ArrayList;
import java.util.List;

public class MainViewModel extends ViewModel {

  private final MutableLiveData<SubCategory> selectedMode = new MutableLiveData<>();

  public void selectMode(SubCategory subCategory) {
    selectedMode.setValue(subCategory);
  }

  public LiveData<SubCategory> getSelectedMode() {
    return selectedMode;
  }

  private final MutableLiveData<String> deviceData = new MutableLiveData<>();
  private final MutableLiveData<List<String>> logMessages = new MutableLiveData<>(new ArrayList<>());

  public void processUsbData(String message) {
    deviceData.setValue(message);

    List<String> currentMessages = logMessages.getValue();
    if (currentMessages != null) {
      String timestamp = new java.text.SimpleDateFormat("[HH:mm:ss.SSS] ", java.util.Locale.getDefault()).format(new java.util.Date());
      currentMessages.add(timestamp + message);
      if (currentMessages.size() > 100) {
        currentMessages.remove(0);
      }
      logMessages.setValue(currentMessages);
    }
  }

  public LiveData<String> getDeviceData() {
    return deviceData;
  }

  public LiveData<List<String>> getLogMessages() {
    return logMessages;
  }

  public void clearLogs() {
    List<String> currentMessages = logMessages.getValue();
    if (currentMessages != null) {
      currentMessages.clear();
      logMessages.setValue(currentMessages);
    }
  }

  private final MutableLiveData<Vehicle> vehicle = new MutableLiveData<>();

  public void setVehicle(Vehicle data) {
    vehicle.setValue(data);
  }

  public LiveData<Vehicle> getVehicle() {
    return vehicle;
  }

  private final MutableLiveData<Boolean> usbStatus = new MutableLiveData<>();

  public void setUsbStatus(Boolean data) {
    usbStatus.setValue(data);
  }

  public LiveData<Boolean> getUsbStatus() {
    return usbStatus;
  }
}
