package org.openbot;

import android.app.Application;
import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.preference.PreferenceManager;
import org.jetbrains.annotations.NotNull;
//import org.openbot.entity.MyObjectBox;
//import org.openbot.objectbox.ObjectBox;
import org.openbot.mqtt.MqttService;
import org.openbot.objectbox.ObjectBox;
import org.openbot.objectbox.entity.LogEntry;
import org.openbot.vehicle.Vehicle;

import io.objectbox.Box;
import io.objectbox.BoxStore;
import timber.log.Timber;

public class OpenBotApplication extends Application {
  private static final String TAG = "OpenBotApp";

  static Context context;
  public static Vehicle vehicle;

  public static MqttService mqttService;


  public static Context getContext() {
    return context;
  }

  @Override
  public void onCreate() {
    super.onCreate();
    context = getApplicationContext();
    ObjectBox.init(this);
    BoxStore store = ObjectBox.get();
//     --- ADD THIS TEST CODE ---
//     1. Get the "Box" for our LogEntry entity
    Box<LogEntry> logBox = store.boxFor(LogEntry.class);

    // 2. Create a new log entry object
    LogEntry testLog = new LogEntry("ObjectBox test successful in HCMC!", System.currentTimeMillis());

    // 3. Save the object to the database
    logBox.put(testLog);
    Log.d(TAG, "Saved Log Entry with ID: " + testLog.id);

    LogEntry retrievedLog = logBox.get(testLog.id);
    Log.d(TAG, "Retrieved Log Entry: " + retrievedLog.toString());

    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
    int baudRate = Integer.parseInt(sharedPreferences.getString("baud_rate", "115200"));
    vehicle = new Vehicle(this, baudRate);
    vehicle.initBle();
    vehicle.connectUsb();
    vehicle.initBle();
    if (BuildConfig.DEBUG) {
      Timber.plant(
          new Timber.DebugTree() {
            @NonNull
            @Override
            protected String createStackElementTag(@NotNull StackTraceElement element) {
              return super.createStackElementTag(element) + ":" + element.getLineNumber();
            }
          });
    }
  }

  @Override
  public void onTerminate() {
    super.onTerminate();
  }
}
