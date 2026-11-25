package org.openbot;

import android.app.Application;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Environment;
import androidx.annotation.NonNull;
import androidx.preference.PreferenceManager;
import java.io.File;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import org.jetbrains.annotations.NotNull;
import org.openbot.vehicle.Vehicle;
import timber.log.Timber;

public class OpenBotApplication extends Application {

  static Context context;
  public static Vehicle vehicle;

  public static Context getContext() {
    return context;
  }

  @Override
  public void onCreate() {
    super.onCreate();
    context = getApplicationContext();
    saveCrashesToFile();

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

  /**
   * Writes every uncaught exception to Documents/OpenBot/crash_<date>.txt, next to the autopilot
   * logs, so a crash can be read without adb. The default handler (Crashlytics, the system dialog)
   * still runs afterwards. Native crashes (e.g. inside TFLite) never get here.
   */
  private void saveCrashesToFile() {
    final Thread.UncaughtExceptionHandler previous = Thread.getDefaultUncaughtExceptionHandler();
    Thread.setDefaultUncaughtExceptionHandler(
        (thread, e) -> {
          try {
            File dir =
                new File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                    getString(R.string.app_name));
            dir.mkdirs();
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            try (PrintWriter out = new PrintWriter(new File(dir, "crash_" + stamp + ".txt"))) {
              out.println("Thread: " + thread.getName());
              out.println(
                  Build.MANUFACTURER + " " + Build.MODEL + ", Android " + Build.VERSION.RELEASE);
              out.println("App: " + BuildConfig.VERSION_NAME + " (" + BuildConfig.VERSION_CODE + ")");
              out.println();
              e.printStackTrace(out);
            }
          } catch (Throwable ignored) {
            // Never let the crash logger hide the original crash.
          }
          if (previous != null) previous.uncaughtException(thread, e);
        });
  }

  @Override
  public void onTerminate() {
    super.onTerminate();
  }
}
