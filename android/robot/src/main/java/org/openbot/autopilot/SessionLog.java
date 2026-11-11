package org.openbot.autopilot;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.os.SystemClock;
import androidx.annotation.Nullable;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import org.openbot.R;
import org.openbot.env.ImageUtils;
import org.openbot.logging.LogDataUtils;
import org.openbot.logging.SensorService;
import org.openbot.vehicle.Vehicle;
import org.zeroturnaround.zip.ZipUtil;
import org.zeroturnaround.zip.commons.FileUtils;
import timber.log.Timber;

/**
 * Records an Autopilot Session run in the data logger's layout, so the policy side reads it like
 * any recording: {@code images/<frame>_crop.jpeg} and {@code sensor_data/} with rgbFrames.txt,
 * ctrlLog.txt, indicatorLog.txt and the phone sensors, all written by {@link SensorService}.
 *
 * <p>On top of that, {@code sensor_data/autopilotLog.txt} has a row per inferred frame: who drove
 * ({@code model}, {@code human} while you take over, {@code pause} on a direction switch, {@code
 * none} without a model for that direction), the model's raw prediction - also while you drive, so
 * it can be compared with what you did - the command sent and the memory the model saw.
 *
 * <p>ctrlLog.txt holds every command sent, the models' included: filter it by autopilotLog's source
 * before training on it. Images and rows are written on a thread of their own, not the inference
 * thread. Stopping zips the folder next to it and deletes it, as the data logger does.
 */
class SessionLog {

    static final String HEADER =
            "timestamp[ns],frame,source,direction,model,pred_steering,pred_throttle,"
                    + "steering,throttle,inference[ms],memory";

    private final Activity activity;
    private final Vehicle vehicle;
    private final File folder;
    private final Intent sensorIntent;
    private final HandlerThread writerThread;
    private final Handler writer;

    /** autopilotLog.txt. Only touched on the writer thread. */
    @Nullable private BufferedWriter rows;

    private volatile Messenger sensorMessenger;

    private final ServiceConnection sensorConnection =
            new ServiceConnection() {
                @Override
                public void onServiceConnected(ComponentName className, IBinder binder) {
                    sensorMessenger = new Messenger(binder);
                    // As the data logger does: the state at the start, once the service has opened
                    // its logs.
                    writer.postDelayed(
                            () -> {
                                control(vehicle.getSteering(), vehicle.getThrottle());
                                indicator(vehicle.getIndicator());
                            },
                            500);
                }

                @Override
                public void onServiceDisconnected(ComponentName className) {
                    sensorMessenger = null;
                }
            };

    SessionLog(Activity activity, Vehicle vehicle) {
        this.activity = activity;
        this.vehicle = vehicle;
        folder =
                new File(
                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                        activity.getString(R.string.app_name)
                                + File.separator
                                + "autopilot_"
                                + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()));
        final File sensorFolder = new File(folder, "sensor_data");

        writerThread = new HandlerThread("session-log");
        writerThread.start();
        writer = new Handler(writerThread.getLooper());
        writer.post(
                () -> {
                    try {
                        if (!sensorFolder.exists() && !sensorFolder.mkdirs())
                            throw new IOException("Cannot create " + sensorFolder);
                        rows = new BufferedWriter(new FileWriter(new File(sensorFolder, "autopilotLog.txt")));
                        rows.write(HEADER + "\n");
                    } catch (IOException e) {
                        Timber.e(e, "Cannot open autopilotLog.txt");
                        rows = null;
                    }
                });

        sensorIntent = new Intent(activity, SensorService.class);
        sensorIntent.putExtra("logFolder", sensorFolder.getAbsolutePath());
        activity.startService(sensorIntent);
        activity.bindService(sensorIntent, sensorConnection, Context.BIND_AUTO_CREATE);
    }

    String getFolderName() {
        return folder.getName();
    }

    /** A frame that goes to the model, with the time it came in. Any thread. */
    void frame(long frameNum) {
        send(LogDataUtils.generateFrameNumberMessage(frameNum));
    }

    /** A command sent to the car, as {@link Vehicle} has it. Any thread. */
    void control(float steering, float throttle) {
        send(LogDataUtils.generateControlDataMessage((int) steering, (int) throttle));
    }

    /** The indicator, on every change. Any thread. */
    void indicator(int indicator) {
        send(LogDataUtils.generateIndicatorMessage(indicator));
    }

    /**
     * One autopilotLog row, timestamped now, and the frame's model input as {@code
     * <frame>_crop.jpeg}.
     *
     * @param row everything after {@code timestamp,frame}, as in {@link #HEADER}
     * @param crop a copy the log can keep and recycle, or null when no model is loaded
     */
    void row(long frameNum, String row, @Nullable Bitmap crop) {
        final long timestamp = SystemClock.elapsedRealtimeNanos();
        writer.post(
                () -> {
                    if (crop != null) {
                        ImageUtils.saveBitmap(
                                crop, new File(folder, "images").getAbsolutePath(), frameNum + "_crop.jpeg");
                        crop.recycle();
                    }
                    if (rows == null) return;
                    try {
                        rows.write(timestamp + "," + frameNum + "," + row + "\n");
                    } catch (IOException e) {
                        Timber.e(e, "Cannot write autopilotLog.txt");
                    }
                });
    }

    /**
     * Stops the sensor logs, writes what is still queued, then zips the folder. Call it after the
     * last {@link #row}.
     */
    void stop() {
        sensorMessenger = null;
        activity.unbindService(sensorConnection);
        activity.stopService(sensorIntent);
        // SensorService closes its logs in onDestroy, on the main thread some time after this;
        // give it that time before zipping, or ctrlLog.txt can lose its last lines.
        writer.postDelayed(
                () -> {
                    try {
                        if (rows != null) rows.close();
                    } catch (IOException e) {
                        Timber.e(e, "Cannot close autopilotLog.txt");
                    }
                    rows = null;
                    ZipUtil.pack(folder, new File(folder.getPath() + ".zip"));
                    FileUtils.deleteQuietly(folder);
                    writerThread.quitSafely();
                },
                500);
    }

    private void send(Message msg) {
        final Messenger messenger = sensorMessenger;
        if (messenger == null) return;
        try {
            messenger.send(msg);
        } catch (RemoteException e) {
            Timber.e(e, "SensorService is gone");
        }
    }
}
