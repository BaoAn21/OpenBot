package org.openbot.autopilot;

import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;
import androidx.core.content.ContextCompat;
import androidx.navigation.Navigation;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import org.jetbrains.annotations.NotNull;
import org.openbot.R;
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentAutopilotSessionBinding;
import org.openbot.env.BotToControllerEventBus;
import org.openbot.env.ImageUtils;
import org.openbot.tflite.Autopilot;
import org.openbot.tflite.AutopilotMem;
import org.openbot.tflite.Model;
import org.openbot.tflite.Network;
import org.openbot.utils.ConnectionUtils;
import org.openbot.utils.Constants;
import org.openbot.utils.Enums;
import org.openbot.utils.FileUtils;
import org.openbot.utils.PermissionUtils;
import org.openbot.vehicle.Control;
import timber.log.Timber;

/**
 * A full autopilot run on a track that has to be driven both ways: one model drives forward
 * (a memory model, {@code CMDNAV_MEM}: donkey_memory or cil_mobile_fast_memory) and another backward ({@code CMDNAV}, e.g.
 * cil_mobile_fast, or {@code CMDNAV_MEM}, e.g. cil_mobile_fast_memory). Both are loaded up front so
 * a switch is instant, but only the active one runs on a frame.
 *
 * <p>Which one drives follows the indicator: REVERSE means backward, anything else forward. The
 * on-screen button, the gamepad's triangle and the phone/web controllers all switch it that way,
 * the car shows its reverse light while reversing, and each model's cmd input is the indicator,
 * as it is logged in data collection. A future camera-based switch (traffic sign, AprilTag) plugs
 * in as an {@link IndicatorSource} and sets the indicator the same way.
 *
 * <p>On every switch the car stops for {@link #SWITCH_PAUSE_MS} before the other model takes over,
 * and the memories are cleared, so a memory model starts from the standstill it starts from in
 * training. Throttle is clamped to >= 0 forward and <= 0 backward: the cil models' output is
 * linear, and with the ESC in F/R mode a small throttle of the wrong sign would change direction
 * in the middle of a run. The switch decides the direction, the model only how much. A memory
 * model remembers the clamped output, what the car did, as its training data recorded it.
 *
 * <p>Moving the gamepad stick while autopilot is on takes over: you drive, the models don't run,
 * and the active direction's memory follows what the car does. Once the stick has been centred for
 * {@link #TAKEOVER_RELEASE_MS} the models drive again.
 *
 * <p>The forward spinner only offers CMDNAV_MEM models, the backward one CMDNAV and CMDNAV_MEM
 * models. Forward needs the memory input; backward needs a model that can reverse - a donkey_memory
 * model there would only stand still, its sigmoid throttle clamped to 0; use donkey_memory_reverse
 * (tanh throttle) instead.
 */
public class AutopilotSessionFragment extends CameraFragment {

    /** Picks the indicator from the camera, e.g. off a traffic sign or an AprilTag. */
    public interface IndicatorSource {
        /** Runs on the inference thread for every inferred frame; null keeps the current one. */
        @Nullable
        Enums.VehicleIndicator onFrame(Bitmap frame);
    }

    /** How long the car stands still on a switch before the other model drives. */
    private static final long SWITCH_PAUSE_MS = 500;

    /** After the stick is back in the centre, how long until the models drive again. */
    private static final long TAKEOVER_RELEASE_MS = 300;

    private FragmentAutopilotSessionBinding binding;
    private Handler handler;
    private HandlerThread handlerThread;

    private long lastProcessingTimeMs;
    private boolean computingNetwork = false;

    /** Null until something sets the indicator from the camera; switching is manual until then. */
    private IndicatorSource autoIndicator = null;

    private AutopilotMem forwardNet;
    /** An {@link Autopilot} (CMDNAV) or an {@link AutopilotMem} (CMDNAV_MEM). */
    private Network backwardNet;
    private Crop forwardCrop;
    private Crop backwardCrop;

    /**
     * The forward model's previous outputs, {@code [s, t] * memLen} oldest first, as in {@link
     * AutopilotMemFragment}. Only touched on the inference thread.
     */
    private float[] forwardMemory;

    /** The same for a backward memory model; null for a CMDNAV one. */
    private float[] backwardMemory;

    /**
     * The memories as last put on screen ("" for a backward model without one); the overlay redraws
     * from them on indicator changes.
     */
    private volatile String forwardMemoryText = "";

    private volatile String backwardMemoryText = "";

    /** Direction of the last inferred frame and end of the stop after a switch. Inference thread. */
    private boolean lastReverse = false;

    private long switchUntilMs = 0;

    /** The stick while you take over from the models. Written on the UI thread. */
    private volatile Control humanControl = new Control(0, 0);

    /** You drive until then: MAX_VALUE while the stick is off centre. */
    private volatile long humanUntilMs = 0;

    /** Set once the camera delivered a frame, so the crops can be sized to it. */
    private volatile boolean frameConfigured = false;

    private int sensorOrientation;

    private Model forwardModel;
    private Model backwardModel;
    private Network.Device device = Network.Device.CPU;
    private int numThreads = -1;

    /** A model's input bitmap and the frame-to-input transform; the two models can differ. */
    private static final class Crop {
        final Bitmap bitmap;
        final Matrix transform;

        Crop(Network network, int frameWidth, int frameHeight, int sensorOrientation) {
            bitmap =
                    Bitmap.createBitmap(
                            network.getImageSizeX(), network.getImageSizeY(), Bitmap.Config.ARGB_8888);
            transform =
                    ImageUtils.getTransformationMatrix(
                            frameWidth,
                            frameHeight,
                            bitmap.getWidth(),
                            bitmap.getHeight(),
                            sensorOrientation,
                            network.getCropRect(),
                            network.getMaintainAspect());
        }

        Bitmap from(Bitmap frame) {
            new Canvas(bitmap).drawBitmap(frame, transform, null);
            return bitmap;
        }
    }

    @Override
    public View onCreateView(
            @NotNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentAutopilotSessionBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        BotToControllerEventBus.emitEvent(ConnectionUtils.createStatus("FRAGMENT_TYPE", "CLOSE"));
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        binding.controllerContainer.speedInfo.setText(getString(R.string.speedInfo, "---,---"));
        initDeviceSpinner(binding.deviceSpinner, preferencesManager.getDevice());
        setNumThreads(preferencesManager.getNumThreads());
        binding.threads.setText(String.valueOf(getNumThreads()));
        binding.cameraToggle.setOnClickListener(v -> toggleCamera());
        forwardMemoryText = getString(R.string.memory_info_empty);
        updateIndicatorInfo();

        if (vehicle.getConnectionType().equals("USB")) {
            binding.usbToggle.setVisibility(View.VISIBLE);
            binding.bleToggle.setVisibility(View.GONE);
        } else if (vehicle.getConnectionType().equals("Bluetooth")) {
            binding.bleToggle.setVisibility(View.VISIBLE);
            binding.usbToggle.setVisibility(View.GONE);
        }

        List<String> forwardModels =
                getModelNames(
                        f -> f.type.equals(Model.TYPE.CMDNAV_MEM) && f.pathType != Model.PATH_TYPE.URL);
        List<String> backwardModels =
                getModelNames(
                        f ->
                                (f.type.equals(Model.TYPE.CMDNAV) || f.type.equals(Model.TYPE.CMDNAV_MEM))
                                        && f.pathType != Model.PATH_TYPE.URL);
        if (forwardModels.isEmpty() || backwardModels.isEmpty()) {
            Toast.makeText(
                            requireContext().getApplicationContext(),
                            R.string.no_session_models,
                            Toast.LENGTH_LONG)
                    .show();
        }
        initForwardModelSpinner(
                binding.forwardModelSpinner,
                forwardModels,
                preferencesManager.getAutopilotSessionForwardModel());
        // The base class' spinner, so models pushed from the server (always CMDNAV) land here.
        initModelSpinner(
                binding.backwardModelSpinner,
                backwardModels,
                preferencesManager.getAutopilotSessionBackwardModel());
        initServerSpinner(binding.serverSpinner);
        setAnalyserResolution(Enums.Preview.HD.getValue());
        binding.deviceSpinner.setOnItemSelectedListener(
                new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                        String selected = parent.getItemAtPosition(position).toString();
                        setDevice(Network.Device.valueOf(selected.toUpperCase()));
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {
                    }
                });

        binding.plus.setOnClickListener(
                v -> {
                    String threads = binding.threads.getText().toString().trim();
                    int numThreads = Integer.parseInt(threads);
                    if (numThreads >= 9) return;
                    setNumThreads(++numThreads);
                    binding.threads.setText(String.valueOf(numThreads));
                });
        binding.minus.setOnClickListener(
                v -> {
                    String threads = binding.threads.getText().toString().trim();
                    int numThreads = Integer.parseInt(threads);
                    if (numThreads == 1) return;
                    setNumThreads(--numThreads);
                    binding.threads.setText(String.valueOf(numThreads));
                });
        BottomSheetBehavior.from(binding.aiBottomSheet).setState(BottomSheetBehavior.STATE_EXPANDED);

        mViewModel
                .getUsbStatus()
                .observe(getViewLifecycleOwner(), status -> binding.usbToggle.setChecked(status));

        binding.usbToggle.setChecked(vehicle.isUsbConnected());
        binding.bleToggle.setChecked(vehicle.bleConnected());

        binding.usbToggle.setOnClickListener(
                v -> {
                    binding.usbToggle.setChecked(vehicle.isUsbConnected());
                    Navigation.findNavController(requireView()).navigate(R.id.open_usb_fragment);
                });
        binding.bleToggle.setOnClickListener(
                v -> {
                    binding.bleToggle.setChecked(vehicle.bleConnected());
                    Navigation.findNavController(requireView()).navigate(R.id.open_bluetooth_fragment);
                });

        setSpeedMode(Enums.SpeedMode.getByID(preferencesManager.getSpeedMode()));
        setControlMode(Enums.ControlMode.getByID(preferencesManager.getControlMode()));
        setDriveMode(Enums.DriveMode.getByID(preferencesManager.getDriveMode()));

        binding.controllerContainer.controlMode.setOnClickListener(
                v -> {
                    Enums.ControlMode controlMode =
                            Enums.ControlMode.getByID(preferencesManager.getControlMode());
                    if (controlMode != null) setControlMode(Enums.switchControlMode(controlMode));
                });
        binding.controllerContainer.driveMode.setOnClickListener(
                v -> setDriveMode(Enums.switchDriveMode(vehicle.getDriveMode())));

        binding.controllerContainer.speedMode.setOnClickListener(
                v ->
                        setSpeedMode(
                                Enums.toggleSpeed(
                                        Enums.Direction.CYCLIC.getValue(),
                                        Enums.SpeedMode.getByID(preferencesManager.getSpeedMode()))));

        // Same path as the gamepad's triangle: reverse on, or off again, with the controller events.
        binding.directionButton.setOnClickListener(
                v ->
                        toggleIndicatorButton(
                                Enums.VehicleIndicator.REVERSE, Constants.CMD_INDICATOR_REVERSE));

        binding.autoSwitch.setOnClickListener(v -> setNetworkEnabled(binding.autoSwitch.isChecked()));
        BotToControllerEventBus.emitEvent(
                ConnectionUtils.createStatus(
                        "FRAGMENT_TYPE", Enums.FragmentType.AUTOPILOT.getFragment()));
    }

    /**
     * Like {@link #initModelSpinner}, which can only drive one spinner, for the forward model. Matches
     * names exactly, so a forward model never resolves to a backward one with a similar name.
     */
    private void initForwardModelSpinner(Spinner spinner, List<String> models, String selected) {
        ArrayAdapter<String> adapter = new ArrayAdapter<>(requireContext(), R.layout.spinner_item, models);
        adapter.setDropDownViewResource(android.R.layout.simple_dropdown_item_1line);
        spinner.setAdapter(adapter);
        if (!selected.isEmpty())
            spinner.setSelection(Math.max(0, adapter.getPosition(FileUtils.nameWithoutExtension(selected))));
        spinner.setOnItemSelectedListener(
                new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                        String name = parent.getItemAtPosition(position).toString();
                        masterList.stream()
                                .filter(
                                        f ->
                                                f.type.equals(Model.TYPE.CMDNAV_MEM)
                                                        && FileUtils.nameWithoutExtension(f.name).equals(name))
                                .findFirst()
                                .ifPresent(model -> setForwardModel(model));
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {
                    }
                });
    }

    /** Sizes the crops to the camera frame; runs once the first frame is in. */
    private void configureFrame() {
        sensorOrientation = 90 - ImageUtils.getScreenOrientation(requireActivity());
        Timber.i("Camera orientation relative to screen canvas: %d", sensorOrientation);
        frameConfigured = true;
        recreateNetworks(true, true);
    }

    protected void onInferenceConfigurationChanged() {
        recreateLater(true, true);
    }

    private void recreateLater(boolean forward, boolean backward) {
        computingNetwork = false;
        // Deferred until the camera delivers frames, the crops need the frame size.
        if (!frameConfigured) return;
        runInBackground(() -> recreateNetworks(forward, backward));
    }

    /** Runs on the inference thread, except for the first time, from the camera's first frame. */
    private void recreateNetworks(boolean forward, boolean backward) {
        final int frameWidth = getMaxAnalyseImageSize().getWidth();
        final int frameHeight = getMaxAnalyseImageSize().getHeight();
        if (forward) {
            if (forwardNet != null) forwardNet.close();
            forwardNet = null;
            forwardCrop = null;
            if (forwardModel != null) {
                try {
                    Timber.d(
                            "Creating forward autopilot (model=%s, device=%s, numThreads=%d)",
                            forwardModel, device, numThreads);
                    AutopilotMem net = new AutopilotMem(requireActivity(), forwardModel, device, numThreads);
                    forwardCrop = new Crop(net, frameWidth, frameHeight, sensorOrientation);
                    forwardMemory = new float[2 * net.getMemLen()];
                    showMemory(false, forwardMemory.clone());
                    forwardNet = net;
                } catch (IllegalArgumentException | IOException e) {
                    showNetworkError(e);
                }
            }
        }
        if (backward) {
            if (backwardNet != null) backwardNet.close();
            backwardNet = null;
            backwardCrop = null;
            backwardMemory = null;
            backwardMemoryText = "";
            if (backwardModel != null) {
                try {
                    Timber.d(
                            "Creating backward autopilot (model=%s, device=%s, numThreads=%d)",
                            backwardModel, device, numThreads);
                    if (backwardModel.type.equals(Model.TYPE.CMDNAV_MEM)) {
                        AutopilotMem net =
                                new AutopilotMem(requireActivity(), backwardModel, device, numThreads);
                        backwardCrop = new Crop(net, frameWidth, frameHeight, sensorOrientation);
                        backwardMemory = new float[2 * net.getMemLen()];
                        showMemory(true, backwardMemory.clone());
                        backwardNet = net;
                    } else {
                        Autopilot net = new Autopilot(requireActivity(), backwardModel, device, numThreads);
                        backwardCrop = new Crop(net, frameWidth, frameHeight, sensorOrientation);
                        backwardNet = net;
                    }
                } catch (IllegalArgumentException | IOException e) {
                    showNetworkError(e);
                }
            }
        }
        requireActivity().runOnUiThread(this::updateInputResolution);
    }

    private void showNetworkError(Exception e) {
        Timber.e(e, "Failed to create network.");
        requireActivity()
                .runOnUiThread(
                        () ->
                                Toast.makeText(
                                                requireContext().getApplicationContext(),
                                                e.getMessage(),
                                                Toast.LENGTH_LONG)
                                        .show());
    }

    /** Input sizes as forward / backward, "-" for a model that is not loaded. */
    private void updateInputResolution() {
        final Network forward = forwardNet;
        final Network backward = backwardNet;
        binding.inputResolution.setText(
                String.format(Locale.US, "%s / %s", sizeOf(forward), sizeOf(backward)));
    }

    private static String sizeOf(@Nullable Network network) {
        if (network == null) return "-";
        return network.getImageSizeX() + "x" + network.getImageSizeY();
    }

    @Override
    public synchronized void onResume() {
        frameConfigured = false;
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        binding.bleToggle.setChecked(vehicle.bleConnected());
        super.onResume();
    }

    @Override
    public synchronized void onPause() {
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            e.printStackTrace();
        }
        super.onPause();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    @Override
    protected void processUSBData(String data) {
        binding.controllerContainer.speedInfo.setText(
                getString(
                        R.string.speedInfo,
                        String.format(
                                Locale.US, "%3.0f,%3.0f", vehicle.getLeftWheelRpm(), vehicle.getRightWheelRpm())));
    }

    @Override
    protected void processKeyEvent(KeyEvent keyCode) {
        if (binding.autoSwitch.isChecked()
                && (keyCode.getKeyCode() == KeyEvent.KEYCODE_BUTTON_THUMBL
                || keyCode.getKeyCode() == KeyEvent.KEYCODE_BUTTON_THUMBR)) {
            audioPlayer.playFromString("Autopilot active. Cannot change speed mode.");
        } else {
            super.processKeyEvent(keyCode);
        }
    }

    @Override
    protected void processControllerKeyData(String commandType) {
        switch (commandType) {
            case Constants.CMD_DRIVE:
                binding.controllerContainer.controlInfo.setText(
                        String.format(Locale.US, "%.0f,%.0f", vehicle.getSteering(), vehicle.getThrottle()));
                break;

            case Constants.CMD_DRIVE_MODE:
                setDriveMode(Enums.switchDriveMode(vehicle.getDriveMode()));
                break;

            case Constants.CMD_SPEED_DOWN:
                setSpeedMode(
                        Enums.toggleSpeed(
                                Enums.Direction.DOWN.getValue(),
                                Enums.SpeedMode.getByID(preferencesManager.getSpeedMode())));
                break;

            case Constants.CMD_SPEED_UP:
                setSpeedMode(
                        Enums.toggleSpeed(
                                Enums.Direction.UP.getValue(),
                                Enums.SpeedMode.getByID(preferencesManager.getSpeedMode())));
                break;

            case Constants.CMD_INDICATOR_LEFT:
            case Constants.CMD_INDICATOR_RIGHT:
            case Constants.CMD_INDICATOR_STOP:
            case Constants.CMD_INDICATOR_REVERSE:
                updateIndicatorInfo();
                break;
        }
    }

    @Override
    protected void handleJoystickControl(Control control) {
        if (binding == null || !binding.autoSwitch.isChecked()) {
            super.handleJoystickControl(control);
            return;
        }
        // Any stick input takes over from the models until the stick has been centred for
        // TAKEOVER_RELEASE_MS. drive() keeps sending it each frame, so the models can't
        // overwrite it.
        humanControl = control;
        if (control.getSteering() != 0 || control.getThrottle() != 0) {
            humanUntilMs = Long.MAX_VALUE;
            super.handleJoystickControl(control); // right away, not on the next frame
        } else if (humanUntilMs == Long.MAX_VALUE) {
            humanUntilMs = SystemClock.elapsedRealtime() + TAKEOVER_RELEASE_MS;
        }
    }

    private boolean isTakenOver() {
        return SystemClock.elapsedRealtime() < humanUntilMs;
    }

    /** Which model drives. Read on the inference thread, so a switch lands between two frames. */
    private boolean isReversing() {
        return vehicle.getIndicator() == Enums.VehicleIndicator.REVERSE.getValue();
    }

    /**
     * The overlay and the direction button. The overlay shows the indicator with its cmd value -
     * every model gets it as cmd - and the active model's memory as it sees it, if it has one. Runs
     * on the UI thread.
     */
    private void updateIndicatorInfo() {
        final boolean reverse = isReversing();
        binding.directionButton.setText(
                reverse ? R.string.direction_backward : R.string.direction_forward);
        binding.directionButton.setBackgroundTintList(
                ColorStateList.valueOf(
                        ContextCompat.getColor(requireContext(), reverse ? R.color.red : R.color.green)));

        int indicator = vehicle.getIndicator();
        int label;
        int color;
        if (indicator == Enums.VehicleIndicator.LEFT.getValue()) {
            label = R.string.indicator_left;
            color = R.color.indicator;
        } else if (indicator == Enums.VehicleIndicator.RIGHT.getValue()) {
            label = R.string.indicator_right;
            color = R.color.indicator;
        } else if (reverse) {
            label = R.string.indicator_reverse;
            color = R.color.red;
        } else {
            label = R.string.indicator_forward;
            color = R.color.green;
        }
        final String memoryText = reverse ? backwardMemoryText : forwardMemoryText;
        binding.indicatorInfo.setText(
                memoryText.isEmpty() ? getString(label) : getString(label) + "\n" + memoryText);
        binding.indicatorInfo.setTextColor(ContextCompat.getColor(requireContext(), color));
    }

    /** Both memories back to donkeycar's starting (0, 0) per pair. Runs on the inference thread. */
    private void resetMemory() {
        if (forwardMemory != null) {
            Arrays.fill(forwardMemory, 0f);
            showMemory(false, forwardMemory.clone());
        }
        if (backwardMemory != null) {
            Arrays.fill(backwardMemory, 0f);
            showMemory(true, backwardMemory.clone());
        }
    }

    /** Shifts the oldest pair out and the newest control in. Runs on the inference thread. */
    private static void pushMemory(float[] memory, Control output) {
        System.arraycopy(memory, 2, memory, 0, memory.length - 2);
        memory[memory.length - 2] = output.getSteering() / Control.MAX;
        memory[memory.length - 1] = output.getThrottle() / Control.MAX;
    }

    /** Puts a memory on screen, oldest pair first, as the model sees it. */
    private void showMemory(boolean backward, float[] snapshot) {
        StringBuilder text = new StringBuilder("MEM");
        for (int i = 0; i < snapshot.length; i += 2) {
            text.append(i == 0 ? "  " : " | ")
                    .append(String.format(Locale.US, "%+.2f,%.2f", snapshot[i], snapshot[i + 1]));
        }
        if (backward) backwardMemoryText = text.toString();
        else forwardMemoryText = text.toString();
        requireActivity().runOnUiThread(this::updateIndicatorInfo);
    }

    @Override
    protected void handleNetworkCommand() {
        setNetworkEnabledWithAudio(!binding.autoSwitch.isChecked());
    }

    @Override
    protected boolean isNetworkModeEnabled() {
        return binding.autoSwitch.isChecked();
    }

    private void setNetworkEnabledWithAudio(boolean b) {
        setNetworkEnabled(b);

        if (b) {
            audioPlayer.play(voice, "network_enabled.mp3");
            runInBackground(
                    () -> {
                        try {
                            TimeUnit.MILLISECONDS.sleep(lastProcessingTimeMs);
                            vehicle.setControl(0, 0);
                            requireActivity()
                                    .runOnUiThread(() -> binding.inferenceInfo.setText(R.string.time_fps));
                        } catch (InterruptedException e) {
                            Timber.e(e, "Got interrupted.");
                        }
                    });
        } else audioPlayer.playDriveMode(voice, vehicle.getDriveMode());
    }

    private void setNetworkEnabled(boolean b) {
        humanUntilMs = 0;
        binding.autoSwitch.setChecked(b);
        emitNetworkStatus(b);
        binding.controllerContainer.controlMode.setEnabled(!b);
        binding.controllerContainer.driveMode.setEnabled(!b);
        binding.controllerContainer.speedMode.setEnabled(!b);

        binding.controllerContainer.controlMode.setAlpha(b ? 0.5f : 1f);
        binding.controllerContainer.driveMode.setAlpha(b ? 0.5f : 1f);
        binding.controllerContainer.speedMode.setAlpha(b ? 0.5f : 1f);

        if (!b) {
            setSpeedMode(Enums.SpeedMode.getByID(preferencesManager.getSpeedMode()));
            handler.postDelayed(() -> vehicle.setControl(0, 0), 500);
        } else {
            binding.controllerContainer.speedMode.setImageResource(R.drawable.ic_speed_high);
            vehicle.setSpeedMultiplier(Enums.SpeedMode.FAST.getValue());
            // Queued on the inference thread, so it lands before the next inference rather than
            // in the middle of one. The run starts in whichever direction the indicator says,
            // without a stop, the car is standing still anyway.
            runInBackground(
                    () -> {
                        resetMemory();
                        lastReverse = isReversing();
                        switchUntilMs = 0;
                    });
        }
    }

    private long frameNum = 0;

    @Override
    protected void processFrame(Bitmap bitmap, ImageProxy image) {
        if (!frameConfigured) configureFrame();

        ++frameNum;
        if (binding != null && binding.autoSwitch.isChecked()) {
            // If network is busy, return.
            if (computingNetwork) {
                return;
            }

            computingNetwork = true;
            Timber.i("Putting image " + frameNum + " for detection in bg thread.");

            runInBackground(
                    () -> {
                        final long startTime = SystemClock.elapsedRealtime();
                        if (autoIndicator != null) {
                            Enums.VehicleIndicator detected = autoIndicator.onFrame(bitmap);
                            if (detected != null && detected.getValue() != vehicle.getIndicator()) {
                                requireActivity().runOnUiThread(() -> setIndicator(detected));
                            }
                        }
                        drive(bitmap);
                        lastProcessingTimeMs = SystemClock.elapsedRealtime() - startTime;
                        computingNetwork = false;
                    });
            // Read once: the inference thread can set it to 0 (a frame of the switch stop) before
            // the UI thread runs this.
            final long processingMs = lastProcessingTimeMs;
            final boolean manual = isTakenOver();
            if (manual || processingMs > 0)
                requireActivity()
                        .runOnUiThread(
                                () ->
                                        binding.inferenceInfo.setText(
                                                manual
                                                        ? getString(R.string.takeover_active)
                                                        : String.format(Locale.US, "%d fps", 1000 / processingMs)));
        }
    }

    /** One frame of driving with the model the indicator picks. Runs on the inference thread. */
    private void drive(Bitmap frame) {
        final boolean reverse = isReversing();
        if (reverse != lastReverse) {
            lastReverse = reverse;
            resetMemory();
            switchUntilMs = SystemClock.elapsedRealtime() + SWITCH_PAUSE_MS;
            Timber.i("Switched to the %s model", reverse ? "backward" : "forward");
        }
        if (isTakenOver()) {
            // You drive. The active model's memory gets what the car actually did, as in the
            // recorded data, so it carries on from your driving when you let go.
            Control human = humanControl;
            float[] memory = reverse ? backwardMemory : forwardMemory;
            if (memory != null) {
                pushMemory(memory, human);
                showMemory(reverse, memory.clone());
            }
            handleDriveCommand(human);
            return;
        }
        if (SystemClock.elapsedRealtime() < switchUntilMs) {
            // Let the car stop before the other model drives.
            handleDriveCommand(new Control(0, 0));
        } else if (reverse) {
            if (backwardNet == null) {
                handleDriveCommand(new Control(0, 0));
                return;
            }
            Bitmap input = backwardCrop.from(frame);
            Control output =
                    backwardNet instanceof AutopilotMem
                            ? ((AutopilotMem) backwardNet)
                                    .recognizeImage(input, backwardMemory, vehicle.getIndicator())
                            : ((Autopilot) backwardNet).recognizeImage(input, vehicle.getIndicator());
            Control reversing = new Control(output.getSteering(), Math.min(output.getThrottle(), 0));
            if (backwardMemory != null) {
                pushMemory(backwardMemory, reversing);
                showMemory(true, backwardMemory.clone());
            }
            handleDriveCommand(reversing);
        } else {
            if (forwardNet == null || forwardMemory == null) {
                handleDriveCommand(new Control(0, 0));
                return;
            }
            Control output =
                    forwardNet.recognizeImage(
                            forwardCrop.from(frame), forwardMemory, vehicle.getIndicator());
            // cil_mobile_fast_memory's throttle is linear: never let it reverse mid-forward.
            Control forward = new Control(output.getSteering(), Math.max(output.getThrottle(), 0));
            pushMemory(forwardMemory, forward);
            showMemory(false, forwardMemory.clone());
            handleDriveCommand(forward);
        }
    }

    /** For an {@link IndicatorSource}: what the gamepad buttons do, from the UI thread. */
    private void setIndicator(Enums.VehicleIndicator indicator) {
        if (vehicle.getIndicator() == indicator.getValue()) return;
        switch (indicator) {
            case LEFT:
                toggleIndicatorButton(indicator, Constants.CMD_INDICATOR_LEFT);
                break;
            case RIGHT:
                toggleIndicatorButton(indicator, Constants.CMD_INDICATOR_RIGHT);
                break;
            case REVERSE:
                toggleIndicatorButton(indicator, Constants.CMD_INDICATOR_REVERSE);
                break;
            case STOP:
                // Pressing the active one again is what turns an indicator off.
                for (Enums.VehicleIndicator active : Enums.VehicleIndicator.values()) {
                    if (active.getValue() == vehicle.getIndicator()) {
                        toggleIndicatorButton(active, Constants.CMD_INDICATOR_STOP);
                        break;
                    }
                }
                break;
        }
    }

    protected void handleDriveCommand(Control control) {
        vehicle.setControl(control);
        float steering = vehicle.getSteering();
        float throttle = vehicle.getThrottle();
        requireActivity()
                .runOnUiThread(
                        () -> binding.controllerContainer.controlInfo.setText(
                                String.format(Locale.US, "%.0f,%.0f", steering, throttle)));
    }

    @Override
    public void onConnectionEstablished(String ipAddress) {
        requireActivity().runOnUiThread(() -> binding.ipAddress.setText(ipAddress));
    }

    private void connectWebController() {
        phoneController.connectWebServer();
        Enums.DriveMode oldDriveMode = currentDriveMode;
        // Currently only dual drive mode supported
        setDriveMode(Enums.DriveMode.GAME);
        binding.controllerContainer.driveMode.setAlpha(0.5f);
        binding.controllerContainer.driveMode.setEnabled(false);
        preferencesManager.setDriveMode(oldDriveMode.getValue());
    }

    private void setForwardModel(Model model) {
        if (forwardModel != model) {
            Timber.d("Updating forward model: %s", model);
            forwardModel = model;
            preferencesManager.setAutopilotSessionForwardModel(model.name);
            recreateLater(true, false);
        }
    }

    /** The backward model; called by the base class' spinner, see {@link #onViewCreated}. */
    @Override
    protected void setModel(Model model) {
        if (backwardModel != model) {
            Timber.d("Updating backward model: %s", model);
            backwardModel = model;
            preferencesManager.setAutopilotSessionBackwardModel(model.name);
            recreateLater(false, true);
        }
    }

    protected Network.Device getDevice() {
        return device;
    }

    private void setDevice(Network.Device device) {
        if (this.device != device) {
            Timber.d("Updating  device: %s", device);
            this.device = device;
            final boolean threadsEnabled = device == Network.Device.CPU;
            binding.plus.setEnabled(threadsEnabled);
            binding.minus.setEnabled(threadsEnabled);
            binding.threads.setText(
                    threadsEnabled ? String.valueOf(numThreads) : getString(R.string.n_a));
            if (threadsEnabled) binding.threads.setTextColor(Color.BLACK);
            else binding.threads.setTextColor(Color.GRAY);
            preferencesManager.setDevice(device.ordinal());
            onInferenceConfigurationChanged();
        }
    }

    protected int getNumThreads() {
        return numThreads;
    }

    private void setNumThreads(int numThreads) {
        if (this.numThreads != numThreads) {
            Timber.d("Updating  numThreads: %s", numThreads);
            this.numThreads = numThreads;
            preferencesManager.setNumThreads(numThreads);
            onInferenceConfigurationChanged();
        }
    }

    private void setSpeedMode(Enums.SpeedMode speedMode) {
        if (speedMode != null && !binding.autoSwitch.isChecked()) {
            switch (speedMode) {
                case SLOW:
                    binding.controllerContainer.speedMode.setImageResource(R.drawable.ic_speed_low);
                    break;
                case NORMAL:
                    binding.controllerContainer.speedMode.setImageResource(R.drawable.ic_speed_medium);
                    break;
                case FAST:
                    binding.controllerContainer.speedMode.setImageResource(R.drawable.ic_speed_high);
                    break;
            }

            Timber.d("Updating  controlSpeed: %s", speedMode);
            preferencesManager.setSpeedMode(speedMode.getValue());
            vehicle.setSpeedMultiplier(speedMode.getValue());
        }
    }

    private void setControlMode(Enums.ControlMode controlMode) {
        if (controlMode != null) {
            switch (controlMode) {
                case GAMEPAD:
                    binding.controllerContainer.controlMode.setImageResource(R.drawable.ic_controller);
                    disconnectPhoneController();
                    break;
                case PHONE:
                    binding.controllerContainer.controlMode.setImageResource(R.drawable.ic_phone);
                    if (!PermissionUtils.hasControllerPermissions(requireActivity()))
                        requestPermissionLauncher.launch(Constants.PERMISSIONS_CONTROLLER);
                    else connectPhoneController();
                    break;
                case WEBSERVER:
                    binding.controllerContainer.controlMode.setImageResource(R.drawable.ic_server);
                    if (!PermissionUtils.hasControllerPermissions(requireActivity()))
                        requestPermissionLauncher.launch(Constants.PERMISSIONS_CONTROLLER);
                    else connectWebController();
                    break;
            }
            Timber.d("Updating  controlMode: %s", controlMode);
            preferencesManager.setControlMode(controlMode.getValue());
        }
    }

    protected void setDriveMode(Enums.DriveMode driveMode) {
        if (driveMode != null) {
            switch (driveMode) {
                case DUAL:
                    binding.controllerContainer.driveMode.setImageResource(R.drawable.ic_dual);
                    break;
                case GAME:
                    binding.controllerContainer.driveMode.setImageResource(R.drawable.ic_game);
                    break;
                case JOYSTICK:
                    binding.controllerContainer.driveMode.setImageResource(R.drawable.ic_joystick);
                    break;
            }

            Timber.d("Updating  driveMode: %s", driveMode);
            vehicle.setDriveMode(driveMode);
            preferencesManager.setDriveMode(driveMode.getValue());
        }
    }

    private void connectPhoneController() {
        phoneController.connect(requireContext());
        Enums.DriveMode oldDriveMode = currentDriveMode;
        // Currently only dual drive mode supported
        setDriveMode(Enums.DriveMode.DUAL);
        binding.controllerContainer.driveMode.setAlpha(0.5f);
        binding.controllerContainer.driveMode.setEnabled(false);
        preferencesManager.setDriveMode(oldDriveMode.getValue());
    }

    private void disconnectPhoneController() {
        phoneController.disconnect();
        setDriveMode(Enums.DriveMode.getByID(preferencesManager.getDriveMode()));
        binding.controllerContainer.driveMode.setEnabled(true);
        binding.controllerContainer.driveMode.setAlpha(1.0f);
    }
}
