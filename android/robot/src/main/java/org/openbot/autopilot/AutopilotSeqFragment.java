package org.openbot.autopilot;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.TypedValue;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;
import androidx.navigation.Navigation;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.jetbrains.annotations.NotNull;
import org.openbot.R;
import org.openbot.common.CameraFragment;
import org.openbot.databinding.FragmentAutopilotBinding;
import org.openbot.env.BorderedText;
import org.openbot.env.BotToControllerEventBus;
import org.openbot.env.ImageUtils;
import org.openbot.tflite.AutopilotSeq;
import org.openbot.tflite.Model;
import org.openbot.tflite.Network;
import org.openbot.tracking.MultiBoxTracker;
import org.openbot.utils.ConnectionUtils;
import org.openbot.utils.Constants;
import org.openbot.utils.Enums;
import org.openbot.utils.PermissionUtils;
import org.openbot.vehicle.Control;
import timber.log.Timber;

/**
 * Autopilot driven by a sequence policy ({@code pilot_net_seq}): the network is fed a stack of
 * recent frames instead of a single one, which is what gives it the motion cue it needs to drive
 * both forward and in reverse.
 *
 * <p>Kept separate from {@link AutopilotFragment} rather than folded into it because the two feed
 * their networks differently - this one has to keep a frame history and sample it on a clock - and
 * because a single-frame model loaded here (or a stack model loaded there) would be a silent
 * mismatch. The model spinner only offers CMDNAV_SEQ models for the same reason.
 */
public class AutopilotSeqFragment extends CameraFragment {

    /**
     * Frame rate the training data was recorded at. The stack offsets count frames on that
     * timeline, so this is what turns them into the ages we look up here. Measured across the
     * recorded sessions at 33.17ms per frame.
     */
    private static final float DATASET_FRAME_PERIOD_MS = 1000f / 30f;

    /** Offsets to fall back on when the model name does not carry the ones it was trained with. */
    private static final int[] DEFAULT_SEQ_STRIDE = {2};

    // options for drop down in object nav?
    private FragmentAutopilotBinding binding;
    private Handler handler;
    private HandlerThread handlerThread;

    private long lastProcessingTimeMs;
    private boolean computingNetwork = false;

    private static final float TEXT_SIZE_DIP = 10;

    private AutopilotSeq autopilot;

    private Matrix frameToCropTransform;
    private volatile CropTarget cropTarget;
    private int sensorOrientation;

    /**
     * The scaled-down frame and the canvas that draws into it, kept as one value so the camera
     * thread can never pick up a canvas from one model and a bitmap from the next.
     */
    private static final class CropTarget {
        final Bitmap bitmap;
        final Canvas canvas;

        CropTarget(int width, int height) {
            bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            canvas = new Canvas(bitmap);
        }
    }

    /** Every preview frame lands here; the stack is sampled out of it by timestamp. */
    private volatile FrameHistory frameHistory;

    /** Scratch the inference thread fills from the history, oldest frame first. */
    private volatile int[][] frameStack;

    /** Age of each frame in the stack, newest first, matching the training offsets. */
    private volatile float[] frameAgesMs;

    /** Span of the stack actually assembled last inference, for the readout. */
    private volatile long lastStackSpanMs = 0;

    private MultiBoxTracker tracker;

    private Model model;
    private Network.Device device = Network.Device.CPU;
    private int numThreads = -1;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(
            @NotNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        binding = FragmentAutopilotBinding.inflate(inflater, container, false);
        return inflateFragment(binding, inflater, container);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        BotToControllerEventBus.emitEvent(
            ConnectionUtils.createStatus("FRAGMENT_TYPE", "CLOSE"));
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        binding.controllerContainer.speedInfo.setText(getString(R.string.speedInfo, "---,---"));
        initDeviceSpinner(binding.deviceSpinner, preferencesManager.getDevice());
        setNumThreads(preferencesManager.getNumThreads());
        binding.threads.setText(String.valueOf(getNumThreads()));
        binding.cameraToggle.setOnClickListener(v -> toggleCamera());

        if (vehicle.getConnectionType().equals("USB")) {
            binding.usbToggle.setVisibility(View.VISIBLE);
            binding.bleToggle.setVisibility(View.GONE);
        } else if (vehicle.getConnectionType().equals("Bluetooth")) {
            binding.bleToggle.setVisibility(View.VISIBLE);
            binding.usbToggle.setVisibility(View.GONE);
        }
        List<String> models =
                getModelNames(
                        f -> f.type.equals(Model.TYPE.CMDNAV_SEQ) && f.pathType != Model.PATH_TYPE.URL);
        if (models.isEmpty()) {
            Toast.makeText(
                            requireContext().getApplicationContext(),
                            R.string.no_seq_model,
                            Toast.LENGTH_LONG)
                    .show();
        }
        initModelSpinner(binding.modelSpinner, models, preferencesManager.getAutopilotSeqModel());
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

        binding.autoSwitch.setOnClickListener(v -> setNetworkEnabled(binding.autoSwitch.isChecked()));
        BotToControllerEventBus.emitEvent(
            ConnectionUtils.createStatus(
                "FRAGMENT_TYPE", Enums.FragmentType.AUTOPILOT.getFragment()));

    }

    private void updateCropImageInfo() {
        //        Timber.i("%s x %s",getPreviewSize().getWidth(), getPreviewSize().getHeight());
        //        Timber.i("%s x %s",getMaxAnalyseImageSize().getWidth(),
        //     getMaxAnalyseImageSize().getHeight());
        frameToCropTransform = null;

        sensorOrientation = 90 - ImageUtils.getScreenOrientation(requireActivity());

        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        BorderedText borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(requireContext());

        Timber.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        recreateNetwork(getModel(), getDevice(), getNumThreads());
        if (autopilot == null) {
            Timber.e("No network on preview!");
            return;
        }

        binding.trackingOverlay.addCallback(
                canvas -> {
                    if (tracker != null) {
                        tracker.draw(canvas);
                    }
                });
        tracker.setFrameConfiguration(
                getMaxAnalyseImageSize().getWidth(),
                getMaxAnalyseImageSize().getHeight(),
                sensorOrientation);
    }

    protected void onInferenceConfigurationChanged() {
        computingNetwork = false;
        final FrameHistory history = frameHistory;
        if (history != null) history.clear();
        if (cropTarget == null) {
            // Defer creation until we're getting camera frames.
            return;
        }
        final Network.Device device = getDevice();
        final Model model = getModel();
        final int numThreads = getNumThreads();
        runInBackground(() -> recreateNetwork(model, device, numThreads));
    }

    private void recreateNetwork(Model model, Network.Device device, int numThreads) {
        if (model == null) return;
        tracker.clearTrackedObjects();
        if (autopilot != null) {
            Timber.d("Closing autoPilot.");
            autopilot.close();
            autopilot = null;
        }

        try {
            Timber.d(
                    "Creating autopilot (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
            autopilot = new AutopilotSeq(requireActivity(), model, device, numThreads);
            final CropTarget target =
                    new CropTarget(autopilot.getImageSizeX(), autopilot.getImageSizeY());
            setupFrameStack(model, autopilot);
            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            getMaxAnalyseImageSize().getWidth(),
                            getMaxAnalyseImageSize().getHeight(),
                            target.bitmap.getWidth(),
                            target.bitmap.getHeight(),
                            sensorOrientation,
                            autopilot.getCropRect(),
                            autopilot.getMaintainAspect());
            // Published last: processFrame starts recording the moment this is visible, and the
            // transform above is what it crops with.
            cropTarget = target;
            requireActivity()
                    .runOnUiThread(
                            () ->
                                    binding.inputResolution.setText(
                                            String.format(
                                                    Locale.getDefault(),
                                                    "%dx%d",
                                                    autopilot.getImageSizeX(),
                                                    autopilot.getImageSizeY())));

            Matrix cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);

        } catch (IllegalArgumentException | IOException e) {
            String msg = "Failed to create network.";
            Timber.e(e, msg);
            requireActivity()
                    .runOnUiThread(
                            () ->
                                    Toast.makeText(
                                                    requireContext().getApplicationContext(),
                                                    e.getMessage(),
                                                    Toast.LENGTH_LONG)
                                            .show());
        }
    }

    /**
     * Sizes the frame history and the stack for the model that was just loaded.
     *
     * <p>How many frames the stack holds comes from the model itself. How far apart they should be
     * does not - it is a property of the data the model was trained on - so it is read from the
     * training tag in the file name when there is one, and falls back to a uniform stride
     * otherwise.
     */
    private void setupFrameStack(Model model, AutopilotSeq network) {
        final int seqLen = network.getSeqLen();
        final int[] offsets = seqOffsets(model, seqLen);

        frameAgesMs = new float[seqLen];
        for (int i = 0; i < seqLen; ++i) frameAgesMs[i] = offsets[i] * DATASET_FRAME_PERIOD_MS;

        final int numPixels = network.getImageSizeX() * network.getImageSizeY();
        frameStack = new int[seqLen][numPixels];
        // Enough room to cover the deepest offset several times over, so a camera stall or a
        // slower-than-recording frame rate still leaves something to sample.
        frameHistory = new FrameHistory(offsets[seqLen - 1] * 3 + 5, numPixels);

        Timber.d(
                "Stack of %d frames, offsets %s, spanning %.0fms",
                seqLen, Arrays.toString(offsets), frameAgesMs[seqLen - 1]);
    }

    /**
     * The frame offsets the model was trained with, newest first.
     *
     * <p>Training writes them into the model name (..._seq0-2-4-6-8_...), which is the only record
     * of them that travels with the file - the tflite itself cannot say how far apart its channels
     * were in time. A name without the tag, or one whose tag does not match the model's depth, gets
     * the uniform fallback.
     */
    private int[] seqOffsets(Model model, int seqLen) {
        // The display name is what the import dialog lets you edit, so the tag can survive in
        // either place - check the name first, then the file it actually points at.
        for (String candidate : new String[] {model.name, model.path}) {
            final int[] offsets = parseOffsets(candidate, seqLen);
            if (offsets != null) return offsets;
        }

        final int[] offsets = new int[seqLen];
        for (int i = 0; i < seqLen; ++i) offsets[i] = i * DEFAULT_SEQ_STRIDE[0];
        Timber.w(
                "No usable offsets in %s, assuming a stride of %d", model.name, DEFAULT_SEQ_STRIDE[0]);
        return offsets;
    }

    private int[] parseOffsets(String candidate, int seqLen) {
        if (candidate == null) return null;
        final Matcher matcher = Pattern.compile("_seq(\\d+(?:-\\d+)*)").matcher(candidate);
        if (!matcher.find()) return null;

        final String[] parts = matcher.group(1).split("-");
        if (parts.length != seqLen) {
            Timber.w(
                    "%s declares %d offsets but the model takes %d frames - ignoring the tag",
                    candidate, parts.length, seqLen);
            return null;
        }
        final int[] offsets = new int[seqLen];
        for (int i = 0; i < seqLen; ++i) offsets[i] = Integer.parseInt(parts[i]);
        return offsets;
    }

    @Override
    public synchronized void onResume() {
        cropTarget = null;
        tracker = null;
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

        }
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
        }
    }

    private long frameNum = 0;

    @Override
    protected void processFrame(Bitmap bitmap, ImageProxy image) {
        if (tracker == null) updateCropImageInfo();

        ++frameNum;

        // Snapshot what the network swap touches: a model change can replace all of it from
        // another thread while this frame is in flight, and a stack half from each model would be
        // silently wrong rather than an error.
        final AutopilotSeq network = autopilot;
        final FrameHistory history = frameHistory;
        final CropTarget target = cropTarget;
        final int[][] stack = frameStack;
        final float[] agesMs = frameAgesMs;
        if (network == null
                || history == null
                || target == null
                || stack == null
                || agesMs == null
                || frameToCropTransform == null) {
            return;
        }

        // Crop and record every preview frame, on this thread, whether or not the network is
        // running. The bitmap handed in here is a single buffer the camera overwrites on the next
        // frame, so it cannot be read later from the inference thread; and recording only the
        // frames inference happened to run on would space the stack by inference latency instead of
        // by the interval the model was trained with. Keeping the history warm also means flipping
        // the switch starts on a full stack rather than on five copies of one frame.
        target.canvas.drawBitmap(bitmap, frameToCropTransform, null);
        history.add(SystemClock.elapsedRealtime(), target.bitmap);

        if (binding != null && binding.autoSwitch.isChecked()) {
            // If network is busy, return.
            if (computingNetwork) {
                return;
            }

            computingNetwork = true;
            Timber.i("Putting image " + frameNum + " for detection in bg thread.");

            runInBackground(
                    () -> {
                        final long span = history.fill(agesMs, stack);
                        if (span >= 0) {
                            Timber.i("Running autopilot on image %s", frameNum);
                            final long startTime = SystemClock.elapsedRealtime();
                            handleDriveCommand(network.recognizeImage(stack, vehicle.getIndicator()));
                            lastProcessingTimeMs = SystemClock.elapsedRealtime() - startTime;
                            lastStackSpanMs = span;
                        }

                        computingNetwork = false;
                    });
            if (lastProcessingTimeMs > 0)
                requireActivity()
                        .runOnUiThread(
                                () ->
                                        binding.inferenceInfo.setText(
                                                String.format(
                                                        Locale.US,
                                                        "%d fps %dms",
                                                        1000 / lastProcessingTimeMs,
                                                        lastStackSpanMs)));
        }
    }

    protected void handleDriveCommand(Control control) {
        vehicle.setControl(control);
        float steering = vehicle.getSteering();
        float throttle = vehicle.getThrottle();
        requireActivity()
                .runOnUiThread(
                        () ->
                                binding.controllerContainer.controlInfo.setText(
                                        String.format(Locale.US, "%.0f,%.0f", steering, throttle)));
    }

    @Override
    public void onConnectionEstablished(String ipAddress) {
        requireActivity().runOnUiThread(() -> binding.ipAddress.setText(ipAddress));
    }

    protected Model getModel() {
        return model;
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

    protected void setModel(Model model) {
        if (this.model != model) {
            Timber.d("Updating  model: %s", model);
            this.model = model;
            preferencesManager.setAutopilotSeqModel(model.name);
            onInferenceConfigurationChanged();
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
