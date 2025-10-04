package org.openbot.mqtt;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import org.openbot.OpenBotApplication;
import org.openbot.R;

public class MqttLogFragment extends Fragment {

    private TextView logTextView;
    private EditText brokerUrlEditText;
    private Button connectButton;

    private EditText topicEditText;
    private EditText messageEditText;
    private Button sendButton;
    private BroadcastReceiver logReceiver;


    private MqttService mqttService;

    private static final String KEY_BROKER_URL = "broker_url_edittext";

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_mqtt_log, container, false);
        logTextView = view.findViewById(R.id.mqtt_log_textview);

        brokerUrlEditText = view.findViewById(R.id.broker_url_edittext);
        connectButton = view.findViewById(R.id.connect_button);
        topicEditText = view.findViewById(R.id.topic_edittext);
        messageEditText = view.findViewById(R.id.message_edittext);
        sendButton = view.findViewById(R.id.send_button);

        connectButton.setOnClickListener(v -> handleConnect());
        sendButton.setOnClickListener(v -> handleSendMessage());

        logReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (intent.hasExtra(MqttService.MQTT_LOG_MESSAGE)) {
                    String message = intent.getStringExtra(MqttService.MQTT_LOG_MESSAGE);
                    appendLog(message);
                }
            }
        };

        return view;
    }

    @Override
    public void onResume() {
        super.onResume();
        LocalBroadcastManager.getInstance(requireContext()).registerReceiver(logReceiver, new IntentFilter(MqttService.MQTT_LOG_EVENT));
    }

    @Override
    public void onPause() {
        super.onPause();
        LocalBroadcastManager.getInstance(requireContext()).unregisterReceiver(logReceiver);
    }

    public void appendLog(String message) {
        if (logTextView != null) {
            logTextView.append(message + "\n");
        }
    }

    private void handleConnect() {
        if (OpenBotApplication.mqttService != null) {
            appendLog("Creating new MQTT connection...");
        }

        String brokerUrl = "tcp://" + brokerUrlEditText.getText().toString().trim() + ":1883";
        String clientId = "OpenBot_" + System.currentTimeMillis();

        OpenBotApplication.mqttService = new MqttService(requireContext(), brokerUrl, clientId);
        OpenBotApplication.mqttService.connect();
    }

    private void handleSendMessage() {
        // First, check if we are even connected
        if (OpenBotApplication.mqttService == null) {
            Toast.makeText(getContext(), "Not connected to a broker", Toast.LENGTH_SHORT).show();
            return;
        }

        String topic = topicEditText.getText().toString().trim();
        String message = messageEditText.getText().toString().trim();

        // Validate that the fields are not empty
        if (TextUtils.isEmpty(topic) || TextUtils.isEmpty(message)) {
            Toast.makeText(getContext(), "Topic and message cannot be empty", Toast.LENGTH_SHORT).show();
            return;
        }

        // Use the shared MqttService instance to publish the message
        OpenBotApplication.mqttService.publish(topic, message);

        // Clear the message field for convenience after sending
        messageEditText.setText("");
    }
}
