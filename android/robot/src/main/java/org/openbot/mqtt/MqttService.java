package org.openbot.mqtt; // Your package name

import android.content.Context;
import android.content.Intent;
import android.util.Log;

import info.mqtt.android.service.MqttAndroidClient;
import info.mqtt.android.service.Ack;

import org.eclipse.paho.client.mqttv3.IMqttActionListener;
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.IMqttToken;
import org.eclipse.paho.client.mqttv3.MqttCallbackExtended;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttMessage;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import java.nio.charset.StandardCharsets;

public class MqttService {

    private static final String TAG = "MqttService";
    private MqttAndroidClient client;
    private LocalBroadcastManager broadcaster;
    private volatile boolean isConnected = false;

    public static final String MQTT_LOG_EVENT = "mqtt-log-event";
    public static final String MQTT_LOG_MESSAGE = "mqtt-log-message";

    public MqttService(Context context, String brokerUrl, String clientId, String topic) {
        client = new MqttAndroidClient(context, brokerUrl, clientId, Ack.AUTO_ACK);
        broadcaster = LocalBroadcastManager.getInstance(context);

        client.setCallback(new MqttCallbackExtended() {
            @Override
            public void connectComplete(boolean reconnect, String serverURI) {
                String message = "Connection to broker complete.";
                isConnected = true;
                Log.d(TAG, message);
                sendUpdate(message);
                // After connecting, we can now subscribe automatically
                subscribe(topic);
            }

            @Override
            public void connectionLost(Throwable cause) {
                String message = "Connection was lost.";
                isConnected = false;
                Log.d(TAG, message);
                sendUpdate(message);
            }

            @Override
            public void messageArrived(String topic, MqttMessage message) throws Exception {
                String payload = new String(message.getPayload(), StandardCharsets.UTF_8);
                String logMessage = "📩 MESSAGE RECEIVED on topic '" + topic + "': " + payload;
//                Log.d(TAG, logMessage);
                sendUpdate(logMessage);
            }

            @Override
            public void deliveryComplete(IMqttDeliveryToken token) {
                // Not critical for this test
            }
        });
    }

    public void connect() {
        MqttConnectOptions options = new MqttConnectOptions();
        options.setAutomaticReconnect(true);
        options.setCleanSession(true);

        client.connect(options, null, new IMqttActionListener() {
            @Override
            public void onSuccess(IMqttToken asyncActionToken) {
                Log.d(TAG, "Connection Success!");
            }

            @Override
            public void onFailure(IMqttToken asyncActionToken, Throwable exception) {
                isConnected = false;
                Log.e(TAG, "Connection Failure!", exception);
            }
        });
    }

    public void disconnect() {

        client.disconnect();
        isConnected = false;
    }

    public void publish(String topic, String payload) {
        try {
            MqttMessage message = new MqttMessage();
            message.setPayload(payload.getBytes());
            client.publish(topic, message);
            String logMessage = "Published message to topic '" + topic + "': " + payload;
            Log.d(TAG, logMessage);
            sendUpdate(logMessage);
        } catch (IllegalAccessError e) {
            Log.e(TAG, "Publish failed!", e);
            sendUpdate("Publish failed: " + e.getMessage());
        }

    }

    public void subscribe(String topic) {
        // No MqttException is thrown by subscribe in the new library.
        client.subscribe(topic, 0, null, new IMqttActionListener() {
            @Override
            public void onSuccess(IMqttToken asyncActionToken) {
                String message = "Subscribed to " + topic;
                Log.d(TAG, message);
                sendUpdate(message);
            }

            @Override
            public void onFailure(IMqttToken asyncActionToken, Throwable exception) {
                String message = "Subscription to " + topic + " failed!";
                Log.e(TAG, message, exception);
                sendUpdate(message);
            }
        });
    }

    private void sendUpdate(String message) {
        Intent intent = new Intent(MQTT_LOG_EVENT);
        intent.putExtra(MQTT_LOG_MESSAGE, message);
        broadcaster.sendBroadcast(intent);
    }

    public boolean isConnected() {
        return isConnected;
    }
}