package org.openbot.depthDetection;

import android.util.Log;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.json.JSONArray; // Using JSON for easier handling on server

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
// okio.ByteString is not needed if sending text (JSON)

public class NetworkUtils {

    private static final String TAG = "NetworkUtils";
    // <-- REPLACE WITH YOUR LAPTOP'S IP and the WebSocket port
    private static final String SERVER_URL = "ws://10.253.175.225:8765";
    private static final int NORMAL_CLOSURE_STATUS = 1000;

    private static OkHttpClient client;
    private static WebSocket webSocket;
    private static OkHttpWebSocketListener listener;

    // --- Initialize the client and listener (call once) ---
    public static synchronized void initializeWebSocket() {
        if (client == null) {
            client = new OkHttpClient.Builder()
                    .readTimeout(0, TimeUnit.MILLISECONDS) // Keep connection alive
                    .retryOnConnectionFailure(true) // Attempt auto-reconnect on network issues
                    .pingInterval(30, TimeUnit.SECONDS) // Keep connection alive
                    .build();
            listener = new OkHttpWebSocketListener();
            Log.i(TAG, "OkHttpClient initialized for WebSocket.");
        }
    }

    // --- Connect to the server ---
    public static synchronized void connectWebSocket() {
        if (client == null) {
            Log.e(TAG, "Client not initialized. Call initializeWebSocket() first.");
            return;
        }
        if (webSocket != null) {
            Log.w(TAG, "WebSocket already connected or attempting connection.");
            return; // Avoid multiple connections
        }

        Request request = new Request.Builder()
                .url(SERVER_URL)
                .build();
        Log.i(TAG, "Attempting to connect WebSocket to: " + SERVER_URL);
        client.newWebSocket(request, listener); // Asynchronous connection
    }

    // --- Send depth data ---
    public static void sendDepthData(float[] depthValues) {
        if (webSocket == null) {
            // Log.w(TAG, "WebSocket not connected, cannot send data."); // Can be noisy
            return;
        }
        if (depthValues == null || depthValues.length == 0) {
            Log.w(TAG, "Depth data array is null or empty, not sending.");
            return;
        }

        try {
            // Convert float array to JSON array string
            JSONArray jsonArray = new JSONArray();
            for (float value : depthValues) {
                if (Float.isFinite(value)) {
                    jsonArray.put(value);
                } else {
                    jsonArray.put(0.0); // Replace NaN/Infinity with 0
                }
            }
            String message = jsonArray.toString();
            // Log.d(TAG, "Sending depth data via WebSocket (first 100 chars): " + message.substring(0, Math.min(message.length(), 100)));

            // Send the JSON string as a text message
            boolean success = webSocket.send(message);
            if (!success) {
                Log.w(TAG, "WebSocket send queue is full or closing. Message dropped.");
                // Handle buffer full state if needed (e.g., maybe close/reconnect)
            }
        } catch (Exception e) { // Catch potential JSONException or WebSocketException
            Log.e(TAG, "Error sending depth data via WebSocket", e);
        }
    }

    // --- Close the connection ---
    public static synchronized void closeWebSocket() {
        if (webSocket != null) {
            Log.i(TAG, "Closing WebSocket connection.");
            webSocket.close(NORMAL_CLOSURE_STATUS, "Client shutting down");
            webSocket = null; // Nullify immediately after requesting close
        }
        // Optional: Clean up the OkHttpClient if no longer needed
        // if (client != null) {
        //    client.dispatcher().executorService().shutdown();
        //    client.connectionPool().evictAll();
        //    client = null;
        //    Log.i(TAG, "OkHttpClient shut down.");
        // }
        listener = null; // Allow listener to be GC'd
    }

    // --- Get connection status ---
    public static boolean isConnected() {
        return webSocket != null; // Simple check if the instance exists (set in onOpen, cleared in onClose/onFailure)
    }

    // --- Listener class (can be static inner or separate) ---
    private static class OkHttpWebSocketListener extends WebSocketListener {
        @Override
        public void onOpen(@NonNull WebSocket ws, @NonNull Response response) {
            Log.i(TAG, "WebSocket connection opened!");
            // Store the WebSocket instance globally within NetworkUtils
            NetworkUtils.webSocket = ws;
            // Optionally notify UI or other components
        }

        @Override
        public void onMessage(@NonNull WebSocket ws, @NonNull String text) {
            Log.i(TAG, "WebSocket received message: " + text);
            // Handle messages from the server if needed
        }

        // onMessage with ByteString can be added if binary data is expected

        @Override
        public void onClosing(@NonNull WebSocket ws, int code, @NonNull String reason) {
            Log.w(TAG, "WebSocket closing: " + code + " / " + reason);
            // Request graceful closure from our side
            ws.close(NORMAL_CLOSURE_STATUS, null);
            // Nullify the global instance
            NetworkUtils.webSocket = null;
        }

        @Override
        public void onClosed(@NonNull WebSocket ws, int code, @NonNull String reason) {
            Log.w(TAG, "WebSocket closed: " + code + " / " + reason);
            // Ensure the global instance is nullified
            NetworkUtils.webSocket = null;
            // Optionally notify UI or trigger reconnect logic
        }

        @Override
        public void onFailure(@NonNull WebSocket ws, @NonNull Throwable t, @Nullable Response response) {
            Log.e(TAG, "WebSocket connection failure: " + t.getMessage(), t);
            // Nullify the global instance on failure
            NetworkUtils.webSocket = null;
            // Optionally notify UI or trigger reconnect logic (with backoff)
            // Consider calling connectWebSocket() again after a delay
        }
    }
}