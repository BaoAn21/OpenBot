package org.openbot.depthDetection;

import android.util.Log;

import androidx.annotation.NonNull;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class NetworkUtils {
    public static void sendDepthData(float[] depthValues) {
        // 1. Define your laptop's server URL
        // <-- REPLACE WITH YOUR LAPTOP'S IP
        final String url = "http://192.168.50.141:8080/upload_depth";

        // 2. Convert the FloatArray to a ByteArray
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        try {
            for (float f : depthValues) {
                dos.writeFloat(f);
            }
        } catch (IOException e) {
            Log.e("NetworkUtils", "Error converting float array", e);
            return; // Can't proceed if conversion fails
        }
        byte[] data = bos.toByteArray();

        // 3. Create the request body
        // "application/octet-stream" is the standard for raw binary data
        RequestBody requestBody = RequestBody.create(
                data,
                MediaType.get("application/octet-stream")
        );

        // 4. Build the POST request
        Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .build();

        // 5. Send the request (asynchronously)
        // Note: OkHttpClient() should ideally be a singleton,
        // but this is fine for a quick test.
        OkHttpClient client = new OkHttpClient();
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                // Handle failure (e.g., server is down, wrong IP)
                Log.e("Network", "Failed to send depth data", e);
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                try {
                    // Check if the server accepted the data
                    if (response.isSuccessful()) {
                        String responseBody = (response.body() != null) ? response.body().string() : "empty";
                        Log.d("Network", "Depth data sent! " + responseBody);
                    } else {
                        Log.w("Network", "Server responded with error: " + response.code());
                    }
                } finally {
                    response.close(); // Always close the response body
                }
            }
        });

    }
}
