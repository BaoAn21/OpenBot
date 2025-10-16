// File: org/openbot/depthDetection/ImageUtils.java

package org.openbot.depthDetection;

import android.graphics.Bitmap;
import android.graphics.Color;

public class ImageUtils {
    public static Bitmap toGrayscaleBitmap(float[] floatArray, int imageDim) {
        Bitmap bitmap = Bitmap.createBitmap(imageDim, imageDim, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[imageDim * imageDim];
        for (int i = 0; i < pixels.length; i++) {
            int gray = (int) floatArray[i];
            pixels[i] = Color.rgb(gray, gray, gray);
        }
        bitmap.setPixels(pixels, 0, imageDim, 0, 0, imageDim, imageDim);
        return bitmap;
    }

    public static Bitmap toHeatMapBitmap(float[] floatArray, int imageDim) {
        Bitmap bitmap = Bitmap.createBitmap(imageDim, imageDim, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[imageDim * imageDim];
        for (int i = 0; i < pixels.length; i++) {
            int c = (int) floatArray[i];
            pixels[i] = Color.rgb(r(c), g(c), b(c));
        }
        bitmap.setPixels(pixels, 0, imageDim, 0, 0, imageDim, imageDim);
        return bitmap;
    }

    private static int r(int v) { return v > 127 ? (v < 192 ? (v - 127) * 4 : 255) : 0; }
    private static int g(int v) { return v < 64 ? v * 4 : (v > 192 ? 255 - (v - 192) * 4 : 255); }
    private static int b(int v) { return v < 64 ? 255 : (v < 127 ? 255 - (v - 64) * 4 : 0); }
}