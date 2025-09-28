package org.openbot.objectbox;

import android.content.Context;


import org.openbot.MyObjectBox;

import io.objectbox.BoxStore;

public class ObjectBox {
    private static BoxStore store;

    public static void init(Context context) {
        store = MyObjectBox.builder()
                .androidContext(context)
                .build();
    }

    public static BoxStore get() { return store; }
}
