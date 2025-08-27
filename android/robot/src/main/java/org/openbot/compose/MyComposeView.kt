package org.openbot.compose

import android.os.Bundle
import android.os.PersistableBundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.padding
import androidx.compose.material.Text
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

//@Composable
//fun Greeting(name: String) {
//    Text(
//        text = "Hello $name from Compose",
//        color = Color.Blue,
//        modifier = Modifier.padding(16.dp)
//    )
//}

class MyComposeView : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Button(onClick = { finish() }) {
                    Text("hello, tap to go back.")
                }
            }
        }
    }
}