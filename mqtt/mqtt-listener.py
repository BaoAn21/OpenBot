import paho.mqtt.client as mqtt
import time

# IMPORTANT: Use the same IP address as your Android app
BROKER_ADDRESS = "172.28.182.95" # e.g., "192.168.1.15"
TOPIC = "openbot/data"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Listener connected to local broker.")
        client.subscribe(TOPIC)
        print(f"👂 Listening for messages on topic: '{TOPIC}'")
    else:
        print(f"❌ Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    # Added a prefix to easily distinguish received messages
    print(f"\n[RECEIVED] <-- '{msg.payload.decode()}'")
    # This line is to re-prompt the user after a message is received
    print("Enter a message to send (or type 'exit'): ", end='', flush=True)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_ADDRESS, 1883, 60)

# 1. loop_start() runs the client in a background thread.
# This is crucial for allowing the main script to accept user input.
client.loop_start()

print("MQTT Client is running. Type a message and press Enter to publish.")
print("Type 'exit' to quit.")

try:
    # 2. This main loop now continuously waits for your input.
    while True:
        message = input("Enter a message to send (or type 'exit'): ")
        if message.lower() == 'exit':
            print("Exiting...")
            break
        
        # 3. Publish the message you typed.
        client.publish(TOPIC, message)
        print(f"[SENT] --> '{message}'")

finally:
    # 4. Clean up gracefully when the loop is broken.
    print("Stopping MQTT client.")
    client.loop_stop()
    client.disconnect()
    

