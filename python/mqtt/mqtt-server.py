import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
# Use the same broker as your Android app
BROKER_ADDRESS = "test.mosquitto.org"
BROKER_PORT = 1883
# Subscribe to the topic your app uses
ROBOT_TOPIC = "robot/control"
# We also subscribe to 'test/#' to see all other test messages
WILDCARD_TOPIC = "test/#"

# --- MQTT CALLBACKS ---

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[Monitor] Connected successfully to {BROKER_ADDRESS}")
        # Subscribe to the topics
        client.subscribe(ROBOT_TOPIC)
        client.subscribe(WILDCARD_TOPIC)
        print(f"[Monitor] Subscribed to '{ROBOT_TOPIC}'")
        print(f"[Monitor] Subscribed to '{WILDCARD_TOPIC}'")
    else:
        print(f"Failed to connect, return code {rc}")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    # Decode the message payload from bytes to a string
    payload_str = msg.payload.decode('utf-8')
    print(f"  [Message Received] Topic: '{msg.topic}' | Payload: '{payload_str}'")

# --- MAIN SCRIPT ---

print("Starting MQTT Monitor...")

# Create an MQTT client instance
client = mqtt.Client(client_id="my_test_monitor_12345") # Use a unique client ID

# Assign callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker
try:
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
except Exception as e:
    print(f"Could not connect to broker: {e}")
    exit()

# Start the network loop. This is a blocking call.
# It will keep the script running and processing incoming messages.
try:
    print(f"Waiting for messages... Press CTRL+C to exit.")
    client.loop_forever()
except KeyboardInterrupt:
    print("\n[Monitor] Disconnecting...")
    client.disconnect()
    print("[Monitor] Exited.")
