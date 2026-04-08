"""
Q-Engage Lite - Service Module
MQTT service for communication and event handling
"""
import sys
import json
import paho.mqtt.client as mqtt
from pathlib import Path


def load_settings():
    """Load settings from settings.default.json"""
    settings_path = Path(__file__).parent.parent / "settings.default.json"
    try:
        with open(settings_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        return {}


def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        print("✓ Connected to MQTT broker successfully")
        client.subscribe("qengage/tracker/#")
        client.subscribe("qengage/commands/#")
    else:
        print(f"✗ Failed to connect, return code {rc}")


def on_message(client, userdata, msg):
    """Callback when message received"""
    print(f"Message received on {msg.topic}: {msg.payload.decode()}")
    
    # TODO: Process different message types
    # TODO: Handle tracker events
    # TODO: Handle command messages


def on_disconnect(client, userdata, rc, properties=None):
    """Callback when disconnected from MQTT broker"""
    if rc != 0:
        print(f"Unexpected disconnection. Code: {rc}")


def main():
    """
    Main entry point for the service module.
    Initializes MQTT client and handles communication.
    """
    print("Q-Engage Lite - Service Module v1.0.0")
    print("Initializing MQTT Service...")
    
    settings = load_settings()
    
    try:
        # Create MQTT client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="q-engage-lite-service")
        
        # Set callbacks
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        # TODO: Get broker details from settings
        broker = "localhost"  # Default broker
        port = 1883
        
        print(f"Connecting to MQTT broker at {broker}:{port}...")
        client.connect(broker, port, 60)
        
        print("Service module ready. Listening for events...")
        print("Press Ctrl+C to stop")
        
        # Start network loop
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\nShutting down service module...")
        if 'client' in locals():
            client.disconnect()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
