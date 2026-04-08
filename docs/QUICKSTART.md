# Quick Start Guide - Q-Engage Lite

## 📥 Installation Steps

### 1. Install the package and dependencies

```bash
# From the project root directory
pip install -e .
```

This will install all required dependencies:
- paho-mqtt (MQTT communication)
- ultralytics (YOLOv8 AI model)
- opencv-python (Computer vision)
- numpy (Numerical computing)
- pillow (Image processing)

### 2. Verify installation

```bash
# Check if commands are available
q-engage-tracker --help || echo "Tracker installed"
q-engage-service --help || echo "Service installed"
```

### 3. Download YOLO model (first run)

The tracker module will automatically download the YOLOv8 model on first run.

## 🎮 Running the modules

### Option 1: Using installed commands

```bash
# Run tracker module
q-engage-tracker

# Run service module (in another terminal)
q-engage-service
```

### Option 2: Using Python modules

```bash
# Run tracker module
python -m src.tracker

# Run service module (in another terminal)
python -m src.service
```

## 🧪 Testing MQTT Service

### Install and run Mosquitto MQTT broker (optional)

```bash
# Ubuntu/Debian
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto

# Test MQTT connection
mosquitto_pub -h localhost -t "qengage/test" -m "Hello Q-Engage"
```

## 🔧 Configuration

Edit `settings.default.json` to configure your camera and application settings.

## 📊 Module Description

### Tracker Module (`src/tracker`)
- AI-powered edge detection
- Object tracking using YOLOv8
- Processes video streams from cameras
- Publishes detection events via MQTT

### Service Module (`src/service`)
- MQTT client for real-time communication
- Subscribes to tracker events
- Handles command messages
- Manages application state

## 🐛 Troubleshooting

### Import errors after installation
If you see import errors, activate your virtual environment or reinstall:
```bash
pip install -e . --force-reinstall
```

### MQTT connection issues
Check that your MQTT broker is running:
```bash
sudo systemctl status mosquitto
```

### Camera/RTSP issues
Update the `rtsp_url` in `settings.default.json` with your camera's stream URL.
