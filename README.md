# Q-Engage Lite

A lightweight version of Q-Engage for small to medium-sized businesses with AI-powered edge detection and MQTT service communication.

## 🚀 Features

- **AI-Powered Tracker**: Edge detection and object tracking using YOLOv8 (Ultralytics)
- **MQTT Service**: Real-time event communication via MQTT protocol
- **Modular Architecture**: Two independent modules that can run separately or together
- **Camera Support**: RTSP stream support with motion detection and night vision capabilities

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip and setuptools

### Install from source

```bash
# Clone or navigate to the project directory
cd /home/fernando/projects/q-engage-lite

# Install in development mode (editable)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## 🎯 Usage

After installation, you can run the modules as commands:

### Tracker Module (AI Edge Detection)
```bash
q-engage-tracker
```

### Service Module (MQTT Communication)
```bash
q-engage-service
```

### Run as Python modules
Alternatively, you can run them directly as Python modules:

```bash
# Tracker module
python -m src.tracker

# Service module
python -m src.service
```

## 🛠️ Configuration

Edit `settings.default.json` to configure:

- Application metadata
- Camera settings (resolution, frame rate, RTSP URL)
- Motion detection and night vision options

```json
{
    "camera": {
        "resolution": "1080p",
        "frame_rate": 30,
        "night_vision": true,
        "motion_detection": true,
        "rtsp_url": "rtsp://your-camera-url"
    }
}
```

## 📚 Dependencies

- **paho-mqtt** (>=2.0.0): MQTT protocol implementation
- **ultralytics** (>=8.0.0): YOLOv8 for AI object detection
- **opencv-python** (>=4.8.0): Computer vision and video processing
- **numpy** (>=1.24.0): Numerical computing
- **pillow** (>=10.0.0): Image processing

## 🏗️ Project Structure

```
q-engage-lite/
├── src/
│   ├── tracker/          # AI edge detection module
│   │   └── __main__.py
│   └── service/          # MQTT service module
│       └── __main__.py
├── db/                   # Database files
├── tmp/frames/           # Temporary frame storage
├── pyproject.toml        # Project configuration
├── settings.default.json # Default settings
├── README.md
└── VERSION
```

## 🧪 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/

# Linting
flake8 src/

# Type checking
mypy src/
```

## 📄 License

MIT License - See LICENSE file for details

## 👥 Author

**Quallity Solutions**

## 📌 Version

Current version: **1.0.0**
