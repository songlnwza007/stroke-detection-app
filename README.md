# Stroke Detection App

An Android application built with Kivy that combines facial stroke detection and hand gesture tracking for rehabilitation exercises.

## Features

### 1. Facial Stroke Detection

- Analyzes facial asymmetry using MediaPipe Face Mesh
- Detects mouth asymmetry (primary stroke indicator)
- Detects eye drooping (secondary indicator)
- Real-time camera-based analysis

### 2. Hand Gesture Detection

- Tracks hand gestures (open hand vs fist)
- Counts gesture transitions for rehabilitation exercises
- Timed exercise mode with phases for left/right/both hands
- Real-time feedback and counting

## Project Structure

```
stroke_detection_app/
├── main.py                    # Kivy app entry point
├── screens.py                 # UI screen definitions
├── buildozer.spec             # Android build configuration
├── requirements.txt           # Python dependencies
├── detectors/
│   ├── __init__.py
│   ├── facial_detector.py     # Stroke detection module
│   └── hand_detector.py       # Hand gesture module
└── utils/
    ├── __init__.py
    └── camera_manager.py      # Camera handling
```

## Installation & Testing (Desktop)

### Prerequisites

- Python 3.8+
- Webcam/Camera

### Setup

```bash
# Navigate to the app directory
cd stroke_detection_app

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

## Building Android APK

### Prerequisites

- Linux or WSL (Windows Subsystem for Linux)
- Python 3.8+
- Java JDK 11
- Android SDK and NDK (auto-downloaded by Buildozer)

### Build Steps

1. **Install Buildozer** (on Linux/WSL):

```bash
pip install buildozer
pip install cython

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y \
    python3-pip \
    build-essential \
    git \
    python3 \
    python3-dev \
    ffmpeg \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev \
    libgstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    openjdk-11-jdk \
    unzip \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libffi-dev \
    cmake
```

2. **Build Debug APK**:

```bash
cd stroke_detection_app
buildozer android debug
```

3. **Deploy to device**:

```bash
buildozer android deploy run logcat
```

The APK will be created in the `bin/` folder.

## Usage

### Facial Stroke Detection

1. Open the app and tap "Facial Stroke Detection"
2. Tap "Start Detection" to begin
3. Face the camera directly
4. The app will analyze your facial symmetry and display:
   - Mouth Symmetry percentage
   - Eye Droop percentage
   - STROKE or NORMAL status

### Hand Gesture Detection

1. Open the app and tap "Hand Gesture Detection"
2. Tap "Start Camera" to enable the camera
3. Tap "Start Exercise" to begin the timed exercise
4. Follow the on-screen instructions (show left/right/both hands)
5. Make fist gestures to increment the counters
6. Use "Reset" to clear counts

## Important Notes

- **Medical Disclaimer**: This app is for screening purposes only and should not be used as a substitute for professional medical diagnosis.
- **Camera Permission**: The app requires camera access to function.
- **Performance**: For best results, ensure good lighting and face the camera directly.

## Troubleshooting

### MediaPipe Issues on Android

If MediaPipe doesn't work properly on Android:

1. Try using a lower resolution (modify `CameraManager`)
2. Consider using TensorFlow Lite models instead
3. Check Buildozer logs for specific errors

### Camera Not Working

1. Ensure camera permissions are granted
2. Check if another app is using the camera
3. Try restarting the app

## License

This project is for educational and medical screening purposes.
