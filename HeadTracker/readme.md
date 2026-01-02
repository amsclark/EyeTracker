# HEAD TRACKING MOUSE CONTROL

This script uses your webcam and MediaPipe's Face Landmarker to estimate your head pose and control the mouse cursor using head orientation.

To help support this software and other open-source projects, please consider subscribing to my YouTube channel: https://www.youtube.com/@jeoresearch, or joining for $1 per month: https://www.youtube.com/@jeoresearch/join. 

The webcam used for tracking can be found here ($35): https://amzn.to/43of401

Output should look like this video: https://youtu.be/hImmJDTgXjw

## REQUIREMENTS

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## INSTALLATION

### Linux (Ubuntu/Debian)

1. **Create and activate a virtual environment:**
```bash
python3 -m venv eyetracker
cd eyetracker
source bin/activate
```

2. **Install required packages:**
```bash
pip install opencv-python mediapipe numpy pyautogui
```

3. **Download the MediaPipe face model:**
```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

4. **Optional - Install PyQt5 for cursor visualization:**
```bash
pip install PyQt5
```

### Windows/Mac

Follow similar steps, creating a virtual environment and installing the same packages. The MediaPipe model will need to be downloaded to the same directory as the script.

## USAGE

1. Ensure a webcam is connected. The script defaults to camera index 0 (built-in camera). Change to 1 if needed.

2. **Activate the virtual environment (Linux):**
```bash
source bin/activate
```

3. **Run the main tracking script:**
```bash
python3 MonitorTracking.py
```

4. **Optional - Run cursor visualization in a separate terminal:**
```bash
python3 CursorCircle.py
```

5. Two windows will appear:
    - "Head-Aligned Cube": shows your face with overlayed bounding box and tracking vector.
    - "Facial Landmarks": shows the detected facial landmarks.

## FEATURES

### Control Modes

**Velocity/Joystick Mode (Default - Recommended)**
- Head tilt controls cursor **speed**, not position (like Eviacam)
- Tilt head slightly → cursor moves in that direction
- Return to center → cursor stops
- Much more stable and ergonomic for extended use
- Ideal for users with limited mobility

**Position Mode (Legacy)**
- Head angle maps directly to screen position
- Requires holding head at specific angles
- Set `USE_VELOCITY_MODE = False` in the script to enable

### Mouse Click Detection

**Mouth Click**
- Open your mouth to trigger a mouse click
- Configurable threshold and cooldown
- Set `ENABLE_MOUTH_CLICK = True/False` to toggle
- Adjust `MOUTH_OPEN_THRESHOLD` for sensitivity

### Stabilization

- Kalman filtering for smooth tracking
- Motion threshold to ignore micro-movements
- Exponential smoothing to reduce jitter
- Configurable dead zone for easier cursor stopping

### Keyboard Controls

- **t** - Toggle mouse control on/off
- **c** - Calibrate (sets current head position as neutral/center)
- **q** - Quit the application

## CONFIGURATION

Edit these parameters at the top of `MonitorTracking.py`:

```python
# Velocity mode settings
USE_VELOCITY_MODE = True      # Enable joystick-style control
MAX_VELOCITY = 8              # Maximum cursor speed (pixels/frame)
DEAD_ZONE = 5                 # Degrees of no-movement zone
VELOCITY_SCALE = 0.5          # Overall sensitivity multiplier

# Click detection
ENABLE_MOUTH_CLICK = True     # Enable mouth-open clicking
MOUTH_OPEN_THRESHOLD = 0.03   # Sensitivity (lower = easier to trigger)
CLICK_COOLDOWN = 0.5          # Seconds between clicks

# Stabilization
MOTION_THRESHOLD = 5          # Ignore movements below this (pixels)
SMOOTHING_ALPHA = 0.15        # Smoothing weight (lower = smoother)
filter_length = 8             # Averaging buffer size
```

## NOTES

- **Calibration** centers the tracking to your current head pose - do this while looking straight ahead
- **Velocity mode** is inspired by accessibility software like Eviacam, designed for users with limited mobility
- The program calculates yaw and pitch from facial landmarks using MediaPipe Face Landmarker
- Movement is smoothed using multiple techniques: averaging, Kalman filtering, and exponential smoothing

## TROUBLESHOOTING

**Script crashes on start:**
- Make sure you're in the virtual environment: `source bin/activate`
- Verify the `face_landmarker.task` model file is in the same directory

**Tracking is too sensitive/fast:**
- Decrease `MAX_VELOCITY` and `VELOCITY_SCALE`
- Increase `DEAD_ZONE` for easier stopping

**Tracking is jittery:**
- Increase `filter_length` for more averaging
- Decrease `SMOOTHING_ALPHA` for heavier smoothing
- Increase `MOTION_THRESHOLD` to ignore small movements

**Wrong camera:**
- Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or another index

**Cursor doesn't stop easily:**
- Increase `DEAD_ZONE` (try 7-10 degrees)
- Make sure you've calibrated with 'c' key

**Mouth clicks too sensitive:**
- Increase `MOUTH_OPEN_THRESHOLD` (try 0.04-0.05)
- Increase `CLICK_COOLDOWN` to prevent double-clicks
