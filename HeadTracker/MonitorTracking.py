import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import math
import threading
import time
import subprocess

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = True
filter_length = 8

# Joystick/velocity mode parameters
USE_VELOCITY_MODE = True  # Set to False for old position mode
MAX_VELOCITY = 9  # Maximum cursor speed in pixels per frame
DEAD_ZONE = 7  # Degrees - no movement within this range from center
VELOCITY_SCALE = 0.7  # Sensitivity multiplier

# Click detection parameters
ENABLE_MOUTH_CLICK = True  # Click when mouth opens
MOUTH_OPEN_THRESHOLD = 0.03  # Ratio of mouth opening to face height
CLICK_COOLDOWN = 1  # Seconds between clicks
last_click_time = 0

# Stabilization parameters (for velocity mode)
MOTION_THRESHOLD = 5  # pixels - ignore small movements
SMOOTHING_ALPHA = 0.15  # weight for exponential smoothing (lower = smoother)


FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

calibration_offset_yaw = 0
calibration_offset_pitch = 0

# Current cursor position for velocity mode
current_cursor_x = CENTER_X
current_cursor_y = CENTER_Y

# Buffers to store recent ray data
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)

# Initialize Kalman filter for cursor position
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Previous cursor position for stabilization
previous_cursor_position = (CENTER_X, CENTER_Y)

# Initialize MediaPipe Face Landmarker (new API)
base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)
face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

# Open camera
cap = cv2.VideoCapture(0)  # Try camera 0 first (default)
frame_counter = 0

# Landmark indices for bounding box
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}

# Mouth landmarks for click detection
MOUTH_TOP = 13      # Upper lip
MOUTH_BOTTOM = 14   # Lower lip

def mouse_mover():
    while True:
        if mouse_control_enabled == True:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)  # adjust for responsiveness

def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def apply_threshold(current_pos, prev_pos, threshold=MOTION_THRESHOLD):
    """Ignores small movements below a certain threshold."""
    dist = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    if dist < threshold:
        return prev_pos
    return current_pos

def smooth_position(current_pos, prev_pos, alpha=SMOOTHING_ALPHA):
    """Applies exponential smoothing to reduce jitter."""
    return tuple((1 - alpha) * np.array(prev_pos) + alpha * np.array(current_pos))

def calculate_velocity(yaw_deviation, pitch_deviation):
    """Calculate cursor velocity based on head deviation from center (joystick mode)."""
    # Apply dead zone
    if abs(yaw_deviation) < DEAD_ZONE:
        yaw_deviation = 0
    if abs(pitch_deviation) < DEAD_ZONE:
        pitch_deviation = 0
    
    # Calculate velocity with scaling
    velocity_x = (yaw_deviation / 20.0) * MAX_VELOCITY * VELOCITY_SCALE
    velocity_y = (pitch_deviation / 10.0) * MAX_VELOCITY * VELOCITY_SCALE
    
    # Clamp to max velocity
    velocity_x = np.clip(velocity_x, -MAX_VELOCITY, MAX_VELOCITY)
    velocity_y = np.clip(velocity_y, -MAX_VELOCITY, MAX_VELOCITY)
    
    return velocity_x, velocity_y

threading.Thread(target=mouse_mover, daemon=True).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Process the frame with timestamp
    results = face_landmarker.detect_for_video(mp_image, frame_counter)
    frame_counter += 1

    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]
        landmarks_frame = np.zeros_like(frame)  # Blank black frame

        outline_pts = []
        # Draw all landmarks as single white pixels
        for i, landmark in enumerate(face_landmarks):
            pt = landmark_to_np(landmark, w, h)
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                color = (155, 155, 155) if i in FACE_OUTLINE_INDICES else (255, 25, 10)
                cv2.circle(landmarks_frame, (x, y), 3, color, -1)
                frame[y, x] = (255, 255, 255)  # optional: also update main frame if needed

        # Mouth click detection
        if ENABLE_MOUTH_CLICK:
            mouth_top = landmark_to_np(face_landmarks[MOUTH_TOP], w, h)
            mouth_bottom = landmark_to_np(face_landmarks[MOUTH_BOTTOM], w, h)
            mouth_distance = np.linalg.norm(mouth_top - mouth_bottom)
            
            # Get face height for normalization
            top_pt = landmark_to_np(face_landmarks[LANDMARKS["top"]], w, h)
            bottom_pt = landmark_to_np(face_landmarks[LANDMARKS["bottom"]], w, h)
            face_height = np.linalg.norm(top_pt - bottom_pt)
            
            mouth_ratio = mouth_distance / face_height if face_height > 0 else 0
            
            # Check if mouth is open and cooldown has passed
            current_time = time.time()
            if mouth_ratio > MOUTH_OPEN_THRESHOLD and (current_time - last_click_time) > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = current_time
                print(f"CLICK! (mouth ratio: {mouth_ratio:.3f})")
            
            # Display mouth status
            cv2.putText(frame, f"Mouth: {mouth_ratio:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if mouth_ratio > MOUTH_OPEN_THRESHOLD else (255, 255, 255), 2)

        

        # Highlight bounding landmarks in pink
        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(face_landmarks[idx], w, h)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)

        # Extract points
        left = key_points["left"]
        right = key_points["right"]
        top = key_points["top"]
        bottom = key_points["bottom"]
        front = key_points["front"]

        # Oriented axes based on head geometry
        right_axis = (right - left)
        right_axis /= np.linalg.norm(right_axis)

        up_axis = (top - bottom)
        up_axis /= np.linalg.norm(up_axis)

        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis)

        # Flip to ensure forward vector comes out of the face
        forward_axis = -forward_axis

        # Compute center of the head
        center = (left + right + top + bottom + front) / 5

        # Half-sizes (width, height, depth)
        half_width = np.linalg.norm(right - left) / 2
        half_height = np.linalg.norm(top - bottom) / 2
        half_depth = 80  # can be tuned or calculated if you have a back landmark

        # Generate cube corners in face-aligned space
        def corner(x_sign, y_sign, z_sign):
            return (center
                    + x_sign * half_width * right_axis
                    + y_sign * half_height * up_axis
                    + z_sign * half_depth * forward_axis)

        cube_corners = [
            corner(-1, 1, -1),   # top-left-front
            corner(1, 1, -1),    # top-right-front
            corner(1, -1, -1),   # bottom-right-front
            corner(-1, -1, -1),  # bottom-left-front
            corner(-1, 1, 1),    # top-left-back
            corner(1, 1, 1),     # top-right-back
            corner(1, -1, 1),    # bottom-right-back
            corner(-1, -1, 1)    # bottom-left-back
        ]

        # Projection function
        def project(pt3d):
            return int(pt3d[0]), int(pt3d[1])

        # Draw wireframe cube
        cube_corners_2d = [project(pt) for pt in cube_corners]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # back face
            (0, 4), (1, 5), (2, 6), (3, 7)   # sides
        ]
        for i, j in edges:
            cv2.line(frame, cube_corners_2d[i], cube_corners_2d[j], (255, 125, 35), 2)

        # Update smoothing buffers
        ray_origins.append(center)
        ray_directions.append(forward_axis)

        # Compute averaged ray origin and direction
        avg_origin = np.mean(ray_origins, axis=0)
        avg_direction = np.mean(ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)  # normalize

        # Reference forward direction (camera looking straight ahead)
        reference_forward = np.array([0, 0, -1])  # Z-axis into the screen

        # Horizontal (yaw) angle from reference (project onto XZ plane)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj /= np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad  # left is negative

        # Vertical (pitch) angle from reference (project onto YZ plane)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive

        #Specify 

        # Convert to degrees and re-center around 0
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        #this results in the center being 180, +10 left = -170, +10 right = +170

        #convert left rotations to 0-180
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg

        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg
        
        #yaw is now converted to 90 (looking directly left) to 270 (looking directly right), wrt camera
        #pitch is now converted to 90 (looking straight down) and 270 (looking straight up), wrt camera
        #print(f"Angles: yaw={yaw_deg}, pitch={pitch_deg}")

        #specify degrees at which screen border will be reached
        yawDegrees = 20 # x degrees left or right
        pitchDegrees = 10 # x degrees up or down
        
        # leftmost pixel position must correspond to 180 - yaw degrees
        # rightmost pixel position must correspond to 180 + yaw degrees
        # topmost pixel position must correspond to 180 + pitch degrees
        # bottommost pixel position must correspond to 180 - pitch degrees

        # Apply calibration offsets
        yaw_deg += calibration_offset_yaw
        pitch_deg += calibration_offset_pitch

        if USE_VELOCITY_MODE:
            # Velocity/joystick mode: head deviation controls cursor speed
            yaw_deviation = yaw_deg - 180  # Deviation from center
            pitch_deviation = pitch_deg - 180
            
            # Calculate velocity
            velocity_x, velocity_y = calculate_velocity(yaw_deviation, pitch_deviation)
            
            # Update cursor position incrementally
            current_cursor_x += velocity_x
            current_cursor_y -= velocity_y  # Invert Y (up = negative pitch deviation)
            
            # Clamp to screen bounds
            current_cursor_x = np.clip(current_cursor_x, 10, MONITOR_WIDTH - 10)
            current_cursor_y = np.clip(current_cursor_y, 10, MONITOR_HEIGHT - 10)
            
            screen_x, screen_y = int(current_cursor_x), int(current_cursor_y)
            
            print(f"Velocity mode - Deviation: yaw={yaw_deviation:.1f}° pitch={pitch_deviation:.1f}° | "
                  f"Velocity: x={velocity_x:.1f} y={velocity_y:.1f} | Position: ({screen_x}, {screen_y})")
        else:
            # Original position mode: head angle maps directly to screen position
            # Map to full screen resolution
            screen_x = int(((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * MONITOR_WIDTH)
            screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

            # Clamp screen position to monitor bounds
            if(screen_x < 10):
                screen_x = 10
            if(screen_y < 10):
                screen_y = 10
            if(screen_x > MONITOR_WIDTH - 10):
                screen_x = MONITOR_WIDTH - 10
            if(screen_y > MONITOR_HEIGHT - 10):
                screen_y = MONITOR_HEIGHT - 10

            # Apply Kalman filter
            measurement = np.array([[np.float32(screen_x)], [np.float32(screen_y)]])
            kalman.correct(measurement)
            predicted = kalman.predict()
            smoothed_x, smoothed_y = int(predicted[0]), int(predicted[1])
            
            # Apply additional stabilization
            stabilized_position = apply_threshold((smoothed_x, smoothed_y), previous_cursor_position)
            stabilized_position = smooth_position(stabilized_position, previous_cursor_position)
            
            # Update previous position
            previous_cursor_position = stabilized_position
            screen_x, screen_y = stabilized_position

            print(f"Screen position: x={screen_x}, y={screen_y}")

        if mouse_control_enabled:
            with mouse_lock:
                mouse_target[0] = screen_x
                mouse_target[1] = screen_y

        # Draw smoothed ray
        ray_length = 2.5 * half_depth
        ray_end = avg_origin - avg_direction * ray_length
        
        # Draw the ray
        cv2.line(frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        cv2.line(landmarks_frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)

    cv2.imshow("Head-Aligned Cube", frame)
    cv2.imshow("Facial Landmarks", landmarks_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_offset_yaw = 180 - raw_yaw_deg
        calibration_offset_pitch = 180 - raw_pitch_deg
        print(f"[Calibrated] Offset Yaw: {calibration_offset_yaw}, Offset Pitch: {calibration_offset_pitch}")
    elif key == ord('t'):  # Toggle mouse control with 't' key
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")


cap.release()
cv2.destroyAllWindows()
