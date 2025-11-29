from flask import Flask, send_from_directory, jsonify, request, Response
import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import copy
from collections import Counter, deque
from model import KeyPointClassifier, PointHistoryClassifier
from sign_language_filter import SignLanguageFilter
# Import gesture processing functions from app.py
from app import (
    calc_bounding_rect, calc_landmark_list, pre_process_landmark,
    pre_process_point_history, draw_landmarks, draw_info_text, draw_point_history
)
from flask_socketio import SocketIO
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory('.', 'frontend.html')

# Serve static files (if needed in the future)
@app.route('/<path:filename>')
def static_files(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return 'File not found', 404

# --- Gesture Recognition Video Feed ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Initialize sign language filter with optimized parameters
sign_filter = SignLanguageFilter(
    window_size=5,  # Smaller window for faster response
    threshold=0.7,  # Higher threshold for more confident detections
    cooldown_period=0.3  # Shorter cooldown for more responsive updates
)

history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

# Global variables for camera handling
camera = None
camera_lock = threading.Lock()
frame_generator = None

# Gesture detection variables
last_emitted_letter = None
last_letter = None
last_emission_time = 0
gesture_start_time = None
min_detection_duration = 0.5  # Time required to hold a gesture
stabilization_buffer = deque(maxlen=5)  # Buffer for gesture stabilization
required_consistent_frames = 3  # Number of consistent frames required

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise Exception("Failed to open camera")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            camera.set(cv2.CAP_PROP_FPS, 30)
            # Wait for camera to initialize
            time.sleep(0.5)
        return camera

def release_camera():
    global camera, frame_generator
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        frame_generator = None

def generate_frames():
    global last_emitted_letter, last_letter, last_emission_time, gesture_start_time, frame_generator
    try:
        cap = get_camera()
        if cap is None:
            return
            
        while True:
            if cap is None:
                break
                
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            debug_image = copy.deepcopy(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            
            current_sign_id = None
            gesture_text = None
            current_time = time.time()

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    current_sign_id = hand_sign_id
                    
                    if hand_sign_id == 2:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        calc_bounding_rect(debug_image, hand_landmarks),
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]]
                    )
                    gesture_text = keypoint_classifier_labels[hand_sign_id]
                    
                    # Process gesture through sign language filter
                    if gesture_text and gesture_text.startswith('ASL '):
                        # Create probability distribution for the filter
                        prob_dist = [0.0] * 26  # Initialize with zeros
                        letter = gesture_text[4:].strip()
                        if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                            letter_index = ord(letter) - ord('A')
                            # Create a more nuanced probability distribution
                            for i in range(26):
                                if i == letter_index:
                                    prob_dist[i] = 0.9  # High probability for detected letter
                                else:
                                    prob_dist[i] = 0.004  # Low probability for other letters
                            
                            # Add to stabilization buffer
                            stabilization_buffer.append(letter)
                            
                            # Check for consistent gesture
                            if len(stabilization_buffer) >= required_consistent_frames:
                                # Count occurrences of each letter in the buffer
                                letter_counts = Counter(stabilization_buffer)
                                most_common = letter_counts.most_common(1)[0]
                                
                                # If we have enough consistent frames
                                if most_common[1] >= required_consistent_frames:
                                    stabilized_letter = most_common[0]
                                    
                                    # Check if we're in cooldown period
                                    if current_time - last_emission_time >= sign_filter.cooldown_period:
                                        # Process through the sign language filter
                                        filtered_letter = sign_filter.process_frame(prob_dist)
                                        
                                        # Emit letter if detected by filter
                                        if filtered_letter and filtered_letter != last_emitted_letter:
                                            socketio.emit('transcript', {'type': 'gesture', 'text': filtered_letter})
                                            last_emitted_letter = filtered_letter
                                            last_letter = filtered_letter
                                            last_emission_time = current_time
                                            # Clear buffer after successful detection
                                            stabilization_buffer.clear()
                            else:
                                # Not enough frames yet, show orange text
                                debug_image = cv2.putText(
                                    debug_image,
                                    f"Stabilizing: {letter}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (0, 165, 255),  # Orange color
                                    2,
                                    cv2.LINE_AA
                                )
            else:
                point_history.append([0, 0])
                # Reset filter state when no hand is detected
                sign_filter.buffer.clear()
                sign_filter.current_output = None
                last_emitted_letter = None
                last_letter = None
                stabilization_buffer.clear()

            debug_image = draw_point_history(debug_image, point_history)
            ret, buffer = cv2.imencode('.jpg', debug_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames: {str(e)}")
        release_camera()
        return

@app.route('/start_camera')
def start_camera():
    try:
        get_camera()  # This will initialize the camera if it's not already running
        return jsonify({"status": "success", "message": "Camera started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_camera')
def stop_camera():
    try:
        release_camera()
        return jsonify({"status": "success", "message": "Camera stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Example API endpoint (expand as needed)
@app.route('/api/example', methods=['GET'])
def example_api():
    return jsonify({'message': 'API is working!'})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000) 