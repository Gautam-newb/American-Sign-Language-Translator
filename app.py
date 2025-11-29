#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KOMMUNIK8 - Sign Language Recognition Application

This is the main Flask application for real-time ASL gesture recognition.
It provides a web interface for live video feed with gesture recognition,
using MediaPipe for hand tracking and TensorFlow Lite models for classification.

Features:
- Real-time hand gesture recognition
- WebSocket-based real-time communication
- Video streaming with gesture overlay
- ASL letter recognition (A-Z)
- Gesture stabilization filter
"""
import argparse
import copy
import csv
import itertools
from collections import Counter, deque
import time

import cv2 as cv
import mediapipe as mp
import numpy as np

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc
from flask import Flask, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from sign_language_filter import SignLanguageFilter
import json
import os

app = Flask(__name__, static_url_path='', static_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the sign language filter with stricter parameters
sign_filter = SignLanguageFilter(
    window_size=30,  # Increased window size for more stability (about 1 second at 30fps)
    threshold=0.8,   # Increased threshold for more confidence
    cooldown_period=1.2  # Longer cooldown to prevent rapid changes
)

# Global variable to store the current video capture
camera = None

def get_args():
    """
    Parse command-line arguments for camera and MediaPipe configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - device: Camera device index (default: 0)
            - width: Frame width in pixels (default: 960)
            - height: Frame height in pixels (default: 540)
            - use_static_image_mode: Use static image mode for MediaPipe
            - min_detection_confidence: Minimum confidence for hand detection (0.0-1.0)
            - min_tracking_confidence: Minimum confidence for hand tracking (0.0-1.0)
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history
    finger_gesture_history = deque(maxlen=history_length)

    # Data collection mode
    mode = 0  # 0: Normal, 1: Data Collection
    number = 0  # Class number for data collection

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == ord('k'):  # Toggle data collection mode
            mode = 1 - mode
            number = 4  # Set to ASL A (index 4)
        elif key == ord('s'):  # Save data
            if mode == 1 and results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    logging_csv(number, mode, pre_processed_landmark_list, point_history)

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                    mode
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)

        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    """
    Calculate bounding rectangle around hand landmarks.
    
    Args:
        image: Input image (for dimensions)
        landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: Bounding rectangle coordinates [x_min, y_min, x_max, y_max]
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    # Convert normalized landmarks to pixel coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    # Calculate bounding rectangle
    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """
    Convert MediaPipe normalized landmarks to pixel coordinates.
    
    Args:
        image: Input image (for dimensions)
        landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: List of [x, y] pixel coordinates for each of 21 landmarks
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Convert normalized coordinates (0.0-1.0) to pixel coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z  # Z coordinate available but not used

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """
    Preprocess landmarks for model input: convert to relative coordinates and normalize.
    
    This function:
    1. Converts absolute coordinates to relative (wrist as origin)
    2. Flattens 2D list to 1D
    3. Normalizes to [-1, 1] range based on maximum absolute value
    
    Args:
        landmark_list: List of [x, y] pixel coordinates (21 landmarks)
    
    Returns:
        list: Normalized feature vector of 42 values (21 landmarks × 2 coordinates)
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates (wrist at index 0 is the origin)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list (flatten)
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization: divide by maximum absolute value to scale to [-1, 1]
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    """
    Preprocess point history for model input: convert to relative coordinates and normalize.
    
    This function processes the history of point movements (e.g., index finger tip)
    for dynamic gesture classification. It:
    1. Converts absolute coordinates to relative (first point as origin)
    2. Normalizes by image dimensions
    3. Flattens 2D list to 1D feature vector
    
    Args:
        image: Input image (for dimensions)
        point_history: List of [x, y] point coordinates over time (deque)
    
    Returns:
        list: Normalized feature vector (32 values: 16 points × 2 coordinates)
    """
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    """
    Log training data to CSV file for model training.
    
    This function appends normalized landmark data to a CSV file for later use
    in training the gesture recognition model. Only saves when in data collection mode.
    
    Args:
        number (int): Class label (gesture ID) for the sample
        mode (int): Mode flag (1 = data collection, 0 = normal)
        landmark_list (list): Preprocessed landmark coordinates (42 values)
        point_history_list: Point history data (currently unused but kept for compatibility)
    """
    if mode == 1:  # Data Collection mode
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    """
    Draw hand landmarks and connections on the image.
    
    This function visualizes the MediaPipe hand landmarks by drawing:
    - Lines connecting joints (fingers and palm)
    - Circles at each landmark point
    - Special highlighting for fingertips
    
    Args:
        image: Input image to draw on (modified in place)
        landmark_point: List of 21 [x, y] landmark coordinates
    
    Returns:
        numpy.ndarray: Image with landmarks drawn
    """
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    """
    Draw bounding rectangle around detected hand.
    
    Args:
        use_brect (bool): Whether to draw the bounding rectangle
        image: Input image to draw on
        brect (list): Bounding rectangle coordinates [x_min, y_min, x_max, y_max]
    
    Returns:
        numpy.ndarray: Image with bounding rectangle drawn (if enabled)
    """
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text, mode=0):
    """
    Draw text annotations on the image showing recognition results.
    
    This function displays:
    - Handedness (Left/Right) and recognized gesture
    - Large letter overlay for ASL letters
    - Data collection mode status
    - Finger gesture information
    
    Args:
        image: Input image to draw on
        brect (list): Bounding rectangle coordinates
        handedness: MediaPipe handedness classification object
        hand_sign_text (str): Recognized hand gesture label (e.g., "ASL A")
        finger_gesture_text (str): Recognized finger gesture label
        mode (int): Mode flag (0 = normal, 1 = data collection)
    
    Returns:
        numpy.ndarray: Image with text annotations drawn
    """
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Display data collection mode status
    if mode == 1:
        cv.putText(image, "Data Collection Mode: ASL A", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(image, "Press 'S' to save sample", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

    # Display large letter in center if it's an ASL letter
    if hand_sign_text.startswith('ASL '):
        letter = hand_sign_text[4:]  # Extract just the letter
        # Draw a semi-transparent background
        overlay = image.copy()
        cv.rectangle(overlay, (image.shape[1]//2 - 60, image.shape[0]//2 - 60),
                    (image.shape[1]//2 + 60, image.shape[0]//2 + 60),
                    (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        # Draw the letter
        cv.putText(image, letter, (image.shape[1]//2 - 40, image.shape[0]//2 + 40),
                  cv.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 3, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    """
    Draw point movement history as a trail on the image.
    
    This function visualizes the trajectory of point movements (e.g., index finger tip)
    by drawing circles that increase in size based on recency. Used for visualizing
    dynamic gestures like pointing or drawing.
    
    Args:
        image: Input image to draw on
        point_history: Deque containing history of [x, y] point coordinates
    
    Returns:
        numpy.ndarray: Image with point history trail drawn
    """
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def generate_frames():
    """
    Generator function that yields video frames with gesture recognition overlay.
    
    This function:
    - Captures frames from the camera
    - Processes them through MediaPipe for hand detection
    - Classifies gestures using the trained model
    - Applies stabilization filter
    - Draws landmarks and annotations
    - Emits recognized letters via WebSocket
    - Yields JPEG-encoded frames for streaming
    
    Yields:
        bytes: JPEG-encoded frame with MJPEG boundary markers
    """
    global camera
    
    # Initialize MediaPipe Hands for hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Load the keypoint classifier
    keypoint_classifier = KeyPointClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    last_emitted_letter = None
    letter_detection_start = None
    min_detection_duration = 1.5  # Time required to hold a gesture
    current_letter = None

    while True:
        if camera is None:
            break
            
        success, frame = camera.read()
        if not success:
            break
        else:
            # Mirror the frame
            frame = cv.flip(frame, 1)
            debug_image = copy.deepcopy(frame)

            # Convert to RGB for MediaPipe
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    # Calculate bounding box
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    
                    # Calculate landmark list
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    
                    # Pre-process landmarks
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    hand_sign_text = keypoint_classifier_labels[hand_sign_id]

                    # Create probability distribution for the filter
                    prob_dist = [0.0] * 26  # Initialize with zeros
                    if hand_sign_text.startswith('ASL '):
                        letter = hand_sign_text[4:]  # Extract the letter
                        if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                            letter_index = ord(letter) - ord('A')
                            prob_dist[letter_index] = 1.0  # Set probability to 1 for detected letter
                    
                    # Process through the sign language filter
                    detected_letter = sign_filter.process_frame(prob_dist)
                    
                    # Draw the results
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        hand_sign_text,
                        "",  # No finger gesture text
                        0    # Normal mode
                    )

                    # Handle letter detection timing
                    current_time = time.time()
                    
                    if detected_letter:
                        if detected_letter != current_letter:
                            # New gesture detected - reset timer
                            current_letter = detected_letter
                            letter_detection_start = current_time
                        elif current_time - letter_detection_start >= min_detection_duration:
                            # Only emit if we've detected the letter for long enough and it's different
                            if detected_letter != last_emitted_letter:
                                socketio.emit('transcript', {
                                    'type': 'gesture',
                                    'text': detected_letter
                                })
                                last_emitted_letter = detected_letter
                                # Reset timer immediately after successful detection
                                letter_detection_start = current_time
                    else:
                        # No gesture detected - reset everything
                        current_letter = None
                        letter_detection_start = None

                    # Add the filtered letter to the frame if detected
                    if detected_letter:
                        detection_time = current_time - letter_detection_start if letter_detection_start else 0
                        # Add color coding based on detection time
                        color = (0, 255, 0) if detection_time >= min_detection_duration else (0, 165, 255)  # Green if ready, Orange if waiting
                        cv.putText(debug_image, 
                                 f"Letter: {detected_letter} ({detection_time:.1f}s)", 
                                 (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Convert the frame to JPEG
            ret, buffer = cv.imencode('.jpg', debug_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Clean up
    hands.close()

@app.route('/')
def index():
    """
    Serve the main web interface.
    
    Returns:
        Response: HTML file (frontend.html)
    """
    return send_from_directory('.', 'frontend.html')

@app.route('/<path:path>')
def serve_static(path):
    """
    Serve static files from the root directory.
    
    Args:
        path: File path relative to root directory
    
    Returns:
        Response: Requested file or 404 if not found
    """
    return send_from_directory('.', path)

@app.route('/video_feed')
def video_feed():
    """
    Stream video feed with gesture recognition overlay.
    
    Returns:
        Response: MJPEG stream of processed video frames
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """
    Initialize and start the camera.
    
    Returns:
        JSON: Status response with success/error message
    """
    global camera
    if camera is None:
        camera = cv.VideoCapture(0)
        return jsonify({"status": "success", "message": "Camera started"})
    return jsonify({"status": "error", "message": "Camera already running"})

@app.route('/stop_camera')
def stop_camera():
    """
    Stop and release the camera.
    
    Returns:
        JSON: Status response with success/error message
    """
    global camera
    if camera is not None:
        camera.release()
        camera = None
        return jsonify({"status": "success", "message": "Camera stopped"})
    return jsonify({"status": "error", "message": "Camera not running"})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
