#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Point History Classifier Module

This module provides a TensorFlow Lite-based classifier for recognizing
dynamic hand gestures based on the history of point movements over time.
"""
import numpy as np
import tensorflow as tf


class PointHistoryClassifier(object):
    """
    Classifier for dynamic hand gestures using point movement history.
    
    This class loads a pre-trained TensorFlow Lite model that classifies
    gestures based on the trajectory of specific points (e.g., index finger tip)
    over a sequence of frames.
    
    Attributes:
        interpreter: TensorFlow Lite interpreter instance
        input_details: Model input tensor details
        output_details: Model output tensor details
        score_th: Confidence threshold for valid predictions
        invalid_value: Value returned when confidence is below threshold
    """
    
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        """
        Initialize the Point History Classifier.
        
        Args:
            model_path (str): Path to the TensorFlow Lite model file
            score_th (float): Confidence threshold (0.0-1.0) for valid predictions
            invalid_value (int): Class index to return when confidence is too low
            num_threads (int): Number of threads for inference (default: 1)
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        """
        Classify dynamic gesture from point movement history.
        
        Args:
            point_history (list): List of normalized point coordinates over time
                                 (32 values: 16 points × 2 coordinates)
        
        Returns:
            int: Class index of the predicted gesture, or invalid_value if
                 confidence is below threshold
        """
        # Get input tensor index
        input_details_tensor_index = self.input_details[0]['index']
        
        # Set input tensor with point history data
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        
        # Run inference
        self.interpreter.invoke()

        # Get output tensor index
        output_details_tensor_index = self.output_details[0]['index']

        # Retrieve prediction results
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Get class with highest probability
        result_index = np.argmax(np.squeeze(result))
        
        # Apply confidence threshold - return invalid if confidence too low
        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
