#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KeyPoint Classifier Module

This module provides a TensorFlow Lite-based classifier for recognizing
static hand gestures (ASL letters) from normalized hand landmark coordinates.
"""
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    """
    Classifier for static hand gestures using TensorFlow Lite model.
    
    This class loads a pre-trained TensorFlow Lite model and provides
    inference capabilities for classifying hand gestures based on
    normalized landmark coordinates.
    
    Attributes:
        interpreter: TensorFlow Lite interpreter instance
        input_details: Model input tensor details
        output_details: Model output tensor details
    """
    
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        """
        Initialize the KeyPoint Classifier.
        
        Args:
            model_path (str): Path to the TensorFlow Lite model file
            num_threads (int): Number of threads for inference (default: 1)
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        """
        Classify hand gesture from normalized landmark coordinates.
        
        Args:
            landmark_list (list): List of 42 normalized coordinates
                                 (21 landmarks × 2 coordinates)
        
        Returns:
            int: Class index of the predicted gesture (0-29)
        """
        # Get input tensor index
        input_details_tensor_index = self.input_details[0]['index']
        
        # Set input tensor with normalized landmark data
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        
        # Run inference
        self.interpreter.invoke()

        # Get output tensor index
        output_details_tensor_index = self.output_details[0]['index']

        # Retrieve prediction results
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Get class with highest probability
        result_index = np.argmax(np.squeeze(result))

        return result_index
