"""
Sign Language Filter Module

This module provides a stabilization filter for sign language gesture recognition.
It reduces false positives by averaging probability distributions over a sliding
window and applying confidence thresholds and cooldown periods.
"""
from collections import deque
import numpy as np
import time

class SignLanguageFilter:
    def __init__(self, window_size=10, threshold=0.6, cooldown_period=0.5):
        """
        Initialize the filter with a window size and confidence threshold.
        
        Args:
            window_size (int): Number of frames to average over (e.g., 10 frames).
            threshold (float): Minimum average probability to accept a new letter (e.g., 0.6).
            cooldown_period (float): Minimum time in seconds between letter detections.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown_period = cooldown_period
        self.buffer = deque(maxlen=window_size)  # Stores probability distributions
        self.current_output = None  # Current detected letter
        self.last_detection_time = 0  # Time of last letter detection
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Assuming 26 letters

    def process_frame(self, prob_dist):
        """
        Process a new frame's probability distribution and return the filtered output.
        
        Args:
            prob_dist (list or np.array): Probability distribution over letters (sums to 1).
        
        Returns:
            str or None: Detected letter or None if no confident detection.
        """
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_detection_time < self.cooldown_period:
            return self.current_output

        # Ensure prob_dist is a numpy array
        prob_dist = np.array(prob_dist)
        if prob_dist.shape[0] != len(self.alphabet):
            raise ValueError("Probability distribution size must match alphabet size.")

        # Add new probability distribution to buffer
        self.buffer.append(prob_dist)

        # If buffer isn't full yet, return current output (or None if first frames)
        if len(self.buffer) < self.window_size:
            return self.current_output

        # Compute average probability over the window
        avg_probs = np.mean(list(self.buffer), axis=0)
        
        # Find the letter with the highest average probability
        max_prob_idx = np.argmax(avg_probs)
        max_prob = avg_probs[max_prob_idx]
        candidate_letter = self.alphabet[max_prob_idx]

        # Update output only if:
        # 1. The max probability exceeds threshold
        # 2. The letter is different from current output
        # 3. We're not in cooldown period
        if (max_prob >= self.threshold and 
            candidate_letter != self.current_output and 
            current_time - self.last_detection_time >= self.cooldown_period):
            self.current_output = candidate_letter
            self.last_detection_time = current_time

        return self.current_output

# Example usage
if __name__ == "__main__":
    filter = SignLanguageFilter(window_size=10, threshold=0.6)
    
    # Simulate some frames (26 letters, A-Z)
    # Frame 1-5: Steady 'A' with high probability
    steady_a = [0.9 if i == 0 else 0.004 for i in range(26)]  # 'A' at index 0
    for _ in range(5):
        print(filter.process_frame(steady_a))  # Should output None, then 'A'
    
    # Frame 6-10: Transition with mixed probabilities
    transition = [0.4 if i == 0 else 0.023 for i in range(26)]  # Lower prob for 'A'
    for _ in range(5):
        print(filter.process_frame(transition))  # Should hold 'A'
    
    # Frame 11-15: Steady 'B' with high probability
    steady_b = [0.9 if i == 1 else 0.004 for i in range(26)]  # 'B' at index 1
    for _ in range(5):
        print(filter.process_frame(steady_b))  # Should switch to 'B'