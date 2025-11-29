"""
FPS Calculator Utility

This module provides a utility class for calculating frames per second (FPS)
using OpenCV's high-resolution timer for accurate measurements.
"""
from collections import deque
import cv2 as cv


class CvFpsCalc(object):
    """
    Calculate frames per second (FPS) using OpenCV's tick counter.
    
    This class maintains a rolling buffer of frame times and calculates
    the average FPS over the specified buffer length for smooth readings.
    
    Attributes:
        _start_tick: Starting tick count for current frame
        _freq: Conversion factor from ticks to milliseconds
        _difftimes: Deque buffer storing frame time differences
    """
    
    def __init__(self, buffer_len=1):
        """
        Initialize the FPS calculator.
        
        Args:
            buffer_len (int): Number of frames to average over for FPS calculation
                            (default: 1 for instant FPS, higher for smoother average)
        """
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        """
        Calculate and return the current FPS.
        
        This method should be called once per frame. It calculates the time
        difference since the last call and updates the rolling average.
        
        Returns:
            float: Frames per second, rounded to 2 decimal places
        """
        # Get current tick count
        current_tick = cv.getTickCount()
        
        # Calculate time difference in milliseconds
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        # Add to rolling buffer
        self._difftimes.append(different_time)

        # Calculate average FPS (1000ms / average_frame_time)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
