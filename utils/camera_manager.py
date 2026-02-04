"""
Camera Manager Module
Handles camera access and frame capture for Kivy applications.
Works on both desktop and Android platforms.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from kivy.graphics.texture import Texture
from kivy.clock import Clock


class CameraManager:
    """
    Manages camera access and provides frames for processing.
    Handles conversion between OpenCV and Kivy formats.
    """
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize the camera manager.
        
        Args:
            camera_index: Camera device index (0 for default camera)
            resolution: Desired resolution (width, height)
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._frame_callback = None
        self._schedule = None
        
    def start(self, fps: int = 30) -> bool:
        """
        Start the camera capture.
        
        Args:
            fps: Target frames per second
            
        Returns:
            True if camera started successfully
        """
        if self.is_running:
            return True
            
        self.capture = cv2.VideoCapture(self.camera_index)
        
        if not self.capture.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.is_running = True
        return True
    
    def stop(self):
        """Stop the camera capture."""
        self.is_running = False
        if self._schedule:
            self._schedule.cancel()
            self._schedule = None
        if self.capture:
            self.capture.release()
            self.capture = None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame from the camera.
        
        Returns:
            BGR frame as numpy array, or None if capture failed
        """
        if not self.is_running or self.capture is None:
            return None
            
        ret, frame = self.capture.read()
        if not ret:
            return None
            
        return frame
    
    @staticmethod
    def frame_to_texture(frame: np.ndarray, flip_vertical: bool = True) -> Texture:
        """
        Convert an OpenCV frame to a Kivy texture.
        
        Args:
            frame: BGR numpy array from OpenCV
            flip_vertical: Whether to flip the texture vertically (needed for Kivy)
            
        Returns:
            Kivy Texture object
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip vertically if needed (Kivy's coordinate system)
        if flip_vertical:
            frame_rgb = cv2.flip(frame_rgb, 0)
        
        # Create texture
        h, w = frame_rgb.shape[:2]
        texture = Texture.create(size=(w, h), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        
        return texture
    
    @staticmethod
    def resize_frame(frame: np.ndarray, max_width: int = 640, max_height: int = 480) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / w, max_height / h)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def schedule_updates(self, callback, fps: int = 30):
        """
        Schedule regular frame updates using Kivy's Clock.
        
        Args:
            callback: Function to call with each frame
            fps: Target frames per second
        """
        self._frame_callback = callback
        interval = 1.0 / fps
        self._schedule = Clock.schedule_interval(self._update_callback, interval)
    
    def _update_callback(self, dt):
        """Internal callback for scheduled updates."""
        if self._frame_callback and self.is_running:
            frame = self.get_frame()
            if frame is not None:
                self._frame_callback(frame)
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get the actual frame size from the camera."""
        if self.capture and self.capture.isOpened():
            w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return self.resolution
