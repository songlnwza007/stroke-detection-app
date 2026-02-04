"""
Facial Stroke Detection Module
Detects potential stroke indicators through facial asymmetry analysis:
- Mouth asymmetry (primary indicator)
- Eye drooping (secondary indicator)
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class StrokeDetectionResult:
    """Results from stroke detection analysis"""
    is_stroke: bool
    mouth_symmetry: float
    eye_droop_ratio: float
    mouth_asymmetry_detected: bool
    eye_droop_detected: bool
    left_eye_opening: float
    right_eye_opening: float
    landmarks: Optional[Dict] = None


class FacialStrokeDetector:
    """
    Detects potential stroke indicators using MediaPipe Face Mesh.
    Analyzes mouth asymmetry and eye drooping patterns.
    """
    
    # Define left and right landmark pairs for mouth (left, right)
    MOUTH_LANDMARK_PAIRS = [
        (61, 291),   # Outer lip left-right
        (146, 375),
        (91, 321),
        (181, 405),
        (84, 314),
        (78, 308),   # Inner lip left-right
        (95, 324),
        (88, 318),
        (178, 402),
        (87, 317),
    ]
    
    # Left eye landmarks
    LEFT_EYE_LANDMARKS = {
        'outer_corner': 33,
        'inner_corner': 133,
        'top': 159,
        'bottom': 145,
        'top_outer': 160,
        'top_inner': 158,
        'bottom_outer': 144,
        'bottom_inner': 153
    }
    
    # Right eye landmarks
    RIGHT_EYE_LANDMARKS = {
        'outer_corner': 362,
        'inner_corner': 263,
        'top': 386,
        'bottom': 374,
        'top_outer': 385,
        'top_inner': 387,
        'bottom_outer': 380,
        'bottom_inner': 373
    }
    
    # Detection thresholds
    EYE_DROOP_THRESHOLD = 15.0      # Eye opening difference percentage
    MOUTH_ASYMMETRY_THRESHOLD = 75.0  # Mouth symmetry percentage (lower = more asymmetric)
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the facial stroke detector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[StrokeDetectionResult], np.ndarray]:
        """
        Process a single frame for stroke detection.
        
        Args:
            frame: BGR image frame from camera/video
            
        Returns:
            Tuple of (StrokeDetectionResult or None if no face detected, annotated frame)
        """
        if frame is None:
            return None, frame
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        
        if not results.multi_face_landmarks:
            return None, annotated_frame
            
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            # Extract all landmark coordinates
            landmarks = {
                idx: (face_landmarks.landmark[idx].x * w,
                      face_landmarks.landmark[idx].y * h)
                for idx in range(468)
            }
            
            # Analyze mouth symmetry
            mouth_symmetry, mouth_asymmetry_detected = self._analyze_mouth_symmetry(
                landmarks, annotated_frame
            )
            
            # Analyze eye drooping
            eye_droop_ratio, eye_droop_detected, left_opening, right_opening = \
                self._analyze_eye_drooping(landmarks, annotated_frame)
            
            # Final stroke determination
            is_stroke = mouth_asymmetry_detected or (mouth_asymmetry_detected and eye_droop_detected)
            
            # Draw results on frame
            self._draw_results(
                annotated_frame, is_stroke, mouth_symmetry, 
                eye_droop_ratio, mouth_asymmetry_detected, eye_droop_detected
            )
            
            return StrokeDetectionResult(
                is_stroke=is_stroke,
                mouth_symmetry=mouth_symmetry,
                eye_droop_ratio=eye_droop_ratio,
                mouth_asymmetry_detected=mouth_asymmetry_detected,
                eye_droop_detected=eye_droop_detected,
                left_eye_opening=left_opening,
                right_eye_opening=right_opening,
                landmarks=landmarks
            ), annotated_frame
            
        return None, annotated_frame
    
    def _analyze_mouth_symmetry(self, landmarks: Dict, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Analyze mouth symmetry from facial landmarks.
        
        Returns:
            Tuple of (symmetry percentage, whether asymmetry is detected)
        """
        # Find vertical midline from mouth landmarks
        mouth_indices = set(sum(self.MOUTH_LANDMARK_PAIRS, ()))
        mouth_x_coords = [landmarks[idx][0] for idx in mouth_indices]
        mid_x = np.mean(mouth_x_coords)
        
        # Compute mouth symmetry based on distances from midline
        symmetry_scores = []
        
        for left_idx, right_idx in self.MOUTH_LANDMARK_PAIRS:
            left_x, left_y = landmarks[left_idx]
            right_x, right_y = landmarks[right_idx]
            
            # Compute distance from mid_x
            left_distance = abs(left_x - mid_x)
            right_distance = abs(right_x - mid_x)
            
            # Symmetry ratio (1 means perfectly symmetric)
            if max(left_distance, right_distance) > 0:
                symmetry_ratio = min(left_distance, right_distance) / max(left_distance, right_distance)
                symmetry_scores.append(symmetry_ratio)
            
            # Draw mouth landmarks
            cv2.circle(frame, (int(left_x), int(left_y)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_x), int(right_y)), 3, (0, 255, 0), -1)
        
        mouth_symmetry = np.mean(symmetry_scores) * 100 if symmetry_scores else 0
        
        # Draw mouth midline
        mouth_y_coords = [landmarks[idx][1] for idx in mouth_indices]
        mouth_top_y = min(mouth_y_coords)
        mouth_bottom_y = max(mouth_y_coords)
        cv2.line(frame, (int(mid_x), int(mouth_top_y)), 
                 (int(mid_x), int(mouth_bottom_y)), (255, 0, 0), 2)
        
        asymmetry_detected = mouth_symmetry < self.MOUTH_ASYMMETRY_THRESHOLD
        
        return mouth_symmetry, asymmetry_detected
    
    def _analyze_eye_drooping(self, landmarks: Dict, frame: np.ndarray) -> Tuple[float, bool, float, float]:
        """
        Analyze eye drooping from facial landmarks.
        
        Returns:
            Tuple of (droop ratio, whether drooping is detected, left opening, right opening)
        """
        # Extract eye landmark coordinates
        left_eye = {key: landmarks[idx] for key, idx in self.LEFT_EYE_LANDMARKS.items()}
        right_eye = {key: landmarks[idx] for key, idx in self.RIGHT_EYE_LANDMARKS.items()}
        
        # Measure eye opening height
        left_eye_opening = abs(left_eye['top'][1] - left_eye['bottom'][1])
        right_eye_opening = abs(right_eye['top'][1] - right_eye['bottom'][1])
        
        # Calculate eye opening difference
        eye_opening_diff = abs(left_eye_opening - right_eye_opening)
        avg_eye_opening = (left_eye_opening + right_eye_opening) / 2
        opening_droop_ratio = (eye_opening_diff / avg_eye_opening) * 100 if avg_eye_opening > 0 else 0
        
        # Draw eye landmarks
        for key, (x, y) in left_eye.items():
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        for key, (x, y) in right_eye.items():
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        droop_detected = opening_droop_ratio > self.EYE_DROOP_THRESHOLD
        
        return opening_droop_ratio, droop_detected, left_eye_opening, right_eye_opening
    
    def _draw_results(self, frame: np.ndarray, is_stroke: bool, mouth_symmetry: float,
                      eye_droop_ratio: float, mouth_asymmetry: bool, eye_droop: bool):
        """Draw detection results on the frame."""
        status = "STROKE" if is_stroke else "NON-STROKE"
        status_color = (0, 0, 255) if is_stroke else (0, 255, 0)
        
        y_offset = 25
        line_height = 25
        
        # Title
        cv2.putText(frame, "Stroke Detection Analysis",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # Mouth symmetry
        cv2.putText(frame, f"Mouth Symmetry: {mouth_symmetry:.1f}%",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += line_height
        
        # Eye drooping
        cv2.putText(frame, f"Eye Droop: {eye_droop_ratio:.1f}%",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        y_offset += line_height + 5
        
        # Final result with background
        text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
        cv2.rectangle(frame, (8, y_offset - 20),
                      (text_size[0] + 15, y_offset + 10), (0, 0, 0), -1)
        cv2.putText(frame, status,
                    (10, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 3)
    
    def release(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()
