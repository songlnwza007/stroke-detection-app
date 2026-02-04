"""
Hand Gesture Detection Module
Detects hand gestures (open hand vs fist) and counts transitions.
Used for stroke rehabilitation exercises.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum


class HandState(Enum):
    """Possible hand states"""
    OPEN = "Open Hand"
    FIST = "Fist"
    UNKNOWN = "Unknown"


class ExercisePhase(Enum):
    """Exercise phases"""
    BOTH_HANDS = 0
    LEFT_HAND = 1
    RIGHT_HAND = 2


@dataclass
class HandDetectionResult:
    """Results from hand detection"""
    left_hand_state: Optional[HandState] = None
    right_hand_state: Optional[HandState] = None
    left_fist_count: int = 0
    right_fist_count: int = 0
    left_open_count: int = 0
    right_open_count: int = 0
    hands_detected: int = 0


@dataclass
class ExerciseState:
    """State for timed exercise"""
    started: bool = False
    start_time: Optional[float] = None
    current_phase: ExercisePhase = ExercisePhase.BOTH_HANDS
    phase_duration: int = 20  # seconds per phase
    remaining_time: int = 20


class HandGestureDetector:
    """
    Detects hand gestures using MediaPipe Hands.
    Tracks open hand vs fist states and counts transitions.
    """
    
    # Finger tip and pip (proximal interphalangeal) landmarks
    FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    FINGER_PIPS = [2, 6, 10, 14, 18]
    
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        """
        Initialize the hand gesture detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Count tracking
        self.open_count_left = 0
        self.fist_count_left = 0
        self.open_count_right = 0
        self.fist_count_right = 0
        
        # Previous states for transition detection
        self.previous_state_left: Optional[HandState] = None
        self.previous_state_right: Optional[HandState] = None
        
        # Exercise state
        self.exercise = ExerciseState()
    
    def reset_counts(self):
        """Reset all gesture counts."""
        self.open_count_left = 0
        self.fist_count_left = 0
        self.open_count_right = 0
        self.fist_count_right = 0
        self.previous_state_left = None
        self.previous_state_right = None
    
    def start_exercise(self, phase_duration: int = 20):
        """
        Start timed exercise mode.
        
        Args:
            phase_duration: Duration of each phase in seconds
        """
        self.reset_counts()
        self.exercise.started = True
        self.exercise.start_time = time.time()
        self.exercise.phase_duration = phase_duration
        self.exercise.current_phase = ExercisePhase.BOTH_HANDS
    
    def stop_exercise(self):
        """Stop timed exercise mode."""
        self.exercise.started = False
        self.exercise.start_time = None
    
    def _is_hand_open(self, landmarks) -> bool:
        """
        Check if the hand is open based on finger positions.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            True if hand is open (4+ fingers extended)
        """
        open_fingers = 0
        
        # Check fingers (excluding thumb)
        for tip, pip in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                open_fingers += 1
        
        # Thumb check (horizontal comparison)
        if landmarks.landmark[self.FINGER_TIPS[0]].x > landmarks.landmark[self.FINGER_PIPS[0]].x:
            open_fingers += 1
        
        return open_fingers >= 4
    
    def _get_hand_state(self, landmarks) -> HandState:
        """Get the current state of a hand."""
        if self._is_hand_open(landmarks):
            return HandState.OPEN
        return HandState.FIST
    
    def _update_exercise_phase(self):
        """Update the current exercise phase based on elapsed time."""
        if not self.exercise.started or self.exercise.start_time is None:
            return
            
        elapsed_time = time.time() - self.exercise.start_time
        phase_index = int(elapsed_time // self.exercise.phase_duration) % 3
        self.exercise.current_phase = ExercisePhase(phase_index)
        self.exercise.remaining_time = self.exercise.phase_duration - (int(elapsed_time) % self.exercise.phase_duration)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[HandDetectionResult, np.ndarray]:
        """
        Process a single frame for hand gesture detection.
        
        Args:
            frame: BGR image frame from camera/video
            
        Returns:
            Tuple of (HandDetectionResult, annotated frame)
        """
        if frame is None:
            return HandDetectionResult(), frame
        
        # Flip frame for mirror view (user-friendly)
        frame = cv2.flip(frame, 1)
        annotated_frame = frame.copy()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Update exercise phase if running
        if self.exercise.started:
            self._update_exercise_phase()
        
        result = HandDetectionResult(
            left_fist_count=self.fist_count_left,
            right_fist_count=self.fist_count_right,
            left_open_count=self.open_count_left,
            right_open_count=self.open_count_right
        )
        
        if results.multi_hand_landmarks and results.multi_handedness:
            result.hands_detected = len(results.multi_hand_landmarks)
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                # Flip hand label for user's view (mirror effect)
                label_from_camera = handedness.classification[0].label
                # MediaPipe returns "Left" or "Right" (capitalized), but make comparison case-insensitive
                user_label = "right" if label_from_camera.lower() == "left" else "left"
                
                current_state = self._get_hand_state(hand_landmarks)
                
                # Update counts based on state transitions
                if user_label == "left":
                    result.left_hand_state = current_state
                    if self.previous_state_left != current_state:
                        if current_state == HandState.OPEN:
                            self.open_count_left += 1
                        else:
                            self.fist_count_left += 1
                        self.previous_state_left = current_state
                elif user_label == "right":
                    result.right_hand_state = current_state
                    if self.previous_state_right != current_state:
                        if current_state == HandState.OPEN:
                            self.open_count_right += 1
                        else:
                            self.fist_count_right += 1
                        self.previous_state_right = current_state
                
                # Display hand label and state
                y_pos = 30 if user_label == "left" else 150
                cv2.putText(annotated_frame, f"{user_label} - {current_state.value}",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Update result counts
            result.left_fist_count = self.fist_count_left
            result.right_fist_count = self.fist_count_right
            result.left_open_count = self.open_count_left
            result.right_open_count = self.open_count_right
        
        # Draw UI elements
        self._draw_ui(annotated_frame)
        
        return result, annotated_frame
    
    def _draw_ui(self, frame: np.ndarray):
        """Draw UI elements on the frame."""
        h, w = frame.shape[:2]
        
        # Draw counts
        cv2.putText(frame, f"Left Fist Count: {self.fist_count_left}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Right Fist Count: {self.fist_count_right}",
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw exercise info if running
        if self.exercise.started:
            # Instruction based on phase
            if self.exercise.current_phase == ExercisePhase.BOTH_HANDS:
                instruction = "Show BOTH Hands"
                color = (0, 0, 0)
            elif self.exercise.current_phase == ExercisePhase.LEFT_HAND:
                instruction = "Show LEFT Hand"
                color = (255, 0, 0)
            else:
                instruction = "Show RIGHT Hand"
                color = (0, 255, 0)
            
            cv2.putText(frame, instruction, (w - 350, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(frame, f"Time left: {self.exercise.remaining_time}s",
                        (w - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "Exercise not started",
                        (w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    
    def get_instruction(self) -> Tuple[str, Tuple[int, int, int]]:
        """
        Get current instruction text and color for exercise.
        
        Returns:
            Tuple of (instruction text, BGR color)
        """
        if not self.exercise.started:
            return "Press Start to begin exercise", (128, 128, 128)
        
        if self.exercise.current_phase == ExercisePhase.BOTH_HANDS:
            return "Show BOTH Hands", (0, 0, 0)
        elif self.exercise.current_phase == ExercisePhase.LEFT_HAND:
            return "Show LEFT Hand", (255, 0, 0)
        else:
            return "Show RIGHT Hand", (0, 255, 0)
    
    def release(self):
        """Release resources."""
        if self.hands:
            self.hands.close()
