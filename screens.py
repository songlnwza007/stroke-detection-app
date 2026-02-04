"""
Screen definitions for the Stroke Detection App.
Contains Home, Facial Detection, and Hand Detection screens.
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.clock import Clock
from kivy.utils import platform
from kivy.metrics import dp
from kivy.core.window import Window

import cv2
import numpy as np
import os

# Import detectors
from detectors import FacialStrokeDetector, HandGestureDetector
from utils import CameraManager


class HomeScreen(Screen):
    """
    Home screen with navigation to detection features.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_ui()
    
    def _build_ui(self):
        """Build the home screen UI."""
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Title
        title = Label(
            text='Stroke Detection App',
            font_size=dp(28),
            size_hint_y=0.15,
            bold=True,
            color=(0.2, 0.2, 0.8, 1)
        )
        layout.add_widget(title)
        
        # Subtitle
        subtitle = Label(
            text='Select a detection mode:',
            font_size=dp(18),
            size_hint_y=0.1,
            color=(0.4, 0.4, 0.4, 1)
        )
        layout.add_widget(subtitle)
        
        # Buttons container
        buttons_layout = BoxLayout(orientation='vertical', spacing=dp(20), size_hint_y=0.5)
        
        # Facial Detection Button
        facial_btn = Button(
            text='Facial Stroke Detection\n\nAnalyze facial asymmetry\nfor stroke indicators',
            font_size=dp(16),
            background_color=(0.3, 0.6, 0.9, 1),
            background_normal='',
            halign='center'
        )
        facial_btn.bind(on_press=self.go_to_facial)
        buttons_layout.add_widget(facial_btn)
        
        # Hand Detection Button
        hand_btn = Button(
            text='Hand Gesture Detection\n\nRehabilitation exercises\nwith gesture tracking',
            font_size=dp(16),
            background_color=(0.3, 0.8, 0.5, 1),
            background_normal='',
            halign='center'
        )
        hand_btn.bind(on_press=self.go_to_hand)
        buttons_layout.add_widget(hand_btn)
        
        layout.add_widget(buttons_layout)
        
        # Footer
        footer = Label(
            text='For medical screening purposes only.\nConsult a healthcare professional.',
            font_size=dp(12),
            size_hint_y=0.15,
            color=(0.5, 0.5, 0.5, 1),
            halign='center'
        )
        layout.add_widget(footer)
        
        self.add_widget(layout)
    
    def go_to_facial(self, instance):
        """Navigate to facial detection screen."""
        self.manager.transition.direction = 'left'
        self.manager.current = 'facial'
    
    def go_to_hand(self, instance):
        """Navigate to hand detection screen."""
        self.manager.transition.direction = 'left'
        self.manager.current = 'hand'


class FaceGuideOverlay(FloatLayout):
    """
    Overlay widget that draws a face positioning guide (oval frame).
    Similar to KYC verification in banking apps.
    Creates a darkened overlay with a transparent oval in the center.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opacity = 0.6  # Make overlay semi-transparent
        self.bind(size=self._update_guide, pos=self._update_guide)
        self._update_guide()
    
    def _update_guide(self, *args):
        """Update the face guide overlay position and size."""
        self.canvas.clear()
        
        with self.canvas:
            # Calculate oval guide position (centered)
            guide_width = min(self.width * 0.65, self.height * 0.45)
            guide_height = guide_width * 1.25  # Slightly taller for face shape
            guide_x = self.x + (self.width - guide_width) / 2
            guide_y = self.y + (self.height - guide_height) / 2
            
            # Draw semi-transparent dark overlay (4 rectangles around the oval)
            Color(0, 0, 0, 0.6)
            
            # Top rectangle
            Rectangle(
                pos=(self.x, guide_y + guide_height),
                size=(self.width, self.height - guide_y - guide_height)
            )
            # Bottom rectangle
            Rectangle(
                pos=(self.x, self.y),
                size=(self.width, guide_y - self.y)
            )
            # Left rectangle
            Rectangle(
                pos=(self.x, guide_y),
                size=(guide_x - self.x, guide_height)
            )
            # Right rectangle
            Rectangle(
                pos=(guide_x + guide_width, guide_y),
                size=(self.x + self.width - guide_x - guide_width, guide_height)
            )
            
            # Draw guide border (oval frame) - green color
            Color(0.2, 0.9, 0.2, 1)
            Line(ellipse=(guide_x, guide_y, guide_width, guide_height), width=4)
            
            # Draw corner guides (L-shaped corners like photo frames)
            corner_size = dp(25)
            corner_width = 3
            
            # Top-left corner
            Line(points=[
                guide_x, guide_y + guide_height,
                guide_x + corner_size, guide_y + guide_height
            ], width=corner_width)
            Line(points=[
                guide_x, guide_y + guide_height,
                guide_x, guide_y + guide_height - corner_size
            ], width=corner_width)
            
            # Top-right corner
            Line(points=[
                guide_x + guide_width, guide_y + guide_height,
                guide_x + guide_width - corner_size, guide_y + guide_height
            ], width=corner_width)
            Line(points=[
                guide_x + guide_width, guide_y + guide_height,
                guide_x + guide_width, guide_y + guide_height - corner_size
            ], width=corner_width)
            
            # Bottom-left corner
            Line(points=[
                guide_x, guide_y,
                guide_x + corner_size, guide_y
            ], width=corner_width)
            Line(points=[
                guide_x, guide_y,
                guide_x, guide_y + corner_size
            ], width=corner_width)
            
            # Bottom-right corner
            Line(points=[
                guide_x + guide_width, guide_y,
                guide_x + guide_width - corner_size, guide_y
            ], width=corner_width)
            Line(points=[
                guide_x + guide_width, guide_y,
                guide_x + guide_width, guide_y + corner_size
            ], width=corner_width)


class FacialDetectionScreen(Screen):
    """
    Facial stroke detection screen with image upload and face positioning guide.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = None
        self.current_image_path = None
        self.original_image = None
        self.camera = None
        self.camera_manager = None
        self.camera_update_event = None
        self.is_camera_mode = False
        self._build_ui()
    
    def _build_ui(self):
        """Build the facial detection screen UI."""
        layout = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        # Header with back button
        header = BoxLayout(size_hint_y=0.08, spacing=dp(10))
        
        back_btn = Button(
            text='< Back',
            size_hint_x=0.25,
            background_color=(0.5, 0.5, 0.5, 1),
            background_normal=''
        )
        back_btn.bind(on_press=self.go_back)
        header.add_widget(back_btn)
        
        title = Label(
            text='Facial Stroke Detection',
            font_size=dp(18),
            bold=True
        )
        header.add_widget(title)
        
        layout.add_widget(header)
        
        # Image preview container with overlay
        image_container = FloatLayout(size_hint_y=0.5)
        
        # Image widget (for displaying captured/uploaded images)
        self.image_widget = Image(allow_stretch=True, keep_ratio=True)
        image_container.add_widget(self.image_widget)
        
        # Camera widget (hidden by default)
        self.camera_widget = Camera(resolution=(640, 480), play=False)
        self.camera_widget.opacity = 0
        self.camera_widget.disabled = True
        image_container.add_widget(self.camera_widget)
        
        # Face guide overlay
        self.face_guide = FaceGuideOverlay()
        image_container.add_widget(self.face_guide)
        
        # Instruction label on overlay
        self.guide_label = Label(
            text='Upload image or take a photo',
            font_size=dp(14),
            color=(1, 1, 1, 1),
            size_hint=(None, None),
            size=(dp(250), dp(30)),
            pos_hint={'center_x': 0.5, 'y': 0.05}
        )
        image_container.add_widget(self.guide_label)
        
        layout.add_widget(image_container)
        
        # Results panel
        results_layout = BoxLayout(orientation='vertical', size_hint_y=0.25, spacing=dp(5))
        
        self.status_label = Label(
            text='Upload an image to begin',
            font_size=dp(18),
            bold=True,
            color=(0.5, 0.5, 0.5, 1)
        )
        results_layout.add_widget(self.status_label)
        
        self.mouth_label = Label(
            text='Mouth Symmetry: --',
            font_size=dp(14)
        )
        results_layout.add_widget(self.mouth_label)
        
        self.eye_label = Label(
            text='Eye Droop: --',
            font_size=dp(14)
        )
        results_layout.add_widget(self.eye_label)
        
        layout.add_widget(results_layout)
        
        # Control buttons
        buttons_layout = BoxLayout(size_hint_y=0.1, spacing=dp(10))
        
        self.upload_btn = Button(
            text='Upload Image',
            font_size=dp(14),
            background_color=(0.3, 0.6, 0.9, 1),
            background_normal=''
        )
        self.upload_btn.bind(on_press=self.show_file_chooser)
        buttons_layout.add_widget(self.upload_btn)
        
        self.camera_btn = Button(
            text='Take Photo',
            font_size=dp(14),
            background_color=(0.6, 0.4, 0.9, 1),
            background_normal=''
        )
        self.camera_btn.bind(on_press=self.toggle_camera)
        buttons_layout.add_widget(self.camera_btn)
        
        self.capture_btn = Button(
            text='Capture',
            font_size=dp(14),
            background_color=(0.9, 0.6, 0.2, 1),
            background_normal='',
            disabled=True,
            opacity=0
        )
        self.capture_btn.bind(on_press=self.capture_photo)
        buttons_layout.add_widget(self.capture_btn)
        
        self.analyze_btn = Button(
            text='Analyze',
            font_size=dp(14),
            background_color=(0.3, 0.8, 0.5, 1),
            background_normal='',
            disabled=True
        )
        self.analyze_btn.bind(on_press=self.analyze_image)
        buttons_layout.add_widget(self.analyze_btn)
        
        layout.add_widget(buttons_layout)
        
        self.add_widget(layout)
    
    def show_file_chooser(self, instance):
        """Show file chooser popup to select an image."""
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        
        # File chooser
        filechooser = FileChooserIconView()
        filechooser.path = os.path.expanduser('~')
        filechooser.filters = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        content.add_widget(filechooser)
        
        # Buttons
        btn_layout = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(10))
        
        cancel_btn = Button(text='Cancel')
        cancel_btn.bind(on_press=lambda x: popup.dismiss())
        btn_layout.add_widget(cancel_btn)
        
        select_btn = Button(text='Select')
        def select_file(instance):
            if filechooser.selection:
                self.load_image(filechooser.selection[0])
                popup.dismiss()
        select_btn.bind(on_press=select_file)
        btn_layout.add_widget(select_btn)
        
        content.add_widget(btn_layout)
        
        popup = Popup(
            title='Select Image',
            content=content,
            size_hint=(0.9, 0.9)
        )
        popup.open()
    
    def load_image(self, image_path):
        """Load and display the selected image."""
        try:
            # Load image with OpenCV
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                self.status_label.text = 'Error: Could not load image'
                return
            
            self.current_image_path = image_path
            
            # Resize for display while maintaining aspect ratio
            display_image = CameraManager.resize_frame(self.original_image, max_width=640, max_height=480)
            
            # Convert to texture and display
            texture = CameraManager.frame_to_texture(display_image, flip_vertical=False)
            self.image_widget.texture = texture
            
            # Update UI
            self.analyze_btn.disabled = False
            self.status_label.text = 'Image loaded. Click Analyze to detect.'
            self.status_label.color = (0.3, 0.3, 0.8, 1)
            self.guide_label.text = 'Position face within the frame'
            self.face_guide.opacity = 0.6  # Show guide clearly
            
            # Clear previous results
            self.mouth_label.text = 'Mouth Symmetry: --'
            self.eye_label.text = 'Eye Droop: --'
            
        except Exception as e:
            self.status_label.text = f'Error loading image: {str(e)}'
            self.status_label.color = (1, 0.2, 0.2, 1)
    
    def analyze_image(self, instance):
        """Analyze the uploaded image for stroke detection."""
        if self.original_image is None:
            self.status_label.text = 'Please upload an image first'
            return
        
        # Initialize detector if not already done
        if self.detector is None:
            self.detector = FacialStrokeDetector()
        
        # Process the image
        result, annotated_frame = self.detector.process_frame(self.original_image.copy())
        
        # Update UI with results
        if result:
            if result.is_stroke:
                self.status_label.text = 'Status: STROKE INDICATORS DETECTED'
                self.status_label.color = (1, 0.2, 0.2, 1)
            else:
                self.status_label.text = 'Status: NORMAL - No stroke indicators'
                self.status_label.color = (0.2, 0.8, 0.2, 1)
            
            self.mouth_label.text = f'Mouth Symmetry: {result.mouth_symmetry:.1f}%'
            self.eye_label.text = f'Eye Droop: {result.eye_droop_ratio:.1f}%'
            
            # Display annotated image
            display_image = CameraManager.resize_frame(annotated_frame, max_width=640, max_height=480)
            texture = CameraManager.frame_to_texture(display_image, flip_vertical=False)
            self.image_widget.texture = texture
            
            # Reduce guide opacity when analysis is complete
            self.face_guide.opacity = 0.2
            self.guide_label.text = 'Analysis complete'
        else:
            self.status_label.text = 'Status: No face detected. Please ensure face is clearly visible.'
            self.status_label.color = (0.8, 0.8, 0.2, 1)
            self.mouth_label.text = 'Mouth Symmetry: --'
            self.eye_label.text = 'Eye Droop: --'
    
    def toggle_camera(self, instance):
        """Toggle camera mode on/off."""
        if self.is_camera_mode:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera preview."""
        # Use OpenCV camera (more reliable across platforms)
        self.camera_manager = CameraManager(camera_index=0)
        if not self.camera_manager.start():
            # Try Kivy camera as fallback
            try:
                self.camera_widget.play = True
                self.camera_widget.opacity = 1
                self.camera_widget.disabled = False
                self.image_widget.opacity = 0
            except:
                self.status_label.text = 'Error: Cannot access camera'
                self.status_label.color = (1, 0.2, 0.2, 1)
                return
        else:
            # Use OpenCV camera with update loop
            self.camera_widget.opacity = 0
            self.camera_widget.play = False
            # Start update loop
            self.camera_update_event = Clock.schedule_interval(self.update_camera_preview, 1.0 / 30)
        
        self.is_camera_mode = True
        
        # Update buttons
        self.camera_btn.text = 'Stop Camera'
        self.camera_btn.background_color = (0.8, 0.3, 0.3, 1)
        self.capture_btn.disabled = False
        self.capture_btn.opacity = 1
        self.upload_btn.disabled = True
        self.analyze_btn.disabled = True
        
        # Show guide overlay
        self.face_guide.opacity = 0.6
        self.guide_label.text = 'Position face within the frame and tap Capture'
    
    def stop_camera(self):
        """Stop camera preview."""
        # Stop Kivy camera
        if self.camera_widget:
            self.camera_widget.play = False
            self.camera_widget.opacity = 0
            self.camera_widget.disabled = True
        
        # Stop OpenCV camera
        if self.camera_update_event:
            self.camera_update_event.cancel()
            self.camera_update_event = None
        
        if self.camera_manager:
            self.camera_manager.stop()
            self.camera_manager = None
        
        # Show image widget again
        self.image_widget.opacity = 1
        self.is_camera_mode = False
        
        # Update buttons
        self.camera_btn.text = 'Take Photo'
        self.camera_btn.background_color = (0.6, 0.4, 0.9, 1)
        self.capture_btn.disabled = True
        self.capture_btn.opacity = 0
        self.upload_btn.disabled = False
    
    def update_camera_preview(self, dt):
        """Update camera preview frame (for OpenCV camera)."""
        if not self.camera_manager or not self.is_camera_mode:
            return
        
        frame = self.camera_manager.get_frame()
        if frame is None:
            return
        
        # Convert frame to texture and display
        texture = CameraManager.frame_to_texture(frame, flip_vertical=False)
        self.image_widget.texture = texture
        self.image_widget.opacity = 1
    
    def capture_photo(self, instance):
        """Capture photo from camera."""
        if self.is_camera_mode:
            try:
                captured_frame = None
                
                # Try OpenCV camera first (more reliable)
                if self.camera_manager:
                    captured_frame = self.camera_manager.get_frame()
                
                # Fallback to Kivy camera
                elif self.camera_widget.play and self.camera_widget.texture:
                    try:
                        texture = self.camera_widget.texture
                        size = texture.size
                        # Get texture data
                        buf = texture.pixels
                        if buf:
                            frame = np.frombuffer(buf, dtype=np.uint8)
                            frame = frame.reshape((size[1], size[0], 4))  # RGBA
                            # Convert RGBA to BGR for OpenCV
                            captured_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    except:
                        pass
                
                if captured_frame is not None:
                    self.original_image = captured_frame.copy()
                    
                    # Stop camera
                    self.stop_camera()
                    
                    # Display captured image
                    display_image = CameraManager.resize_frame(self.original_image, max_width=640, max_height=480)
                    texture = CameraManager.frame_to_texture(display_image, flip_vertical=False)
                    self.image_widget.texture = texture
                    self.image_widget.opacity = 1
                    
                    # Update UI
                    self.analyze_btn.disabled = False
                    self.status_label.text = 'Photo captured. Click Analyze to detect.'
                    self.status_label.color = (0.3, 0.3, 0.8, 1)
                    self.face_guide.opacity = 0.6
                    self.guide_label.text = 'Photo ready for analysis'
                    
                    # Clear previous results
                    self.mouth_label.text = 'Mouth Symmetry: --'
                    self.eye_label.text = 'Eye Droop: --'
                else:
                    self.status_label.text = 'Error: Failed to capture photo'
                    self.status_label.color = (1, 0.2, 0.2, 1)
                    
            except Exception as e:
                self.status_label.text = f'Error capturing photo: {str(e)}'
                self.status_label.color = (1, 0.2, 0.2, 1)
    
    def on_enter(self):
        """Called when entering this screen."""
        pass
    
    def on_leave(self):
        """Called when leaving this screen."""
        self.stop_camera()
        if self.detector:
            self.detector.release()
            self.detector = None
    
    def go_back(self, instance):
        """Go back to home screen."""
        if self.detector:
            self.detector.release()
            self.detector = None
        self.manager.transition.direction = 'right'
        self.manager.current = 'home'
    
    def cleanup(self):
        """Clean up resources."""
        if self.detector:
            self.detector.release()
            self.detector = None


class HandDetectionScreen(Screen):
    """
    Hand gesture detection screen for rehabilitation exercises.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = None
        self.detector = None
        self.update_event = None
        self._build_ui()
    
    def _build_ui(self):
        """Build the hand detection screen UI."""
        layout = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        # Header with back button
        header = BoxLayout(size_hint_y=0.08, spacing=dp(10))
        
        back_btn = Button(
            text='< Back',
            size_hint_x=0.25,
            background_color=(0.5, 0.5, 0.5, 1),
            background_normal=''
        )
        back_btn.bind(on_press=self.go_back)
        header.add_widget(back_btn)
        
        title = Label(
            text='Hand Gesture Detection',
            font_size=dp(18),
            bold=True
        )
        header.add_widget(title)
        
        layout.add_widget(header)
        
        # Instruction label
        self.instruction_label = Label(
            text='Press Start to begin exercise',
            font_size=dp(16),
            size_hint_y=0.06,
            color=(0.3, 0.3, 0.8, 1)
        )
        layout.add_widget(self.instruction_label)
        
        # Timer label
        self.timer_label = Label(
            text='Time: --',
            font_size=dp(14),
            size_hint_y=0.04,
            color=(0.5, 0.5, 0.5, 1)
        )
        layout.add_widget(self.timer_label)
        
        # Camera preview
        self.camera_image = Image(size_hint_y=0.5)
        layout.add_widget(self.camera_image)
        
        # Results panel
        results_layout = GridLayout(cols=2, size_hint_y=0.15, spacing=dp(10), padding=dp(5))
        
        # Left hand counts
        left_box = BoxLayout(orientation='vertical')
        left_box.add_widget(Label(text='LEFT HAND', font_size=dp(14), bold=True, color=(0.8, 0.2, 0.2, 1)))
        self.left_fist_label = Label(text='Fist: 0', font_size=dp(12))
        left_box.add_widget(self.left_fist_label)
        self.left_state_label = Label(text='State: --', font_size=dp(12))
        left_box.add_widget(self.left_state_label)
        results_layout.add_widget(left_box)
        
        # Right hand counts
        right_box = BoxLayout(orientation='vertical')
        right_box.add_widget(Label(text='RIGHT HAND', font_size=dp(14), bold=True, color=(0.2, 0.8, 0.2, 1)))
        self.right_fist_label = Label(text='Fist: 0', font_size=dp(12))
        right_box.add_widget(self.right_fist_label)
        self.right_state_label = Label(text='State: --', font_size=dp(12))
        right_box.add_widget(self.right_state_label)
        results_layout.add_widget(right_box)
        
        layout.add_widget(results_layout)
        
        # Control buttons
        buttons_layout = BoxLayout(size_hint_y=0.1, spacing=dp(10))
        
        self.start_btn = Button(
            text='Start Camera',
            font_size=dp(16),
            background_color=(0.3, 0.6, 0.9, 1),
            background_normal=''
        )
        self.start_btn.bind(on_press=self.toggle_camera)
        buttons_layout.add_widget(self.start_btn)
        
        self.exercise_btn = Button(
            text='Start Exercise',
            font_size=dp(16),
            background_color=(0.3, 0.8, 0.5, 1),
            background_normal='',
            disabled=True
        )
        self.exercise_btn.bind(on_press=self.toggle_exercise)
        buttons_layout.add_widget(self.exercise_btn)
        
        reset_btn = Button(
            text='Reset',
            font_size=dp(16),
            size_hint_x=0.4,
            background_color=(0.8, 0.5, 0.2, 1),
            background_normal=''
        )
        reset_btn.bind(on_press=self.reset_counts)
        buttons_layout.add_widget(reset_btn)
        
        layout.add_widget(buttons_layout)
        
        self.add_widget(layout)
        self.is_camera_running = False
        self.is_exercise_running = False
    
    def on_enter(self):
        """Called when entering this screen."""
        pass
    
    def on_leave(self):
        """Called when leaving this screen."""
        self.stop_camera()
    
    def toggle_camera(self, instance):
        """Toggle camera on/off."""
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera capture."""
        self.camera = CameraManager(camera_index=0)
        if not self.camera.start():
            self.instruction_label.text = 'Error: Cannot access camera'
            return
        
        self.detector = HandGestureDetector()
        self.update_event = Clock.schedule_interval(self.update_frame, 1.0 / 15)
        
        self.is_camera_running = True
        self.start_btn.text = 'Stop Camera'
        self.start_btn.background_color = (0.8, 0.3, 0.3, 1)
        self.exercise_btn.disabled = False
        self.instruction_label.text = 'Camera ready. Press Start Exercise to begin.'
    
    def stop_camera(self):
        """Stop camera capture."""
        if self.is_exercise_running:
            self.stop_exercise()
        
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None
        
        if self.camera:
            self.camera.stop()
            self.camera = None
        
        if self.detector:
            self.detector.release()
            self.detector = None
        
        self.is_camera_running = False
        self.start_btn.text = 'Start Camera'
        self.start_btn.background_color = (0.3, 0.6, 0.9, 1)
        self.exercise_btn.disabled = True
        self.instruction_label.text = 'Press Start Camera to begin'
    
    def toggle_exercise(self, instance):
        """Toggle exercise mode."""
        if self.is_exercise_running:
            self.stop_exercise()
        else:
            self.start_exercise()
    
    def start_exercise(self):
        """Start timed exercise."""
        if self.detector:
            self.detector.start_exercise(phase_duration=20)
            self.is_exercise_running = True
            self.exercise_btn.text = 'Stop Exercise'
            self.exercise_btn.background_color = (0.8, 0.3, 0.3, 1)
    
    def stop_exercise(self):
        """Stop timed exercise."""
        if self.detector:
            self.detector.stop_exercise()
        self.is_exercise_running = False
        self.exercise_btn.text = 'Start Exercise'
        self.exercise_btn.background_color = (0.3, 0.8, 0.5, 1)
        self.instruction_label.text = 'Exercise stopped'
        self.timer_label.text = 'Time: --'
    
    def reset_counts(self, instance):
        """Reset gesture counts."""
        if self.detector:
            self.detector.reset_counts()
        self.left_fist_label.text = 'Fist: 0'
        self.right_fist_label.text = 'Fist: 0'
    
    def update_frame(self, dt):
        """Update frame and run detection."""
        if not self.camera or not self.detector:
            return
        
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        # Run detection
        result, annotated_frame = self.detector.process_frame(frame)
        
        # Update counts
        self.left_fist_label.text = f'Fist: {result.left_fist_count}'
        self.right_fist_label.text = f'Fist: {result.right_fist_count}'
        
        # Update state labels
        if result.left_hand_state:
            self.left_state_label.text = f'State: {result.left_hand_state.value}'
        else:
            self.left_state_label.text = 'State: Not detected'
        
        if result.right_hand_state:
            self.right_state_label.text = f'State: {result.right_hand_state.value}'
        else:
            self.right_state_label.text = 'State: Not detected'
        
        # Update exercise info
        if self.is_exercise_running:
            instruction, color = self.detector.get_instruction()
            self.instruction_label.text = instruction
            # Convert BGR to Kivy color format
            kivy_color = (color[2]/255, color[1]/255, color[0]/255, 1)
            self.instruction_label.color = kivy_color
            self.timer_label.text = f'Time: {self.detector.exercise.remaining_time}s'
        
        # Convert frame to texture and display
        texture = CameraManager.frame_to_texture(annotated_frame)
        self.camera_image.texture = texture
    
    def go_back(self, instance):
        """Go back to home screen."""
        self.stop_camera()
        self.manager.transition.direction = 'right'
        self.manager.current = 'home'
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_camera()
