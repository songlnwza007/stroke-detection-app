"""
Stroke Detection App
Main entry point for the Kivy Android application.
Combines facial stroke detection and hand gesture detection features.
"""

import os
import sys

# Set environment variable for Kivy before importing
os.environ['KIVY_VIDEO'] = 'ffpyplayer'

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.core.window import Window
from kivy.utils import platform

# Import screens
from screens import HomeScreen, FacialDetectionScreen, HandDetectionScreen


class StrokeDetectionApp(App):
    """
    Main application class for the Stroke Detection App.
    Manages screens and navigation between features.
    """
    
    def build(self):
        """Build and return the root widget."""
        # Set window size for desktop testing
        if platform not in ('android', 'ios'):
            Window.size = (400, 700)
        
        # Request camera permission on Android
        if platform == 'android':
            self._request_android_permissions()
        
        # Create screen manager
        self.screen_manager = ScreenManager(transition=SlideTransition())
        
        # Add screens
        self.screen_manager.add_widget(HomeScreen(name='home'))
        self.screen_manager.add_widget(FacialDetectionScreen(name='facial'))
        self.screen_manager.add_widget(HandDetectionScreen(name='hand'))
        
        return self.screen_manager
    
    def _request_android_permissions(self):
        """Request necessary permissions on Android."""
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE,
            ])
        except ImportError:
            pass  # Not on Android
    
    def on_start(self):
        """Called when the app starts."""
        print("Stroke Detection App started")
    
    def on_stop(self):
        """Called when the app stops."""
        # Clean up resources
        for screen in self.screen_manager.screens:
            if hasattr(screen, 'cleanup'):
                screen.cleanup()
        print("Stroke Detection App stopped")
    
    def switch_screen(self, screen_name: str, direction: str = 'left'):
        """
        Switch to a different screen.
        
        Args:
            screen_name: Name of the screen to switch to
            direction: Transition direction ('left', 'right', 'up', 'down')
        """
        self.screen_manager.transition.direction = direction
        self.screen_manager.current = screen_name


def main():
    """Main entry point."""
    StrokeDetectionApp().run()


if __name__ == '__main__':
    main()
