"""
Vision Safety System - Main Entry Point
Combines all three modules:
1. Lane Detection Without Markings
2. Object Detection + Collision Avoidance
3. Driver Monitoring

Run with: python main.py
"""

import cv2
import time
import argparse
import numpy as np
from typing import Optional

# Import modules
from module1_road import LaneDetectionSystem
from module2_collision import CollisionAvoidanceSystem
from module3_driver import DriverMonitor
from utils.video_utils import VideoHandler
from utils.alerts import AlertSystem, AlertLevel
from utils.visualization import Visualizer
from config import SystemConfig, DEFAULT_CONFIG


class VisionSafetySystem:
    """
    Main system combining all three safety modules
    """
    
    def __init__(self, config: SystemConfig = None):
        """
        Initialize the complete safety system
        
        Args:
            config: System configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
        
        print("=" * 60)
        print("   VISION-BASED SAFETY INTELLIGENCE SYSTEM")
        print("=" * 60)
        
        # Initialize modules based on config
        self.lane_system: Optional[LaneDetectionSystem] = None
        self.collision_system: Optional[CollisionAvoidanceSystem] = None
        self.driver_monitor: Optional[DriverMonitor] = None
        
        # Initialize enabled modules
        if self.config.module1.enabled:
            print("\n[MODULE 1] Initializing Lane Detection...")
            self.lane_system = LaneDetectionSystem(
                model_path=self.config.module1.model_path,
                confidence_threshold=self.config.module1.confidence_threshold
            )
            
        if self.config.module2.enabled:
            print("\n[MODULE 2] Initializing Collision Avoidance...")
            self.collision_system = CollisionAvoidanceSystem(
                model_path=self.config.module2.model_path,
                confidence_threshold=self.config.module2.confidence_threshold,
                ttc_danger=self.config.module2.ttc_danger,
                ttc_warning=self.config.module2.ttc_warning
            )
            
        if self.config.module3.enabled:
            print("\n[MODULE 3] Initializing Driver Monitoring...")
            self.driver_monitor = DriverMonitor(
                ear_threshold=self.config.module3.ear_threshold,
                ear_consec_frames=self.config.module3.ear_consec_frames,
                mar_threshold=self.config.module3.mar_threshold,
                pitch_threshold=self.config.module3.pitch_threshold,
                yaw_threshold=self.config.module3.yaw_threshold,
                phone_detection_enabled=self.config.module3.phone_detection_enabled
            )
            
        # Alert system
        self.alert_system = AlertSystem(sound_enabled=True)
        
        # Visualizer
        self.visualizer = Visualizer()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("\n" + "=" * 60)
        print("   System Ready!")
        print("=" * 60 + "\n")
        
    def process_frame(self, road_frame: np.ndarray, 
                      driver_frame: np.ndarray = None) -> np.ndarray:
        """
        Process frames from cameras
        
        Args:
            road_frame: Frame from road-facing camera
            driver_frame: Frame from driver-facing camera (optional)
            
        Returns:
            Annotated road frame
        """
        self.frame_count += 1
        output_frame = road_frame.copy()
        
        # Update visualizer dimensions
        h, w = road_frame.shape[:2]
        self.visualizer.frame_width = w
        self.visualizer.frame_height = h
        
        # Process Module 1: Lane Detection
        lane_state = None
        if self.lane_system:
            lane_state = self.lane_system.process_frame(road_frame)
            self.lane_system.draw_on_frame(output_frame, lane_state)
            
            # Set lane region for collision filtering
            if self.collision_system and lane_state.lane_region:
                self.collision_system.set_lane_region(lane_state.lane_region)
                
            # Add lane alerts
            for level, msg in lane_state.alerts:
                self.alert_system.add_alert(
                    AlertLevel[level], msg, "LANE"
                )
                
        # Process Module 2: Collision Avoidance
        collision_state = None
        if self.collision_system:
            collision_state = self.collision_system.process_frame(road_frame)
            self.collision_system.draw_on_frame(output_frame, collision_state)
            
            # Add collision alerts
            for level, msg in collision_state.alerts:
                self.alert_system.add_alert(
                    AlertLevel[level], msg, "COLLISION"
                )
                
        # Process Module 3: Driver Monitoring
        driver_state = None
        if self.driver_monitor:
            # Use driver frame if provided, otherwise use road frame
            frame_for_driver = driver_frame if driver_frame is not None else road_frame
            driver_state = self.driver_monitor.process_frame(frame_for_driver)
            
            # Draw driver info on road frame (small panel)
            self._draw_driver_panel(output_frame, driver_state)
            
            # Add driver alerts
            for level, msg in driver_state.alerts:
                self.alert_system.add_alert(
                    AlertLevel[level], msg, "DRIVER"
                )
                
        # Draw alerts
        active_alerts = self.alert_system.get_active_alerts()
        self.visualizer.draw_alerts(output_frame, active_alerts)
        
        # Draw FPS
        self._update_fps()
        self.visualizer.draw_fps(output_frame, self.fps)
        
        return output_frame
        
    def _draw_driver_panel(self, frame: np.ndarray, driver_state):
        """Draw driver monitoring panel"""
        if driver_state is None:
            return
            
        h, w = frame.shape[:2]
        panel_x = w - 280
        panel_y = 10
        panel_w = 270
        panel_h = 130
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "DRIVER STATUS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status indicator
        status_colors = {
            "attentive": (0, 255, 0),
            "drowsy": (0, 0, 255),
            "distracted": (0, 165, 255),
            "phone": (0, 0, 255),
        }
        color = status_colors.get(driver_state.driver_status, (255, 255, 255))
        cv2.putText(frame, driver_state.driver_status.upper(), (panel_x + 10, panel_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Stats
        cv2.putText(frame, f"EAR: {driver_state.ear:.2f}", (panel_x + 10, panel_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Attention: {driver_state.overall_attention_score:.0f}%", 
                   (panel_x + 10, panel_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Looking: {driver_state.looking_direction}", 
                   (panel_x + 10, panel_y + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                   
    def _update_fps(self):
        """Update FPS calculation"""
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
    def reset(self):
        """Reset all modules"""
        if self.lane_system:
            self.lane_system.reset()
        if self.collision_system:
            self.collision_system.reset()
        if self.driver_monitor:
            self.driver_monitor.reset()
        self.alert_system.clear()
        
    def release(self):
        """Release resources"""
        if self.driver_monitor:
            self.driver_monitor.release()


def run_demo(video_source=0, driver_camera=None):
    """
    Run the demo with video source
    
    Args:
        video_source: Video file path or camera index for road view
        driver_camera: Camera index for driver view (None = use road camera)
    """
    # Create configuration
    config = SystemConfig()
    config.module1.enabled = True
    config.module2.enabled = True
    config.module3.enabled = True
    
    # Initialize system
    system = VisionSafetySystem(config)
    
    # Open video sources
    print(f"\n[INFO] Opening video source: {video_source}")
    road_cap = cv2.VideoCapture(video_source)
    
    driver_cap = None
    if driver_camera is not None:
        print(f"[INFO] Opening driver camera: {driver_camera}")
        driver_cap = cv2.VideoCapture(driver_camera)
        
    if not road_cap.isOpened():
        print("[ERROR] Could not open video source!")
        return
        
    # Get video properties
    width = int(road_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(road_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = road_cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS")
    
    # Create window
    cv2.namedWindow("Vision Safety System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Safety System", 1280, 720)
    
    print("\n[INFO] Press 'Q' to quit, 'R' to reset\n")
    
    while True:
        # Read frames
        ret, road_frame = road_cap.read()
        if not ret:
            # Loop video
            road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        driver_frame = None
        if driver_cap is not None:
            ret_d, driver_frame = driver_cap.read()
            
        # Process
        output = system.process_frame(road_frame, driver_frame)
        
        # Display
        cv2.imshow("Vision Safety System", output)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            system.reset()
            print("[INFO] System reset")
            
    # Cleanup
    road_cap.release()
    if driver_cap:
        driver_cap.release()
    cv2.destroyAllWindows()
    system.release()
    
    print("\n[INFO] System stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vision-Based Safety Intelligence System")
    parser.add_argument("--video", "-v", type=str, default="0",
                       help="Video source (file path or camera index)")
    parser.add_argument("--driver-camera", "-d", type=int, default=None,
                       help="Driver camera index (optional)")
    
    args = parser.parse_args()
    
    # Parse video source
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
        
    run_demo(video_source, args.driver_camera)


if __name__ == "__main__":
    main()
