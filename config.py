"""
Configuration file for Vision Safety System
All settings in one place for easy modification
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

# =============================================================================
# ENUMS
# =============================================================================

class AlertLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriverState(Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    PHONE = "phone_detected"
    UNKNOWN = "unknown"

class CollisionRisk(Enum):
    SAFE = "safe"
    MONITOR = "monitor"
    WARNING = "warning"
    DANGER = "danger"

# =============================================================================
# MODULE CONFIGURATIONS
# =============================================================================

@dataclass
class CameraConfig:
    """Camera settings"""
    road_camera_id: int = 0          # External camera for road
    driver_camera_id: int = 1        # Internal camera for driver (or same as road)
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30

@dataclass
class Module1Config:
    """Road Segmentation & Virtual Lane Detection"""
    enabled: bool = True
    model_path: str = "models/yolov8n-seg.pt"
    confidence_threshold: float = 0.5
    road_classes: Tuple[str, ...] = ("road", "pavement", "terrain")
    lane_smoothing_window: int = 5
    polynomial_degree: int = 2

@dataclass
class Module2Config:
    """Object Detection & Collision Avoidance"""
    enabled: bool = True
    model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = 0.4
    nms_iou_threshold: float = 0.45
    
    # Target classes for collision detection
    vehicle_classes: Tuple[str, ...] = ("car", "truck", "bus", "motorcycle", "bicycle")
    pedestrian_classes: Tuple[str, ...] = ("person",)
    
    # Tracking settings
    track_buffer: int = 30
    track_threshold: float = 0.5
    match_threshold: float = 0.8
    
    # Distance estimation (reference object heights in meters)
    reference_heights: dict = None
    
    # Collision thresholds (in meters)
    ttc_danger: float = 1.5      # Time-to-collision danger threshold (seconds)
    ttc_warning: float = 3.0     # Time-to-collision warning threshold (seconds)
    distance_danger: float = 5.0  # Distance danger threshold (meters)
    distance_warning: float = 15.0 # Distance warning threshold (meters)
    
    def __post_init__(self):
        if self.reference_heights is None:
            self.reference_heights = {
                "person": 1.7,
                "car": 1.5,
                "truck": 3.0,
                "bus": 3.2,
                "motorcycle": 1.2,
                "bicycle": 1.1,
            }

@dataclass
class Module3Config:
    """Driver Monitoring System"""
    enabled: bool = True
    
    # Eye Aspect Ratio (EAR) settings
    ear_threshold: float = 0.22          # Below this = eyes closed
    ear_consec_frames: int = 15          # Frames to trigger drowsy alert
    
    # Mouth Aspect Ratio (MAR) settings
    mar_threshold: float = 0.6           # Above this = yawning
    yawn_consec_frames: int = 10         # Frames to confirm yawn
    
    # Head Pose settings
    pitch_threshold: float = 20.0        # Looking up/down degrees
    yaw_threshold: float = 30.0          # Looking left/right degrees
    distraction_frames: int = 30         # Frames to trigger distraction
    
    # Phone detection
    phone_detection_enabled: bool = True
    phone_confidence: float = 0.5
    
    # Attention score weights
    weight_ear: float = 0.35
    weight_mar: float = 0.15
    weight_head_pose: float = 0.30
    weight_phone: float = 0.20

@dataclass
class DisplayConfig:
    """Visualization settings"""
    show_fps: bool = True
    show_detections: bool = True
    show_tracking: bool = True
    show_lanes: bool = True
    show_driver_status: bool = True
    show_alerts: bool = True
    
    # Colors (BGR format)
    color_safe: Tuple[int, int, int] = (0, 255, 0)      # Green
    color_warning: Tuple[int, int, int] = (0, 165, 255)  # Orange
    color_danger: Tuple[int, int, int] = (0, 0, 255)     # Red
    color_lane: Tuple[int, int, int] = (255, 255, 0)     # Cyan
    color_info: Tuple[int, int, int] = (255, 255, 255)   # White

# =============================================================================
# MAIN CONFIG
# =============================================================================

@dataclass
class SystemConfig:
    """Main system configuration"""
    camera: CameraConfig = None
    module1: Module1Config = None
    module2: Module2Config = None
    module3: Module3Config = None
    display: DisplayConfig = None
    
    # General settings
    use_openvino: bool = True    # Use Intel OpenVINO for acceleration
    video_source: str = None      # Video file path or camera index
    save_output: bool = False
    output_path: str = "output/result.mp4"
    
    def __post_init__(self):
        if self.camera is None:
            self.camera = CameraConfig()
        if self.module1 is None:
            self.module1 = Module1Config()
        if self.module2 is None:
            self.module2 = Module2Config()
        if self.module3 is None:
            self.module3 = Module3Config()
        if self.display is None:
            self.display = DisplayConfig()

# Default configuration instance
DEFAULT_CONFIG = SystemConfig()
