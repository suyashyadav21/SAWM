"""
SAWM State Definitions & Data Schemas.

Provides structured, typed state representations for:
- 11-body Franka Panda manipulator
- Static obstacles and dynamic kinematic proxies
- Contextual keep-out / restricted zones
- Distance metrics (exact geometric vs. approximate link-center)
- Contact and collision records
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class RobotState:
    """Robot state representing all 11 bodies and active joints."""
    joint_positions: np.ndarray       # (7,) arm joints in radians
    joint_velocities: np.ndarray      # (7,) arm joint velocities in rad/s
    ee_position: np.ndarray           # (3,) wrist/hand world Cartesian pos
    ee_quaternion: np.ndarray         # (4,) [w, x, y, z]
    gripper_position: float           # Opening distance in meters (0 to 0.04)
    # Cartesian world positions (3,) for all 11 tracked bodies
    body_positions: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class HazardState:
    """State of an obstacle or dynamic proxy."""
    name: str
    body_name: str
    geom_name: str
    position: np.ndarray              # (3,) world Cartesian pos
    velocity: np.ndarray              # (3,) Cartesian velocity in m/s
    enabled: bool = True
    is_mocap: bool = True


@dataclass
class RestrictedZoneState:
    """State of a contextual keep-out / restricted zone."""
    name: str
    bounds_min: np.ndarray            # [xmin, ymin, zmin]
    bounds_max: np.ndarray            # [xmax, ymax, zmax]
    warning_buffer: float             # Proximity buffer around zone in meters
    breached: bool = False            # True if any robot geom/body is strictly inside AABB
    warning_active: bool = False      # True if robot is within warning buffer
    min_distance_to_zone: float = 999.0
    breaching_bodies: List[str] = field(default_factory=list)


@dataclass
class DistanceMetrics:
    """
    Separation distance metrics.
    Explicitly distinguishes exact geometric surface-to-surface distance
    from approximate link-center clearance.
    """
    # Exact geometric surface distance (via MuJoCo mj_geomDistance)
    geometric_min_distance: float = 999.0
    nearest_robot_geom: str = "none"
    nearest_hazard_geom: str = "none"
    nearest_points: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (point_on_robot, point_on_hazard)

    # Approximate link-center clearance
    link_center_min_distance: float = 999.0
    nearest_robot_body: str = "none"
    nearest_hazard_body: str = "none"


@dataclass
class ContactMetrics:
    """Physical contact and collision records."""
    total_contacts: int = 0
    robot_obstacle_collision: bool = False
    robot_proxy_collision: bool = False
    robot_table_collision: bool = False
    robot_object_contact: bool = False
    max_penetration: float = 0.0      # Maximum negative distance in meters
    contact_pairs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SafetyTelemetry:
    """Aggregated safety status from the centralized SafetyMonitor."""
    state: str = "SAFE"               # "SAFE", "WARNING", "DANGER"
    is_latched: bool = False          # True if DANGER is latched due to physical collision
    physical_risk_score: float = 0.0  # Normalized [0.0, 1.0]
    contextual_harm_score: float = 0.0# Normalized [0.0, 1.0]
    active_alarms: List[str] = field(default_factory=list)
    near_miss_count: int = 0
    collision_count: int = 0
