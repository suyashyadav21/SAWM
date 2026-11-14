"""
Physical Risk Head / Monitor for SAWM.

Estimates physical safety risks:
- Robot-obstacle proximity (based on exact surface-to-surface distance)
- Physical collisions with obstacles, dynamic proxies, or unintended table impacts
- Imminent contact risk scoring
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from observation.state import DistanceMetrics, ContactMetrics


class PhysicalRiskMonitor:
    """Evaluates physical collision risks from geometric distance and contact signals."""

    def __init__(self, warning_dist: float = 0.20, danger_dist: float = 0.08):
        self.warning_dist = float(warning_dist)
        self.danger_dist = float(danger_dist)

    def evaluate(self, distances: DistanceMetrics, contacts: ContactMetrics) -> Tuple[float, str, List[str]]:
        """
        Evaluate physical risk score and state.
        
        Args:
            distances: DistanceMetrics containing geometric_min_distance.
            contacts: ContactMetrics containing collision flags.
            
        Returns:
            Tuple of:
                - risk_score: float in [0.0, 1.0]
                - physical_state: "SAFE", "WARNING", or "DANGER"
                - alarms: list of active alarm strings
        """
        alarms = []
        is_collision = contacts.robot_obstacle_collision or contacts.robot_proxy_collision

        # 1. Collision check
        if is_collision:
            alarms.append(f"PHYSICAL COLLISION DETECTED (penetration: {contacts.max_penetration*1000:.1f}mm)")
            return 1.0, "DANGER", alarms

        # 2. Geometric distance check
        d_geom = distances.geometric_min_distance

        if d_geom <= self.danger_dist:
            # Danger threshold breached
            # Scale linearly from danger_dist down to 0.0 -> score in [0.8, 1.0]
            ratio = max(0.0, d_geom / max(1e-4, self.danger_dist))
            risk_score = 0.8 + 0.2 * (1.0 - ratio)
            alarms.append(f"CRITICAL PROXIMITY to {distances.nearest_hazard_geom} ({d_geom:.3f}m < {self.danger_dist:.2f}m)")
            return float(risk_score), "DANGER", alarms

        elif d_geom < self.warning_dist:
            # Warning proximity zone
            # Scale linearly from warning_dist down to danger_dist -> score in [0.2, 0.8]
            span = max(1e-4, self.warning_dist - self.danger_dist)
            ratio = (d_geom - self.danger_dist) / span
            risk_score = 0.2 + 0.6 * (1.0 - ratio)
            alarms.append(f"Proximity Warning: {distances.nearest_hazard_geom} at {d_geom:.3f}m")
            return float(risk_score), "WARNING", alarms

        else:
            # Safe separation
            # Slight background score if within 0.50m
            if d_geom < 0.50:
                risk_score = 0.2 * (0.50 - d_geom) / 0.30
            else:
                risk_score = 0.0
            return float(risk_score), "SAFE", alarms
