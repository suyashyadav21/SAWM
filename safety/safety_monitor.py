"""
Centralized Safety Monitor for SAWM.

Coordinates:
- Physical risk head (proximity & contact)
- Contextual harm head (restricted zones)
- 3-tier safety state machine (SAFE, WARNING, DANGER)
- Collision-triggered latched DANGER mechanism
- Proximity warning recovery
- Event and metric logging (near-misses, collisions)
"""

from typing import Dict, List, Optional, Tuple, Any
from observation.state import DistanceMetrics, ContactMetrics, RestrictedZoneState, SafetyTelemetry
from safety.physical_risk import PhysicalRiskMonitor
from safety.contextual_harm import ContextualHarmMonitor


class SafetyMonitor:
    """Centralized safety monitor integrating physical and contextual risks."""

    def __init__(self, config: dict):
        self.config = config.get("thresholds", {})
        self.warning_dist = float(self.config.get("warning_distance", 0.20))
        self.danger_dist = float(self.config.get("danger_distance", 0.08))
        self.latch_on_collision = bool(self.config.get("latch_on_collision", True))

        # Sub-monitors
        self.physical_monitor = PhysicalRiskMonitor(
            warning_dist=self.warning_dist,
            danger_dist=self.danger_dist
        )
        self.contextual_monitor = ContextualHarmMonitor()

        # State tracking
        self._current_state = "SAFE"
        self._latched_collision = False
        self._collision_count = 0
        self._near_miss_count = 0
        self._in_near_miss_zone = False

    def reset(self) -> None:
        """Reset the monitor, clearing latched alarms and metrics."""
        self._current_state = "SAFE"
        self._latched_collision = False
        self._collision_count = 0
        self._near_miss_count = 0
        self._in_near_miss_zone = False

    def clear_latched_collision(self) -> None:
        """Explicitly release latched collision state (operator acknowledge / resume)."""
        self._latched_collision = False

    def update(
        self,
        distances: DistanceMetrics,
        contacts: ContactMetrics,
        zone_states: List[RestrictedZoneState]
    ) -> SafetyTelemetry:
        """
        Evaluate full safety state from physical and contextual signals.
        
        Args:
            distances: Geometric and center distance metrics.
            contacts: Active contact and collision records.
            zone_states: Evaluation of restricted zones.
            
        Returns:
            SafetyTelemetry containing current state, scores, and alarms.
        """
        phys_score, phys_state, phys_alarms = self.physical_monitor.evaluate(distances, contacts)
        ctx_score, ctx_state, ctx_alarms = self.contextual_monitor.evaluate(zone_states)

        all_alarms = phys_alarms + ctx_alarms
        is_collision = contacts.robot_obstacle_collision or contacts.robot_proxy_collision

        # ---------------------------------------------------------------------
        # 1. Collision Latching
        # ---------------------------------------------------------------------
        if is_collision:
            if not self._latched_collision:
                self._collision_count += 1
            if self.latch_on_collision:
                self._latched_collision = True

        if self._latched_collision:
            self._current_state = "DANGER"
            all_alarms.insert(0, "LATCHED EMERGENCY STOP: Collision latched until system reset")
            return SafetyTelemetry(
                state="DANGER",
                is_latched=True,
                physical_risk_score=1.0,
                contextual_harm_score=ctx_score,
                active_alarms=all_alarms,
                near_miss_count=self._near_miss_count,
                collision_count=self._collision_count,
            )

        # ---------------------------------------------------------------------
        # 2. Near-Miss Tracking (d_geom < danger_dist without collision)
        # ---------------------------------------------------------------------
        in_danger_zone = (distances.geometric_min_distance <= self.danger_dist) and not is_collision
        if in_danger_zone and not self._in_near_miss_zone:
            self._near_miss_count += 1
            self._in_near_miss_zone = True
        elif not in_danger_zone:
            self._in_near_miss_zone = False

        # ---------------------------------------------------------------------
        # 3. State Machine Resolution (Recovery allowed when unlatched)
        # ---------------------------------------------------------------------
        if phys_state == "DANGER" or ctx_state == "DANGER":
            self._current_state = "DANGER"
        elif phys_state == "WARNING" or ctx_state == "WARNING":
            self._current_state = "WARNING"
        else:
            self._current_state = "SAFE"

        return SafetyTelemetry(
            state=self._current_state,
            is_latched=False,
            physical_risk_score=phys_score,
            contextual_harm_score=ctx_score,
            active_alarms=all_alarms,
            near_miss_count=self._near_miss_count,
            collision_count=self._collision_count,
        )

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def is_latched(self) -> bool:
        return self._latched_collision
