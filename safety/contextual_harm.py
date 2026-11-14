"""
Contextual Harm Head / Monitor for SAWM.

Evaluates semantic and procedural workspace risks:
- Entering restricted / keep-out volumes
- Proximity to sensitive workstation zones
- Separation of contextual policy hazards from physical collision dynamics
"""

from typing import List, Tuple
from observation.state import RestrictedZoneState


class ContextualHarmMonitor:
    """Evaluates contextual and zone-based hazards independently from physical collisions."""

    def __init__(self):
        pass

    def evaluate(self, zone_states: List[RestrictedZoneState]) -> Tuple[float, str, List[str]]:
        """
        Evaluate contextual harm across all configured zones.
        
        Args:
            zone_states: List of RestrictedZoneState objects.
            
        Returns:
            Tuple of:
                - harm_score: float in [0.0, 1.0]
                - contextual_state: "SAFE", "WARNING", or "DANGER"
                - alarms: list of active alarm strings
        """
        if not zone_states:
            return 0.0, "SAFE", []

        max_harm = 0.0
        overall_state = "SAFE"
        alarms = []

        for z in zone_states:
            if z.breached:
                max_harm = 1.0
                overall_state = "DANGER"
                bodies_str = ", ".join(z.breaching_bodies) if z.breaching_bodies else "robot"
                alarms.append(f"RESTRICTED ZONE BREACH: {bodies_str} inside '{z.name}'")

            elif z.warning_active and overall_state != "DANGER":
                # Inside warning buffer
                dist = z.min_distance_to_zone
                harm = 0.5 * max(0.0, (z.warning_buffer - dist) / max(1e-4, z.warning_buffer))
                max_harm = max(max_harm, float(harm))
                overall_state = "WARNING"
                alarms.append(f"Zone Warning: Approaching '{z.name}' ({dist:.3f}m from boundary)")

        return max_harm, overall_state, alarms
