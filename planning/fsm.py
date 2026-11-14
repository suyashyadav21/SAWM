"""
Finite-State Machine (FSM) for Franka Panda Pick-and-Place Task.

Enforces:
- 8-stage task sequence (HOME -> PREGRASP -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> RELEASE -> RETURN)
- Explicit FAILED states with specific failure reasons:
  - IK_CONVERGENCE_FAILURE
  - GRASP_FAILURE (no contact or failed lift)
  - OBJECT_SLIPPAGE_FAILURE (object dropped during transit)
  - PLACEMENT_FAILURE (object outside goal zone)
  - TIMEOUT_FAILURE (exceeded max steps per stage)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class TaskState(str, Enum):
    """Discrete states of the pick-and-place FSM."""
    HOME = "HOME"
    MOVE_TO_PREGRASP = "MOVE_TO_PREGRASP"
    DESCEND_TO_OBJECT = "DESCEND_TO_OBJECT"
    GRASP = "GRASP"
    LIFT = "LIFT"
    MOVE_TO_TARGET = "MOVE_TO_TARGET"
    RELEASE = "RELEASE"
    RETURN_HOME = "RETURN_HOME"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class TaskResult:
    """Comprehensive outcome and telemetry of a task execution."""
    success: bool
    final_state: str
    failure_reason: Optional[str] = None
    execution_time: float = 0.0
    trajectory_length: float = 0.0
    grasp_success: bool = False
    placement_error: float = 999.0
    joint_limit_violations: int = 0
    state_history: List[str] = field(default_factory=list)
    stages_completed: List[str] = field(default_factory=list)
    safety_summary: Dict[str, Any] = field(default_factory=dict)


class PickPlaceFSM:
    """State machine governing pick-and-place task execution and failure guards."""

    def __init__(self, config: dict):
        self.config = config.get("fsm", {})

        self.table_z = float(self.config.get("table_surface_z", 0.400))
        self.min_lift_height = float(self.config.get("min_lift_height", 0.050))
        self.max_transit_slip = float(self.config.get("max_transit_slip", 0.035))
        self.goal_radius = float(self.config.get("goal_tolerance_radius", 0.060))
        self.timeout_steps = int(self.config.get("fsm_timeout_steps", 200))

        self.grasp_dwell_steps = int(self.config.get("grasp_dwell_steps", 40))
        self.release_dwell_steps = int(self.config.get("release_dwell_steps", 30))

        self.reset()

    def reset(self) -> None:
        """Reset state machine to initial state."""
        self.state = TaskState.HOME
        self.state_step_count = 0
        self.failure_reason: Optional[str] = None
        self.state_history = [self.state.value]
        self.stages_completed: List[str] = []

        # Tracking for slippage detection
        self._initial_grasp_offset: Optional[float] = None
        self.grasp_verified = False
        self.placement_verified = False

    def transition_to(self, new_state: TaskState, reason: Optional[str] = None) -> None:
        """Execute state transition."""
        if new_state != self.state and self.state not in [TaskState.COMPLETED, TaskState.FAILED]:
            self.stages_completed.append(self.state.value)
        self.state = new_state
        self.state_step_count = 0
        if not self.state_history or self.state_history[-1] != new_state.value:
            self.state_history.append(new_state.value)
        if reason:
            self.failure_reason = reason

    def fail(self, reason: str) -> None:
        """Transition to FAILED with an explicit diagnostic reason."""
        self.transition_to(TaskState.FAILED, reason=reason)

    def check_timeout(self) -> bool:
        """Check if current state exceeded allowable step limit."""
        if self.state_step_count > self.timeout_steps:
            self.fail(f"TIMEOUT_FAILURE: State '{self.state.value}' exceeded {self.timeout_steps} steps")
            return True
        return False

    def update_grasp_tracking(self, ee_pos: np.ndarray, obj_pos: np.ndarray, contacts: Any) -> bool:
        """
        Verify grasp during LIFT and TRANSPORT phases.
        
        Checks:
        1. Appropriate robot-object contacts exist.
        2. Object is elevated above table surface.
        3. Object follows gripper without excessive relative drift (slippage).
        """
        curr_dist = float(np.linalg.norm(ee_pos - obj_pos))
        obj_z = obj_pos[2]
        lift_height = obj_z - self.table_z

        has_contact = contacts.get("robot_object_contact", False) if isinstance(contacts, dict) else getattr(contacts, "robot_object_contact", False)

        if self.state == TaskState.LIFT:
            # Track lift progress; mark grasp verified once threshold is attained
            if lift_height >= self.min_lift_height and has_contact:
                self.grasp_verified = True
                self._initial_grasp_offset = curr_dist
            return True  # Allow lift segment to complete smoothly

        elif self.state == TaskState.MOVE_TO_TARGET:
            # Slippage / dropped object check
            if not self.grasp_verified:
                self.fail("GRASP_FAILURE: Object was never properly grasped before transport")
                return False

            if self._initial_grasp_offset is not None:
                drift = abs(curr_dist - self._initial_grasp_offset)
                if drift > self.max_transit_slip:
                    self.fail(f"OBJECT_SLIPPAGE_FAILURE: Object slipped by {drift*1000:.1f}mm during transit")
                    return False

            # Check if dropped to table prematurely
            if lift_height < 0.02:
                self.fail("OBJECT_SLIPPAGE_FAILURE: Object dropped onto table during transit")
                return False

            return True

        return True

    def verify_placement(self, obj_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple[bool, float]:
        """
        Verify that object is seated securely within the goal zone circle.
        """
        h_dist = float(np.linalg.norm(obj_pos[:2] - goal_pos[:2]))
        obj_z = obj_pos[2]
        is_on_table = (obj_z - self.table_z) <= 0.035

        if h_dist <= self.goal_radius and is_on_table:
            self.placement_verified = True
            return True, h_dist
        else:
            self.placement_verified = False
            return False, h_dist

    @property
    def is_terminal(self) -> bool:
        """True if in COMPLETED or FAILED state."""
        return self.state in [TaskState.COMPLETED, TaskState.FAILED]

    @property
    def is_success(self) -> bool:
        """True strictly if task completed successfully."""
        return self.state == TaskState.COMPLETED
