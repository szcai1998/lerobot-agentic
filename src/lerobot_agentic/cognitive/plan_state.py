import threading

import numpy as np

from lerobot_agentic.cognitive.schemas import SpatialGroundingPlan


class LatestFrameBuffer:
    """
    Thread-safe frame buffer that completely decouples simulation rendering from
    the asynchronous cloud supervisory thread.

    Only the simulation thread ever touches MuJoCo rendering contexts and pushes
    immutable NumPy copies. The supervisor thread reads the latest snapshot.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.rgb_overhead: np.ndarray | None = None
        self.depth_overhead: np.ndarray | None = None
        self.rgb_wrist: np.ndarray | None = None
        self.step_idx: int = 0

    def push(
        self,
        rgb_overhead: np.ndarray,
        depth_overhead: np.ndarray | None = None,
        rgb_wrist: np.ndarray | None = None,
        step_idx: int = 0
    ):
        with self._lock:
            self.rgb_overhead = rgb_overhead.copy()
            self.depth_overhead = depth_overhead.copy() if depth_overhead is not None else None
            self.rgb_wrist = rgb_wrist.copy() if rgb_wrist is not None else None
            self.step_idx = step_idx

    def get_latest(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
        with self._lock:
            if self.rgb_overhead is None:
                return None, None, None, 0
            return (
                self.rgb_overhead.copy(),
                self.depth_overhead.copy() if self.depth_overhead is not None else None,
                self.rgb_wrist.copy() if self.rgb_wrist is not None else None,
                self.step_idx
            )


class AtomicPlanState:
    """Thread-safe shared state container for asynchronous supervisory communication."""
    def __init__(self):
        self._lock = threading.Lock()
        self.plan: SpatialGroundingPlan | None = None
        self.target_pos_world: np.ndarray | None = None
        self.dest_pos_world: np.ndarray | None = None
        self.subgoal_id: int = 0
        self.version: int = 0
        self.latest_replan_id: int = 0

    def update(
        self,
        plan: SpatialGroundingPlan,
        target_pos_world: np.ndarray | None = None,
        dest_pos_world: np.ndarray | None = None,
        subgoal_id: int = 0
    ):
        with self._lock:
            self.plan = plan
            self.target_pos_world = target_pos_world
            self.dest_pos_world = dest_pos_world
            self.subgoal_id = subgoal_id
            self.version += 1
            if plan.requires_replanning:
                if plan.replan_id > 0:
                    self.latest_replan_id = plan.replan_id
                else:
                    self.latest_replan_id += 1

    def get_snapshot(self) -> tuple[SpatialGroundingPlan | None, np.ndarray | None, np.ndarray | None, int, int]:
        with self._lock:
            return self.plan, self.target_pos_world, self.dest_pos_world, self.subgoal_id, self.version

    def check_and_consume_replan(self, last_consumed_id: int) -> tuple[bool, int]:
        """
        Edge-triggered recovery detection: Returns True only if a new recovery event
        occurred that has not yet been consumed by the control thread.
        """
        with self._lock:
            if self.latest_replan_id > last_consumed_id:
                return True, self.latest_replan_id
            return False, last_consumed_id
