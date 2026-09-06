import threading

import numpy as np

from lerobot_agentic.cognitive.schemas import SpatialGroundingPlan


class AtomicPlanState:
    """Thread-safe shared state container for asynchronous supervisory communication."""
    def __init__(self):
        self._lock = threading.Lock()
        self.plan: SpatialGroundingPlan | None = None
        self.target_pos_world: np.ndarray | None = None
        self.dest_pos_world: np.ndarray | None = None
        self.subgoal_id: int = 0
        self.version: int = 0

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

    def get_snapshot(self) -> tuple[SpatialGroundingPlan | None, np.ndarray | None, np.ndarray | None, int, int]:
        with self._lock:
            return self.plan, self.target_pos_world, self.dest_pos_world, self.subgoal_id, self.version
