"""
Stuck detection utility.

CompletionTracker removed in V3 — env uses placed_blocks sets directly.
"""
from cuboid_house_rl.config import STUCK_PATIENCE, STUCK_MIN_DELTA


class StuckDetector:
    """
    Detects when the agent is stuck (no progress for too many steps).
    """

    def __init__(self, patience: int = STUCK_PATIENCE, min_delta: float = STUCK_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self):
        self.steps_without_progress = 0
        self.last_completion = 0.0

    def update(self, total_completion: float) -> bool:
        delta = total_completion - self.last_completion
        if delta >= self.min_delta:
            self.steps_without_progress = 0
            self.last_completion = total_completion
        else:
            self.steps_without_progress += 1
        return self.is_stuck

    @property
    def is_stuck(self) -> bool:
        return self.steps_without_progress >= self.patience
