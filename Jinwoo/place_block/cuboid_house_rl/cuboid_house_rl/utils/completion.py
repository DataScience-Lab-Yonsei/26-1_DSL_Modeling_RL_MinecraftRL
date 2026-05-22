"""
Completion ratio tracking and stuck detection.
"""
import numpy as np
from cuboid_house_rl.config import (
    OAK_PLANKS, AIR, TOTAL_BLOCKS,
    STUCK_PATIENCE, STUCK_MIN_DELTA,
    SUBTASK_FLOOR, SUBTASK_WALLS, SUBTASK_CEILING, SUBTASK_DONE,
    DOOR_X, DOOR_Z, DOOR_Y_BOTTOM, DOOR_Y_TOP,
)


class CompletionTracker:
    """
    Tracks construction progress across all phases.
    Compares current world state against the blueprint.
    """

    def __init__(self, blueprint: np.ndarray, phase_positions: dict):
        """
        Args:
            blueprint: 3D array of target block types.
            phase_positions: dict with 'floor', 'walls', 'ceiling' keys,
                             each containing list of (x,y,z) tuples.
        """
        self.blueprint = blueprint
        self.phase_positions = phase_positions
        self.door_positions = [
            (DOOR_X, DOOR_Y_BOTTOM, DOOR_Z),
            (DOOR_X, DOOR_Y_TOP, DOOR_Z),
        ]

        # Precompute counts for efficiency
        self.floor_total = len(phase_positions["floor"])
        self.wall_total = len(phase_positions["walls"])
        self.ceiling_total = len(phase_positions["ceiling"])

    def compute(self, world: np.ndarray) -> dict:
        """
        Compute completion ratios for each phase.

        Args:
            world: 3D array of current block types in the world.

        Returns:
            dict with keys:
                'floor_ratio': 0.0-1.0
                'wall_ratio': 0.0-1.0
                'ceiling_ratio': 0.0-1.0
                'door_ratio': 0.0 or 1.0
                'total_completion': 0.0-1.0 (global progress)
                'total_correct': int (number of correctly placed blocks)
        """
        floor_correct = self._count_correct(world, self.phase_positions["floor"])
        wall_correct = self._count_correct(world, self.phase_positions["walls"])
        ceiling_correct = self._count_correct(world, self.phase_positions["ceiling"])

        # Door: both positions must be AIR
        door_correct = all(
            world[x, y, z] == AIR for x, y, z in self.door_positions
        )

        total_correct = floor_correct + wall_correct + ceiling_correct

        return {
            "floor_ratio": floor_correct / self.floor_total if self.floor_total > 0 else 1.0,
            "wall_ratio": wall_correct / self.wall_total if self.wall_total > 0 else 1.0,
            "ceiling_ratio": ceiling_correct / self.ceiling_total if self.ceiling_total > 0 else 1.0,
            "door_ratio": 1.0 if door_correct else 0.0,
            "total_completion": total_correct / TOTAL_BLOCKS,
            "total_correct": total_correct,
        }

    def _count_correct(self, world: np.ndarray, positions: list) -> int:
        """Count positions where world matches blueprint (OAK_PLANKS)."""
        count = 0
        for x, y, z in positions:
            if world[x, y, z] == OAK_PLANKS:
                count += 1
        return count

    def get_current_subtask(self, completion: dict) -> int:
        """
        Rule-based subtask controller.
        Returns the current subtask ID based on completion ratios.
        """
        if completion["floor_ratio"] < 1.0:
            return SUBTASK_FLOOR
        elif completion["wall_ratio"] < 1.0:
            return SUBTASK_WALLS
        elif completion["ceiling_ratio"] < 1.0:
            return SUBTASK_CEILING
        else:
            return SUBTASK_DONE


class StuckDetector:
    """
    Detects when the agent is stuck (no progress for too many steps).

    Monitors total_completion and counts consecutive steps where
    the change (delta) is below a threshold.
    """

    def __init__(self, patience: int = STUCK_PATIENCE, min_delta: float = STUCK_MIN_DELTA):
        """
        Args:
            patience: number of consecutive no-progress steps before stuck.
            min_delta: minimum change in total_completion to count as progress.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self):
        """Reset the detector for a new episode."""
        self.steps_without_progress = 0
        self.last_completion = 0.0

    def update(self, total_completion: float) -> bool:
        """
        Update with current total_completion.

        Args:
            total_completion: float 0.0-1.0, overall construction progress.

        Returns:
            True if stuck (should terminate episode), False otherwise.
        """
        delta = total_completion - self.last_completion

        if delta >= self.min_delta:
            # Progress made — reset counter
            self.steps_without_progress = 0
            self.last_completion = total_completion
        else:
            # No progress — increment counter
            self.steps_without_progress += 1

        return self.is_stuck

    @property
    def is_stuck(self) -> bool:
        """Whether the agent is considered stuck."""
        return self.steps_without_progress >= self.patience
