"""
Scripted Expert — unified dispatcher.

Chains 5 stage-specific experts:
    floor(0) → wall(1) → door(2) → ceiling(3) → looking(4)

Also provides target generation functions for blueprint creation.
"""
import random
from typing import List, Tuple, Optional

import numpy as np

from cuboid_house_rl.config import (
    AIR, OAK_PLANKS, PLANKS_SLOT, NUM_ACTION_DIMS,
    STAGE_FLOOR, STAGE_WALL, STAGE_DOOR, STAGE_CEILING, STAGE_LOOKING,
)
from cuboid_house_rl.expert.floor_expert import FloorExpert
from cuboid_house_rl.expert.wall_expert import WallExpert
from cuboid_house_rl.expert.door_expert import DoorExpert
from cuboid_house_rl.expert.ceiling_expert import CeilingExpert
from cuboid_house_rl.expert.movement import noop

STAGE_NAMES = ["floor", "wall", "door", "ceiling", "looking"]


class ScriptedExpert:
    """
    Top-level expert that chains stage-specific sub-experts.

    Usage:
        expert = ScriptedExpert(stage="all")
        expert.reset()
        while not expert.is_done():
            action = expert.get_action(env)
            env.step(action)
    """

    def __init__(self, stage: str = "all",
                 width: int = None, depth: int = None):
        """
        Args:
            stage: "floor", "walls", "all"
            width: floor width (5~10). None = random.
            depth: floor depth (5~10). None = random.
        """
        self.stage = stage
        self.width = width
        self.depth = depth

        self._floor_expert: Optional[FloorExpert] = None
        self._wall_expert: Optional[WallExpert] = None
        self._door_expert: Optional[DoorExpert] = None
        self._ceiling_expert: Optional[CeilingExpert] = None
        self._current_stage = "floor"
        self._current_stage_id = STAGE_FLOOR

        self.reset()

    def reset(self):
        """Reset for new episode."""
        w = self.width if self.width is not None else random.randint(5, 10)
        d = self.depth if self.depth is not None else random.randint(5, 10)

        self._floor_expert = FloorExpert(width=w, depth=d)
        self._wall_expert = None
        self._door_expert = None
        self._ceiling_expert = None
        self._current_stage = "floor"
        self._current_stage_id = STAGE_FLOOR
        self._stage_just_done = None  # set when a stage completes
        self._ceiling_done_logged = False
        self._door_open_logged = False

        # Expose for external access
        self.actual_width = w
        self.actual_depth = d

    @property
    def origin_set(self) -> bool:
        return self._floor_expert.origin_set

    @property
    def origin_x(self) -> Optional[int]:
        return self._floor_expert.origin_x

    @property
    def origin_z(self) -> Optional[int]:
        return self._floor_expert.origin_z

    @property
    def current_stage_id(self) -> int:
        return self._current_stage_id

    @property
    def current_stage_name(self) -> str:
        return STAGE_NAMES[self._current_stage_id]

    def is_done(self) -> bool:
        if self.stage == "floor":
            return self._floor_expert.is_done()
        elif self.stage == "walls":
            if self._wall_expert is None:
                return False
            return self._wall_expert.is_done()
        elif self.stage == "all":
            if self._ceiling_expert is None:
                return False
            return self._ceiling_expert.is_done()
        return True

    def get_current_target(self):
        if self._current_stage == "floor":
            return self._floor_expert.get_current_target()
        elif self._current_stage == "walls" and self._wall_expert:
            return self._wall_expert.get_current_target()
        return None

    def get_remaining_targets(self) -> list:
        """Get all remaining targets as (x, y, z) tuples."""
        targets = []
        if self._current_stage == "floor":
            targets.extend(self._floor_expert.get_remaining_targets())
            if self.stage in ("walls", "all") and self._floor_expert.origin_set:
                if self._wall_expert is None:
                    self._init_wall_expert()
                targets.extend(self._wall_expert.get_remaining_targets())
        elif self._current_stage == "walls" and self._wall_expert:
            targets.extend(self._wall_expert.get_remaining_targets())
        return targets

    def get_action(self, env) -> np.ndarray:
        """Dispatch to current stage expert."""

        if self._current_stage == "floor":
            if self._floor_expert.is_done():
                self._stage_just_done = "floor"
                if self.stage in ("walls", "all"):
                    self._transition_to_walls()
                    return noop()
                else:
                    return noop()
            return self._floor_expert.get_action(env)

        elif self._current_stage == "walls":
            if self._wall_expert is None:
                return noop()
            if self._wall_expert.is_done():
                self._stage_just_done = "wall"
                if self.stage == "all":
                    self._transition_to_door()
                    return noop()
                return noop()
            return self._wall_expert.get_action(env)

        elif self._current_stage == "door":
            if self._door_expert is None:
                return noop()
            if self._door_expert.is_done():
                self._stage_just_done = "door"
                self._transition_to_ceiling()
                return noop()
            return self._door_expert.get_action(env)

        elif self._current_stage == "ceiling":
            if self._ceiling_expert is None:
                return noop()
            # Detect ceiling done (before looking sequence)
            if (self._ceiling_expert.state == CeilingExpert.MOVE_TO_DOOR_X and
                    not hasattr(self, '_ceiling_done_logged')):
                self._stage_just_done = "ceiling"
                self._ceiling_done_logged = True
            # Detect door opened
            if (self._ceiling_expert.state == CeilingExpert.WALK_OUT and
                    not hasattr(self, '_door_open_logged')):
                self._stage_just_done = "door_open"
                self._door_open_logged = True
            if self._ceiling_expert.is_done():
                self._stage_just_done = "looking"
                return noop()
            return self._ceiling_expert.get_action(env)

        return noop()

    def _init_wall_expert(self):
        """Create wall expert using floor's origin and dimensions."""
        self._wall_expert = WallExpert(
            origin_x=self._floor_expert.origin_x,
            origin_z=self._floor_expert.origin_z,
            width=self._floor_expert.width,
            depth=self._floor_expert.depth,
        )

    def _transition_to_walls(self):
        """Switch from floor stage to wall stage."""
        self._init_wall_expert()
        self._current_stage = "walls"
        self._current_stage_id = STAGE_WALL
        print(f"[Expert] Floor done → Wall "
              f"(origin=({self.origin_x},{self.origin_z}), "
              f"size={self.actual_width}x{self.actual_depth})")

    def _transition_to_door(self):
        """Switch from wall stage to door stage."""
        self._door_expert = DoorExpert(
            origin_x=self._floor_expert.origin_x,
            origin_z=self._floor_expert.origin_z,
            width=self._floor_expert.width,
            depth=self._floor_expert.depth,
        )
        self._current_stage = "door"
        self._current_stage_id = STAGE_DOOR
        print(f"[Expert] Wall done → Door")

    def _transition_to_ceiling(self):
        """Switch from door stage to ceiling stage (includes looking)."""
        prev_hotbar = getattr(self._door_expert, '_hotbar',
                              getattr(self._wall_expert, '_hotbar', 0))
        self._ceiling_expert = CeilingExpert(
            origin_x=self._floor_expert.origin_x,
            origin_z=self._floor_expert.origin_z,
            width=self._floor_expert.width,
            depth=self._floor_expert.depth,
            initial_hotbar=prev_hotbar,
            door_x=self._door_expert.door_x,
        )
        self._current_stage = "ceiling"
        self._current_stage_id = STAGE_CEILING
        print(f"[Expert] Door done → Ceiling")

    def update_stage_id(self):
        """Update stage_id based on ceiling expert's finish sequence."""
        if (self._current_stage == "ceiling" and
                self._ceiling_expert is not None):
            # Ceiling finish states = looking stage
            looking_states = {
                CeilingExpert.MOVE_TO_DOOR_X,
                CeilingExpert.FACE_DOOR,
                CeilingExpert.WALK_TO_DOOR,
                CeilingExpert.OPEN_DOOR,
                CeilingExpert.WALK_OUT,
                CeilingExpert.TURN_AROUND,
                CeilingExpert.LOOK_AT_HOUSE,
                CeilingExpert.WAIT_FINISH,
            }
            if self._ceiling_expert.state in looking_states:
                self._current_stage_id = STAGE_LOOKING
            else:
                self._current_stage_id = STAGE_CEILING
