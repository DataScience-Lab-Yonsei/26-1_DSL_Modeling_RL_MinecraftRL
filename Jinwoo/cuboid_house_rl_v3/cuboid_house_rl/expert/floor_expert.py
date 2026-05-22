"""
Floor Expert — Stage 1.

Places floor blocks (y=1) on a random-sized rectangle (5~10 x 5~10).

Scenario:
    Phase 0: Walk random 3 ticks → pick direction → aim at ground 2 blocks away → place first block → origin set
    Phase 1: Align to (ox+0.5, oz-0.5) → Row 0 from ground (y=1), +X direction
    Phase 2: Jump onto last floor block → now on y=2
    Phase 3: Rows 1~(depth-1) from atop floor blocks (y=2), serpentine

All movement uses precision sneak-based positioning.
Yaw is kept at 0 (+Z direction) throughout.
"""
import math
import random
import numpy as np
from typing import Optional, Tuple, List

from cuboid_house_rl.config import (
    AIR, OAK_PLANKS, PLANKS_SLOT, SLOT_AXE,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    NUM_ACTION_DIMS,
)
from cuboid_house_rl.expert.movement import (
    noop, place_action, jump_forward_action,
    move_to_xz_action, move_x_action, move_z_action,
    fix_yaw_action,
    aim_action, is_aimed,
    check_raycast_hits_block,
    raycast_aim_action, fine_aim_to_center,
    error_to_camera_idx, compute_aim_error,
    MovementController,
    FINE_THRESHOLD, AIM_TOLERANCE_RAD,
)


class FloorExpert:
    """
    State machine for placing floor blocks.

    Floor size is randomized each episode: width (5~10), depth (5~10).
    Origin is determined by the first block placement.
    """

    # Phases
    PH0_WALK_RANDOM = "ph0_walk_random"
    PH0_STOP = "ph0_stop"
    PH0_PICK_DIRECTION = "ph0_pick_direction"
    PH0_FIX_YAW = "ph0_fix_yaw"
    PH0_AIM = "ph0_aim"
    PH0_VERIFY = "ph0_verify"
    PH0_PLACE = "ph0_place"
    PH0_WAIT = "ph0_wait"
    PH0_ALIGN = "ph0_align"

    PH1_FIX_YAW = "ph1_fix_yaw"
    PH1_MOVE_TO_TARGET = "ph1_move_to_target"
    PH1_AIM = "ph1_aim"
    PH1_VERIFY = "ph1_verify"
    PH1_PLACE = "ph1_place"
    PH1_WAIT = "ph1_wait"

    PH2_FACE_Z = "ph2_face_z"
    PH2_WALK_TO_BLOCK = "ph2_walk_to_block"
    PH2_JUMP = "ph2_jump"
    PH2_CHECK_LANDED = "ph2_check_landed"

    PH3_MOVE_TO_NEXT_ROW = "ph3_move_to_next_row"
    PH3_MOVE_TO_TARGET = "ph3_move_to_target"
    PH3_AIM = "ph3_aim"
    PH3_VERIFY = "ph3_verify"
    PH3_PLACE = "ph3_place"
    PH3_WAIT = "ph3_wait"

    # Break wrong block states
    BREAK_HIT = "break_hit"

    DONE = "done"

    def __init__(self, width: int = None, depth: int = None):
        """
        Args:
            width: floor width in X direction (5~10). None = random.
            depth: floor depth in Z direction (5~10). None = random.
        """
        self.width = width if width is not None else random.randint(5, 10)
        self.depth = depth if depth is not None else random.randint(5, 10)

        # Origin — set when first block is placed
        self.origin_x: Optional[int] = None
        self.origin_z: Optional[int] = None
        self.origin_set = False

        # State machine
        self.state = self.PH0_WALK_RANDOM
        self.walk_random_ticks = 0
        self.wait_counter = 0
        self.jump_attempts = 0

        # Movement controller (stateful, with stuck/jump detection)
        self.mover = MovementController()

        # Hotbar: current planks slot
        self._hotbar = PLANKS_SLOT

        # Stuck detection for move phases without MovementController
        self._move_prev_x = None
        self._move_stuck_ticks = 0
        self._MOVE_STUCK_LIMIT = 15  # force aim after this many stuck ticks

        # Phase 0 direction choice
        self.ph0_direction = None  # "north", "south", "east", "west"
        self.ph0_target_block = None  # (x, z) of first block on ground
        self.ph0_target_yaw = 0.0

        # Gaze lock: once aim confirmed in ph1 row, keep camera still
        self._ph1_gaze_locked = False
        # Per-row gaze lock in ph3
        self._ph3_gaze_locked = False

        # Raycast confirm counter: require N consecutive hits before placing
        self._RAYCAST_CONFIRM_NEEDED = 3
        self._raycast_confirm_count = 0

        # Break wrong block: track which block to break and return state
        self._break_target = None       # (x, y, z) of wrong block to break
        self._break_return_state = None  # state to return to after breaking

        # Targets — generated after origin is set
        self.targets: List[Tuple[int, int, int]] = []
        self.target_idx = 0

        # Current row tracking for serpentine
        self.current_row = 0

    def reset(self):
        """Reset for new episode. Randomize size."""
        self.width = random.randint(5, 10)
        self.depth = random.randint(5, 10)
        self.origin_x = None
        self.origin_z = None
        self.origin_set = False
        self.state = self.PH0_WALK_RANDOM
        self.walk_random_ticks = 0
        self.wait_counter = 0
        self.jump_attempts = 0
        self.mover = MovementController()
        self._move_prev_x = None
        self._move_stuck_ticks = 0
        self.ph0_direction = None
        self.ph0_target_block = None
        self.ph0_target_yaw = 0.0
        self._ph1_gaze_locked = False
        self._ph3_gaze_locked = False
        self._raycast_confirm_count = 0
        self._break_target = None
        self._break_return_state = None
        self._hotbar = PLANKS_SLOT
        self.targets = []
        self.target_idx = 0
        self.current_row = 0

    def is_done(self) -> bool:
        return self.state == self.DONE

    def _check_hotbar(self, env):
        """If current slot is empty, switch to next slot with planks."""
        count = env.get_hotbar_planks_count(self._hotbar)
        if count <= 0:
            new_slot = env.find_planks_slot()
            if new_slot >= 0 and new_slot != self._hotbar:
                pass  # silent hotbar switch
                self._hotbar = new_slot

    def _noop(self):
        return noop(self._hotbar)

    def _place(self):
        return place_action(self._hotbar)

    def get_current_target(self):
        if self.target_idx < len(self.targets):
            return self.targets[self.target_idx]
        return None

    def get_remaining_targets(self) -> list:
        return self.targets[self.target_idx:]

    def _generate_targets(self):
        """Generate serpentine floor target list after origin is set."""
        ox, oz = self.origin_x, self.origin_z
        targets = []
        for i in range(self.depth):
            z = oz + i
            if i % 2 == 0:
                xs = range(ox, ox + self.width)       # +X
            else:
                xs = range(ox + self.width - 1, ox - 1, -1)  # -X
            for x in xs:
                targets.append((x, 1, z))
        self.targets = targets

    # ==================================================================
    # Main dispatch
    # ==================================================================

    def get_action(self, env) -> np.ndarray:
        """Compute next action based on current state."""
        # Check if current hotbar slot is empty → switch
        self._check_hotbar(env)

        action = self._dispatch(env)
        action[ACT_HOTBAR] = self._hotbar
        return action

    def _dispatch(self, env) -> np.ndarray:
        # ---- Phase 0: Origin determination ----

        if self.state == self.PH0_WALK_RANDOM:
            return self._ph0_walk_random(env)

        if self.state == self.PH0_STOP:
            self.state = self.PH0_PICK_DIRECTION
            return noop()

        if self.state == self.PH0_PICK_DIRECTION:
            return self._ph0_pick_direction(env)

        if self.state == self.PH0_FIX_YAW:
            return self._ph0_fix_yaw(env)

        if self.state == self.PH0_AIM:
            return self._ph0_aim(env)

        if self.state == self.PH0_VERIFY:
            return self._ph0_verify(env)

        if self.state == self.PH0_PLACE:
            return self._ph0_place(env)

        if self.state == self.PH0_WAIT:
            return self._wait(env, next_state=self.PH0_ALIGN)

        if self.state == self.PH0_ALIGN:
            return self._ph0_align(env)

        # ---- Phase 1: Row 0 from ground ----

        if self.state == self.PH1_FIX_YAW:
            return self._ph1_fix_yaw(env)

        if self.state == self.PH1_MOVE_TO_TARGET:
            return self._ph1_move_to_target(env)

        if self.state == self.PH1_AIM:
            return self._ph1_aim(env)

        if self.state == self.PH1_VERIFY:
            return self._ph1_verify(env)

        if self.state == self.PH1_PLACE:
            return self._ph1_place(env)

        if self.state == self.PH1_WAIT:
            return self._ph1_wait(env)

        # ---- Phase 2: Jump onto floor ----

        if self.state == self.PH2_FACE_Z:
            return self._ph2_face_z(env)

        if self.state == self.PH2_WALK_TO_BLOCK:
            return self._ph2_walk_to_block(env)

        if self.state == self.PH2_JUMP:
            return self._ph2_jump(env)

        if self.state == self.PH2_CHECK_LANDED:
            return self._ph2_check_landed(env)

        # ---- Phase 3: Rows 1+ from atop floor ----

        if self.state == self.PH3_MOVE_TO_NEXT_ROW:
            return self._ph3_move_to_next_row(env)

        if self.state == self.PH3_MOVE_TO_TARGET:
            return self._ph3_move_to_target(env)

        if self.state == self.PH3_AIM:
            return self._ph3_aim(env)

        if self.state == self.PH3_VERIFY:
            return self._ph3_verify(env)

        if self.state == self.PH3_PLACE:
            return self._ph3_place(env)

        if self.state == self.PH3_WAIT:
            return self._ph3_wait(env)

        # ---- Break wrong block ----

        if self.state == self.BREAK_HIT:
            return self._break_hit(env)

        return noop()

    # ==================================================================
    # Phase 0: First block (origin)
    # ==================================================================

    def _ph0_walk_random(self, env) -> np.ndarray:
        """Walk randomly for 3 ticks."""
        self.walk_random_ticks += 1
        if self.walk_random_ticks >= 3:
            self.state = self.PH0_STOP
            return noop()

        action = noop()
        action[ACT_FWD_BACK] = random.choice([0, 1, 2])
        action[ACT_LEFT_RIGHT] = random.choice([0, 1, 2])
        return action

    def _ph0_pick_direction(self, env) -> np.ndarray:
        """Pick direction for first block — fixed north (+Z) for now."""
        ax = int(math.floor(env.agent_x))
        az = int(math.floor(env.agent_z))

        # Fixed north: agent already faces +Z, no yaw change needed
        direction = "north"
        self.ph0_direction = direction
        self.ph0_target_block = (ax, az + 2)
        self.ph0_target_yaw = 0.0  # face +Z

        self.state = self.PH0_FIX_YAW
        return noop()

    def _ph0_fix_yaw(self, env) -> np.ndarray:
        """Fix yaw to face the chosen direction."""
        action, fixed = fix_yaw_action(env.agent_yaw, self.ph0_target_yaw)
        if fixed:
            self.state = self.PH0_AIM
        return action

    def _ph0_aim(self, env) -> np.ndarray:
        """Aim at the ground block 2 units away.
        Requires 3 consecutive confirmations (block + center).
        If block hit but off-center: fine-tune pitch.
        """
        tx, tz = self.ph0_target_block

        # Full check: block hit + pitch centered
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0), check_center=True):
            self._raycast_confirm_count += 1
            if self._raycast_confirm_count >= self._RAYCAST_CONFIRM_NEEDED:
                self._raycast_confirm_count = 0
                return self._ph0_place(env)
            return noop()

        self._raycast_confirm_count = 0

        # Block hit but pitch off-center → fine-tune pitch
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0), check_center=False):
            return fine_aim_to_center(env, (tx, 0, tz))

        return raycast_aim_action(env, (tx, 0, tz))

    def _ph0_verify(self, env) -> np.ndarray:
        """Fallback verify state (normally not reached — ph0_aim places directly)."""
        tx, tz = self.ph0_target_block
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0)):
            return self._ph0_place(env)
        self.state = self.PH0_AIM
        return noop()

    def _ph0_place(self, env) -> np.ndarray:
        """Place first block — this sets the origin."""
        tx, tz = self.ph0_target_block
        self.origin_x = tx
        self.origin_z = tz
        self.origin_set = True
        self._generate_targets()

        # Sync origin to env immediately (don't wait for block_event)
        if hasattr(env, 'set_origin') and not env.origin_set:
            env.set_origin(tx, tz)

        # First target is (ox, 1, oz) which we're placing now
        self.target_idx = 1  # skip first block, already placed

        self.state = self.PH0_WAIT
        self.wait_counter = 2
        return self._place()

    def _ph0_align(self, env) -> np.ndarray:
        """
        After first block: align to (ox+0.5, oz-1.0).
        First fix yaw to +Z, then move to position.
        """
        # Fix yaw to +Z first
        action, fixed = fix_yaw_action(env.agent_yaw, 0.0)
        if not fixed:
            return action

        # Move to standing position for Row 0
        target_x = self.origin_x + 0.5
        target_z = self.origin_z - 1.0

        action, arrived = self.mover.move_to(
            env, target_x, target_z
        )
        if arrived:
            self.mover.reset()
            self.state = self.PH1_FIX_YAW
        return action

    # ==================================================================
    # Phase 1: Row 0 from ground
    # ==================================================================

    def _ph1_fix_yaw(self, env) -> np.ndarray:
        """Ensure yaw is +Z before starting row."""
        action, fixed = fix_yaw_action(env.agent_yaw, 0.0)
        if fixed:
            self.state = self.PH1_MOVE_TO_TARGET
        return action

    def _ph1_move_to_target(self, env) -> np.ndarray:
        """Move to (tx+0.5, oz-1.5) for current target."""
        if self.target_idx >= len(self.targets):
            self.state = self.DONE
            return noop()

        target = self.targets[self.target_idx]
        tx, ty, tz = target

        # Skip already-placed blocks
        if (tx, ty, tz) in getattr(env, 'correct_blocks', set()) or \
           (tx, ty, tz) in getattr(env, 'placed_blocks', set()):
            self.target_idx += 1
            return noop()

        # Check if this target is still in Row 0
        if tz != self.origin_z:
            # Row 0 done → Phase 2 (jump)
            self.state = self.PH2_FACE_Z
            return noop()

        # Standing position: (tx+0.5, oz-1.0) — correct both X and Z
        target_x = tx + 0.5
        target_z = self.origin_z - 1.0
        action, arrived = move_to_xz_action(env.agent_x, env.agent_z, target_x, target_z)
        # Must be within 0.5 of target x center before aiming
        dx = abs(env.agent_x - target_x)
        if arrived and dx <= 0.5:
            self._move_prev_x = None
            self._move_stuck_ticks = 0
            self.state = self.PH1_AIM
            return action

        # Stuck detection: if x barely changes, force aim from current position
        if self._move_prev_x is not None:
            if abs(env.agent_x - self._move_prev_x) < 0.01:
                self._move_stuck_ticks += 1
            else:
                self._move_stuck_ticks = 0
        self._move_prev_x = env.agent_x

        if self._move_stuck_ticks >= self._MOVE_STUCK_LIMIT:
            self._move_prev_x = None
            self._move_stuck_ticks = 0
            self.state = self.PH1_AIM  # aim from current position
        return action

    def _ph1_aim(self, env) -> np.ndarray:
        """Aim at ground top face for current target.

        Requires 3 consecutive confirmations (block + center) before placing.
        If block hit but off-center: fine-tune pitch.
        """
        target = self.targets[self.target_idx]
        tx, ty, tz = target

        # Full check: block hit + pitch centered
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0), check_center=True):
            self._raycast_confirm_count += 1
            if self._raycast_confirm_count >= self._RAYCAST_CONFIRM_NEEDED:
                self._raycast_confirm_count = 0
                self._ph1_gaze_locked = True
                self.state = self.PH1_WAIT
                self.wait_counter = 2
                return self._place()
            return noop()

        self._raycast_confirm_count = 0

        # Block hit but pitch off-center → fine-tune pitch only
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0), check_center=False):
            return fine_aim_to_center(env, (tx, 0, tz))

        if self._ph1_gaze_locked:
            self._ph1_gaze_locked = False

        return raycast_aim_action(env, (tx, 0, tz))

    def _ph1_verify(self, env) -> np.ndarray:
        """Fallback verify state (normally not reached — ph1_aim places directly)."""
        target = self.targets[self.target_idx]
        tx, ty, tz = target
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0)):
            self.state = self.PH1_WAIT
            self.wait_counter = 2
            return self._place()
        self.state = self.PH1_AIM
        return noop()

    def _ph1_place(self, env) -> np.ndarray:
        self.state = self.PH1_WAIT
        self.wait_counter = 2
        return self._place()

    def _ph1_wait(self, env) -> np.ndarray:
        """After placing, verify with raycast that block actually exists."""
        self.wait_counter -= 1
        if self.wait_counter > 0:
            return noop()

        # Verify: raycast should now hit the PLACED block (tx, 1, tz)
        target = self.targets[self.target_idx]
        tx, ty, tz = target
        if not check_raycast_hits_block(env, (tx, 1, tz), expected_face=None, check_center=False):
            # Check if a wrong block was placed (raycast hits y=1 but wrong x/z)
            hit = env._cg_obs_extractor.extract_raycast(env._cg_obs) if env._cg_obs else None
            if hit and int(hit["position"][1]) == 1:
                # Wrong position — break it first
                wp = tuple(int(v) for v in hit["position"])
                print(f"  [expert] ph1_wait: WRONG block at {wp}, expected ({tx},1,{tz}), breaking")
                self._break_target = wp
                self._break_return_state = self.PH1_AIM
                self._ph1_gaze_locked = False
                self._raycast_confirm_count = 0
                self.state = self.BREAK_HIT
                return noop()
            else:
                # Block not placed at all — retry
                print(f"  [expert] ph1_wait: block NOT at ({tx},1,{tz}), retrying")
                self._ph1_gaze_locked = False
                self._raycast_confirm_count = 0
                self.state = self.PH1_AIM
                return noop()

        self.target_idx += 1

        if self.target_idx >= len(self.targets):
            self.state = self.DONE
            return noop()

        # Check if next target is still row 0
        next_tz = self.targets[self.target_idx][2]
        if next_tz != self.origin_z:
            self._ph1_gaze_locked = False  # leaving ph1
            self.state = self.PH2_FACE_Z
        else:
            self.state = self.PH1_MOVE_TO_TARGET
        return noop()

    # ==================================================================
    # Phase 2: Jump onto floor blocks
    # ==================================================================

    def _ph2_face_z(self, env) -> np.ndarray:
        """Face +Z before walking toward floor blocks."""
        action, fixed = fix_yaw_action(env.agent_yaw, 0.0)
        if fixed:
            self.state = self.PH2_WALK_TO_BLOCK
        return action

    def _ph2_walk_to_block(self, env) -> np.ndarray:
        """Walk forward toward the floor block edge.

        CraftGround block at z=N occupies physical space z=(N-1)~z=N,
        so the south face of floor blocks at z=oz is at z=oz-1.
        Agent (width 0.3) can stand up to z=oz-1.3 before collision.
        Walk to oz-1.2 to be right next to the block before jumping.
        """
        target_z = self.origin_z - 0.7
        action, arrived = move_z_action(env.agent_z, target_z)
        if arrived:
            self.state = self.PH2_JUMP
            self.jump_attempts = 0
        return action

    def _ph2_jump(self, env) -> np.ndarray:
        """Jump + forward to get on top of floor blocks."""
        self.jump_attempts += 1
        if self.jump_attempts > 20:
            # Emergency: skip to phase 3 anyway, agent might be on blocks
            print(f"  [expert] ph2_jump: giving up after {self.jump_attempts} attempts, y={env.agent_y:.2f}")
            self.current_row = 1
            self.state = self.PH3_MOVE_TO_NEXT_ROW
            return noop()
        # After sending jump, check if we landed next tick
        self.state = self.PH2_CHECK_LANDED
        return jump_forward_action()

    def _ph2_check_landed(self, env) -> np.ndarray:
        """Check if agent landed on floor blocks (y ≈ 2)."""
        if env.agent_y >= 1.8:  # successfully on floor block
            self.current_row = 1
            self.state = self.PH3_MOVE_TO_NEXT_ROW
            return noop()

        # Not landed yet — try jumping again
        self.state = self.PH2_JUMP
        return noop()

    # ==================================================================
    # Phase 3: Rows 1+ from atop floor blocks
    # ==================================================================

    def _ph3_move_to_next_row(self, env) -> np.ndarray:
        """Move to the standing z for the current row."""
        if self.target_idx >= len(self.targets):
            self.state = self.DONE
            return noop()

        target = self.targets[self.target_idx]
        tx, ty, tz = target

        # Standing z: on the PREVIOUS row's floor blocks, close to target
        # tz-1 is the previous row block. Stand near its north edge
        # (closer to target) so pitch is steep enough to hit block center.
        stand_z = tz - 1 + 0.5  # = tz-0.5: near edge of prev row block

        # Also need correct x: first block of this row
        stand_x = tx + 0.5

        # Fix yaw first
        action, fixed = fix_yaw_action(env.agent_yaw, 0.0)
        if not fixed:
            return action

        action, arrived = self.mover.move_to(
            env, stand_x, stand_z
        )
        if arrived:
            self.mover.reset()
            self._ph3_gaze_locked = False  # new row → re-aim once
            self.state = self.PH3_MOVE_TO_TARGET
        return action

    def _ph3_move_to_target(self, env) -> np.ndarray:
        """Strafe to current target x, must reach at least block center."""
        if self.target_idx >= len(self.targets):
            self.state = self.DONE
            return noop()

        target = self.targets[self.target_idx]
        tx, ty, tz = target

        # Skip already-placed blocks
        if (tx, ty, tz) in getattr(env, 'correct_blocks', set()):
            self.target_idx += 1
            return noop()

        target_x = tx + 0.5
        # Must be within 0.5 of block center before aiming
        dx = abs(env.agent_x - target_x)
        action, arrived = move_x_action(env.agent_x, target_x)
        if arrived and dx <= 0.5:
            self.state = self.PH3_AIM
        return action

    def _ph3_aim(self, env) -> np.ndarray:
        """Aim at ground top face from y=2.

        Uses raycast feedback control for aiming.
        Requires 3 consecutive confirmations (block + center) before placing.
        If block hit but off-center: fine-tune pitch.
        """
        target = self.targets[self.target_idx]
        tx, ty, tz = target

        # Full check: block hit + pitch centered
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0), check_center=True):
            self._raycast_confirm_count += 1
            if self._raycast_confirm_count >= self._RAYCAST_CONFIRM_NEEDED:
                self._raycast_confirm_count = 0
                self._ph3_gaze_locked = True
                self.state = self.PH3_WAIT
                self.wait_counter = 2
                return self._place()
            return noop()

        self._raycast_confirm_count = 0

        # Block hit but pitch off-center → fine-tune pitch only
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0), check_center=False):
            return fine_aim_to_center(env, (tx, 0, tz))

        if self._ph3_gaze_locked:
            self._ph3_gaze_locked = False

        return raycast_aim_action(env, (tx, 0, tz))

    def _ph3_verify(self, env) -> np.ndarray:
        """Fallback verify state (normally not reached — ph3_aim places directly)."""
        target = self.targets[self.target_idx]
        tx, ty, tz = target
        if check_raycast_hits_block(env, (tx, 0, tz), expected_face=(0, 1, 0)):
            self.state = self.PH3_WAIT
            self.wait_counter = 2
            return self._place()
        self.state = self.PH3_AIM
        return noop()

    def _ph3_place(self, env) -> np.ndarray:
        self.state = self.PH3_WAIT
        self.wait_counter = 2
        return self._place()

    def _ph3_wait(self, env) -> np.ndarray:
        """After placing, verify with raycast that block actually exists."""
        self.wait_counter -= 1
        if self.wait_counter > 0:
            return noop()

        # Verify: raycast should now hit the PLACED block (tx, 1, tz)
        target = self.targets[self.target_idx]
        tx, ty, tz = target
        if not check_raycast_hits_block(env, (tx, 1, tz), expected_face=None, check_center=False):
            # Check if a wrong block was placed (raycast hits y=1 but wrong x/z)
            hit = env._cg_obs_extractor.extract_raycast(env._cg_obs) if env._cg_obs else None
            if hit and int(hit["position"][1]) == 1:
                wp = tuple(int(v) for v in hit["position"])
                print(f"  [expert] ph3_wait: WRONG block at {wp}, expected ({tx},1,{tz}), breaking")
                self._break_target = wp
                self._break_return_state = self.PH3_AIM
                self._ph3_gaze_locked = False
                self._raycast_confirm_count = 0
                self.state = self.BREAK_HIT
                return noop()
            else:
                print(f"  [expert] ph3_wait: block NOT at ({tx},1,{tz}), retrying")
                self._ph3_gaze_locked = False
                self._raycast_confirm_count = 0
                self.state = self.PH3_AIM
                return noop()

        self.target_idx += 1

        if self.target_idx >= len(self.targets):
            self.state = self.DONE
            return noop()

        # Check if we moved to a new row
        next_target = self.targets[self.target_idx]
        curr_target_z = self.targets[self.target_idx - 1][2]
        next_target_z = next_target[2]

        if next_target_z != curr_target_z:
            # New row — walk forward (gaze lock resets in _ph3_move_to_next_row)
            self.state = self.PH3_MOVE_TO_NEXT_ROW
        else:
            # Same row — just strafe, gaze stays locked
            self.state = self.PH3_MOVE_TO_TARGET

        return noop()

    # ==================================================================
    # Break wrong block
    # ==================================================================

    def _break_hit(self, env) -> np.ndarray:
        """Keep attacking the wrong block until raycast no longer hits it."""
        bx, by, bz = self._break_target

        # Check if block is gone
        hit = env._cg_obs_extractor.extract_raycast(env._cg_obs) if env._cg_obs else None
        if hit is None or (int(hit["position"][0]), int(hit["position"][1]), int(hit["position"][2])) != (bx, by, bz):
            # Block broken — switch back to planks and retry
            print(f"  [expert] break done: ({bx},{by},{bz}) removed")
            self._hotbar = PLANKS_SLOT
            self._break_target = None
            self.state = self._break_return_state
            self._break_return_state = None
            return noop()

        # Still there — attack with axe
        action = noop()
        action[ACT_INTERACT] = 2  # attack/break
        action[ACT_HOTBAR] = SLOT_AXE
        self._hotbar = SLOT_AXE
        return action

    # ==================================================================
    # Common
    # ==================================================================

    def _wait(self, env, next_state: str) -> np.ndarray:
        """Generic wait state."""
        self.wait_counter -= 1
        if self.wait_counter <= 0:
            self.state = next_state
        return noop()
