"""
Wall Expert — Stage 2 (V3).

Builds wall columns around the floor perimeter using look-down + jump + place.

Traversal: counterclockwise spiral starting from where floor expert ended.
  - Odd depth  (last row even → ends at east,  x=ox+width-1):
      east wall → south wall → west wall → north wall
  - Even depth (last row odd  → ends at west, x=ox):
      west wall → south wall → east wall → north wall

Per column sequence:
    1. Navigate to (cx+0.5, cz+0.5) at floor level (y≈2)
    2. Fix pitch to +90° (look straight down)
    3. Jump → wait JUMP_PLACE_DELAY ticks → place
    4. Wait LAND_DELAY ticks for landing (y rises by 1)
    5. Repeat 2-4 for WALL_HEIGHT=4 blocks total
    6. Walk off column top (y=6) → fall to floor → next column

No raycast confirmation needed: looking straight down is accurate as long as
the agent is positioned over the correct floor block.
"""
import math
import numpy as np
from typing import List, Tuple

from cuboid_house_rl.config import (
    PLANKS_SLOT, SLOT_PLANKS, SLOT_AXE, SLOT_GLASS,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    NUM_ACTION_DIMS,
)
from cuboid_house_rl.expert.movement import (
    noop, place_action,
    move_to_xz_action, fix_yaw_action,
    error_to_camera_idx, cap_pitch_for_forward,
    MovementController,
    FINE_THRESHOLD,
)


# ── Timing (ticks) ────────────────────────────────────────────────────────────
WALL_JUMP_PLACE_DELAY = 6    # ticks after jump before placing
WALL_LAND_DELAY       = 6    # ticks after place to wait for landing
WALL_HEIGHT           = 4    # blocks per column  (y=2, 3, 4, 5)

# Pitch target for straight-down look
_PITCH_TARGET   = math.pi / 2        # +90° → looking straight down
_PITCH_TOL      = math.radians(5.0)  # tolerance before starting jump


class WallExpert:
    """
    State machine for building wall columns.

    Args:
        origin_x, origin_z : floor origin from FloorExpert
        width, depth        : floor dimensions from FloorExpert
    """

    # States
    MOVE_TO_COLUMN = "move_to_column"
    LOOK_DOWN      = "look_down"
    JUMP           = "jump"
    WAIT_PEAK      = "wait_peak"
    PLACE          = "place"
    WAIT_LAND      = "wait_land"
    MOVE_NEXT      = "move_next"
    MOVE_TO_CENTER = "move_to_center"
    DONE           = "done"

    def __init__(self, origin_x: int, origin_z: int, width: int, depth: int,
                 hotbar: int = PLANKS_SLOT):
        self.ox    = origin_x
        self.oz    = origin_z
        self.width = width
        self.depth = depth
        self._hotbar = hotbar

        # Floor last row direction determines which wall we start on.
        # Row index (0-based): even rows go +X, odd rows go -X.
        # Last row index = depth - 1.
        # depth odd  → last row even → agent ends at east  (x = ox+width-1)
        # depth even → last row odd  → agent ends at west  (x = ox)
        self._start_east = (depth % 2 == 1)

        self.columns: List[Tuple[int, int]] = self._generate_columns()
        self.col_idx          = 0
        self.blocks_in_column = 0

        self.state        = self.MOVE_TO_COLUMN
        self.wait_counter = 0
        self._yaw_fixed   = False
        self._jumped_from_ground = False
        self._pre_jump_y  = 0.0   # y before jump, for landing verification
        self._land_stable_y = None
        self._land_stable_count = 0

    # ── Column list generation ────────────────────────────────────────────────

    # Arrival threshold for forward walking
    _ARRIVE_DIST = 0.4

    def _generate_columns(self) -> List[Tuple[int, int, str]]:
        """
        Generate the spiral perimeter column list.

        Each entry is (x, z, wall_side) — floor position + which wall.
        wall_side: "east", "south", "west", "north"
        Exactly 2*(width+depth-2) columns total (each perimeter cell once).
        """
        ox, oz = self.ox, self.oz
        w, d   = self.width, self.depth
        cols: List[Tuple[int, int, str]] = []

        if self._start_east:
            for z in range(oz + d - 2, oz - 1, -1):
                cols.append((ox + w - 1, z, "east"))
            for x in range(ox + w - 2, ox - 1, -1):
                cols.append((x, oz, "south"))
            for z in range(oz + 1, oz + d):
                cols.append((ox, z, "west"))
            for x in range(ox + 1, ox + w):
                cols.append((x, oz + d - 1, "north"))
        else:
            for z in range(oz + d - 2, oz - 1, -1):
                cols.append((ox, z, "west"))
            for x in range(ox + 1, ox + w):
                cols.append((x, oz, "south"))
            for z in range(oz + 1, oz + d):
                cols.append((ox + w - 1, z, "east"))
            for x in range(ox + w - 2, ox - 1, -1):
                cols.append((x, oz + d - 1, "north"))

        return cols

    # ── Public API ────────────────────────────────────────────────────────────

    def is_done(self) -> bool:
        return self.state == self.DONE

    def get_current_target(self):
        if self.col_idx >= len(self.columns):
            return None
        x, z, _ = self.columns[self.col_idx]
        return (x, 2 + self.blocks_in_column, z)

    def get_remaining_targets(self) -> list:
        targets = []
        for i in range(self.col_idx, len(self.columns)):
            x, z, _ = self.columns[i]
            y0 = 2 + (self.blocks_in_column if i == self.col_idx else 0)
            for y in range(y0, 2 + WALL_HEIGHT):
                targets.append((x, y, z))
        return targets

    def _check_hotbar(self, env):
        """If current planks/glass slot is empty, switch to next slot with same material."""
        if self._hotbar < SLOT_AXE:
            # Planks slot — check if empty
            count = env.get_hotbar_planks_count(self._hotbar)
            if count <= 0:
                new_slot = env.find_planks_slot()
                if new_slot >= 0 and new_slot != self._hotbar:
                    self._hotbar = new_slot
        elif self._hotbar >= SLOT_GLASS:
            # Glass slot — check if empty
            new_slot = env.find_glass_slot()
            if new_slot >= 0 and new_slot != self._hotbar:
                self._hotbar = new_slot

    def _place_block(self):
        return place_action(self._hotbar)

    # ── Main dispatch ─────────────────────────────────────────────────────────

    def get_action(self, env) -> np.ndarray:
        self._check_hotbar(env)

        if self.col_idx >= len(self.columns) and self.state != self.MOVE_TO_CENTER:
            self.state = self.MOVE_TO_CENTER
            self._yaw_fixed = False
            action = noop()
        elif self.state == self.MOVE_TO_CENTER:
            action = self._move_to_center(env)
        elif self.state == self.MOVE_TO_COLUMN:
            action = self._move_to_column(env)
        elif self.state == self.LOOK_DOWN:
            action = self._look_down(env)
        elif self.state == self.JUMP:
            action = self._jump(env)
        elif self.state == self.WAIT_PEAK:
            action = self._wait_peak(env)
        elif self.state == self.PLACE:
            action = self._place(env)
        elif self.state == self.WAIT_LAND:
            action = self._wait_land(env)
        elif self.state == self.MOVE_NEXT:
            action = self._move_next(env)
        elif self.state == self.MOVE_TO_CENTER:
            action = self._move_to_center(env)
        else:
            action = noop()

        action[ACT_HOTBAR] = self._hotbar

        # Stuck detection: if agent doesn't move for 10+ steps, print yaw debug
        curr_pos = (round(env.agent_x, 2), round(env.agent_z, 2))
        if curr_pos == getattr(self, '_stuck_prev_pos', None):
            self._stuck_count = getattr(self, '_stuck_count', 0) + 1
        else:
            self._stuck_count = 0
        self._stuck_prev_pos = curr_pos

        if self._stuck_count >= 100:
            cx, cz, _ = self.columns[min(self.col_idx, len(self.columns) - 1)]
            dx = (cx + 0.5) - env.agent_x
            dz = (cz + 0.5) - env.agent_z
            dist = math.sqrt(dx * dx + dz * dz)
            target_yaw = math.atan2(dx, dz)
            yaw_err = target_yaw - env.agent_yaw
            yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            print(f"  [wall STUCK] state={self.state} col={self.col_idx} "
                  f"yaw={math.degrees(env.agent_yaw):.1f}° "
                  f"target={math.degrees(target_yaw):.1f}° "
                  f"err={math.degrees(yaw_err):.1f}° "
                  f"act_yaw={action[ACT_YAW]} "
                  f"pitch={math.degrees(env.agent_pitch):.1f}° "
                  f"y={env.agent_y:.1f} "
                  f"dx={dx:.2f} dz={dz:.2f} dist={dist:.2f} "
                  f"yaw_fixed={self._yaw_fixed}")

        return action

    # ── States ────────────────────────────────────────────────────────────────

    def _get_travel_yaw(self) -> float:
        """Compute yaw to face from current column toward next column (travel direction).
        For the last column, face same direction as previous travel.
        For the first column, face toward the column from agent's expected position.
        """
        if self.col_idx > 0 and self.col_idx < len(self.columns):
            px, pz, _ = self.columns[self.col_idx - 1]
            cx, cz, _ = self.columns[self.col_idx]
            dx, dz = cx - px, cz - pz
            if abs(dx) + abs(dz) > 0:
                return math.atan2(dx, dz)
        # Fallback: compute from current column to next, or use 0
        if self.col_idx + 1 < len(self.columns):
            cx, cz, _ = self.columns[self.col_idx]
            nx, nz, _ = self.columns[self.col_idx + 1]
            dx, dz = nx - cx, nz - cz
            if abs(dx) + abs(dz) > 0:
                return math.atan2(dx, dz)
        return 0.0

    _SNEAK_DIST = 0.8  # start sneaking when this close

    def _move_to_column(self, env) -> np.ndarray:
        """Navigate to column center. Must be at floor level to start building.
        If still on wall (y > 2.5), just walk forward to fall off first.
        After landing, fix yaw once toward column, then walk forward.
        """
        cx, cz, side = self.columns[self.col_idx]
        target_x, target_z = cx + 0.5, cz + 0.5

        dx = target_x - env.agent_x
        dz = target_z - env.agent_z
        dist = math.sqrt(dx * dx + dz * dz)

        # Still on top of wall → walk forward to fall off (no yaw change)
        if env.agent_y > 2.5:
            action = noop()
            action[ACT_FWD_BACK] = 2
            return action

        # Fell off floor (y ≈ 1) → face column center, jump + forward
        if env.agent_y <= 1.5:
            if not self._yaw_fixed:
                self._ground_prev_x = None
                self._ground_prev_z = None
                self._ground_stuck_count = 0
                target_yaw = math.atan2(dx, dz)
                action, yaw_ok = fix_yaw_action(env.agent_yaw, target_yaw)
                if not yaw_ok:
                    return action
                self._yaw_fixed = True

            # Stuck detection: x or z not changing for 3 ticks → jump
            stuck = False
            if self._ground_prev_x is not None:
                if abs(env.agent_x - self._ground_prev_x) < 0.01 or \
                   abs(env.agent_z - self._ground_prev_z) < 0.01:
                    self._ground_stuck_count += 1
                else:
                    self._ground_stuck_count = 0
                if self._ground_stuck_count >= 3:
                    stuck = True
            self._ground_prev_x = env.agent_x
            self._ground_prev_z = env.agent_z

            action = noop()
            action[ACT_FWD_BACK] = 2
            if dist <= 0.8 or stuck:
                action[ACT_JUMP] = 1
            cap_pitch_for_forward(action, env.agent_pitch)
            self._jumped_from_ground = True
            return action

        # Just jumped back from ground → reset yaw for re-adjustment
        if getattr(self, '_jumped_from_ground', False):
            self._jumped_from_ground = False
            self._yaw_fixed = False

        # At floor level: close enough → start building
        if dist < self._ARRIVE_DIST:
            self._yaw_fixed = False
            if self.blocks_in_column != 0:
                print(f"  [wall] BUG: col={self.col_idx} blocks_in_column={self.blocks_in_column}, resetting")
                self.blocks_in_column = 0
            self.state = self.LOOK_DOWN
            return noop()

        # Fix pitch first if too steep (gimbal lock at 90°)
        if abs(env.agent_pitch) > math.radians(80.0):
            action = noop()
            pitch_err = 0.0 - env.agent_pitch
            action[ACT_PITCH] = error_to_camera_idx(pitch_err)
            return action

        # Face column center (one-time fix after landing)
        if not self._yaw_fixed:
            target_yaw = math.atan2(dx, dz)

            # dx≈0 or dz≈0 → yaw unstable (±π or ±π/2 boundary)
            # Force right-only correction only when error is small (near boundary)
            if abs(dx) < 0.05 or abs(dz) < 0.05:
                yaw_err = target_yaw - env.agent_yaw
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                if abs(yaw_err) < math.radians(5.0):
                    # Small error near boundary → force right to avoid oscillation
                    if abs(yaw_err) < math.radians(1.0):
                        self._yaw_fixed = True
                        return noop()
                    action = noop()
                    action[ACT_YAW] = 6  # +0.3° right always
                    return action
                # Large error → use normal fix_yaw (correct direction)

            action, yaw_ok = fix_yaw_action(env.agent_yaw, target_yaw, tolerance_deg=1.0)
            if not yaw_ok:
                return action
            self._yaw_fixed = True

        # Walk forward — sneak when close for precision
        action = noop()
        action[ACT_FWD_BACK] = 2  # forward
        if dist < self._SNEAK_DIST:
            action[ACT_SNEAK] = 1
        cap_pitch_for_forward(action, env.agent_pitch)
        return action

    def _is_corner_column(self) -> bool:
        """Check if current column is at a house corner."""
        cx, cz, _ = self.columns[self.col_idx]
        corners = {
            (self.ox, self.oz),
            (self.ox + self.width - 1, self.oz),
            (self.ox, self.oz + self.depth - 1),
            (self.ox + self.width - 1, self.oz + self.depth - 1),
        }
        return (cx, cz) in corners

    def _look_down(self, env) -> np.ndarray:
        """Adjust pitch to +90° (straight down). Select material before jump."""
        # Choose material: corner or y=5 → planks, else → glass
        if self._is_corner_column() or self.blocks_in_column == 3:
            self._hotbar = SLOT_PLANKS
        else:
            self._hotbar = SLOT_GLASS

        pitch_err = _PITCH_TARGET - env.agent_pitch
        if abs(pitch_err) < _PITCH_TOL:
            self.state = self.JUMP
            return noop()
        action = noop()
        action[ACT_PITCH] = error_to_camera_idx(pitch_err)
        return action

    def _jump(self, env) -> np.ndarray:
        self._pre_jump_y = env.agent_y
        self._land_stable_y = None
        self._land_stable_count = 0
        action = noop()
        action[ACT_JUMP] = 1
        self.state       = self.WAIT_PEAK
        self.wait_counter = WALL_JUMP_PLACE_DELAY
        return action

    def _wait_peak(self, env) -> np.ndarray:
        self.wait_counter -= 1
        if self.wait_counter <= 0:
            self.state = self.PLACE
        return noop()

    def _place(self, env) -> np.ndarray:
        self.state        = self.WAIT_LAND
        self.wait_counter = 0
        return self._place_block()

    _LAND_STABLE_NEEDED = 3  # 3 ticks of same y = landed
    _LAND_TIMEOUT = 20       # max ticks to wait for landing

    def _wait_land(self, env) -> np.ndarray:
        """Wait until agent lands (3 consecutive ticks at same y), then verify."""
        self.wait_counter += 1

        # Timeout guard
        if self.wait_counter > self._LAND_TIMEOUT:
            print(f"  [wall] _wait_land timeout, y={env.agent_y:.1f}, forcing next")
            self._land_stable_y = None
            self._land_stable_count = 0
            self.blocks_in_column += 1
            return self._check_column_done(env)

        # Track stable y (3 consecutive ticks at same y)
        cur_y = round(env.agent_y, 1)
        if self._land_stable_y is not None and abs(cur_y - self._land_stable_y) < 0.15:
            self._land_stable_count += 1
        else:
            self._land_stable_y = cur_y
            self._land_stable_count = 1

        if self._land_stable_count < self._LAND_STABLE_NEEDED:
            return noop()

        # Landed! Check if y increased (block was placed successfully)
        landed_y = env.agent_y
        cx, cz, side = self.columns[self.col_idx]
        y_gained = landed_y - self._pre_jump_y

        if y_gained > 0.5:
            # Success: y went up ~1 → block placed
            self.blocks_in_column += 1
            self._pre_jump_y = landed_y
        else:
            # Failed: y didn't increase → block not placed, retry
            self._land_stable_y = None
            self._land_stable_count = 0
            self.state = self.LOOK_DOWN
            return noop()

        return self._check_column_done(env)

    def _check_column_done(self, env) -> np.ndarray:
        if self.blocks_in_column >= WALL_HEIGHT:
            # Column complete
            self.blocks_in_column = 0
            self.col_idx += 1
            if self.col_idx >= len(self.columns):
                # Last column done → move to house center
                self.state = self.MOVE_TO_CENTER
                self._yaw_fixed = False
            else:
                self.state = self.MOVE_NEXT
        else:
            # More blocks in this column
            self.state = self.LOOK_DOWN
        return noop()

    _WALK_OFF_TICKS = 1  # walk forward to step off edge, then free-fall

    def _move_next(self, env) -> np.ndarray:
        """
        Fix yaw to travel direction once, then walk forward to fall off.
        Yaw stays locked during fall.
        """
        if self.col_idx >= len(self.columns):
            self.state = self.MOVE_TO_CENTER
            self._yaw_fixed = False
            return noop()

        # Already landed?
        if env.agent_y < 2.5:
            self.state = self.MOVE_TO_COLUMN
            self._yaw_fixed = False
            return noop()

        # Fix yaw once before walking off
        if not self._yaw_fixed:
            target_yaw = self._get_travel_yaw()
            action, yaw_ok = fix_yaw_action(env.agent_yaw, target_yaw, tolerance_deg=1.0)
            if not yaw_ok:
                return action
            self._yaw_fixed = True
            self.wait_counter = 0

        # Timeout guard
        self.wait_counter += 1
        if self.wait_counter > 40:
            print(f"  [wall] _move_next timeout at y={env.agent_y:.2f}, forcing next column")
            self.state        = self.MOVE_TO_COLUMN
            self._yaw_fixed = False
            self.wait_counter = 0
            return noop()

        # Walk forward to step off edge
        if self.wait_counter <= self._WALK_OFF_TICKS:
            action = noop()
            action[ACT_FWD_BACK] = 2  # forward
            return action

        # Brake: press backward for a few ticks to cancel momentum
        if self.wait_counter <= self._WALK_OFF_TICKS + 2:
            action = noop()
            action[ACT_FWD_BACK] = 0  # backward
            return action

        # After braking: just fall straight down
        return noop()

    def _move_to_center(self, env) -> np.ndarray:
        """Face house center, fall off wall, walk to center, then DONE."""
        center_x = self.ox + self.width // 2 + 0.5
        center_z = self.oz + self.depth // 2 + 0.5

        dx = center_x - env.agent_x
        dz = center_z - env.agent_z
        dist = math.sqrt(dx * dx + dz * dz)

        # Still on wall → face center and walk forward to fall off
        if env.agent_y > 2.5:
            self._center_landed = False
            if not self._yaw_fixed:
                target_yaw = math.atan2(dx, dz)
                action, yaw_ok = fix_yaw_action(env.agent_yaw, target_yaw)
                if not yaw_ok:
                    return action
                self._yaw_fixed = True
            action = noop()
            action[ACT_FWD_BACK] = 2  # forward
            return action

        # Just landed → fix pitch to 0° first, then re-fix yaw
        if not getattr(self, '_center_landed', False):
            self._center_landed = True
            self._center_pitch_fixed = False
            self._yaw_fixed = False

        # Fix pitch to 0° after landing (one-time)
        if not getattr(self, '_center_pitch_fixed', True):
            pitch_err = 0.0 - env.agent_pitch
            if abs(pitch_err) > math.radians(3.0):
                action = noop()
                action[ACT_PITCH] = error_to_camera_idx(pitch_err)
                return action
            self._center_pitch_fixed = True

        # Arrived at center
        if dist < 0.4:
            print(f"  [wall] Arrived at house center ({center_x:.1f}, {center_z:.1f})")
            self.state = self.DONE
            return noop()

        # Face center and walk (tolerance=1°)
        if not self._yaw_fixed:
            target_yaw = math.atan2(dx, dz)
            action, yaw_ok = fix_yaw_action(env.agent_yaw, target_yaw, tolerance_deg=1.0)
            if not yaw_ok:
                return action
            self._yaw_fixed = True

        action = noop()
        action[ACT_FWD_BACK] = 2  # forward
        return action
