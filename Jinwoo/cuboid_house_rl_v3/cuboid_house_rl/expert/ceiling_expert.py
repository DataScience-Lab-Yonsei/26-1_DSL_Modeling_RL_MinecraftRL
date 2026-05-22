"""
CeilingExpert — place ceiling blocks at y=5 from floor level.

Agent stands on floor (y=2), faces -Z (yaw=π), looks up to place blocks
on the +Z face of wall/ceiling blocks at y=5.

Movement (yaw=π fixed):
  +x = strafe left    (ACT_LEFT_RIGHT=0)
  -x = strafe right   (ACT_LEFT_RIGHT=2)
  +z = backward        (ACT_FWD_BACK=0)
  -z = forward         (ACT_FWD_BACK=2)

Pattern: serpentine rows from south wall inward
  Row 0 (z=oz+1): stand at z=oz+2, aim at wall top (y=5,z=oz) +Z face → +x direction
  Row 1 (z=oz+2): step back to z=oz+3, aim at prev ceiling +Z face → -x direction
  ...
  Last rows: can't step back → aim steeper to reach further blocks
"""
import math
import numpy as np

from cuboid_house_rl.config import (
    PLANKS_SLOT, SLOT_PLANKS, SLOT_AXE,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    NUM_ACTION_DIMS,
)
from cuboid_house_rl.expert.movement import (
    noop, place_action, fix_yaw_action, reset_fix_yaw_state,
    error_to_camera_idx, check_raycast_hits_block,
    FINE_THRESHOLD,
)

CEILING_Y = 5  # ceiling block y position


class CeilingExpert:
    """Expert that places ceiling blocks at y=5 from floor level."""

    # States
    MOVE_TO_START  = "move_to_start"
    FIX_YAW        = "fix_yaw"
    MOVE_TO_X      = "move_to_x"
    AIM            = "aim"
    PLACE          = "place"
    WAIT           = "wait"
    ADVANCE_ROW    = "advance_row"
    # Finish sequence
    MOVE_TO_DOOR_X = "move_to_door_x"
    FACE_DOOR      = "face_door"
    WALK_TO_DOOR   = "walk_to_door"
    REFACE_DOOR    = "reface_door"
    OPEN_DOOR      = "open_door"
    WALK_OUT       = "walk_out"
    TURN_AROUND    = "turn_around"
    LOOK_AT_HOUSE  = "look_at_house"
    WAIT_FINISH    = "wait_finish"
    DONE           = "done"

    def __init__(self, origin_x: int, origin_z: int, width: int, depth: int,
                 initial_hotbar: int = SLOT_PLANKS, door_x: int = None):
        self.ox = origin_x
        self.oz = origin_z
        self.width = width
        self.depth = depth
        self.door_x = door_x if door_x is not None else origin_x + width // 2

        # Ceiling interior: x from ox+1 to ox+width-2, z from oz+1 to oz+depth-2
        self.x_min = origin_x + 1
        self.x_max = origin_x + width - 2
        self.z_min = origin_z + 1
        self.z_max = origin_z + depth - 2

        # Current target
        self.row_idx = 0  # which ceiling row (0 = z_min, 1 = z_min+1, ...)
        self.total_rows = self.z_max - self.z_min + 1
        self.target_z = self.z_min  # ceiling z being placed
        self.going_plus_x = True  # serpentine direction
        self._init_x_target()

        # Agent standing z: one block behind the target row (+z direction)
        self.stand_z = self.target_z + 1

        self.state = self.MOVE_TO_START
        self._hotbar = initial_hotbar
        self._yaw_fixed = False
        self._aim_ticks = 0
        self._raycast_confirm = 0
        self._RAYCAST_CONFIRM_NEEDED = 3
        self._prev_pitch_sign = 0  # for pitch dampening

    def _init_x_target(self):
        """Set x target based on serpentine direction."""
        if self.going_plus_x:
            self.current_x = self.x_min
        else:
            self.current_x = self.x_max

    def _next_x(self) -> bool:
        """Advance to next x in current row. Returns False if row done."""
        if self.going_plus_x:
            self.current_x += 1
            if self.current_x > self.x_max:
                return False
        else:
            self.current_x -= 1
            if self.current_x < self.x_min:
                return False
        return True

    def _can_step_back(self) -> bool:
        """Check if agent can step back (+z) for next row."""
        # Agent stands at target_z + 1. Next row target_z + 1.
        # Agent would need to stand at target_z + 2.
        # Can't go past oz + depth - 1 (north wall)
        return self.stand_z + 1 <= self.oz + self.depth - 1

    def is_done(self) -> bool:
        return self.state == self.DONE

    def _check_hotbar(self, env):
        """Switch to next planks slot if current is empty."""
        count = env.get_hotbar_planks_count(self._hotbar)
        if count <= 0:
            new_slot = env.find_planks_slot()
            if new_slot is not None and new_slot != self._hotbar:
                pass  # silent hotbar switch
                self._hotbar = new_slot

    def get_action(self, env) -> np.ndarray:
        self._check_hotbar(env)
        if self.state == self.MOVE_TO_START:
            action = self._move_to_start(env)
        elif self.state == self.FIX_YAW:
            action = self._fix_yaw(env)
        elif self.state == self.MOVE_TO_X:
            action = self._move_to_x(env)
        elif self.state == self.AIM:
            action = self._aim(env)
        elif self.state == self.PLACE:
            action = self._place_block(env)
        elif self.state == self.WAIT:
            action = self._wait(env)
        elif self.state == self.ADVANCE_ROW:
            action = self._advance_row(env)
        elif self.state == self.MOVE_TO_DOOR_X:
            action = self._move_to_door_x(env)
        elif self.state == self.FACE_DOOR:
            action = self._face_door(env)
        elif self.state == self.WALK_TO_DOOR:
            action = self._walk_to_door(env)
        elif self.state == self.REFACE_DOOR:
            action = self._reface_door(env)
        elif self.state == self.OPEN_DOOR:
            action = self._open_door(env)
        elif self.state == self.WALK_OUT:
            action = self._walk_out(env)
        elif self.state == self.TURN_AROUND:
            action = self._turn_around(env)
        elif self.state == self.LOOK_AT_HOUSE:
            action = self._look_at_house(env)
        elif self.state == self.WAIT_FINISH:
            action = self._wait_finish(env)
        else:
            action = noop()

        # Stuck detection: 20 ticks no position change → print debug
        curr_pos = (round(env.agent_x, 1), round(env.agent_y, 1), round(env.agent_z, 1))
        if curr_pos == getattr(self, '_ceil_prev_pos', None):
            self._ceil_stuck_count = getattr(self, '_ceil_stuck_count', 0) + 1
        else:
            self._ceil_stuck_count = 0
        self._ceil_prev_pos = curr_pos
        if self._ceil_stuck_count >= 20 and self._ceil_stuck_count % 20 == 0:
            print(f"  [ceil STUCK] state={self.state} row={self.row_idx}/{self.total_rows}"
                  f" yaw={math.degrees(env.agent_yaw):.1f}° pitch={math.degrees(env.agent_pitch):.1f}°"
                  f" pos=({env.agent_x:.1f},{env.agent_y:.1f},{env.agent_z:.1f})"
                  f" flip={getattr(self, '_pitch_flip_count', 0)} sign={getattr(self, '_prev_pitch_sign', 0)}")

        # Enforce hotbar
        action[ACT_HOTBAR] = self._hotbar
        return action

    def _move_to_start(self, env) -> np.ndarray:
        """Move to starting position using yaw=π movement (strafe + backward)."""
        target_x = self.current_x + 0.5
        target_z = self.stand_z + 0.5

        dx = target_x - env.agent_x
        dz = target_z - env.agent_z
        dist = math.sqrt(dx * dx + dz * dz)

        if dist < 0.4:
            self._yaw_fixed = False
            self.state = self.FIX_YAW
            print(f"  [ceiling] Starting at ({target_x:.1f}, {target_z:.1f}), "
                  f"ceiling rows: {self.total_rows}, x range: {self.x_min}-{self.x_max}")
            return noop()

        # Fix yaw to π (-Z) first
        if not self._yaw_fixed:
            action, yaw_ok = fix_yaw_action(env.agent_yaw, math.pi)
            if not yaw_ok:
                return action
            self._yaw_fixed = True

        # Move using yaw=π: +x=strafe right, -x=strafe left, +z=backward, -z=forward
        action = noop()
        if abs(dx) > 0.15:
            if dx > 0:
                action[ACT_LEFT_RIGHT] = 2  # +x = strafe right
            else:
                action[ACT_LEFT_RIGHT] = 0  # -x = strafe left
        elif abs(dz) > 0.15:
            if dz > 0:
                action[ACT_FWD_BACK] = 0  # +z = backward
            else:
                action[ACT_FWD_BACK] = 2  # -z = forward

        if dist < 0.8:
            action[ACT_SNEAK] = 1

        return action

    def _fix_yaw(self, env) -> np.ndarray:
        """Fix yaw to π (-Z direction)."""
        if not getattr(self, '_fix_yaw_reset', False):
            reset_fix_yaw_state()
            self._fix_yaw_reset = True
        action, yaw_ok = fix_yaw_action(env.agent_yaw, math.pi, tolerance_deg=1.0)
        if yaw_ok:
            self._fix_yaw_reset = False
            self._reset_aim_state()
            self.state = self.AIM
            print(f"  [ceiling] Yaw fixed to -Z, placing row {self.row_idx} (z={self.target_z})")
        return action

    def _aim_at_ceiling(self, env) -> np.ndarray:
        """Aim at +Z face of block at (current_x, CEILING_Y, target_z-1).
        Only adjusts pitch (yaw=π is fixed). Dampening on overshoot."""
        tz = self.target_z - 1

        # +Z face center of the aim block
        face_y = CEILING_Y + 0.5   # y center of block
        face_z = tz + 1.0          # +Z face position

        eye_y = env.agent_y + 1.62
        eye_z = env.agent_z

        dy = face_y - eye_y   # positive = above eye
        dz = face_z - eye_z   # negative = in front

        target_pitch = math.atan2(-dy, abs(dz))
        pitch_err = target_pitch - env.agent_pitch

        # Dampening: cumulative halving on direction flips
        curr_sign = 1 if pitch_err > 0 else (-1 if pitch_err < 0 else 0)
        if curr_sign != 0 and self._prev_pitch_sign != 0 and curr_sign != self._prev_pitch_sign:
            self._pitch_flip_count = min(getattr(self, '_pitch_flip_count', 0) + 1, 2)
            pitch_err *= 0.5 ** self._pitch_flip_count  # 1st: 0.5, 2nd: 0.25
        elif curr_sign == self._prev_pitch_sign:
            self._pitch_flip_count = 0
        if curr_sign != 0:
            self._prev_pitch_sign = curr_sign

        action = noop()
        if abs(pitch_err) > math.radians(0.5):
            action[ACT_PITCH] = error_to_camera_idx(pitch_err)
        return action

    def _reset_aim_state(self):
        """Reset pitch dampening state for new block aim."""
        self._prev_pitch_sign = 0
        self._pitch_flip_count = 0

    def _aim(self, env) -> np.ndarray:
        """Aim at ceiling placement position."""
        # The block we aim at: (current_x, CEILING_Y, target_z - 1)
        # Its +Z face → placement at (current_x, CEILING_Y, target_z)
        aim_block = (self.current_x, CEILING_Y, self.target_z - 1)

        # Check if raycast hits the aim block's +Z face
        if check_raycast_hits_block(env, aim_block, expected_face=(0, 0, 1), check_center=False):
            self._raycast_confirm += 1
            if self._raycast_confirm >= self._RAYCAST_CONFIRM_NEEDED:
                self._raycast_confirm = 0
                self.state = self.PLACE
                return noop()
            return noop()

        self._raycast_confirm = 0
        self._aim_ticks += 1

        # Timeout: if can't aim after 50 ticks, skip this block
        if self._aim_ticks > 50:
            print(f"  [ceiling] Aim timeout at ({self.current_x}, {CEILING_Y}, {self.target_z}), skipping")
            self._aim_ticks = 0
            self._advance_to_next_block()
            return noop()

        return self._aim_at_ceiling(env)

    def _place_block(self, env) -> np.ndarray:
        """Place ceiling block."""
        self.state = self.WAIT
        self._wait_counter = 2
        action = noop()
        action[ACT_INTERACT] = 0  # use/place
        return action

    def _wait(self, env) -> np.ndarray:
        """Wait after placing, then advance."""
        self._wait_counter -= 1
        if self._wait_counter > 0:
            return noop()

        self._aim_ticks = 0
        self._advance_to_next_block()
        return noop()

    def _advance_to_next_block(self):
        """Move to next block in row, or next row."""
        if self._next_x():
            self.state = self.MOVE_TO_X
        else:
            # Row done
            self.row_idx += 1
            if self.row_idx >= self.total_rows:
                print(f"  [ceiling] All rows done! Starting finish sequence.")
                self._yaw_fixed = False
                self._finish_ticks = 0
                self._walk_out_ticks = 0
                self.state = self.MOVE_TO_DOOR_X
                return

            self.target_z += 1
            self.going_plus_x = not self.going_plus_x
            self._init_x_target()

            if self._can_step_back():
                self.stand_z += 1
                self.state = self.ADVANCE_ROW
                print(f"  [ceiling] Row {self.row_idx} (z={self.target_z}), stepping back")
            else:
                # Can't step back — aim from current position (steeper angle)
                self._reset_aim_state()
                self.state = self.AIM
                print(f"  [ceiling] Row {self.row_idx} (z={self.target_z}), aiming from current pos (can't step back)")

    def _move_to_x(self, env) -> np.ndarray:
        """Strafe to next x position (yaw=π fixed)."""
        target_x = self.current_x + 0.5
        dx = target_x - env.agent_x

        if abs(dx) < 0.15:
            self._raycast_confirm = 0
            self._reset_aim_state()
            self.state = self.AIM
            return noop()

        action = noop()
        # Facing -z (yaw=π): right = +x, left = -x
        if dx > 0:
            action[ACT_LEFT_RIGHT] = 2  # strafe right → +x
        else:
            action[ACT_LEFT_RIGHT] = 0  # strafe left → -x

        if abs(dx) < 0.5:
            action[ACT_SNEAK] = 1

        return action

    def _advance_row(self, env) -> np.ndarray:
        """Step back one block (+z direction). Facing -z, so backward = +z.
        If stuck (z doesn't change for 3 ticks), aim from current position."""
        target_z = self.stand_z + 0.5
        dz = target_z - env.agent_z

        if abs(dz) < 0.15:
            self._raycast_confirm = 0
            self._advance_stuck_count = 0
            self.state = self.MOVE_TO_X
            return noop()

        # Stuck detection: z not changing for 3 ticks → can't go back
        curr_z = round(env.agent_z, 2)
        prev_z = getattr(self, '_advance_prev_z', None)
        if prev_z is not None and abs(curr_z - prev_z) < 0.01:
            self._advance_stuck_count = getattr(self, '_advance_stuck_count', 0) + 1
        else:
            self._advance_stuck_count = 0
        self._advance_prev_z = curr_z

        if self._advance_stuck_count >= 3:
            # Can't step back — aim from current position with steeper pitch
            self._advance_stuck_count = 0
            self.stand_z -= 1  # revert stand_z since we couldn't move
            self._raycast_confirm = 0
            self._reset_aim_state()
            self.state = self.AIM
            print(f"  [ceiling] Can't step back (stuck), aiming from current pos for z={self.target_z}")
            return noop()

        action = noop()
        # Facing -z: backward = +z
        if dz > 0:
            action[ACT_FWD_BACK] = 0  # backward → +z
        else:
            action[ACT_FWD_BACK] = 2  # forward → -z

        if abs(dz) < 0.5:
            action[ACT_SNEAK] = 1

        return action

    # ── Finish sequence ─────────────────────────────────────────────

    def _move_to_door_x(self, env) -> np.ndarray:
        """Fix pitch to 0° first (with dampening, one-time), then strafe."""
        if not getattr(self, '_doorx_pitch_fixed', False):
            pitch_err = 0.0 - env.agent_pitch
            if abs(pitch_err) > math.radians(3.0):
                # Dampening: halve on direction flip
                p_sign = 1 if pitch_err > 0 else -1
                prev = getattr(self, '_doorx_pitch_prev_sign', 0)
                if prev != 0 and p_sign != prev:
                    pitch_err *= 0.5
                self._doorx_pitch_prev_sign = p_sign
                action = noop()
                action[ACT_PITCH] = error_to_camera_idx(pitch_err)
                return action
            self._doorx_pitch_fixed = True

        target_x = self.door_x + 0.5
        dx = target_x - env.agent_x

        if abs(dx) < 0.15:
            self._yaw_fixed = False
            self.state = self.FACE_DOOR
            print(f"  [ceiling] Aligned with door x={target_x:.1f}")
            return noop()

        action = noop()
        # Facing -z (yaw=π): right = +x, left = -x
        if dx > 0:
            action[ACT_LEFT_RIGHT] = 2  # +x
        else:
            action[ACT_LEFT_RIGHT] = 0  # -x
        return action

    def _face_door(self, env) -> np.ndarray:
        """Fix yaw to π before walking to door. Pitch already fixed in _move_to_door_x."""
        action, yaw_ok = fix_yaw_action(env.agent_yaw, math.pi, tolerance_deg=1.0)
        if yaw_ok:
            self.state = self.WALK_TO_DOOR
            print(f"  [ceiling] Facing door (yaw=π), walking to door")
        return action

    def _walk_to_door(self, env) -> np.ndarray:
        """Walk forward (-z) toward door at z=oz. Pitch fixed."""
        target_z = self.oz + 2 + 0.5  # 2 blocks inside from door
        dz = target_z - env.agent_z

        if abs(dz) < 0.3:
            self._reface_done = False
            self.state = self.REFACE_DOOR
            print(f"  [ceiling] Reached door position (z={self.oz}+2), re-fixing yaw")
            return noop()

        action = noop()
        action[ACT_FWD_BACK] = 2  # forward = -z
        return action

    def _reface_door(self, env) -> np.ndarray:
        """Re-fix yaw to π after reaching door position, then open door."""
        if not getattr(self, '_reface_yaw_reset', False):
            reset_fix_yaw_state()
            self._reface_yaw_reset = True
        action, yaw_ok = fix_yaw_action(env.agent_yaw, math.pi, tolerance_deg=1.0)
        if yaw_ok:
            self.state = self.OPEN_DOOR
            print(f"  [ceiling] Yaw re-fixed, opening door")
        return action

    def _open_door(self, env) -> np.ndarray:
        """Open the door (interact)."""
        self.state = self.WALK_OUT
        self._walk_out_ticks = 0
        self._hotbar = SLOT_AXE  # axe → env won't count as placement
        print(f"  [ceiling] Opening door")
        action = noop()
        action[ACT_INTERACT] = 0  # use/interact = open door
        return action

    def _walk_out(self, env) -> np.ndarray:
        """Walk forward 5 blocks outside. If stuck, sneak-strafe to align x."""
        self._walk_out_ticks += 1

        # Done: walked enough
        target_z = self.oz - 4.5  # 5 blocks outside from door
        if env.agent_z < target_z:
            self._yaw_fixed = False
            self.state = self.TURN_AROUND
            print(f"  [ceiling] Outside, turning around")
            return noop()

        # Stuck detection: z not changing for 3 ticks → blocked by door frame
        curr_z = round(env.agent_z, 2)
        prev_z = getattr(self, '_walkout_prev_z', None)
        if prev_z is not None and abs(curr_z - prev_z) < 0.01:
            self._walkout_stuck = getattr(self, '_walkout_stuck', 0) + 1
        else:
            self._walkout_stuck = 0
        self._walkout_prev_z = curr_z

        if self._walkout_stuck >= 3:
            # Sneak-strafe to align x with door center
            door_cx = self.door_x + 0.5
            dx = door_cx - env.agent_x
            if abs(dx) < 0.05:
                # x is aligned, re-fix yaw to -z then forward
                self._walkout_stuck = 0
                self._walkout_yaw_fixed = False

            # Re-fix yaw after strafe alignment
            if not getattr(self, '_walkout_yaw_fixed', True):
                action, yaw_ok = fix_yaw_action(env.agent_yaw, math.pi, tolerance_deg=2.0)
                if yaw_ok:
                    self._walkout_yaw_fixed = True
                return action
            # Sneak strafe: 1 tick move, alternate with checking
            action = noop()
            action[ACT_SNEAK] = 1
            # Facing -z (yaw=π): right = +x, left = -x
            if dx > 0:
                action[ACT_LEFT_RIGHT] = 2  # +x
            else:
                action[ACT_LEFT_RIGHT] = 0  # -x
            return action

        action = noop()
        action[ACT_FWD_BACK] = 2  # forward = -z
        return action

    def _turn_around(self, env) -> np.ndarray:
        """Turn 180° to face +z (yaw=0), look at house."""
        if not getattr(self, '_turn_yaw_reset', False):
            reset_fix_yaw_state()
            self._turn_yaw_reset = True
        action, yaw_ok = fix_yaw_action(env.agent_yaw, 0.0, tolerance_deg=1.0)
        if yaw_ok:
            self.state = self.LOOK_AT_HOUSE
            print(f"  [ceiling] Facing house")
        return action

    def _look_at_house(self, env) -> np.ndarray:
        """Look at door upper block (y=3)."""
        # Target: door upper block +z face center
        face_y = 3.5  # y=3 block center
        eye_y = env.agent_y + 1.62
        eye_z = env.agent_z
        door_z = self.oz + 1.0  # +z face of door

        dy = face_y - eye_y
        dz = door_z - eye_z
        target_pitch = math.atan2(-dy, abs(dz))
        pitch_err = target_pitch - env.agent_pitch

        if abs(pitch_err) < math.radians(5.0):
            self._finish_ticks = 0
            self.state = self.WAIT_FINISH
            print(f"  [ceiling] Looking at house, waiting 5 ticks")
            return noop()

        action = noop()
        action[ACT_PITCH] = error_to_camera_idx(pitch_err)
        return action

    def _wait_finish(self, env) -> np.ndarray:
        """Wait 5 ticks then DONE."""
        self._finish_ticks += 1
        if self._finish_ticks >= 10:
            print(f"  [ceiling] Episode complete!")
            self.state = self.DONE
        return noop()
