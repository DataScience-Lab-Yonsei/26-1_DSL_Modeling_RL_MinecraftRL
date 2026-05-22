"""
DoorExpert — break 2 wall blocks and place a door.

Flow:
1. MOVE_TO_DOOR: walk to (door_x+0.5, oz+2.5) — 2 blocks inside from south wall
2. FACE_DOOR: fix yaw to -Z (face south wall)
3. AIM_UPPER: aim at upper wall block (door_x, 3, oz)
4. BREAK_UPPER: switch to axe, break until gone
5. AIM_LOWER: aim at lower wall block (door_x, 2, oz)
6. BREAK_LOWER: switch to axe, break until gone
7. PLACE_DOOR: switch to door slot, aim at ground below door, place
8. DONE
"""
import math
import numpy as np

from cuboid_house_rl.config import (
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    NUM_ACTION_DIMS,
    SLOT_AXE, SLOT_DOOR, SLOT_PLANKS,
    DOOR_HEIGHT_BOTTOM, DOOR_HEIGHT_TOP,
)
from cuboid_house_rl.expert.movement import (
    noop, fix_yaw_action, move_to_xz_action,
    error_to_camera_idx, check_raycast_hits_block,
    raycast_aim_action,
    FINE_THRESHOLD,
)


class DoorExpert:
    """Expert that breaks 2 wall blocks and places a door."""

    # States
    MOVE_TO_DOOR = "move_to_door"
    FACE_DOOR    = "face_door"
    AIM_UPPER    = "aim_upper"
    BREAK_UPPER  = "break_upper"
    AIM_LOWER    = "aim_lower"
    BREAK_LOWER  = "break_lower"
    PLACE_DOOR   = "place_door"
    DONE         = "done"

    def __init__(self, origin_x: int, origin_z: int, width: int, depth: int):
        self.ox = origin_x
        self.oz = origin_z
        self.width = width
        self.depth = depth

        # Door position: front wall (south, z=oz) center
        self.door_x = origin_x + width // 2
        self.door_z = origin_z

        self.state = self.MOVE_TO_DOOR
        self._yaw_fixed = False
        self._hotbar = SLOT_PLANKS
        self._break_ticks = 0
        self._raycast_confirm_count = 0
        self._prev_pitch_sign = 0  # for pitch dampening
        self._prev_yaw_sign = 0    # for yaw dampening
        self._pitch_flip_count = 0  # consecutive flips for stronger dampening
        self._yaw_flip_count = 0

    def is_done(self) -> bool:
        return self.state == self.DONE

    def get_action(self, env) -> np.ndarray:
        if self.state == self.MOVE_TO_DOOR:
            action = self._move_to_door(env)
        elif self.state == self.FACE_DOOR:
            action = self._face_door(env)
        elif self.state == self.AIM_UPPER:
            action = self._aim_upper(env)
        elif self.state == self.BREAK_UPPER:
            action = self._break_upper(env)
        elif self.state == self.AIM_LOWER:
            action = self._aim_lower(env)
        elif self.state == self.BREAK_LOWER:
            action = self._break_lower(env)
        elif self.state == self.PLACE_DOOR:
            action = self._place_door(env)
        else:
            action = noop()

        # Enforce hotbar
        action[ACT_HOTBAR] = self._hotbar
        return action

    def _move_to_door(self, env) -> np.ndarray:
        """Walk to 2 blocks inside from the door (door_x+0.5, oz+2.5)."""
        target_x = self.door_x + 0.5
        target_z = self.oz + 2.5  # 2 blocks inside from south wall

        dx = target_x - env.agent_x
        dz = target_z - env.agent_z
        dist = math.sqrt(dx * dx + dz * dz)

        if dist < 0.4:
            self._yaw_fixed = False
            self.state = self.FACE_DOOR
            print(f"  [door] Arrived at door position ({target_x:.1f}, {target_z:.1f})")
            return noop()

        # Face target and walk (tolerance=1°)
        if not self._yaw_fixed:
            target_yaw = math.atan2(dx, dz)
            action, yaw_ok = fix_yaw_action(env.agent_yaw, target_yaw, tolerance_deg=1.0)
            if not yaw_ok:
                return action
            self._yaw_fixed = True

        action = noop()
        action[ACT_FWD_BACK] = 2  # forward
        return action

    def _face_door(self, env) -> np.ndarray:
        """Fix yaw to face south wall (-Z direction)."""
        # Face -Z = yaw π
        action, yaw_ok = fix_yaw_action(env.agent_yaw, math.pi, tolerance_deg=1.0)
        if yaw_ok:
            self._yaw_fixed = True
            self.state = self.AIM_UPPER
            self._hotbar = SLOT_AXE
            print(f"  [door] Facing door, breaking upper block ({self.door_x}, {DOOR_HEIGHT_TOP}, {self.door_z})")
        return action

    def _aim_at_block_face(self, env, target_block) -> np.ndarray:
        """Aim at the +Z face center of a block (the face facing the agent).
        Includes pitch/yaw dampening to prevent oscillation."""
        tx, ty, tz = target_block
        # +Z face center: x+0.5, y+0.5, z+1.0
        face_x = tx + 0.5
        face_y = ty + 0.5
        face_z = tz + 1.0

        eye_x = env.agent_x
        eye_y = env.agent_y + 1.62
        eye_z = env.agent_z

        dx = face_x - eye_x
        dy = face_y - eye_y
        dz = face_z - eye_z
        dist_xz = math.sqrt(dx**2 + dz**2)

        target_pitch = math.atan2(-dy, dist_xz)
        target_yaw = math.atan2(dx, dz)

        pitch_err = target_pitch - env.agent_pitch
        yaw_err = target_yaw - env.agent_yaw
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

        # Pitch dampening: cumulative halving on direction flips
        p_sign = 1 if pitch_err > 0 else (-1 if pitch_err < 0 else 0)
        if p_sign != 0 and self._prev_pitch_sign != 0 and p_sign != self._prev_pitch_sign:
            self._pitch_flip_count = min(self._pitch_flip_count + 1, 2)
            pitch_err *= 0.5 ** self._pitch_flip_count  # 1st: 0.5, 2nd: 0.25
        elif p_sign == self._prev_pitch_sign:
            self._pitch_flip_count = 0
        if p_sign != 0:
            self._prev_pitch_sign = p_sign

        # Yaw dampening: cumulative halving on direction flips
        y_sign = 1 if yaw_err > 0 else (-1 if yaw_err < 0 else 0)
        if y_sign != 0 and self._prev_yaw_sign != 0 and y_sign != self._prev_yaw_sign:
            self._yaw_flip_count = min(self._yaw_flip_count + 1, 2)
            yaw_err *= 0.5 ** self._yaw_flip_count  # 1st: 0.5, 2nd: 0.25
        elif y_sign == self._prev_yaw_sign:
            self._yaw_flip_count = 0
        if y_sign != 0:
            self._prev_yaw_sign = y_sign

        action = noop()
        if abs(pitch_err) > math.radians(0.5):
            action[ACT_PITCH] = error_to_camera_idx(pitch_err)
        if abs(yaw_err) > math.radians(0.5):
            action[ACT_YAW] = error_to_camera_idx(-yaw_err)
        return action

    def _aim_upper(self, env) -> np.ndarray:
        """Aim at upper door block (y=3)."""
        target_block = (self.door_x, DOOR_HEIGHT_TOP, self.door_z)

        if check_raycast_hits_block(env, target_block, check_center=False):
            self._hotbar = SLOT_AXE
            self._break_ticks = 0
            self.state = self.BREAK_UPPER
            print(f"  [door] Breaking upper block {target_block}")
            return noop()

        return self._aim_at_block_face(env, target_block)

    def _break_upper(self, env) -> np.ndarray:
        """Break upper door block with axe."""
        target_block = (self.door_x, DOOR_HEIGHT_TOP, self.door_z)

        # Block gone → immediately move to lower
        if not check_raycast_hits_block(env, target_block, check_center=False):
            self._hotbar = SLOT_AXE
            self.state = self.AIM_LOWER
            print(f"  [door] Upper block broken, aiming at lower block ({self.door_x}, {DOOR_HEIGHT_BOTTOM}, {self.door_z})")
            return noop()

        # Keep attacking
        action = noop()
        action[ACT_INTERACT] = 2  # attack
        return action

    def _aim_lower(self, env) -> np.ndarray:
        """Aim at lower door block (y=2)."""
        target_block = (self.door_x, DOOR_HEIGHT_BOTTOM, self.door_z)

        if check_raycast_hits_block(env, target_block, check_center=False):
            self._hotbar = SLOT_AXE
            self._break_ticks = 0
            self.state = self.BREAK_LOWER
            print(f"  [door] Breaking lower block {target_block}")
            return noop()

        return self._aim_at_block_face(env, target_block)

    def _break_lower(self, env) -> np.ndarray:
        """Break lower door block with axe."""
        target_block = (self.door_x, DOOR_HEIGHT_BOTTOM, self.door_z)

        # Block gone → immediately place door
        if not check_raycast_hits_block(env, target_block, check_center=False):
            self._hotbar = SLOT_DOOR
            self.state = self.PLACE_DOOR
            print(f"  [door] Lower block broken, placing door")
            return noop()

        # Keep attacking
        action = noop()
        action[ACT_INTERACT] = 2  # attack
        return action

    def _aim_at_block_top(self, env, target_block) -> np.ndarray:
        """Aim at the top face center of a block. With dampening."""
        tx, ty, tz = target_block
        # Top face center: x+0.5, y+1.0, z+0.5
        face_x = tx + 0.5
        face_y = ty + 1.0
        face_z = tz + 0.5

        eye_x = env.agent_x
        eye_y = env.agent_y + 1.62
        eye_z = env.agent_z

        dx = face_x - eye_x
        dy = face_y - eye_y
        dz = face_z - eye_z
        dist_xz = math.sqrt(dx**2 + dz**2)

        target_pitch = math.atan2(-dy, dist_xz)
        target_yaw = math.atan2(dx, dz)

        pitch_err = target_pitch - env.agent_pitch
        yaw_err = target_yaw - env.agent_yaw
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

        # Pitch dampening: cumulative halving on direction flips
        p_sign = 1 if pitch_err > 0 else (-1 if pitch_err < 0 else 0)
        if p_sign != 0 and self._prev_pitch_sign != 0 and p_sign != self._prev_pitch_sign:
            self._pitch_flip_count = min(self._pitch_flip_count + 1, 2)
            pitch_err *= 0.5 ** self._pitch_flip_count
        elif p_sign == self._prev_pitch_sign:
            self._pitch_flip_count = 0
        if p_sign != 0:
            self._prev_pitch_sign = p_sign

        # Yaw dampening: cumulative halving on direction flips
        y_sign = 1 if yaw_err > 0 else (-1 if yaw_err < 0 else 0)
        if y_sign != 0 and self._prev_yaw_sign != 0 and y_sign != self._prev_yaw_sign:
            self._yaw_flip_count = min(self._yaw_flip_count + 1, 2)
            yaw_err *= 0.5 ** self._yaw_flip_count
        elif y_sign == self._prev_yaw_sign:
            self._yaw_flip_count = 0
        if y_sign != 0:
            self._prev_yaw_sign = y_sign

        action = noop()
        if abs(pitch_err) > math.radians(0.5):
            action[ACT_PITCH] = error_to_camera_idx(pitch_err)
        if abs(yaw_err) > math.radians(0.5):
            action[ACT_YAW] = error_to_camera_idx(-yaw_err)
        return action

    def _place_door(self, env) -> np.ndarray:
        """Place door at the opening. Aim at floor block below door."""
        # Aim at floor block (door_x, 1, door_z) top face
        target_block = (self.door_x, 1, self.door_z)

        if check_raycast_hits_block(env, target_block, expected_face=(0, 1, 0), check_center=False):
            self._raycast_confirm_count += 1
            if self._raycast_confirm_count >= 3:
                self._raycast_confirm_count = 0
                self.state = self.DONE
                print(f"  [door] Door placed!")
                action = noop()
                action[ACT_INTERACT] = 0  # use/place
                return action
            return noop()

        self._raycast_confirm_count = 0
        return raycast_aim_action(env, target_block)
