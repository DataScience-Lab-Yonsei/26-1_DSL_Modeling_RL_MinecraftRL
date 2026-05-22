"""
Movement and camera control utilities for scripted experts.

All movement assumes yaw is fixed to a known direction.
When yaw = 0 (+Z direction):
    forward/back = ±Z
    strafe right/left = ±X

Precision movement uses sneak for fine-grained control:
    |delta| > 0.5  → normal speed
    |delta| > 0.05 → sneak (slow, precise)
    |delta| <= 0.05 → arrived
"""
import math
import numpy as np

from cuboid_house_rl.config import (
    CAMERA_DELTA_MAP,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    NUM_ACTION_DIMS, PLANKS_SLOT,
)

# Tolerances
COARSE_THRESHOLD = 0.5   # switch to sneak below this
FINE_THRESHOLD = 0.05    # consider arrived below this
AIM_TOLERANCE_RAD = math.radians(3.0)  # ~3° for aim check


# ==============================================================================
# Action construction
# ==============================================================================

def noop(hotbar: int = PLANKS_SLOT) -> np.ndarray:
    """Create a no-op action."""
    action = np.zeros(NUM_ACTION_DIMS, dtype=np.int64)
    action[ACT_FWD_BACK] = 1    # stop
    action[ACT_LEFT_RIGHT] = 1  # stop
    action[ACT_INTERACT] = 1    # nothing
    action[ACT_PITCH] = 4       # 0° delta (index 4 in 9-option map)
    action[ACT_YAW] = 4         # 0° delta
    action[ACT_HOTBAR] = hotbar
    return action


def place_action(hotbar: int = PLANKS_SLOT) -> np.ndarray:
    """Create a place (use) action."""
    action = noop(hotbar)
    action[ACT_INTERACT] = 0  # use/place
    return action


def jump_forward_action(hotbar: int = PLANKS_SLOT) -> np.ndarray:
    """Create a jump + forward action."""
    action = noop(hotbar)
    action[ACT_JUMP] = 1
    action[ACT_FWD_BACK] = 2  # forward
    return action


_PITCH_CAP_RAD = math.radians(70.0)


def cap_pitch_for_forward(action: np.ndarray, agent_pitch: float) -> np.ndarray:
    """If pitch > 70°, add pitch correction to action so agent moves faster."""
    if agent_pitch > _PITCH_CAP_RAD:
        pitch_err = _PITCH_CAP_RAD - agent_pitch  # negative → look up
        action[ACT_PITCH] = error_to_camera_idx(pitch_err)
    return action


# ==============================================================================
# Camera delta selection
# ==============================================================================

def error_to_camera_idx(error_rad: float) -> int:
    """
    Convert angular error (radians) to best camera delta action index.

    CAMERA_DELTA_MAP = [-10, -3, -1, -0.3, 0, +0.3, +1, +3, +10] degrees.

    Picks the delta that reduces error without overshooting.
    """
    error_deg = math.degrees(error_rad)

    if abs(error_deg) < 0.5:
        return 4  # 0° delta (index 4 in 9-option map), close enough

    best_idx = 4
    best_score = abs(error_deg)  # score = remaining error after applying delta

    for i, delta in enumerate(CAMERA_DELTA_MAP):
        remaining = abs(error_deg - delta)
        # Penalize overshooting (delta goes past zero in opposite direction)
        if (error_deg > 0 and delta > error_deg * 1.2) or \
           (error_deg < 0 and delta < error_deg * 1.2):
            remaining += 2.0
        if remaining < best_score:
            best_score = remaining
            best_idx = i

    return best_idx


def _block_err_to_camera_idx(block_err: int, positive_is_down: bool) -> int:
    """
    Map integer block-distance error to a camera delta index.

    Args:
        block_err: signed integer error in block coords (positive = need more)
        positive_is_down: if True, positive error → pitch down (higher index)
                          if False, positive error → turn right (higher index)
    Returns:
        Index into CAMERA_DELTA_MAP.
    """
    if block_err == 0:
        return 4  # 0°
    sign = 1 if block_err > 0 else -1
    abs_err = abs(block_err)
    if abs_err >= 3:
        delta = sign * 10
    elif abs_err == 2:
        delta = sign * 3
    else:  # 1 block
        delta = sign * 1
    # find closest index
    best = min(range(len(CAMERA_DELTA_MAP)),
               key=lambda i: abs(CAMERA_DELTA_MAP[i] - delta))
    return best


# ==============================================================================
# Aim computation
# ==============================================================================

def compute_aim_error(agent_x, agent_y, agent_z, agent_yaw, agent_pitch,
                      target_point: np.ndarray) -> tuple:
    """
    Compute yaw and pitch error from agent gaze to a target point.

    Args:
        agent_x/y/z: agent feet position
        agent_yaw/pitch: current gaze direction (radians)
        target_point: (3,) world coordinate to look at

    Returns:
        (yaw_error, pitch_error) in radians.
        Positive yaw_error = need to turn right (toward +X when facing +Z)
        Positive pitch_error = need to look down more
    """
    eye = np.array([agent_x, agent_y + 1.62, agent_z])
    diff = target_point - eye
    dist = np.linalg.norm(diff)
    if dist < 0.01:
        return 0.0, 0.0

    # Target yaw (our convention: yaw=0 → +Z, +yaw → +X)
    target_yaw = math.atan2(diff[0], diff[2])
    yaw_err = target_yaw - agent_yaw
    yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

    # Target pitch (positive = looking down)
    horiz_dist = math.sqrt(diff[0] ** 2 + diff[2] ** 2)
    target_pitch = math.atan2(-diff[1], horiz_dist) if horiz_dist > 0.01 else 0.0
    pitch_err = target_pitch - agent_pitch

    return yaw_err, pitch_err


def aim_action(agent_x, agent_y, agent_z, agent_yaw, agent_pitch,
               target_point: np.ndarray, hotbar: int = PLANKS_SLOT) -> np.ndarray:
    """Generate camera-only action to aim toward target_point."""
    yaw_err, pitch_err = compute_aim_error(
        agent_x, agent_y, agent_z, agent_yaw, agent_pitch, target_point
    )
    action = noop(hotbar)
    action[ACT_YAW] = error_to_camera_idx(-yaw_err)    # negated: our yaw = -MC yaw
    action[ACT_PITCH] = error_to_camera_idx(pitch_err)
    return action


def fine_aim_to_center(env, target_block: tuple,
                       hotbar: int = PLANKS_SLOT) -> np.ndarray:
    """
    Sub-block fine aiming: when raycast already hits the correct block
    but not near center, adjust pitch first, then sneak-move if needed.

    Returns action that nudges aim toward block top-face center.
    """
    tx, ty, tz = target_block
    center = np.array([tx + 0.5, ty + 1.0, tz + 0.5])
    yaw_err, pitch_err = compute_aim_error(
        env.agent_x, env.agent_y, env.agent_z,
        env.agent_yaw, env.agent_pitch, center
    )

    action = noop(hotbar)

    # First try pitch correction (most common for forward/back offset)
    if abs(pitch_err) > math.radians(1.0):
        action[ACT_PITCH] = error_to_camera_idx(pitch_err)
        return action

    # Pitch is OK but yaw is off → sneak sideways to adjust
    if abs(yaw_err) > math.radians(1.0):
        action[ACT_SNEAK] = 1  # sneak for precision
        # yaw_err > 0 → target is to the right → strafe right (-X when facing +Z)
        if yaw_err > 0:
            action[ACT_LEFT_RIGHT] = 2  # right
        else:
            action[ACT_LEFT_RIGHT] = 0  # left
        return action

    # Both close enough — return noop (should pass center check)
    return action


def raycast_aim_action(env, target_block: tuple,
                       hotbar: int = PLANKS_SLOT) -> np.ndarray:
    """
    Feedback-based camera correction using actual raycast hit position.

    Compares where the ray currently hits vs the target block and corrects
    pitch/yaw accordingly.  This is immune to 1-tick observation lag because
    we correct based on what the game actually reports, not a predicted angle.

    Uses the agent's actual facing direction (yaw) to project the hit/target
    offsets onto forward and right axes — works for any yaw direction.

    Pitch:
        hit_fwd > target_fwd → ray lands further than target → too horizontal
                               → increase pitch (look more down)
        hit_fwd < target_fwd → ray lands closer than target  → too steep
                               → decrease pitch (look more up)

    Yaw:
        hit_right > target_right → ray lands to the right of target
                                  → turn left (positive MC yaw delta)
        hit_right < target_right → ray lands to the left of target
                                  → turn right (negative MC yaw delta)

    When raycast misses (no hit), look more steeply down.
    """
    tx, ty, tz = target_block
    action = noop(hotbar)

    if env._cg_obs is None:
        return action

    hit = env._cg_obs_extractor.extract_raycast(env._cg_obs)

    if hit is None:
        # Ray hits nothing → looking too horizontal, look down more
        action[ACT_PITCH] = 7  # +3°
        return action

    hx, hy, hz = hit["position"]

    # Agent's forward direction in XZ: (sin_yaw, cos_yaw) for (X, Z)
    # (our_yaw=0 → facing +Z → forward=(0,1) ✓)
    yaw = env.agent_yaw
    fwd_x = math.sin(yaw)
    fwd_z = math.cos(yaw)
    # Right axis (rotate forward 90° clockwise in XZ): (cos_yaw, -sin_yaw)
    rgt_x = math.cos(yaw)
    rgt_z = -math.sin(yaw)

    ax, az = env.agent_x, env.agent_z

    # Block centers
    hit_cx, hit_cz = hx + 0.5, hz + 0.5
    tgt_cx, tgt_cz = tx + 0.5, tz + 0.5

    # Forward-projected distance from agent to each block center
    hit_fwd = (hit_cx - ax) * fwd_x + (hit_cz - az) * fwd_z
    tgt_fwd = (tgt_cx - ax) * fwd_x + (tgt_cz - az) * fwd_z

    # Right-projected distance from agent to each block center
    hit_rgt = (hit_cx - ax) * rgt_x + (hit_cz - az) * rgt_z
    tgt_rgt = (tgt_cx - ax) * rgt_x + (tgt_cz - az) * rgt_z

    # --- Pitch correction ---
    # fwd_diff > 0 → hit further than target → too horizontal → more pitch (down)
    fwd_diff = int(round(hit_fwd - tgt_fwd))
    action[ACT_PITCH] = _block_err_to_camera_idx(fwd_diff, positive_is_down=True)

    # --- Yaw correction ---
    # rgt_diff > 0 → hit is to the RIGHT of target → turn LEFT
    # Turn left = decrease our_yaw = increase MC yaw = POSITIVE MC yaw delta
    # Positive delta → higher CAMERA_DELTA_MAP index
    rgt_diff = int(round(hit_rgt - tgt_rgt))
    action[ACT_YAW] = _block_err_to_camera_idx(rgt_diff, positive_is_down=False)

    return action


def is_aimed(agent_x, agent_y, agent_z, agent_yaw, agent_pitch,
             target_point: np.ndarray, tolerance: float = AIM_TOLERANCE_RAD) -> bool:
    """Check if agent is aimed close enough to target_point."""
    yaw_err, pitch_err = compute_aim_error(
        agent_x, agent_y, agent_z, agent_yaw, agent_pitch, target_point
    )
    return abs(yaw_err) < tolerance and abs(pitch_err) < tolerance


# ==============================================================================
# Precision movement (yaw must be fixed to a known direction)
# ==============================================================================

def move_x_action(agent_x: float, target_x: float,
                  hotbar: int = PLANKS_SLOT) -> tuple:
    """
    Generate action to move along X axis (via strafe when facing +Z).

    Returns:
        (action, arrived) — action to take, whether we've arrived.
    """
    dx = target_x - agent_x
    action = noop(hotbar)

    if abs(dx) <= FINE_THRESHOLD:
        return action, True

    # Direction: MC left = +X, MC right = -X (when facing +Z south)
    if dx > 0:
        action[ACT_LEFT_RIGHT] = 0  # strafe left → +X
    else:
        action[ACT_LEFT_RIGHT] = 2  # strafe right → -X

    # Sneak for precision
    if abs(dx) <= COARSE_THRESHOLD:
        action[ACT_SNEAK] = 1

    return action, False


def move_z_action(agent_z: float, target_z: float,
                  hotbar: int = PLANKS_SLOT) -> tuple:
    """
    Generate action to move along Z axis (via forward/back when facing +Z).

    Returns:
        (action, arrived) — action to take, whether we've arrived.
    """
    dz = target_z - agent_z
    action = noop(hotbar)

    if abs(dz) <= FINE_THRESHOLD:
        return action, True

    # Direction: forward = +Z, back = -Z (when facing +Z)
    if dz > 0:
        action[ACT_FWD_BACK] = 2  # forward
    else:
        action[ACT_FWD_BACK] = 0  # back

    # Sneak for precision
    if abs(dz) <= COARSE_THRESHOLD:
        action[ACT_SNEAK] = 1

    return action, False


def move_to_xz_action(agent_x: float, agent_z: float,
                      target_x: float, target_z: float,
                      hotbar: int = PLANKS_SLOT) -> tuple:
    """
    Move to (target_x, target_z): X first, then Z.

    Returns:
        (action, arrived)
    """
    # X first
    action, x_arrived = move_x_action(agent_x, target_x, hotbar)
    if not x_arrived:
        return action, False

    # Then Z
    action, z_arrived = move_z_action(agent_z, target_z, hotbar)
    if not z_arrived:
        return action, False

    return noop(hotbar), True


# ==============================================================================
# Yaw fixing
# ==============================================================================

_fix_yaw_prev_sign = [0.0]  # mutable state: previous error sign

def reset_fix_yaw_state():
    """Reset yaw fix dampening state. Call when switching states/experts."""
    _fix_yaw_prev_sign[0] = 0.0

def fix_yaw_action(agent_yaw: float, target_yaw: float = 0.0,
                   hotbar: int = PLANKS_SLOT,
                   tolerance_deg: float = 2.0) -> tuple:
    """
    Generate action to fix yaw to target_yaw.
    When error direction flips (overshoot detected), halve the correction.

    Returns:
        (action, fixed) — whether yaw is close enough.
    """
    yaw_err = target_yaw - agent_yaw
    yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

    if abs(yaw_err) < math.radians(tolerance_deg):
        _fix_yaw_prev_sign[0] = 0.0
        return noop(hotbar), True

    yaw_err_deg = math.degrees(yaw_err)

    # Near ±180°: always turn right with small steps to avoid oscillation
    if abs(yaw_err_deg) > 170:
        corrected_err = math.radians(3.0)  # always right, 3° step
        _fix_yaw_prev_sign[0] = 1.0
        action = noop(hotbar)
        action[ACT_YAW] = error_to_camera_idx(-corrected_err)
        return action, False

    # Detect direction flip → overshoot → halve correction
    cur_sign = 1.0 if yaw_err > 0 else -1.0
    if _fix_yaw_prev_sign[0] != 0.0 and cur_sign != _fix_yaw_prev_sign[0]:
        corrected_err = yaw_err * 0.5
    else:
        corrected_err = yaw_err
    _fix_yaw_prev_sign[0] = cur_sign

    action = noop(hotbar)
    action[ACT_YAW] = error_to_camera_idx(-corrected_err)
    return action, False


# ==============================================================================
# Raycast verification
# ==============================================================================

def check_raycast_hits_block(env, target_block: tuple,
                             expected_face: tuple = None,
                             check_center: bool = True,
                             center_tolerance_deg: float = 3.0) -> bool:
    """
    Check if current raycast hits the specified block near its center.

    Args:
        env: HouseBuildingEnv with _cg_obs and _cg_obs_extractor
        target_block: (x, y, z) expected hit block
        expected_face: (nx, ny, nz) expected face normal, or None for any face
        check_center: if True, verify gaze aims near block top-face center
        center_tolerance_deg: max angle error from center (degrees)

    Returns:
        True if raycast hits the correct block (and face/center if specified).
    """
    if env._cg_obs is None:
        return False

    hit = env._cg_obs_extractor.extract_raycast(env._cg_obs)
    if hit is None:
        return False

    hx, hy, hz = hit["position"]
    if (int(hx), int(hy), int(hz)) != target_block:
        return False

    if expected_face is not None:
        if hit["face_normal"] != expected_face:
            return False

    # Verify gaze pitch is aimed near block top-face center
    if check_center:
        tx, ty, tz = target_block
        center = np.array([tx + 0.5, ty + 1.0, tz + 0.5])
        _, pitch_err = compute_aim_error(
            env.agent_x, env.agent_y, env.agent_z,
            env.agent_yaw, env.agent_pitch, center
        )
        tol = math.radians(center_tolerance_deg)
        if abs(pitch_err) > tol:
            return False

    return True


# ==============================================================================
# Movement Controller (stateful, with stuck detection + auto-jump)
# ==============================================================================

STUCK_MOVE_THRESHOLD = 0.02  # if moved less than this in one tick, consider stuck
STUCK_TICKS_BEFORE_JUMP = 3  # jump after this many stuck ticks

class MovementController:
    """
    Stateful movement controller that tracks position history
    and automatically adds jump when stuck (blocked by a block).

    Usage:
        mc = MovementController()
        action, arrived = mc.move_to(env, target_x, target_z, hotbar)
        # Call mc.reset() at episode start or phase change
    """

    def __init__(self):
        self.prev_x = None
        self.prev_z = None
        self.stuck_ticks = 0

    def reset(self):
        self.prev_x = None
        self.prev_z = None
        self.stuck_ticks = 0

    def move_to(self, env, target_x: float, target_z: float,
                hotbar: int = PLANKS_SLOT) -> tuple:
        """
        Move to (target_x, target_z) with auto-jump on stuck.

        Returns:
            (action, arrived)
        """
        agent_x = env.agent_x
        agent_z = env.agent_z

        # Check if stuck (position barely changed since last tick)
        if self.prev_x is not None:
            moved = math.sqrt(
                (agent_x - self.prev_x) ** 2 +
                (agent_z - self.prev_z) ** 2
            )
            if moved < STUCK_MOVE_THRESHOLD:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0

        self.prev_x = agent_x
        self.prev_z = agent_z

        # Generate base movement action (X first, then Z)
        action, arrived = move_to_xz_action(
            agent_x, agent_z, target_x, target_z, hotbar
        )

        if arrived:
            self.stuck_ticks = 0
            return action, True

        # If stuck for several ticks, add jump
        if self.stuck_ticks >= STUCK_TICKS_BEFORE_JUMP:
            action[ACT_JUMP] = 1
            # Don't sneak while jumping (sneak prevents jump)
            action[ACT_SNEAK] = 0

        return action, False

