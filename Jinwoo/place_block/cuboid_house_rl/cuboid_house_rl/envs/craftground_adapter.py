"""
CraftGround Adapter.

Bridges between our HouseBuildingEnv and the real CraftGround Minecraft client.
Handles:
- Environment creation via craftground.make()
- Action conversion: MultiDiscrete → CraftGround dict
- Observation extraction: CraftGround obs → world state, agent state, raycast
- Block type mapping: Minecraft block IDs → our internal IDs (AIR/OAK_PLANKS/SOLID)
"""
import math
import numpy as np
from typing import Optional, Dict, Tuple

from cuboid_house_rl.config import (
    WORLD_SIZE, AIR, OAK_PLANKS, SOLID, NUM_BLOCK_TYPES,
    MINECRAFT_GROUND_Y,
    SPAWN_X, SPAWN_Y, SPAWN_Z, SPAWN_YAW, SPAWN_PITCH,
    HOUSE_ORIGIN_X, HOUSE_ORIGIN_Z, HOUSE_WIDTH, HOUSE_DEPTH,
    FLOOR_Y, CEILING_Y,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    CAMERA_DELTA_MAP, PLANKS_SLOT,
    LOCAL_WINDOW_SIZE, RAYCAST_MAX_DISTANCE,
)


# ======================================================================
# Minecraft Block ID Mapping
# ======================================================================

# CraftGround BlockInfo uses `translation_key` which has the format:
#   "block.minecraft.oak_planks", "block.minecraft.air", "block.minecraft.stone"
# We also handle the registry name format just in case:
#   "minecraft:oak_planks", "minecraft:air", "minecraft:stone"
# We map these to our internal 3-type system (AIR=0, OAK_PLANKS=1, SOLID=2).

# Mapping using the short block name (after normalization)
_BLOCK_SHORT_NAME_TO_INTERNAL = {
    "air": AIR,
    "cave_air": AIR,
    "void_air": AIR,
    "oak_planks": OAK_PLANKS,
}
# Everything else maps to SOLID (ground, barriers, bedrock, etc.)
DEFAULT_BLOCK_TYPE = SOLID


def minecraft_block_to_internal(block_name: str) -> int:
    """
    Convert a Minecraft block name to our internal block type.

    Handles multiple formats:
        "block.minecraft.oak_planks"  (translation_key from BlockInfo)
        "minecraft:oak_planks"        (registry name format)
        "oak_planks"                  (short name)
    """
    if not block_name:
        return DEFAULT_BLOCK_TYPE

    name = block_name.strip()

    # Format 1: translation_key "block.minecraft.oak_planks"
    if name.startswith("block.minecraft."):
        short = name[len("block.minecraft."):]
        return _BLOCK_SHORT_NAME_TO_INTERNAL.get(short, DEFAULT_BLOCK_TYPE)

    # Format 2: registry name "minecraft:oak_planks"
    if ":" in name:
        short = name.split(":", 1)[1]
        return _BLOCK_SHORT_NAME_TO_INTERNAL.get(short, DEFAULT_BLOCK_TYPE)

    # Format 3: already a short name
    return _BLOCK_SHORT_NAME_TO_INTERNAL.get(name, DEFAULT_BLOCK_TYPE)


# ======================================================================
# Action Conversion
# ======================================================================

def multi_discrete_to_craftground(action: np.ndarray) -> dict:
    """
    Convert our MultiDiscrete action to CraftGround V2 action dict.

    Our action: MultiDiscrete([3, 3, 2, 2, 3, 9, 7, 7])
        [0] fwd/back:   0=back, 1=stop, 2=forward
        [1] left/right: 0=left, 1=stop, 2=right
        [2] jump:       0=no, 1=yes
        [3] sneak:      0=no, 1=yes
        [4] interact:   0=place(use), 1=nothing, 2=attack
        [5] hotbar:     0-8 (slot index)
        [6] pitch:      index into CAMERA_DELTA_MAP
        [7] yaw:        index into CAMERA_DELTA_MAP

    Returns:
        CraftGround action dict compatible with V2_MINERL_HUMAN.
    """
    # no_op_v2 import — try multiple known paths across CraftGround versions
    try:
        from craftground.environment.action_space import no_op_v2
    except ImportError:
        try:
            from craftground.minecraft import no_op_v2
        except ImportError:
            from craftground import no_op_v2

    a = no_op_v2()

    # Movement (3-way → separate binary keys)
    a["forward"] = 1 if int(action[ACT_FWD_BACK]) == 2 else 0
    a["back"] = 1 if int(action[ACT_FWD_BACK]) == 0 else 0
    a["left"] = 1 if int(action[ACT_LEFT_RIGHT]) == 0 else 0
    a["right"] = 1 if int(action[ACT_LEFT_RIGHT]) == 2 else 0

    # Jump and sneak
    a["jump"] = int(action[ACT_JUMP])
    a["sneak"] = int(action[ACT_SNEAK])

    # Interact: 0=place(use), 1=nothing, 2=attack(break)
    interact = int(action[ACT_INTERACT])
    a["use"] = 1 if interact == 0 else 0
    a["attack"] = 1 if interact == 2 else 0

    # Hotbar slot selection (1-indexed in CraftGround, uses dot notation)
    slot = int(action[ACT_HOTBAR])
    if 0 <= slot < 9:
        a[f"hotbar.{slot + 1}"] = 1

    # Camera deltas (degrees per step)
    pitch_idx = int(action[ACT_PITCH])
    yaw_idx = int(action[ACT_YAW])
    a["camera_pitch"] = CAMERA_DELTA_MAP[pitch_idx]
    a["camera_yaw"] = CAMERA_DELTA_MAP[yaw_idx]

    return a


# ======================================================================
# CraftGround Environment Factory
# ======================================================================

def create_craftground_env(port: int = 8023) -> "gymnasium.Env":
    """
    Create and configure a CraftGround environment for house building.

    Args:
        port: TCP port for the CraftGround Minecraft server instance.

    Returns:
        A CraftGround Gymnasium environment.
    """
    import craftground
    from craftground import InitialEnvironmentConfig, ActionSpaceVersion
    from craftground.initial_environment_config import (
        DaylightMode, WorldType, GameMode,
    )
    from craftground.screen_encoding_modes import ScreenEncodingMode

    env = craftground.make(
        port=port,
        initial_env_config=InitialEnvironmentConfig(
            image_width=64,
            image_height=64,
            gamemode=GameMode.SURVIVAL,
            world_type=WorldType.SUPERFLAT,
            hud_hidden=False,  # Show health/food/hotbar HUD
            render_distance=6,
            simulation_distance=6,
            request_raycast=True,
            # Note: surrounding_blocks provides only a 3x3x3 grid per the API.
            # We use it to incrementally update a persistent world array each step,
            # then extract the 11x11x11 local window from that array.
            requires_surrounding_blocks=True,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            initial_extra_commands=[
                # Give plenty of oak planks (10 stacks)
                "give @p oak_planks 640",
                # Teleport to spawn (correct Y from MINECRAFT_GROUND_Y)
                f"tp @p {SPAWN_X} {SPAWN_Y + get_y_offset()} {SPAWN_Z}",
                # World border to keep agent in area
                f"worldborder center {WORLD_SIZE // 2} {WORLD_SIZE // 2}",
                f"worldborder set {WORLD_SIZE}",
                # Always day, no weather, no mobs
                "gamerule doDaylightCycle false",
                "time set day",
                "gamerule doWeatherCycle false",
                "weather clear",
                "gamerule doMobSpawning false",
                # Prevent hunger drain: permanent saturation keeps food bar full
                "effect give @p saturation 999999 0 true",
                # Disable fall damage during construction
                "gamerule fallDamage false",
            ],
        ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )

    return env


# ======================================================================
# Observation Extraction from CraftGround
# ======================================================================

# Runtime Y offset — set by detect_ground_y() on first env reset.
# When MINECRAFT_GROUND_Y is None in config, this gets auto-detected.
_y_offset = MINECRAFT_GROUND_Y  # may be None initially


def get_y_offset() -> int:
    """Return the current Y offset (internal_y = mc_y - offset)."""
    if _y_offset is None:
        raise RuntimeError(
            "Y offset not yet detected. Call detect_ground_y() first, "
            "or set MINECRAFT_GROUND_Y in config.py."
        )
    return _y_offset


def detect_ground_y(env) -> int:
    """
    Auto-detect the superflat ground level by teleporting high and
    reading where the player lands.

    Args:
        env: CraftGround gymnasium environment (unwrapped).

    Returns:
        Minecraft Y coordinate of the grass/surface block.
    """
    global _y_offset

    if _y_offset is not None:
        return _y_offset

    from craftground.environment.action_space import no_op_v2

    print("  [auto-detect] Detecting superflat ground level...")

    noop = no_op_v2()

    # The env must already be reset (socket connected).
    # initial_extra_commands already teleported to Y=200.
    # Wait for the player to fall and land on the ground.
    obs = None
    for _ in range(80):  # ~4 seconds at 20 tps
        obs, _, _, _, _ = env.step(noop)

    if obs is None:
        raise RuntimeError("Failed to get observation for ground detection")

    proto = obs["full"] if isinstance(obs, dict) else obs

    # proto.y is the player's feet position. The block they stand on
    # is one below: ground_y = floor(proto.y) - 1
    player_y = float(proto.y)
    ground_y = int(math.floor(player_y)) - 1

    # In our internal system, the ground surface is at GROUND_Y=0.
    # So MINECRAFT_GROUND_Y = ground_y maps internal 0 → MC ground.
    # And SPAWN_Y=1 (internal) → MC ground_y+1 (standing on ground).
    _y_offset = ground_y

    print(f"  [auto-detect] Player Y={player_y:.1f}, "
          f"ground block Y={ground_y}, MINECRAFT_GROUND_Y={_y_offset}")

    return _y_offset


class CraftGroundObsExtractor:
    """
    Extracts structured observations from CraftGround's raw observation.

    CraftGround provides:
    - Visual image (via VisionWrapper or raw obs)
    - Player position, rotation
    - Raycast hit data (when request_raycast=True)
    - Surrounding blocks (when requires_surrounding_blocks=True)

    We extract:
    - Agent position (x, y, z)
    - Agent orientation (yaw, pitch) in radians
    - Raycast hit info (position, distance, block type, face normal)
    - Local voxel grid from surrounding blocks
    - Current hotbar slot
    """

    def __init__(self):
        self._prev_hotbar_slot = PLANKS_SLOT

    @staticmethod
    def _get_proto(obs):
        """
        Extract the ObservationSpaceMessage protobuf from the CraftGround obs.

        CraftGround env.step() returns obs as a dict:
            obs["full"]  -> ObservationSpaceMessage (protobuf with x, y, z, yaw, ...)
            obs["pov"]   -> np.ndarray or torch.Tensor (RGB image)

        This helper handles both the dict format and a raw protobuf
        (for forward-compatibility or testing).
        """
        if isinstance(obs, dict):
            return obs["full"]
        # Fallback: assume obs itself is the protobuf
        return obs

    def extract_agent_state(self, obs) -> Dict:
        """
        Extract agent position and orientation from CraftGround observation.

        Yaw convention conversion:
            Minecraft: yaw=0 → south(+z), +90 → west(-x)
                       forward_x = -sin(yaw), forward_z = cos(yaw)
            Our code:  yaw=0 → north(+z),  +yaw → east(+x)
                       forward_x = sin(yaw),  forward_z = cos(yaw)

            Mapping: our_yaw = -minecraft_yaw (in radians)

        Pitch convention: both use positive pitch = looking down,
            with dy = -sin(pitch), so no conversion needed.

        Args:
            obs: CraftGround observation dict from env.step().

        Returns:
            dict with keys: x, y, z, yaw, pitch, hotbar_slot
        """
        proto = self._get_proto(obs)

        x = float(proto.x)
        y = float(proto.y) - get_y_offset()  # convert to internal coords
        z = float(proto.z)

        # Convert Minecraft degrees → our radians convention
        mc_yaw_deg = float(proto.yaw)
        mc_pitch_deg = float(proto.pitch)

        # Negate yaw to convert Minecraft convention → our convention
        yaw_rad = -math.radians(mc_yaw_deg)
        pitch_rad = math.radians(mc_pitch_deg)

        return {
            "x": x,
            "y": y,
            "z": z,
            "yaw": yaw_rad,
            "pitch": pitch_rad,
            "hotbar_slot": self._prev_hotbar_slot,
        }

    def extract_raycast(self, obs) -> Optional[Dict]:
        """
        Extract raycast hit information from CraftGround observation.

        CraftGround HitResult (when request_raycast=True):
            type:          Type enum (MISS=0, BLOCK=1, ENTITY=2)
            target_block:  BlockInfo (x, y, z, translation_key)
            target_entity: EntityInfo (for entity hits)

        Note: The API does NOT provide a face normal. We compute it
        from the agent's eye position and the hit block center.

        Returns:
            dict with position, distance, block_type, face_normal
            or None if no hit.
        """
        proto = self._get_proto(obs)

        if not hasattr(proto, 'raycast_result') or proto.raycast_result is None:
            return None

        raycast = proto.raycast_result

        # HitResult.type: 0=MISS, 1=BLOCK, 2=ENTITY
        # Check for block hit (we only care about blocks, not entities)
        hit_type = raycast.type if hasattr(raycast, 'type') else 0
        if hit_type != 1:  # Not a BLOCK hit
            return None

        # Get target_block (BlockInfo with x, y, z, translation_key)
        target = raycast.target_block
        if target is None:
            return None

        hx = int(target.x)
        hy = int(target.y) - get_y_offset()  # convert to internal coords
        hz = int(target.z)

        # Block type from translation_key
        block_name = str(target.translation_key) if hasattr(target, 'translation_key') else ""
        block_type = minecraft_block_to_internal(block_name)

        # Compute distance from agent eye to hit block center (internal coords)
        agent_x = float(proto.x)
        agent_y = float(proto.y) - get_y_offset()  # internal coords
        agent_z = float(proto.z)
        agent_eye_y = agent_y + 1.6  # eye height above feet
        distance = math.sqrt(
            (hx + 0.5 - agent_x) ** 2 +
            (hy + 0.5 - agent_eye_y) ** 2 +
            (hz + 0.5 - agent_z) ** 2
        )

        if distance > RAYCAST_MAX_DISTANCE:
            return None

        # Compute face normal (API doesn't provide this directly).
        # We determine which face the ray hit by finding which axis
        # has the smallest penetration from the agent's eye position.
        face_normal = self._compute_face_normal(
            agent_x, agent_eye_y, agent_z,
            float(proto.yaw), float(proto.pitch),
            hx, hy, hz,
        )

        return {
            "position": (hx, hy, hz),
            "distance": distance,
            "block_type": block_type,
            "face_normal": face_normal,
        }

    @staticmethod
    def _compute_face_normal(
        eye_x: float, eye_y: float, eye_z: float,
        yaw_deg: float, pitch_deg: float,
        block_x: int, block_y: int, block_z: int,
    ) -> Tuple[int, int, int]:
        """
        Compute which face of the block was hit by the agent's look ray.

        Since CraftGround's HitResult doesn't include face information,
        we cast a short ray from the eye toward the block and determine
        which face boundary is crossed first using DDA-style logic.

        Args:
            eye_x, eye_y, eye_z: agent eye position (world coords).
            yaw_deg, pitch_deg: agent look direction in degrees.
            block_x, block_y, block_z: integer coords of the hit block.

        Returns:
            (nx, ny, nz) face normal tuple with one component ±1, others 0.
        """
        # Ray direction from yaw/pitch (degrees)
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        cos_pitch = math.cos(pitch_rad)

        # Minecraft convention: yaw 0=south(+z), 90=west(-x)
        # sin/cos mapping for Minecraft yaw:
        dx = -math.sin(yaw_rad) * cos_pitch
        dy = -math.sin(pitch_rad)
        dz = math.cos(yaw_rad) * cos_pitch

        # For each axis, compute how far along the ray we'd travel
        # to reach the nearest face of the block from the eye.
        # The block occupies [block_x, block_x+1) × [block_y, block_y+1) × ...
        # The face we hit is the one with the LARGEST entry t (last face crossed
        # to enter the block).

        def axis_entry_t(origin, direction, block_min):
            """Compute parametric t where ray enters the block on one axis."""
            if direction == 0:
                return float('-inf')
            if direction > 0:
                # Ray enters through the low face
                return (block_min - origin) / direction
            else:
                # Ray enters through the high face
                return (block_min + 1.0 - origin) / direction

        tx = axis_entry_t(eye_x, dx, block_x)
        ty = axis_entry_t(eye_y, dy, block_y)
        tz = axis_entry_t(eye_z, dz, block_z)

        # The face we entered through has the largest t value
        max_t = max(tx, ty, tz)

        if max_t == tx:
            return (-1, 0, 0) if dx > 0 else (1, 0, 0)
        elif max_t == ty:
            return (0, -1, 0) if dy > 0 else (0, 1, 0)
        else:
            return (0, 0, -1) if dz > 0 else (0, 0, 1)

    def extract_surrounding_blocks(self, obs) -> np.ndarray:
        """
        Extract the surrounding blocks grid from CraftGround observation.

        CraftGround provides surrounding blocks as repeated BlockInfo
        when requires_surrounding_blocks=True.

        Returns:
            3D numpy array of shape (LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE)
            with our internal block type IDs.
        """
        proto = self._get_proto(obs)
        half = LOCAL_WINDOW_SIZE // 2
        grid = np.full(
            (LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE),
            AIR, dtype=np.int8,
        )

        if not hasattr(proto, 'surrounding_blocks') or proto.surrounding_blocks is None:
            return grid

        blocks = proto.surrounding_blocks

        # BlockInfo.x/y/z are absolute world coordinates (int32).
        # We need to convert them to local grid indices relative to the agent.
        agent_bx = int(math.floor(float(proto.x)))
        agent_by = int(math.floor(float(proto.y)))
        agent_bz = int(math.floor(float(proto.z)))

        if hasattr(blocks, '__iter__'):
            for block in blocks:
                if hasattr(block, 'x') and hasattr(block, 'y') and hasattr(block, 'z'):
                    bx, by, bz = int(block.x), int(block.y), int(block.z)

                    # Convert absolute → agent-relative → grid index
                    rx = (bx - agent_bx) + half
                    ry = (by - agent_by) + half
                    rz = (bz - agent_bz) + half

                    if (0 <= rx < LOCAL_WINDOW_SIZE and
                        0 <= ry < LOCAL_WINDOW_SIZE and
                        0 <= rz < LOCAL_WINDOW_SIZE):
                        block_name = str(block.translation_key) if hasattr(block, 'translation_key') else ""
                        grid[rx, ry, rz] = minecraft_block_to_internal(block_name)

        return grid

    def build_world_from_surrounding(
        self,
        obs,
        existing_world: np.ndarray,
    ) -> np.ndarray:
        """
        Update the world array using surrounding blocks from CraftGround.

        We maintain a persistent world array and update the region around the
        agent each step using the surrounding blocks data.

        Args:
            obs: CraftGround observation.
            existing_world: current world state array.

        Returns:
            Updated world array.
        """
        proto = self._get_proto(obs)

        if not hasattr(proto, 'surrounding_blocks') or proto.surrounding_blocks is None:
            return existing_world

        blocks = proto.surrounding_blocks

        # BlockInfo.x/y/z are absolute Minecraft world coordinates.
        # Convert Y to internal array index: internal_y = mc_y - MINECRAFT_GROUND_Y
        if hasattr(blocks, '__iter__'):
            for block in blocks:
                if hasattr(block, 'x') and hasattr(block, 'y') and hasattr(block, 'z'):
                    wx = int(block.x)
                    wy = int(block.y) - get_y_offset()  # to internal
                    wz = int(block.z)

                    if (0 <= wx < WORLD_SIZE and
                        0 <= wy < WORLD_SIZE and
                        0 <= wz < WORLD_SIZE):
                        block_name = str(block.translation_key) if hasattr(block, 'translation_key') else ""
                        existing_world[wx, wy, wz] = minecraft_block_to_internal(block_name)

        return existing_world

    def update_hotbar_slot(self, action: np.ndarray):
        """Track the current hotbar slot from the action."""
        self._prev_hotbar_slot = int(action[ACT_HOTBAR])


# ======================================================================
# Reset Helpers
# ======================================================================

def get_reset_commands() -> list:
    """
    Get commands to reset the world state for a new episode.

    In CraftGround, we send commands to:
    1. Clear any previously placed blocks in the house region
    2. Teleport the agent back to spawn
    3. Re-give oak planks
    """
    commands = []

    # Convert internal Y to Minecraft world Y
    y_off = get_y_offset()
    mc_floor_y = FLOOR_Y + y_off
    mc_ceiling_y = CEILING_Y + y_off
    mc_spawn_y = SPAWN_Y + y_off

    # Clear the house building region (fill with air)
    x0 = HOUSE_ORIGIN_X
    x1 = HOUSE_ORIGIN_X + HOUSE_WIDTH - 1
    z0 = HOUSE_ORIGIN_Z
    z1 = HOUSE_ORIGIN_Z + HOUSE_DEPTH - 1

    # Clear from floor to above ceiling
    commands.append(
        f"fill {x0} {mc_floor_y} {z0} {x1} {mc_ceiling_y} {z1} air"
    )

    # Also clear a slightly larger area around the house (agent may have placed
    # blocks outside the blueprint area)
    margin = 3
    commands.append(
        f"fill {x0-margin} {mc_floor_y} {z0-margin} "
        f"{x1+margin} {mc_ceiling_y+2} {z1+margin} air"
    )

    # Teleport agent to random spawn around house (2 blocks from edge)
    # tp format: /tp @p X Y Z YAW PITCH (MC degrees)
    import math
    import random

    house_cx = HOUSE_ORIGIN_X + HOUSE_WIDTH / 2.0   # 20.5
    house_cz = HOUSE_ORIGIN_Z + HOUSE_DEPTH / 2.0   # 20.5
    dist = 2  # blocks from house edge

    side = random.choice(["north", "south", "east", "west"])
    if side == "south":    # -Z side
        sx = random.randint(x0, x1)
        sz = z0 - dist
    elif side == "north":  # +Z side
        sx = random.randint(x0, x1)
        sz = z1 + dist
    elif side == "west":   # -X side
        sx = x0 - dist
        sz = random.randint(z0, z1)
    else:                  # east, +X side
        sx = x1 + dist
        sz = random.randint(z0, z1)

    # Yaw: face roughly toward house center (±45° noise for gaze learning)
    dx = house_cx - sx
    dz = house_cz - sz
    base_yaw = -math.degrees(math.atan2(dx, dz))
    mc_yaw = int(base_yaw + random.uniform(-45, 45)) % 360
    mc_pitch = int(math.degrees(SPAWN_PITCH))
    commands.append(f"tp @p {sx} {mc_spawn_y} {sz} {mc_yaw} {mc_pitch}")

    # Re-give planks
    commands.append("clear @p")
    commands.append("give @p oak_planks 640")

    # Restore full health and saturation for survival mode
    commands.append("effect give @p instant_health 1 5 true")
    commands.append("effect give @p saturation 999999 0 true")

    return commands
