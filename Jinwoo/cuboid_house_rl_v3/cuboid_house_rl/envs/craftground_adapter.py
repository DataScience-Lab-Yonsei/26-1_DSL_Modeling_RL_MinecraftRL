"""
CraftGround Adapter (V3).

Bridges between HouseBuildingEnv and the CraftGround Minecraft client.
Handles:
- Environment creation via craftground.make()
- Action conversion: MultiDiscrete → CraftGround dict
- Observation extraction: agent state + raycast only
- Block type mapping

V3 changes:
- No world array, no surrounding_blocks usage
- No fixed spawn/house position
- No worldborder
- Infinite superflat world
- 4-slot inventory (planks, axe, door, glass)
"""
import math
import numpy as np
from typing import Optional, Dict, Tuple

from cuboid_house_rl.config import (
    AIR, OAK_PLANKS, SOLID, NUM_BLOCK_TYPES,
    MINECRAFT_GROUND_Y,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    CAMERA_DELTA_MAP, SLOT_PLANKS,
    RAYCAST_MAX_DISTANCE,
    INVENTORY_ITEMS,
)


# ======================================================================
# Block ID Mapping
# ======================================================================

_BLOCK_SHORT_NAME_TO_INTERNAL = {
    "air": AIR,
    "cave_air": AIR,
    "void_air": AIR,
    "oak_planks": OAK_PLANKS,
}
DEFAULT_BLOCK_TYPE = SOLID


def minecraft_block_to_internal(block_name: str) -> int:
    """Convert Minecraft block name to internal block type."""
    if not block_name:
        return DEFAULT_BLOCK_TYPE
    name = block_name.strip()
    if name.startswith("block.minecraft."):
        short = name[len("block.minecraft."):]
        return _BLOCK_SHORT_NAME_TO_INTERNAL.get(short, DEFAULT_BLOCK_TYPE)
    if ":" in name:
        short = name.split(":", 1)[1]
        return _BLOCK_SHORT_NAME_TO_INTERNAL.get(short, DEFAULT_BLOCK_TYPE)
    return _BLOCK_SHORT_NAME_TO_INTERNAL.get(name, DEFAULT_BLOCK_TYPE)


# ======================================================================
# Action Conversion
# ======================================================================

def multi_discrete_to_craftground(action: np.ndarray) -> dict:
    """
    Convert MultiDiscrete action to CraftGround V2 action dict.

    Action: MultiDiscrete([3, 3, 2, 2, 3, 9, 7, 7])
    """
    try:
        from craftground.environment.action_space import no_op_v2
    except ImportError:
        try:
            from craftground.minecraft import no_op_v2
        except ImportError:
            from craftground import no_op_v2

    a = no_op_v2()

    a["forward"] = 1 if int(action[ACT_FWD_BACK]) == 2 else 0
    a["back"] = 1 if int(action[ACT_FWD_BACK]) == 0 else 0
    a["left"] = 1 if int(action[ACT_LEFT_RIGHT]) == 0 else 0
    a["right"] = 1 if int(action[ACT_LEFT_RIGHT]) == 2 else 0

    a["jump"] = int(action[ACT_JUMP])
    a["sneak"] = int(action[ACT_SNEAK])

    interact = int(action[ACT_INTERACT])
    a["use"] = 1 if interact == 0 else 0
    a["attack"] = 1 if interact == 2 else 0

    slot = int(action[ACT_HOTBAR])
    if 0 <= slot < 9:
        a[f"hotbar.{slot + 1}"] = 1

    pitch_idx = int(action[ACT_PITCH])
    yaw_idx = int(action[ACT_YAW])
    a["camera_pitch"] = CAMERA_DELTA_MAP[pitch_idx]
    a["camera_yaw"] = CAMERA_DELTA_MAP[yaw_idx]

    return a


# ======================================================================
# CraftGround Environment Factory
# ======================================================================

def create_craftground_env(port: int = 8023,
                           image_width: int = 64,
                           image_height: int = 64) -> "gymnasium.Env":
    """
    Create CraftGround environment for house building.

    V3: infinite superflat, no worldborder, no fixed spawn.
    """
    import craftground
    from craftground import InitialEnvironmentConfig, ActionSpaceVersion
    from craftground.initial_environment_config import (
        DaylightMode, WorldType, GameMode,
    )
    from craftground.screen_encoding_modes import ScreenEncodingMode

    img_w, img_h = image_width, image_height

    env = craftground.make(
        port=port,
        initial_env_config=InitialEnvironmentConfig(
            image_width=img_w,
            image_height=img_h,
            gamemode=GameMode.SURVIVAL,
            world_type=WorldType.SUPERFLAT,
            generate_structures=False,
            hud_hidden=False,
            render_distance=6,
            simulation_distance=6,
            request_raycast=True,
            requires_surrounding_blocks=False,  # V3: not needed
            screen_encoding_mode=ScreenEncodingMode.RAW,
            initial_extra_commands=[
                # Kill any pre-spawned mobs before they can attack
                "kill @e[type=!player]",
                # Give inventory items in correct hotbar order:
                # slot 0 (SLOT_PLANKS), slot 1 (SLOT_AXE), slot 2 (SLOT_DOOR), slot 3 (SLOT_GLASS)
                "give @p oak_planks 64",    # hotbar slot 1
                "give @p diamond_axe 1",    # hotbar slot 2
                "give @p oak_door 64",      # hotbar slot 3
                "give @p glass 64",         # hotbar slot 4
                "give @p oak_planks 576",   # extra planks in remaining slots
                # Environment settings
                "gamerule doDaylightCycle false",
                "time set day",
                "gamerule doWeatherCycle false",
                "weather clear",
                "gamerule doMobSpawning false",
                "difficulty peaceful",
                "gamerule fallDamage false",
                "effect give @p saturation 999999 0 true",
                "effect give @p resistance 999999 4 true",
                "effect give @p fire_resistance 999999 0 true",
            ],
        ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )

    return env


# ======================================================================
# Observation Extraction
# ======================================================================

# Runtime Y offset
_y_offset = MINECRAFT_GROUND_Y


def get_y_offset() -> int:
    if _y_offset is None:
        raise RuntimeError("Y offset not yet detected.")
    return _y_offset


def detect_ground_y(env) -> int:
    """Auto-detect superflat ground level."""
    global _y_offset

    if _y_offset is not None:
        return _y_offset

    from craftground.environment.action_space import no_op_v2

    print("  [auto-detect] Detecting superflat ground level...")
    noop = no_op_v2()

    obs = None
    for _ in range(80):
        obs, _, _, _, _ = env.step(noop)

    if obs is None:
        raise RuntimeError("Failed to get observation for ground detection")

    proto = obs["full"] if isinstance(obs, dict) else obs
    player_y = float(proto.y)
    ground_y = int(math.floor(player_y)) - 1
    _y_offset = ground_y

    print(f"  [auto-detect] Player Y={player_y:.1f}, ground Y={ground_y}")
    return _y_offset


class CraftGroundObsExtractor:
    """
    Extracts agent state and raycast from CraftGround observations.

    V3: No surrounding blocks extraction. No world array updates.
    """

    def __init__(self):
        self._prev_hotbar_slot = SLOT_PLANKS

    @staticmethod
    def _get_proto(obs):
        if isinstance(obs, dict):
            return obs["full"]
        return obs

    def extract_agent_state(self, obs) -> Dict:
        """Extract agent position and orientation."""
        proto = self._get_proto(obs)

        x = float(proto.x)
        y = float(proto.y) - get_y_offset()
        z = float(proto.z)

        mc_yaw_deg = float(proto.yaw)
        mc_pitch_deg = float(proto.pitch)

        # Convert: our_yaw = -minecraft_yaw (radians)
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

    def extract_inventory(self, obs) -> list:
        """Extract inventory item counts. Returns list of (translation_key, count)."""
        proto = self._get_proto(obs)
        items = []
        if hasattr(proto, 'inventory'):
            for item in proto.inventory:
                items.append({
                    "translation_key": str(item.translation_key) if hasattr(item, 'translation_key') else "",
                    "count": int(item.count) if hasattr(item, 'count') else 0,
                    "raw_id": int(item.raw_id) if hasattr(item, 'raw_id') else 0,
                })
        return items

    def get_hotbar_planks_count(self, obs, slot: int) -> int:
        """Get planks count in a specific hotbar slot. Returns 0 if empty."""
        items = self.extract_inventory(obs)
        if slot < len(items):
            item = items[slot]
            if "planks" in item["translation_key"]:
                return item["count"]
        return 0

    def extract_raycast(self, obs) -> Optional[Dict]:
        """Extract raycast hit info."""
        proto = self._get_proto(obs)

        if not hasattr(proto, 'raycast_result') or proto.raycast_result is None:
            return None

        raycast = proto.raycast_result
        hit_type = raycast.type if hasattr(raycast, 'type') else 0
        if hit_type != 1:  # Not BLOCK hit
            return None

        target = raycast.target_block
        if target is None:
            return None

        hx = int(target.x)
        hy = int(target.y) - get_y_offset()
        hz = int(target.z)

        block_name = str(target.translation_key) if hasattr(target, 'translation_key') else ""
        block_type = minecraft_block_to_internal(block_name)

        agent_x = float(proto.x)
        agent_y = float(proto.y) - get_y_offset()
        agent_z = float(proto.z)
        agent_eye_y = agent_y + 1.62
        distance = math.sqrt(
            (hx + 0.5 - agent_x) ** 2 +
            (hy + 0.5 - agent_eye_y) ** 2 +
            (hz + 0.5 - agent_z) ** 2
        )

        if distance > RAYCAST_MAX_DISTANCE:
            return None

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
        eye_x, eye_y, eye_z,
        yaw_deg, pitch_deg,
        block_x, block_y, block_z,
    ) -> Tuple[int, int, int]:
        """Compute which face of the block was hit by the look ray."""
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        cos_pitch = math.cos(pitch_rad)

        dx = -math.sin(yaw_rad) * cos_pitch
        dy = -math.sin(pitch_rad)
        dz = math.cos(yaw_rad) * cos_pitch

        def axis_entry_t(origin, direction, block_min):
            if direction == 0:
                return float('-inf')
            if direction > 0:
                return (block_min - origin) / direction
            else:
                return (block_min + 1.0 - origin) / direction

        tx = axis_entry_t(eye_x, dx, block_x)
        ty = axis_entry_t(eye_y, dy, block_y)
        tz = axis_entry_t(eye_z, dz, block_z)

        max_t = max(tx, ty, tz)

        if max_t == tx:
            return (-1, 0, 0) if dx > 0 else (1, 0, 0)
        elif max_t == ty:
            return (0, -1, 0) if dy > 0 else (0, 1, 0)
        else:
            return (0, 0, -1) if dz > 0 else (0, 0, 1)

    def update_hotbar_slot(self, action: np.ndarray):
        self._prev_hotbar_slot = int(action[ACT_HOTBAR])
