from __future__ import annotations

from collections import deque
import math
import random
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parent
CRAFTGROUND_SRC = PROJECT_ROOT / "CraftGround" / "src"
if str(CRAFTGROUND_SRC) not in sys.path:
    sys.path.insert(0, str(CRAFTGROUND_SRC))

from craftground import InitialEnvironmentConfig, LidarConfig, make
from craftground.environment.action_space import no_op
from craftground.initial_environment_config import Difficulty, GameMode, WorldType
from craftground.proto.observation_space_pb2 import EntityInfo, HitResult, ObservationSpaceMessage
from craftground.screen_encoding_modes import ScreenEncodingMode


HOSTILE_MOBS = ("husk", "zombie", "skeleton", "spider")
PASSIVE_ANIMALS = ("sheep", "pig", "chicken", "cow", "rabbit")

VECTOR_OBSERVATION_SIZE = 24
DEFAULT_SURROUNDING_RADII = (4, 8, 16, 24)


@dataclass(frozen=True)
class RewardConfig:
    survival_per_tick: float = 0.0015
    combat_survival_per_tick: float = 0.006
    death_penalty: float = -1.0
    damage_scale: float = -0.06
    danger_penalty: float = -0.02
    retreat_reward: float = 0.04
    target_visible_reward: float = 0.01
    hostile_visible_reward: float = 0.006
    aim_improvement_scale: float = 0.04
    target_centered_reward: float = 0.025
    draw_centered_reward: float = 0.03
    raycast_target_reward: float = 0.06
    target_hit_reward: float = 0.4
    target_kill_reward: float = 1.25
    shot_penalty: float = -0.015
    no_shot_penalty: float = -0.15
    no_hit_penalty: float = -0.35
    boundary_penalty: float = -0.02
    idle_step_penalty: float = -0.015
    idle_timeout_penalty: float = -0.25
    movement_reward_scale: float = 0.05
    danger_radius: float = 5.0
    boundary_margin: float = 2.0
    shot_recent_window: int = 12
    idle_speed_threshold: float = 0.03
    idle_timeout_steps: int = 80
    blocked_move_penalty: float = -0.12
    blocked_forward_threshold: float = 1.2
    blocked_side_threshold: float = 0.8
    damage_dealt_scale: float = 0.08
    scan_penalty: float = -0.01
    blind_shot_penalty: float = -0.18
    stationary_close_combat_penalty: float = -0.08
    stationary_bow_penalty: float = -0.1
    stationary_shot_penalty: float = -0.25
    stationary_combat_timeout_penalty: float = -0.5
    kiting_shot_reward: float = 0.12
    comfortable_combat_distance: float = 4.5
    too_close_distance: float = 2.8
    spacing_reward_scale: float = 0.06
    stationary_combat_timeout_steps: int = 28
    stuck_penalty: float = -0.12
    backtrack_escape_reward: float = 0.08
    stuck_horizontal_tolerance: float = 0.45
    stuck_history_length: int = 8
    backtrack_commit_steps: int = 4
    approach_under_pressure_penalty: float = -0.08
    too_close_pressure_penalty: float = -0.18
    engaged_kiting_reward: float = 0.12
    escape_band_reward: float = 0.14
    engagement_alignment_threshold: float = 0.68
    pursuit_reward_scale: float = 0.025
    evasive_move_reward: float = 0.1
    min_distance_reward_scale: float = 0.2
    max_distance_reward_scale: float = 1.2
    draw_alignment_threshold: float = 0.7
    release_alignment_threshold: float = 0.84
    aim_yaw_small_threshold: float = 10.0
    aim_yaw_large_threshold: float = 28.0
    aim_pitch_threshold: float = 10.0


@dataclass(frozen=True)
class StageConfig:
    name: str
    hostile_count: int
    arena_radius: int
    max_steps: int
    no_shot_timeout: int
    no_hit_timeout: int
    hostile_types: tuple[str, ...] = ("husk",)
    animal_count: int = 0
    respawn_interval: int = 40
    wall_height: int = 4
    focus_hostiles: bool = True


STAGE_CONFIGS: dict[str, StageConfig] = {
    "stage1_archery": StageConfig(
        name="stage1_archery",
        hostile_count=2,
        arena_radius=14,
        max_steps=1200,
        no_shot_timeout=140,
        no_hit_timeout=220,
        hostile_types=("zombie",),
        respawn_interval=34,
    ),
    "stage2_survival": StageConfig(
        name="stage2_survival",
        hostile_count=5,
        arena_radius=16,
        max_steps=1400,
        no_shot_timeout=150,
        no_hit_timeout=240,
        hostile_types=("zombie",),
        respawn_interval=30,
    ),
    "stage3_combined_easy": StageConfig(
        name="stage3_combined_easy",
        hostile_count=8,
        arena_radius=18,
        max_steps=1800,
        no_shot_timeout=150,
        no_hit_timeout=240,
        hostile_types=("zombie",),
        respawn_interval=22,
    ),
    "stage4_combined_full": StageConfig(
        name="stage4_combined_full",
        hostile_count=12,
        arena_radius=20,
        max_steps=3000,
        no_shot_timeout=160,
        no_hit_timeout=260,
        hostile_types=("zombie", "zombie", "zombie", "spider"),
        respawn_interval=20,
    ),
    "stage5_hard_generalization": StageConfig(
        name="stage5_hard_generalization",
        hostile_count=14,
        arena_radius=22,
        max_steps=3600,
        no_shot_timeout=170,
        no_hit_timeout=280,
        hostile_types=("zombie", "zombie", "spider", "skeleton"),
        respawn_interval=18,
    ),
}


class BowMacroAction(IntEnum):
    NO_OP = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    SPRINT_FORWARD = 5
    JUMP_FORWARD = 6
    TURN_LEFT_SMALL = 7
    TURN_RIGHT_SMALL = 8
    TURN_LEFT_LARGE = 9
    TURN_RIGHT_LARGE = 10
    LOOK_UP_SMALL = 11
    LOOK_DOWN_SMALL = 12
    DRAW_BOW = 13
    RELEASE_BOW = 14
    SHOOT_ARROW = 15
    CLIMB_FORWARD = 16
    CLIMB_LEFT = 17
    CLIMB_RIGHT = 18


@dataclass
class TrackedEntity:
    unique_id: str
    translation_key: str
    distance: float
    health: float
    visible: bool
    x: float
    y: float
    z: float


@dataclass
class ObservationSummary:
    health: float
    food_level: float
    saturation_level: float
    is_dead: bool
    yaw: float
    pitch: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    x: float
    y: float
    z: float
    hostile_entities: list[TrackedEntity] = field(default_factory=list)
    animal_entities: list[TrackedEntity] = field(default_factory=list)
    hostile_visible: bool = False
    target_visible: bool = False
    nearest_hostile_distance: float = 999.0
    nearest_target_distance: float = 999.0
    nearest_target_alignment: float = 0.0
    target_centered: bool = False
    raycast_hostile: bool = False
    raycast_target: bool = False
    boundary_pressure: float = 0.0
    arrow_count: int = 0
    inventory_has_bow: bool = False
    speed: float = 0.0
    moved_distance: float = 0.0
    front_clearance: float = 1.0
    left_clearance: float = 1.0
    right_clearance: float = 1.0
    blocked_forward: bool = False
    blocked_left: bool = False
    blocked_right: bool = False
    visible_hostile_ids: set[str] = field(default_factory=set)
    visible_target_ids: set[str] = field(default_factory=set)
    raw: Optional[ObservationSpaceMessage] = None


def normalize_name(name: str) -> str:
    return name.lower().replace("minecraft:", "")


def is_hostile_name(name: str) -> bool:
    normalized = normalize_name(name)
    return any(token in normalized for token in HOSTILE_MOBS)


def is_passive_name(name: str) -> bool:
    normalized = normalize_name(name)
    return any(token in normalized for token in PASSIVE_ANIMALS)


def wrap_degrees(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def angle_delta(current: float, target: float) -> float:
    return abs(wrap_degrees(target - current))


def alignment_score(
    player_x: float,
    player_y: float,
    player_z: float,
    player_yaw: float,
    player_pitch: float,
    entity_x: float,
    entity_y: float,
    entity_z: float,
) -> float:
    dx = entity_x - player_x
    dz = entity_z - player_z
    dy = entity_y - player_y
    horizontal = math.hypot(dx, dz)
    if horizontal < 1e-6:
        return 1.0
    target_yaw = math.degrees(math.atan2(-dx, dz))
    target_pitch = -math.degrees(math.atan2(dy, horizontal))
    yaw_error = angle_delta(player_yaw, target_yaw) / 180.0
    pitch_error = min(abs(player_pitch - target_pitch) / 90.0, 1.0)
    return float(max(0.0, 1.0 - 0.75 * yaw_error - 0.25 * pitch_error))


def target_view_angles(
    player_x: float,
    player_y: float,
    player_z: float,
    entity_x: float,
    entity_y: float,
    entity_z: float,
) -> tuple[float, float]:
    dx = entity_x - player_x
    dz = entity_z - player_z
    dy = entity_y - player_y
    horizontal = math.hypot(dx, dz)
    if horizontal < 1e-6:
        return 0.0, 0.0
    target_yaw = math.degrees(math.atan2(-dx, dz))
    target_pitch = -math.degrees(math.atan2(dy, horizontal))
    return target_yaw, target_pitch


def entity_distance(
    player_x: float, player_y: float, player_z: float, entity: EntityInfo
) -> float:
    return math.dist((player_x, player_y, player_z), (entity.x, entity.y, entity.z))


def inventory_arrow_count(observation: ObservationSpaceMessage) -> int:
    count = 0
    for item in observation.inventory:
        if "arrow" in normalize_name(item.translation_key):
            count += int(item.count)
    return count


def inventory_has_bow(observation: ObservationSpaceMessage) -> bool:
    return any("bow" in normalize_name(item.translation_key) for item in observation.inventory)


def unique_entity_id(entity: EntityInfo, prefix: str) -> str:
    if entity.unique_name:
        return entity.unique_name
    rounded = f"{round(entity.x, 1)}:{round(entity.y, 1)}:{round(entity.z, 1)}"
    return f"{prefix}:{normalize_name(entity.translation_key)}:{rounded}"


def merge_entities(observation: ObservationSpaceMessage) -> list[tuple[EntityInfo, bool]]:
    merged: list[tuple[EntityInfo, bool]] = []
    seen: set[str] = set()
    for index, entity in enumerate(observation.visible_entities):
        uid = unique_entity_id(entity, f"visible-{index}")
        seen.add(uid)
        merged.append((entity, True))
    for bucket in observation.surrounding_entities.values():
        for index, entity in enumerate(bucket.entities):
            uid = unique_entity_id(entity, f"surrounding-{index}")
            if uid in seen:
                continue
            seen.add(uid)
            merged.append((entity, False))
    return merged


def detect_health_drop(
    previous_entities: Iterable[TrackedEntity], current_entities: Iterable[TrackedEntity]
) -> int:
    current_by_id = {entity.unique_id: entity for entity in current_entities}
    hits = 0
    for previous in previous_entities:
        current = current_by_id.get(previous.unique_id)
        if current is None:
            continue
        if previous.health - current.health > 0.1:
            hits += 1
    return hits


def detect_recent_kills(
    previous_entities: Iterable[TrackedEntity], current_entities: Iterable[TrackedEntity]
) -> int:
    current_ids = {entity.unique_id for entity in current_entities}
    kills = 0
    for previous in previous_entities:
        if previous.unique_id not in current_ids and previous.health <= 4.0:
            kills += 1
    return kills


def detect_damage_dealt(
    previous_entities: Iterable[TrackedEntity], current_entities: Iterable[TrackedEntity]
) -> float:
    current_by_id = {entity.unique_id: entity for entity in current_entities}
    damage_dealt = 0.0
    for previous in previous_entities:
        current = current_by_id.get(previous.unique_id)
        if current is None:
            continue
        damage_dealt += max(0.0, previous.health - current.health)
    return damage_dealt


def clearance_from_lidar(observation: ObservationSpaceMessage, target_angle: float) -> float:
    rays = getattr(observation.lidar_result, "rays", [])
    if not rays:
        return 1.0
    distances: list[float] = []
    for ray in rays:
        angle = float(ray.angle_horizontal) % 360.0
        delta = abs(((angle - target_angle + 180.0) % 360.0) - 180.0)
        if delta <= 25.0:
            distances.append(float(ray.distance))
    if not distances:
        return 1.0
    return min(max(min(distances) / 6.0, 0.0), 1.0)


def summarize_observation(
    observation: ObservationSpaceMessage,
    stage_config: StageConfig,
    center_x: float,
    center_z: float,
    reward_config: RewardConfig,
    previous_summary: Optional[ObservationSummary] = None,
) -> ObservationSummary:
    summary = ObservationSummary(
        health=float(observation.health),
        food_level=float(observation.food_level),
        saturation_level=float(observation.saturation_level),
        is_dead=bool(observation.is_dead),
        yaw=float(observation.yaw),
        pitch=float(observation.pitch),
        velocity_x=float(observation.velocity_x),
        velocity_y=float(observation.velocity_y),
        velocity_z=float(observation.velocity_z),
        x=float(observation.x),
        y=float(observation.y),
        z=float(observation.z),
        arrow_count=inventory_arrow_count(observation),
        inventory_has_bow=inventory_has_bow(observation),
        raw=observation,
    )
    summary.speed = math.sqrt(
        summary.velocity_x**2 + summary.velocity_y**2 + summary.velocity_z**2
    )
    if previous_summary is not None:
        summary.moved_distance = math.dist(
            (summary.x, summary.y, summary.z),
            (previous_summary.x, previous_summary.y, previous_summary.z),
        )
    summary.front_clearance = clearance_from_lidar(observation, 0.0)
    summary.right_clearance = clearance_from_lidar(observation, 90.0)
    summary.left_clearance = clearance_from_lidar(observation, 270.0)
    summary.blocked_forward = summary.front_clearance < reward_config.blocked_forward_threshold / 6.0
    summary.blocked_left = summary.left_clearance < reward_config.blocked_side_threshold / 6.0
    summary.blocked_right = summary.right_clearance < reward_config.blocked_side_threshold / 6.0

    max_axis_offset = max(abs(summary.x - center_x), abs(summary.z - center_z))
    distance_to_wall = max(0.0, stage_config.arena_radius - max_axis_offset)
    summary.boundary_pressure = float(
        max(0.0, 1.0 - distance_to_wall / reward_config.boundary_margin)
    )

    best_hostile_alignment = 0.0
    best_animal_alignment = 0.0
    for index, (entity, visible) in enumerate(merge_entities(observation)):
        tracked = TrackedEntity(
            unique_id=unique_entity_id(entity, str(index)),
            translation_key=entity.translation_key,
            distance=entity_distance(summary.x, summary.y, summary.z, entity),
            health=float(entity.health),
            visible=visible,
            x=float(entity.x),
            y=float(entity.y),
            z=float(entity.z),
        )
        if is_hostile_name(entity.translation_key):
            summary.hostile_entities.append(tracked)
            if visible:
                summary.hostile_visible = True
                summary.visible_hostile_ids.add(tracked.unique_id)
                best_hostile_alignment = max(
                    best_hostile_alignment,
                    alignment_score(
                        summary.x,
                        summary.y + 1.62,
                        summary.z,
                        summary.yaw,
                        summary.pitch,
                        tracked.x,
                        tracked.y + 0.9,
                        tracked.z,
                    ),
                )
        elif is_passive_name(entity.translation_key):
            summary.animal_entities.append(tracked)
            if visible:
                best_animal_alignment = max(
                    best_animal_alignment,
                    alignment_score(
                        summary.x,
                        summary.y + 1.62,
                        summary.z,
                        summary.yaw,
                        summary.pitch,
                        tracked.x,
                        tracked.y + 0.9,
                        tracked.z,
                    ),
                )

    if summary.hostile_entities:
        summary.nearest_hostile_distance = min(
            entity.distance for entity in summary.hostile_entities
        )

    target_entities = summary.hostile_entities
    target_alignment = best_hostile_alignment
    if not target_entities:
        target_entities = summary.animal_entities
        target_alignment = best_animal_alignment
    if target_entities:
        summary.target_visible = any(entity.visible for entity in target_entities)
        summary.visible_target_ids = {
            entity.unique_id for entity in target_entities if entity.visible
        }
        summary.nearest_target_distance = min(entity.distance for entity in target_entities)
        summary.nearest_target_alignment = target_alignment
        summary.target_centered = target_alignment >= 0.82

    if observation.raycast_result.type == HitResult.ENTITY:
        target_name = observation.raycast_result.target_entity.translation_key
        summary.raycast_hostile = is_hostile_name(target_name)
        summary.raycast_target = summary.raycast_hostile or is_passive_name(target_name)

    return summary


def build_vector_observation(
    summary: ObservationSummary,
    stage_config: StageConfig,
    steps_since_shot: int,
    idle_steps: int,
) -> np.ndarray:
    radius = max(float(stage_config.arena_radius), 1.0)
    return np.array(
        [
            summary.health / 20.0,
            summary.food_level / 20.0,
            min(summary.saturation_level / 20.0, 1.0),
            wrap_degrees(summary.yaw) / 180.0,
            float(np.clip(summary.pitch / 90.0, -1.0, 1.0)),
            float(np.clip(summary.velocity_x / 2.0, -1.0, 1.0)),
            float(np.clip(summary.velocity_y / 2.0, -1.0, 1.0)),
            float(np.clip(summary.velocity_z / 2.0, -1.0, 1.0)),
            float(min(summary.nearest_hostile_distance / radius, 1.5)),
            float(min(summary.nearest_target_distance / radius, 1.5)),
            float(summary.hostile_visible),
            float(summary.target_visible),
            float(summary.raycast_hostile),
            float(summary.raycast_target),
            float(summary.nearest_target_alignment),
            float(summary.boundary_pressure),
            float(min(steps_since_shot / max(stage_config.no_shot_timeout, 1), 2.0)),
            float(min(idle_steps / max(1, 80), 2.0)),
            float(summary.front_clearance),
            float(summary.left_clearance),
            float(summary.right_clearance),
            float(summary.blocked_forward),
            float(summary.blocked_left),
            float(summary.blocked_right),
        ],
        dtype=np.float32,
    )


class SurviveAndHuntEnvironment(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        stage: str | StageConfig = "stage3_combined_easy",
        *,
        image_width: int = 128,
        image_height: int = 128,
        env_path: Optional[str] = None,
        port: int = 8000,
        seed: int = 0,
        render_action: bool = False,
        use_vglrun: bool = False,
        reward_config: RewardConfig = RewardConfig(),
    ) -> None:
        super().__init__()
        self.stage_config = STAGE_CONFIGS[stage] if isinstance(stage, str) else stage
        self.reward_config = reward_config
        self.image_width = image_width
        self.image_height = image_height
        self.env_path = env_path
        self.port = port
        self.seed_value = seed
        self.random = random.Random(seed)
        self.center_x = 0
        self.center_z = 0
        self.player_y = -59
        self.render_action = render_action
        self.use_vglrun = use_vglrun

        self.action_space = spaces.Discrete(len(BowMacroAction))
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_height, self.image_width, 3),
                    dtype=np.uint8,
                ),
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(VECTOR_OBSERVATION_SIZE,),
                    dtype=np.float32,
                ),
            }
        )

        self.base_env = make(
            initial_env_config=self._build_initial_config(),
            env_path=self.env_path,
            port=self.port,
            use_vglrun=self.use_vglrun,
            render_action=self.render_action,
            cleanup_world=True,
        )

        self.episode_steps = 0
        self.steps_since_shot = 0
        self.steps_since_hit = 0
        self.shots_fired = 0
        self.target_hits = 0
        self.target_kills = 0
        self.target_damage_dealt = 0.0
        self.damage_taken = 0.0
        self.bow_charge_ticks = 0
        self.recent_shot_window = 0
        self.idle_steps = 0
        self.stationary_combat_steps = 0
        self.stuck_steps = 0
        self.last_termination_reason = ""
        self.previous_summary: Optional[ObservationSummary] = None
        self.current_image: Optional[np.ndarray] = None
        self.seen_target_ids: set[str] = set()
        self.prefer_strafe_left = True
        self.position_history: deque[tuple[float, float]] = deque(
            maxlen=self.reward_config.stuck_history_length
        )
        self.backtrack_steps_remaining = 0

    def _build_initial_config(self) -> InitialEnvironmentConfig:
        return InitialEnvironmentConfig(
            image_width=self.image_width,
            image_height=self.image_height,
            gamemode=GameMode.SURVIVAL,
            difficulty=Difficulty.NORMAL,
            world_type=WorldType.SUPERFLAT,
            generate_structures=False,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            hud_hidden=True,
            render_distance=5,
            simulation_distance=7,
            request_raycast=True,
            lidar_config=LidarConfig(
                horizontal_rays=32,
                max_distance=10.0,
                vertical_angle=0.0,
                vertical_rays=3,
                vertical_fov=50.0,
            ),
            surrounding_entity_distances=list(DEFAULT_SURROUNDING_RADII),
            initial_extra_commands=self._arena_commands() + self._player_setup_commands(),
            no_fov_effect=True,
        )

    def _arena_commands(self) -> list[str]:
        radius = self.stage_config.arena_radius
        floor_y = self.player_y - 1
        wall_top = floor_y + self.stage_config.wall_height
        x1 = self.center_x - radius
        x2 = self.center_x + radius
        z1 = self.center_z - radius
        z2 = self.center_z + radius
        commands = [
            "gamerule doMobSpawning false",
            "gamerule keepInventory true",
            "gamerule doDaylightCycle false",
            "gamerule doWeatherCycle false",
            "weather clear",
            "time set day",
            f"fill {x1} {floor_y} {z1} {x2} {floor_y} {z2} minecraft:grass_block replace",
            f"fill {x1} {floor_y + 1} {z1} {x2} {wall_top + 4} {z2} minecraft:air replace",
            f"fill {x1} {floor_y + 1} {z1} {x2} {wall_top} {z1} minecraft:stone_bricks replace",
            f"fill {x1} {floor_y + 1} {z2} {x2} {wall_top} {z2} minecraft:stone_bricks replace",
            f"fill {x1} {floor_y + 1} {z1} {x1} {wall_top} {z2} minecraft:stone_bricks replace",
            f"fill {x2} {floor_y + 1} {z1} {x2} {wall_top} {z2} minecraft:stone_bricks replace",
        ]
        commands.extend(self._obstacle_commands())
        return commands

    def _obstacle_commands(self) -> list[str]:
        y = self.player_y
        commands = [
            f"fill -3 {y - 1} -3 3 {y - 1} 3 minecraft:dirt replace",
            f"fill -2 {y} -2 2 {y} 2 minecraft:grass_block replace",
            f"fill -8 {y - 1} -3 -3 {y + 1} 2 minecraft:stone replace",
            f"fill 3 {y - 1} -2 8 {y + 2} 2 minecraft:stone replace",
            f"fill -14 {y - 1} -12 -9 {y} -6 minecraft:dirt replace",
            f"fill -13 {y} -11 -10 {y + 1} -7 minecraft:dirt replace",
            f"fill -12 {y + 1} -10 -11 {y + 2} -8 minecraft:grass_block replace",
            f"fill 8 {y - 1} 7 13 {y} 12 minecraft:dirt replace",
            f"fill 9 {y} 8 12 {y + 1} 11 minecraft:dirt replace",
            f"fill 10 {y + 1} 9 11 {y + 2} 10 minecraft:grass_block replace",
            f"fill -12 {y - 1} 4 -6 {y + 1} 10 minecraft:stone replace",
            f"fill 5 {y - 1} -12 11 {y + 1} -6 minecraft:stone replace",
            f"fill -10 {y} 6 -4 {y} 6 minecraft:oak_fence replace",
            f"fill 4 {y} -6 10 {y} -6 minecraft:oak_fence replace",
            f"setblock -7 {y} 6 minecraft:air",
            f"setblock 7 {y} -6 minecraft:air",
            f"fill -1 {y} -10 1 {y + 1} -8 minecraft:dirt replace",
            f"fill -1 {y + 1} -9 1 {y + 2} -9 minecraft:dirt replace",
            f"fill -1 {y + 2} -8 1 {y + 3} -8 minecraft:dirt replace",
            f"fill -1 {y + 3} -7 1 {y + 4} -7 minecraft:dirt replace",
            f"fill -1 {y + 4} -6 1 {y + 5} -6 minecraft:grass_block replace",
            f"fill -2 {y - 1} 8 2 {y + 1} 12 minecraft:dirt replace",
            f"fill -1 {y} 9 1 {y + 2} 11 minecraft:grass_block replace",
            f"fill -1 {y + 3} 10 1 {y + 3} 10 minecraft:oak_log replace",
            f"fill -2 {y + 4} 9 2 {y + 5} 11 minecraft:oak_leaves replace",
            f"fill -14 {y - 2} -1 -11 {y - 2} 2 minecraft:water replace",
            f"fill 11 {y - 2} -2 14 {y - 2} 1 minecraft:water replace",
            f"fill -16 {y - 1} 12 -12 {y} 16 minecraft:dirt replace",
            f"fill -15 {y} 13 -13 {y + 1} 15 minecraft:dirt replace",
            f"fill -14 {y + 1} 14 -14 {y + 2} 14 minecraft:grass_block replace",
            f"fill 12 {y - 1} -16 16 {y} -12 minecraft:dirt replace",
            f"fill 13 {y} -15 15 {y + 1} -13 minecraft:dirt replace",
            f"fill 14 {y + 1} -14 14 {y + 2} -14 minecraft:grass_block replace",
        ]
        for tx, tz in [(-10, -7), (9, 8), (-8, 9), (10, -8)]:
            commands.extend(
                [
                    f"fill {tx} {y} {tz} {tx} {y + 3} {tz} minecraft:oak_log replace",
                    f"fill {tx - 1} {y + 3} {tz - 1} {tx + 1} {y + 5} {tz + 1} minecraft:oak_leaves replace",
                    f"fill {tx - 2} {y + 4} {tz - 2} {tx + 2} {y + 4} {tz + 2} minecraft:oak_leaves replace",
                ]
            )
        for sx, sz in [(-6, -11), (6, 11), (-11, 7), (11, -7)]:
            commands.extend(
                [
                    f"fill {sx - 1} {y - 1} {sz - 1} {sx + 1} {y + 1} {sz + 1} minecraft:cobblestone replace",
                    f"fill {sx} {y + 2} {sz} {sx} {y + 4} {sz} minecraft:cobblestone_wall replace",
                ]
            )
        return commands

    def _player_setup_commands(self) -> list[str]:
        return [
            f"tp @p {self.center_x} {self.player_y} {self.center_z} 0 0",
            "effect give @p minecraft:saturation 1000000 10 true",
            "clear @p",
            "item replace entity @p hotbar.0 with minecraft:bow",
            "item replace entity @p weapon.mainhand with minecraft:bow",
            "give @p minecraft:arrow 64",
        ]

    def _reset_commands(self) -> list[str]:
        commands = ["kill @e[type=!minecraft:player]"]
        commands.extend(self._arena_commands())
        commands.extend(self._player_setup_commands())
        commands.extend(self._spawn_commands(self.stage_config.hostile_count))
        return commands

    def _spawn_commands(self, hostile_count: int) -> list[str]:
        commands: list[str] = []
        radius = max(self.stage_config.arena_radius - 2, 4)
        hostile_positions: list[tuple[int, int]] = []
        ring = max(radius - 1, 4)
        for dx in (-ring, -ring // 2, 0, ring // 2, ring):
            for dz in (-ring, -ring // 2, 0, ring // 2, ring):
                if abs(dx) + abs(dz) < ring:
                    continue
                hostile_positions.append((dx, dz))
        hostile_positions = list(dict.fromkeys(hostile_positions))
        self.random.shuffle(hostile_positions)
        for index in range(hostile_count):
            mob_name = self.stage_config.hostile_types[index % len(self.stage_config.hostile_types)]
            dx, dz = hostile_positions[index % len(hostile_positions)]
            commands.append(
                f"summon minecraft:{mob_name} {self.center_x + dx} {self.player_y} {self.center_z + dz} {{PersistenceRequired:1b}}"
            )
        return commands

    def _micro_action(
        self,
        *,
        forward: bool = False,
        back: bool = False,
        strafe_left: bool = False,
        strafe_right: bool = False,
        jump: bool = False,
        sprint: bool = False,
        use: bool = False,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0,
    ) -> list[int]:
        action = no_op()
        if forward:
            action[0] = 1
        elif back:
            action[0] = 2
        if strafe_right:
            action[1] = 1
        elif strafe_left:
            action[1] = 2
        if sprint:
            action[2] = 3
        elif jump:
            action[2] = 1
        action[3] = int(np.clip(round(camera_pitch / 15.0) + 12, 0, 24))
        action[4] = int(np.clip(round(camera_yaw / 15.0) + 12, 0, 24))
        if use:
            action[5] = 1
        return action

    def _priority_target(self, summary: ObservationSummary) -> Optional[TrackedEntity]:
        targets = self._current_targets(summary)
        if not targets:
            return None
        visible_targets = [entity for entity in targets if entity.visible]
        candidates = visible_targets or targets
        return min(candidates, key=lambda entity: entity.distance)

    def _aim_deltas(self, summary: ObservationSummary, target: TrackedEntity) -> tuple[float, float]:
        target_yaw, target_pitch = target_view_angles(
            summary.x,
            summary.y,
            summary.z,
            target.x,
            target.y,
            target.z,
        )
        yaw_delta = wrap_degrees(target_yaw - summary.yaw)
        pitch_delta = float(np.clip(target_pitch - summary.pitch, -45.0, 45.0))
        return yaw_delta, pitch_delta

    def _combat_mobility_flags(
        self, summary: ObservationSummary, target: TrackedEntity
    ) -> tuple[bool, bool, bool, bool]:
        forward = False
        back = False
        strafe_left = False
        strafe_right = False
        under_pressure = target.distance < self.reward_config.danger_radius + 1.0
        if under_pressure:
            use_left = self.prefer_strafe_left and not summary.blocked_left
            use_right = (not self.prefer_strafe_left) and not summary.blocked_right
            if use_left:
                strafe_left = True
            elif use_right:
                strafe_right = True
            elif not summary.blocked_left:
                strafe_left = True
            elif not summary.blocked_right:
                strafe_right = True
            else:
                back = True
            if target.distance < self.reward_config.too_close_distance:
                back = True
        elif target.distance > self.reward_config.comfortable_combat_distance + 1.5:
            if not summary.blocked_forward:
                forward = True
            elif not summary.blocked_left:
                strafe_left = True
            elif not summary.blocked_right:
                strafe_right = True
        return forward, back, strafe_left, strafe_right

    def _tracking_sequence(
        self, summary: ObservationSummary, hold_ticks: int, *, release: bool
    ) -> list[list[int]]:
        target = self._priority_target(summary)
        if target is None:
            sequence = [self._micro_action(use=True) for _ in range(hold_ticks)]
            if release:
                sequence.append(self._micro_action())
            return sequence

        yaw_delta, pitch_delta = self._aim_deltas(summary, target)
        tracking_ticks = max(1, min(4, hold_ticks))
        yaw_step = float(np.clip(round((yaw_delta / tracking_ticks) / 15.0) * 15.0, -45.0, 45.0))
        pitch_step = float(
            np.clip(round((pitch_delta / tracking_ticks) / 15.0) * 15.0, -15.0, 15.0)
        )
        forward, back, strafe_left, strafe_right = self._combat_mobility_flags(summary, target)
        sequence: list[list[int]] = []
        for tick in range(hold_ticks):
            sequence.append(
                self._micro_action(
                    forward=forward,
                    back=back,
                    strafe_left=strafe_left,
                    strafe_right=strafe_right,
                    use=True,
                    camera_yaw=yaw_step if tick < tracking_ticks else 0.0,
                    camera_pitch=pitch_step if tick < tracking_ticks else 0.0,
                )
            )
        if release:
            sequence.append(
                self._micro_action(
                    forward=forward,
                    back=back,
                    strafe_left=strafe_left,
                    strafe_right=strafe_right,
                )
            )
        return sequence

    def _macro_to_sequence(
        self, action_id: int, summary: ObservationSummary
    ) -> tuple[list[list[int]], bool, int]:
        action = BowMacroAction(action_id)
        if action == BowMacroAction.NO_OP:
            return [self._micro_action()], False, 0
        if action == BowMacroAction.FORWARD:
            return [self._micro_action(forward=True) for _ in range(4)], False, 0
        if action == BowMacroAction.BACKWARD:
            return [self._micro_action(back=True) for _ in range(4)], False, 0
        if action == BowMacroAction.STRAFE_LEFT:
            return [self._micro_action(strafe_left=True) for _ in range(4)], False, 0
        if action == BowMacroAction.STRAFE_RIGHT:
            return [self._micro_action(strafe_right=True) for _ in range(4)], False, 0
        if action == BowMacroAction.SPRINT_FORWARD:
            return [self._micro_action(forward=True, sprint=True) for _ in range(4)], False, 0
        if action == BowMacroAction.JUMP_FORWARD:
            return [self._micro_action(forward=True, jump=True) for _ in range(2)] + [
                self._micro_action(forward=True) for _ in range(2)
            ], False, 0
        if action == BowMacroAction.CLIMB_FORWARD:
            return [self._micro_action(forward=True, jump=True, sprint=True) for _ in range(3)] + [
                self._micro_action(forward=True, sprint=True) for _ in range(2)
            ], False, 0
        if action == BowMacroAction.CLIMB_LEFT:
            return [self._micro_action(strafe_left=True, jump=True, sprint=True) for _ in range(2)] + [
                self._micro_action(strafe_left=True, sprint=True) for _ in range(2)
            ], False, 0
        if action == BowMacroAction.CLIMB_RIGHT:
            return [self._micro_action(strafe_right=True, jump=True, sprint=True) for _ in range(2)] + [
                self._micro_action(strafe_right=True, sprint=True) for _ in range(2)
            ], False, 0
        if action == BowMacroAction.TURN_LEFT_SMALL:
            return [self._micro_action(camera_yaw=-15.0)], False, 0
        if action == BowMacroAction.TURN_RIGHT_SMALL:
            return [self._micro_action(camera_yaw=15.0)], False, 0
        if action == BowMacroAction.TURN_LEFT_LARGE:
            return [self._micro_action(camera_yaw=-45.0)], False, 0
        if action == BowMacroAction.TURN_RIGHT_LARGE:
            return [self._micro_action(camera_yaw=45.0)], False, 0
        if action == BowMacroAction.LOOK_UP_SMALL:
            return [self._micro_action(camera_pitch=-15.0)], False, 0
        if action == BowMacroAction.LOOK_DOWN_SMALL:
            return [self._micro_action(camera_pitch=15.0)], False, 0
        if action == BowMacroAction.DRAW_BOW:
            return self._tracking_sequence(summary, 8, release=False), False, 8
        if action == BowMacroAction.RELEASE_BOW:
            return self._tracking_sequence(summary, 0, release=True), self.bow_charge_ticks > 0, 0
        if action == BowMacroAction.SHOOT_ARROW:
            return self._tracking_sequence(summary, 12, release=True), True, 0
        raise ValueError(f"Unsupported macro action: {action}")

    def _resolve_blocked_action(self, action_id: int, summary: ObservationSummary) -> int:
        action = BowMacroAction(action_id)
        if self.backtrack_steps_remaining > 0 or self._is_stuck_from_history(summary):
            if self.backtrack_steps_remaining > 0:
                self.backtrack_steps_remaining -= 1
            else:
                self.backtrack_steps_remaining = self.reward_config.backtrack_commit_steps - 1
            return self._backtrack_action(summary)
        if (
            action == BowMacroAction.NO_OP
            and summary.hostile_entities
            and summary.nearest_hostile_distance < self.reward_config.danger_radius
        ):
            if self.prefer_strafe_left and not summary.blocked_left:
                return int(BowMacroAction.STRAFE_LEFT)
            if not self.prefer_strafe_left and not summary.blocked_right:
                return int(BowMacroAction.STRAFE_RIGHT)
            return int(BowMacroAction.BACKWARD)
        if action in (
            BowMacroAction.FORWARD,
            BowMacroAction.SPRINT_FORWARD,
            BowMacroAction.JUMP_FORWARD,
            BowMacroAction.CLIMB_FORWARD,
        ) and (
            summary.blocked_forward
            or (
                summary.hostile_entities
                and summary.nearest_hostile_distance < self.reward_config.too_close_distance
            )
        ):
            if summary.front_clearance > 0.35:
                return int(BowMacroAction.CLIMB_FORWARD)
            if summary.left_clearance > summary.right_clearance and not summary.blocked_left:
                return int(BowMacroAction.CLIMB_LEFT if summary.left_clearance > 0.35 else BowMacroAction.STRAFE_LEFT)
            if not summary.blocked_right:
                return int(BowMacroAction.CLIMB_RIGHT if summary.right_clearance > 0.35 else BowMacroAction.STRAFE_RIGHT)
            return int(BowMacroAction.BACKWARD)
        if action == BowMacroAction.STRAFE_LEFT and summary.blocked_left:
            self.prefer_strafe_left = False
            if summary.front_clearance > 0.35:
                return int(BowMacroAction.CLIMB_FORWARD)
            return int(BowMacroAction.TURN_RIGHT_SMALL)
        if action == BowMacroAction.STRAFE_RIGHT and summary.blocked_right:
            self.prefer_strafe_left = True
            if summary.front_clearance > 0.35:
                return int(BowMacroAction.CLIMB_FORWARD)
            return int(BowMacroAction.TURN_LEFT_SMALL)
        return self._resolve_aim_action(action_id, summary)

    def _ensure_supplies(self, summary: Optional[ObservationSummary]) -> None:
        if summary is None:
            return
        commands: list[str] = []
        if summary.arrow_count < 8:
            commands.append("give @p minecraft:arrow 16")
        if not summary.inventory_has_bow:
            commands.append("item replace entity @p weapon.mainhand with minecraft:bow")
        if commands:
            self.base_env.add_commands(commands)

    def _maintain_spawn_pressure(self, summary: Optional[ObservationSummary]) -> None:
        if summary is None:
            return
        if self.episode_steps % max(self.stage_config.respawn_interval, 1) != 0:
            return
        hostile_gap = max(0, self.stage_config.hostile_count - len(summary.hostile_entities))
        if hostile_gap > 0:
            self.base_env.add_commands(self._spawn_commands(hostile_gap))

    def _current_targets(self, summary: ObservationSummary) -> list[TrackedEntity]:
        return summary.hostile_entities if summary.hostile_entities else summary.animal_entities

    def _is_stationary_combat_action(self, action: BowMacroAction) -> bool:
        return action in (
            BowMacroAction.NO_OP,
            BowMacroAction.DRAW_BOW,
            BowMacroAction.RELEASE_BOW,
            BowMacroAction.SHOOT_ARROW,
            BowMacroAction.TURN_LEFT_SMALL,
            BowMacroAction.TURN_RIGHT_SMALL,
            BowMacroAction.TURN_LEFT_LARGE,
            BowMacroAction.TURN_RIGHT_LARGE,
            BowMacroAction.LOOK_UP_SMALL,
            BowMacroAction.LOOK_DOWN_SMALL,
        )

    def _update_position_history(self, summary: ObservationSummary) -> None:
        self.position_history.append((summary.x, summary.z))

    def _is_stuck_from_history(self, summary: ObservationSummary) -> bool:
        if len(self.position_history) < self.position_history.maxlen:
            return False
        xs = [position[0] for position in self.position_history]
        zs = [position[1] for position in self.position_history]
        horizontal_range = max(max(xs) - min(xs), max(zs) - min(zs))
        return (
            horizontal_range < self.reward_config.stuck_horizontal_tolerance
            and summary.moved_distance < 0.12
        )

    def _movement_action_from_world_vector(
        self, summary: ObservationSummary, delta_x: float, delta_z: float
    ) -> int:
        if math.hypot(delta_x, delta_z) < 1e-6:
            return int(BowMacroAction.BACKWARD)
        yaw_radians = math.radians(summary.yaw)
        forward_x = -math.sin(yaw_radians)
        forward_z = math.cos(yaw_radians)
        right_x = math.cos(yaw_radians)
        right_z = math.sin(yaw_radians)
        forward_projection = delta_x * forward_x + delta_z * forward_z
        right_projection = delta_x * right_x + delta_z * right_z
        if abs(forward_projection) >= abs(right_projection):
            if forward_projection >= 0:
                return int(BowMacroAction.FORWARD)
            return int(BowMacroAction.BACKWARD)
        if right_projection >= 0:
            return int(BowMacroAction.STRAFE_RIGHT)
        return int(BowMacroAction.STRAFE_LEFT)

    def _backtrack_action(self, summary: ObservationSummary) -> int:
        if len(self.position_history) >= 2:
            oldest_x, oldest_z = self.position_history[0]
            newest_x, newest_z = self.position_history[-1]
            delta_x = oldest_x - newest_x
            delta_z = oldest_z - newest_z
            if math.hypot(delta_x, delta_z) >= 0.1:
                return self._movement_action_from_world_vector(summary, delta_x, delta_z)
        if abs(summary.velocity_x) + abs(summary.velocity_z) > 1e-3:
            return self._movement_action_from_world_vector(
                summary, -summary.velocity_x, -summary.velocity_z
            )
        return int(BowMacroAction.BACKWARD)

    def _aim_correction_action(self, summary: ObservationSummary, target: TrackedEntity) -> int:
        yaw_delta, pitch_delta = self._aim_deltas(summary, target)
        if abs(yaw_delta) >= self.reward_config.aim_yaw_large_threshold:
            return int(
                BowMacroAction.TURN_RIGHT_LARGE if yaw_delta > 0 else BowMacroAction.TURN_LEFT_LARGE
            )
        if abs(yaw_delta) >= self.reward_config.aim_yaw_small_threshold:
            return int(
                BowMacroAction.TURN_RIGHT_SMALL if yaw_delta > 0 else BowMacroAction.TURN_LEFT_SMALL
            )
        if abs(pitch_delta) >= self.reward_config.aim_pitch_threshold:
            return int(
                BowMacroAction.LOOK_DOWN_SMALL if pitch_delta > 0 else BowMacroAction.LOOK_UP_SMALL
            )
        return int(BowMacroAction.NO_OP)

    def _resolve_aim_action(self, action_id: int, summary: ObservationSummary) -> int:
        action = BowMacroAction(action_id)
        target = self._priority_target(summary)
        if target is None:
            return action_id
        if not target.visible:
            return self._resolve_hunt_action(action_id, summary, target)
        alignment = summary.nearest_target_alignment
        release_ready = summary.raycast_target or alignment >= self.reward_config.release_alignment_threshold
        draw_ready = summary.raycast_target or alignment >= self.reward_config.draw_alignment_threshold
        if action == BowMacroAction.SHOOT_ARROW and not release_ready:
            return self._aim_correction_action(summary, target)
        if action == BowMacroAction.RELEASE_BOW and not release_ready:
            if self.bow_charge_ticks > 0 and draw_ready:
                return int(BowMacroAction.DRAW_BOW)
            return self._aim_correction_action(summary, target)
        if action == BowMacroAction.DRAW_BOW and not draw_ready:
            return self._aim_correction_action(summary, target)
        if (
            action == BowMacroAction.NO_OP
            and summary.hostile_entities
            and target.distance <= self.reward_config.comfortable_combat_distance + 2.0
            and not draw_ready
        ):
            return self._aim_correction_action(summary, target)
        if action == BowMacroAction.NO_OP:
            return self._resolve_hunt_action(action_id, summary, target)
        return action_id

    def _resolve_hunt_action(
        self, action_id: int, summary: ObservationSummary, target: TrackedEntity
    ) -> int:
        action = BowMacroAction(action_id)
        yaw_delta, _ = self._aim_deltas(summary, target)
        if abs(yaw_delta) >= self.reward_config.aim_yaw_small_threshold:
            return self._aim_correction_action(summary, target)
        if target.distance > self.reward_config.comfortable_combat_distance + 1.5:
            if not summary.blocked_forward:
                return int(BowMacroAction.SPRINT_FORWARD)
            if summary.front_clearance > 0.35:
                return int(BowMacroAction.CLIMB_FORWARD)
            if summary.left_clearance >= summary.right_clearance and not summary.blocked_left:
                return int(BowMacroAction.CLIMB_LEFT if summary.left_clearance > 0.35 else BowMacroAction.STRAFE_LEFT)
            if not summary.blocked_right:
                return int(BowMacroAction.CLIMB_RIGHT if summary.right_clearance > 0.35 else BowMacroAction.STRAFE_RIGHT)
        if summary.hostile_entities and target.distance < self.reward_config.too_close_distance:
            if self.prefer_strafe_left and not summary.blocked_left:
                return int(BowMacroAction.CLIMB_LEFT if summary.left_clearance > 0.35 else BowMacroAction.STRAFE_LEFT)
            if not self.prefer_strafe_left and not summary.blocked_right:
                return int(BowMacroAction.CLIMB_RIGHT if summary.right_clearance > 0.35 else BowMacroAction.STRAFE_RIGHT)
            return int(BowMacroAction.BACKWARD)
        if (
            action == BowMacroAction.NO_OP
            and target.visible
            and summary.nearest_target_alignment >= self.reward_config.draw_alignment_threshold
        ):
            return int(BowMacroAction.DRAW_BOW)
        return action_id

    def _reward_from_transition(
        self,
        previous: ObservationSummary,
        current: ObservationSummary,
        action_id: int,
        shot_fired: bool,
        macro_ticks: int,
    ) -> tuple[float, int, int]:
        hostile_pressure = bool(current.hostile_entities)
        reward = self.reward_config.survival_per_tick * macro_ticks
        if hostile_pressure:
            reward += self.reward_config.combat_survival_per_tick * macro_ticks
        distance_delta = current.nearest_hostile_distance - previous.nearest_hostile_distance

        damage = max(0.0, previous.health - current.health)
        reward += self.reward_config.damage_scale * damage
        self.damage_taken += damage
        if current.is_dead:
            reward += self.reward_config.death_penalty

        if (
            previous.nearest_hostile_distance < self.reward_config.danger_radius
            and current.nearest_hostile_distance > previous.nearest_hostile_distance
        ):
            reward += self.reward_config.retreat_reward
        if hostile_pressure and current.nearest_hostile_distance < self.reward_config.danger_radius:
            reward += self.reward_config.danger_penalty * macro_ticks
        if (
            hostile_pressure
            and current.nearest_hostile_distance < self.reward_config.comfortable_combat_distance
            and distance_delta < -0.15
        ):
            reward += self.reward_config.approach_under_pressure_penalty * macro_ticks
        if hostile_pressure and current.nearest_hostile_distance < self.reward_config.too_close_distance:
            reward += self.reward_config.too_close_pressure_penalty * macro_ticks
        if (
            hostile_pressure
            and previous.nearest_hostile_distance < self.reward_config.too_close_distance
            and current.nearest_hostile_distance >= self.reward_config.too_close_distance
            and current.moved_distance >= 0.15
        ):
            reward += self.reward_config.escape_band_reward
        if (
            hostile_pressure
            and distance_delta > 0.12
            and BowMacroAction(action_id)
            in (
                BowMacroAction.BACKWARD,
                BowMacroAction.STRAFE_LEFT,
                BowMacroAction.STRAFE_RIGHT,
            )
        ):
            reward += self.reward_config.evasive_move_reward
        if (
            hostile_pressure
            and previous.nearest_target_distance > self.reward_config.comfortable_combat_distance + 1.5
            and current.nearest_target_distance < previous.nearest_target_distance - 0.15
            and current.moved_distance >= 0.12
        ):
            reward += self.reward_config.pursuit_reward_scale

        new_target_ids = current.visible_target_ids - self.seen_target_ids
        if new_target_ids:
            reward += self.reward_config.target_visible_reward
        new_hostile_ids = current.visible_hostile_ids - self.seen_target_ids
        if new_hostile_ids:
            reward += self.reward_config.hostile_visible_reward
        self.seen_target_ids.update(current.visible_target_ids)

        alignment_gain = current.nearest_target_alignment - previous.nearest_target_alignment
        if alignment_gain > 0:
            reward += alignment_gain * self.reward_config.aim_improvement_scale
        if current.target_centered:
            reward += self.reward_config.target_centered_reward
        if BowMacroAction(action_id) == BowMacroAction.DRAW_BOW and current.target_centered:
            reward += self.reward_config.draw_centered_reward
        shot_alignment = max(previous.nearest_target_alignment, current.nearest_target_alignment)
        if shot_fired:
            if previous.raycast_target or current.raycast_target:
                reward += self.reward_config.raycast_target_reward
                if hostile_pressure and current.moved_distance >= 0.12:
                    reward += self.reward_config.kiting_shot_reward
            elif shot_alignment < 0.72:
                reward += self.reward_config.blind_shot_penalty

        hits = 0
        kills = 0
        damage_dealt = 0.0
        if shot_fired or self.recent_shot_window > 0:
            hits = detect_health_drop(self._current_targets(previous), self._current_targets(current))
            kills = detect_recent_kills(self._current_targets(previous), self._current_targets(current))
            damage_dealt = detect_damage_dealt(
                self._current_targets(previous), self._current_targets(current)
            )
        distance_reward_scale = float(
            np.clip(
                current.nearest_target_distance
                / max(self.reward_config.comfortable_combat_distance, 1e-6),
                self.reward_config.min_distance_reward_scale,
                self.reward_config.max_distance_reward_scale,
            )
        )
        if current.nearest_target_distance < self.reward_config.too_close_distance:
            distance_reward_scale *= 0.5
        if hits:
            reward += hits * self.reward_config.target_hit_reward * distance_reward_scale
        if kills:
            reward += kills * self.reward_config.target_kill_reward * max(distance_reward_scale, 0.6)
        if damage_dealt > 0:
            reward += damage_dealt * self.reward_config.damage_dealt_scale
            self.target_damage_dealt += damage_dealt

        if shot_fired:
            reward += self.reward_config.shot_penalty
        if (
            BowMacroAction(action_id)
            in (BowMacroAction.FORWARD, BowMacroAction.SPRINT_FORWARD, BowMacroAction.JUMP_FORWARD)
            and previous.blocked_forward
        ):
            reward += self.reward_config.blocked_move_penalty
        if (
            BowMacroAction(action_id)
            in (
                BowMacroAction.TURN_LEFT_SMALL,
                BowMacroAction.TURN_RIGHT_SMALL,
                BowMacroAction.TURN_LEFT_LARGE,
                BowMacroAction.TURN_RIGHT_LARGE,
                BowMacroAction.LOOK_UP_SMALL,
                BowMacroAction.LOOK_DOWN_SMALL,
            )
            and not shot_fired
            and current.moved_distance < 0.05
            and not current.raycast_target
            and alignment_gain <= 0
        ):
            reward += self.reward_config.scan_penalty * macro_ticks

        if hostile_pressure:
            reward += min(current.moved_distance, 1.0) * self.reward_config.movement_reward_scale
            spacing_error = abs(
                current.nearest_hostile_distance - self.reward_config.comfortable_combat_distance
            )
            spacing_score = max(0.0, 1.0 - spacing_error / self.reward_config.comfortable_combat_distance)
            reward += (
                spacing_score
                * min(current.moved_distance, 1.0)
                * self.reward_config.spacing_reward_scale
            )
            if (
                current.nearest_hostile_distance >= self.reward_config.too_close_distance
                and current.nearest_hostile_distance <= self.reward_config.comfortable_combat_distance + 1.5
                and current.moved_distance >= 0.12
                and (
                    current.raycast_target
                    or current.nearest_target_alignment
                    >= self.reward_config.engagement_alignment_threshold
                )
            ):
                reward += self.reward_config.engaged_kiting_reward
            if (
                current.speed < self.reward_config.idle_speed_threshold
                and current.moved_distance < 0.05
            ):
                reward += self.reward_config.idle_step_penalty * macro_ticks
            if (
                self._is_stationary_combat_action(BowMacroAction(action_id))
                and current.moved_distance < 0.10
            ):
                reward += self.reward_config.stationary_bow_penalty * macro_ticks
                if shot_fired:
                    reward += self.reward_config.stationary_shot_penalty
            if (
                current.nearest_hostile_distance < self.reward_config.too_close_distance
                and current.moved_distance < 0.08
                and BowMacroAction(action_id)
                in (
                    BowMacroAction.NO_OP,
                    BowMacroAction.DRAW_BOW,
                    BowMacroAction.RELEASE_BOW,
                    BowMacroAction.SHOOT_ARROW,
                )
            ):
                reward += self.reward_config.stationary_close_combat_penalty * macro_ticks
        if self.idle_steps >= self.reward_config.idle_timeout_steps:
            reward += self.reward_config.idle_timeout_penalty
        if self.stationary_combat_steps >= self.reward_config.stationary_combat_timeout_steps:
            reward += self.reward_config.stationary_combat_timeout_penalty
        if self.stuck_steps > 0:
            reward += self.reward_config.stuck_penalty * macro_ticks
            if current.moved_distance >= 0.2:
                reward += self.reward_config.backtrack_escape_reward

        if self.steps_since_shot >= self.stage_config.no_shot_timeout:
            reward += self.reward_config.no_shot_penalty
        if hostile_pressure and self.steps_since_hit >= self.stage_config.no_hit_timeout:
            reward += self.reward_config.no_hit_penalty

        reward += current.boundary_pressure * self.reward_config.boundary_penalty * macro_ticks
        return reward, hits, kills

    def _termination_flags(self, summary: ObservationSummary) -> tuple[bool, bool, str]:
        if summary.is_dead:
            return True, False, "death"
        if self.episode_steps >= self.stage_config.max_steps:
            return False, True, "max_steps"
        if self.idle_steps >= self.reward_config.idle_timeout_steps:
            return False, True, "idle_timeout"
        if self.stationary_combat_steps >= self.reward_config.stationary_combat_timeout_steps:
            return False, True, "stationary_combat_timeout"
        if self.steps_since_shot >= self.stage_config.no_shot_timeout:
            return False, True, "no_shot_timeout"
        if summary.hostile_entities and self.steps_since_hit >= self.stage_config.no_hit_timeout:
            return False, True, "no_hit_timeout"
        return False, False, ""

    def _episode_metrics(self, summary: ObservationSummary) -> dict[str, Any]:
        return {
            "stage": self.stage_config.name,
            "survival_steps": self.episode_steps,
            "shots_fired": self.shots_fired,
            "target_hits": self.target_hits,
            "target_kills": self.target_kills,
            "target_damage_dealt": round(self.target_damage_dealt, 3),
            "animal_hits": self.target_hits,
            "animal_kills": self.target_kills,
            "damage_taken": round(self.damage_taken, 3),
            "nearest_hostile_distance": round(summary.nearest_hostile_distance, 3),
            "nearest_target_distance": round(summary.nearest_target_distance, 3),
            "boundary_pressure": round(summary.boundary_pressure, 3),
            "idle_steps": self.idle_steps,
            "stationary_combat_steps": self.stationary_combat_steps,
            "stuck_steps": self.stuck_steps,
            "termination_reason": self.last_termination_reason,
        }

    def _convert_observation(
        self, image: np.ndarray, summary: ObservationSummary
    ) -> dict[str, np.ndarray]:
        return {
            "image": image.astype(np.uint8, copy=False),
            "vector": build_vector_observation(
                summary,
                self.stage_config,
                self.steps_since_shot,
                self.idle_steps,
            ),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.seed_value = seed
            self.random.seed(seed)
        options = options or {}
        reset_options = {"fast_reset": True, "extra_commands": self._reset_commands()}
        reset_options.update(options)
        raw_obs, _ = self.base_env.reset(seed=seed, options=reset_options)
        summary = summarize_observation(
            raw_obs["full"],
            self.stage_config,
            self.center_x,
            self.center_z,
            self.reward_config,
        )
        self.previous_summary = summary
        self.current_image = raw_obs["pov"]
        self.episode_steps = 0
        self.steps_since_shot = 0
        self.steps_since_hit = 0
        self.shots_fired = 0
        self.target_hits = 0
        self.target_kills = 0
        self.target_damage_dealt = 0.0
        self.damage_taken = 0.0
        self.bow_charge_ticks = 0
        self.recent_shot_window = 0
        self.idle_steps = 0
        self.stationary_combat_steps = 0
        self.stuck_steps = 0
        self.last_termination_reason = ""
        self.seen_target_ids = set(summary.visible_target_ids)
        self.prefer_strafe_left = bool(self.random.getrandbits(1))
        self.position_history.clear()
        self._update_position_history(summary)
        self.backtrack_steps_remaining = 0
        return self._convert_observation(raw_obs["pov"], summary), {"stage": self.stage_config.name}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.previous_summary is None:
            raise RuntimeError("Environment must be reset before stepping.")

        self._ensure_supplies(self.previous_summary)
        self._maintain_spawn_pressure(self.previous_summary)

        resolved_action = self._resolve_blocked_action(int(action), self.previous_summary)
        sequence, shot_fired, draw_charge = self._macro_to_sequence(
            resolved_action, self.previous_summary
        )
        final_obs: Optional[dict[str, Any]] = None
        for micro_action in sequence:
            final_obs, _, _, _, _ = self.base_env.step(micro_action)
        assert final_obs is not None

        self.episode_steps += len(sequence)
        if draw_charge:
            self.bow_charge_ticks += draw_charge
        elif shot_fired:
            self.bow_charge_ticks = 0

        if shot_fired:
            self.steps_since_shot = 0
            self.shots_fired += 1
            self.recent_shot_window = self.reward_config.shot_recent_window
        else:
            self.steps_since_shot += len(sequence)
            self.recent_shot_window = max(0, self.recent_shot_window - len(sequence))

        summary = summarize_observation(
            final_obs["full"],
            self.stage_config,
            self.center_x,
            self.center_z,
            self.reward_config,
            self.previous_summary,
        )
        hostile_pressure = bool(summary.hostile_entities)
        if (
            hostile_pressure
            and summary.speed < self.reward_config.idle_speed_threshold
            and summary.moved_distance < 0.05
        ):
            self.idle_steps += len(sequence)
        else:
            self.idle_steps = 0
        if (
            hostile_pressure
            and self._is_stationary_combat_action(BowMacroAction(resolved_action))
            and summary.moved_distance < 0.10
        ):
            self.stationary_combat_steps += len(sequence)
        else:
            self.stationary_combat_steps = 0
        if self._is_stuck_from_history(summary):
            self.stuck_steps += len(sequence)
            if self.backtrack_steps_remaining <= 0:
                self.backtrack_steps_remaining = self.reward_config.backtrack_commit_steps
        else:
            self.stuck_steps = 0
            if summary.moved_distance >= 0.12:
                self.backtrack_steps_remaining = 0

        reward, hits, kills = self._reward_from_transition(
            self.previous_summary,
            summary,
            resolved_action,
            shot_fired,
            len(sequence),
        )
        self.target_hits += hits
        self.target_kills += kills
        if hits:
            self.steps_since_hit = 0
        else:
            self.steps_since_hit += len(sequence)

        terminated, truncated, reason = self._termination_flags(summary)
        self.last_termination_reason = reason
        self.previous_summary = summary
        self.current_image = final_obs["pov"]
        self._update_position_history(summary)

        info = {
            "stage": self.stage_config.name,
            "shots_fired": self.shots_fired,
            "target_hits": self.target_hits,
            "target_kills": self.target_kills,
            "target_damage_dealt": self.target_damage_dealt,
            "animal_hits": self.target_hits,
            "animal_kills": self.target_kills,
            "survival_steps": self.episode_steps,
            "nearest_hostile_distance": summary.nearest_hostile_distance,
            "nearest_target_distance": summary.nearest_target_distance,
            "boundary_pressure": summary.boundary_pressure,
            "damage_taken": self.damage_taken,
            "idle_steps": self.idle_steps,
            "stationary_combat_steps": self.stationary_combat_steps,
            "stuck_steps": self.stuck_steps,
            "termination_reason": reason,
        }
        if terminated or truncated:
            info["episode_metrics"] = self._episode_metrics(summary)
        return self._convert_observation(final_obs["pov"], summary), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()
