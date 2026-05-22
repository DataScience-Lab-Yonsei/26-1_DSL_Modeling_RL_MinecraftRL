"""
House Building Environment (V3).

Key changes from V2:
  - No world array — uses placed_blocks/correct_blocks/incorrect_blocks sets
  - Origin-based coordinates (set by first block placement)
  - Dynamic house size (random 5~10 per axis)
  - 45-dim flat observation with sin/cos angles
  - 4-slot inventory (planks, axe, door, glass)
  - No surrounding_blocks dependency
"""
import math
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cuboid_house_rl.config import (
    # House
    HOUSE_WIDTH_MIN, HOUSE_WIDTH_MAX,
    HOUSE_DEPTH_MIN, HOUSE_DEPTH_MAX,
    HOUSE_HEIGHT, FLOOR_Y, WALL_Y_MIN, WALL_Y_MAX, WALL_HEIGHT,
    DOOR_HEIGHT_BOTTOM, DOOR_HEIGHT_TOP,
    # Block types
    AIR, OAK_PLANKS, SOLID, NUM_BLOCK_TYPES,
    # Inventory
    SLOT_PLANKS, SLOT_AXE, SLOT_DOOR, SLOT_GLASS, NUM_INVENTORY_SLOTS,
    # Observation
    FLAT_OBS_SIZE, RAYCAST_MAX_DISTANCE,
    ORIGIN_SET_SIZE, AGENT_STATE_SIZE, RAYCAST_INFO_SIZE,
    PROGRESS_SIZE, INCORRECT_COUNT_SIZE,
    TIME_REMAINING_SIZE, STUCK_RATIO_SIZE,
    TARGET_DIRECTION_SIZE, TARGET_DISTANCE_SIZE,
    TARGET_ABSOLUTE_SIZE, HOUSE_SIZE_SIZE,
    # Minecraft
    MINECRAFT_GROUND_Y,
    # Actions
    ACTION_DIMS, NUM_ACTION_DIMS,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    CAMERA_DELTA_MAP, PLANKS_SLOT,
    # Rewards
    REWARD_CORRECT_PLACEMENT, REWARD_INCORRECT_PLACEMENT,
    REWARD_INCORRECT_BLOCK_REMOVAL, REWARD_CORRECT_BLOCK_REMOVAL,
    REWARD_TIME_PENALTY, REWARD_STAGE_COMPLETE, REWARD_COLUMN_COMPLETE,
    REWARD_STUCK_PENALTY,
    REWARD_PROXIMITY_PER_BLOCK,
    HOUSE_PROXIMITY_THRESHOLD,
    # Episode
    MAX_EPISODE_STEPS, STUCK_PATIENCE, STUCK_MIN_DELTA,
    # Subtasks
    SUBTASK_FLOOR, SUBTASK_WALLS, SUBTASK_DOOR, SUBTASK_CEILING, SUBTASK_DONE,
)
from cuboid_house_rl.utils.completion import StuckDetector


class HouseBuildingEnv(gym.Env):
    """V3: Origin-based, placed_blocks set, dynamic house size."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, craftground_env):
        super().__init__()

        self.craftground_env = craftground_env
        self._cg_obs = None
        self.max_episode_steps = MAX_EPISODE_STEPS

        from cuboid_house_rl.envs.craftground_adapter import CraftGroundObsExtractor
        self._cg_obs_extractor = CraftGroundObsExtractor()

        # Dynamic house params (set per episode)
        self.house_width = 7
        self.house_depth = 7
        self.house_height = HOUSE_HEIGHT

        # Origin (set by first block placement)
        self.origin_x = None
        self.origin_z = None
        self.origin_set = False

        # Block tracking (replaces world array)
        self.placed_blocks = set()      # all placed blocks {(x,y,z)}
        self.correct_blocks = set()     # placed & in blueprint
        self.incorrect_blocks = set()   # placed & not in blueprint
        self.blueprint_positions = set()  # all target positions
        self.floor_positions = set()
        self.wall_positions = set()
        self.ceiling_positions = set()
        self.door_positions = set()

        # Door installation tracking
        self.door_installed = False

        # Trackers
        self.stuck_detector = StuckDetector()
        self.rewarded_positions = set()

        # Episode state
        self.step_count = 0
        self.ep_reward_sum = 0.0
        self.current_subtask = SUBTASK_FLOOR

        # Agent state
        self.agent_x = 0.0
        self.agent_y = 1.0
        self.agent_z = 0.0
        self.agent_yaw = 0.0
        self.agent_pitch = 0.0
        self.current_hotbar_slot = SLOT_PLANKS

        # Debug
        self.place_attempts = 0
        self.place_successes = 0
        self.correct_placements = 0
        self.incorrect_placements = 0

        # Target queue
        self._target_queue = None
        self._current_target = None

        # LOOKING phase: when True, building completion does not terminate
        # the episode — the training loop runs expert LOOKING instead.
        self.looking_phase = False
        self._building_done_rewarded = False

        # CraftGround init
        self._cg_initialized = False

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FLAT_OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

    # ==================================================================
    # Blueprint generation (origin-based)
    # ==================================================================

    def _generate_blueprint(self):
        """Generate blueprint positions after origin is set."""
        ox, oz = self.origin_x, self.origin_z
        w, d = self.house_width, self.house_depth

        self.floor_positions = set()
        self.wall_positions = set()
        self.ceiling_positions = set()
        self.door_positions = set()
        self.glass_positions = set()

        # Corner positions for glass/planks distinction
        corners = {
            (ox, oz), (ox + w - 1, oz),
            (ox, oz + d - 1), (ox + w - 1, oz + d - 1),
        }

        # Floor (y=1) — oak_planks
        for x in range(ox, ox + w):
            for z in range(oz, oz + d):
                self.floor_positions.add((x, FLOOR_Y, z))

        # Walls (y=2..5, perimeter)
        for y in range(WALL_Y_MIN, WALL_Y_MAX + 1):
            for x in range(ox, ox + w):
                self.wall_positions.add((x, y, oz))           # south
                self.wall_positions.add((x, y, oz + d - 1))   # north
            for z in range(oz + 1, oz + d - 1):
                self.wall_positions.add((ox, y, z))            # west
                self.wall_positions.add((ox + w - 1, y, z))    # east

        # Glass positions: non-corner wall blocks at y < WALL_Y_MAX
        for pos in self.wall_positions:
            x, y, z = pos
            if (x, z) not in corners and y < WALL_Y_MAX:
                self.glass_positions.add(pos)

        # Ceiling (y=5 interior)
        for x in range(ox + 1, ox + w - 1):
            for z in range(oz + 1, oz + d - 1):
                self.ceiling_positions.add((x, WALL_Y_MAX, z))

        # Door: front wall center, y=2,3
        door_x = ox + w // 2
        door_z = oz  # front wall (-Z side)
        self.door_positions = {
            (door_x, DOOR_HEIGHT_BOTTOM, door_z),
            (door_x, DOOR_HEIGHT_TOP, door_z),
        }

        # Remove door from wall positions and glass positions
        self.wall_positions -= self.door_positions
        self.glass_positions -= self.door_positions

        # Blueprint = dict {(x,y,z): block_type}
        self.blueprint_types = {}
        for pos in self.floor_positions:
            self.blueprint_types[pos] = "oak_planks"
        for pos in self.wall_positions:
            if pos in self.glass_positions:
                self.blueprint_types[pos] = "glass"
            else:
                self.blueprint_types[pos] = "oak_planks"
        for pos in self.ceiling_positions:
            self.blueprint_types[pos] = "oak_planks"
        for pos in self.door_positions:
            self.blueprint_types[pos] = "oak_door"

        # Blueprint positions set (for backward compatibility)
        self.blueprint_positions = set(self.blueprint_types.keys())

    # ==================================================================
    # Origin management
    # ==================================================================

    def set_origin(self, x: int, z: int):
        """Set origin from first block placement."""
        self.origin_x = x
        self.origin_z = z
        self.origin_set = True
        self._generate_blueprint()

    def _to_origin_coords(self, x, y, z):
        """Convert world coords to origin-relative coords."""
        if not self.origin_set:
            return x, y, z
        return x - self.origin_x, y, z - self.origin_z

    # ==================================================================
    # Target management
    # ==================================================================

    def set_target_queue(self, queue: list):
        self._target_queue = list(queue)
        self._advance_target()

    def _advance_target(self):
        while self._target_queue:
            t = self._target_queue[0]
            if t in self.blueprint_positions and t not in self.correct_blocks:
                self._current_target = t
                return
            self._target_queue.pop(0)
        self._current_target = None

    def get_current_target(self):
        if self._target_queue is not None:
            self._advance_target()
            return self._current_target
        return self._find_nearest_unbuilt()

    def _find_nearest_unbuilt(self):
        unbuilt = self.blueprint_positions - self.correct_blocks
        if not unbuilt:
            return None
        eye = np.array([self.agent_x, self.agent_y + 1.62, self.agent_z])
        best = None
        best_dist = float('inf')
        for pos in unbuilt:
            center = np.array(pos, dtype=np.float32) + 0.5
            d = np.linalg.norm(center - eye)
            if d < best_dist:
                best_dist = d
                best = pos
        return best

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
        except (AttributeError, TypeError):
            pass

        # Randomize house size
        self.house_width = random.randint(HOUSE_WIDTH_MIN, HOUSE_WIDTH_MAX)
        self.house_depth = random.randint(HOUSE_DEPTH_MIN, HOUSE_DEPTH_MAX)

        # Reset origin
        self.origin_x = None
        self.origin_z = None
        self.origin_set = False

        # Save previous episode's placed blocks for cleanup
        self._prev_placed_blocks = set(self.placed_blocks)

        # Reset block sets
        self.placed_blocks = set()
        self.correct_blocks = set()
        self.incorrect_blocks = set()
        self.blueprint_positions = set()
        self.floor_positions = set()
        self.wall_positions = set()
        self.ceiling_positions = set()
        self.door_positions = set()

        # Reset trackers
        self.stuck_detector.reset()
        self.rewarded_positions = set()
        self.step_count = 0
        self.ep_reward_sum = 0.0
        self.place_attempts = 0
        self.place_successes = 0
        self.correct_placements = 0
        self.incorrect_placements = 0
        self._target_queue = None
        self._current_target = None
        self.current_subtask = SUBTASK_FLOOR
        self.door_installed = False
        self._building_done_rewarded = False

        # Reset CraftGround
        self._reset_craftground()

        self.prev_completion = self._compute_completion()

        obs = self._build_observation()
        info = {
            "completion": self.prev_completion,
            "house_size": (self.house_width, self.house_depth, self.house_height),
        }
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        if int(action[ACT_INTERACT]) == 0:
            self.place_attempts += 1

        block_event = self._execute_action(action)

        if block_event["type"] == "placed":
            self.place_successes += 1
            pos = block_event["position"]

            # First block sets origin
            if not self.origin_set and pos is not None:
                self.set_origin(pos[0], pos[2])

            # Determine placed block type from hotbar slot
            hotbar_slot = int(action[ACT_HOTBAR])
            if hotbar_slot >= SLOT_GLASS:
                placed_type = "glass"
            elif hotbar_slot == SLOT_DOOR:
                placed_type = "oak_door"
            elif hotbar_slot == SLOT_AXE:
                placed_type = "unknown"  # axe doesn't place blocks
            else:
                placed_type = "oak_planks"

            # Track in sets
            self.placed_blocks.add(pos)
            expected_type = self.blueprint_types.get(pos)
            if pos in self.blueprint_positions and (expected_type is None or expected_type == placed_type):
                self.correct_blocks.add(pos)
                self.correct_placements += 1
                # Detect door installation
                if placed_type == "oak_door" and pos in self.door_positions:
                    self.door_installed = True
            else:
                self.incorrect_blocks.add(pos)
                self.incorrect_placements += 1
                if pos not in self.blueprint_positions:
                    print(f"  [env] WRONG block at {pos} (not in blueprint)")
                else:
                    print(f"  [env] WRONG type at {pos} (expected={expected_type}, placed={placed_type})")

            # Advance target
            if (self._target_queue is not None and
                self._current_target is not None and
                pos == self._current_target):
                self._target_queue.pop(0)
                self._advance_target()

        elif block_event["type"] == "removed":
            pos = block_event["position"]
            was_incorrect = pos in self.incorrect_blocks
            was_correct = pos in self.correct_blocks
            self.placed_blocks.discard(pos)
            self.correct_blocks.discard(pos)
            self.incorrect_blocks.discard(pos)
            self.rewarded_positions.discard(pos)
            if was_incorrect and self.incorrect_placements > 0:
                self.incorrect_placements -= 1
            if was_correct and self.correct_placements > 0:
                self.correct_placements -= 1

        completion = self._compute_completion()
        self.current_subtask = self._get_subtask(completion)

        reward, reward_breakdown = self._compute_reward(
            action, block_event, completion
        )

        terminated = False
        truncated = False
        termination_reason = None

        if self.origin_set and completion["total_ratio"] >= 1.0:
            if not self._building_done_rewarded:
                reward += REWARD_STAGE_COMPLETE
                self._building_done_rewarded = True

            if self.looking_phase:
                # Don't terminate — let the training loop run expert LOOKING.
                # Episode will be terminated externally after expert finishes.
                pass
            else:
                terminated = True
                termination_reason = "success"

        # Stuck/timeout only apply during building (not LOOKING phase)
        if self.current_subtask != SUBTASK_DONE:
            if self.stuck_detector.update(completion.get("total_ratio", 0)):
                reward += REWARD_STUCK_PENALTY
                terminated = True
                termination_reason = "stuck"

        if self.step_count >= self.max_episode_steps:
            truncated = True
            termination_reason = "timeout"

        self.ep_reward_sum += reward
        self.prev_completion = completion

        obs = self._build_observation()

        info = {
            "completion": completion,
            "subtask": self.current_subtask,
            "step": self.step_count,
            "termination_reason": termination_reason,
            "reward_total": reward,
            "ep_reward_sum": self.ep_reward_sum,
            "block_event": block_event,
            "agent_pos": (self.agent_x, self.agent_y, self.agent_z),
            "place_attempts": self.place_attempts,
            "place_successes": self.place_successes,
            "correct_placements": self.correct_placements,
            "incorrect_placements": self.incorrect_placements,
            "house_size": (self.house_width, self.house_depth, self.house_height),
        }

        if self.step_count % 200 == 0 or truncated:
            self._debug_log(info)

        return obs, reward, terminated, truncated, info

    # ==================================================================
    # Observation (V3: 45 dimensions)
    # ==================================================================

    def _build_observation(self) -> np.ndarray:
        parts = [
            self._obs_origin_set(),        # 1
            self._obs_agent_state(),        # 10
            self._obs_raycast(),            # 17
            self._obs_progress(),           # 3
            self._obs_incorrect_count(),    # 1
            self._obs_time_remaining(),     # 1
            self._obs_stuck_ratio(),        # 1
            self._obs_target_direction(),   # 4
            self._obs_target_distance(),    # 1
            self._obs_target_absolute(),    # 3
            self._obs_house_size(),         # 3
        ]
        return np.concatenate(parts)

    def _obs_origin_set(self) -> np.ndarray:
        return np.array([1.0 if self.origin_set else 0.0], dtype=np.float32)

    def _obs_agent_state(self) -> np.ndarray:
        """pos(3) + sin/cos yaw(2) + sin/cos pitch(2) + hotbar(1) + has_planks(1) + has_axe(1) = 10"""
        s = np.zeros(AGENT_STATE_SIZE, dtype=np.float32)
        rx, ry, rz = self._to_origin_coords(self.agent_x, self.agent_y, self.agent_z)
        s[0] = rx
        s[1] = ry
        s[2] = rz
        s[3] = math.sin(self.agent_yaw)
        s[4] = math.cos(self.agent_yaw)
        s[5] = math.sin(self.agent_pitch)
        s[6] = math.cos(self.agent_pitch)
        s[7] = self.current_hotbar_slot / 8.0
        s[8] = 1.0 if self.current_hotbar_slot == SLOT_PLANKS else 0.0
        s[9] = 1.0 if self.current_hotbar_slot == SLOT_AXE else 0.0
        return s

    def _obs_raycast(self) -> np.ndarray:
        """17 floats, origin-based absolute coords."""
        info = np.zeros(RAYCAST_INFO_SIZE, dtype=np.float32)

        hit = None
        if self._cg_obs is not None:
            hit = self._cg_obs_extractor.extract_raycast(self._cg_obs)
        if hit is None:
            return info

        idx = 0
        info[idx] = 1.0  # ray_hit
        idx += 1

        # hit_block_type one-hot (3)
        if hit["block_type"] < NUM_BLOCK_TYPES:
            info[idx + hit["block_type"]] = 1.0
        idx += NUM_BLOCK_TYPES

        # hit_pos origin-relative absolute (3)
        hx, hy, hz = hit["position"]
        rx, ry, rz = self._to_origin_coords(hx + 0.5, hy + 0.5, hz + 0.5)
        info[idx:idx + 3] = [rx, ry, rz]
        idx += 3

        # distance (1)
        info[idx] = hit["distance"]
        idx += 1

        # face_normal world absolute (3)
        fn = hit["face_normal"]
        info[idx:idx + 3] = fn
        idx += 3

        # placement_pos origin-relative (3)
        px = hx + fn[0]
        py = hy + fn[1]
        pz = hz + fn[2]
        prx, pry, prz = self._to_origin_coords(px + 0.5, py + 0.5, pz + 0.5)
        info[idx:idx + 3] = [prx, pry, prz]
        idx += 3

        # placement_valid (1)
        pxi, pyi, pzi = int(px), int(py), int(pz)
        info[idx] = 1.0 if self._is_valid_placement(pxi, pyi, pzi) else 0.0
        idx += 1

        # hit_matches_blueprint (1)
        hxi, hyi, hzi = int(hx), int(hy), int(hz)
        if self.origin_set:
            if (hxi, hyi, hzi) in self.correct_blocks:
                info[idx] = 1.0
        idx += 1

        # placement_is_correct (1)
        if self.origin_set and self._is_valid_placement(pxi, pyi, pzi):
            if (pxi, pyi, pzi) in self.blueprint_positions and \
               (pxi, pyi, pzi) not in self.correct_blocks:
                info[idx] = 1.0
        idx += 1

        return info[:RAYCAST_INFO_SIZE]

    def _obs_progress(self) -> np.ndarray:
        c = self.prev_completion if self.prev_completion else {}
        return np.array([
            c.get("floor_ratio", 0.0),
            c.get("wall_ratio", 0.0),
            c.get("ceiling_ratio", 0.0),
        ], dtype=np.float32)

    def _obs_incorrect_count(self) -> np.ndarray:
        # Normalize: assume max ~20 incorrect blocks
        return np.array([len(self.incorrect_blocks) / 20.0], dtype=np.float32)

    def _obs_time_remaining(self) -> np.ndarray:
        return np.array([1.0 - self.step_count / self.max_episode_steps],
                        dtype=np.float32)

    def _obs_stuck_ratio(self) -> np.ndarray:
        return np.array([
            self.stuck_detector.steps_without_progress / self.stuck_detector.patience
        ], dtype=np.float32)

    def _obs_target_direction(self) -> np.ndarray:
        """sin/cos of delta_yaw and delta_pitch to target. 4 floats."""
        result = np.zeros(TARGET_DIRECTION_SIZE, dtype=np.float32)
        target = self.get_current_target()
        if target is None:
            return result

        eye = np.array([self.agent_x, self.agent_y + 1.62, self.agent_z])
        center = np.array(target, dtype=np.float32) + 0.5
        diff = center - eye
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            return result

        target_yaw = math.atan2(diff[0], diff[2])
        delta_yaw = target_yaw - self.agent_yaw
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi

        horiz = math.sqrt(diff[0] ** 2 + diff[2] ** 2)
        target_pitch = math.atan2(-diff[1], horiz) if horiz > 0.01 else 0.0
        delta_pitch = target_pitch - self.agent_pitch

        result[0] = math.sin(delta_yaw)
        result[1] = math.cos(delta_yaw)
        result[2] = math.sin(delta_pitch)
        result[3] = math.cos(delta_pitch)
        return result

    def _obs_target_distance(self) -> np.ndarray:
        target = self.get_current_target()
        if target is None:
            return np.array([0.0], dtype=np.float32)
        eye = np.array([self.agent_x, self.agent_y + 1.62, self.agent_z])
        center = np.array(target, dtype=np.float32) + 0.5
        return np.array([np.linalg.norm(center - eye)], dtype=np.float32)

    def _obs_target_absolute(self) -> np.ndarray:
        """Origin-relative target position. 3 floats."""
        result = np.zeros(TARGET_ABSOLUTE_SIZE, dtype=np.float32)
        target = self.get_current_target()
        if target is None or not self.origin_set:
            return result
        rx, ry, rz = self._to_origin_coords(*target)
        result[:] = [rx, ry, rz]
        return result

    def _obs_house_size(self) -> np.ndarray:
        return np.array([
            self.house_width, self.house_depth, self.house_height
        ], dtype=np.float32)

    # ==================================================================
    # Completion
    # ==================================================================

    def _compute_completion(self) -> dict:
        if not self.origin_set:
            return {"floor_ratio": 0, "wall_ratio": 0, "ceiling_ratio": 0,
                    "total_ratio": 0, "total_correct": 0}

        floor_c = len(self.correct_blocks & self.floor_positions)
        wall_c = len(self.correct_blocks & self.wall_positions)
        ceil_c = len(self.correct_blocks & self.ceiling_positions)
        total_c = floor_c + wall_c + ceil_c
        total_bp = len(self.blueprint_positions)

        return {
            "floor_ratio": floor_c / max(1, len(self.floor_positions)),
            "wall_ratio": wall_c / max(1, len(self.wall_positions)),
            "ceiling_ratio": ceil_c / max(1, len(self.ceiling_positions)),
            "total_ratio": total_c / max(1, total_bp),
            "total_correct": total_c,
        }

    def _get_subtask(self, completion) -> int:
        if completion["floor_ratio"] < 1.0:
            return SUBTASK_FLOOR
        elif completion["wall_ratio"] < 1.0:
            return SUBTASK_WALLS
        elif not self.door_installed:
            return SUBTASK_DOOR
        elif completion["ceiling_ratio"] < 1.0:
            return SUBTASK_CEILING
        return SUBTASK_DONE

    # ==================================================================
    # Action
    # ==================================================================

    def action_masks(self) -> np.ndarray:
        mask = np.ones(sum(ACTION_DIMS), dtype=bool)
        # Mask attack (use axe directly instead)
        interact_start = sum(ACTION_DIMS[:ACT_INTERACT])
        mask[interact_start + 2] = False
        return mask

    def _execute_action(self, action: np.ndarray) -> dict:
        return self._execute_craftground(action)

    def _execute_craftground(self, action: np.ndarray) -> dict:
        from cuboid_house_rl.envs.craftground_adapter import multi_discrete_to_craftground

        interact = int(action[ACT_INTERACT])
        pre_raycast = None
        if interact in (0, 2) and self._cg_obs is not None:
            pre_raycast = self._cg_obs_extractor.extract_raycast(self._cg_obs)

        cg_action = multi_discrete_to_craftground(action)
        cg_obs, _, _, _, _ = self.craftground_env.step(cg_action)
        self._cg_obs = cg_obs

        agent_state = self._cg_obs_extractor.extract_agent_state(cg_obs)
        self.agent_x = agent_state["x"]
        self.agent_y = agent_state["y"]
        self.agent_z = agent_state["z"]
        self.agent_yaw = agent_state["yaw"]
        self.agent_pitch = agent_state["pitch"]
        self._cg_obs_extractor.update_hotbar_slot(action)
        self.current_hotbar_slot = int(action[ACT_HOTBAR])

        # Infer block event from pre-raycast
        if interact == 0 and pre_raycast is not None:
            hotbar_slot = int(action[ACT_HOTBAR])
            # Axe selected → not placing a block (breaking or door opening)
            if hotbar_slot == SLOT_AXE:
                return {"type": "none", "position": None, "correct": None}

            fn = np.array(pre_raycast["face_normal"])
            hp = np.array(pre_raycast["position"])
            place_pos = hp + fn
            px, py, pz = int(place_pos[0]), int(place_pos[1]), int(place_pos[2])
            if self._is_valid_placement(px, py, pz):
                return {"type": "placed", "position": (px, py, pz), "correct": None}

        elif interact == 2 and pre_raycast is not None:
            hx, hy, hz = [int(v) for v in pre_raycast["position"]]
            pos = (hx, hy, hz)
            if pos in self.placed_blocks or pos in self.incorrect_blocks:
                return {"type": "removed", "position": pos, "correct": None}

        return {"type": "none", "position": None, "correct": None}

    def _is_valid_placement(self, x, y, z) -> bool:
        if y < 1:
            return False
        if (x, y, z) in self.placed_blocks:
            return False
        ax = int(round(self.agent_x))
        ay = int(round(self.agent_y))
        az = int(round(self.agent_z))
        if x == ax and z == az and y in (ay, ay + 1):
            return False
        return True

    def get_hotbar_planks_count(self, slot: int) -> int:
        """Get planks count in a specific hotbar slot."""
        if self._cg_obs is None:
            return 64  # assume full if no observation yet
        return self._cg_obs_extractor.get_hotbar_planks_count(self._cg_obs, slot)

    def find_planks_slot(self) -> int:
        """Find first hotbar slot with planks. Returns slot index or -1."""
        if self._cg_obs is None:
            return PLANKS_SLOT
        items = self._cg_obs_extractor.extract_inventory(self._cg_obs)
        for i, item in enumerate(items):
            if i >= 9:  # hotbar is slots 0-8
                break
            if "planks" in item["translation_key"] and item["count"] > 0:
                return i
        return -1

    def find_glass_slot(self) -> int:
        """Find first hotbar slot with glass. Returns slot index or -1."""
        if self._cg_obs is None:
            return SLOT_GLASS
        items = self._cg_obs_extractor.extract_inventory(self._cg_obs)
        for i, item in enumerate(items):
            if i >= 9:
                break
            if "glass" in item["translation_key"] and item["count"] > 0:
                return i
        return -1

    # ==================================================================
    # Reward
    # ==================================================================

    def _compute_reward(self, action, block_event, completion) -> tuple:
        breakdown = {"time_penalty": REWARD_TIME_PENALTY}

        # 1. Block placement
        if block_event["type"] == "placed":
            pos = block_event["position"]
            if self.origin_set and pos in self.blueprint_positions:
                if pos not in self.rewarded_positions:
                    breakdown["correct_placement"] = REWARD_CORRECT_PLACEMENT
                    self.rewarded_positions.add(pos)
            elif self.origin_set:
                breakdown["incorrect_placement"] = REWARD_INCORRECT_PLACEMENT

        # 2. Block removal (axe)
        elif block_event["type"] == "removed":
            pos = block_event["position"]
            if self.origin_set:
                if pos in self.incorrect_blocks or pos not in self.blueprint_positions:
                    # Removed a misplaced block — good
                    breakdown["incorrect_block_removal"] = REWARD_INCORRECT_BLOCK_REMOVAL
                elif pos in self.blueprint_positions:
                    # Removed a correct block — bad
                    breakdown["correct_block_removal"] = REWARD_CORRECT_BLOCK_REMOVAL
                    self.rewarded_positions.discard(pos)

        # 3. Column complete bonus (wall stage: 4 blocks at same x,z)
        if block_event["type"] == "placed" and self.origin_set:
            pos = block_event["position"]
            if pos in self.wall_positions:
                x, y, z = pos
                # Check if all 4 wall heights at this (x,z) are now placed
                column_blocks = {(x, wy, z) for wy in range(WALL_Y_MIN, WALL_Y_MAX + 1)}
                column_correct = column_blocks & self.correct_blocks
                # Only count wall positions (exclude door)
                column_wall = column_blocks & self.wall_positions
                if len(column_correct) == len(column_wall) and len(column_wall) > 0:
                    breakdown["column_complete"] = REWARD_COLUMN_COMPLETE

        # 4. Stage completion bonus
        if self.origin_set and self.prev_completion:
            prev_floor = self.prev_completion.get("floor_ratio", 0)
            prev_wall = self.prev_completion.get("wall_ratio", 0)
            prev_ceil = self.prev_completion.get("ceiling_ratio", 0)
            curr_floor = completion.get("floor_ratio", 0)
            curr_wall = completion.get("wall_ratio", 0)
            curr_ceil = completion.get("ceiling_ratio", 0)

            if curr_floor >= 1.0 and prev_floor < 1.0:
                breakdown["stage_complete"] = REWARD_STAGE_COMPLETE
            elif curr_wall >= 1.0 and prev_wall < 1.0:
                breakdown["stage_complete"] = REWARD_STAGE_COMPLETE
            elif curr_ceil >= 1.0 and prev_ceil < 1.0:
                breakdown["stage_complete"] = REWARD_STAGE_COMPLETE

        # 5. Proximity penalty
        if self.origin_set:
            ox, oz = self.origin_x, self.origin_z
            w, d = self.house_width, self.house_depth
            ax, az = self.agent_x, self.agent_z
            dx = max(ox - ax, 0, ax - (ox + w - 1))
            dz = max(oz - az, 0, az - (oz + d - 1))
            manhattan = dx + dz
            if manhattan > HOUSE_PROXIMITY_THRESHOLD:
                over = manhattan - HOUSE_PROXIMITY_THRESHOLD
                breakdown["proximity_penalty"] = REWARD_PROXIMITY_PER_BLOCK * over

        total = sum(breakdown.values())
        return total, breakdown

    # ==================================================================
    # Debug
    # ==================================================================

    def _debug_log(self, info):
        step = info["step"]
        comp = info["completion"]
        r = info.get("reward_total", 0)
        ep_r = info["ep_reward_sum"]
        pos = info["agent_pos"]
        pc = info["correct_placements"]
        pi = info["incorrect_placements"]

        comp_str = (
            f"F:{comp.get('floor_ratio',0):.0%} "
            f"W:{comp.get('wall_ratio',0):.0%} "
            f"C:{comp.get('ceiling_ratio',0):.0%}"
        )
        pos_str = f"({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
        size = info["house_size"]

        print(
            f"[{step:>5}] r={r:>+7.3f} ep={ep_r:>+8.1f} | "
            f"{comp_str} | pos={pos_str} | "
            f"correct={pc} wrong={pi} | "
            f"size={size[0]}x{size[1]}"
        )

    # ==================================================================
    # CraftGround Reset
    # ==================================================================

    def _reset_craftground(self):
        from cuboid_house_rl.envs.craftground_adapter import (
            multi_discrete_to_craftground, detect_ground_y, get_y_offset,
        )

        if self.craftground_env is None:
            raise RuntimeError("CraftGround env not initialized.")

        if not self._cg_initialized:
            self.craftground_env.reset()
            detect_ground_y(self.craftground_env)
            self._cg_initialized = True

        # Random spawn: teleport to random location
        import random as _rnd
        tp_x = _rnd.randint(-100, 100) + 0.5
        tp_z = _rnd.randint(-100, 100) + 0.5

        # Build noop action (needed for fill wait)
        noop_action = np.zeros(NUM_ACTION_DIMS, dtype=np.int64)
        noop_action[ACT_FWD_BACK] = 1
        noop_action[ACT_LEFT_RIGHT] = 1
        noop_action[ACT_INTERACT] = 1
        noop_action[ACT_PITCH] = 4   # 0° delta (index 4 in 9-option map)
        noop_action[ACT_YAW] = 4
        noop_action[ACT_HOTBAR] = SLOT_PLANKS
        cg_noop = multi_discrete_to_craftground(noop_action)

        # Get ground level for correct Y coordinates
        gy = get_y_offset()  # e.g. 0 or 62
        tp_y = gy + 1        # stand on ground

        # Step 1: tp to new location + wait 1 tick for tp to apply
        self._send_commands([f"tp @p {tp_x} {tp_y} {tp_z} 0 0"])
        self.craftground_env.step(cg_noop)

        # Step 2: flatten area around new spawn (remove villages, old buildings)
        self._send_commands([
            f"fill ~-50 {gy + 1} ~-50 ~50 {gy + 10} ~50 air",
            f"fill ~-50 {gy} ~-50 ~50 {gy} ~50 grass_block",
        ])

        # Step 3: wait 3 ticks for fill to complete
        for _ in range(3):
            self.craftground_env.step(cg_noop)

        # Step 4: setup items
        self._send_commands([
            "clear @p",
            "give @p oak_planks 320",
            "give @p diamond_axe 1",
            "give @p oak_door 64",
            "give @p glass 128",
            "effect give @p saturation 999999 0 true",
            "gamerule fallDamage false",
        ])

        # Probe action: forward + camera yaw delta to detect freeze
        probe_action = noop_action.copy()
        probe_action[ACT_FWD_BACK] = 2   # forward
        probe_action[ACT_YAW] = 6        # +1° yaw delta
        cg_probe = multi_discrete_to_craftground(probe_action)

        # Attempt reset with freeze detection (up to 3 retries)
        MAX_FREEZE_RETRIES = 3
        for attempt in range(MAX_FREEZE_RETRIES):
            # Step noop ticks for commands to take effect
            cg_obs = None
            for _ in range(20):
                cg_obs, _, _, _, _ = self.craftground_env.step(cg_noop)

            # Record baseline position/yaw before probe
            pre_state = self._cg_obs_extractor.extract_agent_state(cg_obs)
            pre_x = pre_state["x"]
            pre_z = pre_state["z"]
            pre_yaw = pre_state["yaw"]

            # Send probe ticks (forward + camera movement)
            for _ in range(5):
                cg_obs, _, _, _, _ = self.craftground_env.step(cg_probe)

            # Check if anything changed
            post_state = self._cg_obs_extractor.extract_agent_state(cg_obs)
            pos_delta = abs(post_state["x"] - pre_x) + abs(post_state["z"] - pre_z)
            yaw_delta = abs(post_state["yaw"] - pre_yaw)

            if pos_delta > 0.01 or yaw_delta > 0.005:
                # Not frozen — success
                if attempt > 0:
                    print(f"  [reset] Freeze resolved after {attempt + 1} attempt(s)")
                break

            # Frozen — try to recover
            print(f"  [reset] Freeze detected (attempt {attempt + 1}/{MAX_FREEZE_RETRIES}): "
                  f"pos_delta={pos_delta:.4f}, yaw_delta={yaw_delta:.4f}")

            # Send ESC key to close any open UI (inventory, chat, etc.)
            self._send_esc_key()

            # Re-send setup commands
            self._send_commands(commands)
        else:
            print("  [reset] WARNING: Could not resolve freeze after all retries")

        # Final: step a few noop ticks to stabilize
        for _ in range(5):
            cg_obs, _, _, _, _ = self.craftground_env.step(cg_noop)

        self._cg_obs = cg_obs

        agent_state = self._cg_obs_extractor.extract_agent_state(cg_obs)
        self.agent_x = agent_state["x"]
        self.agent_y = agent_state["y"]
        self.agent_z = agent_state["z"]
        self.agent_yaw = agent_state["yaw"]
        self.agent_pitch = agent_state["pitch"]
        self.current_hotbar_slot = SLOT_PLANKS

    def _send_commands(self, commands: list):
        """Send commands to CraftGround, trying wrapper then unwrapped."""
        try:
            self.craftground_env.get_wrapper_attr("add_commands")(commands)
        except (AttributeError, TypeError):
            try:
                self.craftground_env.unwrapped.add_commands(commands)
            except (AttributeError, TypeError):
                pass

    def _send_esc_key(self):
        """Send ESC key press to close any open UI in Minecraft."""
        from cuboid_house_rl.envs.craftground_adapter import multi_discrete_to_craftground

        # Method 1: Try sending ESC via key command
        try:
            self._send_commands([""])  # empty command to flush
        except Exception:
            pass

        # Method 2: Build an action that simulates ESC
        # In CraftGround V2, we can try inventory toggle key
        # or send a raw key event. Try multiple approaches.
        try:
            # Some CraftGround versions support raw key events
            esc_commands = [
                "closecontainer",  # Force-close any open container/UI
            ]
            self._send_commands(esc_commands)
        except Exception:
            pass

        # Method 3: Use the action dict to press inventory key (E) twice
        # to open and close inventory, which can unstick UI state
        noop_action = np.zeros(NUM_ACTION_DIMS, dtype=np.int64)
        noop_action[ACT_FWD_BACK] = 1
        noop_action[ACT_LEFT_RIGHT] = 1
        noop_action[ACT_INTERACT] = 1
        noop_action[ACT_PITCH] = 4
        noop_action[ACT_YAW] = 4
        noop_action[ACT_HOTBAR] = SLOT_PLANKS

        cg_action = multi_discrete_to_craftground(noop_action)

        # Try toggling inventory key if available in action space
        try:
            inv_action = dict(cg_action)
            inv_action["inventory"] = 1  # toggle inventory
            self.craftground_env.step(inv_action)
            # Step noop to close it
            for _ in range(3):
                self.craftground_env.step(cg_action)
            inv_action["inventory"] = 1  # toggle again to close
            self.craftground_env.step(inv_action)
            for _ in range(3):
                self.craftground_env.step(cg_action)
        except Exception:
            # If inventory key not supported, just step noops
            for _ in range(5):
                self.craftground_env.step(cg_action)

    # ==================================================================
    # Rendering
    # ==================================================================

    def get_frame(self) -> np.ndarray:
        if self._cg_obs is None:
            return None
        if isinstance(self._cg_obs, dict):
            frame = self._cg_obs.get("pov", self._cg_obs.get("rgb"))
            if frame is not None:
                return np.asarray(frame)
        return None
