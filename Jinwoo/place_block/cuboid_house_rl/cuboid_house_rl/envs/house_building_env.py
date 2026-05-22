"""
House Building Environment.

Gymnasium-compatible wrapper that implements:
- Observation: voxel grids + agent state + raycast info + context
- Action: MultiDiscrete with hotbar masking
- Reward: per-step construction rewards + phase bonuses
- Termination: success, timeout, stuck detection

Supports two backends:
- CraftGround: real Minecraft client via craftground.make()
"""
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cuboid_house_rl.config import (
    # World
    WORLD_SIZE, AIR, OAK_PLANKS, SOLID, NUM_BLOCK_TYPES,
    # House
    TOTAL_BLOCKS,
    HOUSE_ORIGIN_X, HOUSE_ORIGIN_Z, HOUSE_WIDTH, HOUSE_DEPTH,
    FLOOR_Y, CEILING_Y,
    # Spawn
    SPAWN_X, SPAWN_Y, SPAWN_Z, SPAWN_YAW, SPAWN_PITCH,
    # Observation
    LOCAL_WINDOW_SIZE, STACKED_CHANNELS,
    AGENT_STATE_SIZE, RAYCAST_INFO_SIZE, INVENTORY_SIZE,
    COMPLETION_SIZE, SUBTASK_ID_SIZE, PREV_ACTION_SIZE,
    TIME_REMAINING_SIZE, STUCK_RATIO_SIZE, TARGET_DIRECTION_SIZE,
    NON_VOXEL_SIZE,
    RAYCAST_MAX_DISTANCE, NUM_BLOCK_TYPES,
    # Actions
    ACTION_DIMS, NUM_ACTION_DIMS, ACT_FWD_BACK, ACT_LEFT_RIGHT,
    ACT_JUMP, ACT_SNEAK, ACT_INTERACT, ACT_HOTBAR, ACT_PITCH, ACT_YAW,
    CAMERA_DELTA_MAP, PLANKS_SLOT,
    # Rewards
    REWARD_CORRECT_PLACEMENT, REWARD_INCORRECT_PLACEMENT,
    REWARD_INCORRECT_REMOVAL, REWARD_PROGRESS_SCALE,
    REWARD_TIME_PENALTY, REWARD_FLOOR_COMPLETE, REWARD_WALLS_COMPLETE,
    REWARD_CEILING_COMPLETE, REWARD_DOOR_CORRECT, REWARD_HOUSE_COMPLETE,
    REWARD_STUCK_PENALTY,
    REWARD_PROXIMITY_PER_BLOCK, REWARD_LOOKING_AT_TARGET,
    HOUSE_PROXIMITY_THRESHOLD,
    REWARD_STATIONARY_PENALTY, STATIONARY_THRESHOLD,
    HOUSE_ORIGIN_X, HOUSE_ORIGIN_Z, HOUSE_WIDTH, HOUSE_DEPTH,
    # Episode
    MAX_EPISODE_STEPS,
    # Subtasks
    SUBTASK_FLOOR, SUBTASK_WALLS, SUBTASK_CEILING, SUBTASK_DONE,
    NUM_SUBTASKS,
)
from cuboid_house_rl.utils.blueprint import (
    create_blueprint, get_phase_positions, get_door_positions,
)
from cuboid_house_rl.utils.completion import CompletionTracker, StuckDetector
from cuboid_house_rl.utils.coord_transform import (
    world_to_agent_relative, rotate_normal_to_agent,
    extract_local_voxel_window, one_hot_3d,
)


class HouseBuildingEnv(gym.Env):
    """
    Gymnasium environment for cuboid house construction via CraftGround.

    Uses the CraftGround Minecraft client as backend. Handles:
    - Observation: voxel grids + agent state + raycast info + context
    - Action: MultiDiscrete with hotbar masking
    - Reward: per-step construction rewards + phase bonuses
    - Termination: success, timeout, stuck detection
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, craftground_env):
        """
        Args:
            craftground_env: pre-created CraftGround environment
                (from create_craftground_env()).
        """
        super().__init__()

        self.craftground_env = craftground_env
        self._cg_obs = None  # latest CraftGround raw observation

        self.max_episode_steps = MAX_EPISODE_STEPS

        from cuboid_house_rl.envs.craftground_adapter import (
            CraftGroundObsExtractor,
        )
        self._cg_obs_extractor = CraftGroundObsExtractor()

        # Blueprint
        self.blueprint = create_blueprint()
        self.phase_positions = get_phase_positions(self.blueprint)
        self.door_positions = get_door_positions()

        # Trackers
        self.completion_tracker = CompletionTracker(
            self.blueprint, self.phase_positions
        )
        self.stuck_detector = StuckDetector()

        # Per-position reward tracking (first-time-only correct placement)
        self.rewarded_positions = set()

        # Episode state
        self.step_count = 0
        self.prev_action = np.zeros(NUM_ACTION_DIMS, dtype=np.float32)
        self.prev_completion = None
        self.prev_subtask = SUBTASK_FLOOR
        self.current_subtask = SUBTASK_FLOOR

        # Debug counters
        self.place_attempts = 0      # place 액션 선택 횟수
        self.place_successes = 0     # 실제 블록 배치 성공
        self.correct_placements = 0  # 정확한 위치 배치
        self.incorrect_placements = 0  # 틀린 위치 배치

        # Stationary detection
        self.last_block_pos = None
        self.steps_at_same_block = 0

        # CraftGround initialization flag
        self._cg_initialized = False

        # World state (CraftGround updates from surrounding blocks each step)
        self.world = None
        self.agent_x = SPAWN_X
        self.agent_y = SPAWN_Y
        self.agent_z = SPAWN_Z
        self.agent_yaw = SPAWN_YAW
        self.agent_pitch = SPAWN_PITCH
        self.current_hotbar_slot = PLANKS_SLOT

        # ---- Define observation & action spaces ----

        self.observation_space = spaces.Dict({
            "voxel_grids": spaces.Box(
                low=0.0, high=1.0,
                shape=(LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE,
                       LOCAL_WINDOW_SIZE, STACKED_CHANNELS),
                dtype=np.float32,
            ),
            "flat_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(NON_VOXEL_SIZE,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        try:
            super().reset(seed=seed)
        except (AttributeError, TypeError):
            pass

        # Reset world via CraftGround
        self._reset_craftground()

        # Reset trackers
        self.stuck_detector.reset()
        self.rewarded_positions = set()
        self.step_count = 0
        self.prev_action = np.zeros(NUM_ACTION_DIMS, dtype=np.float32)
        self.ep_reward_sum = 0.0  # cumulative episode reward for debugging
        self.place_attempts = 0
        self.place_successes = 0
        self.correct_placements = 0
        self.incorrect_placements = 0
        self.last_block_pos = None
        self.steps_at_same_block = 0

        # Initial completion
        self.prev_completion = self.completion_tracker.compute(self.world)
        self.current_subtask = self.completion_tracker.get_current_subtask(
            self.prev_completion
        )
        self.prev_subtask = self.current_subtask

        obs = self._build_observation()
        info = {
            "completion": self.prev_completion,
            "subtask": self.current_subtask,
        }

        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one step.

        Args:
            action: array of 8 integers (one per MultiDiscrete dimension).

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # 1. Execute action in the world
        if int(action[ACT_INTERACT]) == 0:  # place action selected
            self.place_attempts += 1
        block_event = self._execute_action(action)
        if block_event["type"] == "placed":
            self.place_successes += 1
            if block_event["correct"]:
                self.correct_placements += 1
            else:
                self.incorrect_placements += 1

        # 2. Compute new completion
        completion = self.completion_tracker.compute(self.world)
        self.current_subtask = self.completion_tracker.get_current_subtask(
            completion
        )

        # 3. Compute reward
        reward, reward_breakdown = self._compute_reward(
            action, block_event, completion
        )

        # 4. Check termination
        terminated = False
        truncated = False
        termination_reason = None

        if (completion["floor_ratio"] >= 1.0 and
            completion["wall_ratio"] >= 1.0 and
            completion["ceiling_ratio"] >= 1.0 and
            completion["door_ratio"] >= 1.0):
            reward += REWARD_HOUSE_COMPLETE
            reward_breakdown["house_complete"] = REWARD_HOUSE_COMPLETE
            terminated = True
            termination_reason = "success"

        if self.stuck_detector.update(completion["total_completion"]):
            reward += REWARD_STUCK_PENALTY
            reward_breakdown["stuck_penalty"] = REWARD_STUCK_PENALTY
            terminated = True
            termination_reason = "stuck"

        if self.step_count >= self.max_episode_steps:
            truncated = True
            termination_reason = "timeout"

        # 5. Track cumulative episode reward
        self.ep_reward_sum += reward

        # 6. Update state for next step
        self.prev_completion = completion
        self.prev_action = action.astype(np.float32)
        self.prev_subtask = self.current_subtask

        # 7. Build observation
        obs = self._build_observation()

        # Subtask names for readability
        _subtask_names = {
            SUBTASK_FLOOR: "floor",
            SUBTASK_WALLS: "walls",
            SUBTASK_CEILING: "ceiling",
            SUBTASK_DONE: "done",
        }

        info = {
            "completion": completion,
            "subtask": self.current_subtask,
            "subtask_name": _subtask_names.get(self.current_subtask, "?"),
            "step": self.step_count,
            "termination_reason": termination_reason,
            # Debugging: reward breakdown
            "reward_total": reward,
            "reward_breakdown": reward_breakdown,
            "ep_reward_sum": self.ep_reward_sum,
            # Debugging: place counters
            "place_attempts": self.place_attempts,
            "place_successes": self.place_successes,
            "correct_placements": self.correct_placements,
            "incorrect_placements": self.incorrect_placements,
            # Debugging: block event
            "block_event": block_event,
            # Debugging: agent state
            "agent_pos": (self.agent_x, self.agent_y, self.agent_z),
            "agent_yaw_deg": math.degrees(self.agent_yaw),
            "agent_pitch_deg": math.degrees(self.agent_pitch),
        }

        # Verbose logging (every 50 steps or when something interesting happens)
        # Suppress when wrapped by GazeTrainingEnv
        if not getattr(self, '_gaze_wrapper', False):
            if (self.step_count % 200 == 0 or
                terminated or truncated):
                self._debug_log(info, action)

        return obs, reward, terminated, truncated, info

    def _debug_log(self, info, action):
        """Print a concise debug line for this step."""
        step = info["step"]
        comp = info["completion"]
        be = info["block_event"]
        rb = info["reward_breakdown"]
        r = info["reward_total"]
        ep_r = info["ep_reward_sum"]

        # Action names
        interact_names = {0: "place", 1: "noop", 2: "attack"}
        interact = interact_names.get(int(action[ACT_INTERACT]), "?")
        # Movement action debug
        fwd_names = {0: "back", 1: "stop", 2: "fwd"}
        strafe_names = {0: "left", 1: "stop", 2: "right"}
        move_str = (
            f"{fwd_names.get(int(action[ACT_FWD_BACK]), '?')}/"
            f"{strafe_names.get(int(action[ACT_LEFT_RIGHT]), '?')} "
            f"pitch={self.agent_pitch:.1f}rad"
        )

        # Block event string
        if be["type"] == "placed":
            correct_str = "correct" if be["correct"] else "WRONG"
            event_str = f"placed {be['position']} ({correct_str})"
        elif be["type"] == "removed":
            correct_str = "correct" if be["correct"] else "WRONG"
            event_str = f"removed {be['position']} ({correct_str})"
        else:
            event_str = ""

        # Reward components (only non-zero)
        reward_parts = []
        for k, v in rb.items():
            if abs(v) > 1e-6:
                reward_parts.append(f"{k}={v:+.3f}")
        reward_str = ", ".join(reward_parts) if reward_parts else "time_only"

        # Completion
        comp_str = (
            f"F:{comp['floor_ratio']:.0%} "
            f"W:{comp['wall_ratio']:.0%} "
            f"C:{comp['ceiling_ratio']:.0%} "
            f"D:{'ok' if comp['door_ratio'] >= 1 else 'no'}"
        )

        # Position
        pos = info["agent_pos"]
        pos_str = f"({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"

        # Place stats
        pa = info.get("place_attempts", 0)
        ps = info.get("place_successes", 0)
        pc = info.get("correct_placements", 0)
        pi = info.get("incorrect_placements", 0)
        place_str = f"place:{pa}try/{ps}ok({pc}c/{pi}w)"

        line = (
            f"[{step:>5}] r={r:>+7.3f} ep={ep_r:>+8.1f} | "
            f"{comp_str} | {info['subtask_name']:>7} | "
            f"{interact:>6} {move_str} | pos={pos_str} | {place_str}"
        )
        if event_str:
            line += f" | {event_str}"
        if reward_parts:
            line += f" | [{reward_str}]"

        print(line)

    def action_masks(self) -> np.ndarray:
        """
        Return action masks for MaskablePPO.
        True = action is valid, False = masked.

        Returns:
            1D boolean array of size sum(ACTION_DIMS) = 36.
        """
        mask = np.ones(sum(ACTION_DIMS), dtype=bool)

        # Mask attack action (index 2 in interact dim) — too slow in survival
        interact_start = sum(ACTION_DIMS[:ACT_INTERACT])
        mask[interact_start + 2] = False  # disable attack

        # Mask non-planks hotbar slots
        hotbar_start = sum(ACTION_DIMS[:ACT_HOTBAR])
        for slot in range(ACTION_DIMS[ACT_HOTBAR]):
            if slot != PLANKS_SLOT:
                mask[hotbar_start + slot] = False

        return mask

    # ==================================================================
    # Observation Building
    # ==================================================================

    def _build_observation(self) -> dict:
        """Construct the full observation dict."""
        voxel_grids = self._build_voxel_grids()

        flat_features = np.concatenate([
            self._build_agent_state(),
            self._build_raycast_info(),
            self._build_inventory_info(),
            self._build_completion_features(),
            self._build_subtask_id(),
            self.prev_action,
            self._build_time_remaining(),
            self._build_stuck_ratio(),
            self._build_target_direction(),
        ])

        return {
            "voxel_grids": voxel_grids,
            "flat_features": flat_features,
        }

    def _build_voxel_grids(self) -> np.ndarray:
        """
        Extract and one-hot encode the local voxel window.
        Returns shape (11, 11, 11, 8).

        Channels:
          0-2: block type one-hot (air, planks, solid)
          3-5: visibility one-hot (air, visible, non-visible)
          6:   blueprint target (1 where blueprint wants OAK_PLANKS)
          7:   needs placement (1 where blueprint=OAK_PLANKS AND world=AIR)
        """
        agent_pos = (self.agent_x, self.agent_y, self.agent_z)
        block_window = extract_local_voxel_window(
            self.world, agent_pos, LOCAL_WINDOW_SIZE, default_value=SOLID
        )

        block_onehot = one_hot_3d(block_window, NUM_BLOCK_TYPES)

        visibility_window = self._compute_visibility_grid(block_window)
        visibility_onehot = one_hot_3d(visibility_window, NUM_BLOCK_TYPES)

        # Blueprint channels: extract same local window from blueprint
        blueprint_window = extract_local_voxel_window(
            self.blueprint, agent_pos, LOCAL_WINDOW_SIZE, default_value=AIR
        )

        # Channel 6: blueprint target (where blocks should go)
        blueprint_target = (blueprint_window == OAK_PLANKS).astype(np.float32)

        # Channel 7: needs placement (target AND not yet placed)
        needs_placement = (
            (blueprint_window == OAK_PLANKS) & (block_window == AIR)
        ).astype(np.float32)

        return np.concatenate([
            block_onehot,
            visibility_onehot,
            blueprint_target[..., np.newaxis],
            needs_placement[..., np.newaxis],
        ], axis=-1)

    def _compute_visibility_grid(self, block_window: np.ndarray) -> np.ndarray:
        """
        Compute visibility grid based on distance from agent center.
        0=air, 1=visible solid, 2=non-visible solid.
        """
        center = LOCAL_WINDOW_SIZE // 2
        vis = np.copy(block_window)

        for x in range(LOCAL_WINDOW_SIZE):
            for y in range(LOCAL_WINDOW_SIZE):
                for z in range(LOCAL_WINDOW_SIZE):
                    if block_window[x, y, z] == AIR:
                        vis[x, y, z] = 0
                    else:
                        dist = math.sqrt(
                            (x - center) ** 2 +
                            (y - center) ** 2 +
                            (z - center) ** 2
                        )
                        vis[x, y, z] = 1 if dist <= RAYCAST_MAX_DISTANCE else 2

        return vis

    def _build_agent_state(self) -> np.ndarray:
        """Agent position, orientation, hotbar. ~15 floats."""
        state = np.zeros(AGENT_STATE_SIZE, dtype=np.float32)
        state[0] = self.agent_x
        state[1] = self.agent_y
        state[2] = self.agent_z
        state[3] = self.agent_yaw / math.pi
        state[4] = self.agent_pitch / math.pi
        state[5 + self.current_hotbar_slot] = 1.0
        state[14] = 1.0 if self.current_hotbar_slot == PLANKS_SLOT else 0.0
        return state

    def _build_raycast_info(self) -> np.ndarray:
        """Raycast information in agent-relative coordinates. ~25 floats."""
        info = np.zeros(RAYCAST_INFO_SIZE, dtype=np.float32)

        # Get raycast hit from CraftGround
        if self._cg_obs is not None:
            hit = self._cg_obs_extractor.extract_raycast(self._cg_obs)
        else:
            hit = None

        if hit is None:
            info[0] = 0.0
            return info

        info[0] = 1.0  # ray_hit

        idx = 1
        # hit_block_type: one-hot (3)
        if hit["block_type"] < NUM_BLOCK_TYPES:
            info[idx + hit["block_type"]] = 1.0
        idx += NUM_BLOCK_TYPES

        # hit_position: agent-relative (3)
        agent_pos = np.array([self.agent_x, self.agent_y, self.agent_z],
                             dtype=np.float32)
        hit_pos_world = np.array(hit["position"], dtype=np.float32)
        hit_pos_local = world_to_agent_relative(
            hit_pos_world, agent_pos, self.agent_yaw
        )
        info[idx:idx + 3] = hit_pos_local
        idx += 3

        # hit_distance: normalized (1)
        info[idx] = hit["distance"] / RAYCAST_MAX_DISTANCE
        idx += 1

        # hit_face_normal: agent-relative (3)
        face_normal_world = np.array(hit["face_normal"], dtype=np.float32)
        face_normal_local = rotate_normal_to_agent(
            face_normal_world, self.agent_yaw
        )
        info[idx:idx + 3] = face_normal_local
        idx += 3

        # placement_position: agent-relative (3)
        placement_world = hit_pos_world + face_normal_world
        placement_local = world_to_agent_relative(
            placement_world, agent_pos, self.agent_yaw
        )
        info[idx:idx + 3] = placement_local
        idx += 3

        # placement_is_valid (1)
        px, py, pz = int(placement_world[0]), int(placement_world[1]), int(placement_world[2])
        placement_valid = self._is_valid_placement(px, py, pz)
        info[idx] = 1.0 if placement_valid else 0.0
        idx += 1

        # hit_matches_blueprint (1)
        hx, hy, hz = int(hit_pos_world[0]), int(hit_pos_world[1]), int(hit_pos_world[2])
        if 0 <= hx < WORLD_SIZE and 0 <= hy < WORLD_SIZE and 0 <= hz < WORLD_SIZE:
            info[idx] = 1.0 if self.world[hx, hy, hz] == self.blueprint[hx, hy, hz] else 0.0
        idx += 1

        # hit_blueprint_type: one-hot (2)
        if 0 <= hx < WORLD_SIZE and 0 <= hy < WORLD_SIZE and 0 <= hz < WORLD_SIZE:
            bp_type = self.blueprint[hx, hy, hz]
            if bp_type == AIR:
                info[idx] = 1.0
            elif bp_type == OAK_PLANKS:
                info[idx + 1] = 1.0
        idx += 2

        # placement_blueprint_type: one-hot (2)
        if 0 <= px < WORLD_SIZE and 0 <= py < WORLD_SIZE and 0 <= pz < WORLD_SIZE:
            bp_type = self.blueprint[px, py, pz]
            if bp_type == AIR:
                info[idx] = 1.0
            elif bp_type == OAK_PLANKS:
                info[idx + 1] = 1.0
        idx += 2

        # placement_would_be_correct (1)
        if (placement_valid and
            0 <= px < WORLD_SIZE and 0 <= py < WORLD_SIZE and 0 <= pz < WORLD_SIZE):
            info[idx] = 1.0 if (
                self.blueprint[px, py, pz] == OAK_PLANKS and
                self.current_hotbar_slot == PLANKS_SLOT
            ) else 0.0
        idx += 1

        return info[:RAYCAST_INFO_SIZE]

    def _build_inventory_info(self) -> np.ndarray:
        """Current slot + has_planks. 10 floats."""
        info = np.zeros(INVENTORY_SIZE, dtype=np.float32)
        info[self.current_hotbar_slot] = 1.0
        info[9] = 1.0 if self.current_hotbar_slot == PLANKS_SLOT else 0.0
        return info

    def _build_completion_features(self) -> np.ndarray:
        """Floor, wall, ceiling, door ratios. 4 floats."""
        c = self.prev_completion
        return np.array([
            c["floor_ratio"], c["wall_ratio"],
            c["ceiling_ratio"], c["door_ratio"],
        ], dtype=np.float32)

    def _build_subtask_id(self) -> np.ndarray:
        """One-hot subtask. 4 floats."""
        subtask = np.zeros(NUM_SUBTASKS, dtype=np.float32)
        subtask[self.current_subtask] = 1.0
        return subtask

    def _build_time_remaining(self) -> np.ndarray:
        """Normalized time remaining. 1 float."""
        remaining = 1.0 - (self.step_count / self.max_episode_steps)
        return np.array([remaining], dtype=np.float32)

    def _build_stuck_ratio(self) -> np.ndarray:
        """Stuck progress toward penalty. 0.0=just progressed, 1.0=about to be penalized."""
        ratio = self.stuck_detector.steps_without_progress / self.stuck_detector.patience
        return np.array([ratio], dtype=np.float32)

    def _build_target_direction(self) -> np.ndarray:
        """Direction to nearest unbuilt blueprint block. 3 floats:
        delta_yaw (-1~+1), delta_pitch (-1~+1), normalized distance."""
        result = np.zeros(TARGET_DIRECTION_SIZE, dtype=np.float32)

        # Find all unbuilt blueprint positions
        needs = np.argwhere(
            (self.blueprint == OAK_PLANKS) & (self.world == AIR)
        )
        if len(needs) == 0:
            return result  # all built

        # Agent eye position (block center + eye height)
        eye = np.array([self.agent_x, self.agent_y + 1.6, self.agent_z])

        # Find nearest unbuilt block (use block centers +0.5)
        block_centers = needs.astype(np.float32) + 0.5
        diffs = block_centers - eye
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = np.argmin(dists)
        diff = diffs[nearest_idx]
        dist = dists[nearest_idx]

        if dist < 0.01:
            return result

        # Target yaw: angle in XZ plane (our convention: 0=south/+Z)
        target_yaw = math.atan2(diff[0], diff[2])
        delta_yaw = target_yaw - self.agent_yaw
        # Normalize to [-pi, pi]
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi

        # Target pitch: angle from horizontal
        horiz_dist = math.sqrt(diff[0] ** 2 + diff[2] ** 2)
        target_pitch = math.atan2(-diff[1], horiz_dist) if horiz_dist > 0.01 else 0.0
        delta_pitch = target_pitch - self.agent_pitch

        result[0] = max(-1.0, min(1.0, delta_yaw / math.pi))   # normalized
        result[1] = max(-1.0, min(1.0, delta_pitch / (math.pi / 2)))
        result[2] = min(1.0, dist / WORLD_SIZE)  # normalized distance
        return result

    # ==================================================================
    # Action Execution
    # ==================================================================

    def _execute_action(self, action: np.ndarray) -> dict:
        """Execute the multi-discrete action. Returns block_event dict."""
        return self._execute_craftground(action)

    def _execute_craftground(self, action: np.ndarray) -> dict:
        """
        Execute action via CraftGround and detect block placement/removal.

        Strategy:
        1. Snapshot house region before action
        2. Send action to CraftGround
        3. Update agent state
        4. Update world from surrounding blocks (3x3x3)
        5. Diff house region to detect changes from surrounding blocks
        6. If diff found nothing but action was place/attack, infer from
           raycast (handles blocks outside 3x3x3 range)
        """
        from cuboid_house_rl.envs.craftground_adapter import (
            multi_discrete_to_craftground,
        )

        # Remember the interact action BEFORE stepping
        interact = int(action[ACT_INTERACT])

        # Get pre-step raycast info (needed for fallback inference)
        pre_raycast = None
        if interact in (0, 2) and self._cg_obs is not None:
            pre_raycast = self._cg_obs_extractor.extract_raycast(self._cg_obs)

        # Snapshot house region before action
        before_snapshot = self._snapshot_house_region()

        # Convert and step
        cg_action = multi_discrete_to_craftground(action)

        # Debug: log raw action every 200 steps
        if not getattr(self, '_gaze_wrapper', False):
            if self.step_count <= 3 or self.step_count % 200 == 0:
                print(f"  [DEBUG step={self.step_count}] raw_action={action.tolist()} "
                      f"cg_forward={cg_action.get('forward')}, cg_back={cg_action.get('back')}, "
                      f"cg_pitch={cg_action.get('camera_pitch'):.1f}, "
                      f"cg_yaw={cg_action.get('camera_yaw'):.1f}")

        cg_obs, _, cg_terminated, cg_truncated, cg_info = (
            self.craftground_env.step(cg_action)
        )

        self._cg_obs = cg_obs

        # Update agent state from CraftGround
        agent_state = self._cg_obs_extractor.extract_agent_state(cg_obs)
        self.agent_x = agent_state["x"]
        self.agent_y = agent_state["y"]
        self.agent_z = agent_state["z"]
        self.agent_yaw = agent_state["yaw"]
        self.agent_pitch = agent_state["pitch"]

        # Update hotbar tracking
        self._cg_obs_extractor.update_hotbar_slot(action)
        self.current_hotbar_slot = int(action[ACT_HOTBAR])

        # Update world from surrounding blocks (3x3x3 around agent)
        self.world = self._cg_obs_extractor.build_world_from_surrounding(
            cg_obs, self.world
        )

        # Detect what changed by diffing house region
        block_event = self._detect_block_change(before_snapshot)

        # If diff found a change, we're done — the world is already updated
        # by build_world_from_surrounding.
        if block_event["type"] != "none":
            return block_event

        # Fallback: if the agent performed place/attack but the diff saw
        # nothing, the block may be outside the 3x3x3 surrounding range.
        # Infer the event from the pre-step raycast.
        if interact == 0 and pre_raycast is not None:
            # Place action: block goes at hit_position + face_normal
            fn = np.array(pre_raycast["face_normal"])
            hp = np.array(pre_raycast["position"])
            place_pos = hp + fn
            px, py, pz = int(place_pos[0]), int(place_pos[1]), int(place_pos[2])

            if self._is_valid_placement(px, py, pz):
                self.world[px, py, pz] = OAK_PLANKS
                correct = (self.blueprint[px, py, pz] == OAK_PLANKS)
                return {
                    "type": "placed",
                    "position": (px, py, pz),
                    "correct": correct,
                }

        elif interact == 2 and pre_raycast is not None:
            # Attack action: break the hit block
            hx, hy, hz = pre_raycast["position"]
            if (0 <= hx < WORLD_SIZE and 0 <= hy < WORLD_SIZE and
                0 <= hz < WORLD_SIZE and self.world[hx, hy, hz] == OAK_PLANKS):
                should_stay = (self.blueprint[hx, hy, hz] == OAK_PLANKS)
                self.world[hx, hy, hz] = AIR
                return {
                    "type": "removed",
                    "position": (hx, hy, hz),
                    "correct": not should_stay,
                }

        return {"type": "none", "position": None, "correct": None}

    def _snapshot_house_region(self) -> np.ndarray:
        """Snapshot the house building region from the world array."""
        x0 = HOUSE_ORIGIN_X
        x1 = HOUSE_ORIGIN_X + HOUSE_WIDTH
        z0 = HOUSE_ORIGIN_Z
        z1 = HOUSE_ORIGIN_Z + HOUSE_DEPTH
        return self.world[x0:x1, FLOOR_Y:CEILING_Y + 1, z0:z1].copy()

    def _detect_block_change(self, before_snapshot: np.ndarray) -> dict:
        """Compare house region before/after to detect placement/removal."""
        x0 = HOUSE_ORIGIN_X
        z0 = HOUSE_ORIGIN_Z

        after_snapshot = self.world[
            x0:x0 + HOUSE_WIDTH, FLOOR_Y:CEILING_Y + 1, z0:z0 + HOUSE_DEPTH
        ]

        diff = after_snapshot != before_snapshot
        if not diff.any():
            return {"type": "none", "position": None, "correct": None}

        changed = np.argwhere(diff)
        if len(changed) == 0:
            return {"type": "none", "position": None, "correct": None}

        rx, ry, rz = changed[0]
        wx, wy, wz = x0 + rx, FLOOR_Y + ry, z0 + rz

        old_val = before_snapshot[rx, ry, rz]
        new_val = after_snapshot[rx, ry, rz]

        if old_val == AIR and new_val == OAK_PLANKS:
            correct = (self.blueprint[wx, wy, wz] == OAK_PLANKS)
            return {"type": "placed", "position": (wx, wy, wz), "correct": correct}
        elif old_val == OAK_PLANKS and new_val == AIR:
            should_stay = (self.blueprint[wx, wy, wz] == OAK_PLANKS)
            return {"type": "removed", "position": (wx, wy, wz), "correct": not should_stay}
        else:
            return {"type": "none", "position": None, "correct": None}

    # ==================================================================
    # Reward Computation
    # ==================================================================

    def _compute_reward(self, action, block_event, completion) -> tuple:
        """
        Compute the total reward for this step.

        Returns:
            (total_reward, reward_breakdown) where reward_breakdown is a dict
            with each component for debugging.
        """
        breakdown = {
            "block_placement": 0.0,
            "progress": 0.0,
            "phase_bonus": 0.0,
            "time_penalty": REWARD_TIME_PENALTY,
        }

        # 1. Block placement / removal
        if block_event["type"] == "placed":
            pos = block_event["position"]
            if block_event["correct"]:
                if pos not in self.rewarded_positions:
                    breakdown["block_placement"] = REWARD_CORRECT_PLACEMENT
                    self.rewarded_positions.add(pos)
            else:
                breakdown["block_placement"] = REWARD_INCORRECT_PLACEMENT

        elif block_event["type"] == "removed":
            if not block_event["correct"]:
                breakdown["block_placement"] = REWARD_INCORRECT_REMOVAL
                pos = block_event["position"]
                self.rewarded_positions.discard(pos)

        # 2. Progress
        if self.prev_completion is not None:
            subtask_key = {
                SUBTASK_FLOOR: "floor_ratio",
                SUBTASK_WALLS: "wall_ratio",
                SUBTASK_CEILING: "ceiling_ratio",
                SUBTASK_DONE: "floor_ratio",
            }[self.current_subtask]

            prev_ratio = self.prev_completion[subtask_key]
            curr_ratio = completion[subtask_key]
            breakdown["progress"] = (curr_ratio - prev_ratio) * REWARD_PROGRESS_SCALE

        # 3. Stationary penalty (same block position for too long)
        current_block = (int(round(self.agent_x)), int(round(self.agent_y)), int(round(self.agent_z)))
        if current_block == self.last_block_pos:
            self.steps_at_same_block += 1
        else:
            self.steps_at_same_block = 0
            self.last_block_pos = current_block
        if self.steps_at_same_block >= STATIONARY_THRESHOLD:
            breakdown["stationary_penalty"] = REWARD_STATIONARY_PENALTY

        # 4. Proximity penalty (linear, per block beyond threshold)
        ax, az = self.agent_x, self.agent_z
        dx = max(HOUSE_ORIGIN_X - ax, 0, ax - (HOUSE_ORIGIN_X + HOUSE_WIDTH - 1))
        dz = max(HOUSE_ORIGIN_Z - az, 0, az - (HOUSE_ORIGIN_Z + HOUSE_DEPTH - 1))
        manhattan_dist = dx + dz
        if manhattan_dist > HOUSE_PROXIMITY_THRESHOLD:
            over = manhattan_dist - HOUSE_PROXIMITY_THRESHOLD
            breakdown["proximity_penalty"] = REWARD_PROXIMITY_PER_BLOCK * over

        # 4. Looking at unbuilt target block reward
        if self._cg_obs is not None:
            hit = self._cg_obs_extractor.extract_raycast(self._cg_obs)
            if hit is not None:
                hp = np.array(hit["position"])
                fn = np.array(hit["face_normal"])
                place_pos = hp + fn
                px, py, pz = int(place_pos[0]), int(place_pos[1]), int(place_pos[2])
                if (0 <= px < WORLD_SIZE and 0 <= py < WORLD_SIZE and 0 <= pz < WORLD_SIZE):
                    if (self.blueprint[px, py, pz] == OAK_PLANKS and
                            self.world[px, py, pz] == AIR):
                        breakdown["looking_at_target"] = REWARD_LOOKING_AT_TARGET

        # 5. Phase transition bonuses
        if self.prev_subtask != self.current_subtask:
            if self.prev_subtask == SUBTASK_FLOOR and self.current_subtask == SUBTASK_WALLS:
                breakdown["phase_bonus"] = REWARD_FLOOR_COMPLETE
            elif self.prev_subtask == SUBTASK_WALLS and self.current_subtask == SUBTASK_CEILING:
                breakdown["phase_bonus"] = REWARD_WALLS_COMPLETE
                if completion["door_ratio"] >= 1.0:
                    breakdown["phase_bonus"] += REWARD_DOOR_CORRECT
            elif self.prev_subtask == SUBTASK_CEILING and self.current_subtask == SUBTASK_DONE:
                breakdown["phase_bonus"] = REWARD_CEILING_COMPLETE

        total = sum(breakdown.values())
        return total, breakdown

    def _is_valid_placement(self, x, y, z) -> bool:
        """Check if a block can be placed at (x, y, z)."""
        if not (0 <= x < WORLD_SIZE and 0 <= y < WORLD_SIZE and 0 <= z < WORLD_SIZE):
            return False
        if self.world[x, y, z] != AIR:
            return False
        ax, ay, az = int(round(self.agent_x)), int(round(self.agent_y)), int(round(self.agent_z))
        if x == ax and z == az and y in (ay, ay + 1):
            return False
        return True

    # ==================================================================
    # Rendering
    # ==================================================================

    def get_frame(self) -> np.ndarray:
        """
        Get the current RGB frame from CraftGround.

        Returns:
            np.ndarray of shape (H, W, 3) in RGB format, or None.
        """
        if self._cg_obs is None:
            return None
        if isinstance(self._cg_obs, dict):
            # CraftGround obs dict has "pov" or "rgb" key
            frame = self._cg_obs.get("pov", self._cg_obs.get("rgb"))
            if frame is not None:
                return np.asarray(frame)
        return None

    # ==================================================================
    # CraftGround Reset
    # ==================================================================

    def _reset_craftground(self):
        """Reset the CraftGround environment for a new episode."""
        from cuboid_house_rl.envs.craftground_adapter import (
            get_reset_commands, multi_discrete_to_craftground,
            detect_ground_y,
        )

        if self.craftground_env is None:
            raise RuntimeError(
                "CraftGround env not initialized. Pass craftground_env to "
                "constructor or use create_craftground_env()."
            )

        # First time: start CraftGround server and detect ground level
        if not self._cg_initialized:
            self.craftground_env.reset()  # starts server, runs initial_extra_commands
            detect_ground_y(self.craftground_env)  # wait for landing, read Y
            self._cg_initialized = True

        # Queue reset commands via the official CraftGround API.
        # add_commands() queues commands to execute on the next step().
        reset_cmds = get_reset_commands()

        try:
            # Official API: wrapper method that queues commands
            self.craftground_env.get_wrapper_attr("add_commands")(reset_cmds)
        except (AttributeError, TypeError):
            # Fallback: try unwrapped env's add_commands
            try:
                self.craftground_env.unwrapped.add_commands(reset_cmds)
            except (AttributeError, TypeError):
                pass

        # Build a no-op action to step through while commands execute
        noop_action = np.zeros(NUM_ACTION_DIMS, dtype=np.int64)
        noop_action[ACT_FWD_BACK] = 1      # stop
        noop_action[ACT_LEFT_RIGHT] = 1    # stop
        noop_action[ACT_INTERACT] = 1      # nothing
        noop_action[ACT_PITCH] = 3         # no camera change (index 3 = 0°)
        noop_action[ACT_YAW] = 3
        noop_action[ACT_HOTBAR] = PLANKS_SLOT

        cg_action = multi_discrete_to_craftground(noop_action)

        # Re-initialize world array BEFORE warmup steps so we can
        # accumulate surrounding blocks from every step.
        self.world = np.full(
            (WORLD_SIZE, WORLD_SIZE, WORLD_SIZE), AIR, dtype=np.int8
        )
        self.world[:, 0, :] = SOLID  # ground layer (superflat world)

        # Step several times to let the server process all commands.
        # The first step executes the queued commands.
        # Accumulate surrounding blocks from ALL warmup steps to build
        # a larger initial world view (each step adds ~3x3x3 blocks).
        # Also includes freeze detection: if pos doesn't change after
        # warmup, force full CraftGround re-initialization.
        MAX_FREEZE_RETRIES = 5

        for retry in range(MAX_FREEZE_RETRIES):
            # Warmup: 3 noop steps (let teleport execute), then 2 forward steps
            for i in range(3):
                cg_obs, _, _, _, _ = self.craftground_env.step(cg_action)
                self.world = self._cg_obs_extractor.build_world_from_surrounding(
                    cg_obs, self.world
                )

            # Record pos after noop warmup
            pre_state = self._cg_obs_extractor.extract_agent_state(cg_obs)

            # Send 2 forward movement steps to test responsiveness
            move_action = noop_action.copy()
            move_action[ACT_FWD_BACK] = 2  # forward
            cg_move = multi_discrete_to_craftground(move_action)
            for _ in range(2):
                cg_obs, _, _, _, _ = self.craftground_env.step(cg_move)
                self.world = self._cg_obs_extractor.build_world_from_surrounding(
                    cg_obs, self.world
                )

            # Check if agent moved during the forward steps
            post_state = self._cg_obs_extractor.extract_agent_state(cg_obs)
            dx = abs(post_state["x"] - pre_state["x"])
            dz = abs(post_state["z"] - pre_state["z"])

            if dx > 0.01 or dz > 0.01:
                break  # CraftGround is responding
            else:
                print(f"  [FREEZE DETECTED] retry {retry+1}/{MAX_FREEZE_RETRIES} "
                      f"— pos unchanged after forward cmds "
                      f"({pre_state['x']:.1f},{pre_state['z']:.1f}), "
                      f"re-initializing CraftGround...")
                self._cg_initialized = False
                try:
                    self.craftground_env.close()
                except Exception:
                    pass
                import time
                time.sleep(1)
                self.craftground_env.reset()
                detect_ground_y(self.craftground_env)
                self._cg_initialized = True

                # Re-queue reset commands
                reset_cmds = get_reset_commands()
                try:
                    self.craftground_env.get_wrapper_attr("add_commands")(reset_cmds)
                except (AttributeError, TypeError):
                    try:
                        self.craftground_env.unwrapped.add_commands(reset_cmds)
                    except (AttributeError, TypeError):
                        pass

                # Re-initialize world array
                self.world = np.full(
                    (WORLD_SIZE, WORLD_SIZE, WORLD_SIZE), AIR, dtype=np.int8
                )
                self.world[:, 0, :] = SOLID
        else:
            print(f"  [FREEZE] Failed after {MAX_FREEZE_RETRIES} retries, continuing anyway")

        self._cg_obs = cg_obs

        # Update agent state
        agent_state = self._cg_obs_extractor.extract_agent_state(cg_obs)
        self.agent_x = agent_state["x"]
        self.agent_y = agent_state["y"]
        self.agent_z = agent_state["z"]
        self.agent_yaw = agent_state["yaw"]
        self.agent_pitch = agent_state["pitch"]
        self.current_hotbar_slot = PLANKS_SLOT

