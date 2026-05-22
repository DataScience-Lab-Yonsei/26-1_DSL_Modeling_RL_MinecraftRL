"""
Stage 1: Gaze Training Environment (Floor only).

Wraps HouseBuildingEnv to teach the agent camera control.
The agent must move + aim to look at floor target positions.
Success = raycast hits the top face of the block below a floor target.
Place/attack are masked — only movement and camera actions are available.

Free-form targeting: the agent can gaze at ANY unplaced floor block.
Each block requires 3 gaze hits to be marked as "placed".
target_direction observation points to the nearest unplaced block.

Observation space is identical to HouseBuildingEnv for weight transfer.
"""
import math
import gymnasium as gym
import numpy as np
from collections import deque

from cuboid_house_rl.config import (
    ACT_INTERACT, ACTION_DIMS, TARGET_DIRECTION_SIZE,
    OAK_PLANKS, AIR, WORLD_SIZE, FLOOR_Y,
    GAZE_REWARD_SUCCESS, GAZE_REWARD_ANGULAR_SCALE,
    GAZE_MAX_EPISODE_STEPS,
    GAZE_TIME_PENALTY, GAZE_GRADUATION_THRESHOLD, GAZE_GRADUATION_WINDOW,
)
from cuboid_house_rl.envs.house_building_env import HouseBuildingEnv
from cuboid_house_rl.utils.blueprint import get_phase_positions


# Proximity reward: inverse distance to nearest target from agent body
GAZE_PROXIMITY_REWARD_SCALE = 0.05  # max proximity reward per step
GAZE_HITS_REQUIRED = 3  # hits needed per target to confirm placement


class GazeTrainingEnv(gym.Env):
    """
    Stage 1 curriculum: learn to look at floor target block positions.

    Free-form: agent can gaze at any unplaced floor block in any order.
    Each block needs GAZE_HITS_REQUIRED (3) raycast hits to be "placed".
    target_direction always points to the nearest unplaced block.
    """

    def __init__(self, craftground_env):
        super().__init__()
        self.inner_env = HouseBuildingEnv(craftground_env)

        # Disable inner env's stuck detector and episode limits
        # (GazeTrainingEnv manages its own termination)
        self.inner_env.stuck_detector.patience = 999999
        self.inner_env.max_episode_steps = 999999
        self.inner_env._gaze_wrapper = True  # suppress inner logging

        # Same spaces for weight transfer
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space

        # Floor targets
        self.phase_positions = get_phase_positions(self.inner_env.blueprint)
        self.all_floor_positions = list(self.phase_positions["floor"])
        self.total_targets = len(self.all_floor_positions)

        # Per-target gaze counters: {(x,y,z): hit_count}
        self.gaze_counts = {}
        self.remaining_targets = set()
        self.targets_completed = 0
        self.prev_angular_dist = None
        self.prev_nearest_target = None

        # Episode tracking
        self.step_count = 0
        self.ep_reward_sum = 0.0

        # Graduation tracking
        self.success_history = deque(maxlen=GAZE_GRADUATION_WINDOW)
        self.graduated = False

    def reset(self, seed=None, options=None):
        obs, info = self.inner_env.reset(seed=seed, options=options)

        self.remaining_targets = set(
            tuple(p) for p in self.all_floor_positions
        )
        self.gaze_counts = {t: 0 for t in self.remaining_targets}
        self.targets_completed = 0
        self.prev_angular_dist = None
        self.prev_nearest_target = None
        self.step_count = 0
        self.ep_reward_sum = 0.0

        obs = self._rebuild_observation()
        return obs, info

    def step(self, action):
        self.step_count += 1

        # Force interact to noop
        action = action.copy()
        action[ACT_INTERACT] = 1

        # Step inner env
        obs, _, terminated, truncated, info = self.inner_env.step(action)

        # Find nearest target (used for angular shaping + observation)
        nearest_target = self._find_nearest_target()

        # Compute gaze reward
        reward, gaze_info = self._compute_gaze_reward(nearest_target)

        # Check if raycast hit ANY remaining target
        hit_pos = gaze_info.get("hit_pos")
        if hit_pos is not None and hit_pos in self.remaining_targets:
            self.gaze_counts[hit_pos] += 1
            if self.gaze_counts[hit_pos] >= GAZE_HITS_REQUIRED:
                # Confirmed: mark as placed in world
                self.targets_completed += 1
                self.inner_env.world[hit_pos[0], hit_pos[1], hit_pos[2]] = OAK_PLANKS
                self.remaining_targets.discard(hit_pos)
                # Nearest target may have changed
                nearest_target = self._find_nearest_target()
                self.prev_angular_dist = None

        self.ep_reward_sum += reward

        # Termination
        terminated = len(self.remaining_targets) == 0
        truncated = self.step_count >= GAZE_MAX_EPISODE_STEPS

        if terminated or truncated:
            success_rate = self.targets_completed / max(1, self.total_targets)
            self.success_history.append(success_rate)
            if (len(self.success_history) >= GAZE_GRADUATION_WINDOW and
                    np.mean(self.success_history) >= GAZE_GRADUATION_THRESHOLD):
                self.graduated = True

        # Rebuild obs with nearest target direction
        obs = self._rebuild_observation()

        info["gaze_targets_hit"] = self.targets_completed
        info["gaze_targets_total"] = self.total_targets
        info["gaze_success_rate"] = self.targets_completed / max(1, self.total_targets)
        info["gaze_graduated"] = self.graduated
        info["gaze_angular_dist"] = gaze_info.get("angular_dist", 0.0)

        # Per-step log (every 200 steps + termination)
        if self.step_count % 200 == 0 or (terminated or truncated):
            nt = nearest_target if nearest_target else (0, 0, 0)
            print(
                f"[{self.step_count:>5}] r={reward:>+7.3f} ep={self.ep_reward_sum:>+8.1f} | "
                f"done {self.targets_completed}/{self.total_targets} "
                f"({self.targets_completed/max(1,self.total_targets):.0%}) | "
                f"nearest=({nt[0]},{nt[1]},{nt[2]}) | "
                f"ang={gaze_info.get('angular_dist', 0):.2f} | "
                f"prox={gaze_info.get('proximity_reward', 0):+.3f} | "
                f"pos=({self.inner_env.agent_x:.1f},{self.inner_env.agent_y:.1f},"
                f"{self.inner_env.agent_z:.1f}) | "
                f"pitch={self.inner_env.agent_pitch:.1f}rad"
            )

        return obs, reward, terminated, truncated, info

    def _find_nearest_target(self):
        """Find the nearest remaining target to the agent."""
        if not self.remaining_targets:
            return None
        agent_pos = np.array([
            self.inner_env.agent_x,
            self.inner_env.agent_y,
            self.inner_env.agent_z,
        ])
        best = None
        best_dist = float("inf")
        for t in self.remaining_targets:
            d = abs(agent_pos[0] - t[0]) + abs(agent_pos[1] - t[1]) + abs(agent_pos[2] - t[2])
            if d < best_dist:
                best_dist = d
                best = t
        return best

    def _compute_gaze_reward(self, nearest_target):
        """Reward based on looking at any remaining target."""
        if nearest_target is None:
            return GAZE_TIME_PENALTY, {"hit_pos": None, "angular_dist": 0.0,
                                        "proximity_reward": 0.0}

        # Check raycast — did agent look at any remaining target?
        hit_pos = None
        if self.inner_env._cg_obs is not None:
            hit = self.inner_env._cg_obs_extractor.extract_raycast(
                self.inner_env._cg_obs
            )
            if hit is not None:
                hp = np.array(hit["position"])
                fn = np.array(hit["face_normal"])
                place_pos = (int(hp[0] + fn[0]), int(hp[1] + fn[1]),
                             int(hp[2] + fn[2]))
                if place_pos in self.remaining_targets:
                    hit_pos = place_pos

        reward = GAZE_TIME_PENALTY

        # Gaze success reward (per hit, not just on completion)
        if hit_pos is not None:
            reward += GAZE_REWARD_SUCCESS

        # Proximity reward/penalty based on agent distance to nearest target
        proximity_reward = 0.0
        agent_pos = (
            int(self.inner_env.agent_x),
            int(self.inner_env.agent_y),
            int(self.inner_env.agent_z),
        )
        min_dist = float("inf")
        for t in self.remaining_targets:
            d = abs(agent_pos[0] - t[0]) + abs(agent_pos[1] - t[1]) + abs(agent_pos[2] - t[2])
            min_dist = min(min_dist, d)

        if min_dist <= 5:
            proximity_reward = GAZE_PROXIMITY_REWARD_SCALE * (1.0 - min_dist / 5.0)
        else:
            proximity_reward = -0.01 * (min_dist - 5) ** 2
        reward += proximity_reward

        # Angular shaping: reward for reducing angle to nearest target
        angular_dist = self._angular_distance_to(nearest_target)

        # Reset shaping if nearest target changed
        if nearest_target != self.prev_nearest_target:
            self.prev_angular_dist = None
            self.prev_nearest_target = nearest_target

        if self.prev_angular_dist is not None and angular_dist is not None:
            improvement = self.prev_angular_dist - angular_dist
            shaping = np.clip(improvement * GAZE_REWARD_ANGULAR_SCALE, -0.05, 0.1)
            reward += shaping
        if angular_dist is not None:
            self.prev_angular_dist = angular_dist
            # Continuous penalty for looking away from target
            reward += -0.01 * angular_dist

        return reward, {
            "hit_pos": hit_pos,
            "angular_dist": angular_dist if angular_dist is not None else 0.0,
            "proximity_reward": proximity_reward,
        }

    def _angular_distance_to(self, target):
        """Compute angular distance from agent's gaze to target position."""
        eye = np.array([
            self.inner_env.agent_x,
            self.inner_env.agent_y + 1.6,
            self.inner_env.agent_z,
        ])
        block_center = np.array(target, dtype=np.float32) + 0.5
        diff = block_center - eye
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            return 0.0

        target_yaw = math.atan2(diff[0], diff[2])
        horiz_dist = math.sqrt(diff[0] ** 2 + diff[2] ** 2)
        target_pitch = math.atan2(-diff[1], horiz_dist) if horiz_dist > 0.01 else 0.0

        delta_yaw = target_yaw - self.inner_env.agent_yaw
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi
        delta_pitch = target_pitch - self.inner_env.agent_pitch

        return math.sqrt(delta_yaw ** 2 + delta_pitch ** 2)

    def _rebuild_observation(self):
        """Rebuild observation with target_direction pointing to nearest target."""
        obs = self.inner_env._build_observation()

        nearest = self._find_nearest_target()
        if nearest is not None:
            target_dir = self._target_direction_to(nearest)
            obs["flat_features"][-TARGET_DIRECTION_SIZE:] = target_dir

        return obs

    def _target_direction_to(self, target):
        """Compute target direction features for a specific target position."""
        result = np.zeros(TARGET_DIRECTION_SIZE, dtype=np.float32)

        eye = np.array([
            self.inner_env.agent_x,
            self.inner_env.agent_y + 1.6,
            self.inner_env.agent_z,
        ])
        block_center = np.array(target, dtype=np.float32) + 0.5
        diff = block_center - eye
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            return result

        target_yaw = math.atan2(diff[0], diff[2])
        delta_yaw = target_yaw - self.inner_env.agent_yaw
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi

        horiz_dist = math.sqrt(diff[0] ** 2 + diff[2] ** 2)
        target_pitch = math.atan2(-diff[1], horiz_dist) if horiz_dist > 0.01 else 0.0
        delta_pitch = target_pitch - self.inner_env.agent_pitch

        result[0] = max(-1.0, min(1.0, delta_yaw / math.pi))
        result[1] = max(-1.0, min(1.0, delta_pitch / (math.pi / 2)))
        result[2] = min(1.0, dist / WORLD_SIZE)
        return result

    def action_masks(self):
        """Same as inner env but force interact to noop only."""
        mask = self.inner_env.action_masks()
        interact_start = sum(ACTION_DIMS[:ACT_INTERACT])
        mask[interact_start] = False      # disable place
        mask[interact_start + 1] = True   # enable noop
        mask[interact_start + 2] = False  # disable attack
        return mask
