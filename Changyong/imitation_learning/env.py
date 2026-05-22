"""
VectorBuildingEnv — 팀원 설계와 동일한 41차원 벡터 obs.

HierarchicalBuildingEnv를 상속해서 obs만 바꾸고
보상/Manager 로직은 그대로 유지합니다.

obs 구성 (총 45차원):
  origin_set      (1)  : 항상 1.0
  agent_pos       (3)  : 원점 기준 상대 좌표 (정규화)
  agent_orient    (4)  : sin/cos(yaw), sin/cos(pitch)
  hotbar_info     (3)  : slot/8, 판자 보유, 도끼 보유
  raycast         (17) : 히트 정보, 면 법선, 배치 예측 위치 등
  progress        (3)  : 바닥/벽/천장 완성 비율 (3x3 작업: 바닥만)
  incorrect_count (1)  : 잘못 배치된 블록 수 (미지원 → 0)
  time_remaining  (1)  : 남은 시간 비율
  stuck_ratio     (1)  : subgoal 타임아웃 비율
  target_direction(4)  : 목표까지 delta_yaw, delta_pitch sin/cos
  target_distance (1)  : 목표까지 거리 (정규화)
  target_pos      (3)  : 현재 subgoal 절대 좌표 (정규화)
  house_size      (3)  : 3x1x3 (3x3 바닥 고정)
"""
import math
import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from building_hrl import (
    HierarchicalBuildingEnv, build_action, ACTIONS,
    TARGET_POSITIONS, TOTAL_BLOCKS, MAX_STEPS, MAX_SUBGOAL_STEP,
)

# 빌드 원점 & 정규화 스케일
ORIGIN    = (0.0, 65.0, 0.0)
POS_SCALE = 10.0

# 면 → 법선 벡터
FACE_NORMALS = {
    "UP":    ( 0,  1,  0),
    "DOWN":  ( 0, -1,  0),
    "NORTH": ( 0,  0, -1),
    "SOUTH": ( 0,  0,  1),
    "EAST":  ( 1,  0,  0),
    "WEST":  (-1,  0,  0),
}


class VectorBuildingEnv(HierarchicalBuildingEnv):
    """Image obs → 45차원 벡터 obs로 교체한 환경."""

    OBS_DIM = 45

    # y=64 기반 플랫폼을 유지하고 y=65만 비움
    _RESET_CMDS = [
        "fill -10 64 -10 10 64 15 minecraft:stone",  # 기준면 생성
        "fill -10 65 -10 10 65 15 minecraft:air",    # 이전 에피소드 블록 제거
        "tp @p 0 66 0 0 50",
        "clear @p",
        "give @p minecraft:oak_planks 64",
    ]

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.player_pitch       = 0.0
        self.raycast_can_place  = False   # 현재 시선이 미완성 타겟 위에 배치 가능한지
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.OBS_DIM,), dtype=np.float32
        )

    # ── pitch도 같이 저장 ─────────────────────────────────────────
    def _update_player_state(self, obs_dict: dict):
        super()._update_player_state(obs_dict)
        try:
            full = obs_dict.get("full")
            if full is not None:
                self.player_pitch = float(full.pitch)
        except Exception:
            pass

    # ── raycast 조준 확인 ─────────────────────────────────────────
    def _get_raycast_hit_block(self, obs_dict: dict):
        """
        실제 raycast hit 블록의 (bx, by, bz)를 반환.
        Jinwoo v3 참고: math.floor()로 Minecraft 블록 좌표 정확히 계산.
        """
        try:
            full = obs_dict.get("full")
            if full is None:
                return None
            hit = full.raycast_result
            if hit.type != 1:
                return None
            target = hit.target_block
            bx = int(math.floor(float(target.x)))
            by = int(math.floor(float(target.y)))
            bz = int(math.floor(float(target.z)))
            return (bx, by, bz)
        except Exception:
            return None

    def _check_raycast_target(self, obs_dict: dict) -> bool:
        """
        현재 subgoal 아래 돌 블록(y=64)을 정확히 조준하고 있는지 확인.
        Jinwoo v3 참고: 아무 바닥이 아니라 정확한 타겟 블록 검증.
        """
        sg = self.current_subgoal
        if sg is None:
            return False
        hit = self._get_raycast_hit_block(obs_dict)
        if hit is None:
            return False
        hx, hy, hz = hit
        gx, gy, gz = sg
        # subgoal y=65, 그 아래 stone y=64를 UP 면으로 바라볼 때 여기 배치됨
        return hx == gx and hy == gy - 1 and hz == gz

    # ── raycast 17차원 ────────────────────────────────────────────
    def _get_raycast_vec(self, obs_dict: dict) -> np.ndarray:
        vec = np.zeros(17, dtype=np.float32)
        try:
            full = obs_dict.get("full")
            if full is None:
                return vec
            hit = full.raycast_result
            if hit.type != 1:          # MISS
                return vec

            vec[0] = 1.0               # is_hit

            ox, oy, oz = ORIGIN
            # 히트 블록 위치: target_block.x/y/z + math.floor() (Jinwoo v3 참고)
            target = hit.target_block
            bx = int(math.floor(float(target.x)))
            by = int(math.floor(float(target.y)))
            bz = int(math.floor(float(target.z)))

            vec[1] = (bx - ox) / POS_SCALE     # hit_x
            vec[2] = (by - oy) / POS_SCALE     # hit_y
            vec[3] = (bz - oz) / POS_SCALE     # hit_z

            # 면 법선 (face 정보는 최선 노력)
            try:
                face = str(hit.target_block_face).upper()
            except AttributeError:
                face = "UP"
            nx, ny, nz = FACE_NORMALS.get(face, (0, 1, 0))
            vec[4], vec[5], vec[6] = nx, ny, nz

            # 타겟/배치 여부 (floor() 좌표 사용)
            remaining_set = set(self.remaining_targets)
            vec[7] = 1.0 if (bx, by, bz) in remaining_set else 0.0
            vec[8] = 1.0 if (bx, by, bz) in set(TARGET_POSITIONS) and \
                             (bx, by, bz) not in remaining_set else 0.0

            # 거리
            px, py, pz = self.player_pos
            vec[9] = min(math.sqrt((bx-px)**2+(by-py)**2+(bz-pz)**2) / POS_SCALE, 1.0)

            # 위 면 여부
            vec[10] = 1.0 if face == "UP" else 0.0

            # 배치 예정 위치 (hit 블록 위 면 → 새 블록이 놓일 곳)
            px2, py2, pz2 = bx + nx, by + ny, bz + nz
            vec[11] = (px2 - ox) / POS_SCALE
            vec[12] = (py2 - oy) / POS_SCALE
            vec[13] = (pz2 - oz) / POS_SCALE
            ipx2, ipy2, ipz2 = int(px2), int(py2), int(pz2)
            vec[14] = 1.0 if (ipx2, ipy2, ipz2) in remaining_set else 0.0
            vec[15] = vec[14]

            # 히트 블록 종류
            try:
                key = hit.target_block.translation_key.lower()
                vec[16] = 1.0 if "oak_planks" in key else 0.0
            except Exception:
                pass

        except Exception:
            pass
        return vec

    # ── 41차원 벡터 조립 ─────────────────────────────────────────
    def _build_obs(self, raw_obs: dict) -> np.ndarray:
        ox, oy, oz = ORIGIN
        px, py, pz = self.player_pos
        yaw_r   = math.radians(self.player_yaw)
        pitch_r = math.radians(self.player_pitch)

        sg = self.current_subgoal
        if sg is not None:
            gx, gy, gz = sg
            dx, dz   = gx - px, gz - pz
            dist_xz  = math.sqrt(dx**2 + dz**2)
            bearing  = math.degrees(math.atan2(-dx, dz))
            dyaw     = (bearing - self.player_yaw + 180) % 360 - 180
            dpitch   = 50.0 - self.player_pitch    # 바닥 조준 기준 (50도, 양수=아래)
            t_pos    = [(gx-ox)/POS_SCALE, (gy-oy)/POS_SCALE, (gz-oz)/POS_SCALE]
        else:
            dist_xz = dyaw = dpitch = 0.0
            t_pos   = [0.0, 0.0, 0.0]

        dyaw_r   = math.radians(dyaw)
        dpitch_r = math.radians(dpitch)

        has_planks = 1.0 if self.prev_inv > 0 else 0.0

        parts = [
            [1.0],                                                          # origin_set      (1)
            [(px-ox)/POS_SCALE, (py-oy)/POS_SCALE, (pz-oz)/POS_SCALE],    # agent_pos       (3)
            [math.sin(yaw_r), math.cos(yaw_r),
             math.sin(pitch_r), math.cos(pitch_r)],                        # agent_orient    (4)
            [0.0, has_planks, 0.0],                                        # hotbar_info     (3)
            self._get_raycast_vec(raw_obs),                                 # raycast        (17)
            [self.placed_count / TOTAL_BLOCKS, 0.0, 0.0],                  # progress        (3)
            [0.0],                                                          # incorrect_count (1)
            [(MAX_STEPS - self.total_step) / MAX_STEPS],                   # time_remaining  (1)
            [self.subgoal_step / MAX_SUBGOAL_STEP],                        # stuck_ratio     (1)
            [math.sin(dyaw_r), math.cos(dyaw_r),
             math.sin(dpitch_r), math.cos(dpitch_r)],                      # target_direction(4)
            [dist_xz / POS_SCALE],                                         # target_distance (1)
            t_pos,                                                          # target_pos      (3)
            [3/POS_SCALE, 1/POS_SCALE, 3/POS_SCALE],                      # house_size      (3)
        ]
        obs = np.concatenate([np.array(p, dtype=np.float32) for p in parts])
        assert len(obs) == self.OBS_DIM, f"obs dim mismatch: {len(obs)}"
        return obs

    # ── reset ────────────────────────────────────────────────────
    def reset(self, **kwargs):
        options = kwargs.pop("options", {})
        options["extra_commands"] = self._RESET_CMDS
        raw_obs, _ = self.env.reset(options=options, **kwargs)

        self.remaining_targets  = list(TARGET_POSITIONS)
        self.player_pos         = (0.0, 66.0, 0.0)
        self.player_yaw         = 0.0
        self.player_pitch       = 0.0
        self.raycast_can_place  = False
        self.current_subgoal    = self._select_nearest_subgoal()
        self.subgoal_step       = 0
        self.placed_count       = 0
        self.total_step         = 0

        self._update_player_state(raw_obs)
        self.prev_inv  = self._get_plank_count(raw_obs)
        self.prev_dist = self._dist_to_subgoal()
        self.last_obs  = self._process_image(raw_obs)  # 렌더용 유지

        return self._build_obs(raw_obs), {}

    # ── step ─────────────────────────────────────────────────────
    def step(self, action: int):
        action_arr = build_action(ACTIONS[int(action)])
        raw_obs, _, terminated, truncated, _ = self.env.step(action_arr)

        self._update_player_state(raw_obs)
        self.raycast_can_place = self._check_raycast_target(raw_obs)
        current_inv   = self._get_plank_count(raw_obs)
        newly_placed  = max(0, self.prev_inv - current_inv)
        self.prev_inv     = current_inv
        self.placed_count += newly_placed
        self.total_step   += 1
        self.subgoal_step += 1

        # 보상 (building_hrl.py와 동일)
        reward = -0.01
        curr_dist  = self._dist_to_subgoal()
        dist_delta = self.prev_dist - curr_dist
        if dist_delta > 0:
            reward += dist_delta * 0.5
        self.prev_dist = curr_dist

        if self._is_looking_at_floor(raw_obs):
            reward += 0.02

        if newly_placed > 0:
            reward += 5.0
            if self.current_subgoal in self.remaining_targets:
                self.remaining_targets.remove(self.current_subgoal)
            self.current_subgoal = self._select_nearest_subgoal()
            self.subgoal_step    = 0
            self.prev_dist       = self._dist_to_subgoal()
            print(f"  [PLACED] {self.placed_count}/{TOTAL_BLOCKS}  next: {self.current_subgoal}")
        elif self.subgoal_step >= MAX_SUBGOAL_STEP:
            reward -= 0.5
            self.current_subgoal = self._select_nearest_subgoal()
            self.subgoal_step    = 0
            self.prev_dist       = self._dist_to_subgoal()

        if self.placed_count >= TOTAL_BLOCKS:
            terminated = True
            reward += 10.0
            print(f"  [SUCCESS] 완료! (step {self.total_step})")

        if self._is_dead(raw_obs):
            terminated = True
            reward -= 2.0

        if self.total_step >= MAX_STEPS:
            truncated = True

        self.last_obs = self._process_image(raw_obs)

        return (
            self._build_obs(raw_obs),
            reward,
            terminated,
            truncated,
            {"placed": self.placed_count, "remaining": len(self.remaining_targets),
             "subgoal": self.current_subgoal, "step": self.total_step},
        )
