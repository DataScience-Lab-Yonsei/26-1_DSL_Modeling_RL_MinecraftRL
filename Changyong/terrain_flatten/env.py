"""
FlattenEnv — 12x12 지형 평탄화 환경.

태스크: TARGET_Y 초과 블록을 모두 채굴하여 평탄화.

관측 벡터 (18차원):
  agent_pos   (3): 구역 원점 기준 상대 좌표 (정규화)
  agent_orient(4): sin/cos yaw, sin/cos pitch
  raycast     (9): hit(1), hit_pos_rel(3), above_target(1),
                   distance(1), face_up(1), face_forward(1), is_solid(1)
  progress    (1): 제거된 블록 / 전체 목표 블록 비율
  time        (1): 남은 시간 비율

액션 스페이스: MultiDiscrete([3,3,2,2,3,9,9]) — Jinwoo v3 방식
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    AREA_X, AREA_Z, AREA_W, AREA_D, TARGET_Y, TERRAIN_MAX_Y,
    CAMERA_DELTA_MAP, CAMERA_NEUTRAL,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_PITCH, ACT_YAW,
    ACT_DIMS, NUM_ACT_DIMS, OBS_DIM,
    RAYCAST_MAX_DIST, MAX_STEPS,
)

# 구역 원점 & 정규화
POS_ORIGIN = (AREA_X + AREA_W / 2, float(TARGET_Y + 1), AREA_Z + AREA_D / 2)
POS_SCALE  = max(AREA_W, AREA_D) / 2.0   # ≈ 6.0

# 에피소드마다 생성할 지형 fill 명령 (fixed terrain pattern)
# base(TARGET_Y) + 3가지 언덕 → 항상 동일한 초기 상태
def _build_reset_cmds():
    ax, az = AREA_X, AREA_Z
    aw, ad = AREA_W, AREA_D
    ty = TARGET_Y
    cmds = [
        # 구역 전체 비우기 (TARGET_Y+1 ~ TERRAIN_MAX_Y+5)
        f"fill {ax} {ty+1} {az-4} {ax+aw-1} {TERRAIN_MAX_Y+5} {az+ad-1} minecraft:air",
        # 바닥 평탄화 + 접근로 4블록 남쪽까지 (플레이어 이동 경로 보장)
        f"fill {ax} {ty} {az-4} {ax+aw-1} {ty} {az+ad-1} minecraft:stone",
        # 언덕 1: 서쪽 절반 높이 3
        f"fill {ax} {ty+1} {az} {ax+5} {ty+3} {az+5} minecraft:stone",
        # 언덕 2: 동쪽 뒤편 높이 5
        f"fill {ax+6} {ty+1} {az+6} {ax+aw-1} {ty+5} {az+ad-1} minecraft:stone",
        # 중앙 낮은 언덕 높이 2
        f"fill {ax+3} {ty+1} {az+3} {ax+8} {ty+2} {az+8} minecraft:stone",
        # 플레이어 리셋: 구역 앞 접근로(stone 위), pitch=0 수평
        f"tp @p {ax + aw//2 + 0.5} {ty+1} {az-3} 0 0",
        "gamemode creative @p",
        "give @p minecraft:diamond_shovel 1",
        "enchant @p efficiency 5",
        "time set day",
        "weather clear",
        "gamerule doDaylightCycle false",
        "gamerule doMobSpawning false",
        "difficulty peaceful",
    ]
    return cmds

_RESET_CMDS = _build_reset_cmds()

# 초기 지형에서 제거해야 하는 총 블록 수 (fill 명령 기준)
def _count_initial_blocks():
    count = 0
    # 언덕 1: x[0..5] z[0..5] h[1..3]
    count += 6 * 6 * 3
    # 언덕 2: x[6..11] z[6..11] h[1..5]
    count += 6 * 6 * 5
    # 중앙: x[3..8] z[3..8] h[1..2]
    count += 6 * 6 * 2
    return count

TOTAL_TARGET_BLOCKS = _count_initial_blocks()  # 초기 제거 대상 블록 수


def build_cg_action(action: np.ndarray) -> dict:
    """MultiDiscrete 액션 → craftground V2 dict 변환."""
    try:
        from craftground.environment.action_space import no_op_v2
    except ImportError:
        from craftground import no_op_v2

    a = no_op_v2()
    a["forward"] = 1 if int(action[ACT_FWD_BACK])   == 2 else 0
    a["back"]    = 1 if int(action[ACT_FWD_BACK])   == 0 else 0
    a["left"]    = 1 if int(action[ACT_LEFT_RIGHT]) == 0 else 0
    a["right"]   = 1 if int(action[ACT_LEFT_RIGHT]) == 2 else 0
    a["jump"]    = int(action[ACT_JUMP])
    a["sneak"]   = int(action[ACT_SNEAK])
    interact     = int(action[ACT_INTERACT])
    a["use"]     = 1 if interact == 0 else 0
    a["attack"]  = 1 if interact == 2 else 0
    pitch_idx    = int(action[ACT_PITCH])
    yaw_idx      = int(action[ACT_YAW])
    a["camera_pitch"] = CAMERA_DELTA_MAP[pitch_idx]
    a["camera_yaw"]   = CAMERA_DELTA_MAP[yaw_idx]
    return a


class FlattenEnv(gym.Env):
    """
    12x12 지형 평탄화 환경.

    craftground_env: craftground.make()로 생성한 기본 환경.
    """

    def __init__(self, craftground_env):
        super().__init__()
        self.cg_env = craftground_env
        self._cg_obs = None   # 마지막 원시 관측 (expert가 직접 접근 가능)

        self.action_space = spaces.MultiDiscrete(ACT_DIMS)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (OBS_DIM,), dtype=np.float32
        )

        # 에이전트 상태
        self.agent_x    = 0.0
        self.agent_y    = float(TARGET_Y + 1)
        self.agent_z    = 0.0
        self.agent_yaw  = 0.0   # radians, our_yaw = -mc_yaw
        self.agent_pitch = 0.0  # radians, positive = down

        # 에피소드 상태
        self.blocks_cleared = 0
        self.total_step     = 0
        self._prev_hit      = None   # (bx, by, bz) 직전 raycast
        self._prev_attacked = False

    # ── 에이전트 상태 갱신 ────────────────────────────────────────
    def _update_state(self, obs):
        try:
            proto = obs["full"] if isinstance(obs, dict) else obs
            self.agent_x     = float(proto.x)
            self.agent_y     = float(proto.y)
            self.agent_z     = float(proto.z)
            self.agent_yaw   = -math.radians(float(proto.yaw))   # our_yaw = -mc_yaw
            self.agent_pitch =  math.radians(float(proto.pitch))  # positive = down
        except Exception:
            pass

    # ── Raycast 파싱 ─────────────────────────────────────────────
    def _parse_raycast(self, obs):
        """
        Returns dict with keys: position(bx,by,bz), distance, above_target,
        face_normal(nx,ny,nz), block_name.
        None if miss.
        """
        try:
            proto = obs["full"] if isinstance(obs, dict) else obs
            hit = proto.raycast_result
            if hit.type != 1:
                return None
            target = hit.target_block
            bx = int(math.floor(float(target.x)))
            by = int(math.floor(float(target.y)))
            bz = int(math.floor(float(target.z)))

            # 눈 위치
            eye_x = self.agent_x
            eye_y = self.agent_y + 1.62
            eye_z = self.agent_z
            dist  = math.sqrt((bx+0.5-eye_x)**2+(by+0.5-eye_y)**2+(bz+0.5-eye_z)**2)
            if dist > RAYCAST_MAX_DIST:
                return None

            # 면 법선 계산 (yaw/pitch에서)
            yaw_deg   = float(proto.yaw)
            pitch_deg = float(proto.pitch)
            yaw_rad   = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)
            cos_p = math.cos(pitch_rad)
            dx =  -math.sin(yaw_rad) * cos_p
            dy =  -math.sin(pitch_rad)
            dz =   math.cos(yaw_rad) * cos_p

            def entry_t(origin, direction, block_min):
                if direction == 0: return float('-inf')
                return (block_min - origin) / direction if direction > 0 \
                       else (block_min + 1.0 - origin) / direction

            tx = entry_t(eye_x, dx, bx)
            ty = entry_t(eye_y, dy, by)
            tz = entry_t(eye_z, dz, bz)
            mt = max(tx, ty, tz)
            if mt == tx:
                nx, ny, nz = (-1,0,0) if dx>0 else (1,0,0)
            elif mt == ty:
                nx, ny, nz = (0,-1,0) if dy>0 else (0,1,0)
            else:
                nx, ny, nz = (0,0,-1) if dz>0 else (0,0,1)

            block_name = ""
            try:
                block_name = str(target.translation_key).lower()
            except Exception:
                pass

            return {
                "position": (bx, by, bz),
                "distance": dist,
                "above_target": by > TARGET_Y,
                "in_area": (AREA_X <= bx < AREA_X+AREA_W
                             and AREA_Z <= bz < AREA_Z+AREA_D),
                "face_normal": (nx, ny, nz),
                "block_name": block_name,
            }
        except Exception:
            return None

    # ── 관측 벡터 조립 ─────────────────────────────────────────
    def _build_obs(self, raw_obs) -> np.ndarray:
        ox, oy, oz = POS_ORIGIN
        ax, ay, az = self.agent_x, self.agent_y, self.agent_z

        yaw_r   = self.agent_yaw
        pitch_r = self.agent_pitch

        # Raycast 9차원
        ray_vec = np.zeros(9, dtype=np.float32)
        hit = self._parse_raycast(raw_obs)
        if hit is not None:
            bx, by, bz = hit["position"]
            nx, ny, nz = hit["face_normal"]
            ray_vec[0] = 1.0                               # is_hit
            ray_vec[1] = (bx - ox) / POS_SCALE            # hit_x_rel
            ray_vec[2] = (by - oy) / POS_SCALE            # hit_y_rel
            ray_vec[3] = (bz - oz) / POS_SCALE            # hit_z_rel
            ray_vec[4] = 1.0 if hit["above_target"] else 0.0
            ray_vec[5] = hit["distance"] / RAYCAST_MAX_DIST
            ray_vec[6] = 1.0 if ny > 0 else 0.0           # face_up
            ray_vec[7] = 1.0 if nz > 0 else 0.0           # face toward agent (+Z face)
            ray_vec[8] = 0.0 if "air" in hit["block_name"] else 1.0

        progress = min(self.blocks_cleared / max(TOTAL_TARGET_BLOCKS, 1), 1.0)
        time_rem = max(0.0, (MAX_STEPS - self.total_step) / MAX_STEPS)

        parts = [
            [(ax-ox)/POS_SCALE, (ay-oy)/POS_SCALE, (az-oz)/POS_SCALE],
            [math.sin(yaw_r), math.cos(yaw_r),
             math.sin(pitch_r), math.cos(pitch_r)],
            ray_vec.tolist(),
            [progress],
            [time_rem],
        ]
        obs = np.concatenate([np.array(p, dtype=np.float32) for p in parts])
        assert len(obs) == OBS_DIM, f"obs dim {len(obs)} != {OBS_DIM}"
        return obs

    # ── reset ────────────────────────────────────────────────────
    def reset(self, **kwargs):
        options = kwargs.pop("options", {})
        options["extra_commands"] = _RESET_CMDS
        raw_obs, _ = self.cg_env.reset(options=options, **kwargs)

        self._cg_obs     = raw_obs
        self.blocks_cleared = 0
        self.total_step     = 0
        self._prev_hit      = None
        self._prev_attacked = False

        self._update_state(raw_obs)
        return self._build_obs(raw_obs), {}

    # ── step ─────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        cg_action = build_cg_action(action)
        raw_obs, _, terminated, truncated, _ = self.cg_env.step(cg_action)

        self._cg_obs = raw_obs
        self._update_state(raw_obs)
        self.total_step += 1

        # 현재 raycast
        hit = self._parse_raycast(raw_obs)

        # 블록 제거 감지:
        # 이전 스텝에 공격했고 + 이전 hit 블록이 이제 안 보임 → 제거됨
        newly_cleared = 0
        if self._prev_attacked and self._prev_hit is not None:
            ph = self._prev_hit
            cur_pos = hit["position"] if hit else None
            if cur_pos != ph:
                # 이전에 바라보던 블록이 사라진 것으로 간주
                if ph[1] > TARGET_Y and \
                   AREA_X <= ph[0] < AREA_X+AREA_W and \
                   AREA_Z <= ph[2] < AREA_Z+AREA_D:
                    newly_cleared = 1
                    self.blocks_cleared += 1

        # 보상
        reward = -0.005  # 스텝 패널티
        if newly_cleared > 0:
            reward += 1.0
        if hit is not None and hit["above_target"] and hit["in_area"]:
            reward += 0.05  # 제거 대상 블록 조준 보상

        was_attacking = int(action[ACT_INTERACT]) == 2
        self._prev_attacked = was_attacking
        self._prev_hit = hit["position"] if hit else None

        # 종료 조건
        if self.blocks_cleared >= TOTAL_TARGET_BLOCKS:
            terminated = True
            reward += 50.0
            print(f"  [SUCCESS] 평탄화 완료! step={self.total_step}")
        if self.total_step >= MAX_STEPS:
            truncated = True

        info = {
            "blocks_cleared": self.blocks_cleared,
            "total_target":   TOTAL_TARGET_BLOCKS,
            "step": self.total_step,
        }
        return self._build_obs(raw_obs), reward, terminated, truncated, info

    def close(self):
        self.cg_env.close()
