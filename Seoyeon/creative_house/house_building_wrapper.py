"""
house_building_wrapper.py

집 건축 강화학습 환경 래퍼.

참조: pseudo_village_flat_rl.py (yhs0602 CraftGround-Experiments)

확정된 API (참조 코드에서 직접 확인):
  from craftground import InitialEnvironmentConfig, make
  from craftground.initial_environment_config import WorldType
  from craftground.environment.action_space import no_op

  config = InitialEnvironmentConfig(
      image_width=114, image_height=64,
      seed=str(seed),                        ← str 필수
      world_type=WorldType.SUPERFLAT,
      render_distance=4,
      simulation_distance=4,
      hud_hidden=False,
      initial_extra_commands=[...],          ← 모든 환경 제어는 여기서
  )
  env = make(initial_env_config=config, port=port,
             verbose=False, verbose_gradle=True, render_action=False)

no_op() 인덱스 (참조 코드에서 확정):
  [0]  forward=1 / backward=2 / no=0
  [2]  strafe left=1
  [5]  strafe right=1
  [3]  pitch  : up=11 / center=12 / down=13
  [4]  yaw    : big-left=10 / small-left=11 / center=12 / small-right=13 / big-right=14

  ⚠️  아래 인덱스는 참조 코드에 없어 추론값입니다.
      debug_action.py 실행 후 no_op() 배열 길이와 각 인덱스 동작을 확인하세요.
  [1]  jump=1
  [6]  attack/break=1
  [7]  use/place=1
  [8]  sneak=1
  [9]  sprint=1
  [10] hotbar slot (1~9)
"""

from __future__ import annotations

import io
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op

from raycast_tracker import RaycastTracker
from mode_config import MODES, INITIAL_INVENTORY_CMDS, ModeConfig


# ── 행동 정의 ─────────────────────────────────────────────────────
# no_op()은 "아무것도 안 하는" 기본 action array를 반환합니다.
# build_action()에서 해당 인덱스만 수정해 각 행동을 표현합니다.

ACTION_NAMES: list[str] = [
    "NO_OP",         # 0  — 아무것도 안 함
    "FORWARD",       # 1  — 앞으로
    "BACKWARD",      # 2  — 뒤로
    "LEFT",          # 3  — 좌측 이동(strafe)
    "RIGHT",         # 4  — 우측 이동(strafe)
    "JUMP",          # 5  — 점프
    "SNEAK",         # 6  — 웅크리기 (블록 설치 시 필요)
    "ATTACK",        # 7  — 블록 파괴
    "USE",           # 8  — 블록 설치 ★ 핵심 행동
    "CAMERA_LEFT",   # 9  — 카메라 좌 (small)
    "CAMERA_RIGHT",  # 10 — 카메라 우 (small)
    "CAMERA_UP",     # 11 — 카메라 위
    "CAMERA_DOWN",   # 12 — 카메라 아래
    "HOTBAR_1",      # 13
    "HOTBAR_2",      # 14
    "HOTBAR_3",      # 15
    "HOTBAR_4",      # 16
    "HOTBAR_5",      # 17
    "HOTBAR_6",      # 18
    "HOTBAR_7",      # 19
    "HOTBAR_8",      # 20
    "HOTBAR_9",      # 21
]

USE_ACTION_IDX = ACTION_NAMES.index("USE")  # == 8


def build_action(name: str) -> list:
    """
    행동 이름 → no_op() 기반 raw action array.

    인덱스 출처:
      ✅ 확정 (pseudo_village_flat_rl.py 직접 확인)
         [0]이동  [2]strafe-left  [5]strafe-right  [3]pitch  [4]yaw
      ⚠️ 추론 (debug_action.py 로 검증 권장)
         [1]jump  [6]attack  [7]use  [8]sneak  [10]hotbar
    """
    act = no_op()

    # ── 이동 (✅ 확정) ─────────────────────────────────────────────
    if   name == "FORWARD":       act[0] = 1
    elif name == "BACKWARD":      act[0] = 2
    elif name == "LEFT":          act[2] = 1
    elif name == "RIGHT":         act[5] = 1

    # ── 카메라 (✅ 확정) ──────────────────────────────────────────
    elif name == "CAMERA_LEFT":   act[4] = 11   # small left
    elif name == "CAMERA_RIGHT":  act[4] = 13   # small right
    elif name == "CAMERA_UP":     act[3] = 11
    elif name == "CAMERA_DOWN":   act[3] = 13

    # ── 이하 추론값 — debug_action.py 실행 후 수정 ─────────────────
    elif name == "JUMP":          act[1] = 1
    elif name == "SNEAK":         act[8] = 1
    elif name == "ATTACK":        act[6] = 1
    elif name == "USE":           act[7] = 1

    elif name == "HOTBAR_1":      act[10] = 1
    elif name == "HOTBAR_2":      act[10] = 2
    elif name == "HOTBAR_3":      act[10] = 3
    elif name == "HOTBAR_4":      act[10] = 4
    elif name == "HOTBAR_5":      act[10] = 5
    elif name == "HOTBAR_6":      act[10] = 6
    elif name == "HOTBAR_7":      act[10] = 7
    elif name == "HOTBAR_8":      act[10] = 8
    elif name == "HOTBAR_9":      act[10] = 9

    # NO_OP: 수정 없이 no_op() 그대로 반환
    return act


# ── 보상 상수 ────────────────────────────────────────────────────
BLOCK_REWARDS: dict[str, float] = {
    "building":  0.3,
    "door":      2.0,
    "light":     1.5,
    "furniture": 2.5,
}
ALIVE_REWARD     =  0.01
ENCLOSED_BONUS   = 10.0
NIGHT_BONUS      =  5.0
HEALTH_PENALTY_K = -0.5
FOOD_PENALTY_K   = -0.1
DEATH_PENALTY    = -20.0
REWARD_CLIP      = (-10.0, 10.0)


def _extract_image(obs) -> np.ndarray | None:
    """
    raw obs에서 RGB 이미지를 추출합니다.
    참조 코드(process_rgb)와 동일한 방식으로 세 가지 포맷을 처리합니다.

      1. numpy array (직접 이미지)
      2. dict {"pov": ..., "rgb": ...}
      3. protobuf (obs.image bytes) — CraftGround 내부 proto 포맷
    """
    # Case 1: numpy array 직접
    if isinstance(obs, np.ndarray):
        img = obs

    # Case 2: dict — pov 또는 rgb 키
    elif isinstance(obs, dict):
        img = obs.get("pov") or obs.get("rgb")
        if img is None:
            return None
        img = np.array(img, dtype=np.uint8)

    # Case 3: protobuf — .image 필드 (bytes)
    elif hasattr(obs, "image") and isinstance(getattr(obs, "image", None), bytes):
        try:
            img = np.array(
                Image.open(io.BytesIO(obs.image)).convert("RGB"), dtype=np.uint8
            )
        except Exception:
            return None

    else:
        return None

    # CHW → HWC 변환 (필요한 경우)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    return img.astype(np.uint8)


def _get_scalar(obs, key: str, default: float) -> float:
    """
    raw obs에서 스칼라 값을 추출합니다.
    protobuf 속성(.health) 또는 dict 키("health") 모두 처리합니다.
    """
    if isinstance(obs, dict):
        return float(obs.get(key, default))
    return float(getattr(obs, key, default))


# ── 래퍼 ─────────────────────────────────────────────────────────
class HouseBuildingWrapper(gym.Wrapper):
    """
    Observation Dict:
        "image"     : (64, 114, 3) uint8   — 1인칭 POV
        "state"     : (8,)         float32  — health, food, xyz, sin/cos(yaw), pitch
        "raycast"   : (8,)         float32  — 현재 바라보는 블록 정보
        "structure" : (6,)         float32  — 누적 구조물 요약

    Action:
        Discrete(22) → build_action() → no_op() 기반 raw array → 환경에 전달
    """

    IMG_H = 64
    IMG_W = 114

    def __init__(self, env: gym.Env, cfg: ModeConfig, max_episode_steps: int):
        super().__init__(env)
        self.cfg               = cfg
        self.max_episode_steps = max_episode_steps
        self.tracker           = RaycastTracker()

        # 행동/관찰 공간 재정의
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(self.IMG_H, self.IMG_W, 3), dtype=np.uint8,
            ),
            "state": spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype=np.float32,
            ),
            "raycast": spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype=np.float32,
            ),
            "structure": spaces.Box(
                low=0.0, high=1.0, shape=(6,), dtype=np.float32,
            ),
        })
        self._reset_ep_state()

    # ── 에피소드 상태 초기화 ──────────────────────────────────────
    def _reset_ep_state(self):
        self._step                 = 0
        self._ep_reward            = 0.0
        self._prev_health          = 20.0
        self._prev_food            = 20.0
        self._enclosed_bonus_given = False
        self._night_started        = False

    # ── reset ────────────────────────────────────────────────────
    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        spawn_y = _get_scalar(raw_obs, "y", 4.0)
        self.tracker.reset(spawn_y=spawn_y)
        self._reset_ep_state()
        return self._process_obs(raw_obs), info

    # ── step ─────────────────────────────────────────────────────
    def step(self, action: int):
        action_name    = ACTION_NAMES[int(action)]
        action_was_use = (action_name == "USE")
        action_arr     = build_action(action_name)

        raw_obs, _, terminated, truncated, info = self.env.step(action_arr)
        self._step += 1

        if self._step >= self.max_episode_steps:
            truncated = True

        done   = terminated or truncated
        obs    = self._process_obs(raw_obs)
        reward, reward_info = self._compute_reward(raw_obs, done, action_was_use)

        self._ep_reward += reward
        info.update({
            "episode_step":     self._step,
            "episode_reward":   self._ep_reward,
            "reward_breakdown": reward_info,
            "structure":        self.tracker.analyze_structure(),
            "milestones":       dict(self.tracker.milestones),
            "action_name":      action_name,
        })
        return obs, reward, terminated, truncated, info

    # ── 관찰값 전처리 ─────────────────────────────────────────────
    def _process_obs(self, raw_obs) -> dict:
        # 이미지 (numpy / dict / protobuf 모두 처리)
        img_arr = _extract_image(raw_obs)
        if img_arr is not None:
            img_arr = np.array(
                Image.fromarray(img_arr).resize((self.IMG_W, self.IMG_H), Image.BILINEAR),
                dtype=np.uint8,
            )
        else:
            img_arr = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)

        # 상태 벡터
        health = _get_scalar(raw_obs, "health",     20.0) / 20.0
        food   = _get_scalar(raw_obs, "food_level", 20.0) / 20.0
        x      = _get_scalar(raw_obs, "x",           0.0) / 256.0
        y      = _get_scalar(raw_obs, "y",           0.0) / 256.0
        z      = _get_scalar(raw_obs, "z",           0.0) / 256.0
        yaw    = _get_scalar(raw_obs, "yaw",         0.0)
        pitch  = _get_scalar(raw_obs, "pitch",       0.0)

        state = np.array([
            health, food, x, y, z,
            np.sin(np.radians(yaw)),
            np.cos(np.radians(yaw)),
            pitch / 90.0,
        ], dtype=np.float32)

        return {
            "image":     img_arr,
            "state":     state,
            "raycast":   self.tracker.encode_current(raw_obs),
            "structure": self.tracker.get_obs_summary(),
        }

    # ── 보상 계산 ─────────────────────────────────────────────────
    def _compute_reward(
        self, raw_obs, done: bool, action_was_use: bool
    ) -> tuple[float, dict]:
        reward = 0.0
        info:  dict = {}

        # 생존 보상 (매 스텝)
        if not done:
            reward += ALIVE_REWARD
            info["alive"] = ALIVE_REWARD

        health = _get_scalar(raw_obs, "health",     20.0)
        food   = _get_scalar(raw_obs, "food_level", 20.0)

        # 체력·허기 패널티 (survival만)
        if self.cfg.use_health_penalty and health < self._prev_health:
            p = HEALTH_PENALTY_K * (self._prev_health - health)
            reward += p
            info["health_penalty"] = p

        if self.cfg.use_food_penalty and food < self._prev_food:
            p = FOOD_PENALTY_K * (self._prev_food - food)
            reward += p
            info["food_penalty"] = p

        self._prev_health = health
        self._prev_food   = food

        # 사망 패널티 (survival만)
        is_dead = bool(_get_scalar(raw_obs, "is_dead", 0.0))
        if self.cfg.use_death_penalty and done and is_dead:
            reward += DEATH_PENALTY
            info["death"] = DEATH_PENALTY

        # 블록 설치 보상 (raycast 기반 감지)
        placed = self.tracker.update(raw_obs, action_was_use)
        if placed["newly_placed"]:
            r = BLOCK_REWARDS.get(placed["block_type"], 0.2)
            reward += r
            info[f"placed_{placed['block_type']}"] = r

        # 마일스톤 보너스 (일회성)
        m_bonus, m_info = self.tracker.compute_milestone_rewards()
        if m_bonus > 0:
            reward += m_bonus
            info.update(m_info)

        # 밀폐 완성 보너스 (일회성)
        if not self._enclosed_bonus_given and self.tracker.analyze_structure()["enclosed"]:
            reward += ENCLOSED_BONUS
            info["enclosed_bonus"] = ENCLOSED_BONUS
            self._enclosed_bonus_given = True

        # 밤 생존 보너스 (survival만, 반복)
        if self.cfg.use_night_bonus:
            wtime    = int(_get_scalar(raw_obs, "world_time", 0.0)) % 24000
            is_night = 13000 <= wtime <= 23000
            if is_night:
                self._night_started = True
            elif self._night_started and not is_night:
                reward += NIGHT_BONUS
                info["night_bonus"] = NIGHT_BONUS
                self._night_started = False

        reward = float(np.clip(reward, *REWARD_CLIP))
        info["total"] = reward
        return reward, info


# ── 팩토리 함수 ──────────────────────────────────────────────────
def make_house_env(
    port:              int        = 8030,
    mode:              str        = "creative",
    max_episode_steps: int | None = None,
    seed:              int        = 42,
    render_action:     bool       = False,
) -> HouseBuildingWrapper:
    """
    집 건축 환경을 생성합니다.

    참조 코드(pseudo_village_flat_rl.py)와 동일한 API를 사용합니다:
      InitialEnvironmentConfig + make()

    Args:
        port              : CraftGround 서버 포트
        mode              : "creative" | "safe" | "survival"
        max_episode_steps : None이면 모드 기본값(cfg.max_episode_steps) 사용
        seed              : 월드 시드 (int → str 변환됨)
        render_action     : Minecraft 창에 행동 오버레이 표시 여부
    """
    if mode not in MODES:
        raise ValueError(
            f"mode='{mode}' 는 유효하지 않습니다. "
            f"선택 가능: {list(MODES.keys())}"
        )

    cfg   = MODES[mode]
    steps = max_episode_steps if max_episode_steps is not None else cfg.max_episode_steps

    # 인벤토리 give 커맨드를 initial_extra_commands 앞에 추가
    inv_cmds   = list(INITIAL_INVENTORY_CMDS) if cfg.give_inventory else []
    extra_cmds = inv_cmds + list(cfg.initial_extra_commands)

    config = InitialEnvironmentConfig(
        image_width            = 114,
        image_height           = 64,
        seed                   = str(seed),         # ← str 필수 (참조 코드 확인)
        world_type             = WorldType.SUPERFLAT,    # ← Superflat 평지
        render_distance        = 4,
        simulation_distance    = 4,
        hud_hidden             = False,             # ← 체력/허기바 표시
        initial_extra_commands = extra_cmds,
    )

    base_env = make(
        initial_env_config = config,
        port               = port,
        verbose            = False,
        verbose_gradle     = True,
        render_action      = render_action,
    )

    return HouseBuildingWrapper(base_env, cfg=cfg, max_episode_steps=steps)
