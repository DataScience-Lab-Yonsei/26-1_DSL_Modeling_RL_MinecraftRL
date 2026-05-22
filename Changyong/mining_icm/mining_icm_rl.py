"""
mining_icm_rl.py — ICM 기반 Minecraft 지하 광물 채굴 강화학습

Seoyeon/mining_rl_pack/real_mining_v1_0318.py 와 비교:

  [Seoyeon - PPO + 수동 보상]          [이 파일 - PPO + ICM]
  ─────────────────────────────────    ────────────────────────────────
  Layer 1  Y-level shaping (유지)  →   Layer 1  Y-level shaping (유지)
  Layer 2  visited cell + λ 감쇠  →   ICM      전이 예측 오류 = 내재 보상 (자동)
  Layer 3  광물 인벤토리 delta    →   Layer 2  광물 인벤토리 delta (유지)
  수동 탐험 셀 해시 집합           →   없음 (ICM이 novelty 자동 측정)
  lambda_intrinsic 감쇠 스케줄    →   없음 (eta 고정, 네트워크가 스스로 수렴)

핵심 아이디어:
  ICM = PhiEncoder + ForwardModel + InverseModel
  r_int = eta * ||ForwardModel(phi(s), a) - phi(s')||^2
  → 모델이 예측하지 못하는 새 상태 전이일수록 보상이 크다
  → 에이전트는 자연스럽게 미지의 공간을 탐색하게 됨

실행:
  # 훈련
  python mining_icm_rl.py --mode train --db cave_db.json --total_steps 3000000

  # 평가 (Seoyeon 체크포인트도 로드 가능 — 환경 동일)
  python mining_icm_rl.py --mode eval --resume checkpoints/mining_icm_XXXX/best/best_model
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op_v2, ActionSpaceVersion

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from cave_seed_scanner_v1 import CaveSpawnWrapper
from icm import ICMModule, ICMReplayBuffer

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def _wlog(d: dict, step=None):
    if _WANDB and wandb.run:
        wandb.log(d, step=step)

def _wactive() -> bool:
    return _WANDB and bool(wandb.run)


# =============================================================
# 1. 모드 설정 (Seoyeon과 동일)
# =============================================================

@dataclass(frozen=True)
class ModeConfig:
    initial_extra_commands: tuple[str, ...]
    max_episode_steps: int

_COMMON = (
    "gamerule doWeatherCycle false",
    "gamerule doImmediateRespawn true",
    "gamerule naturalRegeneration true",
    "weather clear",
)

MODES: dict[str, ModeConfig] = {
    "safe": ModeConfig(
        initial_extra_commands=(
            "gamemode survival @p",
            "difficulty peaceful",
            "gamerule doMobSpawning false",
            "gamerule doDaylightCycle false",
            "time set 0",
        ) + _COMMON,
        max_episode_steps=6000,
    ),
    "survival": ModeConfig(
        initial_extra_commands=(
            "gamemode survival @p",
            "difficulty normal",
            "gamerule doMobSpawning true",
            "gamerule doDaylightCycle true",
        ) + _COMMON,
        max_episode_steps=12000,
    ),
}

INVENTORY_CMDS: tuple[str, ...] = (
    "clear @p",
    "give @p minecraft:diamond_pickaxe 5",
    "enchant @p efficiency 5",
    "enchant @p fortune 3",
    "enchant @p unbreaking 3",
    "effect give @p minecraft:night_vision 999999 0 true",
)


# =============================================================
# 2. 광물 상수 (Seoyeon과 동일)
# =============================================================

ORE_BLOCK_IDS: frozenset[str] = frozenset({
    "minecraft:diamond_ore",           "minecraft:deepslate_diamond_ore",
    "minecraft:iron_ore",              "minecraft:deepslate_iron_ore",
    "minecraft:gold_ore",              "minecraft:deepslate_gold_ore",
    "minecraft:redstone_ore",          "minecraft:deepslate_redstone_ore",
    "minecraft:lapis_ore",             "minecraft:deepslate_lapis_ore",
    "minecraft:coal_ore",              "minecraft:deepslate_coal_ore",
    "minecraft:copper_ore",            "minecraft:deepslate_copper_ore",
    "minecraft:emerald_ore",           "minecraft:deepslate_emerald_ore",
})

ORE_DROP_REWARDS: dict[str, float] = {
    "minecraft:diamond":       20.0,
    "minecraft:raw_iron":       4.0,
    "minecraft:raw_gold":       6.0,
    "minecraft:redstone":       3.0,
    "minecraft:lapis_lazuli":   3.0,
    "minecraft:coal":           1.0,
    "minecraft:raw_copper":     2.0,
    "minecraft:emerald":        5.0,
}

_ORE_BLOCK_VALUE: dict[str, float] = {
    "minecraft:diamond_ore":           100.0,
    "minecraft:deepslate_diamond_ore": 100.0,
    "minecraft:emerald_ore":            15.0,
    "minecraft:deepslate_emerald_ore":  15.0,
    "minecraft:gold_ore":               20.0,
    "minecraft:deepslate_gold_ore":     20.0,
    "minecraft:iron_ore":               15.0,
    "minecraft:deepslate_iron_ore":     15.0,
    "minecraft:redstone_ore":           10.0,
    "minecraft:deepslate_redstone_ore": 10.0,
    "minecraft:lapis_ore":              10.0,
    "minecraft:deepslate_lapis_ore":    10.0,
    "minecraft:copper_ore":              5.0,
    "minecraft:deepslate_copper_ore":    5.0,
    "minecraft:coal_ore":                1.0,
    "minecraft:deepslate_coal_ore":      1.0,
}


# =============================================================
# 3. 액션 공간 (Seoyeon과 동일)
# =============================================================

ACTION_NAMES: list[str] = [
    "NO_OP", "FORWARD", "BACKWARD", "LEFT", "RIGHT", "JUMP",
    "ATTACK", "CAMERA_LEFT", "CAMERA_RIGHT", "CAMERA_UP", "CAMERA_DOWN",
    "ATTACK_FORWARD", "ATTACK_DOWN", "STAIRCASE_DOWN",
]
N_ACTIONS = len(ACTION_NAMES)

CAM_DEG = 8.0
ATTACK_HOLD_TICKS = 8

def build_action(name: str) -> dict:
    act = no_op_v2()
    match name:
        case "FORWARD":       act["forward"]      = True
        case "BACKWARD":      act["back"]         = True
        case "LEFT":          act["left"]         = True
        case "RIGHT":         act["right"]        = True
        case "JUMP":          act["jump"]         = True
        case "ATTACK":        act["attack"]       = True
        case "CAMERA_LEFT":   act["camera_yaw"]  = -CAM_DEG
        case "CAMERA_RIGHT":  act["camera_yaw"]  =  CAM_DEG
        case "CAMERA_UP":     act["camera_pitch"] = -CAM_DEG
        case "CAMERA_DOWN":   act["camera_pitch"] =  CAM_DEG
    return act


# =============================================================
# 4. 보상 상수
# [ICM 변경점] LAMBDA_DECAY_STEPS, LAYER2_BASE, EXPLORE_CELL_SIZE 제거
# =============================================================

Y_SURFACE          =  64
Y_TARGET           = -58
Y_RANGE            = float(Y_SURFACE - Y_TARGET)

STEP_PENALTY       = -0.02
DEATH_PENALTY      = -30.0
HEALTH_LOSS_K      =  -0.2
BLOCK_BREAK_BONUS  =   0.15

ORE_AIM_BONUS        = 0.3
ORE_AIM_ATTACK_BONUS = 1.0
LAYER1_MAX         =   0.15
Y_DELTA_BONUS      =   0.5

DEPTH_MILESTONES: list[tuple[float, float]] = [
    (40.0,   2.0),
    ( 0.0,   5.0),
    (-20.0, 10.0),
    (-40.0, 15.0),
    (-58.0, 25.0),
]

REWARD_CLIP = (-30.0, 60.0)

# [ICM 추가] 내재 보상 클립 (외재 보상과 스케일 맞춤)
ICM_REWARD_CLIP = 2.0


# =============================================================
# 5. 헬퍼 함수 (Seoyeon과 동일)
# =============================================================

def _get_full(raw_obs):
    if isinstance(raw_obs, dict):
        return raw_obs.get("full", raw_obs)
    return raw_obs

def _scalar(obs, key: str, default: float = 0.0) -> float:
    full = _get_full(obs)
    if isinstance(full, dict):
        return float(full.get(key, default))
    return float(getattr(full, key, default))

def _get_hit(raw_obs):
    return getattr(_get_full(raw_obs), "raycast_result", None)

def _hit_type(hit) -> str:
    raw = getattr(hit, "type", None)
    if raw is None:
        return "miss"
    return {0: "miss", 1: "block", 2: "entity"}.get(int(raw), "miss")

def _hit_state(hit) -> str:
    tb = getattr(hit, "target_block", None)
    if tb is not None:
        tk = getattr(tb, "translation_key", "") or ""
        if tk:
            return _tk_to_block_id(tk)
    for attr in ("block_state", "block_id", "translation_key"):
        v = getattr(hit, attr, None)
        if v:
            return _tk_to_block_id(str(v))
    return ""

def _tk_to_block_id(tk: str) -> str:
    if tk.startswith("block."):
        parts = tk[len("block."):].split(".", 1)
        if len(parts) == 2:
            return f"{parts[0]}:{parts[1]}"
    return tk

def _strip_state(s: str) -> str:
    return s.split("[")[0].strip()

def _normalize_item_key(tk: str) -> str:
    if tk.startswith("item.") or tk.startswith("block."):
        parts = tk.split(".", 2)
        if len(parts) >= 3:
            return f"{parts[1]}:{parts[2]}"
        elif len(parts) == 2:
            return parts[1]
    return tk

def _get_inv_counts(raw_obs) -> dict[str, int]:
    full = _get_full(raw_obs)
    inv = getattr(full, "inventory", [])
    counts: dict[str, int] = {}
    for item in inv:
        key = _normalize_item_key(item.translation_key)
        counts[key] = counts.get(key, 0) + item.count
    return counts

def _extract_image(obs) -> np.ndarray | None:
    if isinstance(obs, np.ndarray):
        img = obs
    elif isinstance(obs, dict):
        img = obs.get("pov")
        if img is None:
            img = obs.get("rgb")
        if img is not None:
            img = np.asarray(img, dtype=np.uint8)
        else:
            full = obs.get("full")
            if full is not None and isinstance(getattr(full, "image", None), bytes):
                try:
                    arr = np.frombuffer(full.image, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    return None
            else:
                return None
    elif isinstance(getattr(obs, "image", None), bytes):
        try:
            arr = np.frombuffer(obs.image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    else:
        return None
    if img is not None and img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img.astype(np.uint8) if img is not None else None


# =============================================================
# 6. MiningICMWrapper
# [ICM 변경점]
#   - _visited set 제거 (→ ICM이 novelty 자동 측정)
#   - lambda_intrinsic 제거
#   - icm_module 참조 저장 (ICMCallback이 step 정보를 전달)
#   - _last_obs_img, _last_obs_state: ICM 버퍼에 넣기 위한 이전 관찰 저장
#   - info["icm_int_r"]: 현재 스텝 내재 보상 (콜백이 채움)
# =============================================================

class MiningICMWrapper(gym.Wrapper):
    """
    ICM 버전 광물 채굴 환경 래퍼.

    Seoyeon MiningWrapper 대비 차이:
      제거: _visited (탐험 셀 집합), lambda_intrinsic, _current_lambda()
            LAYER2 탐험 보너스 계산 전체
      추가: _last_obs_img, _last_obs_state (ICM 버퍼용 이전 관찰 캐시)
            info["icm_obs"] — ICMCallback이 읽어서 버퍼에 push
    """

    H, W   = 64, 114
    RH, RW = 180, 320

    def __init__(self, env, cfg: ModeConfig, max_steps: int, mode_name: str = ""):
        super().__init__(env)
        self.cfg       = cfg
        self.max_steps = max_steps
        self.mode_name = mode_name

        self.action_space      = spaces.Discrete(N_ACTIONS)
        # [ICM] state 차원 8 (Seoyeon 10에서 lambda·is_ore_val 2개 유지하되
        #       lambda 대신 icm_r_norm 으로 교체 → 여전히 10차원)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (self.H, self.W, 3), dtype=np.uint8),
            "state": spaces.Box(-2.0, 2.0, (10,), dtype=np.float32),
        })

        self._step        = 0
        self._total_steps = 0
        self._ep_r        = 0.0
        self._prev_inv: dict[str, int] = {}
        self._prev_health = 20.0
        self._min_y       = float(Y_SURFACE)
        self._milestone_reached: set[float] = set()
        # ICM용 이전 관찰 캐시 (ICMCallback이 채움)
        self._last_obs_img:   np.ndarray | None = None
        self._last_obs_state: np.ndarray | None = None
        # 최근 내재 보상 (obs state[7] 에 반영)
        self._last_icm_r = 0.0

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        noop = no_op_v2()
        for _ in range(5):
            try:
                raw, *_ = self.env.step(noop)
            except Exception:
                break

        self._step        = 0
        self._ep_r        = 0.0
        self._prev_health = _scalar(raw, "health", 20.0)
        self._min_y       = _scalar(raw, "y")
        self._milestone_reached = set()
        self._prev_inv    = _get_inv_counts(raw)
        self._last_icm_r  = 0.0
        self._ep_ore_counts: dict[str, int] = {k: 0 for k in ORE_DROP_REWARDS}

        obs = self._make_obs(raw)
        self._last_obs_img   = obs["image"].copy()
        self._last_obs_state = obs["state"].copy()
        info["render_frame"] = self._render(_extract_image(raw))
        return obs, info

    def step(self, action: int):
        name = ACTION_NAMES[int(action)]
        terminated = truncated = False

        if name in ("ATTACK", "ATTACK_FORWARD", "ATTACK_DOWN", "STAIRCASE_DOWN"):
            _dug_down = False
            _pitch_offset = 0.0

            if name == "ATTACK_FORWARD":
                fwd = no_op_v2(); fwd["forward"] = True
                for _ in range(2):
                    raw, _, terminated, truncated, info = self.env.step(fwd)
                    if terminated or truncated: break
            elif name == "ATTACK_DOWN":
                dn = no_op_v2(); dn["camera_pitch"] = CAM_DEG
                for _ in range(2):
                    raw, _, terminated, truncated, info = self.env.step(dn)
                    if terminated or truncated: break
                _pitch_offset = CAM_DEG * 2
                _dug_down = True
            elif name == "STAIRCASE_DOWN":
                fwd = no_op_v2(); fwd["forward"] = True
                for _ in range(3):
                    raw, _, terminated, truncated, info = self.env.step(fwd)
                    if terminated or truncated: break
                if not (terminated or truncated):
                    dn = no_op_v2(); dn["camera_pitch"] = CAM_DEG * 4
                    raw, _, terminated, truncated, info = self.env.step(dn)
                    _pitch_offset = CAM_DEG * 4
                _dug_down = True

            if not (terminated or truncated):
                atk = no_op_v2(); atk["attack"] = True
                for _ in range(ATTACK_HOLD_TICKS):
                    raw, _, terminated, truncated, info = self.env.step(atk)
                    if terminated or truncated: break

            if not (terminated or truncated):
                if _dug_down:
                    noop = no_op_v2()
                    for _ in range(5):
                        raw, _, terminated, truncated, info = self.env.step(noop)
                        if terminated or truncated: break
                else:
                    pickup = no_op_v2(); pickup["forward"] = True
                    for _ in range(4):
                        raw, _, terminated, truncated, info = self.env.step(pickup)
                        if terminated or truncated: break

            if not (terminated or truncated) and _pitch_offset > 0.0:
                restore = no_op_v2()
                restore["camera_pitch"] = -_pitch_offset
                raw, _, terminated, truncated, info = self.env.step(restore)

            is_attack = True
        else:
            raw, _, terminated, truncated, info = self.env.step(build_action(name))
            is_attack = False

        self._step        += 1
        self._total_steps += 1
        truncated = truncated or (self._step >= self.max_steps)

        curr_inv = _get_inv_counts(raw)
        obs      = self._make_obs(raw)
        reward   = self._compute_reward(raw, terminated or truncated, is_attack, curr_inv)
        self._ep_r += reward

        # 인벤토리 변화 출력
        for key, count in curr_inv.items():
            prev = self._prev_inv.get(key, 0)
            if count > prev:
                matched = key in ORE_DROP_REWARDS
                tag = "ORE ✓" if matched else "----"
                print(f"  [INV] +{count - prev} {key} (total={count}) [{tag}]")

        for item_key in ORE_DROP_REWARDS:
            gained = max(0, curr_inv.get(item_key, 0) - self._prev_inv.get(item_key, 0))
            if gained > 0:
                self._ep_ore_counts[item_key] += gained

        self._prev_inv    = curr_inv
        self._prev_health = _scalar(raw, "health", 20.0)

        # [ICM] ICMCallback이 읽을 현재 관찰 저장
        info.update({
            "episode_step":   self._step,
            "episode_reward": self._ep_r,
            "y_level":        _scalar(raw, "y"),
            "render_frame":   self._render(_extract_image(raw)),
            "ore_counts":     self._ep_ore_counts.copy(),
            "icm_obs": {           # ICMCallback이 버퍼에 push하기 위해 읽는 정보
                "prev_img":   self._last_obs_img,
                "prev_state": self._last_obs_state,
                "action":     int(action),
                "next_img":   obs["image"],
                "next_state": obs["state"],
            },
            "icm_int_r": self._last_icm_r,  # ICMCallback이 채운 값 (이번 step은 0)
        })

        # 다음 스텝을 위해 현재 관찰 저장
        self._last_obs_img   = obs["image"].copy()
        self._last_obs_state = obs["state"].copy()

        if terminated or truncated:
            self._print_ore_summary()

        return obs, reward, terminated, truncated, info

    # ── observation ──────────────────────────────────────────────
    def _make_obs(self, raw) -> dict:
        img = _extract_image(raw)
        img = (
            cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            if img is not None
            else np.zeros((self.H, self.W, 3), dtype=np.uint8)
        )

        y     = _scalar(raw, "y")
        yaw   = _scalar(raw, "yaw")
        pitch = _scalar(raw, "pitch")
        hp    = _scalar(raw, "health",     20.0)
        food  = _scalar(raw, "food_level", 20.0)
        depth_progress = float(np.clip((Y_SURFACE - y) / Y_RANGE, 0.0, 1.0))

        is_ore, ore_value_norm = 0.0, 0.0
        hit = _get_hit(raw)
        if hit is not None and _hit_type(hit) == "block":
            block_id = _strip_state(_hit_state(hit))
            if block_id in ORE_BLOCK_IDS:
                is_ore         = 1.0
                ore_value_norm = _ORE_BLOCK_VALUE.get(block_id, 0.0) / 100.0

        state = np.array([
            float(np.clip(y / 64.0, -2.0, 2.0)),        # [0] Y 위치
            depth_progress,                               # [1] 목표 깊이 진행도
            float(np.clip(hp   / 20.0, 0.0, 1.0)),       # [2] 체력
            float(np.clip(food / 20.0, 0.0, 1.0)),       # [3] 배고픔
            float(np.sin(np.radians(yaw))),               # [4] 방향 sin
            float(np.cos(np.radians(yaw))),               # [5] 방향 cos
            float(np.clip(pitch / 90.0, -1.0, 1.0)),     # [6] 피치
            # [7] ICM 내재 보상 (Seoyeon: lambda_intrinsic)
            float(np.clip(self._last_icm_r / ICM_REWARD_CLIP, 0.0, 1.0)),
            is_ore,                                       # [8] 광물 조준 여부
            ore_value_norm,                               # [9] 조준 광물 가치
        ], dtype=np.float32)

        return {"image": img, "state": state}

    # ── reward (LAYER2 탐험 보너스 제거) ─────────────────────────
    def _compute_reward(self, raw, done: bool, is_attack: bool, curr_inv: dict) -> float:
        r = 0.0
        r += STEP_PENALTY

        health = _scalar(raw, "health", 20.0)
        if health < self._prev_health:
            r += HEALTH_LOSS_K * (self._prev_health - health)
        if done and health <= 0.0:
            r += DEATH_PENALTY

        y = _scalar(raw, "y")
        depth_progress = float(np.clip((Y_SURFACE - y) / Y_RANGE, 0.0, 1.0))
        r += LAYER1_MAX * depth_progress

        if y < self._min_y:
            r += Y_DELTA_BONUS * (self._min_y - y)
            self._min_y = y

        for threshold, bonus in DEPTH_MILESTONES:
            if y <= threshold and threshold not in self._milestone_reached:
                self._milestone_reached.add(threshold)
                r += bonus

        # [ICM 변경] LAYER2 탐험 보너스 없음 — ICMCallback이 별도로 r_int를 extrinsic에 더함

        hit = _get_hit(raw)
        if hit is not None and _hit_type(hit) == "block":
            block_id = _strip_state(_hit_state(hit))
            if block_id in ORE_BLOCK_IDS:
                ore_val = _ORE_BLOCK_VALUE.get(block_id, 1.0) / 100.0
                r += ORE_AIM_BONUS * ore_val
                if is_attack:
                    r += ORE_AIM_ATTACK_BONUS * ore_val

        prev_total = sum(self._prev_inv.values())
        curr_total = sum(curr_inv.values())
        if curr_total > prev_total:
            r += BLOCK_BREAK_BONUS

        for item_key, reward_per_unit in ORE_DROP_REWARDS.items():
            gained = max(0, curr_inv.get(item_key, 0) - self._prev_inv.get(item_key, 0))
            if gained > 0:
                r += reward_per_unit * float(np.log1p(gained))

        return float(np.clip(r, *REWARD_CLIP))

    _ORE_DISPLAY = {
        "minecraft:diamond":      ("Diamond",  "\033[96m"),
        "minecraft:emerald":      ("Emerald",  "\033[92m"),
        "minecraft:raw_gold":     ("Gold",     "\033[93m"),
        "minecraft:raw_iron":     ("Iron",     "\033[37m"),
        "minecraft:redstone":     ("Redstone", "\033[91m"),
        "minecraft:lapis_lazuli": ("Lapis",    "\033[94m"),
        "minecraft:raw_copper":   ("Copper",   "\033[33m"),
        "minecraft:coal":         ("Coal",     "\033[90m"),
    }
    _RST = "\033[0m"
    _BAR_WIDTH = 20

    def _print_ore_summary(self):
        total_mined = sum(self._ep_ore_counts.values())
        max_count   = max(self._ep_ore_counts.values()) if total_mined > 0 else 1
        print(f"\n{'='*50}")
        print(f"  Episode Summary  |  Steps:{self._step}  R:{self._ep_r:+.1f}")
        print(f"{'='*50}")
        for item_key, (name, color) in self._ORE_DISPLAY.items():
            count   = self._ep_ore_counts.get(item_key, 0)
            bar_len = int(self._BAR_WIDTH * count / max_count) if max_count > 0 else 0
            bar = "\u2588" * bar_len + "\u2591" * (self._BAR_WIDTH - bar_len)
            print(f"  {color}{name:>8s}{self._RST} |{color}{bar}{self._RST}| {count:3d}")
        print(f"{'─'*50}")
        print(f"  {'Total':>8s} | {total_mined:3d} items")
        print(f"{'='*50}\n")

    def _render(self, img_rgb) -> np.ndarray:
        if img_rgb is None:
            frame = np.zeros((self.RH, self.RW, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(
                cv2.resize(img_rgb, (self.RW, self.RH), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_RGB2BGR,
            )
        lam = f"{self._last_icm_r:.3f}"
        lines = [
            f"Step:{self._step:4d}  R:{self._ep_r:+7.1f}  ICM_r:{lam}",
            f"Mode:{self.mode_name.upper()} [ICM]",
        ]
        for i, txt in enumerate(lines):
            ypos = 18 + i * 20
            cv2.putText(frame, txt, (5, ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (5, ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        return frame


# =============================================================
# 7. Feature Extractor (SB3 정책용, Seoyeon과 동일)
# =============================================================

class MiningCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(obs_space, features_dim)

        img_shape = obs_space["image"].shape
        if img_shape[-1] in (1, 3, 4):
            h, w, c = img_shape
        else:
            c, h, w = img_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            _dummy = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn(_dummy).shape[1]

        state_dim = obs_space["state"].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim), nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        img_raw = obs["image"].float()
        if img_raw.shape[-1] in (1, 3, 4):
            img = img_raw.permute(0, 3, 1, 2) / 255.0
        else:
            img = img_raw / 255.0
        return self.fusion(
            torch.cat([self.cnn(img), self.state_mlp(obs["state"].float())], dim=-1)
        )


# =============================================================
# 8. ICMCallback — ICM 업데이트 및 내재 보상 주입
# [ICM 핵심] SB3 step과 ICM을 연결하는 콜백
# =============================================================

class ICMCallback(BaseCallback):
    """
    매 step마다:
      1. info["icm_obs"] 에서 (s, a, s') 추출 → replay buffer 에 push
      2. ICM으로 내재 보상 계산 → 환경 wrapper의 _last_icm_r 갱신
         (다음 관찰의 state[7]에 반영)
      3. SB3 reward buffer에 내재 보상 더하기 (model.rollout_buffer 직접 접근)
      icm_update_freq 스텝마다 ICM 파라미터 업데이트
    """

    def __init__(
        self,
        icm: ICMModule,
        buf: ICMReplayBuffer,
        icm_scale: float = 1.0,
        icm_update_freq: int = 512,
        icm_batch_size: int = 256,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.icm            = icm
        self.buf            = buf
        self.icm_scale      = icm_scale
        self.icm_update_freq = icm_update_freq
        self.icm_batch_size  = icm_batch_size
        self._icm_losses: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards")  # np array (n_envs,)

        for i, info in enumerate(infos):
            icm_data = info.get("icm_obs")
            if icm_data is None:
                continue

            prev_img   = icm_data["prev_img"]
            prev_state = icm_data["prev_state"]
            action     = icm_data["action"]
            next_img   = icm_data["next_img"]
            next_state = icm_data["next_state"]

            if prev_img is None or prev_state is None:
                continue

            # 버퍼에 전이 저장
            self.buf.push(prev_img, prev_state, action, next_img, next_state)

            # 내재 보상 계산
            r_int = self.icm.intrinsic_reward(
                prev_img[None],   # (1, H, W, C)
                prev_state[None], # (1, D)
                np.array([action]),
                next_img[None],
                next_state[None],
            )[0]

            r_int_clipped = float(np.clip(r_int, 0.0, ICM_REWARD_CLIP))

            # 환경 wrapper에 내재 보상 기록 (다음 obs state[7]에 반영)
            # DummyVecEnv → training_env.envs[i] 로 접근
            try:
                env_i = self.training_env.envs[i]
                # Wrapper chain 탐색
                w = env_i
                while w is not None:
                    if isinstance(w, MiningICMWrapper):
                        w._last_icm_r = r_int_clipped
                        break
                    w = getattr(w, "env", None)
            except Exception:
                pass

            # SB3 reward에 내재 보상 추가
            if rewards is not None:
                rewards[i] += self.icm_scale * r_int_clipped

        # ICM 파라미터 업데이트
        if self.num_timesteps % self.icm_update_freq == 0 and len(self.buf) >= self.icm_batch_size:
            batch = self.buf.sample(self.icm_batch_size)
            stats = self.icm.update(*batch)
            self._icm_losses.append(stats["icm/loss"])

            if self.num_timesteps % (self.icm_update_freq * 10) == 0:
                avg_loss = float(np.mean(self._icm_losses[-20:]))
                self.logger.record("icm/loss", avg_loss)
                if _wactive():
                    _wlog({"icm/loss": avg_loss}, step=self.num_timesteps)
                if self.verbose:
                    print(f"  [ICM] step={self.num_timesteps:,}  loss={avg_loss:.4f}")

        return True


# =============================================================
# 9. 기타 콜백 (Seoyeon과 동일)
# =============================================================

class MiningTrackingCallback(BaseCallback):
    _ORE_SHORT = {
        "minecraft:diamond":      "diamond",
        "minecraft:raw_iron":     "iron",
        "minecraft:raw_gold":     "gold",
        "minecraft:redstone":     "redstone",
        "minecraft:lapis_lazuli": "lapis",
        "minecraft:coal":         "coal",
        "minecraft:raw_copper":   "copper",
        "minecraft:emerald":      "emerald",
    }

    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._buf: list[dict] = []
        self._ep_ore_buf: list[dict[str, int]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._buf.append({
                "y":     info.get("y_level",        0.0),
                "r":     info.get("episode_reward", 0.0),
                "icm_r": info.get("icm_int_r",      0.0),
            })
            if len(self._buf) > 2000:
                del self._buf[:1000]

            # 에피소드 종료 시 ore 통계 즉시 logger에 기록
            if "episode" in info and "ore_counts" in info:
                self._ep_ore_buf.append(info["ore_counts"])
                if len(self._ep_ore_buf) > 100:
                    del self._ep_ore_buf[:50]
                self._log_ore()

        if self.num_timesteps % self.log_freq == 0 and self._buf:
            self._log_stats()
        return True

    def _log_ore(self):
        """에피소드 끝날 때마다 ore 평균 기록 → SB3 tabular 출력에 반영."""
        if not self._ep_ore_buf:
            return
        ore_log = {}
        for full_key, short in self._ORE_SHORT.items():
            vals     = [ep.get(full_key, 0) for ep in self._ep_ore_buf]
            mean_val = float(np.mean(vals))
            self.logger.record(f"ore/{short}", mean_val)
            ore_log[f"ore/{short}"] = mean_val
        if _wactive():
            _wlog(ore_log, step=self.num_timesteps)

    def _log_stats(self):
        recent  = self._buf[-400:]
        avg_y   = float(np.mean([e["y"]     for e in recent]))
        avg_icm = float(np.mean([e["icm_r"] for e in recent]))

        ep_buf = self.model.ep_info_buffer
        mean_r = mean_l = None
        if ep_buf:
            mean_r = float(np.mean([e["r"] for e in ep_buf]))
            mean_l = float(np.mean([e["l"] for e in ep_buf]))
            self.logger.record("mining/mean_ep_reward", mean_r)
            self.logger.record("mining/mean_ep_length", mean_l)

        self.logger.record("mining/avg_y_level", avg_y)
        self.logger.record("mining/avg_icm_r",   avg_icm)

        if _wactive():
            _wlog({
                "mining/avg_y_level": avg_y,
                "mining/avg_icm_r":   avg_icm,
                **({"mining/mean_ep_reward": mean_r,
                    "mining/mean_ep_length": mean_l} if mean_r is not None else {}),
            }, step=self.num_timesteps)

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"  [{self.num_timesteps:>8,} steps]  avg_y={avg_y:+.1f}  icm_r={avg_icm:.4f}")
            if mean_r is not None:
                print(f"  ep_reward={mean_r:+.1f}  ep_length={mean_l:.0f}")
            print(f"{'─'*60}")


class RenderCallback(BaseCallback):
    WIN = "Mining ICM RL"

    def __init__(self, freq: int = 4):
        super().__init__()
        self.freq    = freq
        self._active = False
        self._tick   = 0

    def _on_training_start(self):
        try:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            self._active = True
        except Exception as e:
            print(f"⚠️  cv2 창 오픈 실패: {e}")

    def _on_step(self) -> bool:
        if not self._active:
            return True
        self._tick += 1
        if self._tick % self.freq:
            return True
        frame = (self.locals.get("infos") or [{}])[0].get("render_frame")
        if frame is None:
            frame = np.zeros((MiningICMWrapper.RH, MiningICMWrapper.RW, 3), dtype=np.uint8)
        try:
            cv2.imshow(self.WIN, frame)
        except Exception:
            self._active = False
            return True
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._active = False
            cv2.destroyWindow(self.WIN)
        return True

    def _on_training_end(self):
        if self._active:
            cv2.destroyWindow(self.WIN)


# =============================================================
# 10. 환경 생성
# =============================================================

def make_env(
    port: int = 8030,
    mode: str = "safe",
    max_steps: int | None = None,
    seed: int = 42,
    db_path: str | None = None,
) -> gym.Env:
    if mode not in MODES:
        raise ValueError(f"mode='{mode}' 불가. 선택: {list(MODES.keys())}")
    cfg = MODES[mode]

    cmds: list[str] = (
        list(INVENTORY_CMDS)
        + list(cfg.initial_extra_commands)
        + ["tp @p 0 -45 0"]
    )

    raw_env = make(
        initial_env_config=InitialEnvironmentConfig(
            image_width=320,
            image_height=180,
            seed=str(seed),
            world_type=WorldType.DEFAULT,
            render_distance=6,
            simulation_distance=6,
            hud_hidden=False,
            request_raycast=True,
            initial_extra_commands=cmds,
        ),
        port=port,
        verbose=False,
        verbose_gradle=True,
        render_action=False,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )

    mining_env = MiningICMWrapper(
        raw_env,
        cfg=cfg,
        max_steps=max_steps or cfg.max_episode_steps,
        mode_name=mode,
    )

    if db_path and Path(db_path).exists():
        return CaveSpawnWrapper(
            env=mining_env,
            db_path=db_path,
            cmds_list=cmds,
            warmup_episodes=5,
            score_weighted_sampling=True,
        )

    print(f"⚠️  cave_db 없음 (db_path={db_path!r}). 기본 스폰 사용.")
    return mining_env


# =============================================================
# 11. 훈련
# =============================================================

HP: dict[str, dict] = {
    "safe": dict(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
    "survival": dict(
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
}

# ICM 하이퍼파라미터
ICM_HP = dict(
    feat_dim=256,
    eta=0.01,       # 내재 보상 스케일 (forward 예측 MSE 스케일 보정)
    beta=0.2,       # forward:inverse 손실 비율 (논문 권장: 0.2)
    lr=3e-4,
    icm_scale=1.0,      # 외재 보상 대비 내재 보상 가중치
    icm_update_freq=512,
    icm_batch_size=256,
    buffer_capacity=8000,
)


def train(args):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = Path(args.log_dir)  / f"mining_icm_{args.env_mode}_{ts}"
    save_dir = Path(args.save_dir) / f"mining_icm_{args.env_mode}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    hp = HP[args.env_mode]

    if args.wandb_project:
        if not _WANDB:
            print("⚠️  wandb 미설치 → pip install wandb")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"mining_icm_{args.env_mode}_{ts}",
                sync_tensorboard=True,
                save_code=True,
                config={
                    "task":       "mining_icm",
                    "env_mode":   args.env_mode,
                    "total_steps": args.total_steps,
                    "seed":       args.seed,
                    **hp,
                    **ICM_HP,
                },
            )
            print(f"[WandB] {wandb.run.url}")

    # GPU 사용 여부
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ICM] device={device}")

    # ICM 모듈 & 버퍼 생성
    icm = ICMModule(
        n_actions=N_ACTIONS,
        img_h=MiningICMWrapper.H,
        img_w=MiningICMWrapper.W,
        img_c=3,
        state_dim=10,
        feat_dim=ICM_HP["feat_dim"],
        eta=ICM_HP["eta"],
        beta=ICM_HP["beta"],
        lr=ICM_HP["lr"],
        device=device,
    )
    buf = ICMReplayBuffer(capacity=ICM_HP["buffer_capacity"])

    def make_fn(offset: int):
        return lambda: make_env(
            port=args.base_port + offset,
            mode=args.env_mode,
            max_steps=args.max_steps or None,
            seed=args.seed + offset,
            db_path=args.db,
        )

    vec_env = VecMonitor(
        DummyVecEnv([make_fn(0)]) if args.n_envs == 1
        else SubprocVecEnv([make_fn(i) for i in range(args.n_envs)]),
        str(log_dir),
    )
    eval_env = VecMonitor(
        DummyVecEnv([make_fn(100)]),
        str(log_dir / "eval"),
    )

    policy_kwargs = dict(
        features_extractor_class=MiningCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        **hp,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
        device=device,
        seed=args.seed,
    )

    if args.resume:
        print(f"[Resume] {args.resume}")
        model.set_parameters(args.resume)

    callbacks = [
        ICMCallback(
            icm=icm,
            buf=buf,
            icm_scale=ICM_HP["icm_scale"],
            icm_update_freq=ICM_HP["icm_update_freq"],
            icm_batch_size=ICM_HP["icm_batch_size"],
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=str(save_dir / "checkpoints"),
            name_prefix="mining_icm",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(100_000 // args.n_envs, 1),
            n_eval_episodes=3,
            deterministic=True,
        ),
        MiningTrackingCallback(log_freq=2000, verbose=1),
        RenderCallback(freq=4),
    ]

    print(f"\n{'='*60}")
    print(f"  Mining ICM RL  |  mode={args.env_mode}  steps={args.total_steps:,}")
    print(f"  ICM: eta={ICM_HP['eta']}  beta={ICM_HP['beta']}  scale={ICM_HP['icm_scale']}")
    print(f"  비교: Seoyeon LAYER2(λ감쇠) → ICM 내재 보상으로 대체")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume),
    )

    model.save(str(save_dir / "final_model"))
    print(f"\n[Done] 모델 저장: {save_dir}/final_model")
    if _wactive():
        wandb.finish()


# =============================================================
# 12. 평가
# =============================================================

def evaluate(args):
    env = make_env(
        port=args.base_port,
        mode=args.env_mode,
        seed=args.seed,
        db_path=args.db,
    )
    model = PPO.load(args.resume, env=env)

    try:
        cv2.namedWindow("Mining ICM Eval", cv2.WINDOW_NORMAL)
        has_cv2 = True
    except Exception:
        has_cv2 = False

    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        ep_r, ep_steps = 0.0, 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            ep_r += r
            ep_steps += 1
            done = terminated or truncated

            if has_cv2:
                frame = info.get("render_frame")
                if frame is not None:
                    cv2.imshow("Mining ICM Eval", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    has_cv2 = False

        print(f"[Eval ep {ep+1}] steps={ep_steps}  R={ep_r:+.1f}")

    env.close()
    if has_cv2:
        cv2.destroyAllWindows()


# =============================================================
# 13. 진입점
# =============================================================

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",            default="train", choices=["train", "eval"])
    p.add_argument("--env_mode",        default="safe",  choices=["safe", "survival"])
    p.add_argument("--total_steps",     type=int,   default=3_000_000)
    p.add_argument("--max_steps",       type=int,   default=0)
    p.add_argument("--n_envs",          type=int,   default=1)
    p.add_argument("--base_port",       type=int,   default=8030)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--db",              default=None)
    p.add_argument("--resume",          default=None)
    p.add_argument("--log_dir",         default="logs")
    p.add_argument("--save_dir",        default="checkpoints")
    p.add_argument("--wandb_project",   default=None)
    p.add_argument("--wandb_run",       default=None)
    p.add_argument("--eval_episodes",   type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
