"""
hrl_mining.py — Hierarchical RL for Mining (Manager-Worker Pattern)

보상 구조:
  Manager (rule-based): 깊이 단계 [Y=40, 0, -20, -40, -58] 를 순서대로 지정
  Worker  (PPO 학습)  : 현재 서브골 Y 도달 + 채굴 보상을 최대화

baseline (real_mining_v2) vs ICM vs HRL 비교 실험용.
wandb_project 기본값: "mining_compare"

실행:
  python hrl_mining.py --mode train --db cave_db.json --total_steps 3000000
"""

from __future__ import annotations

import argparse
import math
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
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from cave_seed_scanner_v1 import CaveSpawnWrapper
    _CAVE_SCANNER = True
except ImportError:
    _CAVE_SCANNER = False
    print("⚠️  cave_seed_scanner_v1 미발견. CaveSpawnWrapper 없이 실행.")

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
# 1. 모드 설정
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
# 2. 광물 상수
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
# 3. 액션 공간 (v2와 동일: 14개)
# =============================================================

ACTION_NAMES: list[str] = [
    "NO_OP",          # 0
    "FORWARD",        # 1
    "BACKWARD",       # 2
    "LEFT",           # 3
    "RIGHT",          # 4
    "JUMP",           # 5
    "ATTACK",         # 6
    "CAMERA_LEFT",    # 7
    "CAMERA_RIGHT",   # 8
    "CAMERA_UP",      # 9
    "CAMERA_DOWN",    # 10
    "ATTACK_FORWARD", # 11
    "ATTACK_DOWN",    # 12
    "STAIRCASE_DOWN", # 13
]

CAM_DEG        = 8.0
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
# =============================================================

Y_SURFACE     =  64.0
Y_TARGET      = -58.0
Y_RANGE       = float(Y_SURFACE - Y_TARGET)   # 122.0

STEP_PENALTY  = -0.02
DEATH_PENALTY = -30.0
HEALTH_LOSS_K =  -0.2

LAYER1_MAX    =   0.15   # 서브골 방향 shaping 최대 보상/스텝
Y_DELTA_BONUS =   0.5    # 새 최저 Y 갱신 보상

DEPTH_MILESTONES: list[tuple[float, float]] = [
    ( 40.0,  2.0),
    (  0.0,  5.0),
    (-20.0, 10.0),
    (-40.0, 15.0),
    (-58.0, 25.0),
]

BLOCK_BREAK_BONUS  = 0.15
ORE_AIM_BONUS      = 0.3
ORE_AIM_ATTACK_BONUS = 1.0
REWARD_CLIP        = (-30.0, 60.0)

# HRL 전용 상수
DEPTH_STAGES: list[float] = [40.0, 0.0, -20.0, -40.0, -58.0]
STAGE_TIMEOUT         = 800    # 스텝 초과 시 강제 다음 스테이지
STAGE_REACH_THRESHOLD = 5.0   # 서브골 Y ± 5 블록 내 도달 인정
STAGE_REACH_BONUS     = 20.0  # 스테이지 도달 보너스
STAGE_TIMEOUT_PENALTY = -5.0  # 타임아웃 패널티
# 스테이지 idx 2(Y≤0)부터 채굴 보상 활성화, 선형 증가
MINE_GATE_START_IDX   = 2


# =============================================================
# 5. 헬퍼
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
    inv  = getattr(full, "inventory", None)
    if inv is None:
        return {}
    counts: dict[str, int] = {}
    items = getattr(inv, "items", inv) if not hasattr(inv, "__iter__") else inv
    try:
        for slot in items:
            raw_id  = getattr(slot, "translation_key", "") or ""
            item_id = _normalize_item_key(raw_id) if raw_id else ""
            count   = int(getattr(slot, "count", 0))
            if item_id and count > 0:
                counts[item_id] = counts.get(item_id, 0) + count
    except Exception:
        pass
    return counts


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _extract_image(raw_obs) -> np.ndarray | None:
    full = _get_full(raw_obs)
    img  = None
    if isinstance(full, dict):
        img = full.get("rgb", full.get("image", None))
        if img is None:
            return None
        if isinstance(img, bytes):
            try:
                arr = np.frombuffer(img, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                return None
    elif isinstance(getattr(full, "image", None), bytes):
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

    if img is not None and img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img.astype(np.uint8) if img is not None else None


# =============================================================
# 6. HRLMiningWrapper
# =============================================================

class HRLMiningWrapper(gym.Wrapper):
    """
    Manager-Worker 계층적 RL 래퍼.

    Manager (rule-based):
      DEPTH_STAGES = [Y=40, 0, -20, -40, -58] 순서로 서브골 지정.
      에이전트가 서브골 Y ± STAGE_REACH_THRESHOLD 내 도달 → 다음 스테이지로.
      STAGE_TIMEOUT 스텝 초과 시 강제 다음 스테이지 (패널티 부여).

    Worker (PPO):
      Obs: {"image": (H,W,3), "state": (12,)}
        state[0-9]  : v2와 동일 (Y, depth_progress, hp, food, sin/cos yaw, pitch, 0, is_ore, ore_val)
        state[10]   : delta_y_norm = (y - target_y) / Y_RANGE  (양수=목표보다 위)
        state[11]   : stage_norm   = stage_idx / (len(STAGES)-1)
      Reward:
        - 서브골 방향 Y shaping (현재 스테이지 Y 기준)
        - 서브골 도달 보너스 / 타임아웃 패널티
        - 블록 파괴 보너스 (항상)
        - 채굴 보상 (stage_idx >= MINE_GATE_START_IDX 이후, 선형 증가)
    """

    H, W   = 64, 114
    RH, RW = 180, 320

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

    def __init__(self, env, cfg: ModeConfig, max_steps: int, mode_name: str = ""):
        super().__init__(env)
        self.cfg       = cfg
        self.max_steps = max_steps
        self.mode_name = mode_name

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (self.H, self.W, 3), dtype=np.uint8),
            "state": spaces.Box(-2.0, 2.0, (12,), dtype=np.float32),
        })

        # 에피소드 상태
        self._step            = 0
        self._ep_r            = 0.0
        self._prev_health     = 20.0
        self._prev_inv: dict[str, int] = {}
        self._min_y           = Y_SURFACE
        self._milestone_reached: set[float] = set()
        self._ep_ore_counts: dict[str, int] = {}

        # HRL Manager 상태
        self._stage_idx   = 0
        self._stage_steps = 0

    # ── Manager 헬퍼 ──────────────────────────────────────────

    @property
    def _current_target_y(self) -> float:
        return DEPTH_STAGES[self._stage_idx]

    def _try_advance_stage(self, y: float) -> float:
        """서브골 도달 또는 타임아웃 시 스테이지 전진. 보너스/패널티 반환."""
        bonus = 0.0
        self._stage_steps += 1

        reached  = y <= self._current_target_y + STAGE_REACH_THRESHOLD
        timedout = self._stage_steps >= STAGE_TIMEOUT

        if reached:
            bonus += STAGE_REACH_BONUS
        elif timedout:
            bonus += STAGE_TIMEOUT_PENALTY

        if (reached or timedout) and self._stage_idx < len(DEPTH_STAGES) - 1:
            self._stage_idx  += 1
            self._stage_steps = 0

        return bonus

    # ── mine_gate: 스테이지 깊어질수록 채굴 보상 증가 ────────
    @property
    def _mine_gate(self) -> float:
        if self._stage_idx < MINE_GATE_START_IDX:
            return 0.0
        return min(1.0, (self._stage_idx - MINE_GATE_START_IDX + 1) /
                        (len(DEPTH_STAGES) - MINE_GATE_START_IDX))

    # ── reset ─────────────────────────────────────────────────

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)

        noop = no_op_v2()
        for _ in range(5):
            try:
                raw, *_ = self.env.step(noop)
            except Exception:
                break

        self._step             = 0
        self._ep_r             = 0.0
        self._prev_health      = _scalar(raw, "health", 20.0)
        self._min_y            = _scalar(raw, "y")
        self._milestone_reached = set()
        self._prev_inv         = _get_inv_counts(raw)
        self._ep_ore_counts    = {k: 0 for k in ORE_DROP_REWARDS}

        # Manager 초기화
        self._stage_idx   = 0
        self._stage_steps = 0

        obs = self._make_obs(raw)
        info["render_frame"] = self._render(_extract_image(raw))
        return obs, info

    # ── step ──────────────────────────────────────────────────

    def step(self, action: int):
        name = ACTION_NAMES[int(action)]
        terminated = truncated = False

        if name in ("ATTACK", "ATTACK_FORWARD", "ATTACK_DOWN", "STAIRCASE_DOWN"):
            _dug_down    = False
            _pitch_offset = 0.0

            if name == "ATTACK_FORWARD":
                fwd = no_op_v2(); fwd["forward"] = True
                for _ in range(2):
                    raw, _, terminated, truncated, info = self.env.step(fwd)
                    if terminated or truncated:
                        break
            elif name == "ATTACK_DOWN":
                dn = no_op_v2(); dn["camera_pitch"] = CAM_DEG
                for _ in range(2):
                    raw, _, terminated, truncated, info = self.env.step(dn)
                    if terminated or truncated:
                        break
                _pitch_offset = CAM_DEG * 2
                _dug_down     = True
            elif name == "STAIRCASE_DOWN":
                fwd = no_op_v2(); fwd["forward"] = True
                for _ in range(3):
                    raw, _, terminated, truncated, info = self.env.step(fwd)
                    if terminated or truncated:
                        break
                if not (terminated or truncated):
                    dn = no_op_v2(); dn["camera_pitch"] = CAM_DEG * 4
                    raw, _, terminated, truncated, info = self.env.step(dn)
                    _pitch_offset = CAM_DEG * 4
                _dug_down = True

            if not (terminated or truncated):
                atk = no_op_v2(); atk["attack"] = True
                for _ in range(ATTACK_HOLD_TICKS):
                    raw, _, terminated, truncated, info = self.env.step(atk)
                    if terminated or truncated:
                        break

            if not (terminated or truncated):
                if _dug_down:
                    noop = no_op_v2()
                    for _ in range(5):
                        raw, _, terminated, truncated, info = self.env.step(noop)
                        if terminated or truncated:
                            break
                else:
                    pickup = no_op_v2(); pickup["forward"] = True
                    for _ in range(4):
                        raw, _, terminated, truncated, info = self.env.step(pickup)
                        if terminated or truncated:
                            break

            if not (terminated or truncated) and _pitch_offset > 0.0:
                restore = no_op_v2()
                restore["camera_pitch"] = -_pitch_offset
                raw, _, terminated, truncated, info = self.env.step(restore)

            is_attack = True

        else:
            raw, _, terminated, truncated, info = self.env.step(build_action(name))
            is_attack = False

        self._step += 1
        truncated = truncated or (self._step >= self.max_steps)

        # 사망 감지 → 인벤토리 복구
        health = _scalar(raw, "health", 20.0)
        if health <= 0.0 or (self._prev_health > 0.0 and health >= 20.0
                              and self._prev_health < 2.0):
            raw_env = self.env
            while hasattr(raw_env, "env"):
                raw_env = raw_env.env
            if hasattr(raw_env, "add_commands"):
                raw_env.add_commands(list(INVENTORY_CMDS))

        curr_inv = _get_inv_counts(raw)
        obs      = self._make_obs(raw)
        reward   = self._compute_reward(raw, terminated or truncated, is_attack, curr_inv)
        self._ep_r += reward

        # 인벤토리 변화 기록 (광물 카운트)
        for key in ORE_DROP_REWARDS:
            gained = max(0, curr_inv.get(key, 0) - self._prev_inv.get(key, 0))
            if gained > 0:
                self._ep_ore_counts[key] = self._ep_ore_counts.get(key, 0) + gained

        self._prev_health = health
        self._prev_inv    = curr_inv

        done = terminated or truncated
        if done:
            info["episode"]    = {"r": self._ep_r, "l": self._step}
            info["ore_counts"] = dict(self._ep_ore_counts)
            info["final_stage"] = self._stage_idx
            self._print_ore_summary()

        info["y_level"]         = _scalar(raw, "y")
        info["episode_reward"]  = self._ep_r
        info["stage_idx"]       = self._stage_idx
        info["render_frame"]    = self._render(_extract_image(raw))
        return obs, reward, terminated, truncated, info

    # ── 관찰 생성 (12-dim state) ───────────────────────────────

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

        # HRL 전용: 서브골까지 delta (양수 = 목표보다 위에 있음 = 내려가야 함)
        target_y    = self._current_target_y
        delta_y_norm = float(np.clip((y - target_y) / Y_RANGE, -1.0, 1.0))
        stage_norm   = self._stage_idx / max(1, len(DEPTH_STAGES) - 1)

        state = np.array([
            float(np.clip(y / 64.0, -2.0, 2.0)),        # [0]  Y 위치
            depth_progress,                               # [1]  전체 깊이 진행도
            float(np.clip(hp   / 20.0, 0.0, 1.0)),       # [2]  체력
            float(np.clip(food / 20.0, 0.0, 1.0)),       # [3]  배고픔
            float(np.sin(np.radians(yaw))),               # [4]  방향 sin
            float(np.cos(np.radians(yaw))),               # [5]  방향 cos
            float(np.clip(pitch / 90.0, -1.0, 1.0)),     # [6]  피치
            0.0,                                          # [7]  (예약: ICM용 슬롯)
            is_ore,                                       # [8]  광물 조준 여부
            ore_value_norm,                               # [9]  조준 광물 가치
            delta_y_norm,                                 # [10] 서브골까지 y 거리 (HRL)
            float(stage_norm),                            # [11] 현재 스테이지 진행도 (HRL)
        ], dtype=np.float32)

        return {"image": img, "state": state}

    # ── 보상 ─────────────────────────────────────────────────

    def _compute_reward(
        self,
        raw,
        done: bool,
        is_attack: bool,
        curr_inv: dict[str, int],
    ) -> float:
        r   = 0.0
        y   = _scalar(raw, "y")

        # 항상 활성
        r += STEP_PENALTY

        health = _scalar(raw, "health", 20.0)
        if health < self._prev_health:
            r += HEALTH_LOSS_K * (self._prev_health - health)
        if done and health <= 0.0:
            r += DEATH_PENALTY

        # ── Phase 1: 서브골 방향 Y shaping ─────────────────────
        # 글로벌 Y=-58이 아닌 현재 스테이지 Y를 기준으로 shaping
        target_y  = self._current_target_y
        dist      = max(0.0, y - target_y)                    # 목표까지 남은 거리 (양수=위)
        max_dist  = max(1.0, Y_SURFACE - target_y)
        local_progress = 1.0 - dist / max_dist
        r += LAYER1_MAX * float(np.clip(local_progress, 0.0, 1.0))

        # Y 최저점 갱신 보너스
        if y < self._min_y:
            r += Y_DELTA_BONUS * (self._min_y - y)
            self._min_y = y

        # 전역 깊이 마일스톤 (1회성)
        for threshold, bonus in DEPTH_MILESTONES:
            if y <= threshold and threshold not in self._milestone_reached:
                self._milestone_reached.add(threshold)
                r += bonus

        # ── Manager: 스테이지 전진 체크 ────────────────────────
        r += self._try_advance_stage(y)

        # ── Phase 2: 블록 파괴 보너스 (항상) ───────────────────
        prev_total = sum(self._prev_inv.values())
        curr_total = sum(curr_inv.values())
        if curr_total > prev_total:
            r += BLOCK_BREAK_BONUS

        # ── Phase 3: 채굴 보상 (mine_gate 게이팅) ───────────────
        mg = self._mine_gate
        if mg > 0.0:
            hit = _get_hit(raw)
            if hit is not None and _hit_type(hit) == "block":
                block_id = _strip_state(_hit_state(hit))
                if block_id in ORE_BLOCK_IDS:
                    ore_val = _ORE_BLOCK_VALUE.get(block_id, 1.0) / 100.0
                    r += mg * ORE_AIM_BONUS * ore_val
                    if is_attack:
                        r += mg * ORE_AIM_ATTACK_BONUS * ore_val

            for item_key, reward_per_unit in ORE_DROP_REWARDS.items():
                gained = max(0, curr_inv.get(item_key, 0) - self._prev_inv.get(item_key, 0))
                if gained > 0:
                    r += mg * reward_per_unit * float(np.log1p(gained))

        return float(np.clip(r, *REWARD_CLIP))

    # ── 에피소드 종료 요약 출력 ────────────────────────────────

    def _print_ore_summary(self):
        total = sum(self._ep_ore_counts.values())
        max_c = max(self._ep_ore_counts.values()) if total > 0 else 1
        print(f"\n{'='*60}")
        print(f"  [HRL] Steps:{self._step}  R:{self._ep_r:+.1f}"
              f"  Stage:{self._stage_idx}/{len(DEPTH_STAGES)-1}"
              f"  Y:{_scalar(self.env,'y'):+.0f}")
        print(f"{'='*60}")
        for item_key, (name, color) in self._ORE_DISPLAY.items():
            count   = self._ep_ore_counts.get(item_key, 0)
            bar_len = int(self._BAR_WIDTH * count / max_c) if max_c > 0 else 0
            bar     = "█" * bar_len + "░" * (self._BAR_WIDTH - bar_len)
            print(f"  {color}{name:8s}{self._RST} [{bar}] {count:3d}")
        print()

    # ── 렌더링 ────────────────────────────────────────────────

    def _render(self, img: np.ndarray | None) -> np.ndarray:
        if img is None:
            return np.zeros((self.RH, self.RW, 3), dtype=np.uint8)
        return cv2.resize(img, (self.RW, self.RH), interpolation=cv2.INTER_LINEAR)


# =============================================================
# 7. CNN Feature Extractor
# =============================================================

class MiningCNNExtractor(BaseFeaturesExtractor):
    """CNN(image 64×114) + MLP(state 12-dim) → fused 256-dim."""

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
            _dummy      = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn(_dummy).shape[1]

        state_dim = obs_space["state"].shape[0]   # 12
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
# 8. 콜백
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
        self.log_freq      = log_freq
        self._buf: list[dict] = []
        self._ep_ore_buf: list[dict[str, int]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._buf.append({
                "y":     info.get("y_level",        0.0),
                "r":     info.get("episode_reward", 0.0),
                "stage": info.get("stage_idx",      0),
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
        recent    = self._buf[-400:]
        avg_y     = float(np.mean([e["y"]     for e in recent]))
        avg_stage = float(np.mean([e["stage"] for e in recent]))

        ep_buf = self.model.ep_info_buffer
        mean_r = mean_l = None
        if ep_buf:
            mean_r = float(np.mean([e["r"] for e in ep_buf]))
            mean_l = float(np.mean([e["l"] for e in ep_buf]))
            self.logger.record("mining/mean_ep_reward", mean_r)
            self.logger.record("mining/mean_ep_length", mean_l)

        self.logger.record("mining/avg_y_level",  avg_y)
        self.logger.record("mining/avg_stage_idx", avg_stage)

        if _wactive():
            _wlog({
                "mining/avg_y_level":   avg_y,
                "mining/avg_stage_idx": avg_stage,
                **({"mining/mean_ep_reward": mean_r,
                    "mining/mean_ep_length": mean_l} if mean_r is not None else {}),
            }, step=self.num_timesteps)

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"  [{self.num_timesteps:>8,} steps]"
                  f"  avg_y={avg_y:+.1f}  avg_stage={avg_stage:.2f}")
            if mean_r is not None:
                print(f"  ep_reward={mean_r:+.1f}  ep_length={mean_l:.0f}")
            print(f"{'─'*60}")


class RenderCallback(BaseCallback):
    WIN = "HRL Mining"

    def __init__(self, freq: int = 4):
        super().__init__(verbose=0)
        self.freq    = freq
        self._active = True

    def _on_step(self) -> bool:
        if not self._active or self._tick % self.freq:
            return True
        frame = (self.locals.get("infos") or [{}])[0].get("render_frame")
        if frame is None:
            frame = np.zeros((HRLMiningWrapper.RH, HRLMiningWrapper.RW, 3), dtype=np.uint8)
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
# 9. 환경 생성
# =============================================================

def make_env(
    port: int = 8050,
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

    mining_env = HRLMiningWrapper(
        raw_env,
        cfg=cfg,
        max_steps=max_steps or cfg.max_episode_steps,
        mode_name=mode,
    )

    if db_path and Path(db_path).exists() and _CAVE_SCANNER:
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
# 10. 하이퍼파라미터
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


# =============================================================
# 11. 훈련 / 평가
# =============================================================

def train(args):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = Path(args.log_dir)  / f"hrl_mining_{args.env_mode}_{ts}"
    save_dir = Path(args.save_dir) / f"hrl_mining_{args.env_mode}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    hp = HP[args.env_mode]

    if args.wandb_project:
        if not _WANDB:
            print("⚠️  wandb 미설치 → pip install wandb")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"hrl_{args.env_mode}_{ts}",
                group=args.wandb_project,   # 같은 project 내 그룹으로 묶어 비교
                sync_tensorboard=True,
                save_code=True,
                config={
                    "algo":        "hrl",
                    "task":        "mining",
                    "env_mode":    args.env_mode,
                    "total_steps": args.total_steps,
                    "n_envs":      args.n_envs,
                    "seed":        args.seed,
                    "db_path":     args.db,
                    "depth_stages": DEPTH_STAGES,
                    "stage_timeout": STAGE_TIMEOUT,
                    **hp,
                },
            )
            print(f"[WandB] {wandb.run.url}")

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

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, device=args.device)
        model.learning_rate = hp["learning_rate"]
        model.clip_range    = hp["clip_range"]
        model.ent_coef      = hp["ent_coef"]
        print(f"[Resume] {args.resume}")
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            tensorboard_log=str(log_dir),
            verbose=1,
            device=args.device,
            **hp,
            policy_kwargs=dict(
                features_extractor_class=MiningCNNExtractor,
                features_extractor_kwargs={"features_dim": 256},
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=nn.ReLU,
            ),
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(20_000 // args.n_envs, 1),
            save_path=str(save_dir),
            name_prefix="hrl_mining",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ),
        MiningTrackingCallback(log_freq=2000, verbose=1),
        RenderCallback(freq=4),
    ]

    print(f"\n[HRL Mining] {args.env_mode.upper()} | "
          f"{args.total_steps:,} steps | {args.n_envs} envs | "
          f"port={args.base_port}")
    print(f"  Stages: {DEPTH_STAGES}  timeout={STAGE_TIMEOUT}")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume),
        progress_bar=True,
    )

    final_path = str(save_dir / "final_model")
    model.save(final_path)
    print(f"\n[Saved] {final_path}")

    if _wactive():
        art = wandb.Artifact("hrl_mining_final", type="model")
        art.add_file(final_path + ".zip")
        wandb.run.log_artifact(art)
        wandb.finish()

    vec_env.close()
    eval_env.close()


def evaluate(args):
    env   = make_env(args.base_port, args.env_mode,
                     args.max_steps or None, args.seed, args.db)
    model = PPO.load(args.resume, env=env, device=args.device)

    ore_totals: dict[str, int] = {k: 0 for k in ORE_DROP_REWARDS}
    ep_rewards  = []

    for ep in range(args.n_eval_episodes):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        steps  = 0

        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(act))
            ep_r  += r
            done   = term or trunc
            steps += 1

        ore_totals_ep = info.get("ore_counts", {})
        for item in ORE_DROP_REWARDS:
            ore_totals[item] += ore_totals_ep.get(item, 0)

        ep_rewards.append(ep_r)
        print(f"  Ep {ep+1:2d}: R={ep_r:+.1f}  steps={steps}"
              f"  stage={info.get('final_stage', '?')}")

    print(f"\n평균 보상: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print("광물 수집 합계:")
    for item, total in ore_totals.items():
        if total > 0:
            print(f"  {item:35s}: {total:4d}개")
    env.close()


# =============================================================
# 12. CLI
# =============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HRL Mining — Manager-Worker Depth Stages")
    p.add_argument("--mode",        default="train", choices=["train", "eval"])
    p.add_argument("--env_mode",    choices=list(MODES.keys()), default="safe")
    _script_dir = str(Path(__file__).resolve().parent)
    p.add_argument("--db",          default=str(Path(_script_dir).parent.parent / "Seoyeon" / "mining_rl_pack" / "cave_db.json"),
                   help="cave_db.json 경로")
    p.add_argument("--total_steps", type=int, default=3_000_000)
    p.add_argument("--n_envs",      type=int, default=1)
    p.add_argument("--base_port",   type=int, default=8050,
                   help="baseline=8030, icm=8040, hrl=8050 (포트 충돌 방지)")
    p.add_argument("--max_steps",   type=int, default=0)
    p.add_argument("--log_dir",     default="logs")
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--resume",      default=None)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="auto")
    p.add_argument("--n_eval_episodes", type=int, default=5)
    p.add_argument("--wandb_project",   default="mining_compare")
    p.add_argument("--wandb_run",       default=None)
    args = p.parse_args()

    match args.mode:
        case "train":
            train(args)
        case "eval":
            assert args.resume, "--resume 경로를 지정하세요"
            evaluate(args)
