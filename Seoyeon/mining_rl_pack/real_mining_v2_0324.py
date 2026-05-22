"""
real_mining.py — CraftGround 지하 광물 채굴 강화학습 (PPO)

의존: cave_seed_scanner.py (CaveSpawnWrapper), cave_db.json (시드 DB)

실행:
  # 훈련
  python mining_rl.py --mode train --db cave_db.json --total_steps 3000000

  # 평가
  python mining_rl.py --mode eval --resume checkpoints/mining_safe_XXXX/best/best_model

보상 구조 (Hierarchical 3-Phase, Multiplicative Gating):
  Phase 1 (Descend)  게이트 없음      — 깊이 보상 집중 (Y-shaping, delta, milestones)
  Phase 2 (Explore)  depth_gate       — 깊어야 탐험 보상 활성화
  Phase 3 (Mine)     depth×explore    — 깊이+탐험 후 채굴 보상 극대화

  상위 Phase 달성도가 하위 Phase 보상의 가중치(gate)로 작용.
  sigmoid 전환으로 부드러운 커리큘럼 형성.
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

from cave_seed_scanner_v1 import CaveSpawnWrapper
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
            "time set 0",  # 밝게 고정 (시각화 디버그용)
        ) + _COMMON,
        max_episode_steps=6000,   # ↓ 10000→6000: 짧은 에피소드로 집중적 행동 유도
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

# 인벤토리 초기화
# 순서 중요:
#   1) clear @p          → 이전 에피소드 아이템 제거 (false reward 방지)
#   2) give diamond_pickaxe → hotbar 0~4 채움 (빈 인벤토리 기준 순서대로)
#   3) enchant           → 현재 손 아이템(slot 0) 에 적용
#   4) 나머지 아이템
INVENTORY_CMDS: tuple[str, ...] = (
    "clear @p",                                  # ★ 필수: 이전 에피소드 잔존 아이템 제거
    "give @p minecraft:diamond_pickaxe 5",       # hotbar 0-4: 다이아 곡괭이
    "enchant @p efficiency 5",                   # slot 0 곡괭이에 효율 V
    "enchant @p fortune 3",                      # slot 0 곡괭이에 행운 III
    "enchant @p unbreaking 3",                   # slot 0 곡괭이에 내구성 III
    #"give @p minecraft:torch 64",                # 횃불
    #"give @p minecraft:cooked_beef 64",          # 음식 (체력 회복)
    "effect give @p minecraft:night_vision 999999 0 true",  # ★ 야간 투시 (영구, 파티클 숨김)
)


# =============================================================
# 2. 광물 상수
# =============================================================

# raycast 감지용 ore 블록 ID
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

# Layer 3: 인벤토리에 생기는 아이템 → 보상
# 다이아 곡괭이 + Fortune III 기준 예상 드롭:
#   diamond:     1–4개 (Fortune III 효과)
#   redstone:    4–9개 (Fortune III 효과)
#   lapis:       4–28개 (Fortune III 효과)
#   raw_iron/gold/copper: Fortune 미적용 (원광석은 Fortune 영향 없음)
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
# ↑ 스케일 다운 이유:
#   이전: 다이아 4개 드롭 → +400 (탐험 보상 ~30의 13배 → 운에 지배됨)
#   현재: 다이아 4개 드롭 → log 압축 적용 후 ~30 (탐험 보상과 동일 스케일)

# raycast 조준 중 블록 가치 (is_ore obs용, 정규화 기준 100)
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
# 3. 액션 공간
# =============================================================

# 14개 액션 — 채굴 태스크에 필요한 것만
ACTION_NAMES: list[str] = [
    "NO_OP",                 # 0
    "FORWARD",               # 1
    "BACKWARD",              # 2
    "LEFT",                  # 3
    "RIGHT",                 # 4
    "JUMP",                  # 5
    "ATTACK",                # 6  — 블록 파기 (핵심)
    "CAMERA_LEFT",           # 7
    "CAMERA_RIGHT",          # 8
    "CAMERA_UP",             # 9
    "CAMERA_DOWN",           # 10
    "ATTACK_FORWARD",        # 11 — 매크로: 전진하며 채굴
    "ATTACK_DOWN",           # 12 — 매크로: 아래 보며 채굴
    "STAIRCASE_DOWN",        # 13 — 매크로: 계단식 하강 (전진+아래보기+채굴)
]

CAM_DEG = 8.0  # 카메라 회전 1스텝 각도
ATTACK_HOLD_TICKS = 8   # 1회 ATTACK 시 유지 틱 수 (효율V 기준 돌 1-3틱, 여유 포함)

def build_action(name: str) -> dict:
    """액션 이름 → no_op_v2() raw dict 변환. 매크로는 step()에서 처리."""
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
        # 매크로: 여기서는 no-op 반환 → step()에서 별도 처리
    return act


# =============================================================
# 4. 보상 상수 (Hierarchical 3-Phase)
# =============================================================

Y_SURFACE          =  64      # 지표면 기준 Y
Y_TARGET           = -58      # 다이아몬드 피크 Y (1.18+)
Y_RANGE            = float(Y_SURFACE - Y_TARGET)  # 122.0

# ── 항상 활성 (Phase 무관) ──────────────────────────────────
STEP_PENALTY       = -0.02    # 매 스텝 패널티
DEATH_PENALTY      = -30.0    # 사망 패널티
HEALTH_LOSS_K      =  -0.2    # 체력 피해 스텝당 패널티 계수

# ── Phase 1: Descend (게이트 없음) ──────────────────────────
LAYER1_MAX         =   0.15   # Y-level shaping 최대 보상/스텝
Y_DELTA_BONUS      =   0.5    # Y가 이전 최저점보다 낮아질 때 보상

# 깊이 마일스톤 보너스: (Y 임계값, 보상) — 해당 Y 이하 도달 시 1회 지급
DEPTH_MILESTONES: list[tuple[float, float]] = [
    (40.0,   2.0),   # 지하 진입
    ( 0.0,   5.0),   # Y=0 도달
    (-20.0, 10.0),   # 깊은 지하
    (-40.0, 15.0),   # 딥슬레이트 층
    (-58.0, 25.0),   # 다이아몬드 피크 — 최대 보상
]

# ── Phase 2: Explore (depth_gate 게이팅) ────────────────────
BLOCK_BREAK_BONUS  =  0.15    # 블록 파괴 보상 (깊은 곳에서 터널링 유도)
LAYER2_BASE        =   0.50   # 새 셀 탐험 기본 보너스
EXPLORE_CELL_SIZE  =   2      # 탐험 셀 크기 (2블록 단위)
LAMBDA_DECAY_STEPS = 1_500_000  # 이 스텝(per-env) 동안 λ: 1.0 → 0.1

# depth_gate sigmoid 파라미터: sigmoid((Y_SURFACE - y - center) / scale)
#   center=30 → Y≈34에서 gate=0.5, Y≈0에서 gate≈0.95
#   scale=10  → 전환 폭 ~20블록 (부드러운 전환)
DEPTH_GATE_CENTER  =  30.0
DEPTH_GATE_SCALE   =  10.0

# ── Phase 3: Mine (depth_gate × explore_gate 게이팅) ────────
ORE_AIM_BONUS        = 0.3    # 광물 블록 조준 보상
ORE_AIM_ATTACK_BONUS = 1.0    # 광물 블록 조준+공격 보상
EXPLORE_GATE_THRESHOLD = 50   # 깊은 곳(Y<0) 탐험 셀 수 — 이만큼 탐험해야 gate=1.0

REWARD_CLIP = (-30.0, 60.0)


# =============================================================
# 5. 헬퍼 (hollow_box_rl_0314.py 와 동일 패턴)
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
    """translation_key 정규화: 'item.minecraft.diamond' → 'minecraft:diamond' 등."""
    if tk.startswith("item.") or tk.startswith("block."):
        parts = tk.split(".", 2)  # e.g. ["item", "minecraft", "diamond"]
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
    """hollow_box_rl_0314.py 의 _extract_image 와 동일."""
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
# 5b. 게이트 함수
# =============================================================

def _sigmoid(x: float) -> float:
    """수치 안정 sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def depth_gate(y: float) -> float:
    """Phase 2 게이트: 깊을수록 1에 가까움.
    Y=64(지표) → ~0.05,  Y=34 → 0.50,  Y=0 → ~0.97,  Y=-58 → ~1.00
    """
    return _sigmoid((Y_SURFACE - y - DEPTH_GATE_CENTER) / DEPTH_GATE_SCALE)


def explore_gate(deep_visited_count: int) -> float:
    """Phase 3 게이트: 깊은 곳(Y<0) 탐험량에 비례, 최대 1.0.
    0셀 → 0.0,  50셀 → 1.0 (선형 클램프)
    """
    return min(1.0, deep_visited_count / EXPLORE_GATE_THRESHOLD)


# =============================================================
# 6. MiningWrapper
# =============================================================

class MiningWrapper(gym.Wrapper):
    """
    광물 채굴 RL 환경 래퍼 (Hierarchical Reward).

    Obs: {
        "image":  (H, W, 3) uint8   — 1인칭 시점 이미지
        "state":  (12,) float32     — 깊이·체력·방향·λ·광물조준·게이트값
    }
    Act: Discrete(14)

    계층적 보상 설계:
      Phase 1 (Descend): Y-shaping, delta, milestones → 항상 활성
      Phase 2 (Explore): 탐험 + 블록파괴 → depth_gate 게이팅
      Phase 3 (Mine):    광물 조준/채굴 → depth_gate × explore_gate 게이팅
    """

    H, W   = 64, 114   # 관찰 이미지 크기 (hollow_box 와 동일)
    RH, RW = 180, 320  # 렌더링 표시 크기

    def __init__(
        self,
        env,
        cfg: ModeConfig,
        max_steps: int,
        mode_name: str = "",
    ):
        super().__init__(env)
        self.cfg       = cfg
        self.max_steps = max_steps
        self.mode_name = mode_name

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (self.H, self.W, 3), dtype=np.uint8),
            # state: [-2, 2] 범위 — y/64 는 Y=128에서 2.0, Y=-64에서 -1.0
            "state": spaces.Box(-2.0, 2.0, (12,), dtype=np.float32),
        })

        # ── 에피소드 상태 ─────────────────────────────────────────
        self._step         = 0
        self._ep_r         = 0.0
        # _total_steps: 전 에피소드 누적 (리셋해도 유지) → λ 감쇠 기준
        self._total_steps  = 0
        self._visited: set[tuple[int, int, int]] = set()
        self._deep_visited_count = 0   # Y<0 셀 탐험 수 (explore_gate 기준)
        self._prev_inv: dict[str, int] = {}
        self._prev_health  = 20.0
        self._min_y        = Y_SURFACE  # 에피소드 내 최저 Y (Y-delta 보상 기준)
        self._milestone_reached: set[float] = set()  # 달성한 깊이 마일스톤
        self.lambda_intrinsic = 1.0
        # 게이트 값 (obs에 포함, 로깅용으로도 사용)
        self._depth_gate   = 0.0
        self._explore_gate = 0.0

    # ── λ 내부 계산 ───────────────────────────────────────────────
    def _current_lambda(self) -> float:
        """per-env 스텝 기반 선형 감쇠. λ: 1.0 → 0.1 (LAMBDA_DECAY_STEPS 동안)."""
        return float(max(0.1, 1.0 - self._total_steps / LAMBDA_DECAY_STEPS))

    # ── reset ────────────────────────────────────────────────────
    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)

        # 물리 안정화: tp 직후 에이전트가 허공에 있을 수 있음
        # 5 tick 동안 no-op → 낙하 후 지면 착지 보장
        noop = no_op_v2()
        for _ in range(5):
            try:
                raw, *_ = self.env.step(noop)
            except Exception:
                break

        self._step         = 0
        self._ep_r         = 0.0
        self._visited      = set()
        self._deep_visited_count = 0
        self._prev_health  = _scalar(raw, "health", 20.0)
        self._min_y        = _scalar(raw, "y")     # 스폰 Y 기준
        self._milestone_reached = set()
        self._prev_inv     = _get_inv_counts(raw)
        self.lambda_intrinsic = self._current_lambda()
        self._depth_gate   = depth_gate(_scalar(raw, "y"))
        self._explore_gate = 0.0
        self._ep_ore_counts: dict[str, int] = {k: 0 for k in ORE_DROP_REWARDS}

        obs = self._make_obs(raw)
        info["render_frame"] = self._render(_extract_image(raw))
        return obs, info

    # ── step ─────────────────────────────────────────────────────
    def step(self, action: int):
        name = ACTION_NAMES[int(action)]
        terminated = truncated = False

        if name in ("ATTACK", "ATTACK_FORWARD", "ATTACK_DOWN", "STAIRCASE_DOWN"):
            # ── Phase 1: 위치/시선 잡기 ────────────────────────────
            _dug_down = False       # 아래를 팠는지 여부 (수거 방향 결정용)
            _pitch_offset = 0.0     # Phase 1에서 누적된 시선 변화량 (복원용)

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
                _pitch_offset = CAM_DEG * 2   # 2틱 × 8° = 16°
                _dug_down = True
            elif name == "STAIRCASE_DOWN":
                fwd = no_op_v2(); fwd["forward"] = True
                for _ in range(3):
                    raw, _, terminated, truncated, info = self.env.step(fwd)
                    if terminated or truncated:
                        break
                if not (terminated or truncated):
                    dn = no_op_v2(); dn["camera_pitch"] = CAM_DEG * 4
                    raw, _, terminated, truncated, info = self.env.step(dn)
                    _pitch_offset = CAM_DEG * 4   # 1틱 × 32° = 32°
                _dug_down = True

            # ── Phase 2: attack 유지 → 블록 완파 ───────────────────
            if not (terminated or truncated):
                atk = no_op_v2(); atk["attack"] = True
                for _ in range(ATTACK_HOLD_TICKS):
                    raw, _, terminated, truncated, info = self.env.step(atk)
                    if terminated or truncated:
                        break

            # ── Phase 3: 드롭 아이템 수거 ──────────────────────────
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

            # ── Phase 4: 시선 복원 ─────────────────────────────────
            # Phase 1에서 아래로 내린 시선을 같은 양만큼 위로 되돌린다
            if not (terminated or truncated) and _pitch_offset > 0.0:
                restore = no_op_v2()
                restore["camera_pitch"] = -_pitch_offset  # 음수 = 위로
                raw, _, terminated, truncated, info = self.env.step(restore)

            is_attack = True

        else:
            raw, _, terminated, truncated, info = self.env.step(build_action(name))
            is_attack = False

        self._step        += 1
        self._total_steps += 1
        truncated = truncated or (self._step >= self.max_steps)

        # ── 사망 감지 → 인벤토리 복구 ───────────────────────────
        # doImmediateRespawn=true 이므로 사망 시 자동 리스폰되지만
        # 인벤토리는 사라짐 → 다이아 곡괭이 등을 다시 지급
        health = _scalar(raw, "health", 20.0)
        if health <= 0.0 or (self._prev_health > 0.0 and health >= 20.0
                             and self._prev_health < 2.0):
            # health==0 (사망 직후) 또는 갑자기 풀피 복구(리스폰 완료) 감지
            raw_env = self.env
            while hasattr(raw_env, "env"):
                raw_env = raw_env.env
            if hasattr(raw_env, "add_commands"):
                raw_env.add_commands(list(INVENTORY_CMDS))
                print("  [RESPAWN] 인벤토리 복구 커맨드 전송")

        curr_inv = _get_inv_counts(raw)
        obs      = self._make_obs(raw)
        reward   = self._compute_reward(raw, terminated or truncated, is_attack, curr_inv)
        self._ep_r    += reward

        # ── 인벤토리 변화 디버그 (새 아이템 감지 시 출력) ─────────
        for key, count in curr_inv.items():
            prev = self._prev_inv.get(key, 0)
            if count > prev:
                matched = key in ORE_DROP_REWARDS
                tag = "ORE ✓" if matched else "----"
                print(f"  [INV] +{count - prev} {key} (total={count}) [{tag}]")

        # 광물 채굴 카운트 누적
        for item_key in ORE_DROP_REWARDS:
            gained = max(0, curr_inv.get(item_key, 0) - self._prev_inv.get(item_key, 0))
            if gained > 0:
                self._ep_ore_counts[item_key] += gained

        self._prev_inv = curr_inv
        self._prev_health = _scalar(raw, "health", 20.0)

        info.update({
            "episode_step":   self._step,
            "episode_reward": self._ep_r,
            "lambda":         self.lambda_intrinsic,
            "y_level":        _scalar(raw, "y"),
            "depth_gate":     self._depth_gate,
            "explore_gate":   self._explore_gate,
            "render_frame":   self._render(_extract_image(raw)),
            "ore_counts":     self._ep_ore_counts.copy(),
        })

        # ── 에피소드 종료 시 광물 채굴 요약 출력 ──────────────────
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

        # 광물 조준 감지 (raycast)
        is_ore, ore_value_norm = 0.0, 0.0
        hit = _get_hit(raw)
        if hit is not None and _hit_type(hit) == "block":
            block_id = _strip_state(_hit_state(hit))
            if block_id in ORE_BLOCK_IDS:
                is_ore        = 1.0
                ore_value_norm = _ORE_BLOCK_VALUE.get(block_id, 0.0) / 100.0

        # 게이트 값 갱신 (보상 계산 전에 최신화)
        self._depth_gate   = depth_gate(y)
        self._explore_gate = explore_gate(self._deep_visited_count)

        state = np.array([
            float(np.clip(y / 64.0, -2.0, 2.0)),        # [0] Y 위치
            depth_progress,                               # [1] 목표 깊이 진행도
            float(np.clip(hp   / 20.0, 0.0, 1.0)),       # [2] 체력
            float(np.clip(food / 20.0, 0.0, 1.0)),       # [3] 배고픔
            float(np.sin(np.radians(yaw))),               # [4] 방향 sin
            float(np.cos(np.radians(yaw))),               # [5] 방향 cos
            float(np.clip(pitch / 90.0, -1.0, 1.0)),     # [6] 피치
            self.lambda_intrinsic,                        # [7] 현재 탐험 가중치
            is_ore,                                       # [8] 광물 조준 여부
            ore_value_norm,                               # [9] 조준 광물 가치
            float(self._depth_gate),                      # [10] depth gate 값
            float(self._explore_gate),                    # [11] explore gate 값
        ], dtype=np.float32)

        return {"image": img, "state": state}

    # ── reward ───────────────────────────────────────────────────
    def _compute_reward(
        self,
        raw,
        done: bool,
        is_attack: bool,
        curr_inv: dict[str, int],
    ) -> float:
        r = 0.0
        y = _scalar(raw, "y")

        # ══════════════════════════════════════════════════════════
        # 항상 활성 (Phase 무관)
        # ══════════════════════════════════════════════════════════

        r += STEP_PENALTY

        health = _scalar(raw, "health", 20.0)
        if health < self._prev_health:
            r += HEALTH_LOSS_K * (self._prev_health - health)

        if done and health <= 0.0:
            r += DEATH_PENALTY

        # ══════════════════════════════════════════════════════════
        # Phase 1: Descend (게이트 없음 — 항상 활성)
        # ══════════════════════════════════════════════════════════

        depth_progress = float(np.clip((Y_SURFACE - y) / Y_RANGE, 0.0, 1.0))
        r += LAYER1_MAX * depth_progress

        if y < self._min_y:
            r += Y_DELTA_BONUS * (self._min_y - y)
            self._min_y = y

        for threshold, bonus in DEPTH_MILESTONES:
            if y <= threshold and threshold not in self._milestone_reached:
                self._milestone_reached.add(threshold)
                r += bonus

        # ══════════════════════════════════════════════════════════
        # Phase 2: Explore (depth_gate 게이팅)
        #   gate: 지표 ~0.05, Y=34 → 0.5, Y<0 → ~1.0
        # ══════════════════════════════════════════════════════════
        dg = self._depth_gate  # _make_obs에서 이미 갱신됨

        self.lambda_intrinsic = self._current_lambda()
        x = _scalar(raw, "x")
        z = _scalar(raw, "z")
        cs = EXPLORE_CELL_SIZE
        cell = (int(x) // cs, int(y) // cs, int(z) // cs)
        if cell not in self._visited:
            self._visited.add(cell)
            # 깊은 곳(Y<0) 탐험 셀 카운트 → explore_gate 기준
            if y < 0:
                self._deep_visited_count += 1
                self._explore_gate = explore_gate(self._deep_visited_count)
            depth_weight = 1.0 + depth_progress
            r += dg * self.lambda_intrinsic * LAYER2_BASE * depth_weight

        prev_total = sum(self._prev_inv.values())
        curr_total = sum(curr_inv.values())
        if curr_total > prev_total:
            r += dg * BLOCK_BREAK_BONUS

        # ══════════════════════════════════════════════════════════
        # Phase 3: Mine (depth_gate × explore_gate 게이팅)
        #   gate: 깊이+탐험 두 조건 모두 충족해야 보상 극대화
        # ══════════════════════════════════════════════════════════
        mine_gate = dg * self._explore_gate

        hit = _get_hit(raw)
        if hit is not None and _hit_type(hit) == "block":
            block_id = _strip_state(_hit_state(hit))
            if block_id in ORE_BLOCK_IDS:
                ore_val = _ORE_BLOCK_VALUE.get(block_id, 1.0) / 100.0
                r += mine_gate * ORE_AIM_BONUS * ore_val
                if is_attack:
                    r += mine_gate * ORE_AIM_ATTACK_BONUS * ore_val

        for item_key, reward_per_unit in ORE_DROP_REWARDS.items():
            prev_count = self._prev_inv.get(item_key, 0)
            curr_count = curr_inv.get(item_key, 0)
            gained     = max(0, curr_count - prev_count)
            if gained > 0:
                r += mine_gate * reward_per_unit * float(np.log1p(gained))

        return float(np.clip(r, *REWARD_CLIP))

    # ── 에피소드 종료 광물 요약 ─────────────────────────────────
    _ORE_DISPLAY = {
        "minecraft:diamond":      ("Diamond",  "\033[96m"),   # cyan
        "minecraft:emerald":      ("Emerald",  "\033[92m"),   # green
        "minecraft:raw_gold":     ("Gold",     "\033[93m"),   # yellow
        "minecraft:raw_iron":     ("Iron",     "\033[37m"),   # white
        "minecraft:redstone":     ("Redstone", "\033[91m"),   # red
        "minecraft:lapis_lazuli": ("Lapis",    "\033[94m"),   # blue
        "minecraft:raw_copper":   ("Copper",   "\033[33m"),   # dark yellow
        "minecraft:coal":         ("Coal",     "\033[90m"),   # gray
    }
    _RST = "\033[0m"
    _BAR_WIDTH = 20

    def _print_ore_summary(self):
        total_mined = sum(self._ep_ore_counts.values())
        max_count = max(self._ep_ore_counts.values()) if total_mined > 0 else 1

        print(f"\n{'='*60}")
        print(f"  Episode Summary  |  Steps: {self._step}  "
              f"R: {self._ep_r:+.1f}  Y: {_scalar(self.env, 'y'):+.0f}")
        print(f"  depth_gate: {self._depth_gate:.3f}  "
              f"explore_gate: {self._explore_gate:.3f}  "
              f"deep_cells: {self._deep_visited_count}")
        print(f"{'='*60}")

        for item_key, (name, color) in self._ORE_DISPLAY.items():
            count = self._ep_ore_counts.get(item_key, 0)
            bar_len = int(self._BAR_WIDTH * count / max_count) if max_count > 0 else 0
            bar = "\u2588" * bar_len + "\u2591" * (self._BAR_WIDTH - bar_len)
            print(f"  {color}{name:>8s}{self._RST} |{color}{bar}{self._RST}| {count:3d}")

        print(f"{'─'*50}")
        print(f"  {'Total':>8s} | {total_mined:3d} items")
        print(f"{'='*50}\n")

    # ── render ───────────────────────────────────────────────────
    def _render(self, img_rgb: np.ndarray | None) -> np.ndarray:
        if img_rgb is None:
            frame = np.zeros((self.RH, self.RW, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(
                cv2.resize(img_rgb, (self.RW, self.RH), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_RGB2BGR,
            )
        lam = f"{self.lambda_intrinsic:.2f}"
        dg  = f"{self._depth_gate:.2f}"
        eg  = f"{self._explore_gate:.2f}"
        lines = [
            f"Step:{self._step:4d}  R:{self._ep_r:+7.1f}  lam:{lam}",
            f"DG:{dg} EG:{eg}  Mode:{self.mode_name.upper()}",
        ]
        for i, txt in enumerate(lines):
            ypos = 18 + i * 20
            cv2.putText(frame, txt, (5, ypos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),       2, cv2.LINE_AA)
            cv2.putText(frame, txt, (5, ypos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        return frame


# =============================================================
# 7. Feature Extractor (CNN + MLP)
# =============================================================

class MiningCNNExtractor(BaseFeaturesExtractor):
    """
    CNN(image 64×114) + MLP(state 10-dim) → fused 256-dim features

    CNN 출력 차원:
      Conv(3→32, k=5, s=2): (30, 55)
      Conv(32→64, k=5, s=2): (13, 26)
      Conv(64→64, k=3, s=1): (11, 24)
      Flatten: 64 × 11 × 24 = 16896
    """

    def __init__(self, obs_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(obs_space, features_dim)

        img_shape = obs_space["image"].shape
        if img_shape[-1] in (1, 3, 4):
            h, w, c = img_shape       # (H, W, C) — SB3 표준
        else:
            c, h, w = img_shape       # (C, H, W) 폴백

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # 실제 CNN 출력 차원 자동 계산 (수동 계산 오류 방지)
        with torch.no_grad():
            _dummy = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn(_dummy).shape[1]  # 16896

        state_dim = obs_space["state"].shape[0]  # 12
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim), nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        img_raw = obs["image"].float()
        # (B, H, W, C) → (B, C, H, W)
        if img_raw.shape[-1] in (1, 3, 4):
            img = img_raw.permute(0, 3, 1, 2) / 255.0
        else:
            img = img_raw / 255.0
        state = obs["state"].float()
        return self.fusion(
            torch.cat([self.cnn(img), self.state_mlp(state)], dim=-1)
        )


# =============================================================
# 8. 하이퍼파라미터
# =============================================================

HP: dict[str, dict] = {
    "safe": dict(
        learning_rate=3e-4,
        n_steps=1024,    # ↑ 512→1024: 더 긴 trajectory로 마일스톤 보상 학습
        batch_size=128,  # ↑ 64→128: n_steps 증가에 비례
        n_epochs=10,
        gamma=0.995,     # ↑ 0.99→0.995: 마일스톤/채굴 같은 먼 보상에 더 민감
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,   # ↑ 0.02→0.03: 새 매크로 포함 14개 액션 탐색 강화
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
    "survival": dict(
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,     # 긴 에피소드 → 감가율 높게
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
}


# =============================================================
# 9. 콜백
# =============================================================

class MiningTrackingCallback(BaseCallback):
    """채굴 통계 (광물 수집량, 평균 깊이, λ) 로깅."""

    # 광물 이름 축약 (wandb 패널 가독성)
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
        # 에피소드 완료 시 광물 카운트 기록
        self._ep_ore_buf: list[dict[str, int]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._buf.append({
                "y":     info.get("y_level",        0.0),
                "r":     info.get("episode_reward", 0.0),
                "lam":   info.get("lambda",         1.0),
                "dg":    info.get("depth_gate",     0.0),
                "eg":    info.get("explore_gate",   0.0),
            })
            if len(self._buf) > 2000:
                del self._buf[:1000]

            # 에피소드 종료 시 광물 카운트 저장
            if "episode" in info and "ore_counts" in info:
                self._ep_ore_buf.append(info["ore_counts"])
                if len(self._ep_ore_buf) > 100:
                    del self._ep_ore_buf[:50]

        if self.num_timesteps % self.log_freq == 0 and self._buf:
            self._log()
        return True

    def _log(self):
        recent = self._buf[-400:]
        avg_y   = float(np.mean([e["y"]   for e in recent]))
        avg_lam = float(np.mean([e["lam"] for e in recent]))
        avg_dg  = float(np.mean([e["dg"]  for e in recent]))
        avg_eg  = float(np.mean([e["eg"]  for e in recent]))

        ep_buf  = self.model.ep_info_buffer
        if ep_buf:
            mean_r = float(np.mean([e["r"] for e in ep_buf]))
            mean_l = float(np.mean([e["l"] for e in ep_buf]))
            self.logger.record("mining/mean_ep_reward", mean_r)
            self.logger.record("mining/mean_ep_length", mean_l)

        self.logger.record("mining/avg_y_level",    avg_y)
        self.logger.record("mining/lambda",          avg_lam)
        self.logger.record("mining/depth_gate",      avg_dg)
        self.logger.record("mining/explore_gate",    avg_eg)

        ore_log = {}
        if self._ep_ore_buf:
            for full_key, short in self._ORE_SHORT.items():
                vals = [ep.get(full_key, 0) for ep in self._ep_ore_buf]
                mean_val = float(np.mean(vals))
                self.logger.record(f"ore/{short}", mean_val)
                ore_log[f"ore/{short}"] = mean_val

        if _wactive():
            _wlog({
                "mining/avg_y_level":  avg_y,
                "mining/lambda":       avg_lam,
                "mining/depth_gate":   avg_dg,
                "mining/explore_gate": avg_eg,
                **({"mining/mean_ep_reward": mean_r,
                    "mining/mean_ep_length": mean_l}
                   if ep_buf else {}),
                **ore_log,
            }, step=self.num_timesteps)

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"  [{self.num_timesteps:>8,} steps]  "
                  f"avg_y={avg_y:+.1f}  lam={avg_lam:.3f}  "
                  f"DG={avg_dg:.3f}  EG={avg_eg:.3f}")
            if ep_buf:
                print(f"  ep_reward={mean_r:+.1f}  ep_length={mean_l:.0f}")
            if ore_log:
                ore_strs = [f"{k.split('/')[-1]}={v:.1f}" for k, v in ore_log.items()]
                print(f"  ore: {' | '.join(ore_strs)}")
            print(f"{'─'*60}")


class RenderCallback(BaseCallback):
    """학습 중 cv2 창에 실시간 렌더링 + 선택적 영상 녹화."""

    WIN = "Mining RL"

    def __init__(self, freq: int = 4, record_path: str | None = None, fps: float = 10.0):
        """
        freq        : 몇 스텝마다 프레임 표시/녹화할지
        record_path : 저장할 mp4 경로 (None이면 녹화 안 함). 예) "runs/ep_record.mp4"
        fps         : 녹화 영상 fps
        """
        super().__init__()
        self.freq         = freq
        self.record_path  = record_path
        self.fps          = fps
        self._active      = False
        self._tick        = 0
        self._writer: cv2.VideoWriter | None = None

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
            frame = np.zeros((MiningWrapper.RH, MiningWrapper.RW, 3), dtype=np.uint8)

        # VideoWriter 초기화 (첫 프레임 기준으로 크기 확정)
        if self.record_path and self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            Path(self.record_path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(self.record_path, fourcc, self.fps, (w, h))
            print(f"[RenderCallback] 녹화 시작: {self.record_path}  ({w}×{h} @ {self.fps}fps)")

        # BGR 변환 후 저장
        if self._writer is not None:
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        try:
            cv2.imshow(self.WIN, frame)
        except Exception:
            self._active = False
            return True
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._active = False
            cv2.destroyWindow(self.WIN)
        return True

    def _on_training_end(self):
        if self._active:
            cv2.destroyWindow(self.WIN)
        if self._writer is not None:
            self._writer.release()
            print(f"[RenderCallback] 녹화 저장 완료: {self.record_path}")


# =============================================================
# 10. 환경 생성
# =============================================================

def make_env(
    port:      int  = 8030,
    mode:      str  = "safe",
    max_steps: int | None = None,
    seed:      int  = 42,
    db_path:   str | None = None,
) -> gym.Env:
    """
    MiningWrapper + CaveSpawnWrapper 환경 생성.

    핵심 패턴:
      cmds 리스트를 make()와 CaveSpawnWrapper 양쪽에 동일 객체로 전달.
      CaveSpawnWrapper.reset() 에서 cmds[-1] = "tp @p x y z" 로 in-place 패치.
      CraftGround는 reset()마다 initial_extra_commands 리스트를 재실행하므로
      서버 재시작 없이 매 에피소드 다른 동굴 좌표로 스폰 가능.
    
    주의: n_envs > 1 시 make_fn 람다로 각 env가 독립된 cmds 리스트를 가져야 함.
          아래 train()의 make_fn 참고.
    """
    if mode not in MODES:
        raise ValueError(f"mode='{mode}' 불가. 선택: {list(MODES.keys())}")

    cfg = MODES[mode]

    # ★ 반드시 mutable list — CaveSpawnWrapper가 [-1] 인덱스를 패치함
    # INVENTORY_CMDS(tuple) + 모드 커맨드(tuple) → list로 변환
    # 마지막 원소: tp placeholder (CaveSpawnWrapper가 덮어씀)
    cmds: list[str] = (
        list(INVENTORY_CMDS)
        + list(cfg.initial_extra_commands)
        + ["tp @p 0 -45 0"]   # placeholder — 실제 값은 CaveSpawnWrapper가 설정
    )

    raw_env = make(
        initial_env_config=InitialEnvironmentConfig(
            image_width=320,
            image_height=180,
            seed=str(seed),
            world_type=WorldType.DEFAULT,        # ★ SUPERFLAT 아님 — 실제 지형/동굴
            render_distance=6,
            simulation_distance=6,
            hud_hidden=False,
            request_raycast=True,                # 광물 조준 감지용 raycast 필요
            initial_extra_commands=cmds,         # ★ cmds 리스트 참조 전달
        ),
        port=port,
        verbose=False,
        verbose_gradle=True,
        render_action=False,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )

    mining_env = MiningWrapper(
        raw_env,
        cfg=cfg,
        max_steps=max_steps or cfg.max_episode_steps,
        mode_name=mode,
    )

    if db_path and Path(db_path).exists():
        return CaveSpawnWrapper(
            env=mining_env,
            db_path=db_path,
            cmds_list=cmds,               # ★ 동일한 list 객체
            warmup_episodes=5,
            score_weighted_sampling=True,
        )

    print(f"⚠️  cave_db 없음 (db_path={db_path!r}). 기본 스폰(tp @p 0 -45 0) 사용.")
    return mining_env


# =============================================================
# 11. 훈련 / 평가
# =============================================================

def train(args):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = Path(args.log_dir)  / f"mining_{args.env_mode}_{ts}"
    save_dir = Path(args.save_dir) / f"mining_{args.env_mode}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    hp = HP[args.env_mode]

    if args.wandb_project:
        if not _WANDB:
            print("⚠️  wandb 미설치 → pip install wandb")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"mining_ppo_{args.env_mode}_{ts}",
                sync_tensorboard=True,
                save_code=True,
                config={
                    "task":        "mining",
                    "env_mode":    args.env_mode,
                    "total_steps": args.total_steps,
                    "n_envs":      args.n_envs,
                    "seed":        args.seed,
                    "db_path":     args.db,
                    **hp,
                },
            )
            print(f"[WandB] {wandb.run.url}")

    # n_envs > 1: 각 env가 독립된 cmds 리스트와 포트를 사용해야 함
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
            name_prefix="mining",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=3,
            deterministic=True,
            verbose=1,
        ),
        MiningTrackingCallback(log_freq=2000, verbose=1),
        RenderCallback(freq=4, record_path=str(save_dir / "record.avi"), fps=10.0),
    ]

    print(f"\n[Mining RL] {args.env_mode.upper()} | "
          f"{args.total_steps:,} steps | {args.n_envs} envs | "
          f"DB: {args.db}")
    print(f"  보상: Layer1(Y-shape) + Layer2(탐험,λ) + Layer3(광물 채굴)")
    print(f"  장비: 다이아 곡괭이 ×5 (효율V / 행운III / 내구III)\n")

    try:
        model.learn(
            args.total_steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),
        )
    finally:
        final = save_dir / f"mining_{args.env_mode}_final"
        model.save(str(final))
        print(f"\n[저장] {final}.zip")
        if _wactive():
            art = wandb.Artifact(f"ppo_mining_{args.env_mode}_final", type="model")
            art.add_file(str(final.with_suffix(".zip")))
            wandb.run.log_artifact(art)
            wandb.finish()
        vec_env.close()
        eval_env.close()


def evaluate(args):
    env   = make_env(args.base_port, args.env_mode, args.max_steps or None,
                     args.seed, args.db)
    model = PPO.load(args.resume, env=env, device=args.device)

    ore_totals: dict[str, int] = {k: 0 for k in ORE_DROP_REWARDS}
    ep_rewards = []

    for ep in range(args.n_eval_episodes):
        obs, _  = env.reset()
        done    = False
        ep_r    = 0.0
        steps   = 0
        prev_inv: dict[str, int] = {}

        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(act))
            ep_r  += r
            done   = term or trunc
            steps += 1

            # 에피소드 내 광물 수집량 집계
            if hasattr(env, "_prev_inv"):
                curr = getattr(env, "_prev_inv", {})
                for item in ORE_DROP_REWARDS:
                    gained = max(0, curr.get(item, 0) - prev_inv.get(item, 0))
                    ore_totals[item] += gained
                prev_inv = dict(curr)

        ep_rewards.append(ep_r)
        print(f"  Ep {ep+1:2d}: R={ep_r:+.1f}  steps={steps}")

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
    p = argparse.ArgumentParser(description="CraftGround 광물 채굴 PPO")
    p.add_argument("--mode",          default="train",
                   choices=["train", "eval"])
    p.add_argument("--env_mode",      choices=list(MODES.keys()), default="safe")
    _script_dir = str(Path(__file__).resolve().parent)
    p.add_argument("--db",            default=str(Path(_script_dir) / "cave_db.json"),
                   help="cave_seed_scanner.py 로 생성한 DB 경로")
    p.add_argument("--total_steps",   type=int, default=100_000)
    p.add_argument("--n_envs",        type=int, default=1)
    p.add_argument("--base_port",     type=int, default=8030)
    p.add_argument("--max_steps",     type=int, default=0,   help="0=모드 기본값")
    p.add_argument("--log_dir",       default="logs")
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--resume",        default=None)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--device",        default="auto")
    p.add_argument("--n_eval_episodes", type=int, default=5)
    p.add_argument("--wandb_project", default="mining_rl")
    p.add_argument("--wandb_run",     default=None)
    args = p.parse_args()

    match args.mode:
        case "train":
            train(args)
        case "eval":
            assert args.resume, "--resume 경로를 지정하세요"
            evaluate(args)
