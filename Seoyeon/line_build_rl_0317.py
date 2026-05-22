"""
line_build_rl.py — CraftGround 일직선 블록 놓기 강화학습 (PPO)

hollow_box_rl_0314.py 와 동일한 obs space (shape) 사용 → 학습 후 모델 전이 가능

핵심 설계:
  - 액션 10개로 축소 (HOTBAR/JUMP 제거, FORWARD_PLACE 매크로 추가)
  - 에이전트는 yaw=0(+z방향/남쪽)으로 스폰, z축 방향 일직선 놓기
  - 보상: Potential-based shaping (일직선 연장) + 방향 일관성 + 완성 보상

보상 설계 원칙:
  ① Dense signal    → 매 스텝 학습 신호 확보 (바닥 주시, 라인 끝 근접)
  ② 기하학적 제약  → "일자"를 potential-based shaping으로 직접 인코딩
  ③ Reward hacking 방지 → 연속설치 보너스 제거, 완성 보상이 shaping 누적보다 충분히 큼

전이 방법:
  1) python Seoyeon/line_build_rl.py --mode train
  2) python Seoyeon/hollow_box_rl_0314.py --mode train --resume <저장경로>.zip
"""

from __future__ import annotations
import argparse, os, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op, no_op_v2, ActionSpaceVersion

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try: import wandb; _WANDB = True
except ImportError: _WANDB = False

def _wlog(d: dict, step=None):
    try:
        if _WANDB and wandb.run: wandb.log(d, step=step)
    except Exception: pass
def _wactive() -> bool:
    try: return _WANDB and bool(wandb.run)
    except Exception: return False


# =================================================================
# 1. 설정
# =================================================================

LINE_TARGET = 7   # 목표 일직선 길이

@dataclass(frozen=True)
class ModeConfig:
    initial_extra_commands: tuple[str, ...]
    max_episode_steps: int
    give_inventory: bool

_COMMON = (
    "gamerule doWeatherCycle false", "gamerule doImmediateRespawn true",
    "gamerule fallDamage false", "weather clear",
    "gamerule doDaylightCycle false", "gamerule doMobSpawning false",
    "time set 6000",
    "tp @p ~ ~ ~ 0 55",  # yaw=0(+z방향), pitch=55(아래 55도) → 바닥 바로 앞을 봄
)
MODES: dict[str, ModeConfig] = {
    "creative": ModeConfig(
        ("gamemode creative @p",) + _COMMON,
        max_episode_steps=2000,
        give_inventory=True,
    ),
}

INVENTORY_CMDS = (
    "item replace entity @p hotbar.0 with minecraft:oak_planks 64",
    "item replace entity @p hotbar.1 with minecraft:oak_planks 64",
    "item replace entity @p hotbar.2 with minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
)

BUILDING_BLOCKS = frozenset({
    "minecraft:oak_planks",
    "minecraft:oak_log",
    "minecraft:oak_slab",
})
ALL_TARGET_BLOCKS = BUILDING_BLOCKS


# =================================================================
# 2. 마일스톤
# =================================================================
# 설계 원칙:
#   - 마일스톤 총합(18.0) > potential shaping 최대 누적(0.5×6=3.0)
#   - 완성 보상(10.0) >> shaping 누적(3.0) → local optimum 방지
# =================================================================

_MILESTONES = [
    ("block_1",   lambda s: s["total"] >= 1,                  1.0),   # 첫 블록
    ("block_2",   lambda s: s["total"] >= 2,                  0.5),   # 두 번째 블록
    ("line_2",    lambda s: s["longest_line"] >= 2,           1.5),   # 2개 연결
    ("line_3",    lambda s: s["longest_line"] >= 3,           2.0),
    ("line_5",    lambda s: s["longest_line"] >= 5,           3.0),
    ("line_done", lambda s: s["longest_line"] >= LINE_TARGET, 10.0),  # 완성
]
MS_KEYS = [k for k, *_ in _MILESTONES]


# =================================================================
# 3. LineTracker
# =================================================================

class LineTracker:
    """블록 설치 감지 + z축 일직선 추적."""

    WANDER_LIMIT = 8

    def __init__(self): self.reset()

    def reset(self, spawn_y: float = 4.0, spawn_x: float = 0.0, spawn_z: float = 0.0):
        self._placed: dict[tuple, str] = {}
        self._cache: dict = {}
        self._spawn_y = spawn_y
        self._spawn_x = spawn_x
        self._spawn_z = spawn_z
        self.milestones = {k: False for k, *_ in _MILESTONES}

    def update(self, raw_obs, action_was_use: bool, action_was_attack: bool = False) -> tuple[bool, str | None, bool]:
        newly_broken = False
        if action_was_attack:
            hit = _get_hit(raw_obs)
            if hit is not None and _hit_type(hit) == "block":
                pos = _hit_pos(hit)
                if pos and pos in self._placed:
                    del self._placed[pos]
                    self._cache.clear()
                    newly_broken = True
        if not action_was_use: return False, None, newly_broken

        hit = _get_hit(raw_obs)
        if hit is None or _hit_type(hit) != "block": return False, None, newly_broken
        bid = _strip_state(_hit_state(hit))
        target_pos = _hit_pos(hit)
        if not target_pos: return False, None, newly_broken

        # Case 1: 레이캐스트가 새로 놓인 블록을 직접 보여줌
        if bid in ALL_TARGET_BLOCKS and target_pos not in self._placed:
            zone = self._classify(target_pos)
            self._placed[target_pos] = bid
            self._cache.clear()
            return True, zone, newly_broken

        # Case 2: 레이캐스트가 클릭한 블록(바닥/이미 놓은 블록)을 보여줌
        #   → 실제 설치 위치 = target_pos + face_normal
        placed_pos = self._estimate_placed_pos(raw_obs, target_pos)
        if placed_pos and placed_pos not in self._placed:
            zone = self._classify(placed_pos)
            self._placed[placed_pos] = "minecraft:oak_planks"
            self._cache.clear()
            return True, zone, newly_broken

        return False, None, newly_broken

    def _estimate_placed_pos(self, raw_obs, target_pos: tuple) -> tuple | None:
        """플레이어 시선 방향에서 클릭한 면(face)을 추정하여 실제 설치 위치 계산.

        원리: 플레이어 눈 위치 → 블록 중심 벡터에서 가장 큰 축 성분이
              클릭한 면의 법선(normal)이 됨.
              새 블록은 target_pos + normal 위치에 설치됨.
        """
        px = _scalar(raw_obs, "x")
        py = _scalar(raw_obs, "y") + 1.62   # 눈 높이 (발 위치 + 1.62)
        pz = _scalar(raw_obs, "z")
        bx, by, bz = target_pos

        # 플레이어 눈 → 블록 중심까지의 벡터
        dx = px - (bx + 0.5)
        dy = py - (by + 0.5)
        dz = pz - (bz + 0.5)

        adx, ady, adz = abs(dx), abs(dy), abs(dz)

        # 가장 큰 축 = 클릭한 면의 방향
        if ady >= adx and ady >= adz:
            # 위/아래 면 클릭
            return (bx, by + (1 if dy > 0 else -1), bz)
        elif adx >= adz:
            # 동/서 면 클릭
            return (bx + (1 if dx > 0 else -1), by, bz)
        else:
            # 남/북 면 클릭
            return (bx, by, bz + (1 if dz > 0 else -1))

    def _classify(self, pos: tuple) -> str:
        bx, by, bz = pos
        sy = self._spawn_y
        sx = round(self._spawn_x)
        dy = by - sy
        dx = abs(bx - sx)
        if dx > self.WANDER_LIMIT:
            return "out"
        if dy <= 1:
            if dx <= 1:            # x 차이 1 이내면 on_line (허용 범위 확대)
                return "on_line"
            return "ground"
        return "above"

    def is_adjacent_to_placed(self, pos: tuple) -> bool:
        if len(self._placed) <= 1:
            return True
        bx, by, bz = pos
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            if (bx+dx, by+dy, bz+dz) in self._placed:
                return True
        return False

    def _find_longest_line(self) -> int:
        if not self._placed: return 0
        positions = set(self._placed.keys())
        best = 1
        for pos in positions:
            bx, by, bz = pos
            # x축 방향
            length = 1
            nx = bx + 1
            while (nx, by, bz) in positions:
                length += 1; nx += 1
            best = max(best, length)
            # z축 방향
            length = 1
            nz = bz + 1
            while (bx, by, nz) in positions:
                length += 1; nz += 1
            best = max(best, length)
        return best

    def _find_z_line_end(self) -> float | None:
        """z축 라인의 끝(최대 z) 좌표. 없으면 None."""
        sx = round(self._spawn_x)
        max_z = None
        for (bx, by, bz) in self._placed:
            if bx == sx and by - self._spawn_y <= 1:
                if max_z is None or bz > max_z:
                    max_z = bz
        return max_z

    def analyze(self) -> dict:
        if self._cache: return self._cache
        total = len(self._placed)
        longest = self._find_longest_line()
        ground_count = sum(1 for p in self._placed if self._classify(p) in ("ground", "on_line"))
        on_line_count = sum(1 for p in self._placed if self._classify(p) == "on_line")
        z_end = self._find_z_line_end()
        self._cache = {
            "total":          total,
            "longest_line":   longest,
            "ground_blocks":  ground_count,
            "on_line_blocks": on_line_count,
            "line_complete":  longest >= LINE_TARGET,
            "z_line_end":     z_end,
        }
        return self._cache

    def milestone_rewards(self) -> float:
        s = self.analyze(); bonus = 0.0
        for key, fn, val in _MILESTONES:
            if fn(s) and not self.milestones[key]:
                self.milestones[key] = True; bonus += val
        return bonus

    def raycast_vec(self, raw_obs) -> np.ndarray:
        """8-dim 벡터"""
        vec = np.zeros(8, dtype=np.float32)
        hit = _get_hit(raw_obs)
        if hit is None or _hit_type(hit) != "block": return vec
        bid = _strip_state(_hit_state(hit))
        pos = _hit_pos(hit)
        if bid not in ALL_TARGET_BLOCKS: return vec
        vec[0] = 1.0
        zone = self._classify(pos)
        if zone == "on_line":  vec[1] = 1.0
        elif zone == "ground": vec[2] = 1.0
        elif zone == "above":  vec[3] = 1.0
        if pos and self.is_adjacent_to_placed(pos):
            vec[4] = 1.0
        if pos:
            px = _scalar(raw_obs, "x"); py = _scalar(raw_obs, "y"); pz = _scalar(raw_obs, "z")
            vec[5:8] = np.clip([(pos[0]-px)/10., (pos[1]-py)/10., (pos[2]-pz)/10.], -1., 1.)
        return vec

    def struct_vec(self, st: dict) -> np.ndarray:
        """8-dim (hollow_box 호환 shape)"""
        return np.array([
            min(st["total"]          / 15., 1.),
            min(st["longest_line"]   / float(LINE_TARGET), 1.),
            min(st["on_line_blocks"] / float(LINE_TARGET), 1.),
            min(st["ground_blocks"]  / 15., 1.),
            float(st["line_complete"]),
            0.0, 0.0, 0.0,
        ], dtype=np.float32)


# =================================================================
# 4. HitResult 헬퍼
# =================================================================

def _get_full(raw_obs):
    if isinstance(raw_obs, dict): return raw_obs.get("full", raw_obs)
    return raw_obs

def _get_hit(raw_obs):
    full = _get_full(raw_obs)
    return getattr(full, "raycast_result", None)

def _hit_type(hit) -> str:
    raw = getattr(hit, "type", None)
    if raw is None: return "miss"
    return {0: "miss", 1: "block", 2: "entity"}.get(int(raw), "miss")

def _tk_to_block_id(tk: str) -> str:
    if tk.startswith("block."):
        parts = tk[len("block."):].split(".", 1)
        if len(parts) == 2: return f"{parts[0]}:{parts[1]}"
    return tk

def _hit_pos(hit) -> tuple | None:
    tb = getattr(hit, "target_block", None)
    if tb is not None:
        x = getattr(tb, "x", None)
        if x is not None:
            return (int(x), int(getattr(tb, "y", 0)), int(getattr(tb, "z", 0)))
    bp = getattr(hit, "block_pos", None)
    if bp is not None: return (int(bp.x), int(bp.y), int(bp.z))
    return None

def _hit_state(hit) -> str:
    tb = getattr(hit, "target_block", None)
    if tb is not None:
        tk = getattr(tb, "translation_key", "") or ""
        if tk: return _tk_to_block_id(tk)
    for attr in ("block_state", "block_id", "translation_key"):
        v = getattr(hit, attr, None)
        if v: return _tk_to_block_id(str(v))
    return ""

def _strip_state(s: str) -> str: return s.split("[")[0].strip()

def _scalar(obs, key: str, default: float = 0.0) -> float:
    full = _get_full(obs)
    if isinstance(full, dict): return float(full.get(key, default))
    return float(getattr(full, key, default))

_TK_TO_ZONE = {tk: "building" for tk in [
    "block.minecraft.oak_planks",
    "block.minecraft.oak_log",
    "block.minecraft.oak_slab",
]}

def _get_inv_counts(raw_obs) -> dict[str, int]:
    full = _get_full(raw_obs)
    inv  = getattr(full, "inventory", [])
    return {item.translation_key: item.count for item in inv}

def _detect_placed_by_inv(prev_counts: dict, curr_counts: dict) -> bool:
    for tk in _TK_TO_ZONE:
        if curr_counts.get(tk, 0) < prev_counts.get(tk, 0): return True
    return False


# =================================================================
# 5. 액션 — 10개로 축소 + FORWARD_PLACE 매크로
# =================================================================

ACTION_NAMES = [
    "NO_OP",           # 0
    "FORWARD",         # 1  앞으로 이동
    "BACKWARD",        # 2  뒤로 이동
    "LEFT",            # 3  왼쪽 이동
    "RIGHT",           # 4  오른쪽 이동
    "USE",             # 5  블록 설치
    "CAMERA_LEFT",     # 6  카메라 왼쪽
    "CAMERA_RIGHT",    # 7  카메라 오른쪽
    "CAMERA_DOWN",     # 8  카메라 아래
    "FORWARD_PLACE",   # 9  ★ 핵심 매크로: 앞으로 1칸 + 아래 보기 + 블록 설치
]
NUM_ACTIONS = len(ACTION_NAMES)
CAM_DEG = 10.0

def build_action(name: str) -> dict:
    act = no_op_v2()
    match name:
        case "FORWARD":       act["forward"] = True
        case "BACKWARD":      act["back"]    = True
        case "LEFT":          act["left"]     = True
        case "RIGHT":         act["right"]    = True
        case "USE":           act["use"]      = True
        case "CAMERA_LEFT":   act["camera_yaw"]   = -CAM_DEG
        case "CAMERA_RIGHT":  act["camera_yaw"]   =  CAM_DEG
        case "CAMERA_DOWN":   act["camera_pitch"]  =  CAM_DEG
    return act


# =================================================================
# 6. 보상 상수
# =================================================================
#
# 설계 원칙 — reward hacking 방지:
#   ① 매 스텝 공짜 보상 없음 (LOOK_DOWN, NEAR_LINE_END 제거)
#   ② 블록 설치 보상은 on_line일 때만 양수
#   ③ 라인 밖 설치는 반드시 순손실 (ground/above/out 모두 음수)
#   ④ 완성 보상(10.0) >> shaping 최대 누적(0.5×6=3.0) → 끝까지 가야 이득
#
# 보상이 양수가 되는 유일한 경로:
#   on_line 설치(+0.30) + 라인 연장(+0.50) + 마일스톤
#   → "z축 일직선으로 블록을 연장"하는 것만이 이득
# =================================================================

# Potential-based shaping (핵심 — 일직선 연장에만 큰 보상)
LINE_EXTEND_SHAPING  =  1.00   # 일직선 1칸 연장당

# 설치 시 zone별 보상
#   on_line 설치: +0.50 (양수 → 탐색 유도)
#   ground 설치:  +0.05 (아주 작은 양수 → 완전 무시하진 않되, on_line 대비 10배 차이)
#   above 설치:   -0.50 (음수 → 억제)
#   out 설치:     -2.00 (강한 억제)
#
# reward hacking 체크:
#   on_line만 쌓기: +0.50 + 1.00(연장) + 마일스톤 = 매우 이득 ✓
#   ground 무작위: +0.05 × n (마일스톤 없음) = 아주 작은 이득 → on_line이 훨씬 유리 ✓
#   아무것도 안 함: 0.0 → ground보다 약간 불리 → 설치 시도 유도 ✓
ON_LINE_BONUS        =  0.50   # on_line 설치
GROUND_BONUS         =  0.05   # ground 설치 (탐색 최소 유도)
ABOVE_PENALTY        = -0.50   # 위로 쌓기 억제
OUT_PENALTY          = -2.00   # 범위 밖

# 기타 패널티
WANDER_PENALTY       = -0.10   # 범위 초과 거리당
BREAK_PLACED_PENALTY = -2.00   # 이미 설치된 블록 파괴

REWARD_CLIP          = (-10.0, 15.0)
_MODE_COLOR_BGR      = {"creative": (80, 200, 255)}


def _extract_image(obs) -> np.ndarray | None:
    if isinstance(obs, np.ndarray):
        img = obs
    elif isinstance(obs, dict):
        img = obs.get("pov")
        if img is None: img = obs.get("rgb")
        if img is not None:
            img = np.asarray(img, dtype=np.uint8)
        else:
            full = obs.get("full")
            if full is not None and isinstance(getattr(full, "image", None), bytes):
                try:
                    arr = np.frombuffer(full.image, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None: return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception: return None
            else: return None
    elif isinstance(getattr(obs, "image", None), bytes):
        try:
            arr = np.frombuffer(obs.image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception: return None
    else: return None
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img.astype(np.uint8)


def _draw_hud(frame: np.ndarray, st: dict, action: str,
              mode: str, ep_r: float, step: int) -> np.ndarray:
    color = _MODE_COLOR_BGR.get(mode, (200, 200, 200))
    done  = st.get("line_complete", False)
    lines = [
        (f"Step:{step:,}  Rwd:{ep_r:+.1f}  Act:{action}", (220, 220, 220)),
        (f"Blocks:{st.get('total',0)}  OnLine:{st.get('on_line_blocks',0)}  "
         f"Line:{st.get('longest_line',0)}/{LINE_TARGET}",
         (220, 220, 220)),
        (f"LINE: {'DONE!' if done else 'building...'}",
         (50, 255, 80) if done else (180, 180, 180)),
        (f"Mode: {mode.upper()}", color),
    ]
    for i, (txt, c) in enumerate(lines):
        y = 18 + i * 20
        cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c,         1, cv2.LINE_AA)
    return frame


# =================================================================
# 7. 환경 Wrapper
# =================================================================

class LineBuildWrapper(gym.Wrapper):
    """
    일직선 블록 놓기 환경

    Obs: {"image":(64,114,3), "state":(8,), "raycast":(8,), "structure":(8,)}
    Act: Discrete(10)
    """
    H, W   = 64, 114
    RH, RW = 180, 320
    S      = 4

    def __init__(self, env, cfg: ModeConfig, max_steps: int, mode_name: str = ""):
        super().__init__(env)
        self.cfg, self.max_steps, self.mode_name = cfg, max_steps, mode_name
        self.tracker = LineTracker()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "image":     spaces.Box(0,   255, (self.H, self.W, 3), np.uint8),
            "state":     spaces.Box(-1., 1.,  (8,), np.float32),
            "raycast":   spaces.Box(-1., 1.,  (8,), np.float32),
            "structure": spaces.Box(0.,  1.,  (8,), np.float32),
        })
        self._step = 0
        self._ep_r = 0.0
        # NOTE: _consec_place 제거 — 연속 설치 보너스는 reward hacking 위험
        self._prev_inv: dict[str, int] = {}
        self._prev_longest = 0   # Potential-based shaping용 이전 라인 길이

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        self.tracker.reset(spawn_y=_scalar(raw, "y", 4.0),
                           spawn_x=_scalar(raw, "x", 0.0),
                           spawn_z=_scalar(raw, "z", 0.0))
        self._step = 0
        self._ep_r = 0.0
        self._prev_inv = _get_inv_counts(raw)
        self._prev_longest = 0
        raw_img = _extract_image(raw)
        obs = self._make_obs(raw, self.tracker.analyze(), raw_img)
        info["render_frame"] = self._render(raw_img, {}, "")
        return obs, info

    def step(self, action: int):
        name = ACTION_NAMES[int(action)]

        # ★ FORWARD_PLACE 매크로: 앞으로 이동 → 아래 보정 → 설치
        if name == "FORWARD_PLACE":
            fwd = no_op_v2(); fwd["forward"] = True
            for _ in range(2): raw, *_ = self.env.step(fwd)
            pitch_now = _scalar(raw, "pitch")
            if pitch_now < 50:
                look = no_op_v2(); look["camera_pitch"] = min(55 - pitch_now, 15.0)
                raw, *_ = self.env.step(look)
            use = no_op_v2(); use["use"] = True
            raw, _, terminated, truncated, info = self.env.step(use)
        else:
            raw, _, terminated, truncated, info = self.env.step(build_action(name))

        self._step += 1
        truncated = truncated or (self._step >= self.max_steps)

        is_use    = name in ("USE", "FORWARD_PLACE")
        is_attack = False
        placed, zone, broken = self.tracker.update(raw, is_use, is_attack)

        # 인벤토리 감소 기반 보완 감지 — 위치도 추정
        curr_inv = _get_inv_counts(raw)
        if not placed and is_use:
            if _detect_placed_by_inv(self._prev_inv, curr_inv):
                # 인벤토리가 줄었으니 블록은 실제로 놓임 → 위치 추정 시도
                hit = _get_hit(raw)
                target_pos = _hit_pos(hit) if hit else None
                if target_pos:
                    est = self.tracker._estimate_placed_pos(raw, target_pos)
                    if est and est not in self.tracker._placed:
                        self.tracker._placed[est] = "minecraft:oak_planks"
                        self.tracker._cache.clear()
                        zone = self.tracker._classify(est)
                    else:
                        zone = "building"
                else:
                    zone = "building"
                placed = True
        self._prev_inv = curr_inv

        st = self.tracker.analyze()
        raw_img = _extract_image(raw)
        obs = self._make_obs(raw, st, raw_img)
        rew = self._compute_reward(raw, terminated or truncated, placed, zone, name, broken, st)
        self._ep_r += rew

        if st.get("line_complete", False):
            terminated = True

        info.update({
            "structure":    st,
            "milestones":   dict(self.tracker.milestones),
            "action_name":  name,
            "episode_step": self._step,
            "render_frame": self._render(raw_img, st, name),
        })
        return obs, rew, terminated, truncated, info

    def _make_obs(self, raw, st: dict, raw_img: np.ndarray | None = None) -> dict:
        img = raw_img if raw_img is not None else _extract_image(raw)
        img = (cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
               if img is not None else np.zeros((self.H, self.W, 3), dtype=np.uint8))
        yaw   = _scalar(raw, "yaw")
        pitch = _scalar(raw, "pitch")
        state = np.array([
            _scalar(raw, "health",     20.) / 20.,
            _scalar(raw, "food_level", 20.) / 20.,
            _scalar(raw, "x")  / 256.,
            _scalar(raw, "y")  / 256.,
            _scalar(raw, "z")  / 256.,
            np.sin(np.radians(yaw)), np.cos(np.radians(yaw)),
            pitch / 90.,
        ], dtype=np.float32)
        return {
            "image":     img,
            "state":     state,
            "raycast":   self.tracker.raycast_vec(raw),
            "structure": self.tracker.struct_vec(st),
        }

    def _render(self, img_rgb, st, action) -> np.ndarray:
        if img_rgb is None:
            img_rgb = np.zeros((self.RH, self.RW, 3), dtype=np.uint8)
        frame = cv2.cvtColor(
            cv2.resize(img_rgb, (self.RW, self.RH), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_RGB2BGR)
        return _draw_hud(frame, st, action, self.mode_name, self._ep_r, self._step)

    def _compute_reward(self, raw, done: bool, placed: bool, zone: str | None,
                        name: str, broken: bool, st: dict) -> float:
        r = 0.0

        # ── 매 스텝 보상: 없음 (reward hacking 방지) ──────────────────────────
        # LOOK_DOWN, NEAR_LINE_END 등 매 스텝 공짜 보상은 전부 제거.
        # 보상은 "블록을 올바르게 놓았을 때"만 발생.

        # ── 활동반경 패널티 ───────────────────────────────────────────────────
        ax = _scalar(raw, "x")
        az = _scalar(raw, "z")
        dist = max(abs(ax - self.tracker._spawn_x), abs(az - self.tracker._spawn_z))
        if dist > self.tracker.WANDER_LIMIT:
            r += WANDER_PENALTY * (dist - self.tracker.WANDER_LIMIT)

        # ── 블록 파괴 패널티 ──────────────────────────────────────────────────
        if broken:
            r += BREAK_PLACED_PENALTY

        # ── 블록 설치 보상 ────────────────────────────────────────────────────
        if placed and zone is not None:

            # Potential-based shaping: 일직선이 실제로 길어질 때만 보상
            curr_longest = st.get("longest_line", 0)
            if curr_longest > self._prev_longest:
                delta = curr_longest - self._prev_longest
                r += LINE_EXTEND_SHAPING * delta        # +0.50 × 연장된 칸 수
                self._prev_longest = curr_longest

            # Zone별 보상
            if zone == "on_line":
                r += ON_LINE_BONUS                      # +0.50
            elif zone == "ground":
                r += GROUND_BONUS                       # +0.05 (탐색 최소 유도)
            elif zone == "above":
                r += ABOVE_PENALTY                      # -0.50
            elif zone == "out":
                r += OUT_PENALTY                        # -2.00
            # zone == "building" (폴백): 0

        # ── 마일스톤 보상 ───────────────────────────────────────────────────
        r += self.tracker.milestone_rewards()

        return float(np.clip(r, *REWARD_CLIP))


# =================================================================
# 8. 환경 생성
# =================================================================

def make_env(port=8040, mode="creative", max_steps=None, seed=42) -> LineBuildWrapper:
    cfg  = MODES[mode]
    cmds = (list(INVENTORY_CMDS) if cfg.give_inventory else []) + list(cfg.initial_extra_commands)
    env  = make(
        initial_env_config=InitialEnvironmentConfig(
            image_width=320, image_height=180, seed=str(seed),
            world_type=WorldType.SUPERFLAT, render_distance=4, simulation_distance=4,
            hud_hidden=False, initial_extra_commands=cmds,
            request_raycast=True,
        ),
        port=port, verbose=False, verbose_gradle=True, render_action=False,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )
    return LineBuildWrapper(env, cfg=cfg,
                            max_steps=max_steps or cfg.max_episode_steps,
                            mode_name=mode)


# =================================================================
# 9. 모델 (hollow_box 호환 아키텍처)
# =================================================================

class LineBuildCNNExtractor(BaseFeaturesExtractor):
    """CNN(image) + MLP(state+raycast+structure) → features_dim=256"""

    def __init__(self, obs_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(obs_space, features_dim)
        img_shape = obs_space["image"].shape
        if img_shape[-1] in (1, 3, 4):
            h, w, c = img_shape
        else:
            c, h, w = img_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 5, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        vec_dim = sum(obs_space[k].shape[0] for k in ("state", "raycast", "structure"))
        self.mlp    = nn.Sequential(nn.Linear(vec_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(cnn_out + 64, features_dim), nn.ReLU())

    def forward(self, obs: dict) -> torch.Tensor:
        img_raw = obs["image"].float()
        if img_raw.shape[-1] in (1, 3, 4):
            img = img_raw.permute(0, 3, 1, 2) / 255.0
        else:
            img = img_raw
        vec = torch.cat([obs[k].float() for k in ("state", "raycast", "structure")], dim=-1)
        return self.fusion(torch.cat([self.cnn(img), self.mlp(vec)], dim=-1))


# =================================================================
# 10. 하이퍼파라미터
# =================================================================

HP = {
    "creative": dict(
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
}


# =================================================================
# 11. 콜백
# =================================================================

class TrackingCallback(BaseCallback):
    def __init__(self, log_freq=1000, ckpt_freq=20, save_dir=".", mode="creative", verbose=0):
        super().__init__(verbose)
        self.log_freq  = log_freq
        self.ckpt_freq = ckpt_freq
        self.save_dir  = Path(save_dir)
        self.mode      = mode
        self._ridx     = 0
        self._buf: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "structure" in info:
                st = info["structure"]
                self._buf.append({"st": st, "ms": info.get("milestones", {})})
                if len(self._buf) > 1000: del self._buf[:500]
                # 초반 1000스텝: 블록 감지 확인 로그
                if self.num_timesteps <= 1000 and st.get("total", 0) > 0:
                    act = info.get("action_name", "?")
                    print(f"  [t={self.num_timesteps}] act={act}  "
                          f"total={st.get('total',0)} line={st.get('longest_line',0)} "
                          f"on_line={st.get('on_line_blocks',0)}")
        if self.num_timesteps % self.log_freq == 0 and self._buf:
            self._log()
        return True

    def _on_rollout_end(self):
        self._ridx += 1

        # wandb 에피소드 로그 (활성일 때만)
        if _wactive():
            ep_buf = self.model.ep_info_buffer
            if ep_buf:
                _wlog({"rollout/mean_ep_reward": float(np.mean([e["r"] for e in ep_buf])),
                       "rollout/mean_ep_length": float(np.mean([e["l"] for e in ep_buf]))},
                      step=self.num_timesteps)

        # 체크포인트 저장 (wandb 여부와 무관하게 항상 실행)
        if self._ridx % self.ckpt_freq == 0:
            p = self.save_dir / f"ckpt_{self._ridx}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(p))
            if _wactive():
                try:
                    art = wandb.Artifact(f"ppo_line_{self.mode}_ckpt_{self._ridx}", type="model")
                    art.add_file(str(p.with_suffix(".zip")))
                    wandb.run.log_artifact(art)
                except Exception as e:
                    print(f"[WandB] artifact 저장 실패: {e}")

    def _log(self):
        recent = self._buf[-200:]
        line_done = float(np.mean([e["st"].get("line_complete", False) for e in recent]))
        avg_line  = float(np.mean([e["st"].get("longest_line",  0)    for e in recent]))
        blk       = float(np.mean([e["st"].get("total",         0)    for e in recent]))
        on_line   = float(np.mean([e["st"].get("on_line_blocks",0)    for e in recent]))
        ms = {k: float(np.mean([e["ms"].get(k, False) for e in recent])) for k in MS_KEYS}
        for k, v in ms.items(): self.logger.record(f"line/ms_{k}", v)
        self.logger.record("line/complete_rate", line_done)
        self.logger.record("line/avg_line_len",  avg_line)
        self.logger.record("line/avg_blocks",    blk)
        self.logger.record("line/avg_on_line",   on_line)
        if _wactive():
            _wlog({"line/complete_rate": line_done,
                   "line/avg_line_len": avg_line,
                   "line/avg_blocks": blk,
                   "line/avg_on_line": on_line,
                   **{f"line/ms_{k}": v for k, v in ms.items()}},
                  step=self.num_timesteps)
        if self.verbose:
            print(f"\n[{self.num_timesteps:,}] blk={blk:.1f}  line={avg_line:.1f}  "
                  f"on_line={on_line:.1f}  done={line_done:.2f}")


class RenderCallback(BaseCallback):
    WIN = "Line Builder"
    def __init__(self, freq=4):
        super().__init__()
        self.freq = freq; self._active = False; self._tick = 0
    def _on_training_start(self):
        try:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            self._active = True
        except Exception: pass
    def _on_step(self) -> bool:
        if not self._active: return True
        self._tick += 1
        if self._tick % self.freq: return True
        frame = (self.locals.get("infos") or [{}])[0].get("render_frame")
        if frame is None: return True
        try: cv2.imshow(self.WIN, frame)
        except Exception: self._active = False; return True
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._active = False; cv2.destroyWindow(self.WIN)
        return True
    def _on_training_end(self):
        if self._active: cv2.destroyWindow(self.WIN); self._active = False


class VideoRecorderCallback(BaseCallback):
    RW = LineBuildWrapper.RW; RH = LineBuildWrapper.RH
    def __init__(self, make_env_fn, freq=15, video_dir="videos", fps=10):
        super().__init__()
        self._make = make_env_fn; self.freq = freq
        self.dir = Path(video_dir); self.fps = fps
        self._ridx = 0; self._env = None
    def _on_step(self) -> bool: return True
    def _on_rollout_end(self):
        self._ridx += 1
        if self._ridx % self.freq == 0:
            if self._env is None: self._env = self._make()
            self._record()
    def _record(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"ep_{self._ridx:04d}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (self.RW, self.RH))
        obs, _ = self._env.reset(); done = False; total_r = 0.0; info = {}
        try:
            while not done:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self._env.step(int(act))
                total_r += r; done = term or trunc
                if (f := info.get("render_frame")) is not None: writer.write(f)
        finally: writer.release()
        if _wactive():
            try:
                wandb.log({
                    "video/eval_episode": wandb.Video(str(path), fps=self.fps, format="mp4"),
                    "video/eval_reward":  total_r,
                    "video/eval_line":    info.get("structure", {}).get("longest_line", 0),
                }, step=self.num_timesteps)
            except Exception as e:
                print(f"[WandB] video 로그 실패: {e}")
    def _on_training_end(self):
        try:
            if self._env is None: self._env = self._make()
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / "final_episode.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, (self.RW, self.RH))
            obs, _ = self._env.reset(); done = False; total_r = 0.0; info = {}
            try:
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = self._env.step(int(act))
                    total_r += r; done = term or trunc
                    if (f := info.get("render_frame")) is not None: writer.write(f)
            finally: writer.release()
            print(f"\n[Video] 최종 영상: {path}  (R={total_r:+.1f})")
            if _wactive():
                try:
                    wandb.log({
                        "video/final_episode": wandb.Video(str(path), fps=self.fps, format="mp4"),
                        "video/final_reward":  total_r,
                        "video/final_line":    info.get("structure", {}).get("longest_line", 0),
                    })
                except Exception as e:
                    print(f"[WandB] final video 로그 실패: {e}")
        except Exception as e: print(f"최종 영상 저장 실패: {e}")
        finally:
            if self._env: self._env.close()


# =================================================================
# 12. 훈련 / 평가
# =================================================================

def train(args):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = Path(args.log_dir)  / f"line_{args.env_mode}_{ts}"
    save_dir = Path(args.save_dir) / f"line_build_{args.env_mode}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    hp = HP[args.env_mode]

    if args.wandb_project:
        if not _WANDB:
            print("[WandB] wandb 미설치 → pip install wandb")
        else:
            run_name = args.wandb_run or f"line_ppo_{args.env_mode}_{ts}"
            print(f"[WandB] 연결 시도... project={args.wandb_project}, name={run_name}")
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    sync_tensorboard=True,
                    save_code=True,
                    config={"task": "line_build", "line_target": LINE_TARGET,
                            "env_mode": args.env_mode,
                            "total_steps": args.total_steps,
                            "n_envs": args.n_envs, "seed": args.seed,
                            "num_actions": NUM_ACTIONS,
                            "action_names": ACTION_NAMES, **hp},
                )
            except Exception as e:
                print(f"[WandB] init 실패: {e}")
                import traceback; traceback.print_exc()
            if wandb.run:
                print(f"[WandB] 연결 성공! URL: {wandb.run.url}")
                print(f"[WandB] run id: {wandb.run.id}")
            else:
                print("[WandB] 연결 실패 — wandb.run 이 None 입니다")

    def make_fn(offset):
        return lambda: make_env(args.base_port + offset, args.env_mode,
                                args.max_steps or None, args.seed + offset)

    vec_env  = VecMonitor(
        DummyVecEnv([make_fn(0)]) if args.n_envs == 1
        else SubprocVecEnv([make_fn(i) for i in range(args.n_envs)]),
        str(log_dir))
    eval_env = VecMonitor(DummyVecEnv([make_fn(100)]), str(log_dir / "eval"))

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, device=args.device)
        model.learning_rate = hp["learning_rate"]
        model.clip_range    = hp["clip_range"]
        model.ent_coef      = hp["ent_coef"]
        print(f"로드: {args.resume}")
    else:
        model = PPO(
            "MultiInputPolicy", vec_env, tensorboard_log=str(log_dir),
            verbose=1, device=args.device, **hp,
            policy_kwargs=dict(
                features_extractor_class=LineBuildCNNExtractor,
                features_extractor_kwargs={"features_dim": 256},
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=nn.ReLU,
            ),
        )

    callbacks = [
        CheckpointCallback(max(10_000 // args.n_envs, 1), str(save_dir), "line_build"),
        EvalCallback(eval_env, best_model_save_path=str(save_dir / "best"),
                     log_path=str(log_dir / "eval"), eval_freq=max(10_000 // args.n_envs, 1),
                     n_eval_episodes=3, deterministic=True, verbose=1),
        TrackingCallback(log_freq=1000, ckpt_freq=20, save_dir=str(save_dir),
                         mode=args.env_mode, verbose=1),
        RenderCallback(freq=4),
        VideoRecorderCallback(make_fn(200), freq=15, video_dir=str(log_dir / "videos")),
    ]

    print(f"\n{'='*50}")
    print(f"  일직선 블록 놓기 PPO")
    print(f"  모드: {args.env_mode.upper()}  |  {args.total_steps:,} steps  |  {args.n_envs} envs")
    print(f"  액션: {NUM_ACTIONS}개 {ACTION_NAMES}")
    print(f"  목표: 나무 블록 {LINE_TARGET}개를 z축 일직선으로 놓기")
    print(f"  보상: place(+0.1) + shaping(+0.5/칸) + dir(+0.3) + milestone(max+18.0)")
    print(f"{'='*50}\n")
    try:
        model.learn(args.total_steps, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=not bool(args.resume))
    finally:
        final = save_dir / f"line_build_{args.env_mode}_final"
        model.save(str(final))
        print(f"\n{'='*50}")
        print(f"  모델 저장: {final}.zip")
        print(f"")
        print(f"  hollow_box 이어 학습:")
        print(f"  python Seoyeon/hollow_box_rl_0314.py --mode train \\")
        print(f"    --resume {final}.zip")
        print(f"{'='*50}\n")
        if _wactive():
            try:
                art = wandb.Artifact(f"ppo_line_{args.env_mode}_final", type="model")
                art.add_file(str(final.with_suffix(".zip")))
                if wandb.run:
                    wandb.run.log_artifact(art)
                wandb.finish()
            except Exception as e:
                print(f"[WandB] 최종 artifact 저장 실패: {e}")
                try: wandb.finish()
                except Exception: pass
        vec_env.close(); eval_env.close()


def evaluate(args):
    env   = make_env(args.base_port, args.env_mode, args.max_steps or None, args.seed)
    model = PPO.load(args.resume, env=env, device=args.device)
    rewards = []
    for ep in range(args.n_eval_episodes):
        obs, _ = env.reset(); done = False; ep_r = 0.0; steps = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(act))
            ep_r += r; done = term or trunc; steps += 1
        rewards.append(ep_r)
        st = info.get("structure", {})
        print(f"  Ep {ep+1:2d}: R={ep_r:+.1f}  steps={steps}  "
              f"total={st.get('total',0)} line={st.get('longest_line',0)} "
              f"complete={st.get('line_complete',False)}")
    print(f"\n평균: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    env.close()


# =================================================================
# 엔트리포인트
# =================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CraftGround 일직선 블록 놓기 PPO")
    p.add_argument("--mode",      default="train", choices=["train", "eval"])
    p.add_argument("--env_mode",  choices=list(MODES.keys()), default="creative")
    p.add_argument("--total_steps",     type=int, default=100_000)
    p.add_argument("--n_envs",          type=int, default=1)
    p.add_argument("--base_port",       type=int, default=8040)
    p.add_argument("--max_steps",       type=int, default=0, help="0=모드 기본값")
    p.add_argument("--log_dir",         default="Seoyeon/logs")
    p.add_argument("--save_dir",        default="Seoyeon/checkpoints")
    p.add_argument("--resume",          default=None)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--device",          default="auto")
    p.add_argument("--n_eval_episodes", type=int, default=5)
    p.add_argument("--wandb_project",   default="line_build_rl")
    p.add_argument("--wandb_run",       default=None)
    args = p.parse_args()

    match args.mode:
        case "train": train(args)
        case "eval":
            assert args.resume, "--resume 경로를 지정하세요"
            evaluate(args)