"""
hollow_box_rl_0314.py — CraftGround 속이 뚫린 직육면체 건축 강화학습 (PPO)

house_rl_v1_0314.py 코드베이스 기반

환경: CraftGround 슈퍼플랫 세계에 스폰 → 블록 놓기로 속이 뚫린 직육면체 짓기
목적:
  - 바닥(floor)   : 7×7 직사각형 채우기
  - 외벽(wall)    : 4방향 테두리만 3단 쌓기 (내부 공간 비워두기)
  - 천장(ceiling) : 7×7 직사각형 덮기
  - 내부(interior): 블록 없이 비워두기 (hollow)

보상 설계:
- 블록 설치 보상: 구역별 차등 (바닥/외벽/천장 = 양수, 내부벽/범위외 = 강한 패널티)
- 마일스톤 보너스: 바닥 → 외벽 → 천장 → 완성 단계별
- 패널티: 내부 채우기, 범위 밖 설치, 빈 USE, 활동반경 초과

구조 크기:
  HOUSE_HALF = 3  →  7×7 발자국 (x,z 각각 -3~+3)
  HOUSE_HEIGHT = 4  →  바닥 위 외벽 4단, 천장은 y = spawn_y + 5
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
    if _WANDB and wandb.run: wandb.log(d, step=step)
def _wactive() -> bool: return _WANDB and bool(wandb.run)


# =================================================================
# 1. 설정
# =================================================================

@dataclass(frozen=True)
class ModeConfig:
    initial_extra_commands: tuple[str, ...]
    max_episode_steps: int
    use_health_penalty: bool
    use_food_penalty:   bool
    use_death_penalty:  bool
    use_night_bonus:    bool
    give_inventory:     bool

_COMMON = (
    "gamerule doWeatherCycle false", "gamerule doImmediateRespawn true",
    "gamerule fallDamage false", "weather clear",
    "tp @p ~ ~ ~ ~ 45",  # pitch=45 → 아래 45도 방향으로 시작
)
MODES: dict[str, ModeConfig] = {
    "creative": ModeConfig(
        ("gamemode creative @p", "gamerule doDaylightCycle false",
         "gamerule doMobSpawning false", "time set 6000") + _COMMON,
        max_episode_steps=6000,
        use_health_penalty=False, use_food_penalty=False,
        use_death_penalty=False,  use_night_bonus=False, give_inventory=True,
    ),
    "safe": ModeConfig(
        ("gamemode survival @p", "difficulty peaceful", "gamerule doDaylightCycle false",
         "gamerule doMobSpawning false", "time set 6000") + _COMMON,
        max_episode_steps=12000,
        use_health_penalty=False, use_food_penalty=False,
        use_death_penalty=False,  use_night_bonus=False, give_inventory=True,
    ),
    "survival": ModeConfig(
        ("gamemode survival @p", "difficulty normal", "gamerule doDaylightCycle true",
         "gamerule doMobSpawning true", "time set 6000") + _COMMON,
        max_episode_steps=12000,
        use_health_penalty=True, use_food_penalty=True,
        use_death_penalty=True,  use_night_bonus=True, give_inventory=True,
    ),
}

# 속이 뚫린 직육면체 → 나무 블록만 사용
INVENTORY_CMDS = (
    # 핫바 슬롯 0~5: 나무 계열 블록
    "item replace entity @p hotbar.0 with minecraft:oak_planks 64",
    "item replace entity @p hotbar.1 with minecraft:oak_log 64",
    "item replace entity @p hotbar.2 with minecraft:oak_slab 64",
    "item replace entity @p hotbar.3 with minecraft:oak_planks 64",
    "item replace entity @p hotbar.4 with minecraft:oak_log 64",
    "item replace entity @p hotbar.5 with minecraft:oak_slab 64",
    # 추가 인벤토리: 나무 블록으로 채움
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_log 64",
    "give @p minecraft:oak_log 64",
    "give @p minecraft:oak_slab 64",
    "give @p minecraft:oak_slab 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_log 64",
    "give @p minecraft:oak_slab 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_log 64",
    "give @p minecraft:oak_slab 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_log 64",
    "give @p minecraft:oak_slab 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_log 64",
    "give @p minecraft:oak_slab 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_log 64",
    "give @p minecraft:oak_slab 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
)


# =================================================================
# 2. 블록 집합 & 마일스톤
# =================================================================

BUILDING_BLOCKS = frozenset({
    "minecraft:oak_planks",
    "minecraft:oak_log",
    "minecraft:oak_slab",
})
ALL_TARGET_BLOCKS = BUILDING_BLOCKS  # 나무 블록만 사용

# 마일스톤 순서: 바닥 → 외벽 → 천장 → 완성
# 이전 단계가 달성돼야 다음 단계 보상이 의미 있도록 설계
_MILESTONES = [
    # 1단계: 바닥 놓기 (5×5=25칸, 절반이면 충분)
    ("floor_2",       lambda s: s["floor"] >= 2,                                      1.0),
    ("floor_8",       lambda s: s["floor"] >= 8,                                      3.0),
    ("floor_done",    lambda s: s["floor"] >= 16,                                     6.0),
    # 2단계: 4방향 외벽 시작 (바닥 >= 2 이후)
    ("has_wall",      lambda s: s["has_wall"] and s["floor"] >= 2,                    3.0),
    ("four_walls",    lambda s: s["four_walls"],                                      6.0),
    # 3단계: 천장 덮기 (4방향 외벽 완성 후)
    ("has_ceiling",   lambda s: s["has_ceiling"] and s["four_walls"],                 4.0),
    ("ceiling_done",  lambda s: s["ceiling"] >= 9 and s["four_walls"],                6.0),
    # 4단계: 완성
    ("hollow_box",    lambda s: s["hollow_box"],                                     30.0),
]
MS_KEYS = [k for k, *_ in _MILESTONES]


# =================================================================
# 3. RaycastTracker
# =================================================================

class RaycastTracker:
    """raycast_result 기반 블록 설치 감지 + 속이 뚫린 직육면체 완성도 추적."""

    HOUSE_HALF   = 2   # 5×5 발자국 (exterior: ±2) — 작게 줄여 학습 난이도 완화
    HOUSE_HEIGHT = 2   # 외벽 높이 (바닥 위 2단), 천장은 spawn_y + HOUSE_HEIGHT + 1

    def __init__(self): self.reset()

    def reset(self, spawn_y: float = 4.0, spawn_x: float = 0.0, spawn_z: float = 0.0):
        self._placed: dict[tuple, tuple] = {}  # pos → (zone, bid)
        self._cache:  dict               = {}
        self._spawn_y = spawn_y
        self._spawn_x = spawn_x
        self._spawn_z = spawn_z
        self.milestones = {k: False for k, *_ in _MILESTONES}

    def update(self, raw_obs, action_was_use: bool, action_was_attack: bool = False) -> tuple[bool, str | None, bool]:
        """블록 설치/파괴 감지. (newly_placed, zone|None, newly_broken) 반환."""
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
        pos = _hit_pos(hit)
        if not pos or bid not in ALL_TARGET_BLOCKS or pos in self._placed:
            return False, None, newly_broken
        zone = self.block_zone(pos)
        self._placed[pos] = (zone, bid)
        self._cache.clear()
        return True, zone, newly_broken

    def _check_four_walls(self) -> bool:
        """4방향 외벽 각각에 2개 이상의 외벽 블록이 있어야 true."""
        H  = self.HOUSE_HALF
        sx = round(self._spawn_x)
        sz = round(self._spawn_z)
        sy = self._spawn_y
        sides = {"x_neg": 0, "x_pos": 0, "z_neg": 0, "z_pos": 0}
        for (bx, by, bz), (zone, _) in self._placed.items():
            if zone != "wall": continue
            dy = by - sy
            if not (1 < dy <= self.HOUSE_HEIGHT + 1): continue
            dx = bx - sx; dz = bz - sz
            if dx == -H: sides["x_neg"] += 1
            if dx ==  H: sides["x_pos"] += 1
            if dz == -H: sides["z_neg"] += 1
            if dz ==  H: sides["z_pos"] += 1
        return all(v >= 2 for v in sides.values())

    def is_adjacent_to_placed(self, pos: tuple) -> bool:
        if len(self._placed) <= 1:
            return True
        bx, by, bz = pos
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            if (bx+dx, by+dy, bz+dz) in self._placed:
                return True
        return False

    def analyze(self) -> dict:
        if self._cache: return self._cache
        sy = self._spawn_y
        floor = wall = ceiling = interior_wall = 0
        for pos, (zone, _) in self._placed.items():
            if zone == "floor":         floor         += 1
            elif zone == "wall":        wall          += 1
            elif zone == "ceiling":     ceiling       += 1
            elif zone == "wall_inside": interior_wall += 1
        total = len(self._placed)
        four_walls = self._check_four_walls()
        hollow = self._hollow_box_check(floor, four_walls, ceiling, interior_wall)
        self._cache = {
            "total":         total,
            "floor":         floor,
            "wall":          wall,
            "ceiling":       ceiling,
            "interior_wall": interior_wall,
            "has_wall":      wall >= 2,   # 5×5 구조에 맞게 완화
            "has_ceiling":   ceiling >= 2,
            "four_walls":    four_walls,
            "hollow_box":    hollow,
        }
        return self._cache

    def _hollow_box_check(self, floor: int, four_walls: bool, ceiling: int, interior_wall: int) -> bool:
        """속이 뚫린 직육면체 완성 조건 (5×5, 2단):
        - 바닥 >= 9 블록 (5×5=25 중 절반 이상)
        - 4방향 외벽 형성
        - 천장 >= 9 블록
        - 내부 벽 블록 = 0 (hollow 조건)
        - BFS flood-fill로 내부 공기가 탈출 불가
        """
        if floor < 9 or not four_walls or ceiling < 9:
            return False
        if interior_wall > 0:
            return False   # 내부에 블록이 있으면 hollow 아님
        return self._flood_fill_enclosed()

    def _flood_fill_enclosed(self) -> bool:
        """BFS flood-fill: 내부 공기가 외부로 탈출 불가하면 True (완전 밀폐).
        바닥(y <= spawn_y) 아래는 고체로 간주.
        """
        solid = set(self._placed.keys())
        sx = round(self._spawn_x)
        sz = round(self._spawn_z)
        sy = int(self._spawn_y)
        H  = self.HOUSE_HALF + 2   # 이 범위 밖이면 "탈출"로 판정

        # BFS 시작: 내부 공기 (바닥 바로 위, 내부 중심)
        start = None
        for dx in range(0, self.HOUSE_HALF):
            for dz in range(0, self.HOUSE_HALF):
                candidate = (sx + dx, sy + 2, sz + dz)
                if candidate not in solid:
                    start = candidate
                    break
            if start: break
        if start is None:
            return False

        visited = {start}
        queue   = [start]
        while queue:
            bx, by, bz = queue.pop(0)
            if abs(bx - sx) > H or abs(bz - sz) > H or by > sy + self.HOUSE_HEIGHT + 3 or by <= sy:
                return False
            for ddx, ddy, ddz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                nb = (bx+ddx, by+ddy, bz+ddz)
                if nb not in visited and nb not in solid and nb[1] > sy:
                    visited.add(nb)
                    queue.append(nb)
            if len(visited) > 800:
                return False
        return True

    def block_zone(self, pos: tuple | None) -> str:
        """블록 위치가 목표 구조의 어느 구역인지 반환.
        반환값: 'floor' | 'wall' | 'wall_inside' | 'ceiling' | 'out'

        구조:
          floor:      y == sy+1, |dx|<=H and |dz|<=H
          wall:       sy+1 < y <= sy+H+1, (|dx|==H or |dz|==H) and |dx|<=H and |dz|<=H
          wall_inside: sy+1 < y <= sy+H+1, |dx|<H and |dz|<H  (내부 → 비워야 함)
          ceiling:    y == sy+H+2, |dx|<=H and |dz|<=H
          out:        그 외
        """
        if pos is None: return "out"
        bx, by, bz = pos
        H   = self.HOUSE_HALF
        CH  = self.HOUSE_HEIGHT
        sx  = round(self._spawn_x)
        sz  = round(self._spawn_z)
        sy  = self._spawn_y
        dx  = abs(bx - sx)
        dz  = abs(bz - sz)
        dy  = by - sy
        in_area  = dx <= H and dz <= H
        on_perim = in_area and (dx == H or dz == H)
        in_inner = in_area and not on_perim   # 테두리 제외 내부

        if dy <= 1:
            return "floor" if in_area else "out"
        if 1 < dy <= CH + 1:
            if on_perim:  return "wall"
            if in_inner:  return "wall_inside"
            return "out"
        if dy == CH + 2:
            return "ceiling" if in_area else "out"
        return "out"

    def milestone_rewards(self) -> float:
        """달성된 마일스톤 보너스 합계 반환 (일회성)."""
        s = self.analyze(); bonus = 0.0
        for key, fn, val in _MILESTONES:
            if fn(s) and not self.milestones[key]:
                self.milestones[key] = True; bonus += val
        return bonus

    def raycast_vec(self, raw_obs) -> np.ndarray:
        """8-dim: [is_block, is_floor_zone, is_wall_zone, is_ceiling_zone,
                   is_inside_zone, rel_x, rel_y, rel_z]"""
        vec = np.zeros(8, dtype=np.float32)
        hit = _get_hit(raw_obs)
        if hit is None or _hit_type(hit) != "block": return vec
        bid = _strip_state(_hit_state(hit))
        pos = _hit_pos(hit)
        if bid not in ALL_TARGET_BLOCKS: return vec
        vec[0] = 1.0
        zone = self.block_zone(pos)
        zone_map = {"floor": 1, "wall": 2, "ceiling": 3, "wall_inside": 4}
        idx = zone_map.get(zone, 0)
        if idx > 0: vec[idx] = 1.0
        if pos:
            px = _scalar(raw_obs, "x"); py = _scalar(raw_obs, "y"); pz = _scalar(raw_obs, "z")
            vec[5:8] = np.clip([(pos[0]-px)/10., (pos[1]-py)/10., (pos[2]-pz)/10.], -1., 1.)
        return vec

    def struct_vec(self, st: dict) -> np.ndarray:
        """8-dim 구조 벡터: [floor, wall, ceiling, four_walls, has_ceiling, interior_wall, hollow_box, padding]"""
        H  = self.HOUSE_HALF
        max_floor   = (2*H+1)**2        # 49
        max_wall    = (2*H+1)*4 - 4     # 24 per level × HOUSE_HEIGHT
        max_ceiling = (2*H+1)**2        # 49
        return np.array([
            min(st["floor"]         / max_floor,   1.),
            min(st["wall"]          / (max_wall * self.HOUSE_HEIGHT), 1.),
            min(st["ceiling"]       / max_ceiling, 1.),
            float(st["four_walls"]),
            float(st["has_ceiling"]),
            min(st["interior_wall"] / 10., 1.),   # 내부 블록 수 (0이어야 함)
            float(st["hollow_box"]),
            0.0,  # padding (구 코드와 shape 호환)
        ], dtype=np.float32)


# =================================================================
# 4. HitResult 헬퍼
# =================================================================

def _get_full(raw_obs):
    if isinstance(raw_obs, dict):
        return raw_obs.get("full", raw_obs)
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
        if len(parts) == 2:
            return f"{parts[0]}:{parts[1]}"
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


# =================================================================
# 5. 액션
# =================================================================

ACTION_NAMES = [
    "NO_OP",
    "FORWARD", "BACKWARD", "LEFT", "RIGHT", "JUMP",
    "USE",
    "CAMERA_LEFT", "CAMERA_RIGHT", "CAMERA_UP", "CAMERA_DOWN",
    "HOTBAR_1", "HOTBAR_2", "HOTBAR_3", "HOTBAR_4", "HOTBAR_5",
    "HOTBAR_6",
    "LOOK_DOWN_AND_USE",   # 매크로: 카메라 아래 + 블록 설치
    "LOOK_DOWN",           # 매크로: 카메라 아래
]

MACRO_ACTIONS = {"LOOK_DOWN_AND_USE", "LOOK_DOWN"}
CAM_DEG = 10.0

def build_action(name: str) -> dict:
    act = no_op_v2()
    match name:
        case "FORWARD":       act["forward"]      = True
        case "BACKWARD":      act["back"]         = True
        case "LEFT":          act["left"]          = True
        case "RIGHT":         act["right"]         = True
        case "JUMP":          act["jump"]          = True
        case "ATTACK":        act["attack"]        = True
        case "USE":           act["use"]           = True
        case "CAMERA_LEFT":   act["camera_yaw"]   = -CAM_DEG
        case "CAMERA_RIGHT":  act["camera_yaw"]   =  CAM_DEG
        case "CAMERA_UP":     act["camera_pitch"] = -CAM_DEG
        case "CAMERA_DOWN":   act["camera_pitch"] =  CAM_DEG
        case n if n.startswith("HOTBAR_"):
            slot = int(n.split("_")[1]) - 1
            act[f"hotbar.{slot}"] = True
    return act


# =================================================================
# 6. 보상 상수
# =================================================================

# 구역별 블록 설치 보상
ZONE_REWARDS = {
    "floor":       1.5,   # 바닥 설치
    "wall":        2.0,   # 외벽 설치 (핵심)
    "ceiling":     2.0,   # 천장 설치
    "building":    0.8,   # 구역 미확인 블록 (인벤토리 감지 폴백)
    "wall_inside": -3.0,  # 내부 채우기 → 패널티 (hollow 위반) — 너무 크면 학습 방해
    "out":         -1.5,  # 범위 밖 설치
}

ISOLATED_PENALTY    = -0.02
WANDER_PENALTY      = -0.03
ALIVE_REWARD        = -0.001   # time pressure 줄임
FAILED_USE_PENALTY  =  0.0     # 실패 패널티 제거 → USE 시도를 적극적으로 하도록
BREAK_PLACED_PENALTY = -2.0
CONSEC_BONUS_K      =  0.3
NIGHT_BONUS         =  5.0
HEALTH_PENALTY_K    = -0.5
FOOD_PENALTY_K      = -0.1
DEATH_PENALTY       = -20.0
REWARD_CLIP         = (-10.0, 10.0)
_MODE_COLOR_BGR     = {"creative": (80, 200, 255), "safe": (80, 255, 80), "survival": (80, 80, 255)}


def _scalar(obs, key: str, default: float = 0.0) -> float:
    full = _get_full(obs)
    if isinstance(full, dict): return float(full.get(key, default))
    return float(getattr(full, key, default))

# translation_key → zone 매핑 (인벤토리 감소 감지용)
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
        if curr_counts.get(tk, 0) < prev_counts.get(tk, 0):
            return True
    return False


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
            else:
                return None
    elif isinstance(getattr(obs, "image", None), bytes):
        try:
            arr = np.frombuffer(obs.image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception: return None
    else:
        return None
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img.astype(np.uint8)

def _draw_hud(frame: np.ndarray, st: dict, ms: dict,
              action: str, mode: str, ep_r: float, step: int) -> np.ndarray:
    color   = _MODE_COLOR_BGR.get(mode, (200, 200, 200))
    hollow  = st.get("hollow_box", False)
    lines = [
        (f"Step:{step:,}  Rwd:{ep_r:+.1f}  Act:{action}", (220, 220, 220)),
        (f"Fl:{st.get('floor',0):3d}  Wl:{st.get('wall',0):3d}  "
         f"Ceil:{st.get('ceiling',0):3d}  Inside:{st.get('interior_wall',0)}  "
         f"4W:{'Y' if st.get('four_walls') else 'n'}",
         (220, 220, 220)),
        (f"HOLLOW BOX: {'DONE!' if hollow else 'building...'}",
         (50, 255, 80) if hollow else (180, 180, 180)),
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

class HollowBoxWrapper(gym.Wrapper):
    """
    속이 뚫린 직육면체 건축 환경 Wrapper

    Obs: {"image":(64,114,3) u8, "state":(8,) f32, "raycast":(8,) f32, "structure":(8,) f32}
    Act: Discrete(19) → build_action() → no_op_v2() raw dict
    info["render_frame"]: BGR ndarray (HUD 포함)
    """
    H, W   = 64, 114
    RH, RW = 180, 320
    S      = 4

    def __init__(self, env, cfg: ModeConfig, max_steps: int, mode_name: str = ""):
        super().__init__(env)
        self.cfg, self.max_steps, self.mode_name = cfg, max_steps, mode_name
        self.tracker = RaycastTracker()
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict({
            "image":     spaces.Box(0,   255, (self.H, self.W, 3), np.uint8),
            "state":     spaces.Box(-1., 1.,  (8,), np.float32),
            "raycast":   spaces.Box(-1., 1.,  (8,), np.float32),
            "structure": spaces.Box(0.,  1.,  (8,), np.float32),
        })
        self._step = 0; self._ep_r = 0.0
        self._prev_health = self._prev_food = 20.0
        self._night_started = False
        self._consec_place = 0
        self._prev_inv: dict[str, int] = {}

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        self.tracker.reset(spawn_y=_scalar(raw, "y", 4.0),
                           spawn_x=_scalar(raw, "x", 0.0),
                           spawn_z=_scalar(raw, "z", 0.0))
        self._step = 0; self._ep_r = 0.0
        self._prev_health = self._prev_food = 20.0
        self._night_started = False
        self._consec_place = 0
        self._prev_inv = _get_inv_counts(raw)
        raw_img = _extract_image(raw)
        obs = self._make_obs(raw, self.tracker.analyze(), raw_img)
        info["render_frame"] = self._render(raw_img, {}, {}, "")
        return obs, info

    def step(self, action: int):
        name = ACTION_NAMES[int(action)]
        if name == "LOOK_DOWN_AND_USE":
            look = no_op_v2(); look["camera_pitch"] = CAM_DEG
            for _ in range(2): raw, *_ = self.env.step(look)
            use = no_op_v2(); use["use"] = True
            raw, _, terminated, truncated, info = self.env.step(use)
        elif name == "LOOK_DOWN":
            look = no_op_v2(); look["camera_pitch"] = CAM_DEG
            for _ in range(3): raw, *_ = self.env.step(look)
            raw, _, terminated, truncated, info = self.env.step(no_op_v2())
        else:
            raw, _, terminated, truncated, info = self.env.step(build_action(name))
        self._step += 1
        truncated = truncated or (self._step >= self.max_steps)

        is_use    = name in ("USE", "LOOK_DOWN_AND_USE")
        is_attack = name == "ATTACK"
        placed, zone, broken = self.tracker.update(raw, is_use, is_attack)

        # 인벤토리 감소 기반 보완 감지
        curr_inv = _get_inv_counts(raw)
        if not placed and is_use:
            if _detect_placed_by_inv(self._prev_inv, curr_inv):
                placed = True
                zone   = "building"  # 구역 불명 → 일반 건축으로 처리
        self._prev_inv = curr_inv

        st  = self.tracker.analyze()
        raw_img = _extract_image(raw)
        obs = self._make_obs(raw, st, raw_img)
        rew = self._compute_reward(raw, terminated or truncated, placed, zone, name, broken)
        self._ep_r += rew

        # hollow_box 달성 시 에피소드 종료
        if st.get("hollow_box", False):
            terminated = True

        info.update({
            "structure":    st,
            "milestones":   dict(self.tracker.milestones),
            "action_name":  name,
            "episode_step": self._step,
            "render_frame": self._render(raw_img, st, self.tracker.milestones, name),
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

    def _render(self, img_rgb, st, ms, action) -> np.ndarray:
        if img_rgb is None:
            img_rgb = np.zeros((self.RH, self.RW, 3), dtype=np.uint8)
        frame = cv2.cvtColor(
            cv2.resize(img_rgb, (self.RW, self.RH), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_RGB2BGR)
        return _draw_hud(frame, st, ms, action, self.mode_name, self._ep_r, self._step)

    def _compute_reward(self, raw, done: bool, placed: bool, zone: str | None,
                        name: str = "", broken: bool = False) -> float:
        r = 0.0 if done else ALIVE_REWARD

        # 활동반경 패널티
        ax   = _scalar(raw, "x")
        az   = _scalar(raw, "z")
        dist = max(abs(ax - self.tracker._spawn_x), abs(az - self.tracker._spawn_z))
        wander_limit = self.tracker.HOUSE_HALF + 1
        if dist > wander_limit:
            r += WANDER_PENALTY * (dist - wander_limit)

        if broken:
            r += BREAK_PLACED_PENALTY

        health = _scalar(raw, "health",     20.)
        food   = _scalar(raw, "food_level", 20.)
        if self.cfg.use_health_penalty and health < self._prev_health:
            r += HEALTH_PENALTY_K * (self._prev_health - health)
        if self.cfg.use_food_penalty and food < self._prev_food:
            r += FOOD_PENALTY_K * (self._prev_food - food)
        self._prev_health, self._prev_food = health, food

        if self.cfg.use_death_penalty and done and bool(_scalar(raw, "is_dead")):
            r += DEATH_PENALTY

        # USE 시 하늘 보면 소폭 패널티
        if name in ("USE", "LOOK_DOWN_AND_USE"):
            pitch = _scalar(raw, "pitch")
            if pitch < 0: r -= 0.05

        if placed and zone is not None:
            base_r = ZONE_REWARDS.get(zone, 1.0)

            if zone == "wall":
                # 외벽 설치 순서 보너스: 바닥이 어느 정도 있어야
                st = self.tracker.analyze()
                if st["floor"] < 4:
                    base_r -= 0.5   # 바닥 없이 벽 먼저 쌓으면 소폭 감점
            elif zone == "ceiling":
                # 천장 설치 순서 보너스: 4방향 외벽이 있어야
                st = self.tracker.analyze()
                if not st["four_walls"]:
                    base_r = -1.0   # 순서 위반
            elif zone == "floor":
                # 고립 설치 체크
                hit = _get_hit(raw)
                pos = _hit_pos(hit) if hit is not None else None
                if pos and not self.tracker.is_adjacent_to_placed(pos):
                    base_r = ISOLATED_PENALTY

            # 연속 설치 보너스
            self._consec_place += 1
            base_r += min(self._consec_place * CONSEC_BONUS_K, 3.0)
            r += base_r
        else:
            self._consec_place = 0
            if name in ("USE", "LOOK_DOWN_AND_USE"):
                r += FAILED_USE_PENALTY

        r += self.tracker.milestone_rewards()

        if self.cfg.use_night_bonus:
            wtime = int(_scalar(raw, "world_time")) % 24000
            if 13000 <= wtime <= 23000: self._night_started = True
            elif self._night_started:   r += NIGHT_BONUS; self._night_started = False

        return float(np.clip(r, *REWARD_CLIP))


def make_env(port=8030, mode="creative", max_steps=None, seed=42) -> HollowBoxWrapper:
    if mode not in MODES:
        raise ValueError(f"mode='{mode}' 불가. 선택: {list(MODES.keys())}")
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
    return HollowBoxWrapper(env, cfg=cfg,
                            max_steps=max_steps or cfg.max_episode_steps,
                            mode_name=mode)


# =================================================================
# 8. 모델
# =================================================================

class HollowBoxCNNExtractor(BaseFeaturesExtractor):
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
# 9. 하이퍼파라미터
# =================================================================

HP = {
    "creative": dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
                     gamma=0.99,  gae_lambda=0.95, clip_range=0.2,  ent_coef=0.02,
                     vf_coef=0.5, max_grad_norm=0.5),
    "safe":     dict(learning_rate=1e-4, n_steps=512, batch_size=64, n_epochs=10,
                     gamma=0.995, gae_lambda=0.95, clip_range=0.2,  ent_coef=0.01,
                     vf_coef=0.5, max_grad_norm=0.5),
    "survival": dict(learning_rate=5e-5, n_steps=512, batch_size=64, n_epochs=10,
                     gamma=0.995, gae_lambda=0.95, clip_range=0.15, ent_coef=0.005,
                     vf_coef=0.5, max_grad_norm=0.5),
}


# =================================================================
# 10. 콜백
# =================================================================

class TrackingCallback(BaseCallback):
    def __init__(self, log_freq=2000, ckpt_freq=20, save_dir=".", mode="creative", verbose=0):
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
                self._buf.append({"st": info["structure"], "ms": info.get("milestones", {})})
                if len(self._buf) > 1000: del self._buf[:500]
        if self.num_timesteps % self.log_freq == 0 and self._buf:
            self._log()
        if self.num_timesteps <= 200 and self.num_timesteps % 50 == 0:
            infos = self.locals.get("infos", [{}])
            st    = infos[0].get("structure", {})
            print(f"[Debug t={self.num_timesteps}] "
                  f"floor={st.get('floor',0)} wall={st.get('wall',0)} "
                  f"ceiling={st.get('ceiling',0)} inside={st.get('interior_wall',0)}")
        return True

    def _on_rollout_end(self):
        self._ridx += 1
        if not _wactive(): return
        ep_buf = self.model.ep_info_buffer
        if ep_buf:
            _wlog({"rollout/mean_ep_reward": float(np.mean([e["r"] for e in ep_buf])),
                   "rollout/mean_ep_length": float(np.mean([e["l"] for e in ep_buf]))},
                  step=self.num_timesteps)
        if self._ridx % self.ckpt_freq == 0:
            p = self.save_dir / f"ckpt_{self._ridx}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(p))
            art = wandb.Artifact(f"ppo_{self.mode}_ckpt_{self._ridx}", type="model")
            art.add_file(str(p.with_suffix(".zip")))
            wandb.run.log_artifact(art)

    def _log(self):
        recent = self._buf[-200:]
        hollow = float(np.mean([e["st"].get("hollow_box", False) for e in recent]))
        blk    = float(np.mean([e["st"].get("total",      0)     for e in recent]))
        inside = float(np.mean([e["st"].get("interior_wall", 0)  for e in recent]))
        ms     = {k: float(np.mean([e["ms"].get(k, False) for e in recent])) for k in MS_KEYS}
        for k, v in ms.items(): self.logger.record(f"hollow/ms_{k}", v)
        self.logger.record("hollow/hollow_box_rate", hollow)
        self.logger.record("hollow/avg_blocks",      blk)
        self.logger.record("hollow/avg_inside",      inside)
        if _wactive():
            _wlog({"rollout/hollow_box_rate": hollow, "rollout/avg_blocks": blk,
                   "rollout/avg_inside": inside,
                   **{f"hollow/ms_{k}": v for k, v in ms.items()}},
                  step=self.num_timesteps)
        if self.verbose:
            print(f"\n[{self.num_timesteps:,}] blk={blk:.1f}  hollow={hollow:.2f}  inside={inside:.2f}")


class RenderCallback(BaseCallback):
    WIN = "Hollow Box Builder"

    def __init__(self, freq=4):
        super().__init__()
        self.freq = freq; self._active = False; self._tick = 0

    def _on_training_start(self):
        try:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            self._active = True
            print(f"[Render] cv2 창 오픈: '{self.WIN}'")
        except Exception as e:
            print(f"⚠️  cv2 창 오픈 실패: {e}")

    def _on_step(self) -> bool:
        if not self._active: return True
        self._tick += 1
        if self._tick % self.freq: return True
        frame = (self.locals.get("infos") or [{}])[0].get("render_frame")
        if frame is None:
            frame = np.zeros((HollowBoxWrapper.H * HollowBoxWrapper.S,
                              HollowBoxWrapper.W * HollowBoxWrapper.S, 3), dtype='uint8')
        try:
            cv2.imshow(self.WIN, frame)
        except Exception as e:
            print(f"⚠️  렌더 실패: {e}")
            self._active = False; return True
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._active = False
            cv2.destroyWindow(self.WIN)
        return True

    def _on_training_end(self):
        if self._active:
            cv2.destroyWindow(self.WIN)
            self._active = False


class VideoRecorderCallback(BaseCallback):
    RW = HollowBoxWrapper.RW
    RH = HollowBoxWrapper.RH

    def __init__(self, make_env_fn, freq=10, video_dir="videos", fps=10):
        super().__init__()
        self._make = make_env_fn; self.freq = freq
        self.dir = Path(video_dir); self.fps = fps
        self._ridx = 0; self._env = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        self._ridx += 1
        if self._ridx % self.freq == 0:
            if self._env is None: self._env = self._make()
            self._record()

    def _record(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        path   = self.dir / f"ep_{self._ridx:04d}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (self.RW, self.RH))
        obs, _ = self._env.reset(); done = False; total_r = 0.0; info = {}
        try:
            while not done:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self._env.step(int(act))
                total_r += r; done = term or trunc
                if (f := info.get("render_frame")) is not None: writer.write(f)
        finally:
            writer.release()
        if _wactive():
            wandb.log({
                "video/eval_episode":  wandb.Video(str(path), fps=self.fps, format="mp4"),
                "video/eval_reward":   total_r,
                "video/eval_blocks":   info.get("structure", {}).get("total",      0),
                "video/eval_hollow":   int(info.get("structure", {}).get("hollow_box", False)),
            }, step=self.num_timesteps)

    def _on_training_end(self):
        try:
            if self._env is None: self._env = self._make()
            self._ridx = 9999
            final_path = self.dir / "final_episode.mp4"
            self.dir.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(final_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, (self.RW, self.RH))
            obs, _ = self._env.reset(); done = False; total_r = 0.0; info = {}
            try:
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = self._env.step(int(act))
                    total_r += r; done = term or trunc
                    if (f := info.get("render_frame")) is not None: writer.write(f)
            finally:
                writer.release()
            print(f"\n[Video] 최종 영상 저장: {final_path}  (R={total_r:+.1f})")
            if _wactive():
                wandb.log({
                    "video/final_episode":  wandb.Video(str(final_path), fps=self.fps, format="mp4"),
                    "video/final_reward":   total_r,
                    "video/final_blocks":   info.get("structure", {}).get("total",      0),
                    "video/final_hollow":   int(info.get("structure", {}).get("hollow_box", False)),
                })
        except Exception as e:
            print(f"⚠️  최종 영상 저장 실패: {e}")
        finally:
            if self._env: self._env.close()


# =================================================================
# 11. 훈련 / 평가 / 디버그
# =================================================================

def train(args):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = Path(args.log_dir)  / f"hollow_{args.env_mode}_{ts}"
    save_dir = Path(args.save_dir) / f"hollow_{args.env_mode}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    hp = HP[args.env_mode]

    if args.wandb_project:
        if not _WANDB: print("⚠️  wandb 미설치 → pip install wandb")
        else:
            wandb.init(project=args.wandb_project,
                       name=args.wandb_run or f"hollow_ppo_{args.env_mode}_{ts}",
                       sync_tensorboard=True, save_code=True,
                       config={"task": "hollow_box", "env_mode": args.env_mode,
                               "total_steps": args.total_steps,
                               "n_envs": args.n_envs, "seed": args.seed, **hp})
            print(f"[WandB] {wandb.run.url}")

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
                features_extractor_class=HollowBoxCNNExtractor,
                features_extractor_kwargs={"features_dim": 256},
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=nn.ReLU,
            ),
        )

    callbacks = [
        CheckpointCallback(max(10_000 // args.n_envs, 1), str(save_dir), "hollow_box"),
        EvalCallback(eval_env, best_model_save_path=str(save_dir / "best"),
                     log_path=str(log_dir / "eval"), eval_freq=max(20_000 // args.n_envs, 1),
                     n_eval_episodes=3, deterministic=True, verbose=1),
        TrackingCallback(log_freq=2000, ckpt_freq=20, save_dir=str(save_dir),
                         mode=args.env_mode, verbose=1),
        RenderCallback(freq=4),
        VideoRecorderCallback(make_fn(200), freq=10, video_dir=str(log_dir / "videos")),
    ]

    print(f"\n[속이 뚫린 직육면체] {args.env_mode.upper()}  |  {args.total_steps:,} steps  |  {args.n_envs} envs")
    print(f"  목표: 바닥(7×7) + 외벽만(4방향 테두리 {RaycastTracker.HOUSE_HEIGHT}단) + 천장(7×7)\n")
    try:
        model.learn(args.total_steps, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=not bool(args.resume))
    finally:
        final = save_dir / f"hollow_box_{args.env_mode}_final"
        model.save(str(final))
        print(f"\n저장: {final}.zip")
        if _wactive():
            art = wandb.Artifact(f"ppo_hollow_{args.env_mode}_final", type="model")
            art.add_file(str(final.with_suffix(".zip")))
            wandb.run.log_artifact(art)
            wandb.finish()
        vec_env.close(); eval_env.close()


def evaluate(args):
    env   = make_env(args.base_port, args.env_mode, args.max_steps or None, args.seed)
    model = PPO.load(args.resume, env=env, device=args.device)
    rewards = []
    for ep in range(args.n_eval_episodes):
        obs, _ = env.reset(); done = False; ep_r = 0.0; steps = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            ep_r += r; done = term or trunc; steps += 1
        rewards.append(ep_r)
        st = info.get("structure", {})
        print(f"  Ep {ep+1:2d}: R={ep_r:+.1f}  steps={steps}  "
              f"fl={st.get('floor',0)} wl={st.get('wall',0)} "
              f"ceil={st.get('ceiling',0)} inside={st.get('interior_wall',0)}  "
              f"hollow={st.get('hollow_box',False)}")
    print(f"\n평균: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    env.close()


def _make_debug_env(port, gamemode="creative"):
    return make(
        initial_env_config=InitialEnvironmentConfig(
            image_width=114, image_height=64, seed="42",
            world_type=WorldType.SUPERFLAT, render_distance=4, simulation_distance=4,
            hud_hidden=False, request_raycast=True,
            initial_extra_commands=[
                f"gamemode {gamemode} @p", "gamerule doDaylightCycle false",
                "gamerule doMobSpawning false", "gamerule doWeatherCycle false",
                "gamerule doImmediateRespawn true", "weather clear",
                "time set 6000", "give @p minecraft:oak_planks 64",
            ],
        ),
        port=port, verbose=False, verbose_gradle=True, render_action=False,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )


def debug_raycast(args):
    """raycast_result proto 실제 필드명 확인."""
    env = _make_debug_env(args.base_port)
    raw, _ = env.reset()

    print("\n=== Observation 타입 & 키 ===")
    print("  타입:", type(raw))
    if isinstance(raw, dict):
        for k, v in raw.items():
            print(f"  [{k}] type={type(v).__name__}  value={str(v)[:120]}")
    else:
        try:
            for f in raw.DESCRIPTOR.fields:
                print(f"  {f.name:30s} = {str(getattr(raw, f.name, 'N/A'))[:80]}")
        except AttributeError:
            print("  DESCRIPTOR 없음")

    hit = None
    if isinstance(raw, dict):
        for k in raw:
            if "raycast" in k.lower():
                hit = raw[k]
                print(f"\n→ raycast 키 발견: '{k}'  타입: {type(hit)}")
                break
    else:
        hit = getattr(raw, "raycast_result", None)

    if hit is None:
        print("\n⚠️  raycast_result 없음")
        env.close(); return

    print("\n=== block_zone 테스트 (LOOK_DOWN_AND_USE 시퀀스) ===")
    tracker = RaycastTracker()
    tracker.reset(spawn_y=_scalar(raw, "y", 4.0),
                  spawn_x=_scalar(raw, "x", 0.0),
                  spawn_z=_scalar(raw, "z", 0.0))
    for i in range(10):
        act = no_op_v2()
        if i < 3: act["camera_pitch"] = CAM_DEG; label = "LOOK_DOWN"
        elif i < 6: act["use"] = True;            label = "USE"
        else: label = "NO_OP"
        raw, *_ = env.step(act)
        h = _get_hit(raw)
        pos = _hit_pos(h) if h else None
        zone = tracker.block_zone(pos)
        print(f"  [{i:02d}] {label:15s} pos={pos}  zone={zone}")
        time.sleep(0.05)
    env.close()


# =================================================================
# 엔트리포인트
# =================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CraftGround 속이 뚫린 직육면체 건축 PPO")
    p.add_argument("--mode",      default="train",
                   choices=["train", "eval", "debug_raycast"])
    p.add_argument("--env_mode",  choices=list(MODES.keys()), default="creative")
    p.add_argument("--total_steps",     type=int, default=100_000)
    p.add_argument("--n_envs",          type=int, default=1)
    p.add_argument("--base_port",       type=int, default=8030)
    p.add_argument("--max_steps",       type=int, default=0, help="0=모드 기본값")
    p.add_argument("--log_dir",         default="Seoyeon/logs")
    p.add_argument("--save_dir",        default="Seoyeon/checkpoints")
    p.add_argument("--resume",          default=None)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--device",          default="auto")
    p.add_argument("--n_eval_episodes", type=int, default=5)
    p.add_argument("--wandb_project",   default="hollow_box_rl")
    p.add_argument("--wandb_run",       default=None)
    args = p.parse_args()

    match args.mode:
        case "train":         train(args)
        case "eval":
            assert args.resume, "--resume 경로를 지정하세요"
            evaluate(args)
        case "debug_raycast": debug_raycast(args)
