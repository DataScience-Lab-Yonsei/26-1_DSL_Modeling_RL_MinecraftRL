"""
house_rl.py — CraftGround 집 건축 강화학습 (PPO)

환경: CraftGround의 작은 평지(world_type=plain_small)에 스폰 → 블록 놓기로 집 짓기
목적: 바닥/벽/지붕/문/조명/가구 설치 → 집 완성 (enclosed)
보상 설계:
- 블록 설치 보상: 종류별 차등 + 연속 설치 보너스
- 마일스톤 보너스: 바닥/벽/지붕 단계별 + 문/조명/가구 + 집 완성
- 패널티: 행동 안 하면 소폭 패널티, USE 했는데 설치 안 되면 패널티, (생존 모드) 체력/배고픔 감소 패널티, 죽음 패널티

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

#try: import pygame; _PYGAME = True
#except ImportError: _PYGAME = False
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
INVENTORY_CMDS = (
    # 핫바 슬롯 0~5: 주요 블록 직접 배치 (에이전트가 즉시 사용 가능)
    "item replace entity @p hotbar.0 with minecraft:oak_planks 64",
    "item replace entity @p hotbar.1 with minecraft:cobblestone 64",
    "item replace entity @p hotbar.2 with minecraft:oak_slab 32",
    "item replace entity @p hotbar.3 with minecraft:oak_door 4",
    "item replace entity @p hotbar.4 with minecraft:torch 16",
    "item replace entity @p hotbar.5 with minecraft:white_bed 1",
    # 추가 인벤토리
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:cobblestone 64",
    "give @p minecraft:crafting_table 1",
    "give @p minecraft:furnace 1",
    "give @p minecraft:glass_pane 16",
    "give @p minecraft:bread 16",
)


# =================================================================
# 2. 블록 집합 & 마일스톤
# =================================================================

BUILDING_BLOCKS  = frozenset({"minecraft:oak_planks", "minecraft:cobblestone",
                               "minecraft:oak_slab",   "minecraft:stone",
                               "minecraft:oak_log",    "minecraft:glass_pane"})
DOOR_BLOCKS      = frozenset({"minecraft:oak_door", "minecraft:spruce_door"})
LIGHT_BLOCKS     = frozenset({"minecraft:torch", "minecraft:wall_torch", "minecraft:lantern"})
FURNITURE_BLOCKS = frozenset({"minecraft:crafting_table", "minecraft:furnace",
                               "minecraft:white_bed",     "minecraft:red_bed"})
ALL_HOUSE_BLOCKS = BUILDING_BLOCKS | DOOR_BLOCKS | LIGHT_BLOCKS | FURNITURE_BLOCKS

# (key, condition_fn, one-time bonus)
# 순서 유도: 바닥(floor) → 벽(wall) → 지붕(roof) → 문/조명/가구 → 집 완성
# 앞 단계가 달성돼야 다음 단계 보상이 의미 있도록 조건을 누적으로 설계
_MILESTONES = [
    # 1단계: 바닥 놓기 (floor >= 16 = 4×4)
    # HOUSE_HALF=3 기준: 7×7 외벽, 5×5 내부, 바닥=최대 49, 벽=외벽 둘레 3단=4*6*3=72, 지붕=49
    # 1단계: 바닥 놓기
    ("floor_4",       lambda s: s["floor"] >= 4,                               2.0),
    ("floor_16",      lambda s: s["floor"] >= 16,                              5.0),
    # 2단계: 4방향 외벽이 시작됨
    ("has_wall",      lambda s: s["has_wall"] and s["floor"] >= 4,             4.0),
    ("four_walls",    lambda s: s["four_walls"],                               8.0),  # 4면 모두 존재
    # 3단계: 지붕 덮기
    ("has_roof",      lambda s: s["has_roof"] and s["four_walls"],             5.0),
    ("roof_9",        lambda s: s["roof"] >= 9 and s["four_walls"],            8.0),
    # 4단계: 조명 (지붕+4면 벽 완성 후)
    ("has_light",     lambda s: s["has_light"] and s["has_roof"] and s["four_walls"],  2.0),
    # 5단계: 문 (조명 설치 후)
    ("has_door",      lambda s: s["has_door"] and s["has_light"],                      3.0),
    # 6단계: 가구 (문 설치 후)
    ("has_furniture", lambda s: s["has_furniture"] and s["has_door"],                  3.0),
    # 최종: 집 완성
    ("enclosed",      lambda s: s["enclosed"],                                        50.0),
]
MS_KEYS = [k for k, *_ in _MILESTONES]


# =================================================================
# 3. RaycastTracker
# =================================================================

class RaycastTracker:
    """raycast_result 기반 블록 설치 감지 + 구조물 완성도 추적."""

    def __init__(self): self.reset()

    HOUSE_HALF = 3   # 7×7 발자국 (exterior), 5×5 내부 공간 → 4×4 이상 내부 확보

    def reset(self, spawn_y: float = 4.0, spawn_x: float = 0.0, spawn_z: float = 0.0):
        self._placed: dict[tuple, tuple] = {}
        self._cache:  dict               = {}
        self._spawn_y = spawn_y
        self._spawn_x = spawn_x
        self._spawn_z = spawn_z
        self.milestones = {k: False for k, *_ in _MILESTONES}

    def update(self, raw_obs, action_was_use: bool, action_was_attack: bool = False) -> tuple[bool, str | None, bool]:
        """블록 설치/파괴 감지. (newly_placed, btype|None, newly_broken) 반환."""
        newly_broken = False
        # ATTACK으로 설치된 집 블록을 부쉈다면 tracker에서 제거
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
        if not pos or bid not in ALL_HOUSE_BLOCKS or pos in self._placed:
            return False, None, newly_broken
        btype = _classify(bid)
        self._placed[pos] = (btype, bid)
        self._cache.clear()
        return True, btype, newly_broken

    def _check_four_walls(self) -> bool:
        """4방향 외벽(perimeter wall height)에 각각 2개 이상의 블록이 있어야 진짜 enclosure."""
        H  = self.HOUSE_HALF
        sx = round(self._spawn_x)
        sz = round(self._spawn_z)
        sy = self._spawn_y
        sides = {"x_neg": 0, "x_pos": 0, "z_neg": 0, "z_pos": 0}
        for (bx, by, bz), (btype, _) in self._placed.items():
            if btype not in ("building", "door"): continue
            dy = by - sy
            if not (1 < dy <= 4): continue
            dx = bx - sx; dz = bz - sz
            if abs(dx) > H or abs(dz) > H: continue
            if dx == -H: sides["x_neg"] += 1
            if dx ==  H: sides["x_pos"] += 1
            if dz == -H: sides["z_neg"] += 1
            if dz ==  H: sides["z_pos"] += 1
        return all(v >= 2 for v in sides.values())

    def is_adjacent_to_placed(self, pos: tuple) -> bool:
        """설치 위치 6방향 중 하나에 기존 블록이 있으면 True (연결된 설치).
        update() 호출 후이므로 _placed에 이미 자신이 포함된 상태 → <=1이면 첫 블록.
        """
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
        floor = wall = roof = doors = lights = furn = 0
        for (_, y, _), (btype, _) in self._placed.items():
            if   y <= sy + 1: floor += 1
            elif y <= sy + 4: wall  += 1
            else:             roof  += 1
            if   btype == "door":      doors  += 1
            elif btype == "light":     lights += 1
            elif btype == "furniture": furn   += 1
        total = len(self._placed)
        four_walls = self._check_four_walls()
        # enclosed: flood-fill로 내부 공기가 외부로 탈출 불가한지 확인
        truly_enclosed = (floor >= 9 and four_walls and roof >= 9 and doors >= 1
                          and self._flood_fill_enclosed())
        self._cache = {
            "total": total, "floor": floor, "wall": wall, "roof": roof,
            "door": doors, "light": lights, "furniture": furn,
            "has_wall": wall >= 4,   "has_roof": roof >= 4,
            "has_door": doors > 0,   "has_light": lights > 0,
            "has_furniture": furn > 0,
            "four_walls": four_walls,
            "enclosed": truly_enclosed,
        }
        return self._cache

    def _flood_fill_enclosed(self) -> bool:
        """BFS flood-fill: 집 내부 공기가 외부로 탈출 가능하면 False (틈 있음).
        탈출 못하면 True (진짜 밀폐된 구조).
        y <= spawn_y 는 슈퍼플랫 바닥으로 간주(고체).
        """
        solid = set(self._placed.keys())
        sx = round(self._spawn_x)
        sz = round(self._spawn_z)
        sy = int(self._spawn_y)
        H  = self.HOUSE_HALF + 2   # 이 범위 밖으로 나가면 "탈출"로 판정

        # BFS 시작점: 스폰 내부 공기 (y = sy+1, 바닥 바로 위)
        start = None
        for dx in range(0, self.HOUSE_HALF):
            for dz in range(0, self.HOUSE_HALF):
                candidate = (sx + dx, sy + 1, sz + dz)
                if candidate not in solid:
                    start = candidate
                    break
            if start: break
        if start is None:
            return False  # 시작점을 찾을 수 없으면 밀폐 판정 불가

        visited = {start}
        queue   = [start]
        while queue:
            bx, by, bz = queue.pop(0)
            # 탈출 조건: x/z 범위 초과 or 지붕 위(높이 제한) or 지면 아래
            if abs(bx - sx) > H or abs(bz - sz) > H or by > sy + 8 or by <= sy:
                return False   # 탈출 가능 → 밀폐 안 됨
            for ddx, ddy, ddz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                nb = (bx+ddx, by+ddy, bz+ddz)
                if nb not in visited and nb not in solid and nb[1] > sy:
                    visited.add(nb)
                    queue.append(nb)
            if len(visited) > 600:   # 안전 한계: 너무 많이 퍼지면 틈 있음
                return False
        return True   # BFS 종료 = 탈출 못함 = 진짜 밀폐

    def block_zone(self, pos: tuple | None) -> str:
        """블록 위치가 목표 직사각형 구역 어디에 해당하는지 반환.
        반환값: 'floor' | 'wall' | 'wall_inside' | 'roof' | 'out'
        """
        if pos is None: return "out"
        bx, by, bz = pos
        H   = self.HOUSE_HALF
        sx  = round(self._spawn_x)
        sz  = round(self._spawn_z)
        sy  = self._spawn_y
        dx  = abs(bx - sx)
        dz  = abs(bz - sz)
        dy  = by - sy
        in_area   = dx <= H and dz <= H
        on_perim  = in_area and (dx == H or dz == H)
        if dy <= 1:        return "floor"      if in_area  else "out"
        if 1 < dy <= 4:    return "wall"       if on_perim else ("wall_inside" if in_area else "out")
        if dy > 4:         return "roof"       if in_area  else "out"
        return "out"

    def milestone_rewards(self) -> float:
        """달성된 마일스톤 보너스 합계 반환 (일회성)."""
        s = self.analyze(); bonus = 0.0
        for key, fn, val in _MILESTONES:
            if fn(s) and not self.milestones[key]:
                self.milestones[key] = True; bonus += val
        return bonus

    def raycast_vec(self, raw_obs) -> np.ndarray:
        """8-dim: [is_block, is_building, is_door, is_light, is_furniture, rel_x, rel_y, rel_z]"""
        vec = np.zeros(8, dtype=np.float32)
        hit = _get_hit(raw_obs)
        if hit is None or _hit_type(hit) != "block": return vec
        bid = _strip_state(_hit_state(hit))
        pos = _hit_pos(hit)
        vec[0] = 1.0
        if   bid in BUILDING_BLOCKS:  vec[1] = 1.0
        elif bid in DOOR_BLOCKS:      vec[2] = 1.0
        elif bid in LIGHT_BLOCKS:     vec[3] = 1.0
        elif bid in FURNITURE_BLOCKS: vec[4] = 1.0
        if pos:
            px = _scalar(raw_obs, "x"); py = _scalar(raw_obs, "y"); pz = _scalar(raw_obs, "z")
            vec[5:8] = np.clip([(pos[0]-px)/10., (pos[1]-py)/10., (pos[2]-pz)/10.], -1., 1.)
        return vec

    def struct_vec(self, st: dict) -> np.ndarray:
        return np.array([
            min(st["floor"]/50., 1.), min(st["wall"]/50., 1.), min(st["roof"]/30., 1.),
            float(st["four_walls"]),   # 4방향 외벽 형성 여부
            float(st["has_door"]), float(st["has_light"]), float(st["has_furniture"]),
            float(st["enclosed"]),     # 완전 밀폐 여부 (flood-fill 결과)
        ], dtype=np.float32)


# =================================================================
# 4. HitResult 헬퍼
# ⚠️  --mode debug_raycast 실행 후 실제 필드명으로 수정하세요.
# =================================================================

def _get_full(raw_obs):
    """dict 형태 obs에서 protobuf ObservationSpaceMessage 반환."""
    if isinstance(raw_obs, dict):
        return raw_obs.get("full", raw_obs)
    return raw_obs

def _get_hit(raw_obs):
    full = _get_full(raw_obs)
    return getattr(full, "raycast_result", None)

def _hit_type(hit) -> str:
    """type: BLOCK(1) / ENTITY(2) / MISS(0)"""
    raw = getattr(hit, "type", None)
    if raw is None: return "miss"
    return {0: "miss", 1: "block", 2: "entity"}.get(int(raw), "miss")

def _tk_to_block_id(tk: str) -> str:
    """'block.minecraft.oak_planks' → 'minecraft:oak_planks'"""
    if tk.startswith("block."):
        parts = tk[len("block."):].split(".", 1)
        if len(parts) == 2:
            return f"{parts[0]}:{parts[1]}"
    return tk

def _hit_pos(hit) -> tuple | None:
    """target_block 서브메시지에서 x,y,z 추출."""
    tb = getattr(hit, "target_block", None)
    if tb is not None:
        x = getattr(tb, "x", None)
        if x is not None:
            return (int(x), int(getattr(tb, "y", 0)), int(getattr(tb, "z", 0)))
    bp = getattr(hit, "block_pos", None)  # 구형 필드 fallback
    if bp is not None: return (int(bp.x), int(bp.y), int(bp.z))
    return None

def _hit_state(hit) -> str:
    """target_block.translation_key → 'minecraft:oak_planks' 형식."""
    tb = getattr(hit, "target_block", None)
    if tb is not None:
        tk = getattr(tb, "translation_key", "") or ""
        if tk: return _tk_to_block_id(tk)
    for attr in ("block_state", "block_id", "translation_key"):  # 구형 필드 fallback
        v = getattr(hit, attr, None)
        if v: return _tk_to_block_id(str(v))
    return ""

def _strip_state(s: str) -> str: return s.split("[")[0].strip()

def _classify(bid: str) -> str:
    if bid in DOOR_BLOCKS:      return "door"
    if bid in LIGHT_BLOCKS:     return "light"
    if bid in FURNITURE_BLOCKS: return "furniture"
    return "building"


# =================================================================
# 5. 액션
# =================================================================

ACTION_NAMES = [
    "NO_OP",
    "FORWARD", "BACKWARD", "LEFT", "RIGHT", "JUMP",
    "USE",
    "CAMERA_LEFT", "CAMERA_RIGHT", "CAMERA_UP", "CAMERA_DOWN",
    "HOTBAR_1", "HOTBAR_2", "HOTBAR_3", "HOTBAR_4", "HOTBAR_5",
    "HOTBAR_6",  # 슬롯 6까지만 (이후는 아이템 없음 → 인벤토리 UI 오작동 방지)
    "LOOK_DOWN_AND_USE",   # 매크로: 카메라 아래 + 블록 설치
    "LOOK_DOWN",           # 매크로: 카메라 아래
]

# 매크로 액션 (env.step에서 다중 스텝으로 처리)
MACRO_ACTIONS = {"LOOK_DOWN_AND_USE", "LOOK_DOWN"}
CAM_DEG = 10.0   # 카메라 1틱당 회전 각도

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
            slot = int(n.split("_")[1]) - 1  # HOTBAR_1→hotbar.0, HOTBAR_6→hotbar.5
            act[f"hotbar.{slot}"] = True
    return act


# =================================================================
# 6. 환경
# =================================================================

# translation_key prefix → btype 매핑
_TK_TO_BTYPE = {
    "block.minecraft.oak_planks":    "building",
    "block.minecraft.cobblestone":   "building",
    "block.minecraft.oak_slab":      "building",
    "block.minecraft.stone":         "building",
    "block.minecraft.oak_log":       "building",
    "block.minecraft.glass_pane":    "building",
    "block.minecraft.oak_door":      "door",
    "block.minecraft.spruce_door":   "door",
    "block.minecraft.torch":         "light",
    "block.minecraft.wall_torch":    "light",
    "block.minecraft.lantern":       "light",
    "block.minecraft.crafting_table":"furniture",
    "block.minecraft.furnace":       "furniture",
    "block.minecraft.white_bed":     "furniture",
    "block.minecraft.red_bed":       "furniture",
}

def _get_inv_counts(raw_obs) -> dict[str, int]:
    """translation_key → count 딕셔너리 반환."""
    full = _get_full(raw_obs)
    inv  = getattr(full, "inventory", [])
    return {item.translation_key: item.count for item in inv}

def _detect_placed_by_inv(prev_counts: dict, curr_counts: dict) -> tuple[bool, str | None]:
    """인벤토리 감소로 설치된 블록 감지. (placed, btype) 반환."""
    for tk, btype in _TK_TO_BTYPE.items():
        prev = prev_counts.get(tk, 0)
        curr = curr_counts.get(tk, 0)
        if curr < prev:
            return True, btype
    return False, None

BLOCK_REWARDS    = {"building": 1.0, "door": 4.0, "light": 3.0, "furniture": 4.0}
ISOLATED_PENALTY = -0.05   # 고립 설치 패널티 (FAILED_USE_PENALTY보다 작게)
WANDER_PENALTY   = -0.05   # 스폰 반경 초과 시 거리당 패널티
ALIVE_REWARD     = -0.002  # 아무것도 안 하면 손해 (너무 크면 음수 누적 → 0.002)
FAILED_USE_PENALTY = -0.05 # USE 했는데 블록 설치 안 됨 (빈번하므로 작게)
BREAK_PLACED_PENALTY = -2.0  # 이미 설치한 집 블록을 ATTACK으로 부수면 패널티
CONSEC_BONUS_K   =  0.3    # 연속 설치 시 보너스 (설치 횟수 × K)
NIGHT_BONUS      =  5.0
HEALTH_PENALTY_K = -0.5
FOOD_PENALTY_K   = -0.1
DEATH_PENALTY    = -20.0
REWARD_CLIP      = (-10.0, 10.0)
_MODE_COLOR_BGR  = {"creative": (80, 200, 255), "safe": (80, 255, 80), "survival": (80, 80, 255)}


def _scalar(obs, key: str, default: float = 0.0) -> float:
    full = _get_full(obs)
    if isinstance(full, dict): return float(full.get(key, default))
    return float(getattr(full, key, default))

def _extract_image(obs) -> np.ndarray | None:
    """protobuf / dict / ndarray → HWC RGB uint8"""
    if isinstance(obs, np.ndarray):
        img = obs
    elif isinstance(obs, dict):
        # pov/rgb 키 우선, 없으면 full.image (protobuf bytes)
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
    if img.ndim == 3 and img.shape[0] == 3:   # CHW → HWC
        img = img.transpose(1, 2, 0)
    return img.astype(np.uint8)

def _draw_hud(frame: np.ndarray, st: dict, ms: dict,
              action: str, mode: str, ep_r: float, step: int) -> np.ndarray:
    color    = _MODE_COLOR_BGR.get(mode, (200, 200, 200))
    enclosed = st.get("enclosed", False)
    lines = [
        (f"Step:{step:,}  Rwd:{ep_r:+.1f}  Act:{action}", (220, 220, 220)),
        (f"Blk:{st.get('total',0):3d}  Fl:{st.get('floor',0)} Wl:{st.get('wall',0)} Rf:{st.get('roof',0)}  "
         f"4W:{'Y' if st.get('four_walls') else 'n'}",
         (220, 220, 220)),
        (f"Door:{'Y' if ms.get('has_door') else 'n'}  "
         f"Light:{'Y' if ms.get('has_light') else 'n'}  "
         f"Furn:{'Y' if ms.get('has_furniture') else 'n'}  "
         f"ENCLOSED:{'YES!' if enclosed else 'no'}",
         (50, 255, 80) if enclosed else (180, 180, 180)),
        (f"Mode: {mode.upper()}", color),
    ]
    for i, (txt, c) in enumerate(lines):
        y = 18 + i * 20
        cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c,         1, cv2.LINE_AA)
    return frame


class HouseBuildingWrapper(gym.Wrapper):
    """
    Obs: {"image":(64,114,3) u8, "state":(8,) f32, "raycast":(8,) f32, "structure":(6,) f32}
    Act: Discrete(22) → build_action() → no_op() raw array
    info["render_frame"]: BGR ndarray (캡처 원본 해상도), HUD 포함
    """
    H, W   = 64, 114        # CNN obs 크기 (작게 유지)
    RH, RW = 180, 320       # 렌더 캡처 해상도 (인벤토리 보이는 크기)
    S      = 4              # 하위 호환용 (VideoRecorderCallback 등에서 참조)

    def __init__(self, env, cfg: ModeConfig, max_steps: int, mode_name: str = ""):
        super().__init__(env)
        self.cfg, self.max_steps, self.mode_name = cfg, max_steps, mode_name
        self.tracker = RaycastTracker()
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict({
            "image":     spaces.Box(0,   255, (self.H, self.W, 3), np.uint8),
            "state":     spaces.Box(-1., 1.,  (8,), np.float32),
            "raycast":   spaces.Box(-1., 1.,  (8,), np.float32),
            "structure": spaces.Box(0.,  1.,  (8,), np.float32),  # 6→8: four_walls+enclosed 추가
        })
        self._step = 0; self._ep_r = 0.0
        self._prev_health = self._prev_food = 20.0
        self._night_started = False
        self._consec_place = 0   # 연속 블록 설치 횟수
        self._prev_total   = 0   # 이전 스텝 총 블록 수 (fallback 감지용)
        self._prev_inv:  dict[str, int] = {}  # 인벤토리 이전 스텝 counts

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        self.tracker.reset(spawn_y=_scalar(raw, "y", 4.0),
                           spawn_x=_scalar(raw, "x", 0.0),
                           spawn_z=_scalar(raw, "z", 0.0))
        self._step = 0; self._ep_r = 0.0
        self._prev_health = self._prev_food = 20.0
        self._night_started = False
        self._consec_place = 0
        self._prev_total   = 0
        self._prev_inv     = _get_inv_counts(raw)
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
        # 1차: raycast 기반 감지
        placed, btype, broken = self.tracker.update(raw, is_use, is_attack)
        # 2차: 인벤토리 감소 기반 감지 (raycast 실패 시 보완)
        curr_inv = _get_inv_counts(raw)
        if not placed and is_use:
            inv_placed, inv_btype = _detect_placed_by_inv(self._prev_inv, curr_inv)
            if inv_placed:
                placed = True
                btype  = inv_btype or "building"
        self._prev_inv = curr_inv
        st  = self.tracker.analyze()
        self._prev_total = st["total"]
        raw_img = _extract_image(raw)
        obs = self._make_obs(raw, st, raw_img)
        rew = self._compute_reward(raw, terminated or truncated, placed, btype, name, broken)
        self._ep_r += rew

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
        # 캡처 원본(320×180)을 그대로 사용 — 인벤토리/HUD 선명하게 보임
        frame = cv2.cvtColor(
            cv2.resize(img_rgb, (self.RW, self.RH), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_RGB2BGR)
        return _draw_hud(frame, st, ms, action, self.mode_name, self._ep_r, self._step)

    def _compute_reward(self, raw, done: bool, placed: bool, btype: str | None, name: str = "", broken: bool = False) -> float:
        r = 0.0 if done else ALIVE_REWARD

        # 활동반경 패널티: 스폰에서 HOUSE_HALF+1 이상 멀어지면 거리에 비례해 패널티
        ax  = _scalar(raw, "x")
        az  = _scalar(raw, "z")
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

        # pitch 보상: USE 시 하늘 보면 소폭 패널티
        if name in ("USE", "LOOK_DOWN_AND_USE"):
            pitch = _scalar(raw, "pitch")
            if pitch < 0: r -= 0.05

        if placed and btype is not None:
            base_r = BLOCK_REWARDS.get(btype, 1.0)
            hit = _get_hit(raw)
            pos = _hit_pos(hit) if hit is not None else None
            if btype == "building" and pos is not None:
                if not self.tracker.is_adjacent_to_placed(pos):
                    # 고립 설치: 아무것도 안 하는 것(-0.31)보다 낫지만 패널티
                    # zone 보너스 없이 패널티만 부여 (첫 블록은 is_adjacent_to_placed가 True 반환)
                    base_r = ISOLATED_PENALTY
                else:
                    # 인접 설치: pos가 있을 때만 구역 보정 (인벤토리 감지는 pos 없음 → 보정 생략)
                    zone = self.tracker.block_zone(pos)
                    st   = self.tracker.analyze()
                    if zone == "floor":
                        base_r += 0.5
                    elif zone == "wall":
                        base_r += 0.0 if st["floor"] < 4 else 1.5
                    elif zone == "wall_inside":
                        base_r -= 2.0  # 내부 공간을 채우면 강한 패널티
                    elif zone == "roof":
                        base_r += 0.0 if not st["four_walls"] else 1.0
                    elif zone == "out":
                        base_r  = -2.0  # 하우스 범위 밖 설치 강한 패널티
            elif btype == "light":
                # 4단계: 지붕+4면 벽 완성 후에만 조명 보상
                st = self.tracker.analyze()
                if not (st["has_roof"] and st["four_walls"]):
                    base_r = -1.5  # 순서 위반 패널티
            elif btype == "door":
                # 5단계: 조명 설치 후에만 문 보상
                st = self.tracker.analyze()
                if not st["has_light"]:
                    base_r = -1.5  # 순서 위반 패널티
            elif btype == "furniture":
                # 6단계: 문 설치 후에만 가구 보상
                st = self.tracker.analyze()
                if not st["has_door"]:
                    base_r = -1.5  # 순서 위반 패널티
            # 연속 설치 보너스
            self._consec_place += 1
            base_r += min(self._consec_place * CONSEC_BONUS_K, 3.0)
            r += base_r
        else:
            self._consec_place = 0   # 설치 안 하면 연속 카운터 리셋
            # 실패한 USE 패널티
            if name in ("USE", "LOOK_DOWN_AND_USE"):
                r += FAILED_USE_PENALTY

        r += self.tracker.milestone_rewards()  # enclosed 포함, 일회성 자동 처리

        if self.cfg.use_night_bonus:
            wtime = int(_scalar(raw, "world_time")) % 24000
            if 13000 <= wtime <= 23000: self._night_started = True
            elif self._night_started:   r += NIGHT_BONUS; self._night_started = False

        return float(np.clip(r, *REWARD_CLIP))


def make_env(port=8030, mode="creative", max_steps=None, seed=42) -> HouseBuildingWrapper:
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
    return HouseBuildingWrapper(env, cfg=cfg,
                                max_steps=max_steps or cfg.max_episode_steps,
                                mode_name=mode)


# =================================================================
# 7. 모델
# =================================================================

class HouseCNNExtractor(BaseFeaturesExtractor):
    """CNN(image) + MLP(state+raycast+structure) → features_dim=256"""

    def __init__(self, obs_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(obs_space, features_dim)
        # SB3 VecTransposeImage가 HWC(H,W,C) → CHW(C,H,W)로 자동 변환
        img_shape = obs_space["image"].shape
        if img_shape[-1] in (1, 3, 4):  # HWC
            h, w, c = img_shape
        else:  # CHW (VecTransposeImage 적용 후)
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
        # CHW면 그대로, HWC면 permute (SB3 preprocess_obs가 이미 /255 정규화함)
        if img_raw.shape[-1] in (1, 3, 4):  # HWC
            img = img_raw.permute(0, 3, 1, 2) / 255.0
        else:  # CHW — SB3이 이미 [0,1]로 정규화
            img = img_raw
        vec = torch.cat([obs[k].float() for k in ("state", "raycast", "structure")], dim=-1)
        return self.fusion(torch.cat([self.cnn(img), self.mlp(vec)], dim=-1))


# =================================================================
# 8. 하이퍼파라미터
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
# 9. 콜백
# =================================================================

class TrackingCallback(BaseCallback):
    """TensorBoard + WandB 로그 + 주기적 모델 artifact 업로드.
    (HouseProgressCallback + WandBCallback 통합)
    """

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
        # raycast 디버그: 처음 200 스텝만 출력
        if self.num_timesteps <= 200 and self.num_timesteps % 50 == 0:
            env = self.training_env
            try:
                raw_env = env.envs[0].env  # DummyVecEnv → HouseBuildingWrapper → inner env
                obs     = env.envs[0].env.env  # 한 단계 더 내려갈 수도 있음
            except Exception:
                raw_env = None
            infos = self.locals.get("infos", [{}])
            st    = infos[0].get("structure", {})
            print(f"[RaycastDebug t={self.num_timesteps}] "
                  f"total={st.get('total',0)} floor={st.get('floor',0)} "
                  f"wall={st.get('wall',0)} roof={st.get('roof',0)}")
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
        enc = float(np.mean([e["st"].get("enclosed", False) for e in recent]))
        blk = float(np.mean([e["st"].get("total",    0)     for e in recent]))
        ms  = {k: float(np.mean([e["ms"].get(k, False) for e in recent])) for k in MS_KEYS}
        for k, v in ms.items(): self.logger.record(f"house/ms_{k}", v)
        self.logger.record("house/enclosed_rate", enc)
        self.logger.record("house/avg_blocks",    blk)
        if _wactive():
            _wlog({"rollout/enclosed_rate": enc, "rollout/avg_blocks": blk,
                   **{f"house/ms_{k}": v for k, v in ms.items()}},
                  step=self.num_timesteps)
        if self.verbose:
            print(f"\n[{self.num_timesteps:,}] blk={blk:.1f}  enclosed={enc:.2f}")


class RenderCallback(BaseCallback):
    """info["render_frame"] → cv2.imshow 창 실시간 표시 (pygame 불필요).
    'q' 키 또는 창 닫기로 렌더링 비활성화 (학습은 계속).
    """
    WIN = "House Builder"

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
            # frame이 없으면 빈 검정 화면이라도 표시해서 창 유지
            frame = __import__('numpy').zeros((HouseBuildingWrapper.H * HouseBuildingWrapper.S,
                                               HouseBuildingWrapper.W * HouseBuildingWrapper.S, 3),
                                              dtype='uint8')
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
    """N 롤아웃마다 eval 에피소드를 mp4 녹화 → WandB 업로드 (활성 시).
    env는 첫 녹화 시점에 lazy 초기화 (CraftGround 인스턴스 최소화).
    """
    RW = HouseBuildingWrapper.RW
    RH = HouseBuildingWrapper.RH

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
            if self._env is None: self._env = self._make()  # lazy init
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
                "video/eval_blocks":   info.get("structure", {}).get("total",    0),
                "video/eval_enclosed": int(info.get("structure", {}).get("enclosed", False)),
            }, step=self.num_timesteps)

    def _on_training_end(self):
        # 학습 종료 시 최종 에피소드 자동 저장
        try:
            if self._env is None: self._env = self._make()
            self._ridx = 9999  # 최종 저장임을 파일명으로 구분
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
                    "video/final_blocks":   info.get("structure", {}).get("total",    0),
                    "video/final_enclosed": int(info.get("structure", {}).get("enclosed", False)),
                })
        except Exception as e:
            print(f"⚠️  최종 영상 저장 실패: {e}")
        finally:
            if self._env: self._env.close()


# =================================================================
# 10. 훈련 / 평가 / 디버그
# =================================================================

def train(args):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = Path(args.log_dir)  / f"{args.env_mode}_{ts}"
    save_dir = Path(args.save_dir) / f"{args.env_mode}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    hp = HP[args.env_mode]

    if args.wandb_project:
        if not _WANDB: print("⚠️  wandb 미설치 → pip install wandb")
        else:
            wandb.init(project=args.wandb_project,
                       name=args.wandb_run or f"ppo_{args.env_mode}_{ts}",
                       sync_tensorboard=True, save_code=True,
                       config={"env_mode": args.env_mode, "total_steps": args.total_steps,
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
        print(f"🔄 로드: {args.resume}")
    else:
        model = PPO(
            "MultiInputPolicy", vec_env, tensorboard_log=str(log_dir),
            verbose=1, device=args.device, **hp,
            policy_kwargs=dict(
                features_extractor_class=HouseCNNExtractor,
                features_extractor_kwargs={"features_dim": 256},
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=nn.ReLU,
            ),
        )

    callbacks = [
        CheckpointCallback(max(10_000 // args.n_envs, 1), str(save_dir), f"house_{args.env_mode}"),
        EvalCallback(eval_env, best_model_save_path=str(save_dir / "best"),
                     log_path=str(log_dir / "eval"), eval_freq=max(20_000 // args.n_envs, 1),
                     n_eval_episodes=3, deterministic=True, verbose=1),
        TrackingCallback(log_freq=2000, ckpt_freq=20, save_dir=str(save_dir),
                         mode=args.env_mode, verbose=1),
        RenderCallback(freq=4),
        VideoRecorderCallback(make_fn(200), freq=10, video_dir=str(log_dir / "videos")),
    ]

    print(f"\n🌍 {args.env_mode.upper()}  |  {args.total_steps:,} steps  |  {args.n_envs} envs\n")
    try:
        model.learn(args.total_steps, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=not bool(args.resume))
    finally:
        final = save_dir / f"house_{args.env_mode}_final"
        model.save(str(final))
        print(f"\n✅ 저장: {final}.zip")
        if _wactive():
            art = wandb.Artifact(f"ppo_{args.env_mode}_final", type="model")
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
              f"blk={st.get('total',0)}  enclosed={st.get('enclosed',False)}")
    print(f"\n평균: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    env.close()


def _make_debug_env(port, gamemode="survival"):
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
    """raycast_result proto 실제 필드명 확인. 훈련 전 필수."""
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

    # dict / proto 양쪽 모두 raycast 키 탐색
    hit = None
    if isinstance(raw, dict):
        for k in raw:
            if "raycast" in k.lower():
                hit = raw[k]
                print(f"\n→ raycast 키 발견: '{k}'  타입: {type(hit)}")
                break
        if hit is None:
            print("\n⚠️  dict에 raycast 관련 키 없음. 전체 키:", list(raw.keys()))
    else:
        hit = getattr(raw, "raycast_result", None)

    if hit is None:
        print("\n⚠️  raycast_result 없음 — 'full' 키 내부를 검사합니다.")
        raw2 = raw
        for i in range(5):
            act = no_op_v2(); act["camera_pitch"] = CAM_DEG
            raw2, *_ = env.step(act)
        full = raw2.get("full") if isinstance(raw2, dict) else None
        if full is not None:
            print("  full 타입:", type(full))
            if isinstance(full, dict):
                for k, v in full.items():
                    print(f"    full[{k}] = {str(v)[:120]}")
            else:
                try:
                    for f in full.DESCRIPTOR.fields:
                        print(f"    {f.name:30s} = {str(getattr(full, f.name, 'N/A'))[:80]}")
                except Exception:
                    print("  full attrs:", [a for a in dir(full) if not a.startswith("_")])
        else:
            print("  'full' 키도 없음. 사용 가능한 키:", list(raw2.keys()) if isinstance(raw2, dict) else type(raw2))
        env.close(); return

    print("\n=== HitResult / raycast 값 ===")
    if isinstance(hit, dict):
        for k, v in hit.items():
            print(f"  [{k}] = {str(v)[:120]}")
    else:
        try:
            for f in hit.DESCRIPTOR.fields:
                print(f"  {f.name:30s} = {str(getattr(hit, f.name, 'N/A'))[:80]}")
        except AttributeError:
            for attr in dir(hit):
                if not attr.startswith("_"):
                    try: print(f"  {attr:30s} = {str(getattr(hit, attr))[:80]}")
                    except Exception: pass

    print("\n=== HitResult 필드 ===")
    try:
        for f in hit.DESCRIPTOR.fields:
            print(f"  {f.name:30s} = {str(getattr(hit, f.name, 'N/A'))[:80]}")
    except AttributeError:
        for attr in dir(hit):
            if not attr.startswith("_"):
                try: print(f"  {attr:30s} = {str(getattr(hit, attr))[:80]}")
                except Exception: pass

    print("\n=== LOOK_DOWN → USE 시퀀스 (V2) ===")
    for i in range(10):
        act = no_op_v2()
        if i < 3:   act["camera_pitch"] = CAM_DEG; label = "LOOK_DOWN"
        elif i < 6: act["use"] = True;              label = "USE"
        else:       label = "NO_OP"
        raw, *_ = env.step(act)
        full = _get_full(raw)
        h    = getattr(full, "raycast_result", None)
        bstate = None
        if h:
            bstate = next((getattr(h, a, None)
                           for a in ("block_state", "block_id", "translation_key")
                           if getattr(h, a, None) is not None), None)
        hit_type_val  = getattr(h, "type",     None) if h else None
        block_pos_val = getattr(h, "block_pos", None) if h else None
        print(f"  [{i:02d}] {label:15s} type={hit_type_val}  block_pos={block_pos_val}  state={bstate}")
        time.sleep(0.05)

    print("\n=== 감지 헬퍼 실제 동작 확인 ===")
    for i in range(5):
        act = no_op_v2(); act["camera_pitch"] = CAM_DEG
        raw, *_ = env.step(act)
    act = no_op_v2(); act["use"] = True
    raw, *_ = env.step(act)
    hit = _get_hit(raw)
    print(f"  _get_hit       : {hit}")
    print(f"  _hit_type      : {_hit_type(hit) if hit else 'N/A'}")
    print(f"  _hit_pos       : {_hit_pos(hit) if hit else 'N/A'}")
    print(f"  _hit_state     : {_hit_state(hit) if hit else 'N/A'}")
    print(f"  block in BLOCKS: {_strip_state(_hit_state(hit)) in ALL_HOUSE_BLOCKS if hit else False}")

    env.close()
    print("\n→ type=='block' 이고 block_pos 값이 나오면 감지 정상.")
    print("→ 모두 None이면 _hit_type/_hit_pos/_hit_state 헬퍼 필드명 수정 필요.")


def debug_action(args):
    """V2 액션 키 인터랙티브 검증."""
    env = _make_debug_env(args.base_port, "creative")
    env.reset()
    print(f"no_op_v2() 키: {list(no_op_v2().keys())}\n")
    for key, val, label, hint in [
        ("jump",      True,  "JUMP",     "점프하면 정상"),
        ("attack",    True,  "ATTACK",   "블록 파괴/swing이면 정상"),
        ("use",       True,  "USE",      "블록 설치되면 정상 ← 핵심"),
        ("hotbar.1",  True,  "HOTBAR_1", "핫바 1번 이동"),
        ("hotbar.2",  True,  "HOTBAR_2", "핫바 2번 이동"),
        ("camera_pitch", CAM_DEG, "LOOK_DOWN", "카메라 아래"),
    ]:
        print(f"  '{key}'={val} → {label}  |  확인: {hint}")
        for _ in range(5):
            act = no_op_v2()
            if label in ("USE", "ATTACK"): act["camera_pitch"] = CAM_DEG
            act[key] = val
            env.step(act); time.sleep(0.05)
        for _ in range(2): env.step(no_op_v2()); time.sleep(0.05)
        input("  → Enter로 다음: ")
    env.close()
    print("\n→ V2 액션 검증 완료.")


# =================================================================
# 엔트리포인트
# =================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CraftGround 집 건축 PPO")
    p.add_argument("--mode",      default="train",
                   choices=["train", "eval", "debug_raycast", "debug_action"])
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
    p.add_argument("--wandb_project",   default="house_rl")
    p.add_argument("--wandb_run",       default=None)
    args = p.parse_args()

    match args.mode:
        case "train":         train(args)
        case "eval":
            assert args.resume, "--resume 경로를 지정하세요"
            evaluate(args)
        case "debug_raycast": debug_raycast(args)
        case "debug_action":  debug_action(args)