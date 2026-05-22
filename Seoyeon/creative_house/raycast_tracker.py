"""
raycast_tracker.py

raycast_result(HitResult proto)를 이용해 블록 설치를 감지하고
구조물 완성도를 추적합니다.

── HitResult 필드 (MinecraftEnv Fabric 소스 기반 추론) ──────────────
  type        : int   0=MISS, 1=BLOCK, 2=ENTITY
  block_pos   : BlockPos {x, y, z}    (type==BLOCK 일 때 유효)
  block_state : str   "minecraft:oak_planks" 등 ([] 포함 가능)

⚠️  실제 필드명은 debug_raycast.py 로 먼저 확인하세요.
    _hit_type / _hit_pos / _hit_state 헬퍼만 수정하면 됩니다.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── 블록 분류 ──────────────────────────────────────────────────────
BUILDING_BLOCKS = frozenset({
    "minecraft:oak_planks",  "minecraft:cobblestone",
    "minecraft:oak_slab",    "minecraft:stone",
    "minecraft:oak_log",     "minecraft:glass_pane",
})
DOOR_BLOCKS = frozenset({
    "minecraft:oak_door",    "minecraft:spruce_door",
})
LIGHT_BLOCKS = frozenset({
    "minecraft:torch",       "minecraft:wall_torch", "minecraft:lantern",
})
FURNITURE_BLOCKS = frozenset({
    "minecraft:crafting_table", "minecraft:furnace",
    "minecraft:white_bed",      "minecraft:red_bed",
})
ALL_HOUSE_BLOCKS = BUILDING_BLOCKS | DOOR_BLOCKS | LIGHT_BLOCKS | FURNITURE_BLOCKS


@dataclass
class PlacedBlock:
    x: int; y: int; z: int
    block_type: str   # "building" | "door" | "light" | "furniture"
    block_id:   str   # "minecraft:oak_planks" 등


@dataclass
class RaycastTracker:
    """
    매 스텝 raycast_result 를 보고:
      1. obs용 8-dim 벡터 인코딩 (현재 바라보는 블록)
      2. USE 직후 새 건축 블록 설치 감지
      3. 누적 placed_blocks 로 구조물 완성도 평가
    """
    placed_blocks:    dict = field(default_factory=dict)
    milestones:       dict = field(default_factory=dict)
    _spawn_y:         float = 4.0
    _structure_cache: dict  = field(default_factory=dict)

    def reset(self, spawn_y: float = 4.0):
        self.placed_blocks.clear()
        self._structure_cache.clear()
        self._spawn_y = spawn_y
        self.milestones = {
            "placed_5":      False,
            "placed_20":     False,
            "placed_50":     False,
            "placed_100":    False,
            "has_wall":      False,
            "has_roof":      False,
            "has_door":      False,
            "has_light":     False,
            "has_furniture": False,
        }

    # ── 1. obs 인코딩 (8-dim float32) ────────────────────────────
    def encode_current(self, raw_obs) -> np.ndarray:
        """
        [is_block, is_building, is_door, is_light, is_furniture,
         rel_x/10,  rel_y/10,   rel_z/10]
        """
        vec = np.zeros(8, dtype=np.float32)
        hit = _get_hit(raw_obs)
        if hit is None or _hit_type(hit) != "block":
            return vec

        bid = _strip_state(_hit_state(hit))
        pos = _hit_pos(hit)

        vec[0] = 1.0
        if   bid in BUILDING_BLOCKS:  vec[1] = 1.0
        elif bid in DOOR_BLOCKS:      vec[2] = 1.0
        elif bid in LIGHT_BLOCKS:     vec[3] = 1.0
        elif bid in FURNITURE_BLOCKS: vec[4] = 1.0

        if pos is not None:
            if isinstance(raw_obs, dict):
                px = float(raw_obs.get("x", 0.0))
                py = float(raw_obs.get("y", 0.0))
                pz = float(raw_obs.get("z", 0.0))
            else:
                px = float(getattr(raw_obs, "x", 0.0))
                py = float(getattr(raw_obs, "y", 0.0))
                pz = float(getattr(raw_obs, "z", 0.0))
            vec[5] = float(np.clip((pos[0] - px) / 10.0, -1, 1))
            vec[6] = float(np.clip((pos[1] - py) / 10.0, -1, 1))
            vec[7] = float(np.clip((pos[2] - pz) / 10.0, -1, 1))

        return vec

    # ── 2. 설치 감지 ────────────────────────────────────────────
    def update(self, raw_obs, action_was_use: bool) -> dict:
        """
        USE 직후 raycast 위치에 새 건축 블록이 생기면 설치로 판정.
        Returns {"newly_placed": bool, "block_type": str|None}
        """
        result = {"newly_placed": False, "block_type": None}
        if not action_was_use:
            return result

        hit = _get_hit(raw_obs)
        if hit is None or _hit_type(hit) != "block":
            return result

        bid = _strip_state(_hit_state(hit))
        pos = _hit_pos(hit)

        if pos is None or bid not in ALL_HOUSE_BLOCKS:
            return result

        key = tuple(pos)
        if key in self.placed_blocks:
            return result

        btype = _classify(bid)
        self.placed_blocks[key] = PlacedBlock(
            x=pos[0], y=pos[1], z=pos[2], block_type=btype, block_id=bid
        )
        self._structure_cache.clear()
        result["newly_placed"] = True
        result["block_type"]   = btype
        return result

    # ── 3. 구조물 분석 ───────────────────────────────────────────
    def analyze_structure(self) -> dict:
        if self._structure_cache:
            return self._structure_cache

        blocks = list(self.placed_blocks.values())
        n      = len(blocks)
        sy     = self._spawn_y

        floor = [b for b in blocks if b.y <= sy + 1]
        wall  = [b for b in blocks if sy + 1 < b.y <= sy + 4]
        roof  = [b for b in blocks if b.y > sy + 3]
        doors = [b for b in blocks if b.block_type == "door"]
        lights= [b for b in blocks if b.block_type == "light"]
        furn  = [b for b in blocks if b.block_type == "furniture"]

        enclosed = (
            len(floor) >= 4 and len(wall) >= 8 and
            len(roof)  >= 4 and len(doors) >= 1
        )

        s = {
            "total": n,
            "floor": len(floor), "wall":  len(wall),  "roof":      len(roof),
            "door":  len(doors), "light": len(lights), "furniture": len(furn),
            "has_wall":      len(wall)  >= 4,
            "has_roof":      len(roof)  >= 4,
            "has_door":      len(doors) > 0,
            "has_light":     len(lights) > 0,
            "has_furniture": len(furn)  > 0,
            "enclosed":      enclosed,
        }
        self._structure_cache = s
        return s

    def compute_milestone_rewards(self) -> tuple[float, dict]:
        s = self.analyze_structure()
        bonus, info = 0.0, {}
        checks = [
            ("placed_5",      s["total"] >= 5,   1.0),
            ("placed_20",     s["total"] >= 20,  2.0),
            ("placed_50",     s["total"] >= 50,  4.0),
            ("placed_100",    s["total"] >= 100, 6.0),
            ("has_wall",      s["has_wall"],      2.0),
            ("has_roof",      s["has_roof"],      3.0),
            ("has_door",      s["has_door"],      2.0),
            ("has_light",     s["has_light"],     1.5),
            ("has_furniture", s["has_furniture"], 2.0),
        ]
        for key, cond, val in checks:
            if cond and not self.milestones.get(key, False):
                self.milestones[key] = True
                bonus += val
                info[f"ms_{key}"] = val
        return bonus, info

    def get_obs_summary(self) -> np.ndarray:
        """6-dim: [floor/50, wall/50, roof/30, has_door, has_light, has_furniture]"""
        s = self.analyze_structure()
        return np.array([
            min(s["floor"]  / 50.0, 1.0),
            min(s["wall"]   / 50.0, 1.0),
            min(s["roof"]   / 30.0, 1.0),
            float(s["has_door"]),
            float(s["has_light"]),
            float(s["has_furniture"]),
        ], dtype=np.float32)


# ── HitResult 필드 접근 헬퍼 ──────────────────────────────────────
# ⚠️ debug_raycast.py 실행 후 실제 필드명으로 수정하세요.

def _get_hit(raw_obs):
    return getattr(raw_obs, "raycast_result", None)

def _hit_type(hit) -> str:
    """proto HitResult.type → "miss" | "block" | "entity" """
    if hit is None:
        return "miss"
    raw = getattr(hit, "type", None) or getattr(hit, "hit_type", None)
    if raw is None:
        return "miss"
    if isinstance(raw, int):
        return {0: "miss", 1: "block", 2: "entity"}.get(raw, "miss")
    return str(raw).lower().split(".")[-1]

def _hit_pos(hit) -> Optional[tuple]:
    """proto HitResult.block_pos → (x, y, z) int tuple"""
    if hit is None:
        return None
    bp = getattr(hit, "block_pos", None)
    if bp is not None:
        return (int(bp.x), int(bp.y), int(bp.z))
    x = getattr(hit, "x", None)
    if x is not None:
        return (int(x), int(getattr(hit, "y", 0)), int(getattr(hit, "z", 0)))
    return None

def _hit_state(hit) -> str:
    """proto HitResult → block id 문자열"""
    if hit is None:
        return ""
    for attr in ("block_state", "block_id", "translation_key"):
        v = getattr(hit, attr, None)
        if v:
            return str(v)
    return ""

def _strip_state(s: str) -> str:
    """ "minecraft:oak_door[facing=east,...]" → "minecraft:oak_door" """
    return s.split("[")[0].strip()

def _classify(bid: str) -> str:
    if bid in DOOR_BLOCKS:      return "door"
    if bid in LIGHT_BLOCKS:     return "light"
    if bid in FURNITURE_BLOCKS: return "furniture"
    return "building"
