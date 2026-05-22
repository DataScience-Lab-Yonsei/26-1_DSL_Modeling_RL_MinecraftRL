"""
mode_config.py — 세 가지 훈련 모드의 모든 설정을 정의합니다.

커리큘럼 학습 권장 순서:
  Phase 1 │ creative  │ 인벤토리 무한, 비행·즉시파괴 가능, 생존 압박 없음
           │           │ → 블록 배치 행동 시퀀스만 학습
  Phase 2 │ safe      │ 인벤토리 소모 있음, Peaceful(허기·몬스터 없음), 항상 낮
           │           │ → 재료 관리 학습
  Phase 3 │ survival  │ Normal 난이도, 낮밤 사이클, 몬스터 스폰
           │           │ → 완전한 생존 압박 속 건축
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeConfig:
    # ── craftground.make() 파라미터 ───────────────────────────────
    allow_mob_spawn: bool
    always_day:      bool
    always_night:    bool
    extra_cmds:      tuple[str, ...]

    # ── 에피소드 기본값 ────────────────────────────────────────────
    max_episode_steps: int

    # ── 보상 함수 플래그 ───────────────────────────────────────────
    use_health_penalty: bool   # 체력 감소 패널티 활성화
    use_food_penalty:   bool   # 허기 감소 패널티 활성화
    use_death_penalty:  bool   # 사망 패널티 활성화
    use_night_bonus:    bool   # 밤 생존 보너스 활성화

    # ── 인벤토리 ──────────────────────────────────────────────────
    give_inventory: bool  # False(creative)면 give 커맨드 생략


MODES: dict[str, ModeConfig] = {

    # ── Creative ─────────────────────────────────────────────────
    # gamemode creative : 인벤토리 무한, 즉시 파괴, 비행 가능
    # 생존 요소 전부 비활성 → 보상은 오직 "무엇을 어디에 놓았나"만
    "creative": ModeConfig(
        allow_mob_spawn = False,
        always_day      = True,
        always_night    = False,
        extra_cmds = (
            "gamemode creative @p",
            "gamerule doDaylightCycle false",
            "gamerule doMobSpawning false",
            "gamerule doWeatherCycle false",
        ),
        max_episode_steps  = 6000,    # 제약 없으니 짧게
        use_health_penalty = False,
        use_food_penalty   = False,
        use_death_penalty  = False,
        use_night_bonus    = False,
        give_inventory     = False,   # 인벤토리 무한 → give 불필요
    ),

    # ── Safe (Peaceful Survival) ─────────────────────────────────
    # gamemode survival + difficulty peaceful
    # 허기 없음, 몬스터 없음, 낮 고정 → 인벤토리 관리만 추가
    "safe": ModeConfig(
        allow_mob_spawn = False,
        always_day      = True,
        always_night    = False,
        extra_cmds = (
            "gamemode survival @p",
            "difficulty peaceful",
            "time set 6000",
            "gamerule doDaylightCycle false",
            "gamerule doMobSpawning false",
            "gamerule doWeatherCycle false",
        ),
        max_episode_steps  = 12000,
        use_health_penalty = False,   # Peaceful = 체력 자동 회복
        use_food_penalty   = False,   # Peaceful = 허기 안 닳음
        use_death_penalty  = False,
        use_night_bonus    = False,
        give_inventory     = True,
    ),

    # ── Survival ─────────────────────────────────────────────────
    # 완전한 Normal 서바이벌 + 낮밤 사이클 + 몬스터
    "survival": ModeConfig(
        allow_mob_spawn = True,
        always_day      = False,
        always_night    = False,
        extra_cmds = (
            "gamemode survival @p",
            "difficulty normal",
            "time set 6000",
            "gamerule doDaylightCycle true",
            "gamerule doMobSpawning true",
            "gamerule doWeatherCycle false",   # 날씨는 고정 (변수 줄이기)
        ),
        max_episode_steps  = 12000,
        use_health_penalty = True,
        use_food_penalty   = True,
        use_death_penalty  = True,
        use_night_bonus    = True,
        give_inventory     = True,
    ),
}

# 인벤토리 초기화 커맨드 (give_inventory=True인 모드에서 사용)
INITIAL_INVENTORY_CMDS: tuple[str, ...] = (
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:oak_planks 64",
    "give @p minecraft:cobblestone 64",
    "give @p minecraft:cobblestone 64",
    "give @p minecraft:oak_slab 32",
    "give @p minecraft:oak_door 4",
    "give @p minecraft:torch 16",
    "give @p minecraft:crafting_table 1",
    "give @p minecraft:furnace 1",
    "give @p minecraft:white_bed 1",
    "give @p minecraft:glass_pane 16",
    "give @p minecraft:wooden_pickaxe 1",
    "give @p minecraft:wooden_axe 1",
    "give @p minecraft:bread 16",
)
