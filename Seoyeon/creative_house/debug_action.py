"""
debug_action.py

no_op() 배열의 길이와 각 인덱스가 실제로 어떤 행동에 매핑되는지 확인합니다.
house_building_wrapper.py build_action()의 추론값 인덱스를 검증하세요.

  ✅ 확정 (pseudo_village_flat_rl.py에서 직접 확인)
     [0] 이동  [2] strafe-L  [5] strafe-R  [3] pitch  [4] yaw

  ⚠️ 추론 — 이 스크립트로 검증 필요
     [1] jump  [6] attack  [7] use/place  [8] sneak  [10] hotbar

사용법:
  python debug_action.py --port 8030

출력:
  no_op() 배열: [0, 0, 0, 12, 12, 0, 0, 0, 0, 0, 0]  (길이=11)
  인덱스 1  → JUMP    작동 여부: 점프 여부 육안 확인
  인덱스 7  → USE     작동 여부: 블록 설치 여부 육안 확인
  ...
"""

from __future__ import annotations
import argparse
import time

from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op


# 테스트할 인덱스 목록: (index, value, label, 확인 방법)
INFERRED_INDICES = [
    (2,  1, "JUMP",    "플레이어가 위로 점프하면 정상"),
    (5,  3, "ATTACK",  "앞 블록이 파괴되면 정상 (없으면 swing)"),
    (5,  1, "USE",     "아래 바닥에 블록이 설치되면 정상"),
    (2,  2, "SNEAK",   "플레이어가 웅크리면(내려가면) 정상"),
    (2,  3, "SPRINT",  "플레이어가 빠르게 앞으로 달리면 정상"),
    (6,  1, "HOTBAR_1","핫바 1번 슬롯으로 이동하면 정상 (아이템 교체 확인)"),
    (6,  2, "HOTBAR_2","핫바 2번 슬롯으로 이동하면 정상"),
]


def main(port: int):
    config = InitialEnvironmentConfig(
        image_width            = 114,
        image_height           = 64,
        seed                   = "42",
        world_type             = WorldType.SUPERFLAT,
        render_distance        = 4,
        simulation_distance    = 4,
        hud_hidden             = False,
        initial_extra_commands = [
            "gamemode creative @p",    # creative: 블록 무한 / 즉시 파괴 확인에 편함
            "gamerule doDaylightCycle false",
            "gamerule doMobSpawning false",
            "gamerule doWeatherCycle false",
            "gamerule doImmediateRespawn true",
            "weather clear",
            "time set 6000",
            "give @p minecraft:oak_planks 64",
        ],
    )

    env = make(
        initial_env_config = config,
        port               = port,
        verbose            = False,
        verbose_gradle     = True,
        render_action      = False,
    )

    raw_obs, _ = env.reset()

    # ── 1. no_op() 배열 구조 확인 ─────────────────────────────────
    base = no_op()
    print(f"\nno_op() 길이: {len(base)}")
    print(f"no_op() 기본값: {list(base)}")
    print()

    # ── 2. 확정 인덱스 재확인 ────────────────────────────────────
    print("=== 확정 인덱스 확인 (참조 코드에서 직접 확인된 것들) ===")
    confirmed = [
        (0,  1,  "FORWARD"),
        (0,  2,  "BACKWARD"),
        (1,  1,  "STRAFE_LEFT"),
        (1,  2,  "STRAFE_RIGHT"),
        (4, 11,  "CAMERA_LEFT (small)"),
        (4, 13,  "CAMERA_RIGHT (small)"),
        (3, 11,  "CAMERA_UP"),
        (3, 13,  "CAMERA_DOWN"),
    ]
    for idx, val, label in confirmed:
        act = no_op()
        act[idx] = val
        print(f"  act[{idx}]={val:2d} → {label}")

    # ── 3. 추론 인덱스 테스트 ─────────────────────────────────────
    print()
    print("=== 추론 인덱스 테스트 (3스텝씩 적용, 창에서 육안 확인) ===")
    print("※ render_action=False 이므로 Minecraft 창에서 직접 확인하세요.\n")

    for idx, val, label, hint in INFERRED_INDICES:
        print(f"  act[{idx}]={val} → {label}")
        print(f"    확인 방법: {hint}")

        # 5스텝 적용
        for _ in range(5):
            act = no_op()

            # LOOK_DOWN 먼저 (USE/ATTACK은 바닥/앞 블록 필요)
            if label in ("USE", "ATTACK"):
                act[3] = 13    # look down 먼저

            act[idx] = val
            env.step(act)
            time.sleep(0.05)

        # NO_OP 2스텝으로 간격
        for _ in range(2):
            env.step(no_op())
            time.sleep(0.05)

        input("    → 확인 후 Enter를 눌러 다음 인덱스로: ")

    env.close()

    print()
    print("✅ 테스트 완료.")
    print("   house_building_wrapper.py build_action() 의 추론값을 실제 값으로 수정하세요.")
    print()
    print("   현재 추론값:")
    print("     JUMP   → act[2] = 1")
    print("     ATTACK → act[5] = 3")
    print("     USE    → act[5] = 1")
    print("     SNEAK  → act[2] = 2")
    print("     SPRINT → act[2] = 3")
    print("     HOTBAR → act[6] = 1~9")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()
    main(args.port)
