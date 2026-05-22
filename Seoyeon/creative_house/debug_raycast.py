"""
debug_raycast.py

raycast_result(HitResult proto)의 실제 필드명을 확인합니다.
훈련 전 반드시 먼저 실행하세요.

사용법:
  # 로컬 (GUI)
  python debug_raycast.py --port 8030

  # 서버 (headless)
  xvfb-run -a python debug_raycast.py --port 8030

출력 예:
  HitResult 전체 필드:
    type                           = 1
    block_pos                      = x: 10  y: 4  z: 8
    block_state                    = minecraft:oak_planks

→ 출력을 보고 raycast_tracker.py 의 세 헬퍼를 수정하세요:
    _hit_type()   : 'type' 필드명 / int 매핑
    _hit_pos()    : 'block_pos' 필드명 구조
    _hit_state()  : 'block_state' / 'block_id' / 'translation_key' 중 사용할 것
"""

from __future__ import annotations
import argparse
import time

from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op


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
            "gamemode survival @p",
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

    # ── 1. ObservationSpaceMessage 전체 필드 출력 ─────────────────
    print("\n" + "=" * 60)
    print("ObservationSpaceMessage 전체 필드:")
    try:
        for f in raw_obs.DESCRIPTOR.fields:
            val = getattr(raw_obs, f.name, "N/A")
            print(f"  {f.name:30s} = {str(val)[:80]}")
    except AttributeError:
        print("  (DESCRIPTOR 없음 — obs 타입:", type(raw_obs), ")")
        if isinstance(raw_obs, dict):
            for k, v in raw_obs.items():
                print(f"  {k:30s} = {str(v)[:80]}")
        else:
            for attr in dir(raw_obs):
                if not attr.startswith("_"):
                    try:
                        print(f"  {attr:30s} = {str(getattr(raw_obs, attr))[:80]}")
                    except Exception:
                        pass

    # ── 2. raycast_result 필드 출력 ───────────────────────────────
    print("\n" + "=" * 60)
    hit = getattr(raw_obs, "raycast_result", None)
    if hit is None:
        print("⚠️  raycast_result 필드가 없습니다!")
        print("   obs 키 목록:", list(raw_obs.keys()) if isinstance(raw_obs, dict) else "N/A")
        env.close()
        return

    print("HitResult 전체 필드:")
    try:
        for f in hit.DESCRIPTOR.fields:
            val = getattr(hit, f.name, "N/A")
            print(f"  {f.name:30s} = {str(val)[:80]}")
    except AttributeError:
        print("  (DESCRIPTOR 없음 — dir() 출력)")
        for attr in dir(hit):
            if not attr.startswith("_"):
                try:
                    print(f"  {attr:30s} = {str(getattr(hit, attr))[:80]}")
                except Exception:
                    pass

    # ── 3. LOOK_DOWN → USE 후 raycast 상태 확인 ───────────────────
    print("\n" + "=" * 60)
    print("LOOK_DOWN 3스텝 → USE 3스텝 → raycast 관찰 (총 10스텝):")
    for step in range(10):
        act = no_op()
        if step < 3:
            act[3] = 13     # pitch down (확정 인덱스)
            label = "LOOK_DOWN"
        elif step < 6:
            act[7] = 1      # USE (추론 인덱스 — 여기서 실제 동작 확인)
            label = "USE(act[7])"
        else:
            label = "NO_OP"

        raw_obs, _, _, _, _ = env.step(act)
        hit2 = getattr(raw_obs, "raycast_result", None)
        if hit2:
            htype  = getattr(hit2, "type",        None) or getattr(hit2, "hit_type", None)
            bpos   = getattr(hit2, "block_pos",   None)
            bstate = (getattr(hit2, "block_state", None)
                   or getattr(hit2, "block_id",   None)
                   or getattr(hit2, "translation_key", None))
            print(f"  step={step:02d}  [{label}]  type={htype}  pos={bpos}  state={bstate}")
        else:
            print(f"  step={step:02d}  [{label}]  raycast_result=None")
        time.sleep(0.05)

    env.close()
    print("\n✅ 확인 완료.")
    print("   위 출력을 보고 raycast_tracker.py 의 헬퍼 함수를 수정하세요:")
    print("     _hit_type()  — type 필드명 & int 매핑")
    print("     _hit_pos()   — block_pos 구조")
    print("     _hit_state() — block_state / block_id / translation_key 선택")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()
    main(args.port)
