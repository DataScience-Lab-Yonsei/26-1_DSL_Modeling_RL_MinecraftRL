"""
지형 평탄화 데이터 수집.

사용법:
    cd Changyong/terrain_flatten
    python collect_demos.py [에피소드 수]
    ex) python collect_demos.py 20

저장 형식:
    demos/demos.npz
        observations : (N, 18) float32
        actions      : (N, 7)  int64   ← MultiDiscrete 7차원
"""
import os
import sys
import time
import numpy as np

from config import PORT, DEMO_DIR, DEMO_PATH, COLLECT_N_EPISODES
from env    import FlattenEnv, TOTAL_TARGET_BLOCKS
from expert import FlattenExpert

# ── cv2 미리보기 ─────────────────────────────────────────────────
try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

_WINDOW = "Flatten Expert — POV"
_PREVIEW_W = 480
_PREVIEW_H = 360


def _show_frame(env: FlattenEnv, expert: FlattenExpert,
                ep: int, step: int, cleared: int):
    """에이전트 POV를 cv2 창으로 표시."""
    if not _CV2_OK:
        return

    raw = env._cg_obs
    if raw is None:
        return

    # POV 이미지 추출
    img = None
    try:
        if isinstance(raw, dict):
            img = raw.get("pov") or raw.get("rgb")
        if img is None:
            return
        img = np.asarray(img, dtype=np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:   # CHW → HWC
            img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (_PREVIEW_W, _PREVIEW_H),
                         interpolation=cv2.INTER_NEAREST)
    except Exception:
        return

    # 오버레이 텍스트
    hit = env._parse_raycast(raw)
    hit_str = "miss"
    if hit is not None:
        bx, by, bz = hit["position"]
        above = "↑MINE" if hit["above_target"] else "ok"
        hit_str = f"({bx},{by},{bz}) {above}"

    lines = [
        f"EP {ep}  step {step}",
        f"cleared: {cleared}/{TOTAL_TARGET_BLOCKS}",
        f"state: {expert._state}",
        f"pos: ({env.agent_x:.1f}, {env.agent_y:.1f}, {env.agent_z:.1f})",
        f"ray: {hit_str}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1,
                    cv2.LINE_AA)

    # 채굴 진행률 바
    bar_w = int(_PREVIEW_W * cleared / max(TOTAL_TARGET_BLOCKS, 1))
    cv2.rectangle(img, (0, _PREVIEW_H - 8), (_PREVIEW_W, _PREVIEW_H),
                  (40, 40, 40), -1)
    cv2.rectangle(img, (0, _PREVIEW_H - 8), (bar_w, _PREVIEW_H),
                  (0, 200, 100), -1)

    if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(_WINDOW, img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise KeyboardInterrupt("사용자가 q를 눌러 종료")


def create_env(port: int = PORT) -> FlattenEnv:
    """craftground 기본 환경 생성 → FlattenEnv 래핑."""
    import craftground
    from craftground import InitialEnvironmentConfig, ActionSpaceVersion
    from craftground.initial_environment_config import (
        DaylightMode, WorldType, GameMode,
    )
    from craftground.screen_encoding_modes import ScreenEncodingMode

    try:
        cg_env = craftground.make(
            port=port,
            initial_env_config=InitialEnvironmentConfig(
                image_width=64,
                image_height=64,
                gamemode=GameMode.CREATIVE,
                world_type=WorldType.DEFAULT,  # 기본 월드 (자연 지형)
                hud_hidden=False,
                render_distance=6,
                simulation_distance=6,
                request_raycast=True,
                requires_surrounding_blocks=False,
                screen_encoding_mode=ScreenEncodingMode.RAW,
                initial_extra_commands=[
                    "gamerule doDaylightCycle false",
                    "gamerule doMobSpawning false",
                    "difficulty peaceful",
                    "gamerule fallDamage false",
                    "time set day",
                    "weather clear",
                ],
            ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
            action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        )
    except Exception:
        # WorldType.DEFAULT 가 없는 버전이면 FLAT 시도
        print("[warn] WorldType.DEFAULT 실패 → FLAT으로 대체")
        cg_env = craftground.make(
            port=port,
            initial_env_config=InitialEnvironmentConfig(
                image_width=64,
                image_height=64,
                gamemode=GameMode.CREATIVE,
                world_type=WorldType.FLAT,
                hud_hidden=False,
                render_distance=6,
                simulation_distance=6,
                request_raycast=True,
                requires_surrounding_blocks=False,
                screen_encoding_mode=ScreenEncodingMode.RAW,
                initial_extra_commands=[
                    "gamerule doDaylightCycle false",
                    "gamerule doMobSpawning false",
                    "difficulty peaceful",
                    "gamerule fallDamage false",
                    "time set day",
                    "weather clear",
                ],
            ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
            action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        )
    return FlattenEnv(cg_env)


def collect_episode(env: FlattenEnv, expert: FlattenExpert, ep: int):
    """한 에피소드 수집. (obs_list, action_list, blocks_cleared) 반환."""
    obs, _ = env.reset()
    expert.reset()

    observations, actions = [], []
    done  = False
    step  = 0
    info  = {}

    if _CV2_OK:
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    while not done and not expert.is_done():
        action = expert.get_action(env)

        observations.append(obs.copy())
        actions.append(action.copy())

        obs, _, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

        _show_frame(env, expert, ep, step, info.get("blocks_cleared", 0))

    return observations, actions, info.get("blocks_cleared", 0)


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else COLLECT_N_EPISODES
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"[collect] 12x12 지형 평탄화 데모 수집")
    print(f"          에피소드: {n_ep}  포트: {PORT}")
    print(f"          저장 경로: {DEMO_PATH}")
    print(f"          총 목표 블록 수: {TOTAL_TARGET_BLOCKS}")

    env    = create_env(PORT)
    expert = FlattenExpert()

    all_obs, all_actions = [], []
    success = 0
    total_cleared = 0

    for ep in range(n_ep):
        t0 = time.time()
        obs_list, act_list, cleared = collect_episode(env, expert, ep + 1)
        all_obs.extend(obs_list)
        all_actions.extend(act_list)
        total_cleared += cleared

        ok = cleared >= TOTAL_TARGET_BLOCKS
        success += int(ok)
        tag = "SUCCESS" if ok else f"cleared={cleared}/{TOTAL_TARGET_BLOCKS}"
        elapsed = time.time() - t0
        print(
            f"  EP {ep+1:3d}/{n_ep} | steps={len(act_list):4d} "
            f"| {tag} | {elapsed:.1f}s"
        )

    env.close()

    if not all_obs:
        print("[collect] 수집된 데이터 없음. 종료.")
        return

    np.savez_compressed(
        DEMO_PATH,
        observations = np.array(all_obs,     dtype=np.float32),  # (N, 18)
        actions      = np.array(all_actions, dtype=np.int64),    # (N, 7)
    )
    print(
        f"\n[collect] 저장 완료"
        f"  총 스텝: {len(all_actions):,}"
        f"  성공: {success}/{n_ep}"
        f"  평균 채굴: {total_cleared/n_ep:.1f}/{TOTAL_TARGET_BLOCKS}"
    )


if __name__ == "__main__":
    main()
