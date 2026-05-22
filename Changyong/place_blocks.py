"""
블록 배치 실험 스크립트
- 원하는 위치에 원하는 블록을 배치하고 확인
- 건축 태스크 실험 전 블록 배치 동작 테스트용

사용법:
    python place_blocks.py

조작:
    W/A/S/D  : 앞/좌/뒤/우 이동
    Q/E      : 위/아래 보기
    Z/C      : 좌/우 회전
    SPACE    : 점프
    F        : 블록 설치 (USE_ITEM)
    G        : 블록 파괴 (ATTACK)
    ESC/Q키  : 종료
"""
import os
import sys
import glob
import time
import signal
import subprocess
import numpy as np
import cv2

import craftground
from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op

# ─── Java 환경 설정 ────────────────────────────────────────────────
JAVA_PATH = "/opt/miniconda3/envs/exp_craftground/lib/jvm"
os.environ["JAVA_HOME"] = JAVA_PATH
os.environ["PATH"] = os.path.join(JAVA_PATH, "bin") + ":" + os.environ.get("PATH", "")

PORT = 8030

# ── 배치할 블록 구조 정의 ─────────────────────────────────────────
# (x, y, z, block_id) 형식으로 원하는 블록 위치를 지정
# x, y, z 는 절대 좌표
BLOCK_LAYOUT = [
    # 예시: 3x3 바닥 + 벽 구조
    # 바닥 (y=65)
    (0, 65, 2, "minecraft:oak_planks"),
    (1, 65, 2, "minecraft:oak_planks"),
    (-1, 65, 2, "minecraft:oak_planks"),
    (0, 65, 3, "minecraft:oak_planks"),
    (1, 65, 3, "minecraft:oak_planks"),
    (-1, 65, 3, "minecraft:oak_planks"),
    (0, 65, 4, "minecraft:oak_planks"),
    (1, 65, 4, "minecraft:oak_planks"),
    (-1, 65, 4, "minecraft:oak_planks"),
    # 벽 (y=66)
    (1, 66, 2, "minecraft:stone_bricks"),
    (-1, 66, 2, "minecraft:stone_bricks"),
    (1, 66, 4, "minecraft:stone_bricks"),
    (-1, 66, 4, "minecraft:stone_bricks"),
    # 지붕 (y=67)
    (0, 67, 2, "minecraft:glass"),
    (0, 67, 3, "minecraft:glass"),
    (0, 67, 4, "minecraft:glass"),
]


def cleanup():
    for sock in glob.glob("/tmp/minecraftrl_*.sock"):
        try:
            os.remove(sock)
        except OSError:
            pass
    try:
        result = subprocess.run(["pgrep", "-f", "minecraftrl"],
                                capture_output=True, text=True)
        for pid in result.stdout.strip().split():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
    except FileNotFoundError:
        pass
    time.sleep(2)


def build_setblock_commands(layout):
    """BLOCK_LAYOUT → setblock 명령어 리스트 변환."""
    return [f"setblock {x} {y} {z} {block}" for x, y, z, block in layout]


def build_noop():
    return no_op()


def build_action(name: str):
    act = no_op()
    if name == "FORWARD":   act[0] = 1
    elif name == "BACKWARD": act[0] = 2
    elif name == "LEFT":     act[1] = 2
    elif name == "RIGHT":    act[1] = 1
    elif name == "JUMP":     act[2] = 1
    elif name == "LOOK_UP":  act[3] = 11
    elif name == "LOOK_DOWN":act[3] = 13
    elif name == "TURN_LEFT": act[4] = 11
    elif name == "TURN_RIGHT":act[4] = 13
    elif name == "ATTACK":   act[5] = 3
    elif name == "USE_ITEM": act[5] = 1
    return act


def run():
    print("=" * 50)
    print("  블록 배치 실험 환경")
    print(f"  배치할 블록 수: {len(BLOCK_LAYOUT)}개")
    print("=" * 50)

    cleanup()

    # setblock 명령어 생성
    setblock_cmds = build_setblock_commands(BLOCK_LAYOUT)

    config = InitialEnvironmentConfig(
        image_width=512,
        image_height=512,
        seed="42",
        world_type=WorldType.DEFAULT,
        render_distance=4,
        simulation_distance=4,
        hud_hidden=False,
        initial_extra_commands=[
            "time set day",
            "gamerule doDaylightCycle false",
            "gamerule doMobSpawning false",
            "gamerule doWeatherCycle false",
            "weather clear",
            "gamerule fallDamage false",
            "kill @e[type=!player]",
            "gamerule randomTickSpeed 0",
            # 기본 아이템: 원하는 블록 + 곡괭이
            "give @p minecraft:oak_planks 64",
            "give @p minecraft:stone_bricks 64",
            "give @p minecraft:glass 64",
            "give @p minecraft:iron_pickaxe 1",
            "tp @p 0 66 0 0 0",
            # 바닥 생성
            "execute positioned 0 65 0 run fill ~-15 ~-1 ~-15 ~15 ~-1 ~15 minecraft:grass_block",
            # 블록 배치
            *setblock_cmds,
        ],
    )

    print("\n환경 시작 중...")
    env = make(
        initial_env_config=config,
        port=PORT,
        verbose=False,
        verbose_gradle=True,
        render_action=False,
    )

    obs, info = env.reset()

    # 이미지 처리
    def get_frame(obs):
        img = obs.get("pov") or obs.get("rgb")
        if img is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        img = np.array(img, dtype=np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return img

    cv2.namedWindow("Block Placement", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Block Placement", 512, 512)

    print("\n조작법:")
    print("  W/A/S/D   : 이동")
    print("  Q/E       : 위/아래 보기")
    print("  Z/C       : 좌/우 회전")
    print("  SPACE     : 점프")
    print("  F         : 블록 설치")
    print("  G         : 블록 파괴")
    print("  R         : 블록 재배치 (리셋)")
    print("  ESC       : 종료\n")

    action = build_noop()

    while True:
        obs, _, terminated, truncated, info = env.step(action)
        action = build_noop()  # 매 프레임 초기화

        frame = get_frame(obs)
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(display, "F:place  G:break  R:rebuild  ESC:quit",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Block Placement", display)

        key = cv2.waitKey(33) & 0xFF  # ~30fps

        if key == 27 or key == ord('q'):   # ESC or q
            break
        elif key == ord('w'):
            action = build_action("FORWARD")
        elif key == ord('s'):
            action = build_action("BACKWARD")
        elif key == ord('a'):
            action = build_action("LEFT")
        elif key == ord('d'):
            action = build_action("RIGHT")
        elif key == ord(' '):
            action = build_action("JUMP")
        elif key == ord('q'):
            action = build_action("LOOK_UP")
        elif key == ord('e'):
            action = build_action("LOOK_DOWN")
        elif key == ord('z'):
            action = build_action("TURN_LEFT")
        elif key == ord('c'):
            action = build_action("TURN_RIGHT")
        elif key == ord('f'):
            action = build_action("USE_ITEM")
        elif key == ord('g'):
            action = build_action("ATTACK")
        elif key == ord('r'):
            # 블록 재배치
            print("블록 재배치 중...")
            obs, _ = env.reset()

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    cv2.destroyAllWindows()
    print("종료.")


if __name__ == "__main__":
    run()
