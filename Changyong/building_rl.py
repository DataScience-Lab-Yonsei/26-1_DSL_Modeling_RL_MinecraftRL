"""
Oak Planks 건축 강화학습
- Task   : 지정된 3x3 위치에 oak_planks 9개를 배치
- Agent  : PPO (CnnPolicy, Stable Baselines3)
- Obs    : 84x84 RGB 이미지
- Actions: 11개 이산 액션 (이동/회전/점프/블록설치)
- Reward : 블록 배치 +1.0 | 완료 보너스 +10.0 | 스텝 패널티 -0.01 | raycast shaping +0.02

목표 구조:
    x: -1, 0, 1
    y: 65 (stone 바닥 위)
    z: 2, 3, 4
    → 3x3 oak_planks 바닥

사용법:
    학습: python building_rl.py train [total_timesteps]
          ex) python building_rl.py train 1000000
    평가: python building_rl.py eval [model_path]
          ex) python building_rl.py eval building_rl_model
"""
import os
import sys
import glob
import time
import signal
import subprocess
import traceback
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from typing import List

import craftground
from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

# ─── Java 환경 설정 ────────────────────────────────────────────────
JAVA_PATH = "/opt/miniconda3/envs/exp_craftground/lib/jvm"
os.environ["JAVA_HOME"] = JAVA_PATH
os.environ["PATH"] = os.path.join(JAVA_PATH, "bin") + ":" + os.environ.get("PATH", "")

# ─── 태스크 설정 ───────────────────────────────────────────────────
TARGET_COUNT = 9       # 3x3 = 9개 배치하면 성공
MAX_STEPS    = 1000
IMG_SIZE     = 84
PORT         = 8012


# ══════════════════════════════════════════════════════════════════
# 0. 정리
# ══════════════════════════════════════════════════════════════════
def cleanup_old_instances():
    for sock in glob.glob("/tmp/minecraftrl_*.sock"):
        try:
            os.remove(sock)
            print(f"  [cleanup] 소켓 파일 제거: {sock}")
        except OSError:
            pass
    try:
        result = subprocess.run(
            ["pgrep", "-f", "minecraftrl"],
            capture_output=True, text=True
        )
        for pid in result.stdout.strip().split():
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"  [cleanup] Java 프로세스 종료: PID {pid}")
            except (ProcessLookupError, ValueError):
                pass
    except FileNotFoundError:
        pass
    time.sleep(2)


# ══════════════════════════════════════════════════════════════════
# 1. 액션 정의
# ══════════════════════════════════════════════════════════════════
ACTIONS: List[str] = [
    "NO_OP",
    "FORWARD",
    "BACKWARD",
    "LEFT",
    "RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LOOK_UP",
    "LOOK_DOWN",
    "JUMP",
    "USE_ITEM",   # 블록 설치
]


def build_action(name: str) -> List[int]:
    act = no_op()
    if name == "FORWARD":     act[0] = 1
    elif name == "BACKWARD":  act[0] = 2
    elif name == "LEFT":      act[1] = 2
    elif name == "RIGHT":     act[1] = 1
    elif name == "JUMP":      act[2] = 1
    elif name == "LOOK_UP":   act[3] = 11
    elif name == "LOOK_DOWN": act[3] = 13
    elif name == "TURN_LEFT":  act[4] = 11
    elif name == "TURN_RIGHT": act[4] = 13
    elif name == "USE_ITEM":   act[5] = 1
    return act


# ══════════════════════════════════════════════════════════════════
# 2. 건축 태스크 Wrapper
# ══════════════════════════════════════════════════════════════════
class BuildingTaskWrapper(gym.Wrapper):
    """
    건축 태스크 Wrapper
    - 에이전트가 oak_planks 9개를 목표 위치(3x3 바닥)에 배치
    - 인벤토리 감소 감지로 블록 배치 보상
    - raycast로 목표 영역(stone 바닥)을 바라볼 때 reward shaping
    보상:
      - 블록 배치 시 (인벤토리 감소): +1.0
      - raycast가 stone/oak_planks 바닥에 닿을 때: +0.02
      - 목표 달성 (9개 배치): +10.0
      - 매 스텝 패널티: -0.01
      - 사망: -2.0
    """

    def __init__(self, env: gym.Env, image_size: int = IMG_SIZE):
        super().__init__(env)
        self.image_size   = image_size
        self.prev_count   = TARGET_COUNT  # 남은 oak_planks 개수
        self.placed_count = 0
        self.current_step = 0
        self.last_obs     = None

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8,
        )

    def _process_image(self, obs: dict) -> np.ndarray:
        img = obs.get("pov") or obs.get("rgb")
        if img is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = np.array(img, dtype=np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return cv2.resize(img, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_AREA)

    def _get_plank_count(self, obs_dict: dict) -> int:
        """인벤토리의 oak_planks 개수 반환."""
        try:
            full = obs_dict.get("full")
            if full is None:
                return self.prev_count
            count = 0
            for item in full.inventory:
                if "oak_planks" in item.translation_key.lower():
                    count += item.count
            return count
        except Exception:
            return self.prev_count

    def _is_looking_at_floor(self, obs_dict: dict) -> bool:
        """
        raycast가 stone 또는 oak_planks에 닿으면 True.
        (목표 영역 바닥을 바라보고 있을 때 reward shaping)
        """
        try:
            full = obs_dict.get("full")
            if full is None:
                return False
            hit = full.raycast_result
            if hit.type != 1:
                return False
            key = hit.target_block.translation_key.lower()
            return "stone" in key or "oak_planks" in key
        except Exception:
            return False

    def _is_dead(self, obs_dict: dict) -> bool:
        try:
            full = obs_dict.get("full")
            return full is not None and full.is_dead
        except Exception:
            return False

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        self.prev_count   = TARGET_COUNT
        self.placed_count = 0
        self.current_step = 0
        img = self._process_image(obs)
        self.last_obs = img
        return img, {"placed": 0}

    def step(self, action: int):
        action_arr = build_action(ACTIONS[int(action)])
        obs, _, terminated, truncated, info = self.env.step(action_arr)

        current_count = self._get_plank_count(info)
        newly_placed  = max(0, self.prev_count - current_count)
        self.prev_count    = current_count
        self.placed_count += newly_placed
        self.current_step += 1

        looking_at_floor = self._is_looking_at_floor(info)

        reward = newly_placed * 1.0 - 0.01
        if looking_at_floor:
            reward += 0.02

        if self.placed_count >= TARGET_COUNT:
            terminated = True
            reward += 10.0
            print(f"  [SUCCESS] {TARGET_COUNT}개 배치 완료! (step {self.current_step})")

        if self._is_dead(info):
            terminated = True
            reward -= 2.0

        if self.current_step >= MAX_STEPS:
            truncated = True

        img = self._process_image(obs)
        self.last_obs = img

        return (
            img,
            reward,
            terminated,
            truncated,
            {"placed": self.placed_count, "step": self.current_step},
        )


# ══════════════════════════════════════════════════════════════════
# 3. 실시간 렌더 콜백
# ══════════════════════════════════════════════════════════════════
class RenderCallback(BaseCallback):
    WINDOW = "Building Training"

    def __init__(self, vec_env, render_every: int = 4, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env      = vec_env
        self.render_every = render_every
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every != 0:
            return True
        try:
            wrapper = self.vec_env.envs[0].env
            obs = wrapper.last_obs
            if obs is None:
                return True
            display = cv2.resize(obs, (320, 320), interpolation=cv2.INTER_NEAREST)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.putText(
                display,
                f"Placed: {wrapper.placed_count}/{TARGET_COUNT}  Step: {wrapper.current_step}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2,
            )
            if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            cv2.imshow(self.WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
        except Exception:
            pass
        return True

    def _on_training_end(self):
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════
# 4. 환경 팩토리
# ══════════════════════════════════════════════════════════════════
def make_env(port: int = PORT, seed: str = "42", hud_hidden: bool = True):
    def _init() -> gym.Env:
        last_exc = None
        for attempt in range(3):
            try:
                config = InitialEnvironmentConfig(
                    image_width=256,
                    image_height=256,
                    seed=seed,
                    world_type=WorldType.DEFAULT,
                    render_distance=2,
                    simulation_distance=2,
                    hud_hidden=hud_hidden,
                    request_raycast=True,
                    initial_extra_commands=[
                        "kill @e[type=!player]",
                        "time set day",
                        "gamerule doDaylightCycle false",
                        "gamerule doMobSpawning false",
                        "gamerule doWeatherCycle false",
                        "weather clear",
                        "gamerule fallDamage false",
                        "gamerule doImmediateRespawn true",
                        "gamerule randomTickSpeed 0",
                        # stone 바닥 생성 (에이전트가 그 위에 블록을 놓음)
                        "execute positioned 0 65 0 run fill ~-7 ~-1 ~-7 ~7 ~-1 ~7 minecraft:stone",
                        # 목표 영역(y=65) 비우기 → 에이전트가 채워야 할 빈 공간
                        "fill -1 65 2 1 65 4 minecraft:air",
                        # 에이전트 스폰: 목표 영역 앞, 약간 아래를 바라보게
                        "tp @p 0 66 0 0 30",
                        # 인벤토리: oak_planks 9개 (목표 개수와 동일)
                        "give @p minecraft:oak_planks 9",
                    ],
                )
                env = make(
                    initial_env_config=config,
                    port=port,
                    verbose=False,
                    verbose_gradle=True,
                    render_action=False,
                )
                env = BuildingTaskWrapper(env, image_size=IMG_SIZE)
                env = Monitor(env)
                return env
            except Exception as e:
                last_exc = e
                print(f"  [make_env] 환경 생성 실패 (시도 {attempt+1}/3): {e}")
                for sock in glob.glob(f"/tmp/minecraftrl_{port}*.sock"):
                    try:
                        os.remove(sock)
                    except OSError:
                        pass
                time.sleep(5)
        raise RuntimeError("환경 생성 3회 실패") from last_exc
    return _init


# ══════════════════════════════════════════════════════════════════
# 5. 학습
# ══════════════════════════════════════════════════════════════════
def train(total_timesteps: int = 1_000_000):
    print("=" * 55)
    print("   Oak Planks 건축 강화학습")
    print(f"   알고리즘 : PPO (CnnPolicy, Stable Baselines3)")
    print(f"   목표     : oak_planks {TARGET_COUNT}개 배치 (3×3 바닥)")
    print(f"   학습 스텝: {total_timesteps:,}")
    print("=" * 55)

    run = wandb.init(
        project="craftground-building",
        name="ppo-building-3x3",
        config={
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "task": "building_3x3_floor",
            "target_count": TARGET_COUNT,
            "max_steps": MAX_STEPS,
            "total_timesteps": total_timesteps,
            "learning_rate": 2.5e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.01,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    print("\n[1/3] 이전 인스턴스 정리 중...")
    cleanup_old_instances()
    print("[2/3] 환경 생성 중...")
    vec_env = DummyVecEnv([make_env(port=PORT, seed="42")])
    print("[3/3] 모델 초기화 중...")

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/building_rl/",
        device="auto",
    )

    print("\n학습을 시작합니다.\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=[
                WandbCallback(
                    gradient_save_freq=1000,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                RenderCallback(vec_env),
            ],
        )
        save_path = "building_rl_model"
        model.save(save_path)
        print(f"\n모델 저장 완료: {save_path}.zip")
    except KeyboardInterrupt:
        print("\n학습 중단. 모델 저장 중...")
        model.save("building_rl_interrupted")
        print("모델 저장 완료: building_rl_interrupted.zip")
    except Exception as e:
        print(f"\n에러 발생: {e}")
        traceback.print_exc()
    finally:
        vec_env.close()
        run.finish()


# ══════════════════════════════════════════════════════════════════
# 6. 평가
# ══════════════════════════════════════════════════════════════════
def evaluate(model_path: str, episodes: int = 5):
    print(f"\n모델 로드: {model_path}")
    cleanup_old_instances()
    env = make_env(port=PORT, seed="99", hud_hidden=False)()

    model = PPO.load(model_path, env=env)

    cv2.namedWindow("Building Eval", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Building Eval", 512, 512)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            display = cv2.resize(obs, (512, 512), interpolation=cv2.INTER_NEAREST)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.putText(
                display,
                f"EP {ep+1}  Placed: {info['placed']}/{TARGET_COUNT}  R: {total_reward:.1f}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2,
            )
            cv2.imshow("Building Eval", display)
            if cv2.waitKey(33) & 0xFF == ord("q"):
                done = True

        print(f"  EP {ep+1}: placed={info['placed']}/{TARGET_COUNT}, reward={total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════
# 7. 진입점
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
        train(total_timesteps=steps)
    elif mode == "eval":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "building_rl_model"
        evaluate(model_path)
    else:
        print(__doc__)
