"""
생존 강화학습 실험 (Survival RL)
- Task   : 최대한 오래 생존하며 체력/허기 유지, 적 처치
- Agent  : PPO (CnnPolicy, Stable Baselines3)
- Obs    : 84x84 RGB 이미지
- Actions: 13개 이산 액션 (이동/회전/공격/점프/달리기/아이템 사용)
- Reward : 생존 +0.01/스텝 | 체력 회복 +1.5 | 체력 감소 -0.5/포인트 | 사망 -10.0 | 적 주시 +0.05

사용법:
    학습: python survival_rl.py train [total_timesteps]
          ex) python survival_rl.py train 2000000
    평가: python survival_rl.py eval [model_path]
          ex) python survival_rl.py eval survival_ppo_model
"""
import os
import sys
import glob
import time
import signal
import subprocess
import torch
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

# ─── GPU 디바이스 설정 ─────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        print("  [device] CUDA GPU 사용")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("  [device] Apple MPS GPU 사용")
        return "mps"
    else:
        print("  [device] CPU 사용")
        return "cpu"

DEVICE = get_device()

# ─── Java 환경 설정 ────────────────────────────────────────────────
# 본인 환경에 맞게 수정하세요
JAVA_PATH = "/opt/miniconda3/envs/exp_craftground/lib/jvm"
os.environ["JAVA_HOME"] = JAVA_PATH
os.environ["PATH"] = os.path.join(JAVA_PATH, "bin") + ":" + os.environ.get("PATH", "")

# ─── 태스크 설정 ───────────────────────────────────────────────────
MAX_STEPS     = 3000    # 에피소드 최대 스텝 수
IMG_SIZE      = 84      # PPO CnnPolicy 입력 이미지 크기
PORT          = 8020    # 다른 실험과 겹치지 않도록 설정
MAX_HEALTH    = 20.0    # 마인크래프트 최대 체력


# ══════════════════════════════════════════════════════════════════
# 0. 이전 실행 잔여 프로세스/소켓 정리
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
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"  [cleanup] Java 프로세스 종료: PID {pid}")
            except (ProcessLookupError, ValueError):
                pass
    except FileNotFoundError:
        pass

    time.sleep(2)


# ══════════════════════════════════════════════════════════════════
# 1. 액션 정의 (V1_MINEDOJO 포맷)
# ══════════════════════════════════════════════════════════════════
# no_op() 반환: [0, 0, 0, 12, 12, 0, 0, 0]
#   idx 0: 전진(1)/후진(2)
#   idx 1: 우(1)/좌(2) 옆이동
#   idx 2: 점프(1)/스니크(2)/달리기(3)
#   idx 3: pitch 델타 (0=−180°, 24=+180°, 12=중립)
#   idx 4: yaw 델타   (0=−180°, 24=+180°, 12=중립)
#   idx 5: 공격(3)/사용(1)/드롭(2)

ACTIONS: List[str] = [
    "NO_OP",
    "FORWARD",
    "BACKWARD",
    "STRAFE_LEFT",
    "STRAFE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LOOK_UP",
    "LOOK_DOWN",
    "ATTACK",
    "USE_ITEM",      # 음식 먹기 / 아이템 사용
    "JUMP",
    "SPRINT_FORWARD",
]


def build_action(name: str) -> List[int]:
    """액션 이름 → CraftGround V1 액션 배열 변환."""
    act = no_op()
    if name == "FORWARD":
        act[0] = 1
    elif name == "BACKWARD":
        act[0] = 2
    elif name == "STRAFE_LEFT":
        act[1] = 2
    elif name == "STRAFE_RIGHT":
        act[1] = 1
    elif name == "TURN_LEFT":
        act[4] = 11
    elif name == "TURN_RIGHT":
        act[4] = 13
    elif name == "LOOK_UP":
        act[3] = 11
    elif name == "LOOK_DOWN":
        act[3] = 13
    elif name == "ATTACK":
        act[5] = 3
    elif name == "USE_ITEM":
        act[5] = 1
    elif name == "JUMP":
        act[2] = 1
    elif name == "SPRINT_FORWARD":
        act[0] = 1
        act[2] = 3
    return act


# ══════════════════════════════════════════════════════════════════
# 2. 생존 태스크 Wrapper
# ══════════════════════════════════════════════════════════════════
class SurvivalTaskWrapper(gym.Wrapper):
    """
    CraftGround 환경을 생존 태스크로 감싸는 Wrapper.

    관측: (IMG_SIZE, IMG_SIZE, 3) uint8 RGB 이미지
    액션: Discrete(len(ACTIONS))
    보상:
      - 생존 보상     : +0.01/스텝
      - 체력 회복     : +1.5 (이전 체력 대비 증가)
      - 체력 감소     : -0.5 × (감소량) (피해 페널티)
      - 적 주시       : +0.05 (전투 유도)
      - 사망 페널티   : -10.0
    종료:
      - 플레이어 사망
      - MAX_STEPS 초과 (truncated)
    """

    def __init__(self, env: gym.Env, image_size: int = IMG_SIZE):
        super().__init__(env)
        self.image_size   = image_size
        self.prev_health  = MAX_HEALTH
        self.current_step = 0
        self.last_obs     = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        self.last_health  = MAX_HEALTH

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8,
        )

    # ── 내부 헬퍼 ──────────────────────────────────────────────
    def _process_image(self, obs: dict) -> np.ndarray:
        img = obs.get("pov")
        if img is None:
            img = obs.get("rgb")
        if img is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = np.array(img, dtype=np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:    # (C,H,W) → (H,W,C)
            img = img.transpose(1, 2, 0)
        return cv2.resize(img, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_AREA)

    def _get_health(self, obs_dict: dict) -> float:
        """현재 체력(0~20) 반환."""
        try:
            full = obs_dict.get("full")
            if full is None:
                return self.prev_health
            return float(full.health)
        except Exception:
            return self.prev_health

    def _is_dead(self, obs_dict: dict) -> bool:
        try:
            full = obs_dict.get("full")
            return full is not None and full.is_dead
        except Exception:
            return False

    def _is_looking_at_entity(self, obs_dict: dict) -> bool:
        """
        raycast_result로 에이전트가 엔티티(몹)를 바라보고 있는지 확인.
        HitResult.type: 0=MISS, 1=BLOCK, 2=ENTITY
        """
        try:
            full = obs_dict.get("full")
            if full is None:
                return False
            hit = full.raycast_result
            return hit.type == 2
        except Exception:
            return False

    # ── gymnasium API ──────────────────────────────────────────
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        self.prev_health  = MAX_HEALTH
        self.current_step = 0
        self.last_obs     = self._process_image(obs)
        self.last_health  = MAX_HEALTH
        return self.last_obs, {"health": MAX_HEALTH, "step": 0}

    def step(self, action: int):
        action_arr = build_action(ACTIONS[int(action)])
        obs, _, terminated, truncated, info = self.env.step(action_arr)

        current_health = self._get_health(info)
        health_delta   = current_health - self.prev_health
        self.prev_health   = current_health
        self.current_step += 1

        # 보상 계산
        reward = 0.01   # 살아있는 매 스텝 보상

        if health_delta > 0:
            reward += 1.5 * health_delta   # 체력 회복 (음식 섭취, 자연 회복)
        elif health_delta < 0:
            reward += 0.5 * health_delta   # 체력 감소 페널티 (health_delta < 0이므로 자동으로 마이너스)

        # 적 주시 보상 (전투 유도)
        if self._is_looking_at_entity(info):
            reward += 0.05

        # 사망 처리
        if self._is_dead(info):
            terminated = True
            reward -= 10.0
            print(f"  [DEAD] 사망. 생존 스텝: {self.current_step}")

        if self.current_step >= MAX_STEPS:
            truncated = True
            print(f"  [TIMEOUT] {MAX_STEPS}스텝 생존 성공!")

        self.last_obs    = self._process_image(obs)
        self.last_health = current_health
        return (
            self.last_obs,
            reward,
            terminated,
            truncated,
            {"health": current_health, "step": self.current_step},
        )


# ══════════════════════════════════════════════════════════════════
# 3. 실시간 렌더 콜백
# ══════════════════════════════════════════════════════════════════
class RenderCallback(BaseCallback):
    """학습 중 매 스텝 cv2 창으로 에이전트 시점을 실시간 표시."""

    WINDOW = "Survival Training"

    def __init__(self, vec_env, render_every: int = 4, verbose=0):
        super().__init__(verbose)
        self.vec_env      = vec_env
        self.render_every = render_every  # 매 N 스텝마다 렌더 (부하 감소)
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every != 0:
            return True
        try:
            wrapper = self.vec_env.envs[0].env  # Monitor → SurvivalTaskWrapper
            obs     = wrapper.last_obs           # (84, 84, 3)
            health  = wrapper.last_health

            display = cv2.resize(obs, (320, 320), interpolation=cv2.INTER_NEAREST)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.putText(display,
                        f"HP: {health:.1f}/{MAX_HEALTH}  Step: {wrapper.current_step}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

            # 창이 닫혔으면 재생성 (macOS 이벤트 루프 문제 대응)
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
# 4. 환경 팩토리 (기존 3번)
# ══════════════════════════════════════════════════════════════════
def make_env(port: int = PORT, seed: str = "42", hud_hidden: bool = True):
    """SB3 DummyVecEnv 용 환경 생성 팩토리. 연결 실패 시 최대 3회 재시도."""
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
                    request_raycast=True,   # 적 주시 감지용
                    initial_extra_commands=[
                        # ── 누적 엔티티 제거 (리셋마다 실행) ──────────
                        "kill @e[type=!player]",        # 이전 에피소드 몹/아이템 전부 제거
                        # ── 시간/날씨 ──────────────────────────────
                        "time set 13000",               # 황혼 (몹 스폰 직전)
                        "gamerule doDaylightCycle true", # 낮밤 사이클 ON (생존 압박)
                        "gamerule doWeatherCycle false",
                        "weather clear",
                        # ── 게임 규칙 ──────────────────────────────
                        "gamerule doMobSpawning false", # 자연 스폰 OFF → 엔티티 누적 방지
                        "gamerule doImmediateRespawn true",
                        "gamerule randomTickSpeed 0",
                        # ── 플레이어 초기 설정 ─────────────────────
                        "tp @p 0 65 0 0 0",
                        "give @p minecraft:diamond_sword{Enchantments:[{id:sharpness,lvl:5}]} 1",
                        "give @p minecraft:bread 16",
                        "give @p minecraft:diamond_helmet 1",
                        "give @p minecraft:diamond_chestplate 1",
                        "give @p minecraft:diamond_leggings 1",
                        "give @p minecraft:diamond_boots 1",
                        # ── 지형 설정 ──────────────────────────────
                        "execute positioned 0 65 0 run fill ~-10 ~-1 ~-10 ~10 ~-1 ~10 minecraft:grass_block",
                        # ── 좀비 직접 소환 (자연 스폰 대신 제어된 수 유지) ──
                        "summon minecraft:zombie 4 65 4",
                        "summon minecraft:zombie -4 65 4",
                        "summon minecraft:zombie 4 65 -4",
                        "summon minecraft:zombie -4 65 -4",
                        "summon minecraft:zombie 0 65 6",
                    ],
                )
                env = make(
                    initial_env_config=config,
                    port=port,
                    verbose=False,
                    verbose_gradle=True,
                    render_action=False,
                )
                env = SurvivalTaskWrapper(env, image_size=IMG_SIZE)
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
# 4. 학습
# ══════════════════════════════════════════════════════════════════
def train(total_timesteps: int = 100_000):
    print("=" * 55)
    print("   생존 강화학습 실험 (Survival RL)")
    print(f"   알고리즘 : PPO (CnnPolicy, Stable Baselines3)")
    print(f"   목표     : {MAX_STEPS}스텝 동안 생존")
    print(f"   학습 스텝: {total_timesteps:,}")
    print("=" * 55)

    run = wandb.init(
        project="craftground-survival_2",
        name="ppo-survival",
        config={
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "task": "survival",
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
            "reward_survival_per_step": 0.01,
            "reward_health_recovery": 1.5,
            "reward_health_damage": -0.5,
            "reward_looking_at_entity": 0.05,
            "penalty_death": -10.0,
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
        tensorboard_log="./logs/survival_ppo/",
        device=DEVICE,
    )

    print("\n학습을 시작합니다. TensorBoard: tensorboard --logdir ./logs/survival_ppo\n")

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
        save_path = "survival_ppo_model"
        model.save(save_path)
        print(f"\n모델 저장 완료: {save_path}.zip")
    except KeyboardInterrupt:
        print("\n학습 중단. 현재까지 학습된 모델 저장 중...")
        model.save("survival_ppo_interrupted")
        print("모델 저장 완료: survival_ppo_interrupted.zip")
    except Exception as e:
        print(f"\n에러 발생: {e}")
        traceback.print_exc()
    finally:
        vec_env.close()
        run.finish()


# ══════════════════════════════════════════════════════════════════
# 5. 평가
# ══════════════════════════════════════════════════════════════════
def evaluate(model_path: str = "survival_ppo_model", n_episodes: int = 5):
    print(f"\n{'='*55}")
    print(f"   모델 평가: {model_path}")
    print(f"   에피소드: {n_episodes}회")
    print(f"{'='*55}\n")

    cleanup_old_instances()

    config = InitialEnvironmentConfig(
        image_width=256,
        image_height=256,
        seed="9999",
        world_type=WorldType.DEFAULT,
        render_distance=2,
        simulation_distance=2,
        hud_hidden=False,
        request_raycast=True,
        initial_extra_commands=[
            "time set 13000",
            "gamerule doDaylightCycle true",
            "gamerule doWeatherCycle false",
            "weather clear",
            "gamerule doMobSpawning true",
            "gamerule doImmediateRespawn true",
            "tp @p 0 65 0 0 0",
            "give @p minecraft:iron_sword 1",
            "give @p minecraft:bread 16",
            "give @p minecraft:leather_helmet 1",
            "give @p minecraft:leather_chestplate 1",
            "execute positioned 0 65 0 run fill ~-10 ~-1 ~-10 ~10 ~-1 ~10 minecraft:grass_block",
            "summon minecraft:zombie 5 65 5",
            "summon minecraft:zombie -5 65 5",
            "summon minecraft:zombie 5 65 -5",
        ],
    )
    base_env = make(
        initial_env_config=config,
        port=PORT + 1,
        verbose=False,
        verbose_gradle=True,
        render_action=True,
    )
    env = SurvivalTaskWrapper(base_env, image_size=IMG_SIZE)

    model = PPO.load(model_path)
    results = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                display = cv2.resize(obs, (256, 256), interpolation=cv2.INTER_NEAREST)
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                health = info.get("health", 0)
                cv2.putText(display,
                            f"HP: {health:.1f}/{MAX_HEALTH}  Step: {steps}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
                cv2.imshow("Survival Agent", display)
                key = cv2.waitKey(33) & 0xFF
                if key == ord("q"):
                    break

                if terminated or truncated:
                    break

            survived = not terminated   # truncated = 시간 초과 생존
            results.append({
                "ep": ep + 1,
                "steps": steps,
                "reward": total_reward,
                "health": info.get("health", 0),
                "survived": survived,
            })
            print(f"  Episode {ep+1:2d}: Steps={steps:4d}, "
                  f"Reward={total_reward:7.2f}, "
                  f"HP={info.get('health', 0):.1f}, "
                  f"{'✅ SURVIVED' if survived else '❌ DEAD'}")

    except Exception as e:
        print(f"에러 발생: {e}")
        traceback.print_exc()
    finally:
        env.close()
        cv2.destroyAllWindows()

    survival_rate = sum(r["survived"] for r in results) / len(results) * 100 if results else 0
    avg_reward    = np.mean([r["reward"] for r in results]) if results else 0
    avg_steps     = np.mean([r["steps"]  for r in results]) if results else 0
    print(f"\n  평균 보상    : {avg_reward:.2f}")
    print(f"  평균 생존 스텝: {avg_steps:.0f}")
    print(f"  생존율       : {survival_rate:.0f}%")


# ══════════════════════════════════════════════════════════════════
# 6. 진입점
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
        train(total_timesteps=steps)
    elif mode == "eval":
        path = sys.argv[2] if len(sys.argv) > 2 else "survival_ppo_model"
        eps  = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        evaluate(model_path=path, n_episodes=eps)
    else:
        print(__doc__)
