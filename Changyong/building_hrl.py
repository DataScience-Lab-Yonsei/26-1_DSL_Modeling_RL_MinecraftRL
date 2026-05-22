"""
Oak Planks 건축 강화학습 - Hierarchical RL (수동 2-level)

구조:
  High-level (Manager) : Rule-based
    - 9개 목표 위치 중 에이전트에서 가장 가까운 빈 위치를 sub-goal로 선택
    - 블록 배치 성공 or timeout(200 step)마다 다음 sub-goal로 전환

  Low-level (Worker) : PPO (MultiInputPolicy)
    - Obs: 84x84 RGB 이미지 + subgoal (rel_x, rel_z, yaw_sin, yaw_cos)
    - 액션: 이동/회전/점프/블록설치
    - 보상: sub-goal 근접 shaping + 블록 배치 +5.0 + 전체 완료 +10.0 + 스텝 -0.01

목표 구조 (3x3 바닥):
    x: -1, 0, 1  /  y: 65  /  z: 2, 3, 4

사용법:
    학습: python building_hrl.py train [total_timesteps]
          ex) python building_hrl.py train 1_000_000
    평가: python building_hrl.py eval [model_path]
"""
import os
import sys
import glob
import math
import time
import signal
import subprocess
import traceback
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Tuple

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
TOTAL_BLOCKS     = 9       # 3x3 목표 블록 수
MAX_STEPS        = 2000    # 에피소드 최대 스텝
MAX_SUBGOAL_STEP = 200     # sub-goal 당 최대 스텝 (timeout)
IMG_SIZE         = 84
PORT             = 8013

# 목표 배치 위치 (x, y, z)
TARGET_POSITIONS: List[Tuple[int, int, int]] = [
    (x, 65, z) for x in range(-1, 2) for z in range(2, 5)
]
# [(-1,65,2),(0,65,2),(1,65,2), (-1,65,3),(0,65,3),(1,65,3), (-1,65,4),(0,65,4),(1,65,4)]


# ══════════════════════════════════════════════════════════════════
# 0. 정리
# ══════════════════════════════════════════════════════════════════
def cleanup_old_instances():
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
    "USE_ITEM",
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
# 2. Hierarchical Building Env
# ══════════════════════════════════════════════════════════════════
class HierarchicalBuildingEnv(gym.Wrapper):
    """
    2-level Hierarchical Building 환경.

    Manager (rule-based):
      - 남은 목표 위치 중 에이전트와 가장 가까운 위치를 sub-goal로 선택
      - 블록 배치 성공 또는 MAX_SUBGOAL_STEP 초과 시 다음 sub-goal로 전환

    Worker (PPO):
      - Observation:
          "image"  : (84,84,3) RGB
          "subgoal": (4,) = [rel_x/SCALE, rel_z/SCALE, sin(yaw), cos(yaw)]
              → 에이전트가 어느 방향을 바라보는지 + 목표까지의 상대 위치 동시 제공
      - 보상:
          - 목표에 가까워질수록: +0.5 * dist_delta (dense shaping)
          - 바닥을 바라볼 때: +0.02 per step
          - 블록 배치 성공: +5.0
          - 전체 완료: +10.0
          - 매 스텝: -0.01
          - sub-goal timeout: -0.5
          - 사망: -2.0
    """

    SUBGOAL_SCALE = 5.0  # 상대 좌표 정규화 스케일

    def __init__(self, env: gym.Env, image_size: int = IMG_SIZE):
        super().__init__(env)
        self.image_size = image_size

        # 상태 변수
        self.remaining_targets: List[Tuple] = []
        self.current_subgoal: Optional[Tuple] = None
        self.subgoal_step  = 0
        self.placed_count  = 0
        self.prev_inv      = 64
        self.total_step    = 0
        self.player_pos    = (0.0, 66.0, 0.0)
        self.player_yaw    = 0.0
        self.prev_dist     = 0.0
        self.last_obs      = None

        # Gym spaces
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({
            "image":   spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            # [rel_x/SCALE, rel_z/SCALE, sin(yaw_rad), cos(yaw_rad)]
            "subgoal": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        })

    # ── 이미지 처리 ──────────────────────────────────────────────
    def _process_image(self, obs: dict) -> np.ndarray:
        img = obs.get("pov")
        if img is None:
            img = obs.get("rgb")
        if img is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = np.array(img, dtype=np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return cv2.resize(img, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_AREA)

    # ── 플레이어 위치 & yaw ───────────────────────────────────────
    def _update_player_state(self, obs_dict: dict):
        """obs (craftground 원시 관측)에서 위치와 yaw를 갱신."""
        try:
            full = obs_dict.get("full")
            if full is not None:
                self.player_pos = (float(full.x), float(full.y), float(full.z))
                self.player_yaw = float(full.yaw)
        except Exception:
            pass

    # ── 인벤토리 ─────────────────────────────────────────────────
    def _get_plank_count(self, obs_dict: dict) -> int:
        try:
            full = obs_dict.get("full")
            if full is None:
                return self.prev_inv
            count = 0
            for item in full.inventory:
                if "oak_planks" in item.translation_key.lower():
                    count += item.count
            return count
        except Exception:
            return self.prev_inv

    # ── raycast ──────────────────────────────────────────────────
    def _is_looking_at_floor(self, obs_dict: dict) -> bool:
        try:
            full = obs_dict.get("full")
            if full is None:
                return False
            hit = full.raycast_result
            if hit.type != 1:
                return False
            key = hit.target_block.translation_key.lower()
            return "stone" in key or "oak_planks" in key or "grass" in key or "dirt" in key
        except Exception:
            return False

    def _is_dead(self, obs_dict: dict) -> bool:
        try:
            full = obs_dict.get("full")
            return full is not None and full.is_dead
        except Exception:
            return False

    # ── Manager: 가장 가까운 빈 sub-goal 선택 ────────────────────
    def _select_nearest_subgoal(self) -> Optional[Tuple]:
        if not self.remaining_targets:
            return None
        px, _, pz = self.player_pos
        return min(
            self.remaining_targets,
            key=lambda t: (t[0] - px) ** 2 + (t[2] - pz) ** 2,
        )

    # ── 현재 sub-goal까지의 XZ 거리 ──────────────────────────────
    def _dist_to_subgoal(self) -> float:
        if self.current_subgoal is None:
            return 0.0
        px, _, pz = self.player_pos
        gx, _, gz = self.current_subgoal
        return math.sqrt((gx - px) ** 2 + (gz - pz) ** 2)

    # ── Worker 관측: sub-goal 상대 방향 + yaw ────────────────────
    def _get_subgoal_vec(self) -> np.ndarray:
        """
        [rel_x/SCALE, rel_z/SCALE, sin(yaw_rad), cos(yaw_rad)]
        에이전트가 어느 방향을 바라보는지 알아야 TURN을 얼마나 할지 판단 가능.
        """
        rel_x, rel_z = 0.0, 0.0
        if self.current_subgoal is not None:
            px, _, pz = self.player_pos
            gx, _, gz = self.current_subgoal
            rel_x = (gx - px) / self.SUBGOAL_SCALE
            rel_z = (gz - pz) / self.SUBGOAL_SCALE
        yaw_rad = math.radians(self.player_yaw)
        return np.array(
            [np.clip(rel_x, -1.0, 1.0),
             np.clip(rel_z, -1.0, 1.0),
             math.sin(yaw_rad),
             math.cos(yaw_rad)],
            dtype=np.float32,
        )

    # 에피소드마다 실행할 초기화 명령
    _RESET_CMDS = [
        "fill -10 65 -10 10 65 15 minecraft:air",  # 주변 전체 블록 제거
        "tp @p 0 66 0 0 50",                        # 위치/각도 초기화
        "clear @p",                                 # 인벤토리 전체 비우기
        "give @p minecraft:oak_planks 64",          # 정확히 64개만 지급
    ]

    # ── reset ─────────────────────────────────────────────────────
    def reset(self, **kwargs):
        options = kwargs.pop("options", {})
        options["extra_commands"] = self._RESET_CMDS
        obs, _ = self.env.reset(options=options, **kwargs)
        self.remaining_targets = list(TARGET_POSITIONS)
        self.player_pos        = (0.0, 66.0, 0.0)
        self.player_yaw        = 0.0
        self.current_subgoal   = self._select_nearest_subgoal()
        self.subgoal_step      = 0
        self.placed_count      = 0
        self.total_step        = 0

        # 위치 갱신 후 실제 인벤토리 값으로 prev_inv 초기화
        self._update_player_state(obs)
        self.prev_inv  = self._get_plank_count(obs)  # 실제 값으로 맞춤
        self.prev_dist = self._dist_to_subgoal()

        img = self._process_image(obs)
        self.last_obs = img
        return {"image": img, "subgoal": self._get_subgoal_vec()}, {}

    # ── step ──────────────────────────────────────────────────────
    def step(self, action: int):
        action_arr = build_action(ACTIONS[int(action)])
        # obs: craftground 원시 관측 dict ("pov"/"rgb", "full" 포함)
        obs, _, terminated, truncated, _ = self.env.step(action_arr)

        # ── 상태 갱신 (obs에서 읽어야 함!) ──
        self._update_player_state(obs)          # 위치 & yaw 갱신
        current_inv  = self._get_plank_count(obs)
        newly_placed = max(0, self.prev_inv - current_inv)
        self.prev_inv    = current_inv
        self.placed_count += newly_placed
        self.total_step   += 1
        self.subgoal_step += 1

        reward = -0.01  # 스텝 패널티

        # ── Dense shaping: 목표에 가까워질수록 보상 ──
        curr_dist = self._dist_to_subgoal()
        dist_delta = self.prev_dist - curr_dist   # 양수 = 가까워짐
        if dist_delta > 0:
            reward += dist_delta * 0.5
        self.prev_dist = curr_dist

        # ── raycast shaping: 바닥을 바라볼 때 ──
        if self._is_looking_at_floor(obs):
            reward += 0.02

        # ── 블록 배치 성공 → sub-goal 완료 ──
        if newly_placed > 0:
            reward += 5.0
            if self.current_subgoal in self.remaining_targets:
                self.remaining_targets.remove(self.current_subgoal)
            self.current_subgoal = self._select_nearest_subgoal()
            self.subgoal_step    = 0
            self.prev_dist       = self._dist_to_subgoal()
            print(f"  [PLACED] {self.placed_count}/{TOTAL_BLOCKS}  "
                  f"next subgoal: {self.current_subgoal}")

        # ── sub-goal timeout → 다음 sub-goal로 ──
        elif self.subgoal_step >= MAX_SUBGOAL_STEP:
            reward -= 0.5
            self.current_subgoal = self._select_nearest_subgoal()
            self.subgoal_step    = 0
            self.prev_dist       = self._dist_to_subgoal()

        # ── 전체 완료 ──
        if self.placed_count >= TOTAL_BLOCKS:
            terminated = True
            reward += 10.0
            print(f"  [SUCCESS] 전체 완료! (step {self.total_step})")

        if self._is_dead(obs):
            terminated = True
            reward -= 2.0

        if self.total_step >= MAX_STEPS:
            truncated = True

        img = self._process_image(obs)
        self.last_obs = img

        return (
            {"image": img, "subgoal": self._get_subgoal_vec()},
            reward,
            terminated,
            truncated,
            {"placed": self.placed_count,
             "remaining": len(self.remaining_targets),
             "subgoal": self.current_subgoal,
             "step": self.total_step},
        )


# ══════════════════════════════════════════════════════════════════
# 3. 실시간 렌더 콜백
# ══════════════════════════════════════════════════════════════════
class RenderCallback(BaseCallback):
    WINDOW = "Building HRL Training"

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

            sg = wrapper.current_subgoal
            sg_text = f"SG:{sg}" if sg else "SG:done"
            cv2.putText(display,
                        f"Placed:{wrapper.placed_count}/{TOTAL_BLOCKS}  {sg_text}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)

            # sub-goal 방향 화살표 (rel_x, rel_z 사용)
            vec = wrapper._get_subgoal_vec()
            cx, cy = 160, 160
            ax = int(cx + vec[0] * 60)
            az = int(cy + vec[1] * 60)
            cv2.arrowedLine(display, (cx, cy), (ax, az), (0, 200, 255), 2, tipLength=0.3)

            if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            cv2.imshow(self.WINDOW, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
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
                    world_type=WorldType.SUPERFLAT,
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
                        # 주변 전체 y=65 레이어 비우기
                        "fill -10 65 -10 10 65 15 minecraft:air",
                        # 에이전트: pitch=50으로 바닥을 잘 볼 수 있게 스폰
                        "tp @p 0 66 0 0 50",
                        # 인벤토리 초기화 후 oak_planks 64개 지급
                        "clear @p",
                        "give @p minecraft:oak_planks 64",
                    ],
                )
                env = make(
                    initial_env_config=config,
                    port=port,
                    verbose=False,
                    verbose_gradle=True,
                    render_action=False,
                )
                env = HierarchicalBuildingEnv(env, image_size=IMG_SIZE)
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
    print("=" * 60)
    print("   Oak Planks 건축 강화학습 - Hierarchical RL")
    print(f"   Manager : Rule-based (nearest empty sub-goal)")
    print(f"   Worker  : PPO (MultiInputPolicy)")
    print(f"   목표    : oak_planks {TOTAL_BLOCKS}개 배치 (3×3 바닥)")
    print(f"   학습 스텝: {total_timesteps:,}")
    print("=" * 60)

    run = wandb.init(
        project="craftground-building",
        name="ppo-hrl-2level-3x3",
        config={
            "algorithm": "PPO",
            "policy": "MultiInputPolicy",
            "task": "building_3x3_hrl",
            "manager": "rule_based_nearest",
            "total_blocks": TOTAL_BLOCKS,
            "max_steps": MAX_STEPS,
            "max_subgoal_steps": MAX_SUBGOAL_STEP,
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
        policy="MultiInputPolicy",
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
        tensorboard_log="./logs/building_hrl/",
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
        save_path = "building_hrl_model"
        model.save(save_path)
        print(f"\n모델 저장 완료: {save_path}.zip")
    except KeyboardInterrupt:
        print("\n학습 중단. 모델 저장 중...")
        model.save("building_hrl_interrupted")
        print("모델 저장 완료: building_hrl_interrupted.zip")
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

    cv2.namedWindow("Building HRL Eval", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Building HRL Eval", 512, 512)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            img = obs["image"]
            display = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.putText(
                display,
                f"EP {ep+1}  Placed:{info['placed']}/{TOTAL_BLOCKS}  R:{total_reward:.1f}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2,
            )
            cv2.imshow("Building HRL Eval", display)
            if cv2.waitKey(33) & 0xFF == ord("q"):
                done = True

        print(f"  EP {ep+1}: placed={info['placed']}/{TOTAL_BLOCKS}  "
              f"reward={total_reward:.2f}")

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
        path = sys.argv[2] if len(sys.argv) > 2 else "building_hrl_model"
        evaluate(path)
    else:
        print(__doc__)
