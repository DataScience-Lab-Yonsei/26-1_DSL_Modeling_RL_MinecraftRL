import os
import cv2
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.logger import configure
from wandb.integration.sb3 import WandbCallback

from craftground import InitialEnvironmentConfig, make


# =========================
# 0. 환경 변수 설정
# =========================
os.environ["CONDA_PREFIX"] = r"C:/Users/chlgu/anaconda3/envs/craftground"

JAVA_PATH = r"C:/Program Files/Java/jdk-21.0.10"
os.environ["JAVA_HOME"] = JAVA_PATH
os.environ["PATH"] = os.path.join(JAVA_PATH, "bin") + ";" + os.environ.get("PATH", "")


# =========================
# 1. 액션 정의
# =========================
ACTION_MEANINGS = {
    0: "forward",
    1: "turn_left_small",
    2: "turn_right_small",
    3: "turn_left_big",
    4: "turn_right_big",
    5: "look_up_small",
    6: "look_down_small",
}

# CraftGround camera neutral index
CAMERA_NEUTRAL = 12

# yaw / pitch step 크기
YAW_LEFT_SMALL = 10
YAW_RIGHT_SMALL = 14
YAW_LEFT_BIG = 8
YAW_RIGHT_BIG = 16

PITCH_UP_SMALL = 10
PITCH_DOWN_SMALL = 14


def make_macro_action_simple(action_id: int) -> np.ndarray:
    """
    CraftGround action vector:
    [move_fb, strafe, modifier, pitch, yaw, interaction, craft_arg, item_arg]
    """
    action = np.array([0, 0, 0, CAMERA_NEUTRAL, CAMERA_NEUTRAL, 0, 0, 0], dtype=np.int64)

    if action_id == 0:
        action[0] = 1   # forward
    elif action_id == 1:
        action[4] = YAW_LEFT_SMALL
    elif action_id == 2:
        action[4] = YAW_RIGHT_SMALL
    elif action_id == 3:
        action[4] = YAW_LEFT_BIG
    elif action_id == 4:
        action[4] = YAW_RIGHT_BIG
    elif action_id == 5:
        action[3] = PITCH_UP_SMALL
    elif action_id == 6:
        action[3] = PITCH_DOWN_SMALL
    else:
        raise ValueError(f"Invalid action_id: {action_id}")

    return action


def make_noop_action() -> np.ndarray:
    return np.array([0, 0, 0, CAMERA_NEUTRAL, CAMERA_NEUTRAL, 0, 0, 0], dtype=np.int64)


# =========================
# 2. 환경 정의
# =========================
class RedBlockApproachEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        port=8888,
        render=False,
        max_steps=120,
        min_target_dist=8,
        max_target_dist=10,
        arena_half_size=50,
        player_y=1,
        target_block_count=1,
    ):
        super().__init__()

        self.render_enabled = render
        self.max_steps = max_steps
        self.min_target_dist = min_target_dist
        self.max_target_dist = max_target_dist
        self.arena_half_size = arena_half_size
        self.player_y = player_y
        self.target_block_count = target_block_count
        self.step_count = 0
        self.prev_red_ratio = 0.0
        self.prev_center_red = 0.0

        self.last_action = None
        self.same_action_count = 0

        test_config = InitialEnvironmentConfig(
            image_width=256,
            image_height=256,
            seed="42",
            render_distance=2,
        )

        self.env = make(
            initial_env_config=test_config,
            port=port,
            verbose=False,
            render_action=True,
        )

        self.action_space = spaces.Discrete(len(ACTION_MEANINGS))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, 84, 84),
            dtype=np.uint8,
        )

    # -------------------------
    # 관측 처리
    # -------------------------
    def _preprocess_obs(self, obs):
        img = np.array(obs["rgb"], dtype=np.uint8)

        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.uint8)

    def _rgb_image(self, obs):
        img = np.array(obs["rgb"], dtype=np.uint8)

        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        return img

    # -------------------------
    # 빨간색 검출
    # -------------------------
    def _red_ratio(self, obs):
        img = self._rgb_image(obs)

        red_mask = (
            (img[:, :, 0] >= 150) &
            (img[:, :, 1] <= 95) &
            (img[:, :, 2] <= 95)
        )
        return float(red_mask.mean())

    def _center_red_ratio(self, obs):
        img = self._rgb_image(obs)
        h, w, _ = img.shape

        cx1, cx2 = w // 2 - 24, w // 2 + 24
        cy1, cy2 = h // 2 - 24, h // 2 + 24
        center = img[cy1:cy2, cx1:cx2]

        red_mask = (
            (center[:, :, 0] >= 150) &
            (center[:, :, 1] <= 95) &
            (center[:, :, 2] <= 95)
        )
        return float(red_mask.mean())

    # -------------------------
    # 명령 helper
    # -------------------------
    def _try_add_commands(self, cmds):
        if hasattr(self.env, "add_commands"):
            try:
                self.env.add_commands(cmds)
                return True
            except Exception as e:
                print("⚠ env.add_commands 실패:", e)

        if hasattr(self.env, "get_wrapper_attr"):
            try:
                add_commands_fn = self.env.get_wrapper_attr("add_commands")
                add_commands_fn(cmds)
                return True
            except Exception as e:
                print("⚠ get_wrapper_attr('add_commands') 실패:", e)

        return False

    # -------------------------
    # arena 생성
    # -------------------------
    def _build_arena_and_spawn_player(self):
        s = self.arena_half_size

        yaw = int(np.random.choice([0, 90, 180, 270]))
        pitch = 10   # 시작부터 약간 아래 보게 설정해도 됨. 필요 없으면 0으로 바꿔도 됨.

        cmds = [
            "time set day",
            "weather clear",

            f"fill ~-{s} ~-2 ~-{s} ~{s} ~8 ~{s} minecraft:air",
            f"fill ~-{s} ~-1 ~-{s} ~{s} ~-1 ~{s} minecraft:smooth_stone",
            f"fill ~-{s} ~ ~-{s} ~{s} ~2 ~{s} minecraft:air",

            "setblock ~ ~-1 ~ minecraft:stone",
            f"tp @p ~ ~ ~ {yaw} {pitch}",
        ]

        ok = self._try_add_commands(cmds)
        if not ok:
            print("⚠ arena 생성 및 플레이어 스폰 명령을 넣지 못했습니다.")

    # -------------------------
    # 목표 생성
    # -------------------------
    def _sample_target_offset(self):
        for _ in range(100):
            r = np.random.randint(self.min_target_dist, self.max_target_dist + 1)
            theta = np.random.uniform(0.0, 2.0 * math.pi)

            dx = int(np.round(r * math.cos(theta)))
            dz = int(np.round(r * math.sin(theta)))

            if dx == 0 and dz == 0:
                continue

            dist = np.sqrt(dx**2 + dz**2)
            if self.min_target_dist - 0.25 <= dist <= self.max_target_dist + 0.75:
                return dx, dz, dist

        dx, dz = 0, self.min_target_dist
        dist = np.sqrt(dx**2 + dz**2)
        return dx, dz, dist

    def _place_red_target(self):
        dx, dz, dist = self._sample_target_offset()

        cmds = [
            "fill ~-8 ~-1 ~-8 ~8 ~3 ~8 minecraft:air replace minecraft:red_wool",
        ]

        if self.target_block_count == 1:
            cmds.append(f"setblock ~{dx} ~ ~{dz} minecraft:red_wool")

        elif self.target_block_count == 3:
            cmds.extend([
                f"setblock ~{dx-1} ~ ~{dz} minecraft:red_wool",
                f"setblock ~{dx} ~ ~{dz} minecraft:red_wool",
                f"setblock ~{dx+1} ~ ~{dz} minecraft:red_wool",
            ])

        else:
            raise ValueError(f"Unsupported target_block_count: {self.target_block_count}")

        print(
            f"🎯 target spawn offset = (dx={dx}, dz={dz}), "
            f"dist={dist:.2f}, blocks={self.target_block_count}"
        )

        ok = self._try_add_commands(cmds)
        if not ok:
            print("⚠ 목표 블록 배치 명령을 넣지 못했습니다.")

    # -------------------------
    # reset
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.last_action = None
        self.same_action_count = 0

        obs, info = self.env.reset()

        self._build_arena_and_spawn_player()
        self._place_red_target()

        obs, _, terminated, truncated, info = self.env.step(make_noop_action())

        if terminated or truncated:
            obs, info = self.env.reset()

        self.prev_red_ratio = self._red_ratio(obs)
        self.prev_center_red = self._center_red_ratio(obs)

        if not isinstance(info, dict):
            info = {}

        info["red_ratio"] = self.prev_red_ratio
        info["center_red"] = self.prev_center_red

        return self._preprocess_obs(obs), info

    # -------------------------
    # step
    # -------------------------
    def step(self, action_id):
        self.step_count += 1
        action_id = int(action_id)

        if self.last_action == action_id:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        self.last_action = action_id

        real_action = make_macro_action_simple(action_id)
        obs, _, terminated, truncated, info = self.env.step(real_action)

        if not isinstance(info, dict):
            info = {}

        red_ratio = self._red_ratio(obs)
        center_red = self._center_red_ratio(obs)

        delta_red = red_ratio - self.prev_red_ratio
        delta_center = center_red - self.prev_center_red

        reward = -0.03

        if delta_red > 0:
            reward += delta_red * 10.0
        else:
            reward += delta_red * 1.5

        if delta_center > 0:
            reward += delta_center * 50.0
        else:
            reward += delta_center * 3.0

        reward += red_ratio * 1.0
        reward += center_red * 8.0

        # 빨간색이 거의 안 보일 때는 탐색 행동 장려
        if red_ratio < 0.001 and center_red < 0.001:
            if action_id in [1, 2, 3, 4, 5, 6]:
                reward += 0.05
            elif action_id == 0:
                reward -= 0.05

        # 같은 회전/시점 액션 반복 패널티
        if action_id in [1, 2, 3, 4, 5, 6] and self.same_action_count >= 8:
            reward -= 0.05

        # 목표가 보이면 전진 장려
        if red_ratio >= 0.005 and action_id == 0:
            reward += 0.08

        if center_red >= 0.02 and action_id == 0:
            reward += 0.10

        # 중앙에 들어왔는데 계속 회전/시점변경하면 약한 패널티
        if center_red >= 0.02 and action_id in [1, 2, 3, 4, 5, 6]:
            reward -= 0.03

        success = (center_red >= 0.06) or (red_ratio >= 0.045)

        if success:
            reward += 20.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True
            reward -= 3.0

        self.prev_red_ratio = red_ratio
        self.prev_center_red = center_red

        info["is_success"] = success
        info["red_ratio"] = red_ratio
        info["center_red"] = center_red
        info["delta_red"] = delta_red
        info["delta_center"] = delta_center

        if self.render_enabled:
            self.render_with_text(
                obs, action_id, red_ratio, center_red, reward, success
            )

        return self._preprocess_obs(obs), reward, terminated, truncated, info

    # -------------------------
    # 렌더링
    # -------------------------
    def render_with_text(self, obs, action_id, red_ratio, center_red, reward, success):
        img = self._rgb_image(obs)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, f"Action: {ACTION_MEANINGS[int(action_id)]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Red ratio: {red_ratio:.4f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Center red: {center_red:.4f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Reward: {reward:.4f}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Success: {success}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Step: {self.step_count}", (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 255, 220), 2, cv2.LINE_AA)

        h, w, _ = frame.shape
        cx1, cx2 = w // 2 - 24, w // 2 + 24
        cy1, cy2 = h // 2 - 24, h // 2 + 24
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)

        cv2.imshow("Red Block Approach Task", frame)
        key = cv2.waitKey(80) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt("User requested quit")

    def close(self):
        self.env.close()
        cv2.destroyAllWindows()


# =========================
# 3. W&B용 커스텀 콜백
# =========================
class CustomWandbMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            if "red_ratio" in info:
                wandb.log({"env/red_ratio": float(info["red_ratio"])})

            if "center_red" in info:
                wandb.log({"env/center_red": float(info["center_red"])})

            if "delta_red" in info:
                wandb.log({"env/delta_red": float(info["delta_red"])})

            if "delta_center" in info:
                wandb.log({"env/delta_center": float(info["delta_center"])})

            if "is_success" in info:
                wandb.log({"env/is_success_step": float(info["is_success"])})

            done = False
            if i < len(dones):
                done = dones[i]

            if done:
                ep_info = info.get("episode")
                if ep_info is not None:
                    wandb.log(
                        {
                            "episode/reward": float(ep_info["r"]),
                            "episode/length": float(ep_info["l"]),
                            "episode/time": float(ep_info["t"]),
                        }
                    )

                if "is_success" in info:
                    wandb.log({"episode/success": float(info["is_success"])})

                if "red_ratio" in info:
                    wandb.log({"episode/final_red_ratio": float(info["red_ratio"])})

                if "center_red" in info:
                    wandb.log({"episode/final_center_red": float(info["center_red"])})

        return True


# =========================
# 4. env 생성
# =========================
def make_env(
    max_steps=150,
    min_target_dist=8,
    max_target_dist=10,
    port=8888,
    render=False,
    target_block_count=1
):
    return Monitor(
        RedBlockApproachEnv(
            port=port,
            render=render,
            max_steps=max_steps,
            min_target_dist=min_target_dist,
            max_target_dist=max_target_dist,
            target_block_count=target_block_count,
        ),
        info_keywords=("is_success", "red_ratio", "center_red"),
    )


def make_test_env(
    max_steps=150,
    min_target_dist=8,
    max_target_dist=10,
    port=8888,
    target_block_count=1
):
    return make_env(
        max_steps=max_steps,
        min_target_dist=min_target_dist,
        max_target_dist=max_target_dist,
        port=port,
        render=True,
        target_block_count=target_block_count
    )


# =========================
# 5. 랜덤 테스트
# =========================
def random_test():
    env = RedBlockApproachEnv(
        port=8888,
        render=True,
        max_steps=120,
        min_target_dist=8,
        max_target_dist=10,
        target_block_count=1,
    )

    obs, info = env.reset()
    episode_idx = 0

    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"[episode {episode_idx}] "
                f"action={ACTION_MEANINGS[int(action)]}, "
                f"reward={reward:.4f}, "
                f"red={info.get('red_ratio', 0):.4f}, "
                f"center={info.get('center_red', 0):.4f}, "
                f"success={info.get('is_success', False)}"
            )

            if terminated or truncated:
                print(f"episode end -> reset, success={info.get('is_success', False)}")
                episode_idx += 1
                obs, info = env.reset()

    finally:
        env.close()


# =========================
# 6. 초기 학습
# =========================
def train():
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 200000,
        "learning_rate": 2e-5,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.05,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "frame_stack": 8,
        "max_steps": 150,
        "min_target_dist": 8,
        "max_target_dist": 10,
        "task": "craftground_red_block_arena_search_align_approach_pitch_added",
    }

    run = wandb.init(
        project="craftground-rl",
        name="ppo-red-block",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = DummyVecEnv([
        lambda: make_env(
            max_steps=config["max_steps"],
            min_target_dist=config["min_target_dist"],
            max_target_dist=config["max_target_dist"],
            port=8888,
            render=False,
        )
    ])
    env = VecFrameStack(env, n_stack=config["frame_stack"])

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="ppo_red_block",
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"./wandb_models/{run.id}",
        model_save_freq=10000,
        verbose=2,
    )

    custom_metrics_callback = CustomWandbMetricsCallback()

    callback = CallbackList([
        checkpoint_callback,
        wandb_callback,
        custom_metrics_callback,
    ])

    model = PPO(
        config["policy_type"],
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        tensorboard_log="./tb_red_block/",
    )

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
        )

        final_model_path = "ppo_red_block"
        model.save(final_model_path)

        artifact = wandb.Artifact("ppo-red-block-model", type="model")
        artifact.add_file(final_model_path + ".zip")
        wandb.log_artifact(artifact)

    finally:
        env.close()
        run.finish()


# =========================
# 7. 이어 학습
# =========================
def train_continue():
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 200000,
        "learning_rate": 5e-6,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.05,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "frame_stack": 8,
        "max_steps": 150,
        "min_target_dist": 8,
        "max_target_dist": 10,
        "target_block_count": 1,
    }

    run = wandb.init(
        project="craftground-rl",
        name="ppo-red-block",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = DummyVecEnv([
        lambda: make_env(
            max_steps=config["max_steps"],
            min_target_dist=config["min_target_dist"],
            max_target_dist=config["max_target_dist"],
            port=8888,
            render=False,
            target_block_count=config["target_block_count"],
        )
    ])
    
    env = VecFrameStack(env, n_stack=config["frame_stack"])

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="ppo_red_block",
    )

    wandb_callback = WandbCallback(
        model_save_path=f"./wandb_models/{run.id}",
        model_save_freq=10000,
        verbose=2,
    )

    custom_metrics_callback = CustomWandbMetricsCallback()

    callback = CallbackList([
        checkpoint_callback,
        wandb_callback,
        custom_metrics_callback,
    ])

    model = PPO.load(
        "ppo_red_block_arena_search_align_approach_pitch_added_final.zip",
        env=env,
        device="auto",
    )

    new_logger = configure(
        "./tb_red_block/",
        ["stdout", "tensorboard"]
    )
    model.set_logger(new_logger)
    model.verbose = 1

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            reset_num_timesteps=False,
        )

        model.save("ppo_red_block")

        artifact = wandb.Artifact(
            "ppo-red-block-model",
            type="model"
        )

        artifact.add_file("ppo_red_block.zip")
        wandb.log_artifact(artifact)
    finally:
        env.close()
        run.finish()


# =========================
# 8. 테스트
# =========================
def test():
    frame_stack = 8
    max_steps = 150
    min_target_dist = 8
    max_target_dist = 10
    target_block_count = 1

    env = DummyVecEnv([
        lambda: make_test_env(
            max_steps=max_steps,
            min_target_dist=min_target_dist,
            max_target_dist=max_target_dist,
            port=8888,
            target_block_count=target_block_count,
        )
    ])
    env = VecFrameStack(env, n_stack=frame_stack)

    model = PPO.load("ppo_red_block.zip")

    obs = env.reset()

    episode_idx = 0
    success_count = 0
    total_reward = 0.0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = env.step(action)

            pred_a = int(action[0])
            print(f"pred action = {pred_a}, meaning = {ACTION_MEANINGS[pred_a]}")

            total_reward += float(rewards[0])

            if dones[0]:
                info = infos[0]

                is_success = info.get("is_success", False)
                red_ratio = info.get("red_ratio", 0.0)
                center_red = info.get("center_red", 0.0)

                if is_success:
                    success_count += 1

                episode_idx += 1
                success_rate = success_count / episode_idx

                print(
                    f"[episode {episode_idx}] "
                    f"success={is_success}, "
                    f"reward={total_reward:.4f}, "
                    f"red={red_ratio:.4f}, "
                    f"center={center_red:.4f}, "
                    f"success_rate={success_rate:.3f}"
                )

                total_reward = 0.0

    except KeyboardInterrupt:
        print("\n테스트 종료")
        if episode_idx > 0:
            print(
                f"총 에피소드: {episode_idx}, "
                f"성공 횟수: {success_count}, "
                f"최종 성공률: {success_count / episode_idx:.3f}"
            )

    finally:
        env.close()


# =========================
# 9. 실행부
# =========================
if __name__ == "__main__":
    # train()
    # train_continue()
    # random_test()
    test()