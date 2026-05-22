"""
BC 모델 평가 / PPO 파인튜닝.

사용법:
    python bc_eval.py eval      [에피소드수]
    python bc_eval.py finetune  [총스텝수]
"""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from building_hrl import make_env, ACTIONS, TOTAL_BLOCKS, cleanup_old_instances
from config import (
    PORT, BC_MODEL_PATH, OBS_DIM,
    FINETUNE_TOTAL_TIMESTEPS, FINETUNE_LR, FINETUNE_ENT_COEF, LOG_DIR,
)
from env import VectorBuildingEnv
from model import MLPPolicy
from bc_train import N_ACTIONS
from collect_demos import _make_vector_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ── BC 평가 ──────────────────────────────────────────────────────
def evaluate(model_path: str = BC_MODEL_PATH, n_episodes: int = 5):
    model = MLPPolicy.load(model_path, obs_dim=OBS_DIM, n_actions=N_ACTIONS)

    cleanup_old_instances()
    env = _make_vector_env(seed="99")

    cv2.namedWindow("BC Eval", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BC Eval", 512, 512)

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r, done, step = 0.0, False, 0

        while not done:
            action = model.predict(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
            step += 1

            # 렌더 (VectorBuildingEnv는 last_obs에 image 유지)
            if env.last_obs is not None:
                display = cv2.cvtColor(
                    cv2.resize(env.last_obs, (512, 512), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_RGB2BGR,
                )
                cv2.putText(display,
                            f"EP{ep+1} Placed:{info.get('placed',0)}/{TOTAL_BLOCKS} "
                            f"R:{total_r:.1f} [{ACTIONS[action]}]",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)
                cv2.imshow("BC Eval", display)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    done = True

        placed = info.get("placed", 0)
        results.append(placed)
        print(f"  EP {ep+1}: placed={placed}/{TOTAL_BLOCKS}  reward={total_r:.2f}  steps={step}")

    env.close()
    cv2.destroyAllWindows()
    print(f"\n평균 배치: {np.mean(results):.1f}/{TOTAL_BLOCKS} | "
          f"성공률: {np.mean([p >= TOTAL_BLOCKS for p in results])*100:.0f}%")


# ── PPO 파인튜닝 ──────────────────────────────────────────────────
def finetune(total_timesteps: int = FINETUNE_TOTAL_TIMESTEPS):
    import wandb
    from wandb.integration.sb3 import WandbCallback
    from stable_baselines3.common.monitor import Monitor

    print(f"[finetune] PPO 파인튜닝 ({total_timesteps:,} 스텝)")
    cleanup_old_instances()

    def _env_factory():
        env = _make_vector_env()
        return Monitor(env)

    vec_env = DummyVecEnv([_env_factory])

    model = PPO(
        policy="MlpPolicy",        # 벡터 obs → MlpPolicy
        env=vec_env,
        verbose=1,
        learning_rate=FINETUNE_LR,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=FINETUNE_ENT_COEF,
        tensorboard_log=os.path.join(LOG_DIR, "finetune"),
        device="auto",
    )

    run = wandb.init(
        project="craftground-building",
        name="bc-ppo-finetune-vector",
        config={"algorithm": "BC→PPO", "obs": "vector_41d", "total_timesteps": total_timesteps},
        sync_tensorboard=True,
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=WandbCallback(gradient_save_freq=1000,
                                   model_save_path=f"models/{run.id}", verbose=2),
        )
        model.save("bc_ppo_model")
        print("[finetune] 저장: bc_ppo_model.zip")
    except KeyboardInterrupt:
        model.save("bc_ppo_interrupted")
    finally:
        vec_env.close()
        run.finish()


# ── 진입점 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if mode == "eval":
        evaluate(n_episodes=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif mode == "finetune":
        finetune(total_timesteps=int(sys.argv[2]) if len(sys.argv) > 2 else FINETUNE_TOTAL_TIMESTEPS)
    else:
        print(__doc__)
