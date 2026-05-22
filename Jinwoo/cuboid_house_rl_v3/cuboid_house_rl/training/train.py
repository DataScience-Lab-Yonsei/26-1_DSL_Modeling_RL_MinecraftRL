"""
Main training script for Cuboid House Construction Agent (V3 — Hierarchical).

Hierarchical: shared MLP + per-stage (LSTM → MLP → head).
Each timestep is routed through the active stage's head; only that head
(plus the shared MLP) receives gradients.

Usage:
    python -m cuboid_house_rl.training.train --mode train
    python -m cuboid_house_rl.training.train --mode train --resume checkpoints/bc_pretrained.pt
    python -m cuboid_house_rl.training.train --mode eval --resume checkpoints/best.pt
"""
import argparse
import os
import time
from pathlib import Path
from collections import deque

import torch
import numpy as np

from cuboid_house_rl.config import (
    ACTION_DIMS, NUM_ACTION_DIMS, TOTAL_ACTION_LOGITS, PLANKS_SLOT,
    FLAT_OBS_SIZE,
    GAMMA, GAE_LAMBDA, LEARNING_RATE, CLIP_RATIO,
    ENTROPY_COEFF, ENTROPY_COEFF_START, ENTROPY_COEFF_END,
    VALUE_LOSS_COEFF, MAX_GRAD_NORM,
    BATCH_SIZE, MINI_BATCH_SIZE, UPDATE_EPOCHS, SEQUENCE_LENGTH,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    MAX_EPISODE_STEPS,
    NUM_STAGES, SUBTASK_TO_STAGE, SUBTASK_DONE,
    SLOT_PLANKS, SLOT_AXE,
)
from cuboid_house_rl.envs.house_building_env import HouseBuildingEnv
from cuboid_house_rl.expert.ceiling_expert import CeilingExpert
from cuboid_house_rl.models.network import HierarchicalActorCriticNetwork
from cuboid_house_rl.training.ppo import RecurrentPPO
from cuboid_house_rl.training.rollout_buffer import RolloutBuffer

# Maximum steps for expert LOOKING phase
LOOKING_MAX_STEPS = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cuboid House Construction Agent — Training & Evaluation (V3 Hierarchical)"
    )
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--keep-checkpoints", type=int, default=5)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file (PPO or BC pretrained)")

    # WandB
    parser.add_argument("--wandb-project", type=str, default="cuboid-house")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
    parser.add_argument("--clip-ratio", type=float, default=CLIP_RATIO)
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--entropy-start", type=float, default=ENTROPY_COEFF_START)
    parser.add_argument("--entropy-end", type=float, default=ENTROPY_COEFF_END)
    parser.add_argument("--value-coeff", type=float, default=VALUE_LOSS_COEFF)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--mini-batch-size", type=int, default=MINI_BATCH_SIZE)
    parser.add_argument("--update-epochs", type=int, default=UPDATE_EPOCHS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--lstm-layers", type=int, default=LSTM_NUM_LAYERS)

    # Evaluation
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_true")

    args = parser.parse_args()
    if args.no_deterministic:
        args.deterministic = False
    return args


def get_device(args):
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


# ==================================================================
# Environment
# ==================================================================

def make_env(args, seed=0, port=None, looking_phase=True):
    from cuboid_house_rl.envs.craftground_adapter import create_craftground_env
    cg_port = port if port is not None else 8023
    cg_env = create_craftground_env(port=cg_port)
    env = HouseBuildingEnv(craftground_env=cg_env)
    env.looking_phase = looking_phase
    env.reset(seed=seed)
    return env


def make_vec_envs(args, num_envs):
    envs = []
    for i in range(num_envs):
        port = 8023 + i
        env = make_env(args, seed=args.seed + i, port=port)
        envs.append(env)
    return envs


def get_stage_ids(envs):
    """Get current stage_id for each env."""
    return np.array(
        [SUBTASK_TO_STAGE.get(env.current_subtask, 0) for env in envs],
        dtype=np.int64,
    )


def run_looking_expert(env, max_steps=LOOKING_MAX_STEPS):
    """
    Run expert LOOKING sequence after building completion.

    Creates a CeilingExpert in finish-sequence mode and steps the env
    until done. These steps are NOT stored in the PPO buffer.

    Returns:
        obs: final observation after LOOKING (or after reset if looking finishes)
    """
    # Create CeilingExpert starting at the LOOKING (finish) phase
    expert = CeilingExpert(
        origin_x=env.origin_x,
        origin_z=env.origin_z,
        width=env.house_width,
        depth=env.house_depth,
        initial_hotbar=SLOT_AXE,
        door_x=env.origin_x + env.house_width // 2,
    )
    # Jump directly to finish sequence
    expert.state = CeilingExpert.MOVE_TO_DOOR_X
    expert._finish_ticks = 0
    expert._walk_out_ticks = 0

    for _ in range(max_steps):
        if expert.is_done():
            break
        action = expert.get_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return obs


# ==================================================================
# Checkpoints
# ==================================================================

def save_checkpoint(path, network, ppo, global_step, episode_count,
                    best_reward, args, wandb_run_id=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": ppo.get_state_dict(),
        "global_step": global_step,
        "episode_count": episode_count,
        "best_reward": best_reward,
        "config": vars(args),
        "wandb_run_id": wandb_run_id,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, network, ppo=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["model_state_dict"])
    if ppo is not None and "optimizer_state_dict" in checkpoint:
        ppo.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Checkpoint loaded: {path}")
    print(f"    Global step: {checkpoint.get('global_step', 0):,}")
    return checkpoint


def manage_checkpoints(checkpoint_dir, keep):
    ckpt_dir = Path(checkpoint_dir)
    step_files = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda f: int(f.stem.split("_")[1]),
    )
    while len(step_files) > keep:
        old = step_files.pop(0)
        old.unlink()


# ==================================================================
# Observation Helpers
# ==================================================================

def obs_to_tensor(obs_list, device):
    """Convert list of flat observations to batched tensor."""
    return torch.tensor(
        np.stack(obs_list), dtype=torch.float32, device=device
    )


def get_action_masks_batch(envs, device):
    masks = np.stack([env.action_masks() for env in envs])
    return torch.tensor(masks, dtype=torch.bool, device=device)


# ==================================================================
# Training Loop
# ==================================================================

def train(args):
    device = get_device(args)
    print(f"Training on {device}")

    # WandB
    wandb_run = None
    wandb_run_id = None
    if not args.no_wandb:
        try:
            import wandb
            if args.wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
            resume_config = {}
            if args.wandb_resume:
                resume_config["resume"] = args.wandb_resume
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                notes=args.wandb_notes,
                config=vars(args),
                **resume_config,
            )
            wandb_run_id = wandb_run.id
            print(f"  WandB run: {wandb_run.url}")
        except ImportError:
            print("  WandB not installed")
            args.no_wandb = True

    # Environments
    envs = make_vec_envs(args, args.num_envs)
    print(f"  Created {args.num_envs} environments")

    # Network (hierarchical)
    network = HierarchicalActorCriticNetwork(lstm_num_layers=args.lstm_layers).to(device)

    initial_entropy = args.entropy_coeff if args.entropy_coeff is not None else args.entropy_start

    ppo = RecurrentPPO(
        network, device,
        learning_rate=args.lr,
        clip_ratio=args.clip_ratio,
        entropy_coeff=initial_entropy,
        value_coeff=args.value_coeff,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch_size,
    )

    params = network.count_parameters()
    print(f"  Network parameters: {params['total']:,}")
    for key, val in params.items():
        if key != "total":
            print(f"    {key}: {val:,}")

    # Buffer
    steps_per_env = args.batch_size // args.num_envs
    buffer = RolloutBuffer(
        buffer_size=steps_per_env,
        num_envs=args.num_envs,
        device=device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        sequence_length=args.sequence_length,
    )

    # Resume
    global_step = 0
    episode_count = 0
    best_reward = float("-inf")

    if args.resume:
        checkpoint = load_checkpoint(args.resume, network, ppo)
        global_step = checkpoint.get("global_step", 0)
        episode_count = checkpoint.get("episode_count", 0)
        best_reward = checkpoint.get("best_reward", float("-inf"))

    # Tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_completions = deque(maxlen=100)
    episode_successes = deque(maxlen=100)

    # Init envs
    obs_list = []
    for env in envs:
        obs, info = env.reset(seed=args.seed)
        obs_list.append(obs)

    hidden_states = network.get_initial_hidden_state(args.num_envs, device)
    stage_ids = get_stage_ids(envs)
    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting training from step {global_step:,}")
    print(f"Target: {args.total_timesteps:,} timesteps")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
      while global_step < args.total_timesteps:

        network.eval()
        buffer.reset()

        for step in range(steps_per_env):
            with torch.no_grad():
                flat_tensor = obs_to_tensor(obs_list, device)
                action_masks = get_action_masks_batch(envs, device)
                stage_tensor = torch.tensor(stage_ids, dtype=torch.long, device=device)

                result = network(
                    flat_tensor, hidden_states,
                    stage_ids=stage_tensor,
                    action_masks=action_masks,
                )

                actions = result["actions"].cpu().numpy()
                log_probs = result["log_probs"].cpu().numpy()
                values = result["values"].cpu().numpy()
                new_hidden = result["hidden_states"]

            # Store in buffer
            flat_np = np.stack(obs_list)
            masks_np = np.stack([env.action_masks() for env in envs])

            dones = np.zeros(args.num_envs, dtype=np.bool_)
            truncs = np.zeros(args.num_envs, dtype=np.bool_)

            buffer.add(
                flat_np, actions, masks_np,
                np.zeros(args.num_envs), values, log_probs,
                dones, truncs,
                hidden_states,
                stage_ids,
            )

            # Step environments
            new_obs_list = []
            step_rewards = np.zeros(args.num_envs, dtype=np.float32)

            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, info = env.step(actions[i])
                step_rewards[i] = reward
                ep_rewards[i] += reward
                ep_lengths[i] += 1

                # Building complete → run expert LOOKING (not stored in buffer)
                if env.current_subtask == SUBTASK_DONE and not terminated and not truncated:
                    obs = run_looking_expert(env)
                    terminated = True  # treat as episode end for PPO

                if terminated or truncated:
                    episode_rewards.append(ep_rewards[i])
                    episode_lengths.append(ep_lengths[i])
                    episode_completions.append(
                        info.get("completion", {}).get("total_completion", 0.0)
                    )
                    reason = info.get("termination_reason", "unknown")
                    # Building completion + LOOKING = success
                    if env.current_subtask == SUBTASK_DONE or reason == "success":
                        episode_successes.append(1.0)
                    else:
                        episode_successes.append(0.0)
                    episode_count += 1

                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0
                    obs, _ = env.reset(seed=args.seed + episode_count)

                    # Reset ALL stage hidden states for this env
                    with torch.no_grad():
                        for s in range(NUM_STAGES):
                            new_hidden[s]["actor"][0][:, i, :] = 0.0
                            new_hidden[s]["actor"][1][:, i, :] = 0.0
                            new_hidden[s]["critic"][0][:, i, :] = 0.0
                            new_hidden[s]["critic"][1][:, i, :] = 0.0

                    dones[i] = True
                    truncs[i] = False

                new_obs_list.append(obs)

            buffer.rewards[buffer.pos - 1] = step_rewards
            buffer.dones[buffer.pos - 1] = dones
            buffer.truncs[buffer.pos - 1] = truncs

            obs_list = new_obs_list
            hidden_states = new_hidden
            stage_ids = get_stage_ids(envs)
            global_step += args.num_envs

        # Compute advantages
        with torch.no_grad():
            flat_tensor = obs_to_tensor(obs_list, device)
            stage_tensor = torch.tensor(stage_ids, dtype=torch.long, device=device)
            last_values, _ = network.get_value(flat_tensor, hidden_states, stage_tensor)
            last_values = last_values.cpu().numpy()

        last_dones = np.zeros(args.num_envs, dtype=np.bool_)
        buffer.compute_advantages(last_values, last_dones)

        # Entropy annealing
        if args.entropy_coeff is None:
            progress = min(global_step / args.total_timesteps, 1.0)
            ent_coeff = args.entropy_start + (args.entropy_end - args.entropy_start) * progress
            ppo.entropy_coeff = ent_coeff

        # PPO update
        network.train()
        metrics = ppo.update(buffer)

        # Logging
        if len(episode_rewards) > 0:
            log_data = {
                "train/reward_mean": np.mean(episode_rewards),
                "train/completion_mean": np.mean(episode_completions),
                "train/success_rate": np.mean(episode_successes),
                "train/entropy": metrics["entropy"],
                "train/policy_loss": metrics["policy_loss"],
                "train/value_loss": metrics["value_loss"],
                "train/episodes": episode_count,
                "train/global_step": global_step,
            }

            if not args.no_wandb and wandb_run:
                import wandb
                wandb.log(log_data, step=global_step)

            print(
                f"Step {global_step:>10,} | "
                f"Reward {np.mean(episode_rewards):>7.1f} | "
                f"Completion {np.mean(episode_completions):>5.1%} | "
                f"Success {np.mean(episode_successes):>5.1%} | "
                f"Entropy {metrics['entropy']:.3f} | "
                f"Episodes {episode_count}"
            )

        # Save
        if global_step % args.save_interval < args.num_envs * steps_per_env:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"step_{global_step}.pt"),
                network, ppo, global_step, episode_count, best_reward,
                args, wandb_run_id,
            )
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "latest.pt"),
                network, ppo, global_step, episode_count, best_reward,
                args, wandb_run_id,
            )
            manage_checkpoints(args.checkpoint_dir, args.keep_checkpoints)

            if len(episode_rewards) > 0:
                current_reward = np.mean(episode_rewards)
                if current_reward > best_reward:
                    best_reward = current_reward
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, "best.pt"),
                        network, ppo, global_step, episode_count, best_reward,
                        args, wandb_run_id,
                    )

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at step {global_step:,}")

    save_checkpoint(
        os.path.join(args.checkpoint_dir, "latest.pt"),
        network, ppo, global_step, episode_count, best_reward, args, wandb_run_id,
    )
    print(f"Training complete! {global_step:,} steps, {episode_count} episodes")

    if wandb_run:
        wandb_run.finish()


# ==================================================================
# Evaluation
# ==================================================================

def evaluate(args, network=None, device=None, num_episodes=None):
    if device is None:
        device = get_device(args)
    if network is None:
        network = HierarchicalActorCriticNetwork(lstm_num_layers=args.lstm_layers).to(device)
        if args.resume:
            load_checkpoint(args.resume, network)
        else:
            print("Warning: evaluating with random weights")

    if num_episodes is None:
        num_episodes = args.eval_episodes

    network.eval()
    env = make_env(args, seed=args.seed, port=8023, looking_phase=True)

    rewards = []
    completions = []
    successes = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        hidden = network.get_initial_hidden_state(1, device)
        ep_reward = 0.0
        done = False

        while not done:
            # Building complete → run expert LOOKING
            if env.current_subtask == SUBTASK_DONE:
                run_looking_expert(env)
                done = True
                break

            stage_id = SUBTASK_TO_STAGE.get(env.current_subtask, 0)
            with torch.no_grad():
                flat_tensor = torch.tensor(
                    obs[np.newaxis], dtype=torch.float32, device=device
                )
                mask = torch.tensor(
                    env.action_masks()[np.newaxis], dtype=torch.bool, device=device
                )
                stage_tensor = torch.tensor([stage_id], dtype=torch.long, device=device)
                result = network(
                    flat_tensor, hidden,
                    stage_ids=stage_tensor,
                    action_masks=mask,
                    deterministic=args.deterministic,
                )
                action = result["actions"].cpu().numpy()[0]
                hidden = result["hidden_states"]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        comp = info.get("completion", {})
        reason = info.get("termination_reason", "unknown")
        rewards.append(ep_reward)
        completions.append(comp.get("total_completion", 0.0))
        is_success = env.current_subtask == SUBTASK_DONE or reason == "success"
        successes.append(1.0 if is_success else 0.0)

        print(
            f"  Eval ep {ep + 1}/{num_episodes}: "
            f"reward={ep_reward:.1f} | "
            f"completion={comp.get('total_completion', 0):.1%} | "
            f"{reason}"
        )

    metrics = {
        "reward_mean": np.mean(rewards),
        "completion_mean": np.mean(completions),
        "success_rate": np.mean(successes),
    }
    print(f"\nEval summary: reward={metrics['reward_mean']:.1f} | "
          f"completion={metrics['completion_mean']:.1%} | "
          f"success={metrics['success_rate']:.1%}")
    return metrics


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
