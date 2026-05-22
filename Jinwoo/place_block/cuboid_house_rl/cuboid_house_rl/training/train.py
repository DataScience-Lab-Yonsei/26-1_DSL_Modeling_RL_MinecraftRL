"""
Main training script for Cuboid House Construction Agent.

Modes:
    train   — Full training loop with PPO, WandB logging, checkpoints
    eval    — Evaluate a trained agent over multiple episodes
    preview — Watch the agent build in real-time
    record  — Save episode videos to disk

Usage:
    python -m cuboid_house_rl.training.train --mode train
    python -m cuboid_house_rl.training.train --mode eval --resume checkpoints/best.pt
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path
from collections import deque

import torch
import numpy as np

from cuboid_house_rl.config import (
    # Actions
    ACTION_DIMS, NUM_ACTION_DIMS, TOTAL_ACTION_LOGITS, PLANKS_SLOT,
    STACKED_CHANNELS, LOCAL_WINDOW_SIZE, NON_VOXEL_SIZE,
    # PPO
    GAMMA, GAE_LAMBDA, LEARNING_RATE, CLIP_RATIO,
    ENTROPY_COEFF, ENTROPY_COEFF_START, ENTROPY_COEFF_END,
    VALUE_LOSS_COEFF, MAX_GRAD_NORM,
    BATCH_SIZE, MINI_BATCH_SIZE, UPDATE_EPOCHS, SEQUENCE_LENGTH,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    # Episode
    MAX_EPISODE_STEPS,
)
from cuboid_house_rl.envs.house_building_env import HouseBuildingEnv
from cuboid_house_rl.models.network import ActorCriticNetwork
from cuboid_house_rl.training.ppo import RecurrentPPO
from cuboid_house_rl.training.rollout_buffer import RolloutBuffer


# ==================================================================
# Argument Parser
# ==================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cuboid House Construction Agent — Training & Evaluation"
    )

    # General
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "preview", "record"])
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
                        help="Path to checkpoint .pt file")

    # WandB
    parser.add_argument("--wandb-project", type=str, default="cuboid-house")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default=None,
                        help="'must', 'allow', 'never', or run ID")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-notes", type=str, default=None, help="WandB run notes")
    parser.add_argument("--no-wandb", action="store_true")

    # Evaluation
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_true")

    # Recording
    parser.add_argument("--record-path", type=str, default="videos")
    parser.add_argument("--record-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")

    # PPO hyperparameters (override config.py)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
    parser.add_argument("--clip-ratio", type=float, default=CLIP_RATIO)
    parser.add_argument("--entropy-coeff", type=float, default=None,
                        help="Fixed entropy coeff (disables annealing)")
    parser.add_argument("--entropy-start", type=float, default=ENTROPY_COEFF_START)
    parser.add_argument("--entropy-end", type=float, default=ENTROPY_COEFF_END)
    parser.add_argument("--value-coeff", type=float, default=VALUE_LOSS_COEFF)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--mini-batch-size", type=int, default=MINI_BATCH_SIZE)
    parser.add_argument("--update-epochs", type=int, default=UPDATE_EPOCHS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--lstm-layers", type=int, default=LSTM_NUM_LAYERS)

    # Curriculum
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2],
                        help="1=gaze training, 2=house building (default)")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Load Stage 1 weights for Stage 2 training")

    args = parser.parse_args()

    if args.no_deterministic:
        args.deterministic = False

    return args


# ==================================================================
# Environment Creation
# ==================================================================

def make_env(args, seed=0, port=None):
    """
    Create a single environment instance.

    Args:
        args: parsed arguments.
        seed: random seed.
        port: CraftGround server port.
    """
    from cuboid_house_rl.envs.craftground_adapter import (
        create_craftground_env,
    )
    cg_port = port if port is not None else 8023
    cg_env = create_craftground_env(port=cg_port)

    if args.stage == 1:
        from cuboid_house_rl.envs.gaze_training_env import GazeTrainingEnv
        env = GazeTrainingEnv(craftground_env=cg_env)
    else:
        env = HouseBuildingEnv(craftground_env=cg_env)

    env.reset(seed=seed)
    return env


def make_vec_envs(args, num_envs):
    """Create multiple environments (simple sequential for now)."""
    envs = []
    for i in range(num_envs):
        # Each CraftGround env needs its own port
        port = 8023 + i
        env = make_env(args, seed=args.seed + i, port=port)
        envs.append(env)
    return envs


# ==================================================================
# Checkpoint Management
# ==================================================================

def save_checkpoint(
    path: str,
    network: ActorCriticNetwork,
    ppo: RecurrentPPO,
    global_step: int,
    episode_count: int,
    best_reward: float,
    args,
    wandb_run_id: str = None,
):
    """Save training checkpoint."""
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


def load_checkpoint(path: str, network: ActorCriticNetwork, ppo: RecurrentPPO = None):
    """Load checkpoint. Returns metadata dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["model_state_dict"])
    if ppo is not None and "optimizer_state_dict" in checkpoint:
        ppo.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Checkpoint loaded: {path}")
    print(f"    Global step: {checkpoint.get('global_step', 0):,}")
    print(f"    Best reward: {checkpoint.get('best_reward', 0):.2f}")
    return checkpoint


def manage_checkpoints(checkpoint_dir: str, keep: int):
    """Remove old checkpoints, keeping only the most recent N + best."""
    ckpt_dir = Path(checkpoint_dir)
    step_files = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda f: int(f.stem.split("_")[1]),
    )
    # Keep latest N, delete the rest
    while len(step_files) > keep:
        old = step_files.pop(0)
        old.unlink()
        print(f"  Removed old checkpoint: {old}")


# ==================================================================
# Observation Helpers
# ==================================================================

def obs_to_tensors(obs_list, device):
    """Convert list of observations (from multiple envs) to batched tensors."""
    voxels = []
    flats = []
    for obs in obs_list:
        # Voxel grids: (W, W, W, C) -> (C, W, W, W) channels first
        v = np.transpose(obs["voxel_grids"], (3, 0, 1, 2))
        voxels.append(v)
        flats.append(obs["flat_features"])

    voxel_tensor = torch.tensor(np.stack(voxels), dtype=torch.float32, device=device)
    flat_tensor = torch.tensor(np.stack(flats), dtype=torch.float32, device=device)
    return voxel_tensor, flat_tensor


def get_action_masks_batch(envs, device):
    """Get action masks from all environments as a batch tensor."""
    masks = np.stack([env.action_masks() for env in envs])
    return torch.tensor(masks, dtype=torch.bool, device=device)


# ==================================================================
# Training Loop
# ==================================================================

def train(args):
    """Main training loop."""
    device = get_device(args)
    print(f"Training on {device}")

    # WandB setup
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
            if args.resume:
                # Load run ID from checkpoint
                ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
                saved_id = ckpt.get("wandb_run_id")
                if saved_id and args.wandb_resume == "must":
                    resume_config["id"] = saved_id

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
            print("  WandB not installed, logging disabled")
            args.no_wandb = True

    # Create environments
    envs = make_vec_envs(args, args.num_envs)
    print(f"  Created {args.num_envs} environments")

    # Create network and PPO
    network = ActorCriticNetwork(lstm_num_layers=args.lstm_layers).to(device)

    # Stage 1: apply gaze-specific initial bias
    if args.stage == 1:
        from cuboid_house_rl.config import GAZE_INITIAL_BIAS
        network._apply_initial_bias(GAZE_INITIAL_BIAS)

    # Load Stage 1 checkpoint for Stage 2 transfer
    if args.stage == 2 and args.stage1_checkpoint:
        print(f"  Loading Stage 1 weights from {args.stage1_checkpoint}")
        ckpt = torch.load(args.stage1_checkpoint, map_location=device)
        network.load_state_dict(ckpt["network_state_dict"])
        print("  Stage 1 weights loaded successfully")

    # Entropy annealing: use fixed value if --entropy-coeff is set, else anneal
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

    # Resume from checkpoint
    global_step = 0
    episode_count = 0
    best_reward = float("-inf")

    if args.resume:
        checkpoint = load_checkpoint(args.resume, network, ppo)
        global_step = checkpoint.get("global_step", 0)
        episode_count = checkpoint.get("episode_count", 0)
        best_reward = checkpoint.get("best_reward", float("-inf"))

    # Episode tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_completions = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    episode_stuck = deque(maxlen=100)
    episode_timeouts = deque(maxlen=100)

    # Initialize environments
    obs_list = []
    for env in envs:
        obs, info = env.reset(seed=args.seed)
        obs_list.append(obs)

    hidden_states = network.get_initial_hidden_state(args.num_envs, device)
    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting training from step {global_step:,}")
    print(f"Target: {args.total_timesteps:,} timesteps")
    print(f"{'='*60}\n")

    start_time = time.time()
    last_log_step = global_step

    # ---- Main training loop ----
    interrupted = False
    try:
      while global_step < args.total_timesteps:

        # ---- Collect rollout ----
        network.eval()
        buffer.reset()

        for step in range(steps_per_env):
            with torch.no_grad():
                voxel_tensor, flat_tensor = obs_to_tensors(obs_list, device)
                action_masks = get_action_masks_batch(envs, device)

                result = network(
                    voxel_tensor, flat_tensor, hidden_states,
                    action_masks=action_masks,
                )

                actions = result["actions"].cpu().numpy()
                log_probs = result["log_probs"].cpu().numpy()
                values = result["values"].cpu().numpy()
                new_hidden = result["hidden_states"]

            # Store in buffer
            voxels_np = np.stack([
                np.transpose(obs["voxel_grids"], (3, 0, 1, 2))
                for obs in obs_list
            ])
            flats_np = np.stack([obs["flat_features"] for obs in obs_list])
            masks_np = np.stack([env.action_masks() for env in envs])

            dones = np.zeros(args.num_envs, dtype=np.bool_)
            truncs = np.zeros(args.num_envs, dtype=np.bool_)

            buffer.add(
                voxels_np, flats_np, actions, masks_np,
                np.zeros(args.num_envs), values, log_probs,
                dones, truncs,
                hidden_states["actor"], hidden_states["critic"],
            )

            # Step environments
            new_obs_list = []
            step_rewards = np.zeros(args.num_envs, dtype=np.float32)

            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, info = env.step(actions[i])
                step_rewards[i] = reward
                ep_rewards[i] += reward
                ep_lengths[i] += 1

                if terminated or truncated:
                    # Log episode
                    episode_rewards.append(ep_rewards[i])
                    episode_lengths.append(ep_lengths[i])
                    if args.stage == 1:
                        gaze_rate = info.get("gaze_success_rate", 0.0)
                        episode_completions.append(gaze_rate)
                    else:
                        episode_completions.append(
                            info.get("completion", {}).get("total_completion", 0.0)
                        )
                    reason = info.get("termination_reason", "unknown")
                    episode_successes.append(1.0 if reason == "success" else 0.0)
                    episode_stuck.append(1.0 if reason == "stuck" else 0.0)
                    episode_timeouts.append(1.0 if reason == "timeout" else 0.0)
                    episode_count += 1

                    # Stage 1: check graduation
                    if args.stage == 1 and info.get("gaze_graduated", False):
                        grad_path = os.path.join(
                            args.checkpoint_dir, "stage1_graduated.pt"
                        )
                        save_checkpoint(
                            grad_path, network, ppo, global_step,
                            episode_count, best_reward, args, wandb_run_id,
                        )
                        print(f"\n*** STAGE 1 GRADUATED! ***")
                        print(f"  Checkpoint saved: {grad_path}")
                        print(f"  Use --stage 2 --stage1-checkpoint {grad_path}")

                    # Reset
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0
                    obs, _ = env.reset(seed=args.seed + episode_count)

                    # Reset LSTM hidden states for this env
                    with torch.no_grad():
                        new_hidden["actor"][0][:, i, :] = 0.0
                        new_hidden["actor"][1][:, i, :] = 0.0
                        new_hidden["critic"][0][:, i, :] = 0.0
                        new_hidden["critic"][1][:, i, :] = 0.0

                    dones[i] = terminated
                    truncs[i] = truncated

                new_obs_list.append(obs)

            # Update buffer rewards and dones (overwrite the zeros)
            buffer.rewards[buffer.pos - 1] = step_rewards
            buffer.dones[buffer.pos - 1] = dones
            buffer.truncs[buffer.pos - 1] = truncs

            obs_list = new_obs_list
            hidden_states = new_hidden
            global_step += args.num_envs

        # ---- Compute advantages ----
        with torch.no_grad():
            voxel_tensor, flat_tensor = obs_to_tensors(obs_list, device)
            last_values, _ = network.get_value(voxel_tensor, flat_tensor, hidden_states)
            last_values = last_values.cpu().numpy()

        last_dones = np.zeros(args.num_envs, dtype=np.bool_)
        buffer.compute_advantages(last_values, last_dones)

        # ---- Entropy Annealing ----
        if args.entropy_coeff is None:  # annealing enabled
            progress = min(global_step / args.total_timesteps, 1.0)
            ent_coeff = args.entropy_start + (args.entropy_end - args.entropy_start) * progress
            ppo.entropy_coeff = ent_coeff

        # ---- PPO Update ----
        network.train()
        metrics = ppo.update(buffer)

        # ---- Logging ----
        elapsed = time.time() - start_time
        fps = (global_step - last_log_step) / max(elapsed, 1)

        if len(episode_rewards) > 0:
            log_data = {
                "train/reward_mean": np.mean(episode_rewards),
                "train/reward_std": np.std(episode_rewards),
                "train/episode_length_mean": np.mean(episode_lengths),
                "train/completion_mean": np.mean(episode_completions),
                "train/success_rate": np.mean(episode_successes),
                "train/stuck_rate": np.mean(episode_stuck),
                "train/timeout_rate": np.mean(episode_timeouts),
                "train/policy_loss": metrics["policy_loss"],
                "train/value_loss": metrics["value_loss"],
                "train/entropy": metrics["entropy"],
                "train/entropy_coeff": ppo.entropy_coeff,
                "train/clip_fraction": metrics["clip_fraction"],
                "train/approx_kl": metrics["approx_kl"],
                "train/episodes": episode_count,
                "train/global_step": global_step,
            }

            if not args.no_wandb and wandb_run:
                import wandb
                wandb.log(log_data, step=global_step)

            # Console output
            print(
                f"Step {global_step:>10,} | "
                f"Reward {np.mean(episode_rewards):>7.1f} | "
                f"Completion {np.mean(episode_completions):>5.1%} | "
                f"Success {np.mean(episode_successes):>5.1%} | "
                f"Entropy {metrics['entropy']:.3f} (coeff={ppo.entropy_coeff:.4f}) | "
                f"Episodes {episode_count}"
            )

        # ---- Save checkpoint ----
        if global_step % args.save_interval < args.num_envs * steps_per_env:
            # Periodic save
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

            # Save best
            if len(episode_rewards) > 0:
                current_reward = np.mean(episode_rewards)
                if current_reward > best_reward:
                    best_reward = current_reward
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, "best.pt"),
                        network, ppo, global_step, episode_count, best_reward,
                        args, wandb_run_id,
                    )

        # ---- Periodic evaluation ----
        if global_step % args.eval_interval < args.num_envs * steps_per_env:
            eval_metrics = evaluate(args, network, device, args.eval_episodes)
            if not args.no_wandb and wandb_run:
                import wandb
                eval_log = {f"eval/{k}": v for k, v in eval_metrics.items()}
                wandb.log(eval_log, step=global_step)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\nTraining interrupted at step {global_step:,}")

    # Save final checkpoint
    save_checkpoint(
        os.path.join(args.checkpoint_dir, "latest.pt"),
        network, ppo, global_step, episode_count, best_reward, args, wandb_run_id,
    )
    print(f"Training {'interrupted' if interrupted else 'complete'}! "
          f"{global_step:,} steps, {episode_count} episodes")

    if wandb_run:
        wandb_run.finish()


# ==================================================================
# Evaluation
# ==================================================================

def evaluate(args, network=None, device=None, num_episodes=None):
    """
    Evaluate the agent over multiple episodes.

    Can be called standalone (--mode eval) or from training loop.
    """
    if device is None:
        device = get_device(args)

    if network is None:
        network = ActorCriticNetwork(lstm_num_layers=args.lstm_layers).to(device)
        if args.resume:
            load_checkpoint(args.resume, network)
        else:
            print("Warning: evaluating with random weights (no --resume)")

    if num_episodes is None:
        num_episodes = args.eval_episodes

    network.eval()
    env = make_env(args, seed=args.seed + 10000)

    rewards = []
    completions = []
    lengths = []
    successes = []
    floor_completions = []
    wall_completions = []
    ceiling_completions = []
    door_correct = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=args.seed + 10000 + ep)
        hidden = network.get_initial_hidden_state(1, device)
        ep_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                voxel = torch.tensor(
                    np.transpose(obs["voxel_grids"], (3, 0, 1, 2))[np.newaxis],
                    dtype=torch.float32, device=device,
                )
                flat = torch.tensor(
                    obs["flat_features"][np.newaxis],
                    dtype=torch.float32, device=device,
                )
                mask = torch.tensor(
                    env.action_masks()[np.newaxis],
                    dtype=torch.bool, device=device,
                )

                result = network(
                    voxel, flat, hidden, action_masks=mask,
                    deterministic=args.deterministic,
                )
                action = result["actions"].cpu().numpy()[0]
                hidden = result["hidden_states"]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        completion = info.get("completion", {})
        rewards.append(ep_reward)
        completions.append(completion.get("total_completion", 0.0))
        lengths.append(info.get("step", 0))
        successes.append(1.0 if info.get("termination_reason") == "success" else 0.0)
        floor_completions.append(completion.get("floor_ratio", 0.0))
        wall_completions.append(completion.get("wall_ratio", 0.0))
        ceiling_completions.append(completion.get("ceiling_ratio", 0.0))
        door_correct.append(completion.get("door_ratio", 0.0))

    metrics = {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "completion_mean": np.mean(completions),
        "completion_std": np.std(completions),
        "success_rate": np.mean(successes),
        "episode_length_mean": np.mean(lengths),
        "floor_completion": np.mean(floor_completions),
        "wall_completion": np.mean(wall_completions),
        "ceiling_completion": np.mean(ceiling_completions),
        "door_correct_ratio": np.mean(door_correct),
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Reward:     {metrics['reward_mean']:.1f} \u00B1 {metrics['reward_std']:.1f}")
    print(f"  Completion: {metrics['completion_mean']:.1%} \u00B1 {metrics['completion_std']:.1%}")
    print(f"  Success:    {metrics['success_rate']:.0%} ({int(metrics['success_rate']*num_episodes)}/{num_episodes})")
    print(f"  Avg Length: {metrics['episode_length_mean']:.0f} steps")
    print(f"")
    print(f"  Floor:   {metrics['floor_completion']:.0%}")
    print(f"  Walls:   {metrics['wall_completion']:.0%}")
    print(f"  Ceiling: {metrics['ceiling_completion']:.0%}")
    print(f"  Door:    {metrics['door_correct_ratio']:.0%}")
    print(f"{'='*50}\n")

    return metrics


# ==================================================================
# Preview & Record
# ==================================================================

def preview(args):
    """Watch the agent build in real-time with cv2 display."""
    import cv2

    device = get_device(args)
    network = ActorCriticNetwork(lstm_num_layers=args.lstm_layers).to(device)

    if args.resume:
        load_checkpoint(args.resume, network)
    else:
        print("Warning: previewing with random weights (no --resume)")

    network.eval()
    env = make_env(args, seed=args.seed)
    obs, info = env.reset()
    hidden = network.get_initial_hidden_state(1, device)

    print("Preview mode — press 'q' in the window or Ctrl+C to stop")
    print("Watching agent build...")

    try:
        step = 0
        ep_reward = 0.0
        while True:
            with torch.no_grad():
                voxel = torch.tensor(
                    np.transpose(obs["voxel_grids"], (3, 0, 1, 2))[np.newaxis],
                    dtype=torch.float32, device=device,
                )
                flat = torch.tensor(
                    obs["flat_features"][np.newaxis],
                    dtype=torch.float32, device=device,
                )
                mask = torch.tensor(
                    env.action_masks()[np.newaxis],
                    dtype=torch.bool, device=device,
                )
                result = network(
                    voxel, flat, hidden, action_masks=mask,
                    deterministic=True,
                )
                action = result["actions"].cpu().numpy()[0]
                hidden = result["hidden_states"]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            # Display frame via cv2
            frame = env.get_frame()
            if frame is not None:
                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                display = cv2.resize(
                    display, (640, 480), interpolation=cv2.INTER_NEAREST
                )
                # Overlay HUD text
                completion = info.get("completion", {})
                comp_pct = completion.get("total_completion", 0)
                text = (
                    f"Step {step} | Reward {ep_reward:.1f} | "
                    f"Completion {comp_pct:.0%}"
                )
                cv2.putText(
                    display, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )
                cv2.imshow("CraftGround Preview", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  Preview stopped by user (q pressed)")
                    break

            completion = info.get("completion", {})
            if step % 50 == 0:
                print(
                    f"  Step {step:>5} | "
                    f"Reward {ep_reward:>7.1f} | "
                    f"Completion {completion.get('total_completion', 0):.1%}"
                )

            if terminated or truncated:
                reason = info.get("termination_reason", "unknown")
                print(f"\n  Episode ended: {reason}")
                print(f"  Total reward: {ep_reward:.1f}")
                print(f"  Final completion: {completion.get('total_completion', 0):.1%}")
                break

    except KeyboardInterrupt:
        print("\n  Preview stopped by user")
    finally:
        cv2.destroyAllWindows()


def record(args):
    """Record episode videos and JSON data to disk."""
    import cv2

    device = get_device(args)
    network = ActorCriticNetwork(lstm_num_layers=args.lstm_layers).to(device)

    if args.resume:
        load_checkpoint(args.resume, network)
    else:
        print("Warning: recording with random weights (no --resume)")

    network.eval()
    os.makedirs(args.record_path, exist_ok=True)

    print(f"Recording {args.record_episodes} episodes to {args.record_path}/")

    for ep in range(args.record_episodes):
        env = make_env(args, seed=args.seed + ep)
        obs, info = env.reset()
        hidden = network.get_initial_hidden_state(1, device)

        episode_data = {
            "actions": [],
            "rewards": [],
            "completions": [],
            "positions": [],
        }

        # Set up video writer
        video_writer = None
        video_path = os.path.join(args.record_path, f"episode_{ep:03d}.mp4")

        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            with torch.no_grad():
                voxel = torch.tensor(
                    np.transpose(obs["voxel_grids"], (3, 0, 1, 2))[np.newaxis],
                    dtype=torch.float32, device=device,
                )
                flat = torch.tensor(
                    obs["flat_features"][np.newaxis],
                    dtype=torch.float32, device=device,
                )
                mask = torch.tensor(
                    env.action_masks()[np.newaxis],
                    dtype=torch.bool, device=device,
                )
                result = network(
                    voxel, flat, hidden, action_masks=mask,
                    deterministic=True,
                )
                action = result["actions"].cpu().numpy()[0]
                hidden = result["hidden_states"]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

            # Write video frame
            frame = env.get_frame()
            if frame is not None:
                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                display = cv2.resize(
                    display, (640, 480), interpolation=cv2.INTER_NEAREST
                )
                # Overlay text
                completion = info.get("completion", {})
                comp_pct = completion.get("total_completion", 0)
                text = (
                    f"Step {step} | Reward {ep_reward:.1f} | "
                    f"Completion {comp_pct:.0%}"
                )
                cv2.putText(
                    display, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

                if video_writer is None:
                    h, w = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, 20.0, (w, h)
                    )
                video_writer.write(display)

            episode_data["actions"].append(action.tolist())
            episode_data["rewards"].append(float(reward))
            episode_data["completions"].append(
                info.get("completion", {}).get("total_completion", 0.0)
            )
            episode_data["positions"].append([
                float(env.agent_x), float(env.agent_y), float(env.agent_z)
            ])

        if video_writer is not None:
            video_writer.release()

        # Save episode JSON data
        completion = info.get("completion", {})
        episode_data["summary"] = {
            "total_reward": float(ep_reward),
            "total_steps": step,
            "total_completion": completion.get("total_completion", 0.0),
            "termination_reason": info.get("termination_reason", "unknown"),
            "success": info.get("termination_reason") == "success",
        }

        json_path = os.path.join(args.record_path, f"episode_{ep:03d}.json")
        with open(json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        print(
            f"  Episode {ep+1}/{args.record_episodes}: "
            f"reward={ep_reward:.1f}, "
            f"completion={completion.get('total_completion', 0):.1%}, "
            f"reason={info.get('termination_reason', 'unknown')} "
            f"→ {video_path}"
        )

    print(f"\nRecording complete! Files saved to {args.record_path}/")


# ==================================================================
# Utilities
# ==================================================================

def get_device(args):
    """Determine the torch device."""
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


# ==================================================================
# Entry Point
# ==================================================================

def main():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'='*60}")
    print(f"  Cuboid House Construction Agent")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    if args.resume:
        print(f"  Resume: {args.resume}")
    print(f"{'='*60}\n")

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "preview":
        preview(args)
    elif args.mode == "record":
        record(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
