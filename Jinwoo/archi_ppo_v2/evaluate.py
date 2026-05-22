"""
Evaluation & Visualization for the House Builder Agent.

Provides:
  1. Run trained agent and record Minecraft gameplay as MP4 video
  2. Overlay real-time stats (completion %, reward, blocks, etc.)
  3. Side-by-side view: agent's POV + blueprint progress grid
  4. Episode metrics logging to CSV / JSON
  5. Live preview window (optional, requires display)
  6. Training callback for periodic evaluation recordings

Usage:
    # Evaluate a trained checkpoint and record video:
    python evaluate.py --checkpoint checkpoints/best_model.pt \
                       --nbt blueprints/house.nbt \
                       --episodes 5 --record

    # Live preview (needs display):
    python evaluate.py --checkpoint checkpoints/best_model.pt \
                       --nbt blueprints/house.nbt \
                       --live

    # Quick test with synthetic blueprint:
    python evaluate.py --checkpoint checkpoints/best_model.pt \
                       --structure wall_2high --episodes 3 --record
"""

import argparse
import os
import json
import time
import csv
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path

import torch

# ──────────────────────────────────────────────────────────────────────
# Optional imports — graceful fallback if not installed
# ──────────────────────────────────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from nbt_parser import (
    Blueprint, parse_nbt_file, create_simple_blueprint, get_blueprint_stats,
)
from building_env import (
    HouseBuildingWrapper, BuildingConfig, BuildAction,
    make_building_env, NUM_BUILD_ACTIONS,
)
from train import BuilderNetwork


# ──────────────────────────────────────────────────────────────────────
# Stats Overlay Renderer
# ──────────────────────────────────────────────────────────────────────
class StatsOverlay:
    """
    Renders a real-time stats HUD onto video frames.
    Uses OpenCV for text drawing (fast), falls back to PIL if needed.
    """

    # Color palette (BGR for OpenCV)
    COLOR_BG = (30, 30, 30)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GREEN = (0, 200, 0)
    COLOR_YELLOW = (0, 200, 255)
    COLOR_RED = (0, 0, 220)
    COLOR_CYAN = (220, 180, 0)
    COLOR_BAR_BG = (60, 60, 60)

    def __init__(self, width: int = 320, height: int = 480):
        self.width = width
        self.height = height

    def render(self, info: dict, step_reward: float, action: int) -> np.ndarray:
        """
        Render a stats panel as a numpy image (H, W, 3) uint8 BGR.

        Args:
            info: The info dict from env.step().
            step_reward: Reward for the current step.
            action: Action taken this step.

        Returns:
            Stats panel image.
        """
        panel = np.full((self.height, self.width, 3), self.COLOR_BG, dtype=np.uint8)

        if not HAS_CV2:
            return panel

        y = 25
        line_h = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        def put(text, color=self.COLOR_TEXT, y_pos=None):
            nonlocal y
            if y_pos is not None:
                y = y_pos
            cv2.putText(panel, text, (10, y), font, font_scale, color, thickness)
            y += line_h

        # Title
        cv2.putText(panel, "BUILD AGENT EVAL", (10, y),
                    font, 0.7, self.COLOR_CYAN, 2)
        y += 35

        # Completion
        pct = info.get("completion_pct", 0.0)
        placed = info.get("blocks_placed", 0)
        total = info.get("total_blocks", 0)
        put(f"Completion: {pct:.1%}  ({placed}/{total})")

        # Progress bar
        bar_x, bar_w, bar_h = 10, self.width - 20, 16
        cv2.rectangle(panel, (bar_x, y - 5), (bar_x + bar_w, y - 5 + bar_h),
                      self.COLOR_BAR_BG, -1)
        fill_w = int(bar_w * min(pct, 1.0))
        bar_color = self.COLOR_GREEN if pct > 0.5 else (
            self.COLOR_YELLOW if pct > 0.2 else self.COLOR_RED
        )
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x, y - 5),
                          (bar_x + fill_w, y - 5 + bar_h), bar_color, -1)
        y += bar_h + 12

        # Reward
        total_reward = info.get("total_reward", 0.0)
        reward_color = self.COLOR_GREEN if step_reward >= 0 else self.COLOR_RED
        put(f"Step Reward:  {step_reward:+.3f}", reward_color)
        put(f"Total Reward: {total_reward:+.2f}")

        # Step count
        step_count = info.get("step_count", 0)
        put(f"Step: {step_count}")
        y += 5

        # Scaffold info
        scaffold = info.get("scaffold_blocks", 0)
        put(f"Scaffold blocks: {scaffold}",
            self.COLOR_YELLOW if scaffold > 0 else self.COLOR_TEXT)

        # Action taken
        action_name = BuildAction(action).name if 0 <= action < len(BuildAction) else "?"
        put(f"Action: {action_name}", self.COLOR_CYAN)
        y += 5

        # Agent position (from observation)
        agent_pos = info.get("agent_pos", None)
        if agent_pos is not None and len(agent_pos) >= 3:
            put(f"Pos: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f}, {agent_pos[2]:.1f})")

        # Structure complete flag
        if info.get("structure_complete", False):
            y += 10
            cv2.putText(panel, "STRUCTURE COMPLETE!", (10, y),
                        font, 0.7, self.COLOR_GREEN, 2)
            y += line_h

        return panel


# ──────────────────────────────────────────────────────────────────────
# Blueprint Progress Visualizer (2D top-down + side view)
# ──────────────────────────────────────────────────────────────────────
class BlueprintVisualizer:
    """
    Renders a 2D visualization of blueprint vs. current build progress.
    Shows a top-down view (XZ plane) and a side view (XY plane) of the
    blueprint, coloring blocks by status: placed / missing / scaffold.
    """

    COLOR_PLACED = (0, 180, 0)      # green — correct
    COLOR_MISSING = (80, 80, 80)    # dark gray — not yet placed
    COLOR_SCAFFOLD = (0, 180, 255)  # orange — scaffold
    COLOR_BG = (20, 20, 20)
    COLOR_AGENT = (255, 0, 255)     # magenta

    def __init__(self, blueprint: Blueprint, cell_size: int = 8):
        self.blueprint = blueprint
        self.cell_size = cell_size

    def render(
        self,
        placed_positions: set,
        scaffold_positions: set,
        agent_pos: Optional[Tuple[float, float, float]] = None,
        origin: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Render top-down and side views of the build progress.

        Returns:
            Combined image (H, W, 3) uint8 BGR.
        """
        if not HAS_CV2:
            return np.zeros((200, 200, 3), dtype=np.uint8)

        cs = self.cell_size
        bp = self.blueprint
        ox, oy, oz = origin

        # Top-down view (XZ, showing the highest block at each column)
        top_w = bp.size_x * cs
        top_h = bp.size_z * cs
        top_img = np.full((max(top_h, 1), max(top_w, 1), 3), self.COLOR_BG, dtype=np.uint8)

        for b in bp.blocks:
            world_pos = (ox + b.x, oy + b.y, oz + b.z)
            if world_pos in placed_positions:
                color = self.COLOR_PLACED
            else:
                color = self.COLOR_MISSING
            x1, z1 = b.x * cs, b.z * cs
            x2, z2 = x1 + cs - 1, z1 + cs - 1
            if 0 <= x1 < top_w and 0 <= z1 < top_h:
                cv2.rectangle(top_img, (x1, z1), (x2, z2), color, -1)
                cv2.rectangle(top_img, (x1, z1), (x2, z2), (50, 50, 50), 1)

        # Draw scaffold blocks
        for pos in scaffold_positions:
            bx, bz = pos[0] - ox, pos[2] - oz
            x1, z1 = bx * cs, bz * cs
            x2, z2 = x1 + cs - 1, z1 + cs - 1
            if 0 <= x1 < top_w and 0 <= z1 < top_h:
                cv2.rectangle(top_img, (x1, z1), (x2, z2), self.COLOR_SCAFFOLD, -1)

        # Agent marker on top-down view
        if agent_pos is not None:
            ax = int((agent_pos[0] - ox) * cs + cs // 2)
            az = int((agent_pos[2] - oz) * cs + cs // 2)
            if 0 <= ax < top_w and 0 <= az < top_h:
                cv2.circle(top_img, (ax, az), cs // 2 + 1, self.COLOR_AGENT, -1)

        # Side view (XY)
        side_w = bp.size_x * cs
        side_h = bp.size_y * cs
        side_img = np.full((max(side_h, 1), max(side_w, 1), 3), self.COLOR_BG, dtype=np.uint8)

        for b in bp.blocks:
            world_pos = (ox + b.x, oy + b.y, oz + b.z)
            if world_pos in placed_positions:
                color = self.COLOR_PLACED
            else:
                color = self.COLOR_MISSING
            x1 = b.x * cs
            # Flip Y so ground is at the bottom
            y1 = (bp.size_y - 1 - b.y) * cs
            x2, y2 = x1 + cs - 1, y1 + cs - 1
            if 0 <= x1 < side_w and 0 <= y1 < side_h:
                cv2.rectangle(side_img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(side_img, (x1, y1), (x2, y2), (50, 50, 50), 1)

        # Agent marker on side view
        if agent_pos is not None:
            ax = int((agent_pos[0] - ox) * cs + cs // 2)
            ay = int((bp.size_y - 1 - (agent_pos[1] - oy)) * cs + cs // 2)
            if 0 <= ax < side_w and 0 <= ay < side_h:
                cv2.circle(side_img, (ax, ay), cs // 2 + 1, self.COLOR_AGENT, -1)

        # Add labels
        label_h = 20
        top_labeled = np.full((top_h + label_h, top_w, 3), self.COLOR_BG, dtype=np.uint8)
        top_labeled[label_h:, :] = top_img
        cv2.putText(top_labeled, "Top (XZ)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        side_labeled = np.full((side_h + label_h, side_w, 3), self.COLOR_BG, dtype=np.uint8)
        side_labeled[label_h:, :] = side_img
        cv2.putText(side_labeled, "Side (XY)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Stack vertically with padding
        max_w = max(top_labeled.shape[1], side_labeled.shape[1], 1)

        def pad_width(img, target_w):
            if img.shape[1] < target_w:
                pad = np.full((img.shape[0], target_w - img.shape[1], 3),
                              self.COLOR_BG, dtype=np.uint8)
                return np.hstack([img, pad])
            return img

        top_padded = pad_width(top_labeled, max_w)
        side_padded = pad_width(side_labeled, max_w)

        spacer = np.full((8, max_w, 3), self.COLOR_BG, dtype=np.uint8)
        combined = np.vstack([top_padded, spacer, side_padded])

        return combined


# ──────────────────────────────────────────────────────────────────────
# Video Recorder
# ──────────────────────────────────────────────────────────────────────
class VideoRecorder:
    """
    Records agent gameplay as MP4 video with side-by-side panels:
      [Minecraft POV] [Stats Panel] [Blueprint Progress]
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 20,
        minecraft_size: Tuple[int, int] = (256, 256),
    ):
        self.output_path = output_path
        self.fps = fps
        self.mc_size = minecraft_size
        self.writer = None
        self.frame_count = 0

    def start(self, frame_width: int, frame_height: int):
        """Initialize the video writer."""
        if not HAS_CV2:
            print("WARNING: OpenCV not available. Video will not be recorded.")
            return

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (frame_width, frame_height)
        )
        self.frame_count = 0

    def write_frame(self, frame: np.ndarray):
        """Write a BGR frame to the video."""
        if self.writer is not None:
            self.writer.write(frame)
            self.frame_count += 1

    def stop(self):
        """Finalize and close the video file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Video saved: {self.output_path} ({self.frame_count} frames)")


# ──────────────────────────────────────────────────────────────────────
# Composite Frame Builder
# ──────────────────────────────────────────────────────────────────────
def build_composite_frame(
    minecraft_frame: np.ndarray,
    stats_panel: np.ndarray,
    blueprint_panel: np.ndarray,
    target_height: int = 480,
) -> np.ndarray:
    """
    Combine the three panels into a single composite frame.

    Layout: [Minecraft POV | Stats | Blueprint Progress]

    All panels are resized to the same height before combining.
    """
    if not HAS_CV2:
        return minecraft_frame

    def resize_to_height(img, h):
        if img.shape[0] == 0 or img.shape[1] == 0:
            return np.zeros((h, 100, 3), dtype=np.uint8)
        scale = h / img.shape[0]
        new_w = max(int(img.shape[1] * scale), 1)
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_NEAREST)

    mc_resized = resize_to_height(minecraft_frame, target_height)
    stats_resized = resize_to_height(stats_panel, target_height)
    bp_resized = resize_to_height(blueprint_panel, target_height)

    # Vertical separator
    sep = np.full((target_height, 3, 3), (80, 80, 80), dtype=np.uint8)

    composite = np.hstack([mc_resized, sep, stats_resized, sep, bp_resized])
    return composite


# ──────────────────────────────────────────────────────────────────────
# Episode Metrics Logger
# ──────────────────────────────────────────────────────────────────────
class MetricsLogger:
    """Logs per-episode and per-step metrics to CSV and JSON."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.episode_log_path = os.path.join(output_dir, "episode_metrics.csv")
        self.step_log_path = os.path.join(output_dir, "step_metrics.csv")
        self.summary_path = os.path.join(output_dir, "eval_summary.json")

        self.episodes: List[dict] = []
        self.current_steps: List[dict] = []

        # Initialize CSV files with headers
        with open(self.episode_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "stage", "completion_pct", "blocks_placed", "total_blocks",
                "scaffold_used", "total_reward", "steps", "structure_complete",
                "duration_sec",
            ])

        with open(self.step_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "step", "action", "reward",
                "completion_pct", "agent_x", "agent_y", "agent_z",
            ])

    def log_step(
        self,
        episode: int,
        step: int,
        action: int,
        reward: float,
        info: dict,
        obs: dict,
    ):
        """Log a single step."""
        agent_pos = obs.get("agent_pos", [0, 0, 0, 0, 0, 0])
        row = {
            "episode": episode,
            "step": step,
            "action": BuildAction(action).name,
            "reward": reward,
            "completion_pct": info.get("completion_pct", 0.0),
            "agent_x": float(agent_pos[0]),
            "agent_y": float(agent_pos[1]),
            "agent_z": float(agent_pos[2]),
        }
        self.current_steps.append(row)

    def log_episode(
        self,
        episode: int,
        info: dict,
        duration: float,
        stage: int = 1,
    ):
        """Log episode summary."""
        row = {
            "episode": episode,
            "stage": stage,
            "completion_pct": info.get("completion_pct", 0.0),
            "blocks_placed": info.get("blocks_placed", 0),
            "total_blocks": info.get("total_blocks", 0),
            "scaffold_used": info.get("scaffold_blocks", 0),
            "total_reward": info.get("total_reward", 0.0),
            "steps": info.get("step_count", 0),
            "structure_complete": info.get("structure_complete", False),
            "duration_sec": duration,
        }
        self.episodes.append(row)

        # Write to CSV
        with open(self.episode_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row.values())

        # Write step data
        with open(self.step_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for step_row in self.current_steps:
                writer.writerow(step_row.values())
        self.current_steps.clear()

    def save_summary(self):
        """Save overall evaluation summary as JSON."""
        if not self.episodes:
            return

        completions = [e["completion_pct"] for e in self.episodes]
        rewards = [e["total_reward"] for e in self.episodes]
        steps = [e["steps"] for e in self.episodes]
        successes = [e["structure_complete"] for e in self.episodes]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_episodes": len(self.episodes),
            "avg_completion": float(np.mean(completions)),
            "max_completion": float(np.max(completions)),
            "min_completion": float(np.min(completions)),
            "std_completion": float(np.std(completions)),
            "avg_reward": float(np.mean(rewards)),
            "avg_steps": float(np.mean(steps)),
            "success_rate": float(np.mean(successes)),
            "episodes": self.episodes,
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nEvaluation Summary:")
        print(f"  Episodes:       {summary['num_episodes']}")
        print(f"  Avg Completion: {summary['avg_completion']:.1%}")
        print(f"  Max Completion: {summary['max_completion']:.1%}")
        print(f"  Avg Reward:     {summary['avg_reward']:.2f}")
        print(f"  Avg Steps:      {summary['avg_steps']:.0f}")
        print(f"  Success Rate:   {summary['success_rate']:.1%}")
        print(f"  Saved to: {self.summary_path}")


# ──────────────────────────────────────────────────────────────────────
# Main Evaluator
# ──────────────────────────────────────────────────────────────────────
class AgentEvaluator:
    """
    Runs evaluation episodes, records video, and logs metrics.
    """

    def __init__(
        self,
        checkpoint_path: str,
        blueprint: Blueprint,
        output_dir: str = "eval_output",
        port: int = 8024,
        image_size: int = 256,
        record_video: bool = True,
        live_preview: bool = False,
        fps: int = 20,
        device: str = "cpu",
        seed: Optional[int] = 12345,
        curriculum_stage: int = 1,
    ):
        self.checkpoint_path = checkpoint_path
        self.blueprint = blueprint
        self.output_dir = output_dir
        self.port = port
        self.image_size = image_size
        self.record_video = record_video
        self.live_preview = live_preview
        self.fps = fps
        self.device = device
        self.seed = seed
        self.curriculum_stage = curriculum_stage

        os.makedirs(output_dir, exist_ok=True)

        # Load network
        self.network = BuilderNetwork(
            obs_grid_size=11,
            num_actions=NUM_BUILD_ACTIONS,
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.network.load_state_dict(checkpoint["network_state"])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print("  Running with untrained (random) network")
        self.network.to(device)
        self.network.eval()

        # Renderers
        self.stats_overlay = StatsOverlay(width=320, height=480)
        self.bp_visualizer = BlueprintVisualizer(blueprint, cell_size=10)
        self.metrics_logger = MetricsLogger(output_dir)

    def evaluate(self, num_episodes: int = 5, max_steps: int = 2000):
        """
        Run evaluation episodes.

        Args:
            num_episodes: Number of episodes to run.
            max_steps: Max steps per episode.
        """
        stage = self.curriculum_stage
        print(f"[Eval] Stage {stage} — "
              f"{'place any block at correct position' if stage == 0 else 'full building'}")

        # Create environment with larger image for better video quality
        env = make_building_env(
            blueprint=self.blueprint,
            port=self.port,
            build_origin=(0, -59, 0),
            image_size=self.image_size,
            max_timesteps=max_steps,
            seed=self.seed,
            curriculum_stage=stage,
        )
        # Override to capture visual observations for recording
        env.config.use_visual_obs = True

        build_origin = env.config.build_origin

        for ep in range(1, num_episodes + 1):
            print(f"\n--- Episode {ep}/{num_episodes} (Stage {stage}) ---")
            ep_start = time.time()

            # Setup video recorder for this episode
            video_recorder = None
            if self.record_video:
                video_path = os.path.join(
                    self.output_dir, f"episode_{ep:03d}.mp4"
                )
                video_recorder = VideoRecorder(
                    video_path, fps=self.fps,
                    minecraft_size=(self.image_size, self.image_size),
                )

            obs, info = env.reset()
            lstm_state = None
            total_reward = 0.0
            first_frame = True

            for step in range(1, max_steps + 1):
                # Get action from trained policy
                action, lstm_state = self._get_action(obs, lstm_state)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Augment info with observation data for logging/rendering
                info["agent_pos"] = obs["agent_pos"]

                # Log step
                self.metrics_logger.log_step(ep, step, action, reward, info, obs)

                # Render composite frame
                if self.record_video or self.live_preview:
                    frame = self._render_frame(
                        obs, info, reward, action, env, build_origin
                    )

                    # Initialize video writer on first frame
                    if first_frame and video_recorder is not None:
                        video_recorder.start(frame.shape[1], frame.shape[0])
                        first_frame = False

                    # Write to video
                    if video_recorder is not None:
                        video_recorder.write_frame(frame)

                    # Live preview
                    if self.live_preview and HAS_CV2:
                        cv2.imshow("House Builder Agent", frame)
                        key = cv2.waitKey(1000 // self.fps)
                        if key == ord("q"):
                            print("Preview closed by user.")
                            env.close()
                            return

                # Print progress periodically
                if step % 100 == 0:
                    pct = info.get("completion_pct", 0.0)
                    print(
                        f"  Step {step:5d} | "
                        f"Completion: {pct:.1%} | "
                        f"Reward: {total_reward:+.2f} | "
                        f"Scaffold: {info.get('scaffold_blocks', 0)}"
                    )

                if terminated or truncated:
                    break

            # Episode done
            duration = time.time() - ep_start
            self.metrics_logger.log_episode(ep, info, duration, stage=stage)

            if video_recorder is not None:
                video_recorder.stop()

            pct = info.get("completion_pct", 0.0)
            complete = info.get("structure_complete", False)
            print(
                f"  Done: {pct:.1%} complete | "
                f"Reward: {total_reward:+.2f} | "
                f"Steps: {step} | "
                f"{'SUCCESS' if complete else 'incomplete'} | "
                f"Time: {duration:.1f}s"
            )

            # ── Log episode to wandb (if active) ────────────────
            try:
                import wandb
                if wandb.run is not None:
                    ep_log = {
                        "eval/stage": stage,
                        "eval/completion_pct": pct,
                        "eval/total_reward": total_reward,
                        "eval/steps": step,
                        "eval/success": int(complete),
                        "eval/scaffold_blocks": info.get("scaffold_blocks", 0),
                        "eval/duration_sec": duration,
                    }

                    # Upload video if recorded
                    video_path = os.path.join(
                        self.output_dir, f"episode_{ep:03d}.mp4"
                    )
                    if os.path.exists(video_path):
                        ep_log["eval/video"] = wandb.Video(
                            video_path, fps=self.fps, format="mp4"
                        )

                    wandb.log(ep_log)
            except (ImportError, Exception):
                pass  # wandb not available or not active

        # Save summary
        self.metrics_logger.save_summary()

        # ── Log evaluation summary table to wandb ────────────────
        try:
            import wandb
            if wandb.run is not None and self.metrics_logger.episodes:
                episodes = self.metrics_logger.episodes
                completions = [e["completion_pct"] for e in episodes]
                rewards = [e["total_reward"] for e in episodes]
                successes = [e["structure_complete"] for e in episodes]

                wandb.run.summary.update({
                    "eval/stage": stage,
                    "eval/avg_completion": float(np.mean(completions)),
                    "eval/max_completion": float(np.max(completions)),
                    "eval/avg_reward": float(np.mean(rewards)),
                    "eval/success_rate": float(np.mean(successes)),
                    "eval/num_episodes": len(episodes),
                })
        except (ImportError, Exception):
            pass

        if self.live_preview and HAS_CV2:
            cv2.destroyAllWindows()

        env.close()

    def _get_action(
        self, obs: dict, lstm_state: Optional[tuple]
    ) -> Tuple[int, tuple]:
        """Get deterministic action from the trained policy."""
        with torch.no_grad():
            local_t = torch.tensor(
                obs["local_grid"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            target_t = torch.tensor(
                obs["target_grid"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            diff_t = torch.tensor(
                obs["diff_grid"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            ray_t = torch.tensor(
                obs["raycast_grid"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            pos_t = torch.tensor(
                obs["agent_pos"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            prog_t = torch.tensor(
                obs["progress"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            logits, value, new_lstm_state = self.network(
                local_t, target_t, diff_t, ray_t, pos_t, prog_t, lstm_state,
            )

            # Stochastic sampling (matches training behaviour)
            action = torch.distributions.Categorical(logits=logits).sample().item()

        return action, new_lstm_state

    def _render_frame(
        self,
        obs: dict,
        info: dict,
        reward: float,
        action: int,
        env: HouseBuildingWrapper,
        origin: Tuple[int, int, int],
    ) -> np.ndarray:
        """Build the composite frame with all panels."""

        # Minecraft POV (from CraftGround's image observation)
        mc_frame = obs.get("image", None)
        if mc_frame is None or not isinstance(mc_frame, np.ndarray):
            mc_frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Convert RGB to BGR for OpenCV
        if HAS_CV2 and mc_frame.shape[-1] == 3:
            mc_frame_bgr = cv2.cvtColor(mc_frame, cv2.COLOR_RGB2BGR)
        else:
            mc_frame_bgr = mc_frame

        # Stats overlay
        stats_panel = self.stats_overlay.render(info, reward, action)

        # Blueprint progress
        agent_pos_arr = obs.get("agent_pos", np.zeros(6))
        agent_pos_tuple = (
            float(agent_pos_arr[0]),
            float(agent_pos_arr[1]),
            float(agent_pos_arr[2]),
        )
        bp_panel = self.bp_visualizer.render(
            placed_positions=env._correctly_placed,
            scaffold_positions=env._scaffold_blocks,
            agent_pos=agent_pos_tuple,
            origin=origin,
        )

        # Composite
        composite = build_composite_frame(
            mc_frame_bgr, stats_panel, bp_panel, target_height=480,
        )

        return composite


# ──────────────────────────────────────────────────────────────────────
# Training Evaluation Callback
# ──────────────────────────────────────────────────────────────────────
class EvalDuringTrainingCallback:
    """
    Callback for periodic evaluation during training.

    Usage in train.py:
        eval_cb = EvalDuringTrainingCallback(
            blueprint=blueprint,
            checkpoint_dir="checkpoints",
            eval_interval=500,      # every 500 training iterations
            eval_episodes=3,
        )

        for iteration in range(total_iterations):
            # ... training step ...
            eval_cb.on_iteration_end(iteration, trainer)
    """

    def __init__(
        self,
        blueprint: Blueprint,
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "eval_output",
        eval_interval: int = 500,
        eval_episodes: int = 2,
        port: int = 8025,
        image_size: int = 128,
        device: str = "cpu",
        curriculum_stage: int = 1,
    ):
        self.blueprint = blueprint
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.port = port
        self.image_size = image_size
        self.device = device
        self.curriculum_stage = curriculum_stage  # can be updated when stage changes

    def on_iteration_end(self, iteration: int, trainer):
        """Called at the end of each training iteration."""
        if iteration % self.eval_interval != 0 or iteration == 0:
            return

        print(f"\n[Eval Callback] Running evaluation at iteration {iteration} "
              f"(Stage {self.curriculum_stage})...")

        # Save a temporary checkpoint
        tmp_path = os.path.join(self.checkpoint_dir, f"eval_temp_{iteration}.pt")
        trainer.save(tmp_path)

        # Run evaluation
        eval_dir = os.path.join(self.output_dir, f"iter_{iteration:06d}")
        evaluator = AgentEvaluator(
            checkpoint_path=tmp_path,
            blueprint=self.blueprint,
            output_dir=eval_dir,
            port=self.port,
            image_size=self.image_size,
            record_video=True,
            live_preview=False,
            device=self.device,
            curriculum_stage=self.curriculum_stage,
        )

        try:
            evaluator.evaluate(
                num_episodes=self.eval_episodes,
                max_steps=1000,
            )
        except Exception as e:
            print(f"[Eval Callback] Evaluation failed: {e}")

        # Clean up temp checkpoint
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize the house-building RL agent"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--nbt", type=str, default=None,
                       help="Path to .nbt blueprint file")
    parser.add_argument("--structure", type=str, default=None,
                       choices=["single_block", "row", "wall_2high",
                                "wall_3high", "small_room", "cube_3x3x3"],
                       help="Synthetic blueprint (if no .nbt)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=2000,
                       help="Max steps per episode")
    parser.add_argument("--port", type=int, default=8024,
                       help="CraftGround port (use different from training!)")
    parser.add_argument("--image-size", type=int, default=256,
                       help="Minecraft render resolution")
    parser.add_argument("--record", action="store_true",
                       help="Record gameplay as MP4 video")
    parser.add_argument("--live", action="store_true",
                       help="Show live preview window")
    parser.add_argument("--fps", type=int, default=20,
                       help="Video FPS")
    parser.add_argument("--output-dir", type=str, default="eval_output",
                       help="Output directory for videos and metrics")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Torch device")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--stage", type=int, default=1,
                       help="Curriculum stage: 0=place any block, 1+=full building")

    # Weights & Biases
    parser.add_argument("--wandb-project", type=str, default="craftground-builder",
                       help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B logging")

    args = parser.parse_args()

    # ── W&B init for standalone eval ─────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                job_type="eval",
                config={
                    "checkpoint": args.checkpoint,
                    "nbt": args.nbt,
                    "episodes": args.episodes,
                    "max_steps": args.max_steps,
                    "stage": args.stage,
                },
            )
        except (ImportError, Exception) as e:
            print(f"W&B not available: {e}")

    # Load blueprint
    if args.nbt and os.path.exists(args.nbt):
        blueprint = parse_nbt_file(args.nbt)
        print(f"Loaded .nbt blueprint: {args.nbt}")
    elif args.structure:
        blueprint = create_simple_blueprint(args.structure)
        print(f"Using synthetic blueprint: {args.structure}")
    else:
        blueprint = create_simple_blueprint("wall_3high")
        print("No blueprint specified, defaulting to wall_3high")

    stats = get_blueprint_stats(blueprint)
    print(f"Blueprint: {stats['total_blocks']} blocks, dims={stats['dimensions']}")

    # Check dependencies
    if (args.record or args.live) and not HAS_CV2:
        print("\nWARNING: OpenCV (cv2) not installed. Install with:")
        print("  pip install opencv-python")
        if args.live:
            print("Live preview requires OpenCV. Falling back to metrics-only.")
            args.live = False

    # Run evaluation
    evaluator = AgentEvaluator(
        checkpoint_path=args.checkpoint,
        blueprint=blueprint,
        output_dir=args.output_dir,
        port=args.port,
        image_size=args.image_size,
        record_video=args.record,
        live_preview=args.live,
        fps=args.fps,
        device=args.device,
        seed=args.seed,
        curriculum_stage=args.stage,
    )

    evaluator.evaluate(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    # Finish wandb run
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except (ImportError, Exception):
        pass


if __name__ == "__main__":
    main()
