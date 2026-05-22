from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


HYUNJIN_ROOT = Path(__file__).resolve().parent
CRAFTGROUND_SRC = HYUNJIN_ROOT / "CraftGround" / "src"
LEGACY_SRC = HYUNJIN_ROOT / "archive" / "archive"
FIXED_RENDER_DISTANCE = 10
FIXED_SIMULATION_DISTANCE = 10
FIXED_LIDAR_MAX_DISTANCE = 10.0
EVAL_EVERY_EPOCHS = 3
SAFE_MAX_ENVS = 2
DEFAULT_OPEN_WORLD_MODE = True
DEFAULT_NEAR_SPAWN_SCALE = 0.35

for candidate in (CRAFTGROUND_SRC, LEGACY_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from cleanup_minecraft_processes import cleanup_craftground_processes
from survive_and_hunt_environment import RewardConfig, StageConfig, SurviveAndHuntEnvironment


@dataclass(frozen=True)
class RoundSpec:
    name: str
    stage: StageConfig
    reward: RewardConfig
    total_timesteps: int


class RaycastVectorObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        vec_space = env.observation_space["vector"]
        self.observation_space = gym.spaces.Box(
            low=vec_space.low.astype(np.float32, copy=False),
            high=vec_space.high.astype(np.float32, copy=False),
            shape=vec_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["vector"].astype(np.float32, copy=False)


def reward_profile_survival_kite() -> RewardConfig:
    base = RewardConfig()
    return RewardConfig(
        survival_per_tick=0.007,
        combat_survival_per_tick=0.010,
        damage_scale=-0.16,
        danger_penalty=-0.03,
        retreat_reward=0.10,
        target_hit_reward=0.02,
        target_kill_reward=0.05,
        visible_shot_attempt_reward=0.0,
        aligned_shot_attempt_reward=0.0,
        power_shot_attempt_reward=0.0,
        shot_penalty=0.0,
        no_shot_penalty=0.0,
        no_hit_penalty=0.0,
        idle_step_penalty=-0.01,
        stuck_penalty=-0.06,
        too_close_pressure_penalty=-0.22,
        weak_hit_penalty=-0.05,
        draw_opportunity_reward=0.03,
        aligned_charge_reward=0.03,
        draw_centered_reward=0.03,
        target_centered_reward=0.03,
        quick_shot_hold_ticks=2,
        power_shot_hold_ticks=5,
        shot_window_reward=0.03,
        shot_window_miss_penalty=-0.05,
        followup_shot_reward=0.03,
        movement_reward_scale=0.10,
        engaged_kiting_reward=0.26,
        evasive_move_reward=0.24,
        stationary_bow_penalty=-0.16,
        stationary_shot_penalty=-0.26,
        stationary_close_combat_penalty=-0.24,
        stationary_combat_timeout_penalty=-0.70,
        mobile_shot_reward=0.08,
        static_blind_shot_penalty=-0.16,
        draw_alignment_threshold=base.draw_alignment_threshold,
        release_alignment_threshold=base.release_alignment_threshold,
    )


def reward_profile_hitfocus_raycast() -> RewardConfig:
    base = RewardConfig()
    return RewardConfig(
        survival_per_tick=0.006,
        combat_survival_per_tick=0.010,
        damage_scale=-0.14,
        danger_penalty=-0.030,
        retreat_reward=0.10,
        target_visible_reward=0.04,
        hostile_visible_reward=0.03,
        aim_improvement_scale=0.06,
        yaw_tracking_reward_scale=0.05,
        pitch_tracking_reward_scale=0.06,
        target_hit_reward=0.32,
        target_kill_reward=0.90,
        visible_shot_attempt_reward=0.0,
        aligned_shot_attempt_reward=0.03,
        power_shot_attempt_reward=0.0,
        shot_penalty=-0.05,
        no_shot_penalty=-0.02,
        no_hit_penalty=-0.26,
        idle_step_penalty=-0.016,
        stuck_penalty=-0.06,
        shot_window_reward=0.08,
        draw_opportunity_reward=0.10,
        aligned_charge_reward=0.10,
        draw_centered_reward=0.08,
        target_centered_reward=0.08,
        shot_choice_reward=0.04,
        blind_shot_penalty=-0.34,
        shot_window_miss_penalty=-0.12,
        followup_shot_reward=0.10,
        aim_track_reward=0.12,
        scan_acquire_reward=0.14,
        focus_target_visible_reward=0.10,
        same_target_hit_bonus=0.18,
        same_target_hit_bonus_power=1.25,
        same_target_hit_window_steps=110,
        completion_kill_multiplier_base=1.35,
        completion_kill_multiplier_scale=0.35,
        completion_window_steps=110,
        movement_reward_scale=0.16,
        engaged_kiting_reward=0.40,
        evasive_move_reward=0.38,
        too_close_pressure_penalty=-0.48,
        stationary_bow_penalty=-0.24,
        stationary_shot_penalty=-0.56,
        stationary_close_combat_penalty=-0.38,
        stationary_combat_timeout_penalty=-0.85,
        mobile_shot_reward=0.10,
        static_blind_shot_penalty=-0.30,
        draw_alignment_threshold=0.72,
        release_alignment_threshold=max(base.release_alignment_threshold, 0.73),
    )


def reward_profile_finishconvert_raycast() -> RewardConfig:
    base = RewardConfig()
    return RewardConfig(
        survival_per_tick=0.007,
        combat_survival_per_tick=0.011,
        damage_scale=-0.15,
        danger_penalty=-0.030,
        retreat_reward=0.11,
        target_visible_reward=0.04,
        hostile_visible_reward=0.03,
        aim_improvement_scale=0.06,
        yaw_tracking_reward_scale=0.05,
        pitch_tracking_reward_scale=0.06,
        target_hit_reward=0.30,
        target_kill_reward=2.00,
        visible_shot_attempt_reward=0.0,
        aligned_shot_attempt_reward=0.04,
        power_shot_attempt_reward=0.03,
        power_hit_reward=0.12,
        no_shot_penalty=-0.06,
        no_hit_penalty=-0.28,
        idle_step_penalty=-0.016,
        stuck_penalty=-0.07,
        shot_window_reward=0.05,
        draw_opportunity_reward=0.05,
        aligned_charge_reward=0.05,
        draw_centered_reward=0.05,
        target_centered_reward=0.05,
        shot_choice_reward=0.04,
        blind_shot_penalty=-0.24,
        shot_window_miss_penalty=-0.10,
        followup_shot_reward=0.12,
        aim_track_reward=0.12,
        scan_acquire_reward=0.16,
        same_target_hit_bonus=0.28,
        same_target_hit_bonus_power=1.35,
        same_target_hit_window_steps=140,
        completion_kill_multiplier_base=2.10,
        completion_kill_multiplier_scale=0.75,
        completion_window_steps=160,
        focus_target_visible_reward=0.18,
        focus_target_switch_penalty=-0.20,
        weak_hit_penalty=-0.14,
        weak_hit_damage_threshold=2.0,
        movement_reward_scale=0.14,
        engaged_kiting_reward=0.38,
        evasive_move_reward=0.34,
        stationary_bow_penalty=-0.20,
        stationary_shot_penalty=-0.56,
        stationary_close_combat_penalty=-0.34,
        stationary_combat_timeout_penalty=-0.80,
        mobile_shot_reward=0.12,
        static_blind_shot_penalty=-0.30,
        finish_window_power_reward=0.24,
        quick_shot_finish_penalty=-0.14,
        finish_target_health_threshold=8.0,
        finish_target_distance_threshold=8.5,
        draw_alignment_threshold=0.70,
        release_alignment_threshold=max(base.release_alignment_threshold, 0.74),
    )


def reward_profile_powerfinish_raycast() -> RewardConfig:
    base = reward_profile_finishconvert_raycast()
    return replace(
        base,
        survival_per_tick=0.006,
        combat_survival_per_tick=0.012,
        damage_scale=-0.16,
        too_close_pressure_penalty=-0.42,
        target_hit_reward=0.26,
        target_kill_reward=2.20,
        power_shot_attempt_reward=0.04,
        power_hit_reward=0.32,
        weak_hit_penalty=-0.18,
        quick_shot_finish_penalty=-0.22,
        finish_window_power_reward=0.36,
        finish_target_health_threshold=8.0,
        finish_target_distance_threshold=8.5,
        movement_reward_scale=0.15,
        engaged_kiting_reward=0.40,
        evasive_move_reward=0.36,
        stationary_bow_penalty=-0.22,
        stationary_shot_penalty=-0.60,
        stationary_close_combat_penalty=-0.36,
        stationary_combat_timeout_penalty=-0.86,
        blind_shot_penalty=-0.28,
        shot_window_miss_penalty=-0.12,
        mobile_shot_reward=0.14,
        static_blind_shot_penalty=-0.34,
    )


def build_rounds(timesteps_scale: float) -> list[RoundSpec]:
    s1 = StageConfig(
        name="s1_survival_bootstrap",
        hostile_count=1,
        arena_radius=30,
        max_steps=600,
        no_shot_timeout=280,
        no_hit_timeout=9999,
        hostile_types=("zombie",),
        animal_count=0,
        respawn_interval=9999,
        wall_height=4,
        focus_hostiles=True,
    )
    s2 = StageConfig(
        name="s2_duel_aim",
        hostile_count=1,
        arena_radius=27,
        max_steps=800,
        no_shot_timeout=240,
        no_hit_timeout=320,
        hostile_types=("zombie",),
        animal_count=0,
        respawn_interval=9999,
        wall_height=4,
        focus_hostiles=True,
    )
    s3 = StageConfig(
        name="s3_duel_finish",
        hostile_count=2,
        arena_radius=25,
        max_steps=900,
        no_shot_timeout=220,
        no_hit_timeout=300,
        hostile_types=("zombie",),
        animal_count=0,
        respawn_interval=160,
        wall_height=4,
        focus_hostiles=True,
    )
    s4 = StageConfig(
        name="s4_kite_pair",
        hostile_count=3,
        arena_radius=24,
        max_steps=1100,
        no_shot_timeout=220,
        no_hit_timeout=280,
        hostile_types=("zombie",),
        animal_count=0,
        respawn_interval=120,
        wall_height=4,
        focus_hostiles=True,
    )
    s5 = StageConfig(
        name="s5_clear_easy",
        hostile_count=4,
        arena_radius=25,
        max_steps=1400,
        no_shot_timeout=220,
        no_hit_timeout=280,
        hostile_types=("zombie",),
        animal_count=0,
        respawn_interval=90,
        wall_height=4,
        focus_hostiles=True,
    )
    s6 = StageConfig(
        name="s6_generalize",
        hostile_count=6,
        arena_radius=24,
        max_steps=1800,
        no_shot_timeout=230,
        no_hit_timeout=300,
        hostile_types=("zombie",),
        animal_count=0,
        respawn_interval=70,
        wall_height=4,
        focus_hostiles=True,
    )

    def ts(value: int) -> int:
        return max(512, int(round(value * timesteps_scale)))

    return [
        RoundSpec("survival_bootstrap", s1, reward_profile_survival_kite(), ts(8_000)),
        RoundSpec("hitfocus_recovery", s2, reward_profile_hitfocus_raycast(), ts(12_000)),
        RoundSpec("duel_finishconvert", s3, reward_profile_finishconvert_raycast(), ts(12_000)),
        RoundSpec("kite_pair_finish", s4, reward_profile_finishconvert_raycast(), ts(16_000)),
        RoundSpec("clear_easy_powerfinish", s5, reward_profile_powerfinish_raycast(), ts(20_000)),
        RoundSpec("generalize_combined", s6, reward_profile_powerfinish_raycast(), ts(24_000)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CraftGround raycast survival-hunt curriculum trainer")
    parser.add_argument("--run-name", default="", help="Optional fixed run name. Default: timestamp-based")
    parser.add_argument("--timesteps-scale", type=float, default=1.0)
    parser.add_argument("--start-round-index", type=int, default=1, help="1-based round index to start from")
    parser.add_argument("--end-round-index", type=int, default=0, help="1-based round index to stop at (0: all)")
    parser.add_argument("--steps-per-epoch", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--port-start", type=int, default=9300)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument("--render-distance", type=int, default=FIXED_RENDER_DISTANCE)
    parser.add_argument("--simulation-distance", type=int, default=FIXED_SIMULATION_DISTANCE)
    parser.add_argument("--lidar-horizontal-rays", type=int, default=32)
    parser.add_argument("--lidar-vertical-rays", type=int, default=3)
    parser.add_argument("--lidar-max-distance", type=float, default=FIXED_LIDAR_MAX_DISTANCE)
    parser.add_argument("--action-repeat-scale", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ent-coef", type=float, default=3e-4)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-recorded-frames", type=int, default=280)
    parser.add_argument("--gif-frame-duration-ms", type=int, default=120)
    parser.add_argument("--gif-end-hold-ms", type=int, default=900)
    parser.add_argument("--gif-min-duration-ms", type=int, default=6000)
    parser.add_argument("--save-root", default="artifacts/raycast_hunt")
    parser.add_argument("--tensorboard-root", default="runs/raycast_hunt")
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint to continue from")
    parser.add_argument(
        "--legacy-root",
        default="/mnt/e/RL_pjt/Hyunjin",
        help="Reference path for old experiments (read-only metadata use)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--use-vglrun", action="store_true")
    parser.add_argument("--play-checkpoint", default="", help="Checkpoint path to replay with cv2 instead of training")
    parser.add_argument("--play-round-index", type=int, default=0, help="1-based round index for replay stage (0: use start-round-index)")
    parser.add_argument("--play-max-steps-scale", type=float, default=1.5, help="Scale max_steps and timeout windows for replay")
    parser.add_argument("--play-fps", type=float, default=12.0)
    parser.add_argument("--play-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-open-arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-near-spawn-scale", type=float, default=0.4)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_env_factory(round_spec: RoundSpec, args: argparse.Namespace, rank: int):
    def _factory() -> gym.Env:
        env = SurviveAndHuntEnvironment(
            stage=round_spec.stage,
            image_width=args.image_width,
            image_height=args.image_height,
            env_path=args.env_path,
            port=args.port_start + rank,
            seed=args.seed + rank,
            use_vglrun=args.use_vglrun,
            reward_config=round_spec.reward,
            render_distance=args.render_distance,
            simulation_distance=args.simulation_distance,
            lidar_horizontal_rays=args.lidar_horizontal_rays,
            lidar_vertical_rays=args.lidar_vertical_rays,
            lidar_max_distance=args.lidar_max_distance,
            action_repeat_scale=args.action_repeat_scale,
        )
        wrapped = RaycastVectorObsWrapper(env)
        return Monitor(
            wrapped,
            info_keywords=(
                "survival_steps",
                "shots_fired",
                "target_hits",
                "target_kills",
                "target_damage_dealt",
                "damage_taken",
                "termination_reason",
            ),
        )

    return _factory


def build_vec_env(round_spec: RoundSpec, args: argparse.Namespace):
    factories = [make_env_factory(round_spec, args, rank) for rank in range(args.num_envs)]
    vec_cls = DummyVecEnv if args.num_envs == 1 else SubprocVecEnv
    vec_env = vec_cls(factories)
    return VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)


def annotate_frame(frame: np.ndarray, overlay_lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGBA")
    if not overlay_lines:
        return np.asarray(image.convert("RGB"))
    left = 4
    top = 4
    line_height = 11
    pad_x = 5
    pad_y = 4
    # Keep overlay compact on 16:9 low-res captures (e.g., 160x90).
    est_char_px = 6
    max_text_width = max(len(line) for line in overlay_lines) * est_char_px
    box_width = min(image.width - 8, max_text_width + pad_x * 2)
    box_height = pad_y * 2 + line_height * len(overlay_lines)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((left, top, left + box_width, top + box_height), fill=(0, 0, 0, 120))
    y = top + pad_y
    for line in overlay_lines:
        draw.text((left + pad_x, y), line, fill=(255, 255, 255, 235))
        y += line_height
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def save_gif(frames: list[np.ndarray], output_path: Path, frame_ms: int, end_hold_ms: int, min_ms: int) -> str | None:
    if not frames:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    durations = [frame_ms] * len(pil_frames)
    durations[-1] += max(end_hold_ms, 0)
    total_ms = sum(durations)
    if total_ms < max(min_ms, 0):
        scale = max(min_ms, 1) / max(total_ms, 1)
        durations = [max(40, int(round(d * scale))) for d in durations]
        durations[-1] += max(min_ms - sum(durations), 0)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
    )
    return str(output_path)


def evaluate_epoch(
    model: PPO,
    round_id: str,
    round_spec: RoundSpec,
    epoch: int,
    args: argparse.Namespace,
    round_dir: Path,
) -> dict[str, Any]:
    eval_port = args.port_start + 700 + epoch
    eval_seed = args.seed + epoch * 17
    env = SurviveAndHuntEnvironment(
        stage=round_spec.stage,
        image_width=args.image_width,
        image_height=args.image_height,
        env_path=args.env_path,
        port=eval_port,
        seed=eval_seed,
        use_vglrun=args.use_vglrun,
        reward_config=round_spec.reward,
        render_distance=args.render_distance,
        simulation_distance=args.simulation_distance,
        lidar_horizontal_rays=args.lidar_horizontal_rays,
        lidar_vertical_rays=args.lidar_vertical_rays,
        lidar_max_distance=args.lidar_max_distance,
        action_repeat_scale=args.action_repeat_scale,
    )

    episodes: list[dict[str, Any]] = []
    gif_paths: list[str] = []
    try:
        for ep in range(args.eval_episodes):
            obs, _ = env.reset(seed=eval_seed + ep)
            vec_obs = obs["vector"].astype(np.float32, copy=False)
            hp = max(float(vec_obs[0]) * 20.0, 0.0)
            shots = 0
            hits = 0
            kills = 0
            dealt = 0.0
            taken = 0.0
            steps = 0
            frames: list[np.ndarray] = [
                annotate_frame(
                    obs["image"],
                    [
                        f"{round_id.split('_')[0]} e{epoch:03d} ep{ep + 1}",
                        f"hp={hp:.1f} shots={shots} hits={hits} kills={kills}",
                        f"dealt={dealt:.2f} taken={taken:.2f} step={steps}",
                    ],
                )
            ]
            done = False
            truncated = False
            ep_reward = 0.0
            last_info: dict[str, Any] = {}

            while not done and not truncated:
                action, _ = model.predict(vec_obs, deterministic=args.eval_deterministic)
                obs, reward, done, truncated, info = env.step(int(action))
                vec_obs = obs["vector"].astype(np.float32, copy=False)
                ep_reward += float(reward)
                last_info = info
                hp = max(float(vec_obs[0]) * 20.0, 0.0)
                shots = int(info.get("shots_fired", shots))
                hits = int(info.get("target_hits", hits))
                kills = int(info.get("target_kills", kills))
                dealt = float(info.get("target_damage_dealt", dealt))
                taken = float(info.get("damage_taken", taken))
                steps = int(info.get("survival_steps", steps))
                if len(frames) < args.max_recorded_frames:
                    frames.append(
                        annotate_frame(
                            obs["image"],
                            [
                                f"{round_id.split('_')[0]} e{epoch:03d} ep{ep + 1}",
                                f"hp={hp:.1f} shots={shots} hits={hits} kills={kills}",
                                f"dealt={dealt:.2f} taken={taken:.2f} step={steps}",
                            ],
                        )
                    )

            metrics = dict(last_info.get("episode_metrics", {}))
            metrics["episode_reward"] = ep_reward
            episodes.append(metrics)

            gif_path = save_gif(
                frames,
                round_dir / "gifs" / f"{round_id}_epoch_{epoch:03d}_ep_{ep + 1:02d}.gif",
                frame_ms=args.gif_frame_duration_ms,
                end_hold_ms=args.gif_end_hold_ms,
                min_ms=args.gif_min_duration_ms,
            )
            if gif_path:
                gif_paths.append(gif_path)
    finally:
        env.close()

    def avg(key: str) -> float:
        values = [float(item.get(key, 0.0)) for item in episodes]
        return mean(values) if values else 0.0

    mean_shots = avg("shots_fired")
    mean_hits = avg("target_hits")
    mean_kills = avg("target_kills")

    summary = {
        "round_id": round_id,
        "round_name": round_spec.name,
        "stage_name": round_spec.stage.name,
        "epoch": epoch,
        "episodes": len(episodes),
        "mean_episode_reward": avg("episode_reward"),
        "mean_survival_steps": avg("survival_steps"),
        "mean_shots_fired": mean_shots,
        "mean_target_hits": mean_hits,
        "mean_target_kills": mean_kills,
        "mean_target_damage_dealt": avg("target_damage_dealt"),
        "mean_damage_taken": avg("damage_taken"),
        "mean_hit_rate": (mean_hits / mean_shots) if mean_shots > 0 else 0.0,
        "mean_hit_to_kill": (mean_kills / mean_hits) if mean_hits > 0 else 0.0,
        "mean_wasted_shots": max(mean_shots - mean_hits, 0.0),
        "mean_move_macro_count": avg("move_macro_count"),
        "mean_turn_macro_count": avg("turn_macro_count"),
        "mean_shoot_macro_count": avg("shoot_macro_count"),
        "mean_no_op_macro_count": avg("no_op_macro_count"),
        "mean_idle_steps": avg("idle_steps"),
        "mean_stationary_combat_steps": avg("stationary_combat_steps"),
        "mean_stuck_steps": avg("stuck_steps"),
        "termination_reason": episodes[-1].get("termination_reason", "") if episodes else "",
        "gifs": gif_paths,
    }

    summary_path = round_dir / f"epoch_{epoch:03d}_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_round_plot(round_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    epochs = [int(item["epoch"]) for item in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [item["mean_survival_steps"] for item in history], label="survival_steps")
    plt.plot(epochs, [item["mean_target_hits"] for item in history], label="hits")
    plt.plot(epochs, [item["mean_target_kills"] for item in history], label="kills")
    plt.plot(epochs, [item["mean_hit_rate"] for item in history], label="hit_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"Round Metrics: {history[0]['round_id']}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(round_dir / "round_metrics.png", dpi=140)
    plt.close()


def fmt_num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    return f"{number:.{digits}f}"


def cleanup_existing_clients() -> None:
    report = cleanup_craftground_processes(wait_seconds=5.0)
    matched = len(report.get("processes", []))
    terminated = len(report.get("terminated_pids", []))
    killed = len(report.get("killed_pids", []))
    if matched or terminated or killed:
        print(
            "CLEANUP "
            f"matched={matched} terminated={terminated} killed={killed}"
        )


def build_play_round(rounds: list[RoundSpec], args: argparse.Namespace) -> RoundSpec:
    round_index = args.play_round_index if args.play_round_index > 0 else args.start_round_index
    if round_index < 1 or round_index > len(rounds):
        raise SystemExit("--play-round-index must be within curriculum range")
    base_round = rounds[round_index - 1]
    scale = max(args.play_max_steps_scale, 1.0)
    stage = replace(
        base_round.stage,
        max_steps=max(1, int(round(base_round.stage.max_steps * scale))),
        no_shot_timeout=max(9999, int(round(base_round.stage.no_shot_timeout * scale * 4))),
        no_hit_timeout=max(9999, int(round(base_round.stage.no_hit_timeout * scale * 4))),
    )
    reward = replace(
        base_round.reward,
        idle_timeout_steps=max(9999, int(round(base_round.reward.idle_timeout_steps * scale * 4))),
        stationary_combat_timeout_steps=max(
            9999,
            int(round(base_round.reward.stationary_combat_timeout_steps * scale * 4)),
        ),
        idle_timeout_penalty=0.0,
        stationary_combat_timeout_penalty=0.0,
    )
    return RoundSpec(
        name=f"{base_round.name}_play",
        stage=stage,
        reward=reward,
        total_timesteps=base_round.total_timesteps,
    )


def replay_checkpoint_cv2(args: argparse.Namespace, rounds: list[RoundSpec]) -> None:
    if not args.play_checkpoint.strip():
        raise SystemExit("--play-checkpoint is required for replay mode")
    checkpoint = Path(args.play_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = HYUNJIN_ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Replay checkpoint not found: {checkpoint}")

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "cv2 is not installed in the project venv. Install opencv-python first."
        ) from exc

    round_spec = build_play_round(rounds, args)
    model = PPO.load(str(checkpoint), device=resolve_device(args.device))
    round_id = f"play_r{args.play_round_index if args.play_round_index > 0 else args.start_round_index:02d}_{round_spec.name}"
    replay_dir = HYUNJIN_ROOT / args.save_root / "playbacks"
    replay_dir.mkdir(parents=True, exist_ok=True)
    playback_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = f"{checkpoint.parent.parent.name}_{checkpoint.parent.name}_{checkpoint.stem}"
    mode_tag = "open" if args.play_open_arena else "arena"
    spawn_tag = f"spawn{args.play_near_spawn_scale:.2f}".replace(".", "p")
    video_path = replay_dir / (
        f"{version_tag}_{round_id}_{playback_started_at}_{mode_tag}_{spawn_tag}.mp4"
    )
    eval_port = args.port_start + 1700
    env = SurviveAndHuntEnvironment(
        stage=round_spec.stage,
        image_width=args.image_width,
        image_height=args.image_height,
        env_path=args.env_path,
        port=eval_port,
        seed=args.seed,
        use_vglrun=args.use_vglrun,
        reward_config=round_spec.reward,
        render_distance=args.render_distance,
        simulation_distance=args.simulation_distance,
        lidar_horizontal_rays=args.lidar_horizontal_rays,
        lidar_vertical_rays=args.lidar_vertical_rays,
        lidar_max_distance=args.lidar_max_distance,
        action_repeat_scale=args.action_repeat_scale,
        open_arena_mode=DEFAULT_OPEN_WORLD_MODE if args.play_open_arena else False,
        spawn_distance_scale=args.play_near_spawn_scale if args.play_open_arena else 1.0,
    )
    writer = None
    try:
        obs, _ = env.reset(seed=args.seed)
        vec_obs = obs["vector"].astype(np.float32, copy=False)
        done = False
        truncated = False
        episode_reward = 0.0
        final_reason = ""
        while not done and not truncated:
            action, _ = model.predict(vec_obs, deterministic=args.play_deterministic)
            obs, reward, done, truncated, info = env.step(int(action))
            vec_obs = obs["vector"].astype(np.float32, copy=False)
            episode_reward += float(reward)
            final_reason = str(info.get("termination_reason", "") or "")
            overlay = annotate_frame(
                obs["image"],
                [
                    f"{round_id} step={int(info.get('survival_steps', 0))}",
                    f"reward={episode_reward:.2f} hp={max(float(vec_obs[0]) * 20.0, 0.0):.1f}",
                    f"shots={int(info.get('shots_fired', 0))} hits={int(info.get('target_hits', 0))} kills={int(info.get('target_kills', 0))}",
                    f"reason={info.get('termination_reason', '') or '-'}",
                ],
            )
            bgr_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            if args.play_save_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        max(args.play_fps, 1.0),
                        (bgr_frame.shape[1], bgr_frame.shape[0]),
                    )
                writer.write(bgr_frame)
            cv2.imshow("CraftGround Replay", bgr_frame)
            key = cv2.waitKey(max(1, int(round(1000.0 / max(args.play_fps, 1.0))))) & 0xFF
            if key in (27, ord("q")):
                break
        print(
            "PLAYBACK "
            f"checkpoint={checkpoint} round={round_id} "
            f"open_arena={args.play_open_arena} near_spawn_scale={args.play_near_spawn_scale:.2f} "
            f"reward={episode_reward:.3f} "
            f"reason={final_reason or '-'} "
            f"video={video_path if args.play_save_video else '-'}"
        )
    finally:
        if writer is not None:
            writer.release()
        env.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    cleanup_existing_clients()
    if args.num_envs > SAFE_MAX_ENVS:
        print(f"WARN num_envs={args.num_envs} is high for WSL; capping to {SAFE_MAX_ENVS}")
        args.num_envs = SAFE_MAX_ENVS
    # User-requested fixed environment sensing/rendering configuration.
    args.render_distance = FIXED_RENDER_DISTANCE
    args.simulation_distance = FIXED_SIMULATION_DISTANCE
    args.lidar_max_distance = FIXED_LIDAR_MAX_DISTANCE
    run_name = args.run_name.strip() or datetime.now().strftime("raycast_%Y%m%d_%H%M%S")
    save_root = HYUNJIN_ROOT / args.save_root / run_name
    tb_root = HYUNJIN_ROOT / args.tensorboard_root
    save_root.mkdir(parents=True, exist_ok=True)

    legacy_root = Path(args.legacy_root)
    legacy_exists = legacy_root.exists()

    run_meta = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "legacy_root": str(legacy_root),
        "legacy_root_exists": legacy_exists,
        "hyunjin_root": str(HYUNJIN_ROOT),
        "craftground_src": str(CRAFTGROUND_SRC),
        "legacy_src": str(LEGACY_SRC),
        "args": vars(args),
    }
    (save_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    rounds = build_rounds(args.timesteps_scale)
    if args.play_checkpoint.strip():
        replay_checkpoint_cv2(args, rounds)
        return
    if args.start_round_index < 1:
        raise SystemExit("--start-round-index must be >= 1")
    if args.end_round_index < 0:
        raise SystemExit("--end-round-index must be >= 0")
    end_round_index = args.end_round_index if args.end_round_index > 0 else len(rounds)
    if end_round_index < args.start_round_index:
        raise SystemExit("--end-round-index must be >= --start-round-index")
    if args.start_round_index > 1 and not args.init_checkpoint.strip():
        raise SystemExit("Starting from round > 1 requires --init-checkpoint")

    selected_rounds = rounds[args.start_round_index - 1 : end_round_index]
    device = resolve_device(args.device)
    print(
        "RUN start "
        f"run={run_name} device={device} "
        f"rounds={','.join([r.name for r in selected_rounds])} "
        f"range={args.start_round_index}-{end_round_index}"
    )

    model: PPO | None = None
    total_epochs = 0
    global_history: list[dict[str, Any]] = []

    for round_index, round_spec in enumerate(selected_rounds, start=args.start_round_index):
        round_id = f"r{round_index:02d}_{round_spec.name}"
        round_dir = save_root / round_id
        round_dir.mkdir(parents=True, exist_ok=True)

        vec_env = build_vec_env(round_spec, args)
        if model is None:
            if args.init_checkpoint.strip():
                init_ckpt = Path(args.init_checkpoint)
                if not init_ckpt.is_absolute():
                    init_ckpt = HYUNJIN_ROOT / init_ckpt
                if not init_ckpt.exists():
                    raise SystemExit(f"Init checkpoint not found: {init_ckpt}")
                model = PPO.load(str(init_ckpt), env=vec_env, device=device)
                model.learning_rate = args.learning_rate
                model.ent_coef = args.ent_coef
                if args.n_steps != int(getattr(model, "n_steps", args.n_steps)):
                    print(
                        "WARN continuing from checkpoint: ignoring --n-steps "
                        f"{args.n_steps} and keeping checkpoint n_steps={model.n_steps}"
                    )
                if args.batch_size != int(getattr(model, "batch_size", args.batch_size)):
                    print(
                        "WARN continuing from checkpoint: ignoring --batch-size "
                        f"{args.batch_size} and keeping checkpoint batch_size={model.batch_size}"
                    )
                if args.n_epochs != int(getattr(model, "n_epochs", args.n_epochs)):
                    print(
                        "WARN continuing from checkpoint: ignoring --n-epochs "
                        f"{args.n_epochs} and keeping checkpoint n_epochs={model.n_epochs}"
                    )
            else:
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    verbose=1,
                    tensorboard_log=str(tb_root),
                    device=device,
                    learning_rate=args.learning_rate,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=args.ent_coef,
                    clip_range=0.2,
                    policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
                )
        else:
            model.set_env(vec_env)

        round_history: list[dict[str, Any]] = []
        epoch_count = max(1, math.ceil(round_spec.total_timesteps / max(args.steps_per_epoch, 1)))

        for epoch in range(1, epoch_count + 1):
            remaining = round_spec.total_timesteps - (epoch - 1) * args.steps_per_epoch
            train_steps = min(args.steps_per_epoch, max(remaining, 1))
            model.learn(
                total_timesteps=train_steps,
                reset_num_timesteps=False,
                tb_log_name=run_name,
                progress_bar=True,
            )
            total_epochs += 1

            ckpt = round_dir / f"epoch_{epoch:03d}.zip"
            model.save(str(ckpt))

            should_eval = (epoch % EVAL_EVERY_EPOCHS == 0) or (epoch == epoch_count)
            if should_eval:
                eval_summary = evaluate_epoch(model, round_id, round_spec, epoch, args, round_dir)
                eval_summary["total_epochs"] = total_epochs
                eval_summary["global_timesteps"] = int(getattr(model, "num_timesteps", 0))

                round_history.append(eval_summary)
                global_history.append(eval_summary)

                append_csv_row(round_dir / "epoch_metrics.csv", eval_summary)
                append_csv_row(save_root / "global_epoch_metrics.csv", eval_summary)

                print(
                    "EVAL "
                    f"round={eval_summary.get('round_id', '-')} "
                    f"stage={eval_summary.get('stage_name', '-')} "
                    f"epoch={eval_summary.get('epoch', '-')} "
                    f"global_ts={eval_summary.get('global_timesteps', '-')} "
                    f"survival={fmt_num(eval_summary.get('mean_survival_steps'), 1)} "
                    f"hit_rate={fmt_num(eval_summary.get('mean_hit_rate'), 3)} "
                    f"hit2kill={fmt_num(eval_summary.get('mean_hit_to_kill'), 3)} "
                    f"kills={fmt_num(eval_summary.get('mean_target_kills'), 2)} "
                    f"wasted={fmt_num(eval_summary.get('mean_wasted_shots'), 1)} "
                    f"reward={fmt_num(eval_summary.get('mean_episode_reward'), 3)}"
                )
            else:
                print(
                    "TRAIN "
                    f"round={round_id} stage={round_spec.stage.name} "
                    f"epoch={epoch}/{epoch_count} global_ts={int(getattr(model, 'num_timesteps', 0))}"
                )

        save_round_plot(round_dir, round_history)
        vec_env.close()

    assert model is not None
    model.save(str(save_root / "final_model.zip"))
    (save_root / "global_summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "rounds": len(rounds),
                "epochs": len(global_history),
                "final_timesteps": int(getattr(model, "num_timesteps", 0)),
                "history": global_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"RUN completed run_dir={save_root}")


if __name__ == "__main__":
    main()
