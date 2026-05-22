"""
Demo verification and visualization.

Usage:
    # Basic stats
    python -m cuboid_house_rl.expert.verify_demos --demo-path demos/demos_floor.npz

    # With action distribution visualization
    python -m cuboid_house_rl.expert.verify_demos --demo-path demos/demos_floor.npz --preview
"""
import argparse
import os
import sys
import numpy as np

from cuboid_house_rl.config import (
    FLAT_OBS_SIZE, ACTION_DIMS, NUM_ACTION_DIMS,
    CAMERA_DELTA_MAP,
    ORIGIN_SET_SIZE, AGENT_STATE_SIZE, RAYCAST_INFO_SIZE,
    PROGRESS_SIZE, INCORRECT_COUNT_SIZE,
    TIME_REMAINING_SIZE, STUCK_RATIO_SIZE,
    TARGET_DIRECTION_SIZE, TARGET_DISTANCE_SIZE,
    TARGET_ABSOLUTE_SIZE, HOUSE_SIZE_SIZE,
)


# Observation field layout
OBS_FIELDS = []
idx = 0
OBS_FIELDS.append(("origin_set", idx, idx + ORIGIN_SET_SIZE)); idx += ORIGIN_SET_SIZE
OBS_FIELDS.append(("agent_pos", idx, idx + 3)); idx += 3
OBS_FIELDS.append(("agent_yaw_sin_cos", idx, idx + 2)); idx += 2
OBS_FIELDS.append(("agent_pitch_sin_cos", idx, idx + 2)); idx += 2
OBS_FIELDS.append(("hotbar_slot", idx, idx + 1)); idx += 1
OBS_FIELDS.append(("has_planks", idx, idx + 1)); idx += 1
OBS_FIELDS.append(("has_axe", idx, idx + 1)); idx += 1
OBS_FIELDS.append(("raycast", idx, idx + RAYCAST_INFO_SIZE)); idx += RAYCAST_INFO_SIZE
OBS_FIELDS.append(("progress", idx, idx + PROGRESS_SIZE)); idx += PROGRESS_SIZE
OBS_FIELDS.append(("incorrect_count", idx, idx + 1)); idx += 1
OBS_FIELDS.append(("time_remaining", idx, idx + 1)); idx += 1
OBS_FIELDS.append(("stuck_ratio", idx, idx + 1)); idx += 1
OBS_FIELDS.append(("target_direction", idx, idx + TARGET_DIRECTION_SIZE)); idx += TARGET_DIRECTION_SIZE
OBS_FIELDS.append(("target_distance", idx, idx + 1)); idx += 1
OBS_FIELDS.append(("target_absolute", idx, idx + TARGET_ABSOLUTE_SIZE)); idx += TARGET_ABSOLUTE_SIZE
OBS_FIELDS.append(("house_size", idx, idx + HOUSE_SIZE_SIZE)); idx += HOUSE_SIZE_SIZE

ACTION_DIM_NAMES = [
    "fwd/back", "left/right", "jump", "sneak",
    "interact", "hotbar", "pitch", "yaw"
]
ACTION_DIM_LABELS = [
    ["back", "stop", "fwd"],
    ["left", "stop", "right"],
    ["no", "yes"],
    ["no", "yes"],
    ["place", "noop", "attack"],
    [f"slot{i}" for i in range(9)],
    [f"{d:+.0f}°" for d in CAMERA_DELTA_MAP],
    [f"{d:+.0f}°" for d in CAMERA_DELTA_MAP],
]


def load_demos(path: str) -> dict:
    """Load and validate demo file."""
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    data = np.load(path, allow_pickle=True)

    obs = data["observations"]
    actions = data["actions"]
    episode_ids = data["episode_ids"]
    stage = str(data.get("stage", "unknown"))

    # Validate shapes
    assert obs.shape[1] == FLAT_OBS_SIZE, \
        f"Obs dim mismatch: got {obs.shape[1]}, expected {FLAT_OBS_SIZE}"
    assert actions.shape[1] == NUM_ACTION_DIMS, \
        f"Action dim mismatch: got {actions.shape[1]}, expected {NUM_ACTION_DIMS}"
    assert len(obs) == len(actions) == len(episode_ids), \
        "Length mismatch between obs/actions/episode_ids"

    return {
        "observations": obs,
        "actions": actions,
        "episode_ids": episode_ids,
        "stage": stage,
    }


def print_basic_stats(data: dict, path: str):
    """Print basic statistics about the demo file."""
    obs = data["observations"]
    actions = data["actions"]
    episode_ids = data["episode_ids"]
    stage = data["stage"]

    unique_episodes = np.unique(episode_ids)
    file_size = os.path.getsize(path) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"Demo file: {path}")
    print(f"{'='*60}")
    print(f"  Stage:          {stage}")
    print(f"  File size:      {file_size:.1f} MB")
    print(f"  Total steps:    {len(obs):,}")
    print(f"  Episodes:       {len(unique_episodes)}")
    print(f"  Obs shape:      {obs.shape} (expected (N, {FLAT_OBS_SIZE}))")
    print(f"  Action shape:   {actions.shape} (expected (N, {NUM_ACTION_DIMS}))")

    # Per-episode stats
    ep_lengths = []
    for ep_id in unique_episodes:
        mask = episode_ids == ep_id
        ep_lengths.append(mask.sum())

    ep_lengths = np.array(ep_lengths)
    print(f"\n  Episode lengths:")
    print(f"    mean:  {ep_lengths.mean():.0f}")
    print(f"    min:   {ep_lengths.min()}")
    print(f"    max:   {ep_lengths.max()}")
    print(f"    std:   {ep_lengths.std():.0f}")


def print_episode_details(data: dict):
    """Print per-episode completion and success info."""
    obs = data["observations"]
    episode_ids = data["episode_ids"]
    unique_episodes = np.unique(episode_ids)

    # Progress is at a known offset in obs
    progress_start = (ORIGIN_SET_SIZE + AGENT_STATE_SIZE + RAYCAST_INFO_SIZE)
    progress_end = progress_start + PROGRESS_SIZE

    # House size
    house_size_start = FLAT_OBS_SIZE - HOUSE_SIZE_SIZE

    print(f"\n  Per-episode details:")
    print(f"  {'Ep':>4} {'Steps':>6} {'Floor%':>7} {'Wall%':>7} {'Ceil%':>7} {'Size':>7} {'Origin':>7}")
    print(f"  {'-'*4} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    floor_completions = []
    for ep_id in unique_episodes:
        mask = episode_ids == ep_id
        ep_obs = obs[mask]
        ep_len = len(ep_obs)

        # Last observation's progress
        last_obs = ep_obs[-1]
        floor_r = last_obs[progress_start]
        wall_r = last_obs[progress_start + 1]
        ceil_r = last_obs[progress_start + 2]

        # House size (from any obs, should be constant)
        w = int(last_obs[house_size_start])
        d = int(last_obs[house_size_start + 1])
        h = int(last_obs[house_size_start + 2])

        # Origin set?
        origin = "yes" if last_obs[0] > 0.5 else "no"

        floor_completions.append(floor_r)

        print(f"  {ep_id:>4} {ep_len:>6} {floor_r:>6.0%} {wall_r:>6.0%} "
              f"{ceil_r:>6.0%} {w}x{d}x{h} {origin:>7}")

    floor_completions = np.array(floor_completions)
    print(f"\n  Floor completion: mean={floor_completions.mean():.0%}, "
          f"100%={np.sum(floor_completions >= 0.99)}/{len(floor_completions)} episodes")


def print_action_distribution(data: dict):
    """Print action distribution per dimension."""
    actions = data["actions"]

    print(f"\n  Action distributions:")
    offset = 0
    for i, (dim_name, dim_size) in enumerate(zip(ACTION_DIM_NAMES, ACTION_DIMS)):
        dim_actions = actions[:, i]
        counts = np.bincount(dim_actions, minlength=dim_size)
        pcts = counts / len(actions) * 100
        labels = ACTION_DIM_LABELS[i]

        print(f"\n  [{i}] {dim_name} ({dim_size} options):")
        for j in range(dim_size):
            bar = "█" * int(pcts[j] / 2)
            label = labels[j] if j < len(labels) else f"#{j}"
            print(f"    {label:>8}: {counts[j]:>7,} ({pcts[j]:>5.1f}%) {bar}")


def print_obs_ranges(data: dict):
    """Print observation value ranges per field."""
    obs = data["observations"]

    print(f"\n  Observation value ranges:")
    print(f"  {'Field':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, start, end in OBS_FIELDS:
        field = obs[:, start:end]
        if field.shape[1] == 1:
            field = field.flatten()
            print(f"  {name:<25} {field.min():>10.3f} {field.max():>10.3f} "
                  f"{field.mean():>10.3f} {field.std():>10.3f}")
        else:
            for j in range(field.shape[1]):
                sub = field[:, j]
                label = f"{name}[{j}]"
                print(f"  {label:<25} {sub.min():>10.3f} {sub.max():>10.3f} "
                      f"{sub.mean():>10.3f} {sub.std():>10.3f}")


def preview_episode(data: dict, episode: int = 0, max_steps: int = 50):
    """Print step-by-step replay of a specific episode."""
    obs = data["observations"]
    actions = data["actions"]
    episode_ids = data["episode_ids"]

    mask = episode_ids == episode
    if not mask.any():
        print(f"  Episode {episode} not found.")
        return

    ep_obs = obs[mask]
    ep_actions = actions[mask]

    progress_start = ORIGIN_SET_SIZE + AGENT_STATE_SIZE + RAYCAST_INFO_SIZE
    pos_start = ORIGIN_SET_SIZE  # agent_pos starts after origin_set

    print(f"\n  Episode {episode} replay (first {max_steps} steps):")
    print(f"  {'Step':>5} {'Action':>40} {'Pos(x,y,z)':>20} {'Floor%':>7} {'Ray':>4}")
    print(f"  {'-'*5} {'-'*40} {'-'*20} {'-'*7} {'-'*4}")

    for step in range(min(len(ep_obs), max_steps)):
        o = ep_obs[step]
        a = ep_actions[step]

        # Decode action
        parts = []
        for i, (name, labels) in enumerate(zip(ACTION_DIM_NAMES, ACTION_DIM_LABELS)):
            val = int(a[i])
            label = labels[val] if val < len(labels) else f"#{val}"
            if label not in ("stop", "no", "noop", "slot0", "+0°"):
                parts.append(f"{name}={label}")
        action_str = ", ".join(parts) if parts else "(noop)"

        # Position
        px, py, pz = o[pos_start], o[pos_start + 1], o[pos_start + 2]

        # Progress
        floor_r = o[progress_start]

        # Raycast hit
        ray_start = ORIGIN_SET_SIZE + AGENT_STATE_SIZE
        ray_hit = "hit" if o[ray_start] > 0.5 else "-"

        print(f"  {step:>5} {action_str:>40} "
              f"({px:>5.1f},{py:>4.1f},{pz:>5.1f}) "
              f"{floor_r:>6.0%} {ray_hit:>4}")


def main():
    parser = argparse.ArgumentParser(description="Verify and visualize demo data")
    parser.add_argument("--demo-path", type=str, required=True)
    parser.add_argument("--preview", action="store_true",
                        help="Show detailed action distributions and episode replay")
    parser.add_argument("--replay-episode", type=int, default=0,
                        help="Episode index to replay in preview mode")
    parser.add_argument("--replay-steps", type=int, default=50,
                        help="Max steps to show in replay")
    args = parser.parse_args()

    data = load_demos(args.demo_path)

    print_basic_stats(data, args.demo_path)
    print_episode_details(data)

    if args.preview:
        print_action_distribution(data)
        print_obs_ranges(data)
        preview_episode(data, episode=args.replay_episode,
                        max_steps=args.replay_steps)

    print(f"\n{'='*60}")
    print("Verification complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
