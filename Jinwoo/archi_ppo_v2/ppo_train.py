"""
PPO Training Script (MLP only, no LSTM).

Identical to train.py except the recurrent LSTM is replaced by a deeper
feed-forward trunk (ppo_network.BuilderNetwork).

Usage:
    python ppo_train.py --curriculum
    python ppo_train.py --nbt path/to/house.nbt
    python ppo_train.py --resume checkpoints_ppo/latest.pt
"""

import argparse
import os
import signal
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List


def _kill_java_children():
    """Kill any lingering Minecraft Java child processes (WSL2 / killpg misses them)."""
    try:
        import psutil
        current = psutil.Process()
        for child in current.children(recursive=True):
            try:
                if "java" in child.name().lower():
                    child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass
    except Exception:
        pass


from nbt_parser import (
    Blueprint,
    parse_nbt_file,
    create_simple_blueprint,
    get_blueprint_stats,
)
from building_env import (
    HouseBuildingWrapper,
    BuildingConfig,
    make_building_env,
    NUM_BUILD_ACTIONS,
    BuildAction,
)
from ppo_network import BuilderNetwork  # MLP-only network (no LSTM)



# ──────────────────────────────────────────────────────────────────────
# PPO Algorithm  (identical to train.py — network API is compatible)
# ──────────────────────────────────────────────────────────────────────
class PPOTrainer:
    def __init__(
        self,
        network: BuilderNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str = "cpu",
    ):
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device

    def collect_rollout(self, env: HouseBuildingWrapper, num_steps: int = 256) -> dict:
        obs_local, obs_target, obs_diff, obs_raycast = [], [], [], []
        obs_pos, obs_progress = [], []
        obs_block_id = []
        obs_struct_info = []
        actions, log_probs, values, rewards, dones = [], [], [], [], []

        self._rollout_episode_completions: list = []
        self._rollout_aim_accuracies: list = []
        self._rollout_reward_components: list = []

        obs, info = env.reset()

        for step in range(num_steps):
            def to_tensor(arr):
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                return torch.from_numpy(arr).unsqueeze(0).to(self.device)

            local_t  = to_tensor(obs["local_grid"])
            target_t = to_tensor(obs["target_grid"])
            diff_t   = to_tensor(obs["diff_grid"].astype(np.float32))
            ray_t    = to_tensor(obs["raycast_grid"])
            pos_t    = to_tensor(obs["agent_pos"])
            prog_t   = to_tensor(obs["progress"])
            nblk_t   = to_tensor(obs["next_block_id"])
            struct_t = to_tensor(obs["structure_info"])

            with torch.no_grad():
                logits, value, _ = self.network(
                    local_t, target_t, diff_t, ray_t, pos_t, prog_t,
                    next_block_id=nblk_t, structure_info=struct_t,
                )
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            obs_local.append(obs["local_grid"])
            obs_target.append(obs["target_grid"])
            obs_diff.append(obs["diff_grid"].astype(np.float32))
            obs_raycast.append(obs["raycast_grid"])
            obs_pos.append(obs["agent_pos"])
            obs_progress.append(obs["progress"])
            obs_block_id.append(obs["next_block_id"])
            obs_struct_info.append(obs["structure_info"])
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())

            obs, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            dones.append(terminated or truncated)

            if terminated or truncated:
                self._rollout_episode_completions.append(info.get("completion_pct", 0.0))
                self._rollout_aim_accuracies.append(info.get("aim_accuracy", 0.0))
                rc = info.get("reward_components")
                if rc:
                    self._rollout_reward_components.append(rc)
                obs, info = env.reset()

        # Final value for GAE bootstrap
        with torch.no_grad():
            def to_tensor(arr):
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                return torch.from_numpy(arr).unsqueeze(0).to(self.device)
            _, last_value, _ = self.network(
                to_tensor(obs["local_grid"]),
                to_tensor(obs["target_grid"]),
                to_tensor(obs["diff_grid"].astype(np.float32)),
                to_tensor(obs["raycast_grid"]),
                to_tensor(obs["agent_pos"]),
                to_tensor(obs["progress"]),
                next_block_id=to_tensor(obs["next_block_id"]),
                structure_info=to_tensor(obs["structure_info"]),
            )

        advantages, returns = self._compute_gae(rewards, values, dones, last_value.item())

        return {
            "obs_local":        np.array(obs_local),
            "obs_target":       np.array(obs_target),
            "obs_diff":         np.array(obs_diff),
            "obs_raycast":      np.array(obs_raycast),
            "obs_pos":          np.array(obs_pos),
            "obs_progress":     np.array(obs_progress),
            "obs_block_id":     np.array(obs_block_id),
            "obs_struct_info":  np.array(obs_struct_info),
            "actions":          np.array(actions),
            "log_probs":        np.array(log_probs),
            "values":           np.array(values),
            "returns":          returns,
            "advantages":       advantages,
        }

    def update(self, rollout: dict) -> dict:
        n = len(rollout["actions"])

        local_t       = torch.tensor(rollout["obs_local"],        dtype=torch.float32, device=self.device)
        target_t      = torch.tensor(rollout["obs_target"],       dtype=torch.float32, device=self.device)
        diff_t        = torch.tensor(rollout["obs_diff"],         dtype=torch.float32, device=self.device)
        pos_t         = torch.tensor(rollout["obs_pos"],          dtype=torch.float32, device=self.device)
        prog_t        = torch.tensor(rollout["obs_progress"],     dtype=torch.float32, device=self.device)
        ray_t         = torch.tensor(rollout["obs_raycast"],      dtype=torch.float32, device=self.device)
        blk_t         = torch.tensor(rollout["obs_block_id"],     dtype=torch.float32, device=self.device)
        struct_t      = torch.tensor(rollout["obs_struct_info"],  dtype=torch.float32, device=self.device)
        actions_t     = torch.tensor(rollout["actions"],         dtype=torch.long,    device=self.device)
        old_log_probs = torch.tensor(rollout["log_probs"],       dtype=torch.float32, device=self.device)
        returns_t     = torch.tensor(rollout["returns"],         dtype=torch.float32, device=self.device)
        advantages_t  = torch.tensor(rollout["advantages"],      dtype=torch.float32, device=self.device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = total_value_loss = total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > n:
                    continue
                mb = indices[start:end]

                logits, vals, _ = self.network(
                    local_t[mb], target_t[mb], diff_t[mb], ray_t[mb],
                    pos_t[mb], prog_t[mb],
                    next_block_id=blk_t[mb], structure_info=struct_t[mb],
                )
                dist         = torch.distributions.Categorical(logits=logits)
                new_lp       = dist.log_prob(actions_t[mb])
                entropy      = dist.entropy().mean()
                ratio        = torch.exp(new_lp - old_log_probs[mb])
                surr1        = ratio * advantages_t[mb]
                surr2        = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t[mb]
                policy_loss  = -torch.min(surr1, surr2).mean()
                value_loss   = nn.functional.mse_loss(vals, returns_t[mb])
                loss         = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()

        num_updates = max(self.ppo_epochs * (n // self.mini_batch_size), 1)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss":  total_value_loss  / num_updates,
            "entropy":     total_entropy     / num_updates,
        }

    def _compute_gae(self, rewards, values, dones, last_value):
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            mask       = 1.0 - float(dones[t])
            delta      = rewards[t] + self.gamma * next_value * mask - values[t]
            gae        = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]
            next_value    = values[t]
        return advantages, returns

    def save(self, path: str, extra_state: dict = None):
        checkpoint = {
            "network_state":   self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if extra_state:
            checkpoint["extra_state"] = extra_state
        torch.save(checkpoint, path)

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint.get("extra_state", {})


# ──────────────────────────────────────────────────────────────────────
# Curriculum
# ──────────────────────────────────────────────────────────────────────
CURRICULUM_STAGES = [
    {"name": "Stage 0: 2-Block Row",               "structure": "row_2",        "max_timesteps":  100, "success_threshold": 0.9, "min_episodes":  100, "randomize_per_episode": False, "place_aim_gate": 90.0},
    {"name": "Stage 1: 2-High Pillar",             "structure": "pillar_2",     "max_timesteps":  200, "success_threshold": 0.85,"min_episodes":  200, "randomize_per_episode": False, "place_aim_gate": 45.0},
    {"name": "Stage 2: Row of Blocks (2-10)",      "structure": "row",          "max_timesteps":  200, "success_threshold": 0.8, "min_episodes":  200, "randomize_per_episode": True,  "place_aim_gate": 25.0},
    {"name": "Stage 3: 3-High Wall",               "structure": "wall_3high",   "max_timesteps": 1000, "success_threshold": 0.7, "min_episodes":  500, "randomize_per_episode": False},
    {"name": "Stage 4: Small Room",                "structure": "small_room",   "max_timesteps": 2000, "success_threshold": 0.6, "min_episodes": 1000, "randomize_per_episode": False},
    {"name": "Stage 5: Full Blueprint (.nbt)",     "structure": None,           "max_timesteps": 5000, "success_threshold": 0.5, "min_episodes": 2000, "randomize_per_episode": False},
]


class BlueprintPool:
    def __init__(self, nbt_dir=None, nbt_files=None):
        self.blueprints: List[Blueprint] = []
        self.file_paths: List[str] = []
        self.stats: Dict[int, dict] = {}

        paths = list(nbt_files) if nbt_files else []
        if nbt_dir and os.path.isdir(nbt_dir):
            for fname in sorted(os.listdir(nbt_dir)):
                if fname.endswith(".nbt"):
                    paths.append(os.path.join(nbt_dir, fname))

        for path in paths:
            try:
                bp = parse_nbt_file(path)
                self.blueprints.append(bp)
                self.file_paths.append(path)
                idx = len(self.blueprints) - 1
                self.stats[idx] = {"attempts": 0, "total_completion": 0.0}
                print(f"  Loaded: {os.path.basename(path):40s} ({len(bp.blocks):4d} blocks, {bp.size_x}×{bp.size_y}×{bp.size_z})")
            except Exception as e:
                print(f"  SKIP:  {os.path.basename(path):40s} Error: {e}")

        if self.blueprints:
            sizes = [len(bp.blocks) for bp in self.blueprints]
            print(f"\n  Pool: {len(self.blueprints)} blueprints (blocks: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f})")

        self._sorted_indices = sorted(range(len(self.blueprints)), key=lambda i: len(self.blueprints[i].blocks))

    def __len__(self):
        return len(self.blueprints)

    def sample_random(self) -> Blueprint:
        if not self.blueprints:
            return create_simple_blueprint("wall_3high")
        return self.blueprints[np.random.randint(len(self.blueprints))]

    def report_completion(self, blueprint: Blueprint, completion: float):
        for i, bp in enumerate(self.blueprints):
            if bp is blueprint:
                self.stats[i]["attempts"] += 1
                self.stats[i]["total_completion"] += completion
                break

    def get_hardest(self, n: int = 5) -> List[int]:
        scored = [(s["total_completion"] / s["attempts"], i)
                  for i, s in self.stats.items() if s["attempts"] > 0]
        scored.sort()
        return [i for _, i in scored[:n]]

    def get_summary(self) -> str:
        lines = [f"Blueprint Pool: {len(self.blueprints)} houses"]
        for i, bp in enumerate(self.blueprints):
            s = self.stats.get(i, {"attempts": 0, "total_completion": 0})
            avg = s["total_completion"] / s["attempts"] if s["attempts"] > 0 else 0
            lines.append(f"  [{i:3d}] {os.path.basename(self.file_paths[i]):30s} {len(bp.blocks):4d} blocks  {bp.size_x}×{bp.size_y}×{bp.size_z}  avg={avg:.1%}  n={s['attempts']}")
        return "\n".join(lines)


class CurriculumManager:
    def __init__(self, nbt_blueprint=None, blueprint_pool=None):
        self.current_stage = 0
        self.nbt_blueprint = nbt_blueprint
        self.blueprint_pool = blueprint_pool
        self.episode_completions: List[float] = []
        self.total_episodes = 0
        self._current_blueprint: Optional[Blueprint] = None

    def get_current_blueprint(self) -> Blueprint:
        stage = CURRICULUM_STAGES[self.current_stage]
        if stage["structure"] is not None:
            bp = create_simple_blueprint(stage["structure"])
        elif self.blueprint_pool and len(self.blueprint_pool) > 0:
            bp = self.blueprint_pool.sample_random()
        elif self.nbt_blueprint is not None:
            bp = self.nbt_blueprint
        else:
            bp = create_simple_blueprint("cube_3x3x3")
        self._current_blueprint = bp
        return bp

    def get_max_timesteps(self) -> int:
        base = CURRICULUM_STAGES[self.current_stage]["max_timesteps"]
        if self._current_blueprint:
            return max(base, min(len(self._current_blueprint.blocks) * 10, 20000))
        return base

    def report_episode(self, completion_pct: float, aim_accuracy: float = 0.0):
        self.total_episodes += 1
        if self.blueprint_pool and self._current_blueprint:
            self.blueprint_pool.report_completion(self._current_blueprint, completion_pct)
        stage = CURRICULUM_STAGES[self.current_stage]
        metric = stage.get("success_metric", "completion")
        self.episode_completions.append(aim_accuracy if metric == "aim_accuracy" else completion_pct)
        if len(self.episode_completions) >= stage["min_episodes"]:
            if np.mean(self.episode_completions[-50:]) >= stage["success_threshold"]:
                if self.current_stage < len(CURRICULUM_STAGES) - 1:
                    self.current_stage += 1
                    self.episode_completions.clear()
                    print(f"\n{'='*60}\nCURRICULUM ADVANCED: {CURRICULUM_STAGES[self.current_stage]['name']}\n{'='*60}\n")

    @property
    def stage_name(self) -> str:
        return CURRICULUM_STAGES[self.current_stage]["name"]

    def get_blueprint_generator(self):
        """Return a callable that generates a new blueprint each call, or None."""
        stage = CURRICULUM_STAGES[self.current_stage]
        if stage.get("randomize_per_episode") and stage["structure"] is not None:
            struct = stage["structure"]
            return lambda: create_simple_blueprint(struct)
        return None

    def get_structure_name(self) -> str:
        return CURRICULUM_STAGES[self.current_stage]["structure"] or ""

    def get_place_aim_gate(self) -> float:
        return CURRICULUM_STAGES[self.current_stage].get("place_aim_gate", 25.0)


# ──────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device:
        device = args.device
    print(f"Using device: {device}")

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                group=args.wandb_group,     name=args.wandb_name,
                config=vars(args), save_code=True, monitor_gym=True,
            )
            print(f"W&B run: {run.url}")
        except ImportError:
            print("WARNING: wandb not installed. Continuing without W&B.")
            use_wandb = False
        except Exception as e:
            print(f"WARNING: wandb init failed: {e}")
            use_wandb = False

    nbt_blueprint = None
    blueprint_pool = None

    if args.nbt_dir and os.path.isdir(args.nbt_dir):
        print(f"Loading blueprints from directory: {args.nbt_dir}")
        blueprint_pool = BlueprintPool(nbt_dir=args.nbt_dir)

    if args.nbt and os.path.exists(args.nbt):
        print(f"Loading single blueprint: {args.nbt}")
        nbt_blueprint = parse_nbt_file(args.nbt)
        print(f"  Blueprint stats: {get_blueprint_stats(nbt_blueprint)}")

    curriculum = CurriculumManager(nbt_blueprint=nbt_blueprint, blueprint_pool=blueprint_pool)
    print(f"Starting curriculum: {curriculum.stage_name}")

    blueprint = curriculum.get_current_blueprint()
    stats = get_blueprint_stats(blueprint)
    print(f"Current blueprint: {stats['total_blocks']} blocks, dims={stats['dimensions']}")

    _kill_java_children()

    env = make_building_env(
        blueprint=blueprint, port=args.port,
        build_origin=(0, -60, 0), curriculum_stage=curriculum.current_stage,
        image_size=args.image_size, debug_visual=getattr(args, "debug_visual", False),
        max_timesteps=curriculum.get_max_timesteps(), seed=args.seed,
        blueprint_generator=curriculum.get_blueprint_generator(),
        structure_name=curriculum.get_structure_name(),
        place_aim_gate=curriculum.get_place_aim_gate(),
    )

    network = BuilderNetwork(
        obs_grid_size=2 * 5 + 1,
        num_actions=NUM_BUILD_ACTIONS,
        hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden,  # ignored by MLP network, kept for API compat
    )

    trainer = PPOTrainer(
        network=network, lr=args.lr, gamma=args.gamma,
        entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.batch_size, device=device,
    )

    start_iteration = 1
    best_completion = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        extra = trainer.load(args.resume)
        if extra:
            start_iteration = extra.get("iteration", 0) + 1
            best_completion  = extra.get("best_completion", 0.0)
            curriculum.current_stage = extra.get("curriculum_stage", 0)
            curriculum.total_episodes = extra.get("total_episodes", 0)
            print(f"  Restored: iter={start_iteration}, stage={curriculum.stage_name}, best={best_completion:.1%}")

        blueprint = curriculum.get_current_blueprint()
        env.close(); _kill_java_children()
        env = make_building_env(
            blueprint=blueprint, port=args.port,
            build_origin=(0, -60, 0), curriculum_stage=curriculum.current_stage,
            image_size=args.image_size, debug_visual=getattr(args, "debug_visual", False),
            max_timesteps=curriculum.get_max_timesteps(), seed=args.seed,
            blueprint_generator=curriculum.get_blueprint_generator(),
            structure_name=curriculum.get_structure_name(),
            place_aim_gate=curriculum.get_place_aim_gate(),
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save_checkpoint(path, iteration):
        trainer.save(path, extra_state={
            "iteration": iteration, "curriculum_stage": curriculum.current_stage,
            "total_episodes": curriculum.total_episodes,
            "best_completion": best_completion, "args": vars(args),
        })

    _env_ref = [env]
    def _shutdown(sig, frame):
        print("\n[Shutdown] Ctrl+C — closing Minecraft process...")
        try:   _env_ref[0].close()
        except Exception as e: print(f"  env.close() error: {e}")
        _kill_java_children()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    end_iteration = args.total_iterations + 1
    print(f"\nTraining iterations {start_iteration} → {end_iteration - 1}")
    print(f"Rollout steps per iteration: {args.rollout_steps}")
    prev_stage = curriculum.current_stage

    for iteration in range(start_iteration, end_iteration):
        start_time = time.time()

        # Swap blueprint in final stage with pool
        should_swap = (
            args.swap_interval > 0 and iteration % args.swap_interval == 0
            and curriculum.current_stage >= 5
            and blueprint_pool is not None and len(blueprint_pool) > 1
        )
        if should_swap:
            blueprint = curriculum.get_current_blueprint()
            env.close(); _kill_java_children()
            env = make_building_env(
                blueprint=blueprint, port=args.port,
                build_origin=(0, -60, 0), curriculum_stage=curriculum.current_stage,
                image_size=args.image_size, debug_visual=getattr(args, "debug_visual", False),
                max_timesteps=curriculum.get_max_timesteps(), seed=args.seed,
                blueprint_generator=curriculum.get_blueprint_generator(),
                structure_name=curriculum.get_structure_name(),
                place_aim_gate=curriculum.get_place_aim_gate(),
            )
            _env_ref[0] = env
            if iteration % args.log_interval == 0:
                print(f"  [Swap] New blueprint: {len(blueprint.blocks)} blocks, {blueprint.size_x}×{blueprint.size_y}×{blueprint.size_z}")

        rollout = trainer.collect_rollout(env, num_steps=args.rollout_steps)
        metrics = trainer.update(rollout)

        rollout_completions = getattr(trainer, '_rollout_episode_completions', [])
        rollout_aims        = getattr(trainer, '_rollout_aim_accuracies', [])
        rollout_rc_list     = getattr(trainer, '_rollout_reward_components', [])

        if rollout_rc_list:
            rc_keys = list(rollout_rc_list[0].keys())
            avg_rc = {k: float(np.mean([ep.get(k, 0.0) for ep in rollout_rc_list])) for k in rc_keys}
        else:
            avg_rc = {}

        if rollout_completions:
            completion_pct = float(np.mean(rollout_completions))
            aim_accuracy   = float(np.mean(rollout_aims)) if rollout_aims else 0.0
        else:
            completion_pct = float(rollout["obs_progress"][-1][0])
            aim_accuracy   = 0.0

        curriculum.report_episode(completion_pct, aim_accuracy=aim_accuracy)

        elapsed      = time.time() - start_time
        avg_reward   = np.mean(rollout["returns"])
        avg_advantage= np.mean(rollout["advantages"])

        if use_wandb:
            import wandb
            log_dict = {
                "train/policy_loss":  metrics["policy_loss"],
                "train/value_loss":   metrics["value_loss"],
                "train/entropy":      metrics["entropy"],
                "train/avg_return":   avg_reward,
                "train/avg_advantage":avg_advantage,
                "build/completion_pct":   completion_pct,
                "build/best_completion":  best_completion,
                "curriculum/stage":       curriculum.current_stage,
                "curriculum/total_episodes": curriculum.total_episodes,
                "perf/iteration_time_sec":   elapsed,
                "perf/steps_per_sec":        args.rollout_steps / max(elapsed, 0.001),
            }
            if curriculum._current_blueprint:
                log_dict["build/blueprint_blocks"] = len(curriculum._current_blueprint.blocks)
                log_dict["build/blueprint_height"]  = curriculum._current_blueprint.size_y
            for k, v in avg_rc.items():
                log_dict[f"rc/{k}"] = v
            wandb.log(log_dict, step=iteration)

        if iteration % args.log_interval == 0:
            n_eps = len(rollout_completions)
            print(
                f"Iter {iteration:5d} | Stage: {curriculum.stage_name[:25]:25s} | "
                f"Completion: {completion_pct:.1%} | Eps: {n_eps:2d} | "
                f"Avg Return: {avg_reward:7.2f} | Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | Time: {elapsed:.1f}s"
            )
            if avg_rc:
                r_parts = [(k[2:], v) for k, v in avg_rc.items() if k.startswith('r_')]
                p_parts = [(k[2:], v) for k, v in avg_rc.items() if k.startswith('p_')]
                r_total = sum(v for _, v in r_parts)
                p_total = sum(v for _, v in p_parts)
                print(f"  Rewards   (total={r_total:+.3f}): " + "  ".join(f"{k}={v:+.4f}" for k, v in r_parts))
                print(f"  Penalties (total={p_total:+.3f}): " + "  ".join(f"{k}={v:+.4f}" for k, v in p_parts))
                print(f"  Net reward: {r_total + p_total:+.3f}")

        # Curriculum advance → recreate env
        if curriculum.current_stage != prev_stage:
            blueprint = curriculum.get_current_blueprint()
            env.close(); _kill_java_children()
            env = make_building_env(
                blueprint=blueprint, port=args.port,
                build_origin=(0, -60, 0), curriculum_stage=curriculum.current_stage,
                image_size=args.image_size, debug_visual=getattr(args, "debug_visual", False),
                max_timesteps=curriculum.get_max_timesteps(), seed=args.seed,
                blueprint_generator=curriculum.get_blueprint_generator(),
                structure_name=curriculum.get_structure_name(),
                place_aim_gate=curriculum.get_place_aim_gate(),
            )
            _env_ref[0] = env
            prev_stage = curriculum.current_stage
            if use_wandb:
                import wandb
                wandb.log({"curriculum/stage_advanced_at": iteration, "curriculum/new_stage": curriculum.current_stage}, step=iteration)

        if iteration % args.save_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(path, iteration)
            save_checkpoint(os.path.join(args.checkpoint_dir, "latest.pt"), iteration)
            print(f"  Saved checkpoint: {path}")

        if completion_pct > best_completion:
            best_completion = completion_pct
            save_checkpoint(os.path.join(args.checkpoint_dir, "best_model.pt"), iteration)

    save_checkpoint(os.path.join(args.checkpoint_dir, "final_model.pt"), iteration)
    print(f"\nTraining complete. Best completion: {best_completion:.1%}")
    print(f"Resume with:  python ppo_train.py --resume {args.checkpoint_dir}/latest.pt ...")

    if blueprint_pool and len(blueprint_pool) > 0:
        print(f"\n{blueprint_pool.get_summary()}")

    if use_wandb:
        import wandb
        wandb.run.summary["best_completion"]  = best_completion
        wandb.run.summary["final_stage"]      = curriculum.current_stage
        wandb.run.summary["total_episodes"]   = curriculum.total_episodes
        best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            artifact = wandb.Artifact(name=f"ppo-builder-{wandb.run.id}", type="model",
                                      description=f"Best model (completion={best_completion:.1%})")
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)
        wandb.finish()

    env.close()
    _kill_java_children()


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Minecraft building agent with MLP-only PPO")

    # Environment
    parser.add_argument("--nbt",           type=str, default=None)
    parser.add_argument("--nbt-dir",       type=str, default=None)
    parser.add_argument("--swap-interval", type=int, default=10)
    parser.add_argument("--port",          type=int, default=8023)
    parser.add_argument("--seed",          type=int, default=12345)
    parser.add_argument("--image-size",    type=int, default=64)

    # Training
    parser.add_argument("--total-iterations", type=int,   default=10000)
    parser.add_argument("--rollout-steps",    type=int,   default=512)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--ppo-epochs",       type=int,   default=4)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--entropy-coef",     type=float, default=0.1)
    parser.add_argument("--hidden-dim",       type=int,   default=256)
    parser.add_argument("--lstm-hidden",      type=int,   default=128,  help="Ignored (MLP only — kept for arg compat)")

    # Curriculum
    parser.add_argument("--curriculum",   action="store_true")
    parser.add_argument("--start-stage",  type=int, default=0)

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--log-interval",   type=int, default=10)
    parser.add_argument("--save-interval",  type=int, default=100)

    # Other
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--debug-visual", action="store_true")

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=2)

    # W&B
    parser.add_argument("--wandb-project", type=str, default="craftground-builder-ppo")
    parser.add_argument("--wandb-entity",  type=str, default=None)
    parser.add_argument("--wandb-group",   type=str, default=None)
    parser.add_argument("--wandb-name",    type=str, default=None)
    parser.add_argument("--no-wandb",      action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
