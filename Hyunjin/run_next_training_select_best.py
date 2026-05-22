from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

HYUNJIN_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = HYUNJIN_ROOT / "archive" / "archive"
if str(ARCHIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_ROOT))

from cleanup_minecraft_processes import cleanup_craftground_processes

SAFE_MAX_ENVS = 2


@dataclass(frozen=True)
class Candidate:
    name: str
    seed: int
    learning_rate: float
    ent_coef: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run next-stage training candidates and auto-select the best improved model"
    )
    parser.add_argument("--run-prefix", default="s84_next")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--start-round-index", type=int, default=2)
    parser.add_argument("--end-round-index", type=int, default=6)
    parser.add_argument("--timesteps-scale", type=float, default=1.0)
    parser.add_argument("--steps-per-epoch", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--lidar-max-distance", type=float, default=10.0)
    parser.add_argument("--port-start-base", type=int, default=9400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python-bin", default="/home/hj/dsl/modeling/venv/bin/python")
    parser.add_argument("--save-root", default="artifacts/raycast_hunt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    return parser.parse_args()


def load_metrics(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def selection_score(row: dict[str, Any]) -> float:
    survival_norm = min(max(_f(row, "mean_survival_steps") / 900.0, 0.0), 1.0)
    hit_rate = min(max(_f(row, "mean_hit_rate"), 0.0), 1.0)
    hit_to_kill = min(max(_f(row, "mean_hit_to_kill"), 0.0), 1.0)
    kills_norm = min(max(_f(row, "mean_target_kills") / 1.5, 0.0), 1.0)
    wasted_norm = min(max(_f(row, "mean_wasted_shots") / 20.0, 0.0), 1.0)
    move_norm = min(max(_f(row, "mean_move_macro_count") / 120.0, 0.0), 1.0)
    idle_norm = min(max(_f(row, "mean_idle_steps") / 120.0, 0.0), 1.0)
    stationary_norm = min(max(_f(row, "mean_stationary_combat_steps") / 120.0, 0.0), 1.0)
    return (
        0.30 * survival_norm
        + 0.18 * kills_norm
        + 0.16 * hit_rate
        + 0.12 * hit_to_kill
        + 0.10 * move_norm
        - 0.08 * wasted_norm
        - 0.03 * idle_norm
        - 0.03 * stationary_norm
    )


def infer_baseline_score(init_checkpoint: Path) -> float | None:
    # expected: .../<run_name>/rXX_.../epoch_YYY.zip
    round_dir = init_checkpoint.parent
    run_dir = round_dir.parent
    global_csv = run_dir / "global_epoch_metrics.csv"
    rows = load_metrics(global_csv)
    if not rows:
        return None
    return selection_score(rows[-1])


def read_candidate_summary(run_dir: Path) -> dict[str, Any]:
    rows = load_metrics(run_dir / "global_epoch_metrics.csv")
    if not rows:
        return {
            "run_dir": str(run_dir),
            "status": "no_metrics",
            "best_score": -1e9,
            "final_score": -1e9,
        }
    best_row = max(rows, key=selection_score)
    final_row = rows[-1]
    return {
        "run_dir": str(run_dir),
        "status": "ok",
        "rows": len(rows),
        "best_score": selection_score(best_row),
        "final_score": selection_score(final_row),
        "best_row": best_row,
        "final_row": final_row,
    }


def cleanup_all(skip_cleanup: bool) -> dict[str, Any]:
    if skip_cleanup:
        return {"skipped": True}
    subprocess.run(["pkill", "-f", "train_craftground_raycast_curriculum.py"], check=False)
    return cleanup_craftground_processes(wait_seconds=5.0)


def build_candidates() -> list[Candidate]:
    return [
        Candidate(name="c1_balanced", seed=84, learning_rate=2e-4, ent_coef=3e-4),
        Candidate(name="c2_lowent", seed=101, learning_rate=1.5e-4, ent_coef=2e-4),
        Candidate(name="c3_stable", seed=202, learning_rate=1.0e-4, ent_coef=1.5e-4),
    ]


def run_candidate(args: argparse.Namespace, candidate: Candidate, run_name: str, port_start: int) -> int:
    cmd = [
        args.python_bin,
        str(HYUNJIN_ROOT / "train_craftground_raycast_curriculum.py"),
        "--run-name",
        run_name,
        "--seed",
        str(candidate.seed),
        "--learning-rate",
        str(candidate.learning_rate),
        "--ent-coef",
        str(candidate.ent_coef),
        "--start-round-index",
        str(args.start_round_index),
        "--end-round-index",
        str(args.end_round_index),
        "--init-checkpoint",
        str(Path(args.init_checkpoint).resolve()),
        "--timesteps-scale",
        str(args.timesteps_scale),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--num-envs",
        str(args.num_envs),
        "--eval-episodes",
        str(args.eval_episodes),
        "--lidar-max-distance",
        str(args.lidar_max_distance),
        "--port-start",
        str(port_start),
        "--device",
        str(args.device),
    ]
    print(json.dumps({"launch": run_name, "cmd": " ".join(shlex.quote(token) for token in cmd)}))
    if args.dry_run:
        return 0
    result = subprocess.run(cmd)
    return int(result.returncode)


def write_selection_outputs(selection_dir: Path, summary: dict[str, Any]) -> None:
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    best = summary.get("best_candidate")
    if not isinstance(best, dict):
        return
    best_run_dir = Path(best["summary"]["run_dir"])
    final_model = best_run_dir / "final_model.zip"
    if not final_model.exists():
        return
    promoted = selection_dir / "best_model.zip"
    if promoted.exists() or promoted.is_symlink():
        promoted.unlink()
    try:
        promoted.symlink_to(final_model)
    except OSError:
        shutil.copy2(final_model, promoted)


def main() -> None:
    args = parse_args()
    if args.num_envs > SAFE_MAX_ENVS:
        print(json.dumps({"warn": f"num_envs={args.num_envs} too high for WSL, capping to {SAFE_MAX_ENVS}"}))
        args.num_envs = SAFE_MAX_ENVS
    init_checkpoint = Path(args.init_checkpoint)
    if not init_checkpoint.is_absolute():
        init_checkpoint = (HYUNJIN_ROOT / init_checkpoint).resolve()
    if not init_checkpoint.exists() and not args.dry_run:
        raise SystemExit(f"init checkpoint not found: {init_checkpoint}")

    baseline = infer_baseline_score(init_checkpoint) if init_checkpoint.exists() else None

    print(json.dumps({"cleanup": cleanup_all(args.skip_cleanup)}, default=str))

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = build_candidates()
    results: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, start=1):
        run_name = f"{args.run_prefix}_{now_tag}_{candidate.name}"
        port_start = args.port_start_base + idx * 100
        if not args.skip_cleanup:
            print(json.dumps({"cleanup_before_candidate": cleanup_all(False)}, default=str))

        rc = run_candidate(args, candidate, run_name, port_start)
        run_dir = HYUNJIN_ROOT / args.save_root / run_name
        summary = read_candidate_summary(run_dir)
        improvement = None
        if baseline is not None and summary.get("status") == "ok":
            improvement = float(summary["best_score"]) - float(baseline)
        result_item = {
            "candidate": {
                "name": candidate.name,
                "seed": candidate.seed,
                "learning_rate": candidate.learning_rate,
                "ent_coef": candidate.ent_coef,
            },
            "run_name": run_name,
            "return_code": rc,
            "summary": summary,
            "improvement_vs_baseline": improvement,
        }
        results.append(result_item)
        print(json.dumps({"candidate_result": result_item}, default=str))

    best = None
    for item in results:
        if item["summary"].get("status") != "ok":
            continue
        score_key = (
            float(item.get("improvement_vs_baseline"))
            if item.get("improvement_vs_baseline") is not None
            else float(item["summary"].get("best_score", -1e9))
        )
        if best is None or score_key > best[0]:
            best = (score_key, item)

    selection = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_score": baseline,
        "args": vars(args),
        "results": results,
        "best_candidate": best[1] if best else None,
    }

    selection_dir = HYUNJIN_ROOT / args.save_root / f"selection_{now_tag}"
    write_selection_outputs(selection_dir, selection)
    print(json.dumps({"selection_dir": str(selection_dir), "best": selection.get("best_candidate")}, default=str))


if __name__ == "__main__":
    main()
