from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ModuleNotFoundError as exc:
    raise SystemExit(
        "tensorboard is not installed in this Python environment.\n"
        "Use the project venv Python:\n"
        "  /home/hj/dsl/modeling/venv/bin/python "
        "/home/hj/dsl/modeling/Hyunjin/monitor_raycast_training_speed.py\n"
        "Or install it in the current interpreter:\n"
        "  python3 -m pip install tensorboard"
    ) from exc


HYUNJIN_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = HYUNJIN_ROOT / "artifacts" / "raycast_hunt"
DEFAULT_TB_ROOT = HYUNJIN_ROOT / "runs" / "raycast_hunt"

BASE_CURRICULUM_TIMESTEPS = {
    "survival_bootstrap": 8_000,
    "hitfocus_recovery": 12_000,
    "duel_finishconvert": 12_000,
    "kite_pair_finish": 16_000,
    "clear_easy_powerfinish": 20_000,
    "generalize_combined": 24_000,
}

ROUND_ORDER = [
    "survival_bootstrap",
    "hitfocus_recovery",
    "duel_finishconvert",
    "kite_pair_finish",
    "clear_easy_powerfinish",
    "generalize_combined",
]
ROUND_COUNT = len(ROUND_ORDER)


@dataclass
class ScalarPoint:
    step: int
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor speed/progress for raycast curriculum training")
    parser.add_argument("--run-name", default="", help="Run name. If omitted, newest run is used.")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--tb-root", default=str(DEFAULT_TB_ROOT))
    parser.add_argument("--refresh-sec", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "table"],
        default="table",
        help="Output format for each snapshot",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Pretty print JSON output with indentation",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit compact summary JSON/text instead of full raw fields",
    )
    parser.add_argument(
        "--max-card-cols",
        type=int,
        default=4,
        help="Maximum number of cards per row for table output",
    )
    return parser.parse_args()


def choose_run_dir(artifact_root: Path, run_name: str) -> Path:
    if run_name.strip():
        run_dir = artifact_root / run_name.strip()
        if not run_dir.exists():
            raise SystemExit(f"Run not found: {run_dir}")
        return run_dir

    candidates = [path for path in artifact_root.iterdir() if path.is_dir()]
    if not candidates:
        raise SystemExit(f"No run directories under: {artifact_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_run_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def find_event_file(tb_root: Path, run_name: str) -> Path | None:
    if not tb_root.exists():
        return None
    # stable-baselines3 default: <run_name>_0/events.out.tfevents.*
    dirs = [p for p in tb_root.iterdir() if p.is_dir() and p.name.startswith(run_name)]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for tb_dir in dirs:
        event_files = sorted(tb_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if event_files:
            return event_files[0]
    return None


def latest_scalar(event_file: Path, tag: str) -> ScalarPoint | None:
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    scalar = acc.Scalars(tag)[-1]
    return ScalarPoint(step=int(scalar.step), value=float(scalar.value))


def read_latest_epoch_metrics(run_dir: Path) -> dict[str, Any] | None:
    csv_path = run_dir / "global_epoch_metrics.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return rows[-1]


def read_all_epoch_metrics(run_dir: Path) -> list[dict[str, Any]]:
    csv_path = run_dir / "global_epoch_metrics.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def count_epoch_rows(run_dir: Path) -> int:
    csv_path = run_dir / "global_epoch_metrics.csv"
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def _build_best_epoch_payload(best_score: float, best_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": round(best_score, 4),
        "round_id": best_row.get("round_id"),
        "stage_name": best_row.get("stage_name"),
        "epoch": best_row.get("epoch"),
        "global_timesteps": best_row.get("global_timesteps"),
        "mean_survival_steps": best_row.get("mean_survival_steps"),
        "mean_hit_rate": best_row.get("mean_hit_rate"),
        "mean_hit_to_kill": best_row.get("mean_hit_to_kill"),
        "mean_target_kills": best_row.get("mean_target_kills"),
        "mean_wasted_shots": best_row.get("mean_wasted_shots"),
        "mean_episode_reward": best_row.get("mean_episode_reward"),
    }


def compute_best_epoch(rows: list[dict[str, Any]], profile: str = "balanced") -> dict[str, Any] | None:
    if not rows:
        return None

    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidates.append(
            {
                "row": row,
                "survival": _to_float(row.get("mean_survival_steps"), 0.0),
                "hit_rate": _to_float(row.get("mean_hit_rate"), 0.0),
                "hit_to_kill": _to_float(row.get("mean_hit_to_kill"), 0.0),
                "kills": _to_float(row.get("mean_target_kills"), 0.0),
                "wasted": _to_float(row.get("mean_wasted_shots"), 0.0),
            }
        )

    mins = {
        "survival": min(item["survival"] for item in candidates),
        "hit_rate": min(item["hit_rate"] for item in candidates),
        "hit_to_kill": min(item["hit_to_kill"] for item in candidates),
        "kills": min(item["kills"] for item in candidates),
        "wasted": min(item["wasted"] for item in candidates),
    }
    maxs = {
        "survival": max(item["survival"] for item in candidates),
        "hit_rate": max(item["hit_rate"] for item in candidates),
        "hit_to_kill": max(item["hit_to_kill"] for item in candidates),
        "kills": max(item["kills"] for item in candidates),
        "wasted": max(item["wasted"] for item in candidates),
    }

    if profile == "survival":
        w_survival, w_hit_rate, w_hit_to_kill, w_kills, w_wasted = 0.55, 0.20, 0.10, 0.10, 0.15
    elif profile == "finish":
        w_survival, w_hit_rate, w_hit_to_kill, w_kills, w_wasted = 0.15, 0.20, 0.35, 0.25, 0.15
    else:
        w_survival, w_hit_rate, w_hit_to_kill, w_kills, w_wasted = 0.35, 0.20, 0.20, 0.15, 0.10

    best_item = None
    best_score = -1e18
    for item in candidates:
        score = (
            w_survival * _norm(item["survival"], mins["survival"], maxs["survival"])
            + w_hit_rate * _norm(item["hit_rate"], mins["hit_rate"], maxs["hit_rate"])
            + w_hit_to_kill * _norm(item["hit_to_kill"], mins["hit_to_kill"], maxs["hit_to_kill"])
            + w_kills * _norm(item["kills"], mins["kills"], maxs["kills"])
            - w_wasted * _norm(item["wasted"], mins["wasted"], maxs["wasted"])
        )
        if score > best_score:
            best_score = score
            best_item = item

    if not best_item:
        return None

    return _build_best_epoch_payload(best_score, best_item["row"])


def _selected_round_names(meta: dict[str, Any]) -> list[str]:
    args = meta.get("args", {}) if isinstance(meta, dict) else {}
    try:
        start_idx = int(args.get("start_round_index", 1))
    except Exception:
        start_idx = 1
    try:
        end_idx = int(args.get("end_round_index", 0))
    except Exception:
        end_idx = 0

    if start_idx < 1:
        start_idx = 1
    if end_idx <= 0:
        end_idx = ROUND_COUNT
    if end_idx > ROUND_COUNT:
        end_idx = ROUND_COUNT
    if end_idx < start_idx:
        start_idx, end_idx = 1, ROUND_COUNT
    return ROUND_ORDER[start_idx - 1 : end_idx]


def planned_total_timesteps(meta: dict[str, Any]) -> int:
    args = meta.get("args", {}) if isinstance(meta, dict) else {}
    scale = float(args.get("timesteps_scale", 1.0))
    selected = _selected_round_names(meta)
    base = sum(BASE_CURRICULUM_TIMESTEPS[name] for name in selected)
    return int(round(base * scale))


def planned_total_epochs(meta: dict[str, Any]) -> int:
    args = meta.get("args", {}) if isinstance(meta, dict) else {}
    scale = float(args.get("timesteps_scale", 1.0))
    steps_per_epoch = int(args.get("steps_per_epoch", 2048))
    if steps_per_epoch <= 0:
        steps_per_epoch = 2048
    total = 0
    for round_name in _selected_round_names(meta):
        ts = max(512, int(round(BASE_CURRICULUM_TIMESTEPS[round_name] * scale)))
        total += math.ceil(ts / steps_per_epoch)
    return total


def find_training_process(run_name: str) -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "ps",
                "-eo",
                "pid=,etimes=,pcpu=,pmem=,args=",
            ],
            text=True,
        )
    except Exception:
        return None
    candidates: list[dict[str, Any]] = []
    for line in out.splitlines():
        if "train_craftground_raycast_curriculum.py" not in line:
            continue
        if f"--run-name {run_name}" not in line:
            continue
        parts = line.strip().split(maxsplit=4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            etimes = int(parts[1])
            pcpu = float(parts[2])
            pmem = float(parts[3])
        except ValueError:
            continue
        candidates.append(
            {
            "pid": pid,
            "elapsed_sec": etimes,
            "cpu_percent": pcpu,
            "mem_percent": pmem,
            "args": parts[4],
            }
        )
    if not candidates:
        return None
    # Prefer the actual python trainer process over shell wrapper commands.
    candidates.sort(
        key=lambda item: (
            0 if "/python" in item["args"] else 1,
            -item["cpu_percent"],
        )
    )
    return candidates[0]


def fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return "-"
    return str(timedelta(seconds=int(seconds)))


def snapshot(run_dir: Path, tb_root: Path) -> dict[str, Any]:
    run_name = run_dir.name
    meta = read_run_meta(run_dir)
    event_file = find_event_file(tb_root, run_name)

    fps = timesteps = elapsed = ep_rew = ep_len = None
    if event_file is not None:
        p = latest_scalar(event_file, "time/fps")
        if p:
            fps = p.value
        p = latest_scalar(event_file, "time/total_timesteps")
        if p:
            timesteps = p.value
        p = latest_scalar(event_file, "time/time_elapsed")
        if p:
            elapsed = p.value
        p = latest_scalar(event_file, "rollout/ep_rew_mean")
        if p:
            ep_rew = p.value
        p = latest_scalar(event_file, "rollout/ep_len_mean")
        if p:
            ep_len = p.value

    planned = planned_total_timesteps(meta)
    latest_epoch = read_latest_epoch_metrics(run_dir)
    all_epoch_rows = read_all_epoch_metrics(run_dir)
    best_epoch_balanced = compute_best_epoch(all_epoch_rows, profile="balanced")
    best_epoch_survival = compute_best_epoch(all_epoch_rows, profile="survival")
    best_epoch_finish = compute_best_epoch(all_epoch_rows, profile="finish")
    completed_epochs = count_epoch_rows(run_dir)
    total_epochs = planned_total_epochs(meta)
    proc = find_training_process(run_name)
    latest_global_timesteps = None
    if latest_epoch and latest_epoch.get("global_timesteps"):
        try:
            latest_global_timesteps = float(latest_epoch["global_timesteps"])
        except Exception:
            latest_global_timesteps = None

    effective_timesteps = timesteps if timesteps is not None else latest_global_timesteps
    progress = (float(effective_timesteps) / planned) if effective_timesteps is not None and planned > 0 else None

    remaining_steps = None
    if effective_timesteps is not None:
        remaining_steps = max(planned - int(effective_timesteps), 0)

    eta = None
    eta_source = None
    if fps and effective_timesteps is not None and fps > 0:
        eta = (planned - float(effective_timesteps)) / float(fps)
        eta_source = "tensorboard_fps"
    elif proc and effective_timesteps is not None and proc.get("elapsed_sec", 0) > 0:
        pseudo_fps = float(effective_timesteps) / float(proc["elapsed_sec"])
        if pseudo_fps > 0:
            eta = (planned - float(effective_timesteps)) / pseudo_fps
            eta_source = "process_elapsed_estimate"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "event_file": str(event_file) if event_file else "(none)",
        "planned_total_timesteps": planned,
        "time_total_timesteps": timesteps,
        "effective_total_timesteps": effective_timesteps,
        "time_fps": fps,
        "time_elapsed_sec": elapsed,
        "rollout_ep_rew_mean": ep_rew,
        "rollout_ep_len_mean": ep_len,
        "progress_ratio": progress,
        "eta_sec": eta,
        "remaining_steps": remaining_steps,
        "eta_source": eta_source,
        "completed_epochs": completed_epochs,
        "planned_epochs": total_epochs,
        "remaining_epochs": max(total_epochs - completed_epochs, 0),
        "process": proc,
        "latest_epoch_metrics": latest_epoch,
        "best_epoch_metrics_balanced": best_epoch_balanced,
        "best_epoch_metrics_survival": best_epoch_survival,
        "best_epoch_metrics_finish": best_epoch_finish,
    }


def print_snapshot(data: dict[str, Any]) -> None:
    print(f"[{data['timestamp']}] run={data['run_name']}")
    print(f"run_dir: {data['run_dir']}")
    print(f"event_file: {data['event_file']}")
    print(
        "speed: fps={fps} timesteps={steps}/{planned} elapsed={elapsed}s eta={eta}".format(
            fps=(f"{data['time_fps']:.2f}" if data["time_fps"] is not None else "-"),
            steps=(f"{int(data['time_total_timesteps'])}" if data["time_total_timesteps"] is not None else "-"),
            planned=int(data["planned_total_timesteps"]),
            elapsed=(f"{int(data['time_elapsed_sec'])}" if data["time_elapsed_sec"] is not None else "-"),
            eta=fmt_eta(data["eta_sec"]),
        )
    )
    process = data.get("process")
    if process:
        print(
            "process: pid={pid} cpu={cpu:.1f}% mem={mem:.1f}% etime={etime}".format(
                pid=process["pid"],
                cpu=process["cpu_percent"],
                mem=process["mem_percent"],
                etime=fmt_eta(process["elapsed_sec"]),
            )
        )
    else:
        print("process: (not found)")
    if data["progress_ratio"] is not None:
        print(f"progress: {100.0 * float(data['progress_ratio']):.2f}%")
    if data["remaining_steps"] is not None:
        print(f"remaining_steps: {int(data['remaining_steps'])}")
    print(
        "epochs: completed={done}/{total} remaining={remain}".format(
            done=int(data["completed_epochs"]),
            total=int(data["planned_epochs"]),
            remain=int(data["remaining_epochs"]),
        )
    )
    if data["rollout_ep_rew_mean"] is not None or data["rollout_ep_len_mean"] is not None:
        print(
            "rollout: ep_rew_mean={rew} ep_len_mean={length}".format(
                rew=(f"{data['rollout_ep_rew_mean']:.3f}" if data["rollout_ep_rew_mean"] is not None else "-"),
                length=(f"{data['rollout_ep_len_mean']:.2f}" if data["rollout_ep_len_mean"] is not None else "-"),
            )
        )

    latest = data.get("latest_epoch_metrics")
    if latest:
        print(
            "latest_epoch: round={round_id} epoch={epoch} hit_rate={hit_rate} kill={kills} survival={surv}".format(
                round_id=latest.get("round_id", "-"),
                epoch=latest.get("epoch", "-"),
                hit_rate=latest.get("mean_hit_rate", "-"),
                kills=latest.get("mean_target_kills", "-"),
                surv=latest.get("mean_survival_steps", "-"),
            )
        )
    print("-" * 72)


def print_json_snapshot(data: dict[str, Any], pretty: bool) -> None:
    if pretty:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(data, ensure_ascii=False))


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _render_kv_table(title: str, rows: list[tuple[str, Any]]) -> None:
    key_width = max([len(k) for k, _ in rows] + [3])
    val_width = max([len(_fmt(v)) for _, v in rows] + [5])
    line = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"
    print(f"[{title}]")
    print(line)
    print(f"| {'key'.ljust(key_width)} | {'value'.ljust(val_width)} |")
    print(line)
    for k, v in rows:
        print(f"| {k.ljust(key_width)} | {_fmt(v).ljust(val_width)} |")
    print(line)


def _build_card(title: str, rows: list[tuple[str, Any]], min_width: int = 34) -> list[str]:
    content = [f"{k}: {_fmt(v)}" for k, v in rows]
    width = max([len(title)] + [len(line) for line in content] + [min_width])
    border = "+" + "-" * (width + 2) + "+"
    lines = [border, f"| {title.ljust(width)} |", border]
    for line in content:
        lines.append(f"| {line.ljust(width)} |")
    lines.append(border)
    return lines


def _print_card_row(cards: list[list[str]], spacing: int = 2) -> None:
    if not cards:
        return
    max_lines = max(len(card) for card in cards)
    widths = [len(card[0]) for card in cards]
    for i in range(max_lines):
        parts: list[str] = []
        for c_idx, card in enumerate(cards):
            line = card[i] if i < len(card) else " " * widths[c_idx]
            parts.append(line)
        print((" " * spacing).join(parts))


def _chunk_cards(cards: list[list[str]], cols: int) -> list[list[list[str]]]:
    if cols <= 0:
        cols = 1
    return [cards[i : i + cols] for i in range(0, len(cards), cols)]


def _card_row_width(cards: list[list[str]], spacing: int = 2) -> int:
    if not cards:
        return 0
    return sum(len(card[0]) for card in cards) + spacing * (len(cards) - 1)


def _choose_cols(cards: list[list[str]], term_width: int, max_cols: int = 4) -> int:
    max_cols = max(1, min(max_cols, len(cards)))
    for cols in range(max_cols, 0, -1):
        rows = _chunk_cards(cards, cols)
        if all(_card_row_width(row) <= term_width for row in rows):
            return cols
    return 1


def print_table_summary(data: dict[str, Any], max_card_cols: int = 4) -> None:
    print(f"[{data.get('timestamp', '-')}] run={data.get('run_name', '-')}")
    term_width = shutil.get_terminal_size((140, 40)).columns

    proc = data.get("process", {}) or {}
    latest = data.get("latest_epoch", {}) or {}
    b_surv = data.get("best_epoch_survival", {}) or {}
    b_fin = data.get("best_epoch_finish", {}) or {}

    progress_card = _build_card(
        "Progress",
        [
            ("overall_progress_%", data.get("overall_progress_percent")),
            ("global_timesteps", data.get("global_timesteps")),
            ("planned_timesteps", data.get("planned_total_timesteps")),
            ("remaining_steps", data.get("remaining_steps")),
            ("completed_epochs", data.get("completed_epochs")),
            ("planned_epochs", data.get("planned_epochs")),
            ("remaining_epochs", data.get("remaining_epochs")),
            ("eta", data.get("eta")),
            ("eta_source", data.get("eta_source")),
        ],
    )
    process_card = _build_card(
        "Process",
        [
            ("running", proc.get("running")),
            ("pid", proc.get("pid")),
            ("cpu_%", proc.get("cpu_percent")),
            ("mem_%", proc.get("mem_percent")),
        ],
    )
    latest_card = _build_card(
        "Latest Epoch",
        [
            ("round_id", latest.get("round_id")),
            ("stage_name", latest.get("stage_name")),
            ("epoch", latest.get("epoch")),
            ("global_ts", latest.get("global_timesteps")),
            ("survival", latest.get("mean_survival_steps")),
            ("hit_rate", latest.get("mean_hit_rate")),
            ("hit_to_kill", latest.get("mean_hit_to_kill")),
            ("kills", latest.get("mean_target_kills")),
            ("wasted_shots", latest.get("mean_wasted_shots")),
        ],
    )
    best_surv_card = _build_card(
        "Best Epoch (Survival)",
        [
            ("round_id", b_surv.get("round_id")),
            ("stage_name", b_surv.get("stage_name")),
            ("epoch", b_surv.get("epoch")),
            ("global_ts", b_surv.get("global_timesteps")),
            ("score", b_surv.get("score")),
            ("survival", b_surv.get("mean_survival_steps")),
            ("hit_rate", b_surv.get("mean_hit_rate")),
            ("hit_to_kill", b_surv.get("mean_hit_to_kill")),
            ("kills", b_surv.get("mean_target_kills")),
            ("wasted_shots", b_surv.get("mean_wasted_shots")),
        ],
    )
    best_finish_card = _build_card(
        "Best Epoch (Finish)",
        [
            ("round_id", b_fin.get("round_id")),
            ("stage_name", b_fin.get("stage_name")),
            ("epoch", b_fin.get("epoch")),
            ("global_ts", b_fin.get("global_timesteps")),
            ("score", b_fin.get("score")),
            ("survival", b_fin.get("mean_survival_steps")),
            ("hit_rate", b_fin.get("mean_hit_rate")),
            ("hit_to_kill", b_fin.get("mean_hit_to_kill")),
            ("kills", b_fin.get("mean_target_kills")),
            ("wasted_shots", b_fin.get("mean_wasted_shots")),
        ],
    )

    cards = [progress_card, process_card, latest_card, best_surv_card, best_finish_card]
    cols = _choose_cols(cards, term_width, max_cols=max_card_cols)
    for row_cards in _chunk_cards(cards, cols):
        _print_card_row(row_cards)


def to_summary(data: dict[str, Any]) -> dict[str, Any]:
    latest = data.get("latest_epoch_metrics") or {}
    progress_ratio = data.get("progress_ratio")
    progress_percent = None if progress_ratio is None else round(float(progress_ratio) * 100.0, 2)
    return {
        "timestamp": data.get("timestamp"),
        "run_name": data.get("run_name"),
        "overall_progress_percent": progress_percent,
        "global_timesteps": data.get("effective_total_timesteps"),
        "planned_total_timesteps": data.get("planned_total_timesteps"),
        "remaining_steps": data.get("remaining_steps"),
        "eta": fmt_eta(data.get("eta_sec")),
        "eta_source": data.get("eta_source"),
        "completed_epochs": data.get("completed_epochs"),
        "planned_epochs": data.get("planned_epochs"),
        "remaining_epochs": data.get("remaining_epochs"),
        "process": {
            "running": data.get("process") is not None,
            "pid": (data.get("process") or {}).get("pid"),
            "cpu_percent": (data.get("process") or {}).get("cpu_percent"),
            "mem_percent": (data.get("process") or {}).get("mem_percent"),
        },
        "latest_epoch": {
            "round_id": latest.get("round_id"),
            "stage_name": latest.get("stage_name"),
            "epoch": latest.get("epoch"),
            "global_timesteps": latest.get("global_timesteps"),
            "mean_survival_steps": latest.get("mean_survival_steps"),
            "mean_hit_rate": latest.get("mean_hit_rate"),
            "mean_hit_to_kill": latest.get("mean_hit_to_kill"),
            "mean_target_kills": latest.get("mean_target_kills"),
            "mean_wasted_shots": latest.get("mean_wasted_shots"),
        },
        "best_epoch_balanced": data.get("best_epoch_metrics_balanced"),
        "best_epoch_survival": data.get("best_epoch_metrics_survival"),
        "best_epoch_finish": data.get("best_epoch_metrics_finish"),
    }


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    tb_root = Path(args.tb_root)

    run_dir = choose_run_dir(artifact_root, args.run_name)

    while True:
        data = snapshot(run_dir, tb_root)
        if args.summary_only:
            data = to_summary(data)
        if args.output_format == "json":
            print_json_snapshot(data, pretty=args.pretty_json)
        elif args.output_format == "table":
            print_table_summary(data, max_card_cols=max(1, int(args.max_card_cols)))
        else:
            print_snapshot(data)
        if args.once:
            return
        time.sleep(max(args.refresh_sec, 0.5))


if __name__ == "__main__":
    main()
