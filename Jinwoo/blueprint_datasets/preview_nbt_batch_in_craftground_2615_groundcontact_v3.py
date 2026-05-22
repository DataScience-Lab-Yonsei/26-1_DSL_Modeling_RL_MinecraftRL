#!/usr/bin/env python3
"""
Batch preview multiple .nbt structures in CraftGround 2.6.15.

What it does
------------
- Scans a directory (or a .zip package) for .nbt files
- For each structure:
  - launches CraftGround
  - moves the structure so its bottom sits on the existing superflat ground
  - captures four exterior views:
    preview_01.png (front)
    preview_02.png (left)
    preview_03.png (back)
    preview_04.png (right)
- Saves each structure into its own subdirectory under --out-dir
- Overwrites --out-dir if it already exists
- Writes batch_manifest.json with success / failure information

Notes
-----
- This is intentionally conservative: one CraftGround env per structure.
  That is slower, but avoids structure-path reconfiguration issues.
- Designed for CraftGround 2.6.15 style API:
    craftground.make(initial_env_config=InitialEnvironmentConfig(...), ...)
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import shutil
import struct
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple

# ---------------------------------------------------------------------
# Disable audio before importing CraftGround to reduce ALSA/OpenAL issues.
# ---------------------------------------------------------------------
os.environ.setdefault("ALSOFT_DRIVERS", "null")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("AUDIODEV", "null")
os.environ.setdefault("ALSOFT_LOGLEVEL", "0")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import craftground
from craftground import InitialEnvironmentConfig, ActionSpaceVersion
from craftground.initial_environment_config import (
    DaylightMode,
    GameMode,
    WorldType,
)
from craftground.screen_encoding_modes import ScreenEncodingMode

try:
    from craftground.minecraft import no_op_v2
except Exception:
    from craftground.environment.action_space import no_op_v2


# =========================
# Minimal NBT parser
# =========================

TAG_End = 0
TAG_Byte = 1
TAG_Short = 2
TAG_Int = 3
TAG_Long = 4
TAG_Float = 5
TAG_Double = 6
TAG_Byte_Array = 7
TAG_String = 8
TAG_List = 9
TAG_Compound = 10
TAG_Int_Array = 11
TAG_Long_Array = 12


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"Expected {n} bytes, got {len(b)}")
    return b


def _read_u8(f: BinaryIO) -> int:
    return struct.unpack(">B", _read_exact(f, 1))[0]


def _read_i8(f: BinaryIO) -> int:
    return struct.unpack(">b", _read_exact(f, 1))[0]


def _read_i16(f: BinaryIO) -> int:
    return struct.unpack(">h", _read_exact(f, 2))[0]


def _read_i32(f: BinaryIO) -> int:
    return struct.unpack(">i", _read_exact(f, 4))[0]


def _read_i64(f: BinaryIO) -> int:
    return struct.unpack(">q", _read_exact(f, 8))[0]


def _read_f32(f: BinaryIO) -> float:
    return struct.unpack(">f", _read_exact(f, 4))[0]


def _read_f64(f: BinaryIO) -> float:
    return struct.unpack(">d", _read_exact(f, 8))[0]


def _read_string(f: BinaryIO) -> str:
    n = _read_i16(f)
    return _read_exact(f, n).decode("utf-8")


def _read_payload(f: BinaryIO, tag_type: int):
    if tag_type == TAG_Byte:
        return _read_i8(f)
    if tag_type == TAG_Short:
        return _read_i16(f)
    if tag_type == TAG_Int:
        return _read_i32(f)
    if tag_type == TAG_Long:
        return _read_i64(f)
    if tag_type == TAG_Float:
        return _read_f32(f)
    if tag_type == TAG_Double:
        return _read_f64(f)
    if tag_type == TAG_Byte_Array:
        n = _read_i32(f)
        return list(_read_exact(f, n))
    if tag_type == TAG_String:
        return _read_string(f)
    if tag_type == TAG_List:
        inner = _read_u8(f)
        n = _read_i32(f)
        return [_read_payload(f, inner) for _ in range(n)]
    if tag_type == TAG_Compound:
        out = {}
        while True:
            t = _read_u8(f)
            if t == TAG_End:
                break
            name = _read_string(f)
            out[name] = _read_payload(f, t)
        return out
    if tag_type == TAG_Int_Array:
        n = _read_i32(f)
        return [_read_i32(f) for _ in range(n)]
    if tag_type == TAG_Long_Array:
        n = _read_i32(f)
        return [_read_i64(f) for _ in range(n)]
    raise ValueError(f"Unsupported NBT tag type: {tag_type}")


def load_nbt_root(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as gz:
        data = gz.read()
    f = io.BytesIO(data)
    root_type = _read_u8(f)
    if root_type != TAG_Compound:
        raise ValueError("NBT root is not a compound.")
    _root_name = _read_string(f)
    root_payload = _read_payload(f, TAG_Compound)
    if not isinstance(root_payload, dict):
        raise ValueError("Parsed NBT root payload is not a compound dict.")
    return root_payload


@dataclass
class StructureInfo:
    size_x: int
    size_y: int
    size_z: int
    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int


def parse_structure_info(nbt_path: Path) -> StructureInfo:
    root = load_nbt_root(nbt_path)

    size = root.get("size", [1, 1, 1])
    if len(size) != 3:
        raise ValueError("NBT structure 'size' is malformed.")

    blocks = root.get("blocks", [])
    if not blocks:
        return StructureInfo(
            int(size[0]), int(size[1]), int(size[2]),
            0, 0, 0,
            int(size[0]) - 1, int(size[1]) - 1, int(size[2]) - 1,
        )

    xs, ys, zs = [], [], []
    for b in blocks:
        pos = b.get("pos", [0, 0, 0])
        if len(pos) != 3:
            continue
        xs.append(int(pos[0]))
        ys.append(int(pos[1]))
        zs.append(int(pos[2]))

    if not xs:
        return StructureInfo(
            int(size[0]), int(size[1]), int(size[2]),
            0, 0, 0,
            int(size[0]) - 1, int(size[1]) - 1, int(size[2]) - 1,
        )

    return StructureInfo(
        size_x=int(size[0]),
        size_y=int(size[1]),
        size_z=int(size[2]),
        min_x=min(xs), min_y=min(ys), min_z=min(zs),
        max_x=max(xs), max_y=max(ys), max_z=max(zs),
    )


# =========================
# CraftGround helpers
# =========================

def safe_name(text: str) -> str:
    name = text.lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "structure"


def copy_structure_to_workdir(src: Path, work_dir: Path) -> tuple[Path, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    structure_name = safe_name(src.stem)
    dst = work_dir / f"{structure_name}.nbt"
    shutil.copy2(src, dst)
    return dst, structure_name


def compute_world_origin(args: argparse.Namespace, info: StructureInfo) -> tuple[int, int, int]:
    world_x = args.x
    world_z = args.z
    world_y = args.ground_y - info.min_y
    return world_x, world_y, world_z


def build_world_commands(
    structure_name: str,
    info: StructureInfo,
    world_origin: tuple[int, int, int],
    tp: tuple[float, float, float, float, float],
    base_margin: int,
) -> list[str]:
    x0, y0, z0 = world_origin
    px, py, pz, yaw, pitch = tp

    # NOTE:
    # We intentionally do NOT create a floating grass/dirt pad here.
    # The structure itself is moved so that its lowest occupied block
    # touches the existing superflat ground.
    cmds = [
        "time set day",
        "gamerule doDaylightCycle false",
        "weather clear",
        "gamerule doWeatherCycle false",
        "gamerule doMobSpawning false",
        f"place template minecraft:{structure_name} {x0} {y0} {z0}",
        f"tp @p {px:.2f} {py:.2f} {pz:.2f} {yaw:.2f} {pitch:.2f}",
    ]
    return cmds


def compute_view_teleport(
    view_name: str,
    args: argparse.Namespace,
    info: StructureInfo,
    world_origin: tuple[int, int, int],
) -> tuple[float, float, float, float, float]:
    """
    Returns (x, y, z, yaw, pitch) for four cardinal exterior views.

    Minecraft yaw convention:
    - 0   : south
    - 90  : west
    - 180 : north
    - -90 : east
    """
    x0, y0, z0 = world_origin

    occ_min_x = x0 + info.min_x
    occ_max_x = x0 + info.max_x
    occ_min_y = y0 + info.min_y
    occ_max_y = y0 + info.max_y
    occ_min_z = z0 + info.min_z
    occ_max_z = z0 + info.max_z

    cx = (occ_min_x + occ_max_x) / 2.0
    cz = (occ_min_z + occ_max_z) / 2.0

    span_x = occ_max_x - occ_min_x + 1
    span_z = occ_max_z - occ_min_z + 1
    span_y = occ_max_y - occ_min_y + 1

    dist = max(span_x, span_z) * args.distance_scale + args.distance_bias
    eye_y = occ_min_y + max(3.0, span_y * args.eye_height_scale) + args.eye_height_bias
    pitch = args.pitch

    if view_name == "front":
        # South side looking north
        return (cx, eye_y, occ_max_z + dist, 180.0, pitch)
    if view_name == "left":
        # West side looking east
        return (occ_min_x - dist, eye_y, cz, -90.0, pitch)
    if view_name == "back":
        # North side looking south
        return (cx, eye_y, occ_min_z - dist, 0.0, pitch)
    if view_name == "right":
        # East side looking west
        return (occ_max_x + dist, eye_y, cz, 90.0, pitch)

    raise ValueError(f"Unknown view name: {view_name}")


def build_initial_env_config(
    args: argparse.Namespace,
    structure_file: Path,
    structure_name: str,
    info: StructureInfo,
) -> InitialEnvironmentConfig:
    world_origin = compute_world_origin(args, info)
    initial_tp = compute_view_teleport("front", args, info, world_origin)
    initial_cmds = build_world_commands(
        structure_name=structure_name,
        info=info,
        world_origin=world_origin,
        tp=initial_tp,
        base_margin=args.base_margin,
    )

    cfg = InitialEnvironmentConfig(
        image_width=args.width,
        image_height=args.height,
        gamemode=GameMode.CREATIVE,
        world_type=WorldType.SUPERFLAT,
        seed=str(args.seed),
        generate_structures=False,
        initial_extra_commands=initial_cmds,
        misc_stat_keys=[],
        hud_hidden=True,
        render_distance=args.render_distance,
        simulation_distance=args.simulation_distance,
        structure_paths=[str(structure_file.resolve())],
        no_fov_effect=True,
        screen_encoding_mode=ScreenEncodingMode.RAW,
    ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY)

    cfg.set_allow_mob_spawn(False)
    cfg.freeze_weather(True)
    return cfg


def make_env(
    args: argparse.Namespace,
    structure_file: Path,
    structure_name: str,
    info: StructureInfo,
):
    initial_env_config = build_initial_env_config(args, structure_file, structure_name, info)
    env = craftground.make(
        initial_env_config=initial_env_config,
        port=args.port,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        use_vglrun=args.use_vglrun,
        render_action=True,
        cleanup_world=True,
        verbose_python=args.verbose_python,
        verbose_gradle=args.verbose_gradle,
        verbose_jvm=args.verbose_jvm,
    )
    return env


def extract_rgb(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        for key in ("rgb", "pov", "image"):
            if key in obs:
                arr = np.asarray(obs[key])
                if arr.ndim == 3:
                    return arr
        if "full" in obs:
            full = obs["full"]
            for attr in ("rgb", "pov", "image"):
                if hasattr(full, attr):
                    arr = np.asarray(getattr(full, attr))
                    if arr.ndim == 3:
                        return arr
    arr = np.asarray(obs)
    if arr.ndim == 3:
        return arr
    raise RuntimeError("Could not extract an RGB observation from CraftGround output.")


def save_rgb(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def reset_env(env, *, extra_commands: list[str] | None = None):
    options = {"fast_reset": True}
    if extra_commands:
        options["extra_commands"] = extra_commands
    out = env.reset(options=options)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def reset_env_initial(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def step_noop(env):
    action = no_op_v2()
    out = env.step(action)
    return out[0]


def terminate_env(env):
    for method_name in ("terminate", "close"):
        method = getattr(env, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


# =========================
# Batch utilities
# =========================

@dataclass
class StructureResult:
    source_nbt: str
    structure_name: str
    success: bool
    error: str | None
    size: List[int] | None
    occupied_min: List[int] | None
    occupied_max: List[int] | None
    world_origin: List[int] | None
    output_dir: str
    preview_files: List[str]


def maybe_extract_input(input_path: Path, temp_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = temp_dir / "extracted_input"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir
    raise FileNotFoundError(f"Input must be a directory or .zip file: {input_path}")


def discover_nbt_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*.nbt"))


def build_contact_sheet(per_structure_dirs: List[Path], output_path: Path, thumb_w: int = 320) -> None:
    previews = []
    labels = []
    for d in per_structure_dirs:
        front = d / "preview_01.png"
        if front.exists():
            previews.append(front)
            labels.append(d.name)

    if not previews:
        return

    images = []
    font = ImageFont.load_default()
    for img_path, label in zip(previews, labels):
        img = Image.open(img_path).convert("RGB")
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, max(1, int(img.height * ratio))))
        canvas = Image.new("RGB", (thumb.width, thumb.height + 26), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, thumb.height + 6), label, fill="black", font=font)
        images.append(canvas)

    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    max_h = max(im.height for im in images)
    cell_w = thumb_w
    pad = 16

    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * max_h + (rows + 1) * pad), "white")
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (max_h + pad)
        sheet.paste(im, (x, y))
    sheet.save(output_path)


def preview_one_structure(
    src_nbt: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> StructureResult:
    structure_out = out_dir / safe_name(src_nbt.stem)
    structure_out.mkdir(parents=True, exist_ok=True)
    work_dir = structure_out / "_structure_work"

    env = None
    structure_name = safe_name(src_nbt.stem)
    try:
        structure_file, structure_name = copy_structure_to_workdir(src_nbt, work_dir)
        info = parse_structure_info(structure_file)
        world_origin = compute_world_origin(args, info)

        if args.verbose_python or args.verbose_gradle or args.verbose_jvm:
            print(f"\n=== Previewing {src_nbt.name} ===")
            print(f"  template: minecraft:{structure_name}")
            print(f"  world_origin: {world_origin}")
            print(f"  size: {[info.size_x, info.size_y, info.size_z]}")
            print(f"  occupied_min: {[info.min_x, info.min_y, info.min_z]}")
            print(f"  occupied_max: {[info.max_x, info.max_y, info.max_z]}")

        env = make_env(args, structure_file, structure_name, info)

        # Boot reset: discard this loading frame
        reset_env_initial(env)
        time.sleep(args.sleep_seconds)

        saved = []
        world_origin = compute_world_origin(args, info)
        for idx, view_name in enumerate(["front", "left", "back", "right"], start=1):
            tp = compute_view_teleport(view_name, args, info, world_origin)
            commands = build_world_commands(
                structure_name=structure_name,
                info=info,
                world_origin=world_origin,
                tp=tp,
                base_margin=args.base_margin,
            )
            obs, _ = reset_env(env, extra_commands=commands)
            time.sleep(args.sleep_seconds)
            for _ in range(args.warmup_steps):
                obs = step_noop(env)

            save_path = structure_out / f"preview_{idx:02d}.png"
            save_rgb(save_path, extract_rgb(obs))
            saved.append(save_path.name)

        return StructureResult(
            source_nbt=str(src_nbt),
            structure_name=structure_name,
            success=True,
            error=None,
            size=[info.size_x, info.size_y, info.size_z],
            occupied_min=[info.min_x, info.min_y, info.min_z],
            occupied_max=[info.max_x, info.max_y, info.max_z],
            world_origin=list(world_origin),
            output_dir=str(structure_out),
            preview_files=saved,
        )

    except Exception as e:
        return StructureResult(
            source_nbt=str(src_nbt),
            structure_name=structure_name,
            success=False,
            error=repr(e),
            size=None,
            occupied_min=None,
            occupied_max=None,
            world_origin=None,
            output_dir=str(structure_out),
            preview_files=[],
        )
    finally:
        if env is not None:
            terminate_env(env)


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Directory or .zip containing .nbt files")
    p.add_argument("--out-dir", required=True, help="Output directory (overwritten if exists)")
    p.add_argument("--port-start", type=int, default=8001, help="Starting port; each structure uses port-start + idx")
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument("--x", type=int, default=8)
    p.add_argument("--z", type=int, default=8)
    p.add_argument("--ground-y", type=int, default=-60)
    p.add_argument("--base-margin", type=int, default=2)

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--render-distance", type=int, default=8)
    p.add_argument("--simulation-distance", type=int, default=8)

    p.add_argument("--distance-scale", type=float, default=0.55)
    p.add_argument("--distance-bias", type=float, default=1.5)
    p.add_argument("--eye-height-scale", type=float, default=0.52)
    p.add_argument("--eye-height-bias", type=float, default=1.0)
    p.add_argument("--pitch", type=float, default=7.0)

    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--sleep-seconds", type=float, default=0.4)

    p.add_argument("--max-files", type=int, default=0, help="0 means no limit")
    p.add_argument("--sample-random", action="store_true")
    p.add_argument("--seed-random", type=int, default=42)

    p.add_argument("--make-contact-sheet", action="store_true")
    p.add_argument("--use-vglrun", action="store_true")
    p.add_argument("--verbose-python", action="store_true")
    p.add_argument("--verbose-gradle", action="store_true")
    p.add_argument("--verbose-jvm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "_batch_temp"
    input_root = maybe_extract_input(input_path, temp_dir)
    nbt_files = discover_nbt_files(input_root)
    if not nbt_files:
        raise FileNotFoundError(f"No .nbt files found under: {input_root}")

    if args.sample_random:
        rng = random.Random(args.seed_random)
        rng.shuffle(nbt_files)

    if args.max_files > 0:
        nbt_files = nbt_files[:args.max_files]

    print("Audio env:")
    for key in ("ALSOFT_DRIVERS", "SDL_AUDIODRIVER", "AUDIODEV", "ALSOFT_LOGLEVEL"):
        print(f"  {key}={os.environ.get(key)}")
    print(f"Found {len(nbt_files)} .nbt files")

    results: List[StructureResult] = []
    per_structure_dirs: List[Path] = []

    for idx, nbt_path in enumerate(nbt_files):
        args.port = args.port_start + idx
        result = preview_one_structure(nbt_path, out_dir, args)
        results.append(result)
        if result.success:
            per_structure_dirs.append(Path(result.output_dir))
            print(f"[OK]   {nbt_path.name} -> {result.output_dir}")
        else:
            print(f"[FAIL] {nbt_path.name} -> {result.error}")

    manifest = {
        "input": str(input_path),
        "resolved_input_root": str(input_root),
        "count": len(results),
        "success_count": sum(r.success for r in results),
        "failure_count": sum(not r.success for r in results),
        "results": [asdict(r) for r in results],
    }

    manifest_path = out_dir / "batch_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.make_contact_sheet and per_structure_dirs:
        contact_path = out_dir / "contact_sheet.png"
        build_contact_sheet(per_structure_dirs, contact_path)
        print(f"Saved contact sheet: {contact_path}")

    print(f"\nSaved batch previews to: {out_dir}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
    os._exit(0)
