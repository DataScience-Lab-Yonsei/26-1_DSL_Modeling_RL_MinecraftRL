"""
Run floor + wall expert — standalone test/demo script.

Runs FloorExpert to completion, then WallExpert on the same episode.
Does NOT save demonstration files (use collect_demos.py for that).

Usage:
    python -m cuboid_house_rl.expert.run_wall --episodes 1
    python -m cuboid_house_rl.expert.run_wall --preview
    python -m cuboid_house_rl.expert.run_wall --record wall_run.mp4
    python -m cuboid_house_rl.expert.run_wall --preview --record out.mp4 --record-fps 20
"""
import argparse
import math
import os
import time
import numpy as np

from cuboid_house_rl.config import MAX_EPISODE_STEPS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (shared with run_floor.py)
# ──────────────────────────────────────────────────────────────────────────────

def _get_game_image(env):
    try:
        obs = env._cg_obs
        if obs is None:
            return None
        if isinstance(obs, dict):
            # Try common image keys
            for key in ("pov", "rgb", "image", "obs"):
                img = obs.get(key)
                if img is not None:
                    return np.asarray(img, dtype=np.uint8)
            # If dict has 'full' proto, try image attribute
            proto = obs.get("full")
            if proto is not None and hasattr(proto, "image"):
                raw = proto.image
                if raw is not None and len(raw) > 0:
                    img_w = getattr(proto, "imageSizeX", 640)
                    img_h = getattr(proto, "imageSizeY", 360)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    if len(arr) == img_w * img_h * 3:
                        return arr.reshape(img_h, img_w, 3)
        # Not a dict — try direct proto
        if hasattr(obs, "image"):
            raw = obs.image
            if raw is not None and len(raw) > 0:
                img_w = getattr(obs, "imageSizeX", 640)
                img_h = getattr(obs, "imageSizeY", 360)
                arr = np.frombuffer(raw, dtype=np.uint8)
                if len(arr) == img_w * img_h * 3:
                    return arr.reshape(img_h, img_w, 3)
    except Exception:
        pass
    return None


class VideoRecorder:
    GAME_W = 480
    GAME_H = 360
    PANEL_W = 370

    def __init__(self, path: str, fps: float = 20.0, with_panel: bool = False):
        import cv2
        self._cv2 = cv2
        self.path = path
        self.fps = fps
        self.with_panel = with_panel
        w = self.GAME_W + self.PANEL_W if with_panel else self.GAME_W
        h = self.GAME_H
        ext = os.path.splitext(path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
        self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        self.w, self.h = w, h
        self.frames = 0
        print(f"[recorder] Writing {w}×{h} @ {fps:.0f}fps → {path}")

    def write_game_image(self, img_rgb):
        if img_rgb is None:
            frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        else:
            ih, iw = img_rgb.shape[:2]
            scale = min(self.w / iw, self.h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            resized = self._cv2.resize(img_rgb, (nw, nh),
                                       interpolation=self._cv2.INTER_NEAREST)
            frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            y0 = (self.h - nh) // 2
            x0 = (self.w - nw) // 2
            frame[y0:y0+nh, x0:x0+nw] = resized[:, :, ::-1]
        self.writer.write(frame)
        self.frames += 1

    def write_canvas(self, canvas: np.ndarray):
        import cv2
        frame = canvas
        if frame.shape[:2] != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h))
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        self.writer.release()
        print(f"[recorder] Saved {self.frames} frames → {self.path}")


# ──────────────────────────────────────────────────────────────────────────────
# Expert shim for PreviewWindow
# ──────────────────────────────────────────────────────────────────────────────

class _ExpertShim:
    """Wraps an expert so PreviewWindow._draw_debug can call get_current_target()."""
    def __init__(self, exp, floor_exp=None):
        # PreviewWindow looks for ._floor_expert
        self._floor_expert = floor_exp if floor_exp is not None else exp
        self._exp = exp

    def get_current_target(self):
        return self._exp.get_current_target()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _step(env, expert, obs, step, args, preview, recorder, be_out):
    """Take one env step, update preview/recorder. Returns (obs, reward, terminated, truncated, info)."""
    action = expert.get_action(env)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    be = info.get("block_event", {})
    be_out.update(be)

    if preview is not None:
        shim = _ExpertShim(expert)
        preview.update(env, shim, step, be)
        if args.preview_delay > 0:
            time.sleep(args.preview_delay)

    if recorder is not None:
        if recorder.with_panel and preview is not None:
            import cv2
            canvas = np.zeros((preview.height, preview.width, 3), dtype=np.uint8)
            img = _get_game_image(env)
            if img is not None:
                ih, iw = img.shape[:2]
                scale = min(preview.GAME_W / iw, preview.height / ih)
                nw, nh = int(iw * scale), int(ih * scale)
                img_bgr = cv2.resize(img, (nw, nh),
                                     interpolation=cv2.INTER_NEAREST)[:, :, ::-1]
                y0 = (preview.height - nh) // 2
                x0 = (preview.GAME_W - nw) // 2
                canvas[y0:y0+nh, x0:x0+nw] = img_bgr
            shim2 = _ExpertShim(expert)
            preview._draw_debug(canvas, env, shim2, step, be)
            recorder.write_canvas(canvas)
        else:
            recorder.write_game_image(_get_game_image(env))

    return obs, reward, terminated, truncated, info, step


def run(args):
    from cuboid_house_rl.envs.craftground_adapter import create_craftground_env
    from cuboid_house_rl.envs.house_building_env import HouseBuildingEnv
    from cuboid_house_rl.expert.floor_expert import FloorExpert
    from cuboid_house_rl.expert.wall_expert import WallExpert

    print(f"Creating CraftGround env on port {args.port}...")
    cg_env = create_craftground_env(port=args.port, image_width=640, image_height=360)
    env = HouseBuildingEnv(craftground_env=cg_env)

    preview = None
    if args.preview:
        from cuboid_house_rl.utils.preview_window import PreviewWindow
        preview = PreviewWindow(title="Wall Expert")

    recorder = None
    if args.record:
        try:
            import cv2  # noqa
        except ImportError:
            print("[run_wall] cv2 not found — cannot record.")
            args.record = None

    try:
      _run_episodes(env, args, preview)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    except Exception as e:
        print(f"\n[error] {e}")
    finally:
        _cleanup(env, recorder, preview)


def _run_episodes(env, args, preview):
    from cuboid_house_rl.expert.floor_expert import FloorExpert
    from cuboid_house_rl.expert.wall_expert import WallExpert
    from cuboid_house_rl.expert.door_expert import DoorExpert
    from cuboid_house_rl.expert.ceiling_expert import CeilingExpert
    recorder = None
    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*60}")

        floor_expert = FloorExpert()
        floor_expert.reset()

        obs, info = env.reset(seed=args.seed + ep)
        env.house_width  = floor_expert.width
        env.house_depth  = floor_expert.depth
        print(f"House size: {floor_expert.width} × {floor_expert.depth}")

        if args.record and recorder is None:
            with_panel = args.preview and preview is not None and preview.enabled
            recorder = VideoRecorder(args.record, fps=args.record_fps,
                                     with_panel=with_panel)

        step = 0
        ep_placed = 0
        be = {}

        # ── Stage 1: Floor ────────────────────────────────────────────────────
        print("  [stage] Floor building...")
        while step < args.max_steps and not floor_expert.is_done():
            obs, _, terminated, truncated, info, step = _step(
                env, floor_expert, obs, step, args, preview, recorder, be
            )
            if be.get("type") == "placed":
                ep_placed += 1

            if floor_expert.origin_set:
                remaining = floor_expert.get_remaining_targets()
                if remaining:
                    env.set_target_queue(remaining)

            if step % 200 == 0:
                comp = info.get("completion", {})
                print(f"  [floor {step:>4}] F:{comp.get('floor_ratio',0):.0%} "
                      f"pos=({env.agent_x:.1f},{env.agent_y:.1f},{env.agent_z:.1f})")
            if terminated or truncated:
                break

        comp = info.get("completion", {})
        print(f"  Floor done: {comp.get('floor_ratio',0):.0%}  "
              f"(steps={step}, placed={ep_placed})")

        if floor_expert.origin_x is None:
            print("  Floor origin never set — skipping wall stage.")
            continue

        # ── Stage 2: Wall ─────────────────────────────────────────────────────
        wall_expert = WallExpert(
            origin_x=floor_expert.origin_x,
            origin_z=floor_expert.origin_z,
            width=floor_expert.width,
            depth=floor_expert.depth,
            hotbar=floor_expert._hotbar,
        )
        n_cols = len(wall_expert.columns)
        print(f"  [stage] Wall building: {n_cols} columns "
              f"({'east' if wall_expert._start_east else 'west'} first)")

        wall_placed = 0
        while step < args.max_steps and not wall_expert.is_done():
            obs, _, terminated, truncated, info, step = _step(
                env, wall_expert, obs, step, args, preview, recorder, be
            )
            if be.get("type") == "placed":
                ep_placed += 1
                wall_placed += 1

            if step % 200 == 0:
                comp = info.get("completion", {})
                print(f"  [wall  {step:>4}] col={wall_expert.col_idx}/{n_cols} "
                      f"W:{comp.get('wall_ratio',0):.0%} "
                      f"pos=({env.agent_x:.1f},{env.agent_y:.1f},{env.agent_z:.1f}) "
                      f"pitch={math.degrees(env.agent_pitch):.1f}° "
                      f"state={wall_expert.state}")
            if terminated or truncated:
                break

        # ── Stage 3: Door ────────────────────────────────────────────────────
        door_expert = DoorExpert(
            origin_x=floor_expert.origin_x,
            origin_z=floor_expert.origin_z,
            width=floor_expert.width,
            depth=floor_expert.depth,
        )
        print(f"  [stage] Door: breaking wall at ({door_expert.door_x}, {door_expert.door_z})")

        while step < args.max_steps and not door_expert.is_done():
            obs, _, terminated, truncated, info, step = _step(
                env, door_expert, obs, step, args, preview, recorder, be
            )

            if step % 200 == 0:
                print(f"  [door  {step:>4}] state={door_expert.state} "
                      f"pos=({env.agent_x:.1f},{env.agent_y:.1f},{env.agent_z:.1f})")
            if terminated or truncated:
                break

        if door_expert.is_done():
            print(f"  Door done! (step={step})")

        # ── Stage 4: Ceiling ─────────────────────────────────────────────
        # Inherit hotbar from door expert (or wall expert)
        prev_hotbar = getattr(door_expert, '_hotbar', getattr(wall_expert, '_hotbar', 0))
        ceiling_expert = CeilingExpert(
            origin_x=floor_expert.origin_x,
            origin_z=floor_expert.origin_z,
            width=floor_expert.width,
            depth=floor_expert.depth,
            initial_hotbar=prev_hotbar,
            door_x=door_expert.door_x,
        )
        print(f"  [stage] Ceiling: {ceiling_expert.total_rows} rows, "
              f"x range {ceiling_expert.x_min}-{ceiling_expert.x_max}")

        while step < args.max_steps and not ceiling_expert.is_done():
            obs, _, terminated, truncated, info, step = _step(
                env, ceiling_expert, obs, step, args, preview, recorder, be
            )
            if be.get("type") == "placed":
                ep_placed += 1

            if step % 200 == 0:
                comp = info.get("completion", {})
                print(f"  [ceil  {step:>4}] row={ceiling_expert.row_idx}/{ceiling_expert.total_rows} "
                      f"C:{comp.get('ceiling_ratio',0):.0%} "
                      f"pos=({env.agent_x:.1f},{env.agent_y:.1f},{env.agent_z:.1f}) "
                      f"state={ceiling_expert.state}")
            # Don't break on terminated — finish sequence needs to complete
            if truncated:
                break

        if ceiling_expert.is_done():
            print(f"  Looking done! Episode complete. (step={step})")

        comp = info.get("completion", {})
        print(f"\n  Summary: steps={step} | total placed={ep_placed} | "
              f"F:{comp.get('floor_ratio',0):.0%} "
              f"W:{comp.get('wall_ratio',0):.0%} "
              f"C:{comp.get('ceiling_ratio',0):.0%}")
        print(f"  Origin: ({floor_expert.origin_x}, {floor_expert.origin_z}) | "
              f"Size: {floor_expert.width}x{floor_expert.depth}")

    _cleanup(env, recorder, preview)


def _cleanup(env, recorder=None, preview=None):
    """Ensure Java/resources are cleaned up even on crash."""
    try:
        if recorder is not None:
            recorder.close()
    except Exception:
        pass
    try:
        if preview is not None:
            preview.close()
    except Exception:
        pass
    try:
        env.craftground_env.close()
    except Exception:
        pass
    # Kill any leftover Java processes
    import subprocess
    subprocess.run(["pkill", "-9", "-f", "net.fabricmc"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "GradleDaemon"], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Run floor + wall expert")
    parser.add_argument("--episodes",      type=int,   default=1)
    parser.add_argument("--max-steps",     type=int,   default=MAX_EPISODE_STEPS)
    parser.add_argument("--port",          type=int,   default=8023)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--preview",       action="store_true",
                        help="Show live debug window (requires opencv-python)")
    parser.add_argument("--preview-delay", type=float, default=0.0)
    parser.add_argument("--record",        type=str,   default=None, metavar="FILE",
                        help="Save video to FILE (.mp4 or .avi)")
    parser.add_argument("--record-fps",    type=float, default=20.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
