"""
Run the floor expert — standalone test/demo script.

Runs N episodes of the floor expert with optional live preview and/or video recording.
Does NOT save demonstration files (use collect_demos.py for that).

Usage:
    python -m cuboid_house_rl.expert.run_floor --episodes 3
    python -m cuboid_house_rl.expert.run_floor --preview
    python -m cuboid_house_rl.expert.run_floor --record floor_run.mp4
    python -m cuboid_house_rl.expert.run_floor --preview --record out.mp4 --record-fps 20
"""
import argparse
import math
import os
import time
import numpy as np

from cuboid_house_rl.config import MAX_EPISODE_STEPS


# ──────────────────────────────────────────────────────────────────────────────
# Video recorder
# ──────────────────────────────────────────────────────────────────────────────

class VideoRecorder:
    """
    Records frames to an mp4/avi file using cv2.VideoWriter.

    Captures the full preview canvas when a PreviewWindow is supplied,
    otherwise upscales the raw 64×64 game image.
    """

    GAME_W = 480
    GAME_H = 360
    PANEL_W = 370  # matches PreviewWindow.PANEL_W

    def __init__(self, path: str, fps: float = 20.0, with_panel: bool = False):
        import cv2
        self._cv2 = cv2
        self.path = path
        self.fps = fps
        self.with_panel = with_panel

        if with_panel:
            w, h = self.GAME_W + self.PANEL_W, self.GAME_H
        else:
            w, h = self.GAME_W, self.GAME_H

        ext = os.path.splitext(path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
        self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        self.w = w
        self.h = h
        self.frames = 0
        print(f"[recorder] Writing {w}×{h} @ {fps:.0f}fps → {path}")

    def write_canvas(self, canvas: np.ndarray):
        """Write a pre-rendered BGR canvas (e.g. from PreviewWindow)."""
        frame = canvas
        if frame.shape[:2] != (self.h, self.w):
            frame = self._cv2.resize(frame, (self.w, self.h))
        self.writer.write(frame)
        self.frames += 1

    def write_game_image(self, img_rgb: np.ndarray):
        """Write a raw game image (RGB), upscaled to fill the frame."""
        if img_rgb is None:
            # Write blank frame
            frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        else:
            ih, iw = img_rgb.shape[:2]
            scale = min(self.w / iw, self.h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            resized = self._cv2.resize(img_rgb, (nw, nh), interpolation=self._cv2.INTER_NEAREST)
            frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            y0 = (self.h - nh) // 2
            x0 = (self.w - nw) // 2
            frame[y0:y0+nh, x0:x0+nw] = resized[:, :, ::-1]  # RGB→BGR
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        self.writer.release()
        print(f"[recorder] Saved {self.frames} frames → {self.path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _get_game_image(env) -> np.ndarray | None:
    """Extract raw RGB image from env CraftGround observation."""
    try:
        obs = env._cg_obs
        if obs is None:
            return None
        if isinstance(obs, dict):
            img = obs.get("pov")
            if img is None:
                img = obs.get("rgb")
            if img is not None:
                return np.asarray(img, dtype=np.uint8)
    except Exception:
        pass
    return None


def run(args):
    from cuboid_house_rl.envs.craftground_adapter import create_craftground_env
    from cuboid_house_rl.envs.house_building_env import HouseBuildingEnv
    from cuboid_house_rl.expert.floor_expert import FloorExpert

    print(f"Creating CraftGround env on port {args.port}...")
    cg_env = create_craftground_env(port=args.port, image_width=640, image_height=360)
    env = HouseBuildingEnv(craftground_env=cg_env)

    # Preview window (optional)
    preview = None
    if args.preview:
        from cuboid_house_rl.utils.preview_window import PreviewWindow

        class _ExpertShim:
            """Shim so PreviewWindow can call expert.get_current_target()."""
            def __init__(self, exp):
                self._floor_expert = exp
            def get_current_target(self):
                return self._floor_expert.get_current_target()

        preview = PreviewWindow(title="Floor Expert")
        _shim_cls = _ExpertShim

    # Video recorder (optional)
    recorder = None
    if args.record:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("[run_floor] cv2 not found — cannot record. Install opencv-python.")
            args.record = None

    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*60}")

        expert = FloorExpert()
        expert.reset()

        obs, info = env.reset(seed=args.seed + ep)
        env.house_width = expert.width
        env.house_depth = expert.depth
        print(f"House size: {expert.width} × {expert.depth}")

        # Create recorder on first episode (so preview is set up first)
        if args.record and recorder is None:
            with_panel = args.preview and preview is not None and preview.enabled
            recorder = VideoRecorder(args.record, fps=args.record_fps, with_panel=with_panel)

        shim = _shim_cls(expert) if args.preview else None

        ep_reward = 0.0
        ep_placed = 0
        step = 0
        be = {}

        while step < args.max_steps and not expert.is_done():
            action = expert.get_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            be = info.get("block_event", {})
            if be.get("type") == "placed":
                ep_placed += 1

            # Update preview
            if preview is not None:
                preview.update(env, shim, step, be)
                if args.preview_delay > 0:
                    time.sleep(args.preview_delay)

            # Record frame
            if recorder is not None:
                if recorder.with_panel and preview is not None:
                    # Capture the canvas that was just rendered
                    import cv2
                    canvas = np.zeros((preview.height, preview.width, 3), dtype=np.uint8)
                    # Re-render canvas for recording
                    img = _get_game_image(env)
                    if img is not None:
                        ih, iw = img.shape[:2]
                        scale = min(preview.GAME_W / iw, preview.height / ih)
                        nw, nh = int(iw * scale), int(ih * scale)
                        img_bgr = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
                        if img_bgr.shape[2] == 3:
                            img_bgr = img_bgr[:, :, ::-1]
                        y0 = (preview.height - nh) // 2
                        x0 = (preview.GAME_W - nw) // 2
                        canvas[y0:y0+nh, x0:x0+nw] = img_bgr
                    preview._draw_debug(canvas, env, shim, step, be)
                    recorder.write_canvas(canvas)
                else:
                    recorder.write_game_image(_get_game_image(env))

            # Sync target queue after origin is set
            if expert.origin_set:
                remaining = expert.get_remaining_targets()
                if remaining:
                    env.set_target_queue(remaining)

            if step % 200 == 0 or step <= 50:
                comp = info.get("completion", {})
                print(
                    f"  [{step:>4}] state={expert.state} | "
                    f"placed={ep_placed} | "
                    f"F:{comp.get('floor_ratio', 0):.0%} | "
                    f"pos=({env.agent_x:.2f},{env.agent_y:.2f},{env.agent_z:.2f}) | "
                    f"yaw={math.degrees(env.agent_yaw):.1f}° "
                    f"pitch={math.degrees(env.agent_pitch):.1f}° | "
                    f"origin={'set' if env.origin_set else 'no'}"
                )

            if terminated or truncated:
                break

        comp = info.get("completion", {})
        print(
            f"\n  Summary: steps={step} | placed={ep_placed} | "
            f"F:{comp.get('floor_ratio', 0):.0%} | reward={ep_reward:.1f}"
        )

    if recorder is not None:
        recorder.close()
    if preview is not None:
        preview.close()
    env.craftground_env.close()


def main():
    parser = argparse.ArgumentParser(description="Run floor expert")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true",
                        help="Show live debug window (requires opencv-python, set DISPLAY=:0 on WSL2)")
    parser.add_argument("--preview-delay", type=float, default=0.0,
                        help="Seconds between steps in preview mode (default: 0 = no delay)")
    parser.add_argument("--record", type=str, default=None, metavar="FILE",
                        help="Save video to FILE (e.g. floor.mp4 or floor.avi); requires opencv-python")
    parser.add_argument("--record-fps", type=float, default=20.0,
                        help="Video frame rate (default: 20)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
