"""
Collect expert demonstrations for Behaviour Cloning (V3).

Syncs random house size between expert and env.

Usage:
    python -m cuboid_house_rl.expert.collect_demos --stage floor --episodes 20
    python -m cuboid_house_rl.expert.collect_demos --stage all --episodes 20
    python -m cuboid_house_rl.expert.collect_demos --stage floor --episodes 5 --record demos.mp4
"""
import argparse
import os
import time
import numpy as np

from cuboid_house_rl.config import MAX_EPISODE_STEPS, BC_DEMO_DIR


def _get_game_image(env):
    """Extract raw RGB image from env CraftGround observation."""
    try:
        obs = env._cg_obs
        if obs is None:
            return None
        if isinstance(obs, dict):
            img = obs.get("pov")
        else:
            img = getattr(obs, "image", None)
            if img is None:
                ext = env._cg_obs_extractor
                img = ext.extract_image(obs) if ext else None
        return img
    except Exception:
        return None


class _VideoRecorder:
    """Lightweight video recorder for collect_demos."""

    GAME_W = 480
    GAME_H = 360
    PANEL_W = 370

    def __init__(self, path, fps=20.0, with_panel=False):
        import cv2
        self._cv2 = cv2
        self.path = path
        self.with_panel = with_panel
        w = self.GAME_W + self.PANEL_W if with_panel else self.GAME_W
        h = self.GAME_H
        ext = os.path.splitext(path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
        self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        self.w, self.h = w, h
        self.frames = 0
        print(f"[recorder] Writing {w}×{h} @ {fps:.0f}fps → {path}")

    def write_canvas(self, canvas):
        frame = canvas
        if frame.shape[:2] != (self.h, self.w):
            frame = self._cv2.resize(frame, (self.w, self.h))
        self.writer.write(frame)
        self.frames += 1

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
            y0, x0 = (self.h - nh) // 2, (self.w - nw) // 2
            frame[y0:y0+nh, x0:x0+nw] = resized[:, :, ::-1]  # RGB→BGR
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        self.writer.release()
        print(f"[recorder] Saved {self.frames} frames → {self.path}")


def collect_demos(args):
    from cuboid_house_rl.envs.craftground_adapter import create_craftground_env
    from cuboid_house_rl.envs.house_building_env import HouseBuildingEnv
    from cuboid_house_rl.expert.scripted_expert import ScriptedExpert

    os.makedirs(args.output_dir, exist_ok=True)

    preview = None
    if args.preview:
        from cuboid_house_rl.utils.preview_window import PreviewWindow
        preview = PreviewWindow()

    recorder = None
    if args.record:
        recorder = _VideoRecorder(
            args.record, fps=args.record_fps,
            with_panel=(preview is not None),
        )

    print(f"Stage: {args.stage}")
    print(f"Creating CraftGround env on port {args.port}...")
    # Always use 64x64 for consistent timing — preview upscales this for display
    cg_env = create_craftground_env(port=args.port, image_width=64, image_height=64)
    env = HouseBuildingEnv(craftground_env=cg_env)

    all_obs = []
    all_actions = []
    all_episode_ids = []
    all_stage_ids = []

    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*60}")

        # Create expert first (generates random size)
        expert = ScriptedExpert(stage=args.stage)
        expert.reset()

        # Reset env, then override house size to match expert
        obs, info = env.reset(seed=args.seed + ep)
        env.house_width = expert.actual_width
        env.house_depth = expert.actual_depth
        print(f"House size: {env.house_width}x{env.house_depth}")

        ep_obs = []
        ep_actions = []
        ep_stage_ids = []
        ep_reward = 0.0
        ep_placed = 0
        step = 0

        while step < args.max_steps and not expert.is_done():
            action = expert.get_action(env)
            expert.update_stage_id()  # update looking detection

            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            ep_stage_ids.append(expert.current_stage_id)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            be = info.get("block_event", {})
            if be.get("type") == "placed":
                ep_placed += 1

            if preview is not None:
                preview.update(env, expert, step, be)
                if recorder is not None:
                    recorder.write_canvas(preview.canvas)
                time.sleep(args.preview_delay)
            elif recorder is not None:
                recorder.write_game_image(_get_game_image(env))

            # Sync target queue after origin is set
            if expert.origin_set:
                remaining = expert.get_remaining_targets()
                if remaining:
                    env.set_target_queue(remaining)

            # Stage done messages
            if hasattr(expert, '_stage_just_done'):
                done_name = expert._stage_just_done
                if done_name:
                    comp = info.get("completion", {})
                    if done_name == "floor":
                        print(f"  Floor done: {comp.get('floor_ratio',0):.0%} (steps={step}, placed={ep_placed})")
                    elif done_name == "wall":
                        print(f"  Wall done: {comp.get('wall_ratio',0):.0%} (steps={step})")
                    elif done_name == "door":
                        print(f"  Door done! (step={step})")
                    elif done_name == "ceiling":
                        print(f"  Ceiling done: {comp.get('ceiling_ratio',0):.0%} (steps={step})")
                    elif done_name == "door_open":
                        print(f"  Door opened! (step={step})")
                    elif done_name == "looking":
                        print(f"  Looking done! (step={step})")
                    expert._stage_just_done = None

            # 200-step progress log
            if step % 200 == 0:
                comp = info.get("completion", {})
                print(
                    f"[{step:>5}] "
                    f"F:{comp.get('floor_ratio', 0):.0%} "
                    f"W:{comp.get('wall_ratio', 0):.0%} "
                    f"C:{comp.get('ceiling_ratio', 0):.0%} | "
                    f"correct={info.get('correct_placements', 0)} "
                    f"wrong={info.get('incorrect_placements', 0)} | "
                    f"pos=({env.agent_x:.1f},{env.agent_y:.1f},{env.agent_z:.1f}) | "
                    f"size={env.house_width}x{env.house_depth}"
                )

            # For 'all' stage, don't break on terminated (finish sequence needs to complete)
            if args.stage == "all":
                if truncated:
                    break
            else:
                if terminated or truncated:
                    break

        comp = info.get("completion", {})
        f_r = comp.get('floor_ratio', 0)
        w_r = comp.get('wall_ratio', 0)
        c_r = comp.get('ceiling_ratio', 0)
        correct = info.get('correct_placements', 0)
        wrong = info.get('incorrect_placements', 0)

        # Only save episodes where expert completed all stages (including looking)
        comp_ok = False
        if args.stage == "floor":
            comp_ok = comp.get("floor_ratio", 0) >= 1.0
        elif args.stage == "walls":
            comp_ok = comp.get("wall_ratio", 0) >= 1.0
        elif args.stage == "all":
            comp_ok = expert.is_done()

        if comp_ok and len(ep_obs) > 0:
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            all_episode_ids.extend([ep] * len(ep_obs))
            all_stage_ids.extend(ep_stage_ids)
            print(f"Episode {ep+1}: F:{f_r:.0%} W:{w_r:.0%} C:{c_r:.0%} | "
                  f"correct={correct} wrong={wrong} | steps={step} | "
                  f"size={env.house_width}x{env.house_depth} | saved ✓")
        else:
            print(f"Episode {ep+1}: F:{f_r:.0%} W:{w_r:.0%} C:{c_r:.0%} | "
                  f"correct={correct} wrong={wrong} | steps={step} | "
                  f"size={env.house_width}x{env.house_depth} | skipped (incomplete)")

    filename = f"demos_{args.stage}.npz"
    save_path = os.path.join(args.output_dir, filename)

    if len(all_obs) == 0:
        print(f"\nNo successful episodes — nothing to save.")
    else:
        # Accumulate: load existing data if file already exists
        if os.path.exists(save_path):
            existing = np.load(save_path, allow_pickle=False)
            prev_obs      = existing["observations"]
            prev_actions  = existing["actions"]
            prev_ep_ids   = existing["episode_ids"]
            # Skip accumulation if existing file is empty/corrupt
            if prev_obs.ndim >= 2 and len(prev_obs) > 0:
                ep_offset = int(prev_ep_ids.max()) + 1
                new_ep_ids = np.array(all_episode_ids, dtype=np.int32) + ep_offset
                obs_array    = np.concatenate([prev_obs,     np.array(all_obs,     dtype=np.float32)], axis=0)
                action_array = np.concatenate([prev_actions, np.array(all_actions, dtype=np.int64)],   axis=0)
                episode_ids  = np.concatenate([prev_ep_ids,  new_ep_ids],                              axis=0)
                # Accumulate stage_ids
                prev_stage_ids = existing.get("stage_ids", np.zeros(len(prev_obs), dtype=np.int32))
                stage_id_array = np.concatenate([prev_stage_ids, np.array(all_stage_ids, dtype=np.int32)], axis=0)
                print(f"\nAccumulating: {len(prev_obs)} existing + {len(all_obs)} new = {len(obs_array)} transitions")
            else:
                obs_array    = np.array(all_obs,     dtype=np.float32)
                action_array = np.array(all_actions, dtype=np.int64)
                episode_ids  = np.array(all_episode_ids, dtype=np.int32)
                stage_id_array = np.array(all_stage_ids, dtype=np.int32)
                print(f"\nExisting file was empty, overwriting.")
        else:
            obs_array    = np.array(all_obs,     dtype=np.float32)
            action_array = np.array(all_actions, dtype=np.int64)
            episode_ids  = np.array(all_episode_ids, dtype=np.int32)
            stage_id_array = np.array(all_stage_ids, dtype=np.int32)
        np.savez_compressed(
            save_path,
            observations=obs_array,
            actions=action_array,
            episode_ids=episode_ids,
            stage_ids=stage_id_array,
            stage=args.stage,
        )

        print(f"Saved {len(obs_array)} transitions total → {save_path}")
        print(f"  Size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")

    if recorder is not None:
        recorder.close()
    if preview is not None:
        preview.close()
    env.craftground_env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument("--stage", type=str, default="floor",
                        choices=["floor", "walls", "all"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=BC_DEMO_DIR)
    parser.add_argument("--preview", action="store_true",
                        help="Show debug window with game image and expert state")
    parser.add_argument("--preview-delay", type=float, default=0.1,
                        help="Seconds to wait between steps when --preview is on (default: 0.1)")
    parser.add_argument("--record", type=str, default=None, metavar="FILE",
                        help="Save video to FILE (.mp4 or .avi)")
    parser.add_argument("--record-fps", type=float, default=20.0,
                        help="Video frame rate (default: 20)")
    args = parser.parse_args()
    collect_demos(args)


if __name__ == "__main__":
    main()
