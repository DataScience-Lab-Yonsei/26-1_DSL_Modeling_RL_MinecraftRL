"""
Expert debug preview window (cv2-based).

Shows game image on the left and expert debug info on the right.
Does NOT change CraftGround image_width/height — avoids the 0% issue.
"""
import math
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class PreviewWindow:
    PANEL_W = 370     # right panel (debug text)
    GAME_H = 360      # game panel height
    GAME_W = 480      # game panel width

    def __init__(self, title: str = "Floor Expert Debug"):
        self.title = title
        self.enabled = _CV2_AVAILABLE
        self.width = self.GAME_W + self.PANEL_W
        self.height = self.GAME_H
        if not self.enabled:
            print("[PreviewWindow] cv2 not available — preview disabled")
            return
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.width, self.height)

    def update(self, env, expert, step: int, be: dict):
        if not self.enabled:
            return

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Left: game image — scale to fit panel maintaining aspect ratio
        img = self._get_game_image(env)
        if img is not None:
            ih, iw = img.shape[:2]
            scale = min(self.GAME_W / iw, self.height / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if img_resized.ndim == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            elif img_resized.shape[2] == 3:
                img_resized = img_resized[:, :, ::-1]  # RGB → BGR
            y0 = (self.height - nh) // 2
            x0 = (self.GAME_W - nw) // 2
            canvas[y0:y0+nh, x0:x0+nw] = img_resized

        # Right: debug panel
        self._draw_debug(canvas, env, expert, step, be)

        cv2.imshow(self.title, canvas)
        cv2.waitKey(1)

    # ------------------------------------------------------------------

    def _get_game_image(self, env):
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

    def _draw_debug(self, canvas, env, expert, step: int, be: dict):
        x0 = self.GAME_W + 8
        y_ref = [18]
        lh = 19
        font = cv2.FONT_HERSHEY_SIMPLEX

        def put(text, color=(200, 200, 200), scale=0.44, bold=False):
            cv2.putText(canvas, text, (x0, y_ref[0]), font, scale,
                        color, 2 if bold else 1, cv2.LINE_AA)
            y_ref[0] += lh

        def sep():
            yy = y_ref[0]
            cv2.line(canvas, (x0, yy),
                     (x0 + self.PANEL_W - 12, yy), (55, 55, 55), 1)
            y_ref[0] += 7

        def bar(pct, color=(0, 200, 80)):
            bw = int((self.PANEL_W - 20) * max(0.0, min(1.0, pct)))
            by = y_ref[0]
            cv2.rectangle(canvas, (x0, by),
                          (x0 + self.PANEL_W - 20, by + 10), (50, 50, 50), -1)
            if bw > 0:
                cv2.rectangle(canvas, (x0, by), (x0 + bw, by + 10), color, -1)
            y_ref[0] += 13

        # ── Header ──
        put("=== Expert Debug ===", (80, 220, 80), scale=0.48, bold=True)
        sep()

        # Step & state
        floor_exp = getattr(expert, '_floor_expert', None)
        state = floor_exp.state if floor_exp else "?"
        put(f"Step: {step}", (180, 180, 180))
        put(f"State: {state}", (100, 220, 255))
        sep()

        # Position
        put(f"X:{env.agent_x:6.2f}  Y:{env.agent_y:.2f}  Z:{env.agent_z:6.2f}")
        yaw_d = math.degrees(env.agent_yaw)
        pit_d = math.degrees(env.agent_pitch)
        put(f"Yaw:{yaw_d:6.1f}  Pitch:{pit_d:5.1f}")
        sep()

        # Origin & target
        if env.origin_set:
            put(f"Origin: ({env.origin_x}, {env.origin_z})", (160, 160, 255))
            tgt = expert.get_current_target() if hasattr(expert, 'get_current_target') else None
            if tgt:
                put(f"Target: {tgt}", (255, 220, 60))
            else:
                put("Target: done", (100, 200, 100))
        else:
            put("Origin: not set", (160, 80, 80))
        sep()

        # Floor completion bar + numbers
        if env.origin_set:
            n_fl = max(1, len(env.floor_positions))
            c_fl = len(env.correct_blocks & env.floor_positions)
            pct = c_fl / n_fl
            bar(pct, (0, 200, 80))
            put(f"Floor: {pct*100:.0f}%  ({c_fl}/{n_fl})", (0, 220, 100))
        else:
            put("Floor: 0%  (no origin)", (100, 100, 100))
        put(f"Correct:{env.correct_placements}  Wrong:{env.incorrect_placements}")
        sep()

        # Raycast
        if env._cg_obs is not None:
            hit = env._cg_obs_extractor.extract_raycast(env._cg_obs)
            if hit:
                hx, hy, hz = hit["position"]
                fn = hit["face_normal"]
                put(f"Ray: ({hx},{hy},{hz})", (255, 180, 60))
                put(f"     face:{fn}  d:{hit['distance']:.2f}", (200, 200, 200))
            else:
                put("Ray: MISS", (120, 120, 120))
        else:
            put("Ray: --", (100, 100, 100))
        sep()

        # Last block event
        be_type = be.get("type", "none")
        be_pos = be.get("position")
        c_ev = (60, 220, 60) if be_type == "placed" \
               else (80, 80, 200) if be_type == "removed" \
               else (110, 110, 110)
        put(f"Event: {be_type}", c_ev)
        if be_pos:
            put(f"  at: {be_pos}", (180, 180, 180))

    def close(self):
        if self.enabled:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
