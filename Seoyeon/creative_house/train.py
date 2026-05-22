"""
train.py — CraftGround 집 건축 PPO 훈련 스크립트

사용법:
  # Phase 1: creative 모드로 빠른 사전학습
  python train.py --env_mode creative --total_steps 1_000_000

  # Phase 2: safe 모드로 파인튜닝
  python train.py --env_mode safe --total_steps 1_000_000 \\
                  --resume checkpoints/<ts>/best/best_model

  # Phase 3: survival 모드로 최종 훈련
  python train.py --env_mode survival --total_steps 2_000_000 \\
                  --resume checkpoints/<ts>/best/best_model

  # 평가
  python train.py --mode eval --env_mode survival \\
                  --resume checkpoints/<ts>/best/best_model
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from house_building_wrapper import make_house_env
from mode_config import MODES

# pygame은 선택 의존성 — 없으면 렌더 콜백 비활성화
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


# ── 모드별 PPO 하이퍼파라미터 권장값 ──────────────────────────────
PPO_HYPERPARAMS: dict[str, dict] = {
    "creative": dict(
        learning_rate = 3e-4,
        n_steps       = 512,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,    # 제약 없음 → 단기 집중해도 됨
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.02,    # 탐색 강조 (행동 공간이 넓음)
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
    ),
    "safe": dict(
        learning_rate = 1e-4,
        n_steps       = 512,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.995,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
    ),
    "survival": dict(
        learning_rate = 5e-5,    # 파인튜닝 → 낮은 lr
        n_steps       = 512,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.995,
        gae_lambda    = 0.95,
        clip_range    = 0.15,    # 기존 정책 보존
        ent_coef      = 0.005,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
    ),
}


# ── CNN 특징 추출기 ────────────────────────────────────────────────
class HouseCNNExtractor(BaseFeaturesExtractor):
    """
    이미지(CNN) + 상태 벡터(MLP) 융합 특징 추출기.
    입력 Dict: {"image": (64,114,3), "state": (8,), "raycast": (8,), "structure": (6,)}
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        img_space = observation_space["image"]
        h, w, c   = img_space.shape

        # CNN — CraftGround-Experiments 와 동일한 설정 (stride=2, kernel=5)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, c, h, w)).shape[1]

        # 비이미지 벡터 MLP (state 8 + raycast 8 + structure 6 = 22)
        vec_dim = (
            observation_space["state"].shape[0]
            + observation_space["raycast"].shape[0]
            + observation_space["structure"].shape[0]
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        img = obs["image"].float() / 255.0
        img = img.permute(0, 3, 1, 2)          # (B,H,W,C) → (B,C,H,W)
        cnn_feat = self.cnn(img)

        vec = torch.cat([
            obs["state"].float(),
            obs["raycast"].float(),
            obs["structure"].float(),
        ], dim=-1)
        vec_feat = self.vec_mlp(vec)

        return self.fusion(torch.cat([cnn_feat, vec_feat], dim=-1))


# ── 진행 상황 로깅 콜백 ───────────────────────────────────────────
class HouseProgressCallback(BaseCallback):
    """TensorBoard에 마일스톤 달성률과 구조물 완성도를 기록합니다."""

    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_info: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "milestones" in info:
                self._ep_info.append({
                    "milestones": info["milestones"],
                    "structure":  info.get("structure", {}),
                })

        if self.num_timesteps % self.log_freq != 0 or not self._ep_info:
            return True

        recent = self._ep_info[-200:]
        ms_keys = [
            "placed_5", "placed_20", "placed_50", "placed_100",
            "has_wall", "has_roof", "has_door", "has_light", "has_furniture",
        ]
        for k in ms_keys:
            rate = np.mean([e["milestones"].get(k, False) for e in recent])
            self.logger.record(f"house/ms_{k}", rate)

        enc_rate = np.mean([e["structure"].get("enclosed", False) for e in recent])
        avg_blk  = np.mean([e["structure"].get("total", 0) for e in recent])
        self.logger.record("house/enclosed_rate",  enc_rate)
        self.logger.record("house/avg_blocks",     avg_blk)

        if self.verbose:
            print(
                f"\n[{self.num_timesteps:,}] "
                f"blk={avg_blk:.1f}  "
                f"wall={np.mean([e['milestones'].get('has_wall',False) for e in recent]):.2f}  "
                f"enclosed={enc_rate:.2f}"
            )
        return True


# ── 실시간 영상 표시 콜백 ─────────────────────────────────────────
class RenderCallback(BaseCallback):
    """
    훈련 중 에이전트 POV 영상을 실시간으로 pygame 창에 표시합니다.

    표시 내용:
      - 왼쪽: 에이전트 시점 (obs["image"])
      - 하단 HUD: 스텝, 보상, 블록 수, 마일스톤

    pygame이 없으면 자동으로 비활성화됩니다.
    서버(headless) 환경에서는 DISPLAY 환경변수가 없으면 비활성화됩니다.
    """

    WIN_W = 456   # 114 × 4
    WIN_H = 256 + 80  # 이미지 + HUD 영역
    IMG_SCALE = 4

    def __init__(self, render_freq: int = 4, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq   # N스텝마다 갱신 (너무 자주 하면 느림)
        self._screen    = None
        self._font      = None
        self._active    = False
        self._step_count = 0
        self._ep_reward  = 0.0
        self._last_info: dict = {}

    def _on_training_start(self) -> None:
        if not _PYGAME_AVAILABLE:
            print("⚠️  pygame 미설치 — 실시간 영상 비활성화")
            print("   pip install pygame  으로 설치하면 활성화됩니다.")
            return

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("SDL_VIDEODRIVER"):
            # headless 환경 — SDL을 offscreen 모드로 강제
            os.environ["SDL_VIDEODRIVER"] = "offscreen"
            print("⚠️  DISPLAY 미설정 — pygame offscreen 모드 (창 표시 안 됨)")
            print("   VNC/X11 포워딩 환경에서는 DISPLAY=:0 설정 후 재실행하세요.")
            return

        pygame.init()
        pygame.display.set_caption("🏠 CraftGround House Builder — 학습 중")
        self._screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        self._font   = pygame.font.SysFont("monospace", 14)
        self._active = True
        print("🖥️  실시간 영상 창 활성화됨 (pygame)")

    def _on_step(self) -> bool:
        self._step_count += 1
        if not self._active:
            return True
        if self._step_count % self.render_freq != 0:
            return True

        # pygame 이벤트 처리 (창 닫기 버튼)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._active = False
                pygame.quit()
                return True

        # obs, info 가져오기
        obs_list = self.locals.get("obs_tensor") or self.locals.get("new_obs")
        infos    = self.locals.get("infos", [{}])
        info     = infos[0] if infos else {}
        self._last_info = info

        rewards = self.locals.get("rewards", [0.0])
        dones   = self.locals.get("dones", [False])
        if dones is not None and len(dones) > 0 and dones[0]:
            self._ep_reward = 0.0   # 에피소드 종료 시 리셋
        self._ep_reward += float(rewards[0]) if rewards is not None else 0.0

        # 이미지 추출
        img_np = None
        if obs_list is not None:
            try:
                if hasattr(obs_list, "cpu"):       # Tensor
                    img_np = obs_list["image"][0].cpu().numpy()
                elif isinstance(obs_list, dict):   # numpy dict (VecEnv)
                    img_np = obs_list["image"][0]
            except Exception:
                pass

        self._screen.fill((30, 30, 30))

        # 이미지 렌더링
        if img_np is not None:
            try:
                import numpy as np
                img_np = img_np.astype(np.uint8)
                # (H, W, C) → pygame surface
                surf = pygame.surfarray.make_surface(img_np.transpose(1, 0, 2))
                surf = pygame.transform.scale(
                    surf, (img_np.shape[1] * self.IMG_SCALE,
                           img_np.shape[0] * self.IMG_SCALE)
                )
                self._screen.blit(surf, (0, 0))
            except Exception:
                pass

        # HUD 렌더링
        structure = info.get("structure", {})
        ms        = info.get("milestones", {})
        hud_y     = self.WIN_H - 80

        def draw_text(text, x, y, color=(220, 220, 220)):
            surf = self._font.render(text, True, color)
            self._screen.blit(surf, (x, y))

        draw_text(f"Step: {self.num_timesteps:,}   Ep reward: {self._ep_reward:+.1f}", 8, hud_y)
        draw_text(
            f"Blocks: {structure.get('total', 0):3d}  "
            f"Floor:{structure.get('floor',0)} "
            f"Wall:{structure.get('wall',0)} "
            f"Roof:{structure.get('roof',0)}",
            8, hud_y + 18,
        )
        enclosed_color = (80, 255, 80) if structure.get("enclosed") else (180, 180, 180)
        draw_text(
            f"Door:{'✓' if ms.get('has_door') else '✗'}  "
            f"Light:{'✓' if ms.get('has_light') else '✗'}  "
            f"Furn:{'✓' if ms.get('has_furniture') else '✗'}  "
            f"ENCLOSED:{'YES' if structure.get('enclosed') else 'no'}",
            8, hud_y + 36, color=enclosed_color,
        )
        mode_color = {"creative": (120, 200, 255), "safe": (120, 255, 120), "survival": (255, 160, 80)}
        env_mode = getattr(self, "_env_mode", "?")
        draw_text(f"Mode: {env_mode.upper()}", 8, hud_y + 54, color=mode_color.get(env_mode, (200, 200, 200)))

        pygame.display.flip()
        return True

    def _on_training_end(self) -> None:
        if self._active:
            pygame.quit()
            self._active = False


# ── 훈련 ─────────────────────────────────────────────────────────
def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = Path(args.log_dir)  / f"{args.env_mode}_{timestamp}"
    save_dir  = Path(args.save_dir) / f"{args.env_mode}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    mode_desc = {
        "creative": "인벤토리 무한 / 항상 낮 / 비행 가능  (Phase 1 사전학습)",
        "safe":     "Peaceful 서바이벌 / 항상 낮           (Phase 2 재료 관리)",
        "survival": "Normal / 낮밤 사이클 / 몬스터 스폰    (Phase 3 최종 목표)",
    }
    print(f"\n🌍 모드: {args.env_mode.upper()} — {mode_desc[args.env_mode]}")
    print(f"📁 로그: {log_dir}")
    print(f"💾 저장: {save_dir}\n")

    _mode = args.env_mode

    def make_fn(port_offset: int):
        def _init():
            return make_house_env(
                port              = args.base_port + port_offset,
                mode              = _mode,
                seed              = (args.seed + port_offset) if args.seed else None,
                max_episode_steps = args.max_episode_steps or None,
                render_action     = True,
            )
        return _init

    if args.n_envs == 1:
        vec_env = DummyVecEnv([make_fn(0)])
    else:
        vec_env = SubprocVecEnv([make_fn(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env, str(log_dir))

    eval_env = DummyVecEnv([make_fn(100)])
    eval_env = VecMonitor(eval_env, str(log_dir / "eval"))

    # PPO 하이퍼파라미터
    hp = PPO_HYPERPARAMS[args.env_mode]
    policy_kwargs = dict(
        features_extractor_class  = HouseCNNExtractor,
        features_extractor_kwargs = {"features_dim": 256},
        net_arch = dict(pi=[128, 128], vf=[128, 128]),
        activation_fn = nn.ReLU,
    )

    if args.resume:
        print(f"🔄 체크포인트 로드: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=args.device)
        # 모드 전환 시 하이퍼파라미터 갱신
        model.learning_rate = hp["learning_rate"]
        model.clip_range    = hp["clip_range"]
        model.ent_coef      = hp["ent_coef"]
    else:
        model = PPO(
            policy          = "MultiInputPolicy",
            env             = vec_env,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = str(log_dir),
            verbose         = 1,
            device          = args.device,
            **hp,
        )

    render_cb = RenderCallback(render_freq=4, verbose=0)
    render_cb._env_mode = args.env_mode   # HUD에 모드 표시용

    callbacks = [
        CheckpointCallback(
            save_freq   = max(10_000 // args.n_envs, 1),
            save_path   = str(save_dir),
            name_prefix = f"house_ppo_{args.env_mode}",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path = str(save_dir / "best"),
            log_path             = str(log_dir / "eval"),
            eval_freq            = max(20_000 // args.n_envs, 1),
            n_eval_episodes      = 3,
            deterministic        = True,
            verbose              = 1,
        ),
        HouseProgressCallback(log_freq=2000, verbose=1),
    ]
    if not args.no_render:
        callbacks.append(render_cb)

    print(f"🚀 훈련 시작! 총 {args.total_steps:,} steps × {args.n_envs}개 환경\n")
    model.learn(
        total_timesteps     = args.total_steps,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = not bool(args.resume),
    )

    final_path = save_dir / f"house_ppo_{args.env_mode}_final"
    model.save(str(final_path))
    print(f"\n✅ 완료. 저장: {final_path}")
    vec_env.close()
    eval_env.close()


# ── 평가 ─────────────────────────────────────────────────────────
def evaluate(args):
    env = make_house_env(
        port              = args.base_port,
        mode              = args.env_mode,
        max_episode_steps = args.max_episode_steps or None,
        render_action     = True,
    )
    model   = PPO.load(args.resume, env=env, device=args.device)
    rewards = []

    for ep in range(args.n_eval_episodes):
        obs, _    = env.reset()
        done      = False
        ep_reward = 0.0
        steps     = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            done       = terminated or truncated
            steps     += 1

        rewards.append(ep_reward)
        ms = info.get("milestones", {})
        st = info.get("structure", {})
        print(
            f"  Ep {ep+1:2d}: R={ep_reward:+.1f}  steps={steps}  "
            f"blocks={st.get('total',0)}  "
            f"enclosed={st.get('enclosed',False)}  "
            f"night={ms.get('night_survived', False)}"
        )

    print(f"\n평균: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    env.close()


# ── 엔트리포인트 ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CraftGround 집 건축 PPO 훈련")

    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train",
        help="train: 훈련 실행  /  eval: 저장된 모델 평가",
    )
    parser.add_argument(
        "--env_mode", choices=list(MODES.keys()), default="creative",
        help=(
            "creative : 인벤토리 무한·항상 낮 (Phase 1)\n"
            "safe     : Peaceful 서바이벌     (Phase 2)\n"
            "survival : Normal·낮밤·몬스터    (Phase 3)"
        ),
    )
    parser.add_argument("--total_steps",       type=int, default=1_000_000)
    parser.add_argument("--n_envs",            type=int, default=1,
                        help="병렬 환경 수 (각각 JVM 실행, RAM 주의)")
    parser.add_argument("--base_port",         type=int, default=8023)
    parser.add_argument("--max_episode_steps", type=int, default=0,
                        help="0 이면 모드 기본값 사용")
    parser.add_argument("--log_dir",           type=str, default="logs")
    parser.add_argument("--save_dir",          type=str, default="checkpoints")
    parser.add_argument("--resume",            type=str, default=None,
                        help="파인튜닝 또는 평가 시 체크포인트 경로")
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--device",            type=str, default="auto")
    parser.add_argument("--n_eval_episodes",   type=int, default=5)
    parser.add_argument(
        "--no_render", action="store_true",
        help="실시간 pygame 영상 창 비활성화 (headless 서버에서 자동 비활성화됨)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        assert args.resume, "--resume 경로를 지정하세요"
        evaluate(args)
