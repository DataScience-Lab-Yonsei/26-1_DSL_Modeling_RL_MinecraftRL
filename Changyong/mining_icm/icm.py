"""
icm.py — Intrinsic Curiosity Module (ICM) for Minecraft mining

논문: Pathak et al. 2017 "Curiosity-driven Exploration by Self-Supervised Prediction"

구성:
  PhiEncoder   : (image, state) → 256-dim feature embedding
  ForwardModel : (phi_s, a_onehot) → pred_phi_s'
  InverseModel : (phi_s, phi_s') → pred_action_logits

intrinsic reward = eta * || pred_phi_s' - phi_s' ||^2

비교 (Seoyeon real_mining_v1 vs 이 ICM 버전):
  Seoyeon  → 수동 설계된 탐험 보너스: visited cell set + lambda 감쇠 (LAYER 2)
  ICM      → 자동 학습: 예측 못하는 상태 전이 = 새로운 → 높은 내재 보상
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# =============================================================
# 1. PhiEncoder: 관찰 공간 → 특징 벡터
# =============================================================

class PhiEncoder(nn.Module):
    """
    (image H×W×C, state D) → feat_dim 벡터

    SB3 정책 네트워크와 별도로 ICM 전용 인코더를 유지.
    (파라미터 공유는 학습 불안정 초래 가능 → 분리 설계)
    """

    def __init__(
        self,
        img_h: int = 64,
        img_w: int = 114,
        img_c: int = 3,
        state_dim: int = 10,
        feat_dim: int = 256,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # 경량 CNN (ICM용 — SB3 정책 CNN보다 작게)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_c, 32, kernel_size=8, stride=4), nn.ELU(),
            nn.Conv2d(32,    64, kernel_size=4, stride=2), nn.ELU(),
            nn.Conv2d(64,    64, kernel_size=3, stride=1), nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            _dummy = torch.zeros(1, img_c, img_h, img_w)
            cnn_out = self.cnn(_dummy).shape[1]

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, feat_dim), nn.ELU(),
        )

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        img  : (B, H, W, C) uint8 또는 float32
        state: (B, D) float32
        → (B, feat_dim)
        """
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        # (B, H, W, C) → (B, C, H, W)
        if img.shape[-1] in (1, 3, 4):
            img = img.permute(0, 3, 1, 2)
        cnn_feat = self.cnn(img)
        state_feat = self.state_mlp(state.float())
        return self.fusion(torch.cat([cnn_feat, state_feat], dim=-1))


# =============================================================
# 2. ForwardModel: (phi_s, a) → pred_phi_s'
# =============================================================

class ForwardModel(nn.Module):
    """예측 오류 = 내재적 보상의 크기."""

    def __init__(self, feat_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + n_actions, 256), nn.ELU(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, phi_s: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_s, a_onehot], dim=-1))


# =============================================================
# 3. InverseModel: (phi_s, phi_s') → pred_action
# =============================================================

class InverseModel(nn.Module):
    """
    행동 예측 → 상태 표현이 행동 관련 정보를 담도록 강제.
    환경 노이즈(나뭇잎 흔들림 등) 를 phi에서 제거하는 효과.
    """

    def __init__(self, feat_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, 256), nn.ELU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, phi_s: torch.Tensor, phi_s_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_s, phi_s_next], dim=-1))


# =============================================================
# 4. ICMModule: 세 네트워크 통합 + 학습
# =============================================================

class ICMModule(nn.Module):
    """
    ICM 전체 (PhiEncoder + ForwardModel + InverseModel).

    intrinsic_reward = eta * || forward(phi_s, a) - phi_s' ||^2
    loss = beta * forward_loss + (1 - beta) * inverse_loss

    하이퍼파라미터:
      eta   : 내재 보상 스케일 (기본 0.01)
      beta  : forward/inverse 손실 가중치 비율 (기본 0.2)
    """

    def __init__(
        self,
        n_actions: int,
        img_h: int = 64,
        img_w: int = 114,
        img_c: int = 3,
        state_dim: int = 10,
        feat_dim: int = 256,
        eta: float = 0.01,
        beta: float = 0.2,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_actions = n_actions
        self.eta       = eta
        self.beta      = beta
        self.device    = torch.device(device)

        self.phi     = PhiEncoder(img_h, img_w, img_c, state_dim, feat_dim)
        self.forward_m  = ForwardModel(feat_dim, n_actions)
        self.inverse_m  = InverseModel(feat_dim, n_actions)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ── 내재적 보상 계산 (no_grad, 빠름) ───────────────────────
    @torch.no_grad()
    def intrinsic_reward(
        self,
        obs_img: np.ndarray,      # (B, H, W, C) uint8
        obs_state: np.ndarray,    # (B, D) float32
        actions: np.ndarray,      # (B,) int
        next_img: np.ndarray,     # (B, H, W, C) uint8
        next_state: np.ndarray,   # (B, D) float32
    ) -> np.ndarray:              # (B,) float32
        img_t  = torch.as_tensor(obs_img,    device=self.device)
        st_t   = torch.as_tensor(obs_state,  device=self.device)
        nimg_t = torch.as_tensor(next_img,   device=self.device)
        nst_t  = torch.as_tensor(next_state, device=self.device)

        phi_s      = self.phi(img_t,  st_t)
        phi_s_next = self.phi(nimg_t, nst_t)

        a_onehot = F.one_hot(
            torch.as_tensor(actions, device=self.device, dtype=torch.long),
            num_classes=self.n_actions,
        ).float()

        pred_phi_next = self.forward_m(phi_s, a_onehot)

        # MSE per sample
        reward = self.eta * (pred_phi_next - phi_s_next).pow(2).mean(dim=-1)
        return reward.cpu().numpy().astype(np.float32)

    # ── ICM 파라미터 업데이트 ───────────────────────────────────
    def update(
        self,
        obs_img: np.ndarray,
        obs_state: np.ndarray,
        actions: np.ndarray,
        next_img: np.ndarray,
        next_state: np.ndarray,
    ) -> dict[str, float]:
        img_t  = torch.as_tensor(obs_img,    device=self.device).float()
        st_t   = torch.as_tensor(obs_state,  device=self.device)
        nimg_t = torch.as_tensor(next_img,   device=self.device).float()
        nst_t  = torch.as_tensor(next_state, device=self.device)
        act_t  = torch.as_tensor(actions,    device=self.device, dtype=torch.long)

        phi_s      = self.phi(img_t,  st_t)
        phi_s_next = self.phi(nimg_t, nst_t)

        a_onehot = F.one_hot(act_t, num_classes=self.n_actions).float()

        # Forward loss: 다음 상태 특징 예측
        pred_phi_next = self.forward_m(phi_s.detach(), a_onehot)
        fwd_loss = F.mse_loss(pred_phi_next, phi_s_next.detach())

        # Inverse loss: 행동 예측 (cross-entropy)
        pred_act = self.inverse_m(phi_s, phi_s_next)
        inv_loss = F.cross_entropy(pred_act, act_t)

        loss = self.beta * fwd_loss + (1.0 - self.beta) * inv_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {
            "icm/loss":         loss.item(),
            "icm/fwd_loss":     fwd_loss.item(),
            "icm/inv_loss":     inv_loss.item(),
        }


# =============================================================
# 5. ICMReplayBuffer: (s, a, s') 저장
# =============================================================

class ICMReplayBuffer:
    """
    순환 버퍼 — ICM 업데이트용 최근 전이 저장.
    DummyVecEnv(n=1) 기준 설계.
    """

    def __init__(self, capacity: int = 8000):
        self.capacity = capacity
        self._obs_img:    deque = deque(maxlen=capacity)
        self._obs_state:  deque = deque(maxlen=capacity)
        self._actions:    deque = deque(maxlen=capacity)
        self._next_img:   deque = deque(maxlen=capacity)
        self._next_state: deque = deque(maxlen=capacity)

    def push(
        self,
        obs_img:    np.ndarray,   # (H, W, C) uint8
        obs_state:  np.ndarray,   # (D,) float32
        action:     int,
        next_img:   np.ndarray,
        next_state: np.ndarray,
    ):
        self._obs_img.append(obs_img)
        self._obs_state.append(obs_state)
        self._actions.append(action)
        self._next_img.append(next_img)
        self._next_state.append(next_state)

    def sample(self, batch_size: int):
        n = len(self._obs_img)
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return (
            np.stack([self._obs_img[i]    for i in idx]),
            np.stack([self._obs_state[i]  for i in idx]),
            np.array([self._actions[i]    for i in idx], dtype=np.int64),
            np.stack([self._next_img[i]   for i in idx]),
            np.stack([self._next_state[i] for i in idx]),
        )

    def __len__(self):
        return len(self._obs_img)
