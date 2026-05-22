"""
정책 모델 모음.

BCPolicy  : CNN + MLP (image 84x84 + subgoal 4 → n_actions)  ← 기존
MLPPolicy : MLP only  (벡터 41차원 → n_actions)              ← 팀원 방식 (권장)
"""
import torch
import torch.nn as nn
from config import DEVICE


# ══════════════════════════════════════════════════════════════════
# MLPPolicy  (팀원 obs 기준, 41차원 입력)
# ══════════════════════════════════════════════════════════════════
class MLPPolicy(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) → logits: (B, n_actions)"""
        return self.net(obs)

    @torch.no_grad()
    def predict(self, obs: "np.ndarray") -> int:
        """1D numpy array → action int"""
        self.eval()
        t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        return int(self.forward(t).argmax(dim=-1).item())

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[MLPPolicy] 저장: {path}")

    @classmethod
    def load(cls, path: str, obs_dim: int, n_actions: int) -> "MLPPolicy":
        model = cls(obs_dim=obs_dim, n_actions=n_actions).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        print(f"[MLPPolicy] 로드: {path}")
        return model


# ══════════════════════════════════════════════════════════════════
# BCPolicy  (기존 image+subgoal 방식, 필요 시 사용)
# ══════════════════════════════════════════════════════════════════
class BCPolicy(nn.Module):

    def __init__(self, n_actions: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.subgoal_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, image: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(image), self.subgoal_mlp(subgoal)], dim=-1))

    @torch.no_grad()
    def predict(self, obs: dict) -> int:
        self.eval()
        img = torch.from_numpy(obs["image"]).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
        sg  = torch.from_numpy(obs["subgoal"]).float().unsqueeze(0).to(DEVICE)
        return int(self.forward(img, sg).argmax(dim=-1).item())

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, n_actions: int) -> "BCPolicy":
        model = cls(n_actions=n_actions).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model
