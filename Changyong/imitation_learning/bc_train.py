"""
Behavioral Cloning 학습 (41차원 벡터 obs 기준).

사용법:
    python bc_train.py [에폭수] [배치사이즈]
    ex) python bc_train.py 50 64
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from building_hrl import ACTIONS
from config import (
    DEMO_PATH, BC_MODEL_PATH, DEVICE, OBS_DIM,
    BC_N_EPOCHS, BC_BATCH_SIZE, BC_LR, BC_VAL_RATIO,
)
from model import MLPPolicy

N_ACTIONS = len(ACTIONS)


def load_demos() -> TensorDataset:
    if not os.path.exists(DEMO_PATH):
        print(f"[ERROR] 데모 없음: {DEMO_PATH}  →  collect_demos.py 먼저 실행")
        sys.exit(1)

    data = np.load(DEMO_PATH)
    obs     = torch.from_numpy(data["observations"])   # (N, 41)
    actions = torch.from_numpy(data["actions"]).long() # (N,)

    print(f"[bc_train] 데모 로드: {len(actions):,} 스텝  obs_dim={obs.shape[1]}")
    print("[bc_train] 액션 분포:")
    for i, name in enumerate(ACTIONS):
        cnt = (data["actions"] == i).sum()
        print(f"  {i:2d} {name:15s}: {cnt:5d} ({cnt/len(actions)*100:4.1f}%)")

    return TensorDataset(obs, actions)


def train(n_epochs: int = BC_N_EPOCHS, batch_size: int = BC_BATCH_SIZE, lr: float = BC_LR):
    print(f"[bc_train] DEVICE={DEVICE} | epochs={n_epochs} | batch={batch_size}")
    dataset = load_demos()

    N = len(dataset)
    val_n, train_n = max(1, int(N * BC_VAL_RATIO)), N - max(1, int(N * BC_VAL_RATIO))
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = MLPPolicy(obs_dim=OBS_DIM, n_actions=N_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    best_val  = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0

        for obs_b, acts_b in train_loader:
            obs_b, acts_b = obs_b.to(DEVICE), acts_b.to(DEVICE)
            logits = model(obs_b)
            loss   = criterion(logits, acts_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(acts_b)
            correct    += (logits.argmax(1) == acts_b).sum().item()

        scheduler.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for obs_b, acts_b in val_loader:
                obs_b, acts_b = obs_b.to(DEVICE), acts_b.to(DEVICE)
                val_correct += (model(obs_b).argmax(1) == acts_b).sum().item()

        train_acc = correct     / train_n
        val_acc   = val_correct / val_n
        print(f"  Epoch {epoch:3d}/{n_epochs} | loss={total_loss/train_n:.4f} | "
              f"train={train_acc:.3f} | val={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            model.save(BC_MODEL_PATH)

    print(f"\n[bc_train] 완료 | best val_acc={best_val:.3f}")


if __name__ == "__main__":
    n_epochs   = int(sys.argv[1]) if len(sys.argv) > 1 else BC_N_EPOCHS
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else BC_BATCH_SIZE
    train(n_epochs=n_epochs, batch_size=batch_size)
