"""
Behaviour Cloning trainer for Hierarchical network.

Trains each stage's actor head on its own expert demonstrations.
The shared MLP is trained by all stages; each stage head only sees
its own demo data.

Usage:
    # Train all stages from a single demo file (with stage_ids)
    python -m cuboid_house_rl.training.train_bc \
        --demo-path demos/demos_all.npz --stage all

    # Train a single stage
    python -m cuboid_house_rl.training.train_bc \
        --demo-path demos/demos_floor.npz --stage floor

    # Resume from previous checkpoint
    python -m cuboid_house_rl.training.train_bc \
        --demo-path demos/demos_all.npz --stage all \
        --resume checkpoints/bc_floor.pt

    # Then PPO fine-tune:
    python -m cuboid_house_rl.training.train \
        --mode train --resume checkpoints/bc_all.pt
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from cuboid_house_rl.config import (
    FLAT_OBS_SIZE, ACTION_DIMS,
    BC_LEARNING_RATE, BC_BATCH_SIZE, BC_EPOCHS, BC_DEMO_DIR,
    LSTM_NUM_LAYERS,
    NUM_STAGES, STAGE_NAMES,
    STAGE_FLOOR, STAGE_WALL, STAGE_DOOR, STAGE_CEILING, STAGE_LOOKING,
)
from cuboid_house_rl.models.network import HierarchicalActorCriticNetwork

STAGE_NAME_TO_ID = {
    "floor": STAGE_FLOOR,
    "wall": STAGE_WALL,
    "walls": STAGE_WALL,       # alias
    "door": STAGE_DOOR,
    "ceiling": STAGE_CEILING,
}


class DemoDataset(Dataset):
    """Dataset of (observation, action, stage_id) triples from expert demos."""

    def __init__(self, demo_path: str, stage_filter: str = None):
        data = np.load(demo_path, allow_pickle=True)
        observations = data["observations"]
        actions = data["actions"]

        # stage_ids: per-transition stage labels
        if "stage_ids" in data:
            stage_ids = data["stage_ids"]
        else:
            # Legacy: infer from stage name
            stage_name = str(data.get("stage", "floor"))
            sid = STAGE_NAME_TO_ID.get(stage_name, 0)
            stage_ids = np.full(len(observations), sid, dtype=np.int64)

        # Always exclude LOOKING (expert-only, not trainable)
        trainable_mask = stage_ids < NUM_STAGES  # LOOKING = 4 >= NUM_STAGES = 4
        if trainable_mask.sum() < len(stage_ids):
            excluded = len(stage_ids) - trainable_mask.sum()
            observations = observations[trainable_mask]
            actions = actions[trainable_mask]
            stage_ids = stage_ids[trainable_mask]
            print(f"  Excluded {excluded} LOOKING transitions (expert-only)")

        # Filter to a single stage if requested
        if stage_filter is not None and stage_filter != "all":
            sid = STAGE_NAME_TO_ID[stage_filter]
            mask = stage_ids == sid
            observations = observations[mask]
            actions = actions[mask]
            stage_ids = stage_ids[mask]
            print(f"  Filtered to stage '{stage_filter}': {mask.sum()} / {len(mask)} transitions")

        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.stage_ids = torch.tensor(stage_ids, dtype=torch.long)

        print(f"Loaded {len(self.observations)} transitions from {demo_path}")
        print(f"  Obs shape:    {self.observations.shape}")
        print(f"  Action shape: {self.actions.shape}")

        # Stage distribution
        for s in range(NUM_STAGES):
            count = (self.stage_ids == s).sum().item()
            if count > 0:
                name = STAGE_NAMES[s] if s < len(STAGE_NAMES) else f"stage_{s}"
                print(f"  {name}: {count} ({count/len(self.stage_ids):.1%})")

        assert self.observations.shape[1] == FLAT_OBS_SIZE, \
            f"Obs dim mismatch: got {self.observations.shape[1]}, expected {FLAT_OBS_SIZE}"
        assert self.actions.shape[1] == len(ACTION_DIMS), \
            f"Action dim mismatch: got {self.actions.shape[1]}, expected {len(ACTION_DIMS)}"

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.stage_ids[idx]


def compute_bc_loss(network, obs, actions, stage_ids, device):
    """
    Per-dimension cross-entropy loss, routed through stage heads.
    Only trains actor path (shared MLP + stage actor LSTM/MLP/head).
    """
    shared = network.shared_mlp(obs)
    shared_seq = shared.unsqueeze(1)  # (B, 1, feat)

    B = obs.shape[0]
    n = network.lstm_num_layers
    from cuboid_house_rl.config import LSTM_HIDDEN_SIZE, TOTAL_ACTION_LOGITS

    all_logits = torch.zeros(B, TOTAL_ACTION_LOGITS, device=device)

    for s in range(network.num_stages):
        mask = (stage_ids == s)
        if not mask.any():
            continue

        idx = mask.nonzero(as_tuple=True)[0]
        K = idx.shape[0]

        actor_hidden = (
            torch.zeros(n, K, LSTM_HIDDEN_SIZE, device=device),
            torch.zeros(n, K, LSTM_HIDDEN_SIZE, device=device),
        )

        logits, _ = network.stage_heads[s].forward_actor(shared_seq[idx], actor_hidden)
        all_logits[idx] = logits

    split_logits = torch.split(all_logits, ACTION_DIMS, dim=-1)
    total_loss = 0.0
    per_dim_loss = []
    per_dim_acc = []

    for i, (dim_logits, dim_size) in enumerate(zip(split_logits, ACTION_DIMS)):
        target = actions[:, i]
        loss = nn.functional.cross_entropy(dim_logits, target)
        total_loss += loss
        per_dim_loss.append(loss.item())

        pred = dim_logits.argmax(dim=-1)
        acc = (pred == target).float().mean().item()
        per_dim_acc.append(acc)

    return total_loss, per_dim_loss, per_dim_acc


def train_bc(args):
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")

    # Load demos
    dataset = DemoDataset(args.demo_path, stage_filter=args.stage)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(dataset) > args.batch_size,
        num_workers=0,
    )

    # Create hierarchical network
    network = HierarchicalActorCriticNetwork(lstm_num_layers=args.lstm_layers).to(device)

    # Resume from previous checkpoint
    if args.resume:
        print(f"Loading weights from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["model_state_dict"])
        prev_stage = ckpt.get("stage", "unknown")
        prev_loss = ckpt.get("loss", "?")
        print(f"  Loaded stage={prev_stage}, loss={prev_loss}")

    params = network.count_parameters()
    print(f"Network parameters: {params['total']:,}")

    # Optimizer: shared MLP + all stage actor parameters
    actor_params = list(network.shared_mlp.parameters())
    for head in network.stage_heads:
        actor_params += list(head.actor_lstm.parameters())
        actor_params += list(head.actor_mlp.parameters())
        actor_params += list(head.action_head.parameters())

    optimizer = torch.optim.Adam(actor_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    dim_names = ["fwd/back", "left/right", "jump", "sneak",
                 "interact", "hotbar", "pitch", "yaw"]

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    print(f"\nBC training: {args.epochs} epochs, "
          f"{len(dataset)} transitions, batch_size={args.batch_size}")

    for epoch in range(args.epochs):
        network.train()
        epoch_loss = 0.0
        epoch_dim_acc = np.zeros(len(ACTION_DIMS))
        num_batches = 0

        for obs_batch, action_batch, stage_batch in dataloader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            stage_batch = stage_batch.to(device)

            loss, dim_losses, dim_accs = compute_bc_loss(
                network, obs_batch, action_batch, stage_batch, device
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dim_acc += np.array(dim_accs)
            num_batches += 1

        scheduler.step()
        epoch_loss /= max(num_batches, 1)
        epoch_dim_acc /= max(num_batches, 1)

        # Log (compact)
        key_accs = (
            f"mv:{epoch_dim_acc[0]:.0%}/{epoch_dim_acc[1]:.0%} "
            f"int:{epoch_dim_acc[4]:.0%} "
            f"cam:{epoch_dim_acc[6]:.0%}/{epoch_dim_acc[7]:.0%}"
        )
        print(
            f"Epoch {epoch + 1:>3}/{args.epochs} | "
            f"loss={epoch_loss:.4f} | {key_accs}"
        )

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model_state_dict": network.state_dict(),
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "stage": args.stage,
                "dim_accuracy": epoch_dim_acc.tolist(),
            }, os.path.join(args.checkpoint_dir, f"bc_{args.stage}.pt"))

    # Final save
    final_path = os.path.join(args.checkpoint_dir, f"bc_{args.stage}_final.pt")
    torch.save({
        "model_state_dict": network.state_dict(),
        "epoch": args.epochs,
        "loss": epoch_loss,
        "stage": args.stage,
    }, final_path)

    print(f"\nBC [{args.stage}] complete! Best loss: {best_loss:.4f}")
    print(f"  Best: checkpoints/bc_{args.stage}.pt")
    print(f"  Final: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Behaviour Cloning Training (Hierarchical)")
    parser.add_argument("--demo-path", type=str, required=True)
    parser.add_argument("--stage", type=str, default="all",
                        choices=["floor", "wall", "walls", "door", "ceiling", "all"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Previous checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BC_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BC_LEARNING_RATE)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--lstm-layers", type=int, default=LSTM_NUM_LAYERS)
    args = parser.parse_args()
    train_bc(args)


if __name__ == "__main__":
    main()
