"""
Recurrent PPO Algorithm.

Implements PPO with:
- Sequence-based training for LSTM continuity
- Separate actor/critic LSTM hidden states
- Action masking
- GAE advantage estimation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from cuboid_house_rl.config import (
    CLIP_RATIO, ENTROPY_COEFF, VALUE_LOSS_COEFF,
    MAX_GRAD_NORM, UPDATE_EPOCHS, MINI_BATCH_SIZE,
    LEARNING_RATE, LSTM_HIDDEN_SIZE,
)
from cuboid_house_rl.models.network import ActorCriticNetwork
from cuboid_house_rl.training.rollout_buffer import RolloutBuffer


class RecurrentPPO:
    """
    Recurrent Proximal Policy Optimization.

    Handles the PPO update logic including:
    - Sequential processing through LSTM with hidden state resets
    - Clipped surrogate objective
    - Value function loss with optional clipping
    - Entropy bonus
    - Gradient clipping
    """

    def __init__(
        self,
        network: ActorCriticNetwork,
        device: torch.device,
        learning_rate: float = LEARNING_RATE,
        clip_ratio: float = CLIP_RATIO,
        entropy_coeff: float = ENTROPY_COEFF,
        value_coeff: float = VALUE_LOSS_COEFF,
        max_grad_norm: float = MAX_GRAD_NORM,
        update_epochs: int = UPDATE_EPOCHS,
        mini_batch_size: int = MINI_BATCH_SIZE,
    ):
        self.network = network
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=learning_rate, eps=1e-5
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO update using data from the rollout buffer.

        Args:
            buffer: filled RolloutBuffer with computed advantages.

        Returns:
            dict of training metrics (losses, clip fraction, entropy, etc.)
        """
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
            "total_loss": 0.0,
        }
        num_updates = 0

        for epoch in range(self.update_epochs):
            for batch in buffer.generate_batches(self.mini_batch_size):
                batch_metrics = self._update_batch(batch)

                for key in metrics:
                    metrics[key] += batch_metrics[key]
                num_updates += 1

        # Average metrics
        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates

        return metrics

    def _update_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Process one mini-batch of sequences through the network and compute PPO loss.

        The key challenge: we must process each sequence step-by-step through
        the LSTM, resetting hidden states at episode boundaries.
        """
        num_seqs = batch["num_seqs"]
        seq_length = batch["seq_length"]

        # Initialize hidden states from stored values at sequence start
        actor_hidden = (
            batch["actor_hidden"][0].detach(),
            batch["actor_hidden"][1].detach(),
        )
        critic_hidden = (
            batch["critic_hidden"][0].detach(),
            batch["critic_hidden"][1].detach(),
        )

        # Process sequence step by step
        all_log_probs = []
        all_values = []
        all_entropy = []

        for t in range(seq_length):
            # Extract timestep t: (num_seqs, ...)
            voxel_t = batch["voxel_grids"][:, t]
            flat_t = batch["flat_features"][:, t]
            actions_t = batch["actions"][:, t]
            masks_t = batch["action_masks"][:, t]

            # Reset hidden states at episode boundaries.
            # If step t-1 was done/truncated, step t is a new episode
            # and needs fresh hidden states.
            if t > 0:
                prev_ended = batch["dones"][:, t - 1]
                done_mask = prev_ended.float().unsqueeze(0).unsqueeze(-1)
                # Zero out hidden state where previous step ended an episode
                actor_hidden = (
                    actor_hidden[0] * (1.0 - done_mask),
                    actor_hidden[1] * (1.0 - done_mask),
                )
                critic_hidden = (
                    critic_hidden[0] * (1.0 - done_mask),
                    critic_hidden[1] * (1.0 - done_mask),
                )

            # Forward pass (evaluate given actions, don't sample new ones)
            hidden_states = {"actor": actor_hidden, "critic": critic_hidden}
            result = self.network.evaluate_actions(
                voxel_t, flat_t, actions_t, hidden_states, action_masks=masks_t
            )

            all_log_probs.append(result["log_probs"])
            all_values.append(result["values"])
            all_entropy.append(result["entropy"])

            # Update hidden states for next timestep
            actor_hidden = result["hidden_states"]["actor"]
            critic_hidden = result["hidden_states"]["critic"]

        # Stack across time: (num_seqs, seq_length)
        new_log_probs = torch.stack(all_log_probs, dim=1)
        new_values = torch.stack(all_values, dim=1)
        new_entropy = torch.stack(all_entropy, dim=1)

        # Flatten for loss computation: (num_seqs * seq_length,)
        new_log_probs = new_log_probs.reshape(-1)
        new_values = new_values.reshape(-1)
        new_entropy = new_entropy.reshape(-1)

        old_log_probs = batch["old_log_probs"].reshape(-1)
        advantages = batch["advantages"].reshape(-1)
        returns = batch["returns"].reshape(-1)
        old_values = batch["old_values"].reshape(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- Policy loss (clipped surrogate) ----
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ---- Value loss (clipped) ----
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -self.clip_ratio, self.clip_ratio
        )
        value_loss_unclipped = (new_values - returns) ** 2
        value_loss_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # ---- Entropy bonus ----
        entropy_loss = -new_entropy.mean()

        # ---- Total loss ----
        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        # ---- Backward pass ----
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ---- Metrics ----
        with torch.no_grad():
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()
            )
            approx_kl = (old_log_probs - new_log_probs).mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "total_loss": total_loss.item(),
        }

    def get_state_dict(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state from checkpoint."""
        self.optimizer.load_state_dict(state_dict)

    def set_learning_rate(self, lr: float):
        """Update learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
