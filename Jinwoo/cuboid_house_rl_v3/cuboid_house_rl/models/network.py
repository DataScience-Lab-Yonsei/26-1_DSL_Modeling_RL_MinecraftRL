"""
Actor-Critic Network (V3) — Hierarchical.

Architecture:
  Shared MLP (all stages) → per-stage Actor LSTM → Actor MLP → Action Head
                           → per-stage Critic LSTM → Critic MLP → Value Head

Each construction stage (floor, walls, ceiling) has its own LSTM, MLP, and
output heads. The shared MLP extracts common features; stage-specific modules
specialise for each stage's distinct behaviour.

During rollout the current stage_id selects which head to use. During PPO
update, gradients only flow through the shared MLP + the active stage's head.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict

from cuboid_house_rl.config import (
    FLAT_OBS_SIZE,
    SHARED_MLP_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    ACTOR_MLP_SIZE, CRITIC_MLP_SIZE,
    ACTION_DIMS, TOTAL_ACTION_LOGITS, INITIAL_BIAS,
    NUM_STAGES, STAGE_NAMES,
)
from cuboid_house_rl.models.action_dist import MaskedMultiDiscreteDistribution


class StageHead(nn.Module):
    """One stage's actor + critic (LSTM → MLP → head)."""

    def __init__(self, input_size, lstm_num_layers):
        super().__init__()
        self.lstm_num_layers = lstm_num_layers

        self.actor_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.actor_mlp = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, ACTOR_MLP_SIZE),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(ACTOR_MLP_SIZE, TOTAL_ACTION_LOGITS)

        self.critic_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.critic_mlp = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, CRITIC_MLP_SIZE),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(CRITIC_MLP_SIZE, 1)

    def forward_actor(self, shared_seq, actor_hidden):
        """Returns (action_logits, new_actor_hidden)."""
        actor_out, new_h = self.actor_lstm(shared_seq, actor_hidden)
        actor_out = actor_out.squeeze(1)
        logits = self.action_head(self.actor_mlp(actor_out))
        return logits, new_h

    def forward_critic(self, shared_seq, critic_hidden):
        """Returns (values, new_critic_hidden)."""
        critic_out, new_h = self.critic_lstm(shared_seq, critic_hidden)
        critic_out = critic_out.squeeze(1)
        values = self.value_head(self.critic_mlp(critic_out)).squeeze(-1)
        return values, new_h


class HierarchicalActorCriticNetwork(nn.Module):
    """Shared MLP + per-stage (LSTM → MLP → Head) for actor and critic."""

    def __init__(self, obs_size=FLAT_OBS_SIZE, lstm_num_layers=LSTM_NUM_LAYERS):
        super().__init__()
        self.lstm_num_layers = lstm_num_layers
        self.obs_size = obs_size
        self.num_stages = NUM_STAGES

        # Shared feature extractor
        self.shared_mlp = nn.Sequential(
            nn.Linear(obs_size, SHARED_MLP_SIZE),
            nn.ReLU(),
            nn.Linear(SHARED_MLP_SIZE, SHARED_MLP_SIZE),
            nn.ReLU(),
        )

        # Per-stage heads
        self.stage_heads = nn.ModuleList([
            StageHead(SHARED_MLP_SIZE, lstm_num_layers)
            for _ in range(NUM_STAGES)
        ])

        self._init_weights()
        self._apply_initial_bias()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif "bias" in name:
                        nn.init.zeros_(param)
        # Small init for value heads
        for head in self.stage_heads:
            nn.init.orthogonal_(head.value_head.weight, gain=0.01)
            nn.init.zeros_(head.value_head.bias)

    def _apply_initial_bias(self, bias=None):
        if bias is None:
            bias = INITIAL_BIAS
        bias_tensor = torch.tensor(bias, dtype=torch.float32)
        with torch.no_grad():
            for head in self.stage_heads:
                head.action_head.bias.copy_(bias_tensor)

    def get_initial_hidden_state(self, batch_size=1, device=None):
        """Returns per-stage hidden states: list of dicts."""
        if device is None:
            device = next(self.parameters()).device
        n = self.lstm_num_layers

        def _zeros():
            return (
                torch.zeros(n, batch_size, LSTM_HIDDEN_SIZE, device=device),
                torch.zeros(n, batch_size, LSTM_HIDDEN_SIZE, device=device),
            )

        return [
            {"actor": _zeros(), "critic": _zeros()}
            for _ in range(self.num_stages)
        ]

    def forward(self, flat_obs, hidden_states, stage_ids,
                action_masks=None, deterministic=False):
        """
        Forward pass routing each sample through its stage head.

        Args:
            flat_obs: (B, obs_size)
            hidden_states: list[NUM_STAGES] of {"actor": (h,c), "critic": (h,c)}
                           each h,c: (num_layers, B, hidden_size)
            stage_ids: (B,) int tensor — stage per sample
            action_masks: (B, TOTAL_ACTION_LOGITS) bool or None
            deterministic: if True, use mode instead of sample

        Returns:
            dict with actions, log_probs, values, entropy, hidden_states
        """
        B = flat_obs.shape[0]
        device = flat_obs.device

        shared = self.shared_mlp(flat_obs)
        shared_seq = shared.unsqueeze(1)  # (B, 1, feat)

        # Output buffers
        all_logits = torch.zeros(B, TOTAL_ACTION_LOGITS, device=device)
        all_values = torch.zeros(B, device=device)
        new_hidden = [
            {
                "actor": (h["actor"][0].clone(), h["actor"][1].clone()),
                "critic": (h["critic"][0].clone(), h["critic"][1].clone()),
            }
            for h in hidden_states
        ]

        # Route per stage
        for s in range(self.num_stages):
            mask = (stage_ids == s)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            s_shared = shared_seq[idx]  # (K, 1, feat)

            # Gather hidden states for this stage's samples
            s_actor_h = (
                hidden_states[s]["actor"][0][:, idx, :].contiguous(),
                hidden_states[s]["actor"][1][:, idx, :].contiguous(),
            )
            s_critic_h = (
                hidden_states[s]["critic"][0][:, idx, :].contiguous(),
                hidden_states[s]["critic"][1][:, idx, :].contiguous(),
            )

            logits, new_actor_h = self.stage_heads[s].forward_actor(s_shared, s_actor_h)
            values, new_critic_h = self.stage_heads[s].forward_critic(s_shared, s_critic_h)

            all_logits[idx] = logits
            all_values[idx] = values

            # Write back updated hidden states
            new_hidden[s]["actor"][0][:, idx, :] = new_actor_h[0]
            new_hidden[s]["actor"][1][:, idx, :] = new_actor_h[1]
            new_hidden[s]["critic"][0][:, idx, :] = new_critic_h[0]
            new_hidden[s]["critic"][1][:, idx, :] = new_critic_h[1]

        dist = MaskedMultiDiscreteDistribution(all_logits, action_masks, ACTION_DIMS)
        actions = dist.mode() if deterministic else dist.sample()

        return {
            "actions": actions,
            "log_probs": dist.log_prob(actions),
            "values": all_values,
            "entropy": dist.entropy(),
            "hidden_states": new_hidden,
        }

    def evaluate_actions(self, flat_obs, actions, hidden_states, stage_ids,
                         action_masks=None):
        """
        Re-evaluate actions for PPO update. Same routing as forward.

        Args:
            flat_obs: (B, obs_size)
            actions: (B, num_action_dims)
            hidden_states: list[NUM_STAGES] of {"actor": (h,c), "critic": (h,c)}
            stage_ids: (B,) int tensor
            action_masks: (B, TOTAL_ACTION_LOGITS) bool or None
        """
        B = flat_obs.shape[0]
        device = flat_obs.device

        shared = self.shared_mlp(flat_obs)
        shared_seq = shared.unsqueeze(1)

        all_logits = torch.zeros(B, TOTAL_ACTION_LOGITS, device=device)
        all_values = torch.zeros(B, device=device)
        new_hidden = [
            {
                "actor": (h["actor"][0].clone(), h["actor"][1].clone()),
                "critic": (h["critic"][0].clone(), h["critic"][1].clone()),
            }
            for h in hidden_states
        ]

        for s in range(self.num_stages):
            mask = (stage_ids == s)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            s_shared = shared_seq[idx]

            s_actor_h = (
                hidden_states[s]["actor"][0][:, idx, :].contiguous(),
                hidden_states[s]["actor"][1][:, idx, :].contiguous(),
            )
            s_critic_h = (
                hidden_states[s]["critic"][0][:, idx, :].contiguous(),
                hidden_states[s]["critic"][1][:, idx, :].contiguous(),
            )

            logits, new_actor_h = self.stage_heads[s].forward_actor(s_shared, s_actor_h)
            values, new_critic_h = self.stage_heads[s].forward_critic(s_shared, s_critic_h)

            all_logits[idx] = logits
            all_values[idx] = values

            new_hidden[s]["actor"][0][:, idx, :] = new_actor_h[0]
            new_hidden[s]["actor"][1][:, idx, :] = new_actor_h[1]
            new_hidden[s]["critic"][0][:, idx, :] = new_critic_h[0]
            new_hidden[s]["critic"][1][:, idx, :] = new_critic_h[1]

        dist = MaskedMultiDiscreteDistribution(all_logits, action_masks, ACTION_DIMS)

        return {
            "log_probs": dist.log_prob(actions),
            "values": all_values,
            "entropy": dist.entropy(),
            "hidden_states": new_hidden,
        }

    def get_value(self, flat_obs, hidden_states, stage_ids):
        """Get values only (for GAE bootstrap)."""
        B = flat_obs.shape[0]
        device = flat_obs.device

        shared = self.shared_mlp(flat_obs)
        shared_seq = shared.unsqueeze(1)

        all_values = torch.zeros(B, device=device)
        new_hidden = [
            {
                "actor": h["actor"],
                "critic": (h["critic"][0].clone(), h["critic"][1].clone()),
            }
            for h in hidden_states
        ]

        for s in range(self.num_stages):
            mask = (stage_ids == s)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            s_shared = shared_seq[idx]

            s_critic_h = (
                hidden_states[s]["critic"][0][:, idx, :].contiguous(),
                hidden_states[s]["critic"][1][:, idx, :].contiguous(),
            )

            values, new_critic_h = self.stage_heads[s].forward_critic(s_shared, s_critic_h)
            all_values[idx] = values

            new_hidden[s]["critic"][0][:, idx, :] = new_critic_h[0]
            new_hidden[s]["critic"][1][:, idx, :] = new_critic_h[1]

        return all_values, new_hidden

    def count_parameters(self):
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        result = {"shared_mlp": _count(self.shared_mlp), "total": 0}
        for i, head in enumerate(self.stage_heads):
            name = STAGE_NAMES[i] if i < len(STAGE_NAMES) else f"stage_{i}"
            result[f"{name}_actor_lstm"] = _count(head.actor_lstm)
            result[f"{name}_actor_mlp"] = _count(head.actor_mlp)
            result[f"{name}_action_head"] = _count(head.action_head)
            result[f"{name}_critic_lstm"] = _count(head.critic_lstm)
            result[f"{name}_critic_mlp"] = _count(head.critic_mlp)
            result[f"{name}_value_head"] = _count(head.value_head)

        result["total"] = sum(p.numel() for p in self.parameters())
        return result
