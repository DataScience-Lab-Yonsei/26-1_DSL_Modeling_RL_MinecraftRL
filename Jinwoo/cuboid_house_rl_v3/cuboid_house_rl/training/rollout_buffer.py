"""
Sequence-aware Rollout Buffer for Hierarchical Recurrent PPO.

Stores per-timestep stage_ids so that PPO can route each sample
through the correct stage head during updates.
"""
import torch
import numpy as np
from typing import Dict, Generator

from cuboid_house_rl.config import (
    ACTION_DIMS, NUM_ACTION_DIMS, TOTAL_ACTION_LOGITS,
    FLAT_OBS_SIZE,
    LSTM_HIDDEN_SIZE, SEQUENCE_LENGTH, GAMMA, GAE_LAMBDA,
    NUM_STAGES,
)


class RolloutBuffer:
    """
    Stores rollout data and computes GAE advantages.
    Includes stage_ids for hierarchical network routing.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        device: torch.device,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.sequence_length = sequence_length

        self.pos = 0
        self.full = False

        T, N = buffer_size, num_envs

        self.flat_obs = np.zeros((T, N, FLAT_OBS_SIZE), dtype=np.float32)
        self.actions = np.zeros((T, N, NUM_ACTION_DIMS), dtype=np.int64)
        self.action_masks = np.zeros((T, N, TOTAL_ACTION_LOGITS), dtype=np.bool_)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.bool_)
        self.truncs = np.zeros((T, N), dtype=np.bool_)
        self.stage_ids = np.zeros((T, N), dtype=np.int64)

        # Per-stage LSTM hidden states at the START of each step
        # Shape: (T, NUM_STAGES, N, LSTM_HIDDEN_SIZE)
        self.actor_h = np.zeros((T, NUM_STAGES, N, LSTM_HIDDEN_SIZE), dtype=np.float32)
        self.actor_c = np.zeros((T, NUM_STAGES, N, LSTM_HIDDEN_SIZE), dtype=np.float32)
        self.critic_h = np.zeros((T, NUM_STAGES, N, LSTM_HIDDEN_SIZE), dtype=np.float32)
        self.critic_c = np.zeros((T, NUM_STAGES, N, LSTM_HIDDEN_SIZE), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

    def add(
        self,
        flat_obs: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray,
        truncs: np.ndarray,
        hidden_states: list,
        stage_ids: np.ndarray,
    ):
        """
        Add one timestep of data from all environments.

        Args:
            hidden_states: list[NUM_STAGES] of {"actor": (h,c), "critic": (h,c)}
            stage_ids: (N,) int array — active stage per env
        """
        t = self.pos

        self.flat_obs[t] = flat_obs
        self.actions[t] = actions
        self.action_masks[t] = action_masks
        self.rewards[t] = rewards
        self.values[t] = values
        self.log_probs[t] = log_probs
        self.dones[t] = dones
        self.truncs[t] = truncs
        self.stage_ids[t] = stage_ids

        for s in range(NUM_STAGES):
            self.actor_h[t, s] = hidden_states[s]["actor"][0].squeeze(0).cpu().numpy()
            self.actor_c[t, s] = hidden_states[s]["actor"][1].squeeze(0).cpu().numpy()
            self.critic_h[t, s] = hidden_states[s]["critic"][0].squeeze(0).cpu().numpy()
            self.critic_c[t, s] = hidden_states[s]["critic"][1].squeeze(0).cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """Compute GAE advantages and returns."""
        T = self.buffer_size
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
            else:
                next_values = self.values[t + 1]
                next_ended = (
                    self.dones[t].astype(np.float32)
                    + self.truncs[t].astype(np.float32)
                )
                next_non_terminal = 1.0 - np.clip(next_ended, 0.0, 1.0)

            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

            episode_ended = self.dones[t] | self.truncs[t]
            last_gae = last_gae * (1.0 - episode_ended.astype(np.float32))

        self.returns = self.advantages + self.values

    def generate_batches(
        self, mini_batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield sequential mini-batches for recurrent PPO."""
        T = self.buffer_size
        N = self.num_envs
        L = self.sequence_length

        num_seqs_per_env = T // L
        total_sequences = num_seqs_per_env * N

        sequence_indices = []
        for env in range(N):
            for seq in range(num_seqs_per_env):
                start_t = seq * L
                sequence_indices.append((env, start_t))

        perm = np.random.permutation(total_sequences)
        seqs_per_batch = max(1, mini_batch_size // L)

        for batch_start in range(0, total_sequences, seqs_per_batch):
            batch_indices = perm[batch_start:batch_start + seqs_per_batch]

            batch_flat = []
            batch_actions = []
            batch_masks = []
            batch_log_probs = []
            batch_advantages = []
            batch_returns = []
            batch_values = []
            batch_dones = []
            batch_stage_ids = []
            # Per-stage hidden states: [NUM_STAGES] lists of (h, c)
            batch_actor_h = [[] for _ in range(NUM_STAGES)]
            batch_actor_c = [[] for _ in range(NUM_STAGES)]
            batch_critic_h = [[] for _ in range(NUM_STAGES)]
            batch_critic_c = [[] for _ in range(NUM_STAGES)]

            for idx in batch_indices:
                env, start_t = sequence_indices[idx]
                end_t = start_t + L

                batch_flat.append(self.flat_obs[start_t:end_t, env])
                batch_actions.append(self.actions[start_t:end_t, env])
                batch_masks.append(self.action_masks[start_t:end_t, env])
                batch_log_probs.append(self.log_probs[start_t:end_t, env])
                batch_advantages.append(self.advantages[start_t:end_t, env])
                batch_returns.append(self.returns[start_t:end_t, env])
                batch_values.append(self.values[start_t:end_t, env])
                batch_dones.append(
                    self.dones[start_t:end_t, env] | self.truncs[start_t:end_t, env]
                )
                batch_stage_ids.append(self.stage_ids[start_t:end_t, env])

                for s in range(NUM_STAGES):
                    batch_actor_h[s].append(self.actor_h[start_t, s, env])
                    batch_actor_c[s].append(self.actor_c[start_t, s, env])
                    batch_critic_h[s].append(self.critic_h[start_t, s, env])
                    batch_critic_c[s].append(self.critic_c[start_t, s, env])

            def to_tensor(arrays, dtype=torch.float32):
                return torch.tensor(np.stack(arrays), dtype=dtype, device=self.device)

            num_seqs = len(batch_indices)

            # Build per-stage hidden state tensors
            stage_hidden = []
            for s in range(NUM_STAGES):
                stage_hidden.append({
                    "actor": (
                        torch.tensor(
                            np.stack(batch_actor_h[s]), dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(0),
                        torch.tensor(
                            np.stack(batch_actor_c[s]), dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(0),
                    ),
                    "critic": (
                        torch.tensor(
                            np.stack(batch_critic_h[s]), dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(0),
                        torch.tensor(
                            np.stack(batch_critic_c[s]), dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(0),
                    ),
                })

            yield {
                "flat_obs": to_tensor(batch_flat),
                "actions": to_tensor(batch_actions, dtype=torch.long),
                "action_masks": to_tensor(batch_masks, dtype=torch.bool),
                "old_log_probs": to_tensor(batch_log_probs),
                "advantages": to_tensor(batch_advantages),
                "returns": to_tensor(batch_returns),
                "old_values": to_tensor(batch_values),
                "dones": to_tensor(batch_dones, dtype=torch.bool),
                "stage_ids": to_tensor(batch_stage_ids, dtype=torch.long),
                "stage_hidden": stage_hidden,
                "num_seqs": num_seqs,
                "seq_length": L,
            }

    def reset(self):
        self.pos = 0
        self.full = False
