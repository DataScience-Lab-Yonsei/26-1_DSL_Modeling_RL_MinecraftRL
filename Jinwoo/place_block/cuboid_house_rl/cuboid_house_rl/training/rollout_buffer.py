"""
Sequence-aware Rollout Buffer for Recurrent PPO.

Stores transitions from parallel environments and generates
sequential mini-batches that preserve LSTM hidden state continuity.
"""
import torch
import numpy as np
from typing import Dict, Optional, Generator

from cuboid_house_rl.config import (
    ACTION_DIMS, NUM_ACTION_DIMS, TOTAL_ACTION_LOGITS,
    LOCAL_WINDOW_SIZE, STACKED_CHANNELS, NON_VOXEL_SIZE,
    LSTM_HIDDEN_SIZE, SEQUENCE_LENGTH, GAMMA, GAE_LAMBDA,
)


class RolloutBuffer:
    """
    Stores rollout data and computes GAE advantages.

    For recurrent PPO, data must be processed in sequential chunks
    to maintain LSTM state continuity. This buffer stores transitions
    per-environment and yields sequences of fixed length.
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
        """
        Args:
            buffer_size: steps per environment per rollout.
            num_envs: number of parallel environments.
            device: torch device.
            gamma: discount factor.
            gae_lambda: GAE lambda.
            sequence_length: LSTM sequence length for training.
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.sequence_length = sequence_length

        self.pos = 0
        self.full = False

        # Pre-allocate storage (all on CPU, moved to device during iteration)
        T, N = buffer_size, num_envs
        W = LOCAL_WINDOW_SIZE

        self.voxel_grids = np.zeros(
            (T, N, STACKED_CHANNELS, W, W, W), dtype=np.float32
        )
        self.flat_features = np.zeros(
            (T, N, NON_VOXEL_SIZE), dtype=np.float32
        )
        self.actions = np.zeros((T, N, NUM_ACTION_DIMS), dtype=np.int64)
        self.action_masks = np.zeros(
            (T, N, TOTAL_ACTION_LOGITS), dtype=np.bool_
        )
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.bool_)  # episode terminated
        self.truncs = np.zeros((T, N), dtype=np.bool_)  # episode truncated

        # LSTM hidden states at the START of each step
        # Shape: (T, N, LSTM_HIDDEN_SIZE) for both h and c
        self.actor_h = np.zeros((T, N, LSTM_HIDDEN_SIZE), dtype=np.float32)
        self.actor_c = np.zeros((T, N, LSTM_HIDDEN_SIZE), dtype=np.float32)
        self.critic_h = np.zeros((T, N, LSTM_HIDDEN_SIZE), dtype=np.float32)
        self.critic_c = np.zeros((T, N, LSTM_HIDDEN_SIZE), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

    def add(
        self,
        voxel_grids: np.ndarray,
        flat_features: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray,
        truncs: np.ndarray,
        actor_hidden: tuple,
        critic_hidden: tuple,
    ):
        """
        Add one timestep of data from all environments.

        Args:
            voxel_grids: (N, C, W, W, W)
            flat_features: (N, NON_VOXEL_SIZE)
            actions: (N, NUM_ACTION_DIMS)
            action_masks: (N, TOTAL_ACTION_LOGITS)
            rewards: (N,)
            values: (N,)
            log_probs: (N,)
            dones: (N,) boolean
            truncs: (N,) boolean
            actor_hidden: (h, c) each (1, N, LSTM_HIDDEN_SIZE) tensor
            critic_hidden: (h, c) each (1, N, LSTM_HIDDEN_SIZE) tensor
        """
        t = self.pos

        self.voxel_grids[t] = voxel_grids
        self.flat_features[t] = flat_features
        self.actions[t] = actions
        self.action_masks[t] = action_masks
        self.rewards[t] = rewards
        self.values[t] = values
        self.log_probs[t] = log_probs
        self.dones[t] = dones
        self.truncs[t] = truncs

        # Store hidden states (squeeze out the num_layers dimension)
        self.actor_h[t] = actor_hidden[0].squeeze(0).cpu().numpy()
        self.actor_c[t] = actor_hidden[1].squeeze(0).cpu().numpy()
        self.critic_h[t] = critic_hidden[0].squeeze(0).cpu().numpy()
        self.critic_c[t] = critic_hidden[1].squeeze(0).cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Compute GAE advantages and returns.

        Args:
            last_values: (N,) bootstrap values from final observation.
            last_dones: (N,) whether final step was terminal.
        """
        T = self.buffer_size
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
            else:
                next_values = self.values[t + 1]
                # Both terminated AND truncated at step t cut the bootstrap,
                # because step t+1 belongs to a new episode.
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

            # Store advantage BEFORE resetting the GAE accumulator,
            # so that the terminal step's advantage (which includes
            # terminal rewards like house_complete or stuck_penalty)
            # is preserved.
            self.advantages[t] = last_gae

            # Reset GAE accumulator at episode boundaries so that
            # advantages from future steps don't leak into prior episodes.
            episode_ended = self.dones[t] | self.truncs[t]
            last_gae = last_gae * (1.0 - episode_ended.astype(np.float32))

        self.returns = self.advantages + self.values

    def generate_batches(
        self, mini_batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield sequential mini-batches for recurrent PPO training.

        Data is split into sequences of `sequence_length` steps.
        Each mini-batch contains multiple sequences from different
        environments and time offsets.

        Yields:
            dict with all tensors needed for PPO update.
        """
        T = self.buffer_size
        N = self.num_envs
        L = self.sequence_length

        # Number of complete sequences per environment
        num_seqs_per_env = T // L
        total_sequences = num_seqs_per_env * N

        # Build list of (env_idx, start_time) for each sequence
        sequence_indices = []
        for env in range(N):
            for seq in range(num_seqs_per_env):
                start_t = seq * L
                sequence_indices.append((env, start_t))

        # Shuffle sequence order (but NOT within sequences)
        perm = np.random.permutation(total_sequences)

        # Yield mini-batches of sequences
        seqs_per_batch = mini_batch_size // L
        if seqs_per_batch < 1:
            seqs_per_batch = 1

        for batch_start in range(0, total_sequences, seqs_per_batch):
            batch_indices = perm[batch_start:batch_start + seqs_per_batch]

            # Collect sequences
            batch_voxel = []
            batch_flat = []
            batch_actions = []
            batch_masks = []
            batch_log_probs = []
            batch_advantages = []
            batch_returns = []
            batch_values = []
            batch_dones = []
            # Hidden states at sequence START
            batch_actor_h = []
            batch_actor_c = []
            batch_critic_h = []
            batch_critic_c = []

            for idx in batch_indices:
                env, start_t = sequence_indices[idx]
                end_t = start_t + L

                batch_voxel.append(self.voxel_grids[start_t:end_t, env])
                batch_flat.append(self.flat_features[start_t:end_t, env])
                batch_actions.append(self.actions[start_t:end_t, env])
                batch_masks.append(self.action_masks[start_t:end_t, env])
                batch_log_probs.append(self.log_probs[start_t:end_t, env])
                batch_advantages.append(self.advantages[start_t:end_t, env])
                batch_returns.append(self.returns[start_t:end_t, env])
                batch_values.append(self.values[start_t:end_t, env])
                # Merge dones and truncs: both are episode boundaries
                # where LSTM hidden state must be reset
                batch_dones.append(
                    self.dones[start_t:end_t, env] | self.truncs[start_t:end_t, env]
                )

                # Hidden state at the start of this sequence
                batch_actor_h.append(self.actor_h[start_t, env])
                batch_actor_c.append(self.actor_c[start_t, env])
                batch_critic_h.append(self.critic_h[start_t, env])
                batch_critic_c.append(self.critic_c[start_t, env])

            # Stack into tensors: (num_seqs, L, ...) then reshape to (num_seqs*L, ...)
            def to_tensor(arrays, dtype=torch.float32):
                stacked = np.stack(arrays)  # (num_seqs, L, ...)
                return torch.tensor(stacked, dtype=dtype, device=self.device)

            num_seqs = len(batch_indices)

            yield {
                # Observations: (num_seqs, L, ...)
                "voxel_grids": to_tensor(batch_voxel),
                "flat_features": to_tensor(batch_flat),
                # Actions and policy: (num_seqs, L, ...)
                "actions": to_tensor(batch_actions, dtype=torch.long),
                "action_masks": to_tensor(batch_masks, dtype=torch.bool),
                "old_log_probs": to_tensor(batch_log_probs),
                # Targets: (num_seqs, L)
                "advantages": to_tensor(batch_advantages),
                "returns": to_tensor(batch_returns),
                "old_values": to_tensor(batch_values),
                # Episode boundaries: (num_seqs, L)
                "dones": to_tensor(batch_dones, dtype=torch.bool),
                # Hidden states at sequence start: (1, num_seqs, LSTM_HIDDEN_SIZE)
                "actor_hidden": (
                    torch.tensor(np.stack(batch_actor_h), dtype=torch.float32,
                                 device=self.device).unsqueeze(0),
                    torch.tensor(np.stack(batch_actor_c), dtype=torch.float32,
                                 device=self.device).unsqueeze(0),
                ),
                "critic_hidden": (
                    torch.tensor(np.stack(batch_critic_h), dtype=torch.float32,
                                 device=self.device).unsqueeze(0),
                    torch.tensor(np.stack(batch_critic_c), dtype=torch.float32,
                                 device=self.device).unsqueeze(0),
                ),
                "num_seqs": num_seqs,
                "seq_length": L,
            }

    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False
