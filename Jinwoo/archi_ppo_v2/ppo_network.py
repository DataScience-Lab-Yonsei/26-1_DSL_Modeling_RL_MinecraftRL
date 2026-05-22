"""
BuilderNetwork (MLP-only, no LSTM).

Drop-in replacement for the LSTM version in train.py.
Identical observation/action interface — just replaces the LSTM with
two additional fully-connected layers so the policy is feed-forward.
"""

import torch
import torch.nn as nn
from typing import Optional

from building_env import NUM_BUILD_ACTIONS, BuildAction


class BuilderNetwork(nn.Module):
    """
    Actor-Critic network for the house-building agent (MLP only, no LSTM).

    Architecture:
        - 3D CNN encoder  for local/target/diff/raycast voxel grids (4 channels)
        - MLP encoders    for agent state, progress
        - Linear encoder  for next_block_id (what block type is needed)
        - Shared MLP trunk (2 layers)
        - Separate policy (actor) and value (critic) heads
    """

    def __init__(
        self,
        obs_grid_size: int = 11,
        num_actions: int = NUM_BUILD_ACTIONS,
        hidden_dim: int = 256,
        # lstm_hidden kept as a no-op kwarg so callers don't need to change
        lstm_hidden: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 3D CNN for voxel grids (local + target + diff + raycast = 4 channels)
        self.grid_encoder = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2),
            nn.Flatten(),
        )
        # After AdaptiveAvgPool3d(2): 64 * 2 * 2 * 2 = 512
        grid_out_dim = 64 * 2 * 2 * 2

        # MLP for agent state: [x, y, z, yaw, pitch, health] = 6
        self.state_encoder = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # MLP for progress: [pct, layer, total_layers, remaining] = 4
        self.progress_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
        )

        # Encoder for next_block_id: [block_id_normalised] = 1
        # Tells the agent which block type is needed at the next target.
        # Allows the policy to learn to select the correct hotbar slot.
        self.block_encoder = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(),
        )

        # Encoder for structure_info: [type_id, n_blocks, size_x, size_y, size_z] = 5
        # Tells the agent what kind of structure it is building and its dimensions.
        self.structure_encoder = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(),
        )

        # Combined feature dimension: 512 + 64 + 32 + 8 + 16 = 632
        combined_dim = grid_out_dim + 64 + 32 + 8 + 16

        # Shared trunk — two FC layers instead of FC + LSTM
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        # Bias PLACE_BLOCK action so the agent tries placing early in training
        with torch.no_grad():
            self.policy_head[-1].bias[BuildAction.PLACE_BLOCK] += 1.0

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        local_grid: torch.Tensor,
        target_grid: torch.Tensor,
        diff_grid: torch.Tensor,
        raycast_grid: torch.Tensor,
        agent_pos: torch.Tensor,
        progress: torch.Tensor,
        next_block_id: torch.Tensor = None,
        structure_info: torch.Tensor = None,
        # lstm_state kept as a dummy kwarg for API compatibility
        lstm_state=None,
    ):
        """
        Forward pass (no recurrent state).

        Args:
            local_grid:    (batch, obs_size, obs_size, obs_size)
            target_grid:   (batch, obs_size, obs_size, obs_size)
            diff_grid:     (batch, obs_size, obs_size, obs_size)
            raycast_grid:  (batch, obs_size, obs_size, obs_size) 1.0 at looked-at block
            agent_pos:     (batch, 6)
            progress:      (batch, 4)
            next_block_id: (batch, 1)  normalised block ID in [0, 1]
            structure_info:(batch, 5)  [type_id, n_blocks, sx, sy, sz] normalised
            lstm_state:    ignored (kept for API compatibility with LSTM version)

        Returns:
            action_logits, value, None  (None replaces lstm_state)
        """
        batch = local_grid.shape[0]

        # Stack grids as channels: (batch, 4, D, H, W)
        grids = torch.stack([
            local_grid.float() / 255.0,
            target_grid.float() / 255.0,
            diff_grid.float(),
            raycast_grid.float(),
        ], dim=1)

        grid_features     = self.grid_encoder(grids)
        state_features    = self.state_encoder(agent_pos)
        progress_features = self.progress_encoder(progress)

        # next_block_id: default to zeros if not provided (backward compat)
        if next_block_id is None:
            next_block_id = torch.zeros(batch, 1,
                                        dtype=torch.float32,
                                        device=local_grid.device)
        block_features = self.block_encoder(next_block_id)

        # structure_info: default to zeros if not provided (backward compat)
        if structure_info is None:
            structure_info = torch.zeros(batch, 5,
                                         dtype=torch.float32,
                                         device=local_grid.device)
        struct_features = self.structure_encoder(structure_info)

        combined = torch.cat([
            grid_features,
            state_features,
            progress_features,
            block_features,
            struct_features,
        ], dim=-1)

        shared_out = self.shared(combined)

        action_logits = self.policy_head(shared_out)
        value         = self.value_head(shared_out).squeeze(-1)

        return action_logits, value, None  # None = no lstm state
