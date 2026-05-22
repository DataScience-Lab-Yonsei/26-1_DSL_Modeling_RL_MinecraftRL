"""
Actor-Critic Network for House Building Agent.

Architecture:
    Voxel grids (11,11,11,6) → 3D CNN → 3,456 features
    Concat with flat features (~67) → 3,523
    → Shared MLP(512)
    → Actor LSTM(256) → MLP(128) → 8 action heads
    → Critic LSTM(256) → MLP(128) → value head

Separate LSTMs for actor and critic to avoid gradient conflicts.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict

from cuboid_house_rl.config import (
    # Observation
    LOCAL_WINDOW_SIZE, STACKED_CHANNELS, NON_VOXEL_SIZE,
    # Network
    CNN_CHANNELS, CNN_KERNEL_SIZE, CNN_STRIDE_LAST, CNN_OUTPUT_SIZE,
    SHARED_MLP_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    ACTOR_MLP_SIZE, CRITIC_MLP_SIZE,
    # Actions
    ACTION_DIMS, TOTAL_ACTION_LOGITS, INITIAL_BIAS,
)
from cuboid_house_rl.models.action_dist import MaskedMultiDiscreteDistribution


class VoxelCNN(nn.Module):
    """
    3D Convolutional feature extractor for voxel grids.

    Input:  (batch, 6, 11, 11, 11)   — channels first
    Output: (batch, 3456)             — flattened
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=STACKED_CHANNELS,
            out_channels=CNN_CHANNELS[0],
            kernel_size=CNN_KERNEL_SIZE,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv3d(
            in_channels=CNN_CHANNELS[0],
            out_channels=CNN_CHANNELS[1],
            kernel_size=CNN_KERNEL_SIZE,
            stride=1,
            padding=0,
        )
        self.conv3 = nn.Conv3d(
            in_channels=CNN_CHANNELS[1],
            out_channels=CNN_CHANNELS[2],
            kernel_size=CNN_KERNEL_SIZE,
            stride=CNN_STRIDE_LAST,
            padding=0,
        )
        self.relu = nn.ReLU()

        # Verify output size
        self._verify_output_size()

    def _verify_output_size(self):
        """Compute and verify the CNN output size."""
        dummy = torch.zeros(1, STACKED_CHANNELS,
                            LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE)
        with torch.no_grad():
            out = self.forward(dummy)
        actual_size = out.shape[1]
        assert actual_size == CNN_OUTPUT_SIZE, \
            f"CNN output size mismatch: expected {CNN_OUTPUT_SIZE}, got {actual_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, W, H, D) voxel grids.

        Returns:
            (batch, CNN_OUTPUT_SIZE) flattened features.
        """
        x = self.relu(self.conv1(x))   # (batch, 32, 9, 9, 9)
        x = self.relu(self.conv2(x))   # (batch, 64, 7, 7, 7)
        x = self.relu(self.conv3(x))   # (batch, 128, 3, 3, 3)
        return x.flatten(start_dim=1)   # (batch, 3456)


class ActorCriticNetwork(nn.Module):
    """
    Full actor-critic network with:
    - Shared 3D CNN for voxel processing
    - Shared MLP for feature fusion
    - Separate LSTM + MLP branches for actor and critic
    - Multi-head action output (one per MultiDiscrete dimension)

    Hidden states for actor and critic LSTMs are maintained externally
    and passed in/out of the forward method.
    """

    def __init__(self, lstm_num_layers=LSTM_NUM_LAYERS):
        super().__init__()
        self.lstm_num_layers = lstm_num_layers

        # ---- Shared backbone ----
        self.cnn = VoxelCNN()

        fusion_input_size = CNN_OUTPUT_SIZE + NON_VOXEL_SIZE
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_input_size, SHARED_MLP_SIZE),
            nn.ReLU(),
        )

        # ---- Actor branch ----
        self.actor_lstm = nn.LSTM(
            input_size=SHARED_MLP_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.actor_mlp = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, ACTOR_MLP_SIZE),
            nn.ReLU(),
        )
        # Single linear layer for all action logits
        self.action_head = nn.Linear(ACTOR_MLP_SIZE, TOTAL_ACTION_LOGITS)

        # ---- Critic branch ----
        self.critic_lstm = nn.LSTM(
            input_size=SHARED_MLP_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.critic_mlp = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, CRITIC_MLP_SIZE),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(CRITIC_MLP_SIZE, 1)

        # ---- Initialize ----
        self._init_weights()
        self._apply_initial_bias()

    def _init_weights(self):
        """Orthogonal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif "bias" in name:
                        nn.init.zeros_(param)

        # Value head with small init for stable value estimates
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)

    def _apply_initial_bias(self, bias=None):
        """Apply initial policy bias from config."""
        if bias is None:
            bias = INITIAL_BIAS
        with torch.no_grad():
            self.action_head.bias.copy_(
                torch.tensor(bias, dtype=torch.float32)
            )

    def get_initial_hidden_state(
        self, batch_size: int = 1, device: torch.device = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create zero-initialized hidden states for both LSTMs.

        Args:
            batch_size: number of parallel environments.
            device: torch device.

        Returns:
            dict with 'actor' and 'critic' keys, each containing
            (h, c) tuple of shape (num_layers, batch_size, LSTM_HIDDEN_SIZE).
        """
        if device is None:
            device = next(self.parameters()).device

        n = self.lstm_num_layers

        def _zeros():
            return (
                torch.zeros(n, batch_size, LSTM_HIDDEN_SIZE, device=device),
                torch.zeros(n, batch_size, LSTM_HIDDEN_SIZE, device=device),
            )

        return {"actor": _zeros(), "critic": _zeros()}

    def forward(
        self,
        voxel_grids: torch.Tensor,
        flat_features: torch.Tensor,
        hidden_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict:
        """
        Forward pass for action selection (single timestep during rollout).

        Args:
            voxel_grids: (batch, 6, 11, 11, 11) — channels first.
            flat_features: (batch, NON_VOXEL_SIZE).
            hidden_states: dict with 'actor' and 'critic' (h, c) tuples.
            action_masks: (batch, TOTAL_ACTION_LOGITS) boolean. None = no masking.
            deterministic: if True, use mode instead of sample.

        Returns:
            dict with keys:
                'actions': (batch, 8) sampled actions
                'log_probs': (batch,) joint log probability
                'values': (batch,) state value estimate
                'entropy': (batch,) distribution entropy
                'hidden_states': updated hidden states dict
        """
        # ---- Shared backbone ----
        cnn_features = self.cnn(voxel_grids)                     # (batch, 3456)
        fused = torch.cat([cnn_features, flat_features], dim=-1)  # (batch, 3523)
        shared = self.shared_mlp(fused)                           # (batch, 512)

        # Add sequence dimension for LSTM: (batch, 1, 512)
        shared_seq = shared.unsqueeze(1)

        # ---- Actor ----
        actor_out, new_actor_hidden = self.actor_lstm(
            shared_seq, hidden_states["actor"]
        )
        actor_out = actor_out.squeeze(1)                 # (batch, 256)
        actor_features = self.actor_mlp(actor_out)       # (batch, 128)
        action_logits = self.action_head(actor_features)  # (batch, 36)

        # ---- Critic ----
        critic_out, new_critic_hidden = self.critic_lstm(
            shared_seq, hidden_states["critic"]
        )
        critic_out = critic_out.squeeze(1)               # (batch, 256)
        critic_features = self.critic_mlp(critic_out)    # (batch, 128)
        values = self.value_head(critic_features).squeeze(-1)  # (batch,)

        # ---- Action distribution ----
        dist = MaskedMultiDiscreteDistribution(
            action_logits, action_masks, ACTION_DIMS
        )

        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        new_hidden_states = {
            "actor": new_actor_hidden,
            "critic": new_critic_hidden,
        }

        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy,
            "hidden_states": new_hidden_states,
        }

    def evaluate_actions(
        self,
        voxel_grids: torch.Tensor,
        flat_features: torch.Tensor,
        actions: torch.Tensor,
        hidden_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Evaluate given actions (used during PPO update on stored rollout data).

        Unlike forward(), this does NOT sample new actions — it computes
        log_prob, value, and entropy for the given actions.

        Args:
            voxel_grids: (batch, 6, 11, 11, 11).
            flat_features: (batch, NON_VOXEL_SIZE).
            actions: (batch, 8) integer actions to evaluate.
            hidden_states: dict with 'actor' and 'critic' (h, c) tuples.
            action_masks: (batch, TOTAL_ACTION_LOGITS) boolean. None = no masking.

        Returns:
            dict with keys:
                'log_probs': (batch,) log probability of given actions
                'values': (batch,) state value estimates
                'entropy': (batch,) distribution entropy
                'hidden_states': updated hidden states
        """
        # ---- Shared backbone ----
        cnn_features = self.cnn(voxel_grids)
        fused = torch.cat([cnn_features, flat_features], dim=-1)
        shared = self.shared_mlp(fused)

        shared_seq = shared.unsqueeze(1)

        # ---- Actor ----
        actor_out, new_actor_hidden = self.actor_lstm(
            shared_seq, hidden_states["actor"]
        )
        actor_out = actor_out.squeeze(1)
        actor_features = self.actor_mlp(actor_out)
        action_logits = self.action_head(actor_features)

        # ---- Critic ----
        critic_out, new_critic_hidden = self.critic_lstm(
            shared_seq, hidden_states["critic"]
        )
        critic_out = critic_out.squeeze(1)
        critic_features = self.critic_mlp(critic_out)
        values = self.value_head(critic_features).squeeze(-1)

        # ---- Evaluate ----
        dist = MaskedMultiDiscreteDistribution(
            action_logits, action_masks, ACTION_DIMS
        )

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        new_hidden_states = {
            "actor": new_actor_hidden,
            "critic": new_critic_hidden,
        }

        return {
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy,
            "hidden_states": new_hidden_states,
        }

    def get_value(
        self,
        voxel_grids: torch.Tensor,
        flat_features: torch.Tensor,
        hidden_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Get only the value estimate (used for bootstrapping at rollout end).

        Returns:
            values: (batch,) state value estimates
            new_hidden_states: updated hidden states
        """
        cnn_features = self.cnn(voxel_grids)
        fused = torch.cat([cnn_features, flat_features], dim=-1)
        shared = self.shared_mlp(fused)

        shared_seq = shared.unsqueeze(1)

        # Only need critic LSTM
        critic_out, new_critic_hidden = self.critic_lstm(
            shared_seq, hidden_states["critic"]
        )
        critic_out = critic_out.squeeze(1)
        critic_features = self.critic_mlp(critic_out)
        values = self.value_head(critic_features).squeeze(-1)

        # Actor hidden state passes through unchanged
        new_hidden_states = {
            "actor": hidden_states["actor"],
            "critic": new_critic_hidden,
        }

        return values, new_hidden_states

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "cnn": _count(self.cnn),
            "shared_mlp": _count(self.shared_mlp),
            "actor_lstm": _count(self.actor_lstm),
            "actor_mlp": _count(self.actor_mlp),
            "action_head": _count(self.action_head),
            "critic_lstm": _count(self.critic_lstm),
            "critic_mlp": _count(self.critic_mlp),
            "value_head": _count(self.value_head),
            "total": sum(p.numel() for p in self.parameters()),
        }


# ==================================================================
# Quick test
# ==================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create network
    net = ActorCriticNetwork().to(device)

    # Parameter count
    params = net.count_parameters()
    print("\nParameter count:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Test forward pass
    batch_size = 4
    voxel = torch.randn(batch_size, STACKED_CHANNELS,
                         LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE,
                         LOCAL_WINDOW_SIZE, device=device)
    flat = torch.randn(batch_size, NON_VOXEL_SIZE, device=device)
    hidden = net.get_initial_hidden_state(batch_size, device)

    # Action mask: only slot 0 valid for hotbar
    mask = torch.ones(batch_size, TOTAL_ACTION_LOGITS, dtype=torch.bool, device=device)
    hotbar_start = sum(ACTION_DIMS[:5])  # index 13
    mask[:, hotbar_start + 1:hotbar_start + 9] = False  # mask slots 1-8

    # Forward (sampling)
    result = net(voxel, flat, hidden, action_masks=mask)
    print(f"\nForward pass (sampling):")
    print(f"  actions: {result['actions'].shape} = {result['actions'][0].tolist()}")
    print(f"  log_probs: {result['log_probs'].shape}")
    print(f"  values: {result['values'].shape}")
    print(f"  entropy: {result['entropy'].shape}")
    print(f"  All hotbar actions are slot 0: {(result['actions'][:, 5] == 0).all()}")

    # Evaluate actions
    eval_result = net.evaluate_actions(
        voxel, flat, result["actions"],
        result["hidden_states"], action_masks=mask
    )
    print(f"\nEvaluate actions:")
    print(f"  log_probs: {eval_result['log_probs'].shape}")
    print(f"  values: {eval_result['values'].shape}")
    print(f"  entropy: {eval_result['entropy'].shape}")

    # Get value only
    values, _ = net.get_value(voxel, flat, hidden)
    print(f"\nGet value: {values.shape}")

    # Verify initial bias
    with torch.no_grad():
        bias = net.action_head.bias.cpu().numpy()
        print(f"\nInitial bias (interact dim): {bias[10:13]}")
        print(f"  Expected: [1.5, 0.5, -2.0] (place-biased)")

    print("\nAll network tests passed!")
