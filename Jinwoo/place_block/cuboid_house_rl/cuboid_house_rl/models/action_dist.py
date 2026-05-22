"""
Masked Multi-Discrete Action Distribution.

Handles independent categorical distributions per action dimension
with optional action masking (invalid actions get -inf logits).
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional, List

from cuboid_house_rl.config import ACTION_DIMS


class MaskedMultiDiscreteDistribution:
    """
    A factored distribution over MultiDiscrete action spaces.

    Each dimension is an independent Categorical distribution.
    Action masking sets logits of invalid actions to -inf before softmax.

    Usage:
        dist = MaskedMultiDiscreteDistribution(logits, action_masks)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
    """

    def __init__(
        self,
        logits: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        action_dims: List[int] = ACTION_DIMS,
    ):
        """
        Args:
            logits: (batch, sum(action_dims)) raw logits from the network.
            action_masks: (batch, sum(action_dims)) boolean tensor.
                          True = valid, False = masked. None = no masking.
            action_dims: list of sizes per dimension [3, 3, 2, 2, 3, 9, 7, 7].
        """
        self.action_dims = action_dims
        self.distributions = []

        # Split logits into per-dimension chunks
        split_logits = torch.split(logits, action_dims, dim=-1)

        if action_masks is not None:
            split_masks = torch.split(action_masks, action_dims, dim=-1)
        else:
            split_masks = [None] * len(action_dims)

        for dim_logits, dim_mask in zip(split_logits, split_masks):
            if dim_mask is not None:
                # Set masked actions to -inf (they get ~0 probability after softmax)
                masked_logits = dim_logits.clone()
                masked_logits[~dim_mask] = float("-inf")
            else:
                masked_logits = dim_logits

            self.distributions.append(Categorical(logits=masked_logits))

    def sample(self) -> torch.Tensor:
        """
        Sample one action per dimension.

        Returns:
            (batch, num_dims) tensor of integer actions.
        """
        samples = [d.sample() for d in self.distributions]
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute joint log probability.

        Joint log prob = sum of per-dimension log probs (independence assumption).

        Args:
            actions: (batch, num_dims) integer tensor.

        Returns:
            (batch,) tensor of log probabilities.
        """
        log_probs = []
        for i, dist in enumerate(self.distributions):
            log_probs.append(dist.log_prob(actions[:, i]))
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """
        Compute total entropy.

        Total entropy = sum of per-dimension entropies.

        Returns:
            (batch,) tensor of entropy values.
        """
        entropies = [d.entropy() for d in self.distributions]
        return torch.stack(entropies, dim=-1).sum(dim=-1)

    def mode(self) -> torch.Tensor:
        """
        Return the most likely action per dimension (greedy/deterministic).

        Returns:
            (batch, num_dims) tensor of integer actions.
        """
        modes = [d.probs.argmax(dim=-1) for d in self.distributions]
        return torch.stack(modes, dim=-1)
