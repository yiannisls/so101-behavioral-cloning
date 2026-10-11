"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import device, nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """
    def __init__(
    self,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int = 128,
    depth: int = 2,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.d_model = d_model
        self.depth = depth

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.ReLU(),
                )
                for _ in range(self.depth - 1)
            ],
            nn.Linear(
                self.d_model,
                self.chunk_size * self.action_dim,
            ),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        # Take a batch of observations and return predicted action chunks
        predictions = self.network(state)  # (B, chunk_size * action_dim)
        return predictions.reshape(
            state.shape[0],
            self.chunk_size,
            self.action_dim
        )  # (B, chunk_size, action_dim  )

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        # Compare predicted action chunks to ground truth action chunks using MSE loss
        predictions = self(state)  # (B, chunk_size, action_dim)

        return nn.functional.mse_loss(predictions, action_chunk)
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        #provide a method to sample actions from the policy given a state
        # despite the name, this method does not actually sample actions,
        # it just returns the predicted action chunk
        return self(state)  # (B, chunk_size, action_dim)

# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int = 128,
    depth: int = 2,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
