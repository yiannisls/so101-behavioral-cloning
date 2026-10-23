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

class MultiTaskPolicy(BasePolicy):
    """MLP predicting action chunks from robot, cube and goal inputs."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        depth: int = 2,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
                for _ in range(self.depth - 1)
            ],
            nn.Linear(self.d_model, self.chunk_size * self.action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return an action chunk for each goal-conditioned observation."""
        predictions = self.network(state)
        return predictions.reshape(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:

        return nn.functional.mse_loss(self(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)

class ACTPolicy(BasePolicy):
    """State-based action-chunk policy with a variational encoder."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        depth: int = 2,
        latent_dim: int = 32,
        nhead: int = 4,
        kl_weight: float = 0.001,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.d_model = d_model
        self.depth = depth
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.kl_weight = kl_weight

        # 1. Convert observations and expert actions into tokens.
        self.encoder_state_projection = nn.Linear(state_dim, d_model)
        self.encoder_action_projection = nn.Linear(action_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.encoder_positions = nn.Embedding(chunk_size + 2, d_model)

        # 2. Read the demonstration and infer a latent distribution.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )
        self.latent_parameters = nn.Linear(d_model, 2 * latent_dim)

        # 3. Convert the current state and latent into decoder context.
        self.decoder_state_projection = nn.Linear(state_dim, d_model)
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.context_positions = nn.Embedding(2, d_model)

        # 4. Predict one action for each future step.
        self.action_queries = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=depth,
        )
        self.action_head = nn.Linear(d_model, action_dim)

        nn.init.normal_(self.cls_token, std=0.02)

    def _encode_latent(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]

        # Convert the observation and expert actions into tokens.
        state_token = self.encoder_state_projection(state).unsqueeze(1)
        action_tokens = self.encoder_action_projection(action_chunk)
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        # Shape: (B, chunk_size + 2, d_model)
        tokens = torch.cat(
            [cls_token, state_token, action_tokens],
            dim=1,
        )

        # Add information identifying each token's position.
        positions = torch.arange(tokens.shape[1], device=state.device)
        tokens = tokens + self.encoder_positions(positions).unsqueeze(0)

        encoded = self.latent_encoder(tokens)

        # The first token summarizes the observation and demonstration.
        summary = encoded[:, 0]
        latent_parameters = self.latent_parameters(summary)

        mu, log_var = latent_parameters.chunk(2, dim=-1)

        # Reparameterization: sample z while preserving gradient flow.
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        return z, mu, log_var

    def _build_context(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        state_token = self.decoder_state_projection(state)
        latent_token = self.latent_projection(z)

        context = torch.stack(
            [state_token, latent_token],
            dim=1,
        )

        positions = torch.arange(2, device=state.device)
        return (
            context
            + self.context_positions(positions).unsqueeze(0)
        )

    def forward(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        batch_size = state.shape[0]

        if action_chunk is not None:
            # Expert actions are available: infer and sample z.
            z, mu, log_var = self._encode_latent(state, action_chunk)
        else:
            # Robot execution: use the standard ACT zero latent.
            z = state.new_zeros(batch_size, self.latent_dim)
            mu, log_var = None, None

        context = self._build_context(state, z)

        # One learned query for each future action.
        queries = self.action_queries.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, chunk_size, d_model)

        decoded = self.action_decoder(
            tgt=queries,
            memory=context,
        )  # (B, chunk_size, d_model)

        predictions = self.action_head(decoded)

        return predictions, mu, log_var

    def compute_loss_components(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predictions, mu, log_var = self(state, action_chunk)

        reconstruction = nn.functional.mse_loss(
            predictions, action_chunk
        )

        kl = -0.5 * (
            1 + log_var - mu.square() - log_var.exp()
        ).sum(dim=-1).mean()

        total = reconstruction + self.kl_weight * kl

        return {
            "total": total,
            "reconstruction": reconstruction,
            "kl": kl,
        }

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_loss_components(
            state, action_chunk
        )["total"]

    @torch.no_grad()
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        predictions, _, _ = self(state)
        return predictions


class ACTScenePolicy(ACTPolicy):
    """ACT with separate decoder tokens for scene features."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        depth: int = 2,
        latent_dim: int = 32,
        nhead: int = 4,
        kl_weight: float = 0.1,
    ) -> None:
        if state_dim != 25:
            raise ValueError(
                "ACTScenePolicy requires the ordered 25-feature "
                "multicube observation."
            )

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            latent_dim=latent_dim,
            nhead=nhead,
            kl_weight=kl_weight,
        )

        # Replace the single decoder observation projection.
        del self.decoder_state_projection

        self.scene_dimensions = (4, 3, 3, 3, 3, 3, 3, 3)

        self.scene_projections = nn.ModuleList(
            nn.Linear(size, d_model)
            for size in self.scene_dimensions
        )

        # Eight scene tokens plus one latent token.
        self.context_positions = nn.Embedding(9, d_model)

    def _build_context(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.split(
            state,
            self.scene_dimensions,
            dim=-1,
        )

        tokens = [
            projection(feature)
            for projection, feature
            in zip(self.scene_projections, features)
        ]

        tokens.append(self.latent_projection(z))

        context = torch.stack(tokens, dim=1)
        positions = torch.arange(
            context.shape[1],
            device=state.device,
        )

        return (
            context
            + self.context_positions(positions).unsqueeze(0)
        )


PolicyType: TypeAlias = Literal[
    "obstacle", "multitask", "act", "act_scene"
]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int = 128,
    depth: int = 2,
    latent_dim: int = 32,
    nhead: int = 4,
    kl_weight: float = 10.0,
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
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth
        )
    if policy_type == "act":
        return ACTPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            latent_dim=latent_dim,
            nhead=nhead,
            kl_weight=kl_weight
        )
    if policy_type == "act_scene":
        return ACTScenePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            latent_dim=latent_dim,
            nhead=nhead,
            kl_weight=kl_weight,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
