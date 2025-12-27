"""Exploration bonus modules for RL algorithms."""

import torch
import torch.nn as nn


class RND(nn.Module):
    """Random Network Distillation for exploration bonuses.

    RND provides intrinsic rewards based on prediction error of a fixed
    random target network. States that are novel/rare will have high
    prediction error, encouraging exploration.

    Paper: https://arxiv.org/abs/1810.12894
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        lr: float = 1e-4,
    ):
        """Initialize RND module.

        Args:
            obs_dim: Observation dimension.
            hidden_dim: Hidden layer size.
            output_dim: Output feature dimension.
            lr: Learning rate for predictor network.
        """
        super().__init__()

        # Fixed random target network (never updated)
        self.target = self._make_network(obs_dim, hidden_dim, output_dim)
        for param in self.target.parameters():
            param.requires_grad = False

        # Trainable predictor network
        self.predictor = self._make_network(obs_dim, hidden_dim, output_dim)

        # Optimizer for predictor
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running statistics for reward normalization
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_m2', torch.zeros(1))
        self.register_buffer('reward_count', torch.zeros(1))

    def _make_network(
        self, obs_dim: int, hidden_dim: int, output_dim: int
    ) -> nn.Module:
        """Create a simple feedforward network.

        Args:
            obs_dim: Input dimension.
            hidden_dim: Hidden layer size.
            output_dim: Output dimension.

        Returns:
            Neural network module.
        """
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @torch.no_grad()
    def compute_bonus(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward bonus for observations.

        Args:
            obs: Observations, shape (batch, obs_dim).

        Returns:
            Intrinsic rewards, shape (batch,).
        """
        # Get target features (fixed)
        target_features = self.target(obs)

        # Get predicted features
        pred_features = self.predictor(obs)

        # Compute prediction error as intrinsic reward
        bonus = (target_features - pred_features).pow(2).mean(dim=-1)

        return bonus

    def update(self, obs: torch.Tensor) -> float:
        """Update predictor network on a batch of observations.

        Args:
            obs: Observations, shape (batch, obs_dim).

        Returns:
            Prediction loss value.
        """
        # Get target features (fixed)
        with torch.no_grad():
            target_features = self.target(obs)

        # Get predicted features
        pred_features = self.predictor(obs)

        # Mean squared error loss
        loss = (target_features - pred_features).pow(2).mean()

        # Update predictor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize intrinsic rewards using running statistics.

        Args:
            rewards: Raw intrinsic rewards, shape (batch,).

        Returns:
            Normalized rewards, shape (batch,).
        """
        rewards_flat = rewards.flatten()

        # Update running statistics (Welford's algorithm)
        for reward in rewards_flat:
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = reward - self.reward_mean
            self.reward_m2 += delta * delta2

        # Compute std
        if self.reward_count > 1:
            reward_std = torch.sqrt(self.reward_m2 / (self.reward_count - 1))
        else:
            reward_std = torch.ones_like(self.reward_mean)

        # Normalize
        normalized = (rewards - self.reward_mean) / (reward_std + 1e-8)

        return normalized


class ICM(nn.Module):
    """Intrinsic Curiosity Module for exploration bonuses.

    ICM provides intrinsic rewards based on prediction error of
    a forward dynamics model. Novel state transitions have high
    prediction error, encouraging exploration.

    Paper: https://arxiv.org/abs/1705.05363
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        lr: float = 1e-4,
        beta: float = 0.2,
    ):
        """Initialize ICM module.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            hidden_dim: Hidden layer size.
            feature_dim: Feature encoding dimension.
            lr: Learning rate.
            beta: Weight for forward loss vs inverse loss.
        """
        super().__init__()
        self.beta = beta

        # Feature encoder (shared)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Inverse model: predicts action from (s_t, s_{t+1})
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        # Forward model: predicts next state features from (s_t, a_t)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    @torch.no_grad()
    def compute_bonus(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute intrinsic reward bonus.

        Args:
            obs: Current observations, shape (batch, obs_dim).
            action: Actions taken, shape (batch, act_dim).
            next_obs: Next observations, shape (batch, obs_dim).

        Returns:
            Intrinsic rewards, shape (batch,).
        """
        # Encode observations
        features = self.encoder(obs)
        next_features = self.encoder(next_obs)

        # Predict next features
        pred_next_features = self.forward_model(torch.cat([features, action], dim=-1))

        # Intrinsic reward is forward prediction error
        bonus = (next_features - pred_next_features).pow(2).mean(dim=-1)

        return bonus

    def update(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> tuple[float, float]:
        """Update ICM on a batch of transitions.

        Args:
            obs: Current observations, shape (batch, obs_dim).
            action: Actions taken, shape (batch, act_dim).
            next_obs: Next observations, shape (batch, obs_dim).

        Returns:
            Tuple of (forward_loss, inverse_loss).
        """
        # Encode observations
        features = self.encoder(obs)
        next_features = self.encoder(next_obs)

        # Forward model loss
        pred_next_features = self.forward_model(torch.cat([features, action], dim=-1))
        forward_loss = (next_features.detach() - pred_next_features).pow(2).mean()

        # Inverse model loss
        pred_action = self.inverse_model(torch.cat([features, next_features], dim=-1))
        inverse_loss = (action - pred_action).pow(2).mean()

        # Total loss
        loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return forward_loss.item(), inverse_loss.item()
