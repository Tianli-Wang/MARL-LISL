"""Centralized scalar state-value critic."""

from torch import nn

from .utils import build_mlp


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "relu",
    ):
        super().__init__()
        # Environment state mixes counts, metres, seconds, and mutex scores.
        # Per-sample normalization keeps the minimal critic numerically stable.
        self.input_norm = nn.LayerNorm(state_dim)
        self.value_net = build_mlp(state_dim, hidden_dim, num_layers, 1, activation)

    def forward(self, state):
        return self.value_net(self.input_norm(state)).squeeze(-1)
