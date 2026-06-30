"""Shared per-agent actor scoring candidate path features."""

from __future__ import annotations

import torch
from torch import nn

from .utils import build_mlp


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.action_scorer = build_mlp(obs_dim, hidden_dim, num_layers, 1, activation)

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.action_scorer(obs).squeeze(-1)
        if action_mask is not None:
            mask = action_mask > 0
            all_invalid = ~mask.any(dim=-1)
            if all_invalid.any():
                mask = mask.clone()
                flat_mask = mask.reshape(-1, mask.shape[-1])
                flat_invalid = all_invalid.reshape(-1)
                flat_mask[flat_invalid, 0] = True
            logits = logits.masked_fill(~mask, -1e9)
        return logits
