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
        normalize_input: bool = False,
        heuristic_prior: dict | None = None,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim) if normalize_input else nn.Identity()
        self.action_scorer = build_mlp(obs_dim, hidden_dim, num_layers, 1, activation)
        prior = dict(heuristic_prior or {})
        self.prior_enabled = bool(prior.get("enabled", False))
        self.prior_scale = float(prior.get("scale", 1.0))
        self.prior_delay = float(prior.get("delay", 0.0))
        self.prior_setup = float(prior.get("setup", 0.0))
        self.prior_new_link = float(prior.get("new_link", 0.0))
        self.prior_hops = float(prior.get("hops", 0.0))
        self.prior_mutex = float(prior.get("mutex", 0.0))
        self.prior_avoid = float(prior.get("avoid", 0.0))
        self.prior_keep_bonus = float(prior.get("keep_bonus", 0.0))

    def _heuristic_logits(self, obs: torch.Tensor) -> torch.Tensor:
        score = (
            -self.prior_delay * obs[..., 0]
            - self.prior_setup * obs[..., 1]
            - self.prior_new_link * obs[..., 3]
            - self.prior_hops * obs[..., 4]
            - self.prior_mutex * obs[..., 6]
            + self.prior_avoid * obs[..., 7]
        )
        if self.prior_keep_bonus:
            score = score.clone()
            score[..., 0] += self.prior_keep_bonus
        return self.prior_scale * score

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.action_scorer(self.input_norm(obs)).squeeze(-1)
        if self.prior_enabled:
            logits = logits + self._heuristic_logits(obs)
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
