"""Actor/centralized-critic policy wrapper."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .actor import Actor
from .critic import Critic
from .utils import to_tensor


class MAPPOPolicy(nn.Module):
    def __init__(self, actor: Actor, critic: Critic, device: str | torch.device = "cpu"):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.device = torch.device(device)
        self.to(self.device)

    @torch.no_grad()
    def act(self, obs, state, action_mask, deterministic: bool = False):
        obs_t = to_tensor(obs, self.device)
        state_t = to_tensor(state, self.device)
        mask_t = to_tensor(action_mask, self.device)
        logits = self.actor(obs_t, mask_t)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = self.critic(state_t)
        return (
            actions.cpu().numpy().astype(np.int64),
            log_probs.cpu().numpy().astype(np.float32),
            float(value.item()),
            entropy.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def get_value(self, state) -> float:
        return float(self.critic(to_tensor(state, self.device)).item())

    def evaluate_actions(self, obs, state, actions, action_mask):
        obs_t = to_tensor(obs, self.device)
        state_t = to_tensor(state, self.device)
        actions_t = to_tensor(actions, self.device, dtype=torch.long)
        mask_t = to_tensor(action_mask, self.device)
        distribution = Categorical(logits=self.actor(obs_t, mask_t))
        return (
            distribution.log_prob(actions_t),
            distribution.entropy(),
            self.critic(state_t),
        )
