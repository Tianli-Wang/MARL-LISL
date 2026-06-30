"""Clipped PPO objective for shared actors and a centralized critic."""

from __future__ import annotations

import torch

from .utils import to_tensor


def compute_mappo_loss(
    policy,
    batch,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    normalize_advantages: bool = True,
):
    device = policy.device
    obs = to_tensor(batch["obs"], device)
    states = to_tensor(batch["states"], device)
    masks = to_tensor(batch["action_masks"], device)
    actions = to_tensor(batch["actions"], device, dtype=torch.long)
    old_log_probs = to_tensor(batch["old_log_probs"], device)
    returns = to_tensor(batch["returns"], device)
    advantages = to_tensor(batch["advantages"], device)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    new_log_probs, entropy, values = policy.evaluate_actions(obs, states, actions, masks)
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages_agent = advantages.unsqueeze(-1)
    surrogate_1 = ratio * advantages_agent
    surrogate_2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_agent
    actor_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
    critic_loss = torch.mean((values - returns) ** 2)
    entropy_mean = entropy.mean()
    total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_mean
    approx_kl = (old_log_probs - new_log_probs).mean()
    info = {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "entropy": float(entropy_mean.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
    }
    return total_loss, info
