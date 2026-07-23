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
    value_clip_range: float | None = None,
):
    """分别构建 Actor 与 Critic 目标，并返回统一诊断指标。

    Critic 网络输出和 loss 位于 ValueNorm 坐标；batch 中的 returns、old_values
    则始终保留环境 reward 的原始尺度。这样 buffer/GAE/评估口径不会因是否启用
    ValueNorm 而改变。
    """

    device = policy.device
    obs = to_tensor(batch["obs"], device)
    states = to_tensor(batch["states"], device)
    masks = to_tensor(batch["action_masks"], device)
    actions = to_tensor(batch["actions"], device, dtype=torch.long)
    old_log_probs = to_tensor(batch["old_log_probs"], device)
    returns = to_tensor(batch["returns"], device)
    advantages = to_tensor(batch["advantages"], device)
    old_values = to_tensor(batch["old_values"], device)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    new_log_probs, entropy, values_normalized = policy.evaluate_actions(
        obs, states, actions, masks
    )
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages_agent = advantages.unsqueeze(-1)
    surrogate_1 = ratio * advantages_agent
    surrogate_2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_agent
    actor_loss = -torch.minimum(surrogate_1, surrogate_2).mean()

    # returns 与 old_values 在 buffer 中均为 raw reward 尺度，必须一起转换；
    # 只归一化 returns、却拿 raw old_values 做 value clipping 会产生毫无意义的
    # 数千量级裁剪中心。
    returns_normalized = policy.normalize_values(returns)
    old_values_normalized = policy.normalize_values(old_values)
    if value_clip_range is None:
        critic_loss = torch.mean((values_normalized - returns_normalized) ** 2)
    else:
        value_clip_range = float(value_clip_range)
        values_clipped = old_values_normalized + torch.clamp(
            values_normalized - old_values_normalized,
            -value_clip_range,
            value_clip_range,
        )
        critic_unclipped = (values_normalized - returns_normalized) ** 2
        critic_clipped = (values_clipped - returns_normalized) ** 2
        critic_loss = torch.maximum(critic_unclipped, critic_clipped).mean()
    entropy_mean = entropy.mean()
    actor_objective = actor_loss - entropy_coef * entropy_mean
    critic_objective = value_coef * critic_loss
    total_loss = actor_objective + critic_objective

    # 该二阶近似形式理论上非负，比简单 mean(old_log-new_log) 在小 batch 上更
    # 稳定，适合用于 target_kl 提前停止。
    log_ratio = new_log_probs - old_log_probs
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = (
        (torch.abs(ratio - 1.0) > float(clip_ratio)).to(torch.float32).mean()
    )
    info = {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "entropy": float(entropy_mean.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "return_mean": float(returns.mean().detach().cpu()),
        "return_std": float(returns.std(unbiased=False).detach().cpu()),
        "value_mean": float(old_values.mean().detach().cpu()),
        "value_std": float(old_values.std(unbiased=False).detach().cpu()),
    }
    return actor_objective, critic_objective, info
