"""Actor/centralized-critic policy wrapper."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .actor import Actor
from .critic import Critic
from .normalization import ValueNormalizer
from .utils import to_tensor


class MAPPOPolicy(nn.Module):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        device: str | torch.device = "cpu",
        value_normalizer: ValueNormalizer | None = None,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        # 旧代码直接构造 MAPPOPolicy 时默认关闭 ValueNorm，保持 Critic 输出
        # 就是原始 value 的历史行为；Trainer 的新配置会显式传入启用实例。
        self.value_normalizer = (
            value_normalizer
            if value_normalizer is not None
            else ValueNormalizer(enabled=False)
        )
        self.device = torch.device(device)
        self.to(self.device)

    def _raw_value(self, state_t: torch.Tensor) -> torch.Tensor:
        """把 Critic 的标准化网络输出还原为 GAE 使用的原始 value。"""

        normalized_value = self.critic(state_t)
        return self.value_normalizer.denormalize(normalized_value)

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
        # rollout buffer 和 GAE 必须保存 reward 原始尺度的 value；若把标准化
        # 网络输出直接写入 GAE，TD delta 会混合两个坐标系。
        value = self._raw_value(state_t)
        value_np = value.detach().cpu().numpy().astype(np.float32)
        # 单环境沿用标量接口，向量环境返回一维数组；显式联合类型可防止静态
        # 检查器仅根据第一个分支把 value_out 错误推断为永远是 float。
        value_out: float | np.ndarray
        if value_np.size == 1:
            value_out = float(value_np.reshape(-1)[0])
        else:
            value_out = value_np
        return (
            actions.cpu().numpy().astype(np.int64),
            log_probs.cpu().numpy().astype(np.float32),
            value_out,
            entropy.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def get_value(self, state):
        value = self._raw_value(to_tensor(state, self.device))
        value_np = value.detach().cpu().numpy().astype(np.float32)
        if value_np.size == 1:
            return float(value_np.reshape(-1)[0])
        return value_np

    @torch.no_grad()
    def update_state_statistics(self, states) -> None:
        """在 Trainer 指定的 rollout 边界更新状态统计，评估时不会自动调用。"""

        self.critic.update_state_statistics(to_tensor(states, self.device))

    @torch.no_grad()
    def update_value_statistics(self, returns) -> None:
        """用完整 rollout 的 raw returns 更新一次 ValueNorm。"""

        self.value_normalizer.update(to_tensor(returns, self.device))

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """供 loss 将 raw returns/old values 映射到 Critic 训练坐标。"""

        return self.value_normalizer.normalize(values)

    def evaluate_actions(self, obs, state, actions, action_mask):
        obs_t = to_tensor(obs, self.device)
        state_t = to_tensor(state, self.device)
        actions_t = to_tensor(actions, self.device, dtype=torch.long)
        mask_t = to_tensor(action_mask, self.device)
        distribution = Categorical(logits=self.actor(obs_t, mask_t))
        return (
            distribution.log_prob(actions_t),
            distribution.entropy(),
            # PPO Critic loss 在标准化坐标中计算；只有离开 policy、进入 GAE
            # 或日志时才反标准化。
            self.critic(state_t),
        )
