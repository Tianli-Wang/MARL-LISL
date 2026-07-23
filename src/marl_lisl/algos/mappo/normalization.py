"""MAPPO 状态与价值目标使用的运行统计归一化工具。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """用数值稳定的并行公式维护逐特征均值与方差。

    统计量以 float64 保存，避免长时间训练后累计误差明显放大；真正送入网络
    时再转换为输入张量的 dtype。该类继承 ``nn.Module``，因此 mean、var 和
    count 会自动进入 checkpoint，并随模型一起移动到 CPU/GPU。
    """

    def __init__(
        self,
        shape: int | Sequence[int] | torch.Size = (),
        *,
        epsilon: float = 1e-4,
        normalization_epsilon: float = 1e-8,
        clip_range: float | None = None,
    ):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(int(size) for size in shape)
        if epsilon <= 0.0:
            raise ValueError("RunningMeanStd epsilon 必须大于 0")
        if normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon 必须大于 0")
        if clip_range is not None and float(clip_range) <= 0.0:
            raise ValueError("clip_range 必须为正数或 None")

        # 以一个极小的伪样本初始化，可保证第一次遇到常量 batch 时方差仍大于 0，
        # 同时不会对正常 batch 的真实统计量产生可见偏移。
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float64))
        self.normalization_epsilon = float(normalization_epsilon)
        self.clip_range = None if clip_range is None else float(clip_range)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """把一个 batch 合并进运行统计；空 batch 不改变任何状态。"""

        values = torch.as_tensor(values, device=self.mean.device)
        feature_shape = tuple(self.mean.shape)
        if feature_shape:
            if values.ndim == 0 or tuple(values.shape[-len(feature_shape) :]) != feature_shape:
                raise ValueError(
                    "RunningMeanStd 输入末尾维度与统计特征不一致: "
                    f"期望 {feature_shape}，实际 {tuple(values.shape)}"
                )
            values = values.reshape((-1,) + feature_shape)
        else:
            values = values.reshape(-1)
        if values.shape[0] == 0:
            return

        # batch moments 和合并过程统一使用 float64；unbiased=False 对单样本 batch
        # 也有定义，避免训练初始阶段出现 NaN。
        values_64 = values.to(dtype=torch.float64)
        batch_mean = values_64.mean(dim=0)
        batch_var = values_64.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            float(values_64.shape[0]), dtype=torch.float64, device=self.mean.device
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        """按 Chan 并行方差公式合并两组 moments。"""

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        correction = delta.square() * self.count * batch_count / total_count
        new_var = (old_m2 + batch_m2 + correction) / total_count

        self.mean.copy_(new_mean)
        # 浮点舍入偶尔可能产生极小负数；截断到 0 后，normalize 仍由 epsilon
        # 保证分母为正。
        self.var.copy_(torch.clamp(new_var, min=0.0))
        self.count.copy_(total_count)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """按当前冻结统计量逐特征标准化，并可选裁剪极端异常值。"""

        mean = self.mean.to(device=values.device, dtype=values.dtype)
        var = self.var.to(device=values.device, dtype=values.dtype)
        normalized = (values - mean) / torch.sqrt(var + self.normalization_epsilon)
        if self.clip_range is not None:
            normalized = torch.clamp(
                normalized, min=-self.clip_range, max=self.clip_range
            )
        return normalized

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """把标准化数值恢复到环境 reward/return 的原始坐标。"""

        mean = self.mean.to(device=values.device, dtype=values.dtype)
        var = self.var.to(device=values.device, dtype=values.dtype)
        return values * torch.sqrt(var + self.normalization_epsilon) + mean


class ValueNormalizer(nn.Module):
    """维护 return 的运行尺度，并定义 Critic 输出的坐标转换。

    启用后 Critic 网络拟合标准化 return，但交给 GAE、日志和评估的 value 会
    被反标准化回原始 reward 尺度。关闭后所有方法均为恒等变换，用于无缝读取
    旧 checkpoint。
    """

    def __init__(
        self,
        enabled: bool,
        *,
        epsilon: float = 1e-4,
        normalization_epsilon: float = 1e-8,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        # ValueNorm 不裁剪目标，否则极端但真实的 outage/mutex return 会被静默
        # 改写；异常梯度由 Critic 独立梯度裁剪负责。
        self.running = RunningMeanStd(
            (),
            epsilon=epsilon,
            normalization_epsilon=normalization_epsilon,
            clip_range=None,
        )

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> None:
        """每个 rollout 只调用一次，PPO 多个 epoch 内保持统计量不变。"""

        if self.enabled:
            self.running.update(returns)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """把原始 return/value 转换到 Critic 的训练坐标。"""

        return self.running.normalize(values) if self.enabled else values

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """把 Critic 网络输出恢复到 GAE 使用的原始 reward 坐标。"""

        return self.running.denormalize(values) if self.enabled else values
