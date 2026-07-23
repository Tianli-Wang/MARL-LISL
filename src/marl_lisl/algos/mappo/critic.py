"""集中式标量状态价值 Critic。"""

from __future__ import annotations

import torch
from torch import nn

from .normalization import RunningMeanStd
from .utils import build_mlp


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "relu",
        state_normalization: str = "layer_norm",
        state_normalization_clip: float = 10.0,
        normalization_epsilon: float = 1e-8,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.state_normalization = str(state_normalization).strip().lower()
        if self.state_normalization == "running":
            # LayerNorm 会在每个样本内部混合“秒、米、计数、互斥量”等不同物理
            # 量纲，并丢失绝对水平；逐特征运行统计能保留这些全局大小关系。
            self.input_norm: nn.Module = RunningMeanStd(
                self.state_dim,
                normalization_epsilon=normalization_epsilon,
                clip_range=state_normalization_clip,
            )
        elif self.state_normalization == "layer_norm":
            # 旧 checkpoint 没有 state_normalization 配置，默认走此分支，模块名
            # 与参数形状均保持不变，从而仍可严格加载原来的 Critic 权重。
            self.input_norm = nn.LayerNorm(self.state_dim)
        elif self.state_normalization in {"none", "identity"}:
            self.input_norm = nn.Identity()
        else:
            raise ValueError(
                "critic.state_normalization 仅支持 running/layer_norm/none，"
                f"实际为 {state_normalization!r}"
            )
        self.value_net = build_mlp(
            self.state_dim, hidden_dim, num_layers, 1, activation
        )

    def _select_state_features(self, state: torch.Tensor) -> torch.Tensor:
        """选择本模型训练时使用的 state 前缀，并校验输入长度。

        新环境会输出扩展 state；旧 checkpoint 的 Critic 仍只认识最前面的
        7 个 legacy 全局特征。允许安全截取前缀后，统一评估脚本无需为了旧模型
        重建另一套环境。
        """

        if state.shape[-1] < self.state_dim:
            raise ValueError(
                "Critic state 维度不足: "
                f"至少需要 {self.state_dim}，实际为 {state.shape[-1]}"
            )
        if state.shape[-1] > self.state_dim:
            state = state[..., : self.state_dim]
        return state

    @torch.no_grad()
    def update_state_statistics(self, state: torch.Tensor) -> None:
        """显式更新逐特征统计；forward 和确定性评估绝不会隐式修改统计量。"""

        if isinstance(self.input_norm, RunningMeanStd):
            self.input_norm.update(self._select_state_features(state))

    @property
    def state_statistics_count(self) -> float:
        """返回累计状态样本数，便于训练日志诊断统计是否按预期更新。"""

        if isinstance(self.input_norm, RunningMeanStd):
            return float(self.input_norm.count.detach().cpu())
        return 0.0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = self._select_state_features(state)
        if isinstance(self.input_norm, RunningMeanStd):
            state = self.input_norm.normalize(state)
        else:
            state = self.input_norm(state)
        return self.value_net(state).squeeze(-1)
