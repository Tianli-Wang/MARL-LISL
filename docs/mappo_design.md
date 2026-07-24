# 第四阶段：最小 MAPPO 训练闭环

## 目标

本阶段把 future-mutex 多流环境接入 MAPPO，完成动作采样、并行 rollout、GAE、
PPO 更新、日志、checkpoint 和确定性评估。默认配置面向 512 线程/A100 机器，
通过 subprocess vectorized env 并行收集样本。

## 输入输出与网络

- Actor 输入：`obs (F, A, 8)` 与 `action_mask (F, A)`；
- Actor 输出：每个 agent 对 A 个候选动作的 logits；
- Centralized critic 输入：紧凑全局 `state (7 + 10F,)`，16 条流时为
  `state (167,)`；
- Critic 输出：共享标量 `V(state)`。

共享 Actor 对每个动作的 8 维路径特征独立应用同一个 MLP。非法动作 logits 被
填为 `-1e9`；若 mask 全为 0，临时强制动作 0 可采样以避免 Categorical NaN。
state 前 7 维保留时间、全图边统计、活跃流数量和 keep future mutex。之后每条
流依次拼接：

```text
keep feasible / propagation / minimum lifetime / hops
candidate legal ratio
minimum-delay candidate: total delay / new-link count
minimum-mutex candidate: mutex / total delay / new-link count
```

候选最小值只从 action mask 合法的候选中计算，避免不可行候选的全零填充值被
误判为最优。Critic 使用逐特征 RunningMeanStd 加普通 MLP，不读取 6080 节点
全图。state 的最前面 7 维保持旧顺序，因此历史 `state_dim=7` checkpoint 仍可
在新环境中评估。

## Rollout、GAE 与 PPO

`RolloutBuffer` 保留 `(T, num_envs, ...)` 维度，支持 `num_envs>=1`。它存储
obs、state、mask、actions、old log-prob、reward、done 和 value。团队 reward
对应全局 GAE advantage，并在 actor loss 中广播给所有 agents：

```text
delta_t = r_t + gamma * V_{t+1} * (1-done_t) - V_t
A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
R_t = A_t + V_t
```

Critic 网络拟合 ValueNorm 标准化后的 return；送入 GAE、explained variance 和
评估日志的 value 会反标准化回原始 reward 尺度。状态统计在完整 rollout 后只
更新一次，随后统一重算 old values，并在该轮所有 PPO epoch 内冻结。
训练日志同时记录采样时的 `explained_variance` 和当前 batch 更新后的
`explained_variance_after_update`；前者衡量跨 rollout 预测能力，后者只表示
本批数据的拟合能力，二者应结合判断是否过拟合。

PPO 使用 clipped surrogate actor loss、标准化 return MSE critic loss和 entropy
bonus，支持 advantage 标准化。Actor/Critic 使用相互独立的 Adam、学习率和
gradient clipping，避免大尺度 Critic 误差污染 Actor 的优化器状态。checkpoint
版本 2 会分别保存两套 optimizer、状态 RunningMeanStd 和 ValueNorm；读取旧版
单 optimizer checkpoint 时恢复网络权重，但安全跳过无法可靠拆分的 Adam moments。

## 训练

先依次生成图、traffic pairs 和 node mutex，再运行：

```bash
python scripts/run/03_train_mappo.py \
  --env-config configs/env.yaml \
  --mappo-config configs/mappo.yaml
```

输出结构：

```text
outputs/runs/<timestamp>_mappo_debug/
├── config.json
├── metrics/
│   ├── train_metrics.csv
│   └── validation_metrics.csv
└── checkpoints/
    ├── best.pt
    ├── checkpoint_000050.pt
    └── latest.pt
```

默认 250 updates 面向真实运行；冒烟调试时可减小 `num_envs`、rollout length、
updates、PPO epochs 和候选路径数。

训练与选模采用两套严格分离的轨迹：

- rollout 使用 train traffic 和随机起点，用于扩大训练覆盖范围；
- 每 25 updates 使用 eval traffic、固定 `k=0` 和确定性 argmax 完整验证；
- `best.pt` 按 `outage → future mutex → total reward → delay` 的优先级保存；
- 连续 3 次验证没有改善且训练已达到 75 updates 时 early stop。

因此 `latest.pt` 只表示训练停止时的最后参数，不保证是最优模型。正式比较和
论文结果应优先显式指定 `best.pt`，并通过 `validation_metrics.csv` 追溯其轮次。

## 评估

```bash
python scripts/run/05_evaluate_methods.py \
  --env-config configs/env.yaml \
  --mappo-config configs/mappo.yaml \
  --checkpoint outputs/runs/<run>/checkpoints/best.pt \
  --methods mappo
```

统一评估入口使用 action mask 后 logits 的 argmax。使用 `--methods all`
可让 MAPPO 与所有已注册 baseline 在相同 traffic 上按同一口径比较。

评估会同时生成两份 CSV：

- `method_compare.csv`：方法级汇总，包含 reward、平均/P95/P99/峰值时延、
  最差业务流平均时延、future mutex 总量与发生率、outage/switch/new-link
  总量与每 flow-step 归一化比率、链路保持率、平均跳数、平均路径距离和
  在线决策耗时；
- `method_compare_per_flow.csv`：逐流明细，用于发现总体平均值掩盖的“问题流”，
  包含每条流的时延分布、outage、切换、新链路、保持率、跳数与路径距离。

其中决策耗时只测量策略的 `act` 调用，不包含环境推进、NetworkX 路径搜索和
future-mutex 计算时间；因此它表示算法在线选路开销，而不是整个仿真步耗时。
时延分位数只统计成功建立路径的流，路径失败则单独进入 outage/setup-failure
指标，避免用虚假的零时延美化结果。

## 当前限制

- 无 GNN，仅处理候选路径特征；
- vectorized env 使用 subprocess，环境交互仍是 CPU-bound；
- 未实现 recurrent policy、学习率调度或 PopArt；ValueNorm 模式默认关闭 value
  clipping，由独立 Critic 梯度裁剪限制更新幅度；
- 当前候选路径为离线 SciPy/Dijkstra 近似多路径；如需精确 K-shortest，可把
  `path_weight.method` 改成 `yen`，但预处理会显著变慢。
