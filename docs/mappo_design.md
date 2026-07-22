# 第四阶段：最小 MAPPO 训练闭环

## 目标

本阶段把 future-mutex 多流环境接入 MAPPO，完成动作采样、并行 rollout、GAE、
PPO 更新、日志、checkpoint 和确定性评估。默认配置面向 512 线程/A100 机器，
通过 subprocess vectorized env 并行收集样本。

## 输入输出与网络

- Actor 输入：`obs (F, A, 8)` 与 `action_mask (F, A)`；
- Actor 输出：每个 agent 对 A 个候选动作的 logits；
- Centralized critic 输入：全局 `state (7,)`；
- Critic 输出：共享标量 `V(state)`。

共享 Actor 对每个动作的 8 维路径特征独立应用同一个 MLP。非法动作 logits 被
填为 `-1e9`；若 mask 全为 0，临时强制动作 0 可采样以避免 Categorical NaN。
Critic 使用输入 LayerNorm 加普通 MLP，以适应 state 中边数、米、秒等不同量纲；
它不读取 6080 节点全图。

## Rollout、GAE 与 PPO

`RolloutBuffer` 保留 `(T, num_envs, ...)` 维度，支持 `num_envs>=1`。它存储
obs、state、mask、actions、old log-prob、reward、done 和 value。团队 reward
对应全局 GAE advantage，并在 actor loss 中广播给所有 agents：

```text
delta_t = r_t + gamma * V_{t+1} * (1-done_t) - V_t
A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
R_t = A_t + V_t
```

PPO 使用 clipped surrogate actor loss、return MSE critic loss和 entropy bonus，
支持 advantage 标准化与 gradient clipping。

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
├── metrics/train_metrics.csv
└── checkpoints/
    ├── checkpoint_000050.pt
    └── latest.pt
```

默认 250 updates 面向真实运行；冒烟调试时可减小 `num_envs`、rollout length、
updates、PPO epochs 和候选路径数。

## 评估

```bash
python scripts/run/05_evaluate_methods.py \
  --env-config configs/env.yaml \
  --mappo-config configs/mappo.yaml \
  --checkpoint outputs/runs/<run>/checkpoints/latest.pt \
  --methods mappo
```

统一评估入口使用 action mask 后 logits 的 argmax，输出并保存
total reward、时延、future mutex、outage、switch、新链路数量和
非法动作数。使用 `--methods all` 可让 MAPPO 与所有已注册 baseline
在相同 traffic 上按同一口径比较。

## 当前限制

- 无 GNN，仅处理候选路径特征；
- vectorized env 使用 subprocess，环境交互仍是 CPU-bound；
- 未实现 recurrent policy、学习率调度或复杂 value clipping；
- 当前候选路径为离线 SciPy/Dijkstra 近似多路径；如需精确 K-shortest，可把
  `path_weight.method` 改成 `yen`，但预处理会显著变慢。
