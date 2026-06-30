# 第五阶段：Baseline、Stress Traffic 与统一评估

## 为什么需要 baseline

单独看到 MAPPO 能输出 reward 和 delay，并不能证明方法优质。尤其当
`future_mutex = 0` 时，可能只是 traffic 场景本身没有共享中继节点冲突，无法验证
“提前切换规避未来互斥”的核心机制。

因此需要同时比较短视低时延策略、保持策略、future-mutex 贪心策略和主动规避规则。

## Baseline 策略

- `ShortestDelay`：每条 flow 选择合法动作中 `T_prop + T_setup` 最小者。
- `MaintainUntilConflict`：保持动作合法就保持；否则选择最低时延合法候选。
- `GreedyConflictAware`：选择 `A_mutex` 最小的合法动作，并用时延打破并列。
- `ProactiveRule`：默认保持；当候选动作的 `B_avoid >= min_b_avoid` 时主动切换。

所有策略只读取 env 输出的 `obs/state/action_mask`，不直接访问或修改 env 内部状态。

## Mutex stress traffic

Stress traffic 从 `graph_0000.npz` 随机采样源宿对，计算 shortest path，并优先选择
中继节点集合有重叠的业务流。目标是让保持/短视策略更容易产生未来节点互斥。

生成命令：

```bash
python scripts/preprocess/06_build_mutex_stress_traffic.py --config configs/env.yaml
```

输出：

```text
data/traffic/traffic_pairs_stress.npy
data/traffic/traffic_stress_config.json
```

如果仍然没有 future mutex，可增大 `traffic.stress.num_flows`、增大
`future_mutex.future_window`，或增加 `traffic.stress.num_candidate_pairs`。

## 诊断 future mutex 压力

```bash
python scripts/diagnose_future_mutex.py \
  --config configs/env.yaml \
  --traffic data/traffic/traffic_pairs_stress.npy
```

如果所有方法 `future_mutex == 0`，该场景不能验证 proactive mutex avoidance。

## 运行 baseline 对比

```bash
python scripts/run_baselines.py \
  --config configs/env.yaml \
  --traffic data/traffic/traffic_pairs_stress.npy
```

结果保存到：

```text
outputs/tables/baseline_compare.csv
```

## MAPPO + Baselines 统一对比

```bash
python scripts/evaluate_all_methods.py \
  --env-config configs/env.yaml \
  --mappo-config configs/mappo.yaml \
  --checkpoint outputs/runs/<run_name>/checkpoints/latest.pt \
  --traffic data/traffic/traffic_pairs_stress.npy
```

结果保存到：

```text
outputs/tables/all_methods_compare.csv
```

如果 stress traffic 的 flow 数和当前 MAPPO checkpoint 的 agent 数不一致，脚本会跳过
MAPPO 并保留 baseline 对比。要评估 MAPPO，请保证训练配置和 traffic flow 数一致。

## 如何判断结果

- 如果所有方法 `future_mutex == 0`：场景没有 mutex 压力，需要 stress traffic。
- 如果 `ProactiveRule.future_mutex < MaintainUntilConflict.future_mutex`：主动规避机制有效。
- 如果 MAPPO 的 `future_mutex` 接近或低于 ProactiveRule，且 switch/new links 更低：
  MAPPO 可能学到了更好的 trade-off。
- 如果 MAPPO 的 switch 或 new links 远高于 baseline：可能过度切换，需要提高对应 reward 权重。
