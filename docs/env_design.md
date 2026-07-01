# 最小多流 LISL 路由环境

## 目标与边界

第二阶段提供了能稳定执行 `reset/step` 的多源多宿路由环境。第三阶段在此基础上
加入短窗口未来节点互斥检测和主动规避规则。每条业务流视为一个 agent，所有
agent 共享团队 reward。当前仍不包含 MAPPO、actor/critic 或训练缓冲区。

## 输入数据与懒加载

环境从 `data/graphs/dmax_2000km/graph_XXXX.npz` 读取动态图。`GraphStore` 使用
默认容量为 3 的 LRU cache，只按当前时隙加载 `edge_index` 与 `edge_attr`，不会
一次读入 721 张图，也不会构建 `6080 × 6080` dense adjacency。

业务文件位于 `data/traffic/`，数组形状为 `(num_flows, 3)`，每行为：

```text
[source_sat_id, dest_sat_id, demand]
```

当前 demand 固定为 `1.0`。运行以下命令从 `graph_0000` 生成可达的训练和评估对：

```bash
python scripts/preprocess/04_build_traffic.py --config configs/env.yaml --split normal
python scripts/preprocess/06_build_candidates.py --config configs/env.yaml --split both
```

## 候选路径与动作

默认配置 `candidates.enabled: true`，环境会从 `data/candidates/` 读取离线预计算
的 K 条候选路径。离线构建阶段使用 NetworkX `shortest_simple_paths`，边权为
传播时延、建链时延和剩余寿命倒数的加权和。若关闭预计算，环境才会退回在线
`PathGenerator`，但这会显著拖慢 reset/step。

每个 agent 的离散动作：

```text
0     保持 current_path
1..K  切换到对应候选路径
```

不可用动作由 `action_mask` 标为 0。若仍执行非法动作，该流路径变为 `None`，并
产生 `action_invalid_penalty`。

## Observation 与全局 State

每条流的 observation 形状为 `(K + 1, 8)`，第 0 行是保持动作，后续是候选路径：

```text
[T_prop, T_setup, R_min, N_new, hop_count, feasible, A_mutex, B_avoid]
```

- `T_prop`：路径传播时延之和；
- `T_setup`：相对当前路径新增链路的最大建链时延；
- `R_min`：路径最小剩余寿命；
- `N_new`：新增链路数；
- `hop_count`：跳数；
- `feasible`：所有路径边当前均存在时为 1。
- `A_mutex`：选择该动作对应路径后，未来窗口内的折扣互斥冲突量；
- `B_avoid`：保持当前路径的冲突量减去 `A_mutex`，正值表示该动作能规避冲突。

全局 state 不包含完整卫星图，固定为 7 维：

```text
[k_normalized, num_edges, mean_distance,
 mean_prop_delay, mean_residual_lifetime, num_active_flows, future_mutex]
```

## Reward

共同 reward 为：

```text
-(w_avg * avg_delay
  + w_peak * peak_delay
  + w_switch * switch_count
  + w_new * new_link_count
  + w_outage * outage_count
  + w_mutex * future_mutex)
```

非法动作惩罚在上述结果外额外扣除。流时延为传播时延之和加新增链路的最大
setup delay；不可用路径时延记 0，由 outage 项惩罚。

## API 与测试

```python
obs, state, action_mask = env.reset()
next_obs, next_state, next_action_mask, reward, done, info = env.step(actions)
```

形状分别为 `(F, K+1, 8)`、`(7,)` 和 `(F, K+1)`。测试合法随机动作：

```bash
python scripts/01_test_env.py --config configs/env.yaml --mode basic --steps 20
```

由于默认配置已启用 future mutex，首次创建环境前必须先运行下文的
`05_build_mutex.py`。

episode 默认从时隙 0 开始，最多执行 721 步。每步仅加载所需图快照，返回的
`info` 包含时延、切换、新链路、中断和非法动作统计。

## 第三阶段：未来节点互斥

先生成紧凑的一维节点容量数组；它不是 `6080 × 6080` 互斥矩阵：

```bash
python scripts/preprocess/05_build_mutex.py --config configs/env.yaml
```

`FutureMutexDetector` 从当前时隙向后检查 `future_window` 个时隙。仅当一条路径的
全部边在未来图中仍存在时，它才占用对应中继节点。节点占用超过
`node_mutex[node]` 的部分计为冲突，并乘以 `future_discount ** delta`。源宿节点
默认不计入互斥资源。

为避免对每个候选动作反复构造巨大 Python 边集合，检测器只缓存活动未来窗口的
排序整数边键，并用二分查找判断路径边是否存在。这仍是稀疏窗口缓存；图文件始终
通过 `GraphStore` 懒加载，且不会修改快照。

检查未来互斥特征：

```bash
python scripts/01_test_env.py --config configs/env.yaml --mode mutex
```

## ProactiveRulePolicy

手工规则在合法候选动作中选择 `B_avoid` 最大且达到 `min_b_avoid` 的路径，可选
限制 setup cost；互斥规避收益相同时，默认选择传播与建链总时延更低的动作。

```bash
python scripts/03_run_proactive_rule.py --config configs/env.yaml
```

该脚本逐步比较 `future_mutex_keep` 与 `future_mutex_after`。第三阶段的验证目标是
观察到 `future_mutex_keep > future_mutex_after` 和 `mutex_avoided > 0`。如果固定
traffic pairs 没有共享中继节点，零冲突是合法结果，可增加 flow 数、降低节点容量
或调整 traffic pairs 制造更强的竞争场景。第一版只处理节点互斥，不处理链路互斥。
