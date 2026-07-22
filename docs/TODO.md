下一步先不要盲目调学习率或继续加训练轮数。现有证据表明，主要问题不是“训练时间不够”，而是评估口径、奖励尺度和 Critic 本身需要先校正。

## 目前最明显的三个问题

1. Critic 几乎没有学到

训练前 25 个 update 到最后 25 个 update：

- `critic_loss`：约 65.9 万 → 4500，数值仍然很大。
- `explained_variance`：始终接近 `0`。
- `approx_kl` 极小，Actor 的 PPO 更新幅度也很弱。

这意味着价值网络基本不能解释回报变化。继续从 250 轮加到 1000 轮，未必解决问题。

根本原因可能包括：

- 回报量级过大且变化剧烈。
- 代码注释提到 value normalization，但当前 loss 中没有真正实现 return/value normalization。
- Critic 的 7 维全局状态信息太少，只包含时间、全图统计、活跃流数量和总互斥量，没有描述各流当前路径和候选路径状况。

2. 奖励被 future mutex 主导

训练初期平均每步大约：

```text
mutex:      12.9 × 5.0 ≈ 64.5
new links:  76.2 × 0.1 ≈ 7.6
switches:    9.7 × 0.1 ≈ 1.0
delay:                    < 0.2
```

mutex 惩罚比时延大几百倍。模型首先学会“不要冲突、少切换”，但很难进一步优化时延。

这也解释了训练现象：

- future mutex：`12.89 → 0.56`
- 切换：`9.66 → 2.25`
- 新链路：`76.19 → 14.91`
- 平均时延只从 `0.0673 → 0.0537`

模型确实学到了东西，只是学到的几乎全是“保守”。

3. 当前评估不足以证明 MAPPO 好坏

现有结果只比较了 RSMR 和 MAPPO，而且其他 baseline 在 [registry.py](D:/MARL-LISL/src/marl_lisl/baselines/registry.py) 中被注释掉了。

此外，当前评估结果中两者的 future mutex 都是 `0`。这意味着评估场景没有测到这个项目最核心的“未来冲突规避”能力。

目前 train/eval 也各只有一套固定 traffic。确定性 MAPPO 重复评估多个 episode 很可能只是重复同一个场景，不是真正的泛化测试。

## 建议按这个顺序推进

### 第一步：先建立可信的评估矩阵

恢复并运行这些 baseline：

- Maintain Until Conflict
- Shortest Delay
- Greedy Conflict Aware
- Proactive Rule
- RSMR
- MAPPO

至少分别评估：

- 普通 eval traffic：看时延、切换、建链。
- stress traffic：看 future mutex 规避能力。
- 多套未参与训练的随机 traffic：看泛化。

不要只看总奖励，重点分别比较：

```text
avg_delay
peak_delay
future_mutex
switch_count
new_link_count
outage_count
```

总奖励完全由人为权重决定，单看它容易误判。

### 第二步：修复 Critic，而不是先调超参数

优先修改：

1. 实现 running return/value normalization。
2. 对 Critic 输入逐维标准化，而不只是使用 `LayerNorm`。
3. 扩充 centralized state，使其包含多流路由状态，例如：

```text
各流当前路径时延
各流剩余寿命
各流候选可行率
各流 keep mutex
候选 mutex 的均值/最小值
当前路径切换年龄
```

不建议把 16 条流的全部原始候选特征直接摊平，可以先为每条流做统计汇总，再拼接成全局状态。

修复后的首要验收指标是：`explained_variance` 应明显大于 0，并随训练上升。如果仍长期接近 0，就不值得进行大规模训练。

### 第三步：重新标定奖励

先把不同代价归一化到相近量级。例如：

- 时延转换为毫秒或除以参考时延。
- `switch_count / num_flows`
- `new_link_count / num_flows`
- `outage_count / num_flows`
- future mutex 除以流对数量和未来窗口权重总和。

归一化之后再设权重。一个健康目标是：正常训练期间，各奖励分项的平均贡献不要相差数百倍。

同时把每个 reward component 单独写入训练 CSV，否则只能看到总结果，无法判断策略在牺牲什么。

### 第四步：做小规模消融，不要立刻跑完整训练

每组先跑约 30–50 updates，至少 3 个随机种子：

| 实验 | 目的 |
|---|---|
| 当前配置 | 对照组 |
| value normalization | 检查 Critic |
| 归一化奖励 | 检查目标尺度 |
| 关闭 heuristic prior | 判断是不是启发式先验主导 |
| 扩充 Critic state | 检查状态表达 |
| mutex 权重降低 | 找时延与冲突的折中 |

筛选标准：

- explained variance 是否上升。
- KL 是否不再接近零。
- stress mutex 是否下降。
- 普通 eval 时延是否恶化。
- 是否减少切换但不是退化成“一直保持”。

### 第五步：最后才调 PPO 参数

前四步完成后，再考虑：

- rollout 从 32 增加到 64/128。
- minibatch 与总样本数匹配。
- Actor/Critic 使用不同学习率。
- 适当提高 entropy，防止过早坍缩到保持动作。
- 根据 KL 调整 `ppo_epochs` 和学习率。

## 我认为最值得立即做的事情

第一轮应当完成这三个改动：

1. 恢复全部 baseline，建立 eval/stress 两套对照表。
2. 给 Critic 加真正的 value/return normalization。
3. 记录归一化前后的每个奖励分项，并降低 mutex 的尺度支配。

然后只跑 50 updates 的小实验。只有当 `explained_variance` 明显脱离 0、stress 场景优于规则方法时，再投入完整的 250+ updates 训练。

## 最终论文或实验中可以采用四级对照：
Shortest Path：传统最短时延基线。
RSMR + 动态排序 + Rip-up/Reroute：快速启发式。
Rolling-horizon MILP/MCF：小规模最优或近似最优基准。
MAPPO：大规模、快速在线推理方法。


RSMR 的局限主要有：
固定流顺序造成优先级偏置：前面的流先占用优质节点，后面的流只能绕路。
每条流只考虑当前局部最优，不会为了整体收益让前面的流稍微绕一点。
能保持就保持，可能保留一条虽然可行、但已经不适合全局路径组合的旧路径。
不考虑多条流联合换路带来的长期收益。
结果可能依赖业务流排列顺序。
例如：
RSMR：
流 1 选择代价 5 的最好路径
流 2 被遮罩后只能选择代价 20 的路径
总代价 = 25

全局协同：
流 1 改选代价 7 的路径
流 2 可以选择代价 8 的路径
总代价 = 15