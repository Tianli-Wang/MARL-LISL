# MARL-LISL

面向低轨巨星座星间激光通信（LISL）的多业务流协同路由项目。项目包含 STK
轨迹预处理、动态稀疏图构建、离线候选路径、未来路径互斥检测、规则 baseline
和 MAPPO 训练评估闭环。

## 目录

```text
configs/    环境、预处理与 MAPPO 参数
data/       原始数据和全部预处理产物
docs/       环境、算法及实验说明
scripts/    用户直接运行的命令行入口
  run/      环境测试、规则、训练和评估入口
  preprocess/ 数据预处理与离线打包入口
src/        可复用的核心实现
```

入口脚本只负责解析参数和调用 `src/marl_lisl/` 中的实现。运行脚本共享
`src/marl_lisl/utils/runtime_config.py`，统一解析 graph、traffic 和 candidate
路径，避免训练、测试、评估使用不同的数据集合。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell 使用 .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 脚本功能

### 运行与评估入口

| 脚本 | 功能 |
|---|---|
| `scripts/run/01_test_env.py` | 统一的环境冒烟测试；支持基础 `reset/step`、future mutex 详情以及 train/eval/stress 数据集 |
| `scripts/run/02_run_proactive_rule.py` | 逐时隙运行主动互斥规避规则，打印 keep/after mutex、动作、奖励和耗时 |
| `scripts/run/03_train_mappo.py` | 创建单环境或多进程向量环境，训练 MAPPO 并保存指标和 checkpoint |
| `scripts/run/04_evaluate_mappo.py` | 确定性评估 MAPPO checkpoint，支持多 episode 和多进程 |
| `scripts/run/05_evaluate_methods.py` | 统一评估全部 baseline、轻量互斥诊断策略和可选 MAPPO，并输出同口径 CSV |

### 数据预处理入口

| 脚本 | 功能 |
|---|---|
| `scripts/preprocess/01_build_sat_state.py` | 把逐时隙 STK 文件转换为固定卫星编号的状态数组和有效掩码 |
| `scripts/preprocess/02_build_graph_snapshots.py` | 构建逐时隙 LISL 稀疏图，并反向计算链路剩余寿命 |
| `scripts/preprocess/03_check_processed_data.py` | 检查状态数组和抽样图快照的形状、数值及边结构 |
| `scripts/preprocess/04_build_traffic.py` | 统一生成普通 train/eval traffic 和 future-mutex stress traffic |
| `scripts/preprocess/05_build_candidates.py` | 离线计算 train/eval/stress 每个时隙、每条 flow 的候选路径 |
| `scripts/preprocess/06_pack_data.py` | 统一把图快照和候选路径打包成多进程可共享的 memmap 查找表 |

## 完整数据准备流程

### 一键执行全部流程

在项目根目录运行以下命令，即可按依赖顺序完成数据预处理、环境检查、规则诊断、
MAPPO 训练、专项评估和统一方法比较：

```bash
python run_all.py
```

入口支持断点式执行，也能临时缩小训练规模，且不会修改 YAML 配置：

```bash
python run_all.py --start-at env_test
python run_all.py --start-at train --num-envs 2 --rollout-length 4 --total-updates 1 --max-eval-steps 5
python run_all.py --dry-run
```

运行 `python run_all.py --help` 可查看起止步骤、跳过步骤、并行数、checkpoint
和评估规模等选项。任一子流程失败时，入口会立即停止并返回失败步骤。

### 手动执行数据准备

将 721 个 STK 文件放到 `data/raw_stk/by_step/`，默认名称为
`step_0000.txt` 至 `step_0720.txt`，然后在项目根目录运行：

```bash
# 1. STK → 固定编号卫星状态
python scripts/preprocess/01_build_sat_state.py --config configs/preprocess.yaml

# 2. 卫星状态 → 721 张动态稀疏图
python scripts/preprocess/02_build_graph_snapshots.py --config configs/preprocess.yaml

# 3. 检查状态和图数据
python scripts/preprocess/03_check_processed_data.py --config configs/preprocess.yaml

# 4. 生成 train/eval 和 stress 三套业务流
python scripts/preprocess/04_build_traffic.py --config configs/env.yaml --split all --workers 2

# 5. 先把图打包为共享 memmap，供并行候选路径预处理读取
python scripts/preprocess/06_pack_data.py --config configs/env.yaml --target graphs

# 6. 离线生成三套候选路径；packed 后端下生成完成后会自动打包
python scripts/preprocess/05_build_candidates.py \
  --config configs/env.yaml --split all --workers 128

# 7. 如需单独重建或校验全部 pack
python scripts/preprocess/06_pack_data.py \
  --config configs/env.yaml --target all --split all --force
```

若省略第 5 步且 `graph_pack_build_if_missing: true`，候选路径脚本会在启动 worker
之前由主进程自动补建一次 graph pack；不会让多个 worker 并发写同一 pack。

前两个预处理步骤可用 `--workers N` 覆盖 `configs/preprocess.yaml` 中的并行数。
链路剩余寿命存在反向时序依赖，因此图生成可以并行，寿命回填必须串行。

主要输出为：

```text
data/sat_state/                         卫星状态、有效掩码和时间索引
data/graphs/dmax_2000km/                逐时隙图及 _packed 图查找表
data/traffic/                           train/eval/stress traffic pairs
data/candidates/{train,eval,stress}/    离线候选路径及 _packed 查找表
```

只要图、traffic、`num_flows`、`num_candidates` 或 `path_weight` 改变，就应重新生成
候选路径；图快照发生变化时还应重新生成 graph pack。

## 环境与 future mutex 测试

统一测试入口取代了原来的 `01_test_env_step.py` 和
`02_test_future_mutex.py`：

```bash
# 同时检查环境 API 和 future mutex
python scripts/run/01_test_env.py --config configs/env.yaml --mode all --split train --steps 20

# 只做基础 reset/step 冒烟测试
python scripts/run/01_test_env.py --config configs/env.yaml --mode basic --steps 5

# 使用 stress traffic 专门检查互斥压力
python scripts/run/01_test_env.py --config configs/env.yaml --mode mutex --split stress --steps 20
```

每条 flow 是一个 agent。动作 `0` 保持当前路径，动作 `1..K` 切换到对应离线候选
路径。每个候选动作的 observation 为：

```text
[T_prop, T_setup, R_min, N_new, hop_count, feasible, A_mutex, B_avoid]
```

其中 `A_mutex` 是执行该动作后的未来路径对冲突量，`B_avoid` 是相对保持动作减少的
冲突量。任意两条有效路径共享至少一个中继卫星即记 1 次互斥；同一路径对共享
多个节点仍只计 1 次。该定义不读取节点或链路容量，默认也不统计源宿卫星。

## 主动规则 baseline

```bash
# 默认快速运行 20 步
python scripts/run/02_run_proactive_rule.py --config configs/env.yaml --steps 20

# 显式运行完整 episode
python scripts/run/02_run_proactive_rule.py --config configs/env.yaml --full-episode
```

脚本打印 `future_mutex_keep`、联合动作后的 `future_mutex_after`、实际规避量、路径
切换、新链路和掉线数量。它是逐步诊断入口，不用于批量方法对比。

## MAPPO 训练与专项评估

```bash
python scripts/run/03_train_mappo.py \
  --env-config configs/env.yaml --mappo-config configs/mappo.yaml

python scripts/run/04_evaluate_mappo.py \
  --env-config configs/env.yaml \
  --mappo-config configs/mappo.yaml \
  --checkpoint outputs/runs/<run>/checkpoints/latest.pt \
  --episodes 4 --workers 4
```

训练支持 subprocess vectorized env、共享 Actor、centralized Critic、action mask、
GAE、clipped PPO、CSV metrics 和 checkpoint。`num_agents` 必须等于 `num_flows`，
`num_actions` 必须等于 `num_candidates + 1`。

## 统一方法评估

`05_evaluate_methods.py` 取代了原来的 `run_baselines.py`、
`diagnose_future_mutex.py` 和 `evaluate_all_methods.py`。

```bash
# 所有 baseline
python scripts/run/05_evaluate_methods.py \
  --traffic data/traffic/traffic_pairs_stress.npy \
  --methods baselines

# 只运行 Maintain、Shortest、Proactive，诊断 stress traffic 是否有互斥压力
python scripts/run/05_evaluate_methods.py \
  --traffic data/traffic/traffic_pairs_stress.npy \
  --methods diagnose --max-steps 100

# baseline 与 MAPPO 使用同一 traffic、同一指标口径比较
python scripts/run/05_evaluate_methods.py \
  --env-config configs/env.yaml \
  --mappo-config configs/mappo.yaml \
  --checkpoint outputs/runs/<run>/checkpoints/latest.pt \
  --traffic data/traffic/traffic_pairs_stress.npy \
  --methods all \
  --output outputs/tables/stress_compare.csv
```

如果传入配置之外的自定义 traffic，统一配置加载器会关闭离线候选路径并回退在线
路径生成，避免把 train/eval/stress 候选路径错配给其他源宿对。

## 性能说明

- `graph_backend: packed`：图快照通过共享 memmap 读取，避免多进程重复解压 NPZ。
- `candidates.backend: packed`：候选路径离线计算，环境不再逐时隙运行最短路搜索。
- future mutex 会缓存路径边编码和中继节点集合，降低路径对求交的重复开销。
- GPU 只负责 MAPPO Actor/Critic 的前向、反向与 PPO 更新；环境 step、预处理和
  future mutex 仍是 CPU 任务。
- 多环境训练时建议把 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、
  `OPENBLAS_NUM_THREADS` 设为 1，避免每个环境进程再次创建大量 BLAS 线程。

更详细的数据格式和算法说明见：

- [预处理设计](docs/preprocess.md)
- [环境设计](docs/env_design.md)
- [MAPPO 设计](docs/mappo_design.md)
- [Baseline 评估](docs/baseline_evaluation.md)
