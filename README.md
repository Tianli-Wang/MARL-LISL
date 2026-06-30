# MARL-LISL

低轨巨星座星间激光通信（LISL）多流路由项目。当前已完成 STK 卫星状态到
稀疏动态图快照的转换、可执行 `reset/step` 的最小多源多宿路由环境，以及未来
节点互斥检测、主动规避规则和最小 MAPPO 训练闭环。

## 目录职责

```text
data/       只放数据
src/        只放源码
scripts/    只放运行入口
configs/    只放参数
docs/       只放说明文档
```

核心实现位于 `src/marl_lisl/preprocess/`，入口脚本只读取配置并调用核心函数。
所有图按时隙单独保存，便于后续环境懒加载。

第二阶段源码位于 `src/marl_lisl/store/` 和 `src/marl_lisl/envs/`；运行入口仍只
放在 `scripts/`，环境参数集中在 `configs/env.yaml`。

根目录运行脚本已按推荐执行顺序编号：先环境/互斥冒烟测试，再规则策略、MAPPO
训练与评估；`scripts/preprocess/` 仍按数据预处理流水线编号。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据位置

将 721 个 STK 文件放在 `data/raw_stk/by_step/`。当前数据已按时间顺序命名为
`step_0000.txt` 至 `step_0720.txt`。如果规范目录不存在，程序会兼容读取项目
根目录下的 `raw_stk/`。

## 运行

在项目根目录依次执行：

```bash
python scripts/preprocess/01_build_sat_state.py
python scripts/preprocess/02_build_graph_snapshots.py
python scripts/preprocess/03_check_processed_data.py
python scripts/preprocess/04_build_traffic_pairs.py --config configs/env.yaml --workers 2
python scripts/preprocess/05_build_mutex.py --config configs/env.yaml
python scripts/preprocess/07_build_mutex_stress_traffic.py --config configs/env.yaml
python scripts/preprocess/08_pack_graphs.py --config configs/env.yaml
python scripts/preprocess/06_build_candidates.py --config configs/env.yaml --split all --workers 128
```

前两个构建入口默认使用 `configs/preprocess.yaml` 中的 `parallel_workers: 128`，
也可以通过 `--workers N` 覆盖。03 检查脚本已设为串行；链路剩余寿命因存在
反向时序依赖，也按顺序计算。

也可显式指定配置：

```bash
python scripts/preprocess/01_build_sat_state.py --config configs/preprocess.yaml
python scripts/preprocess/02_build_graph_snapshots.py --config configs/preprocess.yaml
python scripts/preprocess/03_check_processed_data.py --config configs/preprocess.yaml
```

输出位置：

- `data/sat_state/`：固定卫星编号的状态数组与时间索引；
- `data/graphs/dmax_2000km/`：逐时隙稀疏动态图快照。

参数见 [configs/preprocess.yaml](configs/preprocess.yaml)，数据格式、边特征和
MAPPO 环境读取方式见 [docs/preprocess.md](docs/preprocess.md)。

## 第二阶段：最小路由环境

先基于 `graph_0000` 生成固定的训练与评估 traffic pairs：

```bash
python scripts/preprocess/04_build_traffic_pairs.py --config configs/env.yaml --workers 4
python scripts/preprocess/08_pack_graphs.py --config configs/env.yaml
python scripts/preprocess/06_build_candidates.py --config configs/env.yaml --split both --workers 128
```

再用合法随机动作执行环境：

```bash
python scripts/01_test_env_step.py --config configs/env.yaml --steps 20
```

当前 `future_mutex.enabled: true`，首次运行环境前还需执行第三阶段的
`05_build_mutex.py` 命令生成节点容量文件。当前 `candidates.enabled: true` 且
`candidates.backend: packed`，环境会直接 mmap 读取
`data/candidates/{train,eval,stress}/_packed/`，不再在线执行慢速路径搜索；如果候选
文件缺失或维度不匹配，请先运行 `06_build_candidates.py`。该脚本会先生成
`cand_XXXX.npz`，再自动打包成查表用的 packed candidates。

每条 flow 是一个 agent，动作 `0` 表示保持当前路径，`1..K` 表示切换到对应
候选路径。默认 `PackedGraphStore` / `PackedCandidateStore` 使用 memmap 查表；
若改回 lazy 后端，环境只按缓存窗口懒加载图和候选。
Observation、全局 state、reward、traffic 格式和当前实现边界详见
[docs/env_design.md](docs/env_design.md)。

## 第三阶段：未来互斥与主动规避

```bash
python scripts/preprocess/05_build_mutex.py --config configs/env.yaml
python scripts/preprocess/08_pack_graphs.py --config configs/env.yaml
python scripts/preprocess/06_build_candidates.py --config configs/env.yaml --split both --workers 128
python scripts/02_test_future_mutex.py --config configs/env.yaml
python scripts/03_run_proactive_rule.py --config configs/env.yaml --steps 20
```

环境 observation 已扩展为 8 维候选路径特征，新增 `A_mutex`（动作后的未来冲突）
和 `B_avoid`（相对保持动作的规避收益）；全局 state 增至 7 维，reward 增加未来
互斥惩罚。`ProactiveRulePolicy` 用于验证提前切换能否降低未来共享中继节点冲突。

## 第四阶段：最小 MAPPO 闭环

```bash
python scripts/04_train_mappo.py --env-config configs/env.yaml --mappo-config configs/mappo.yaml
python scripts/05_evaluate_mappo.py \
  --env-config configs/env.yaml --mappo-config configs/mappo.yaml \
  --checkpoint outputs/runs/<run>/checkpoints/latest.pt --episodes 4 --workers 4
```

当前 MAPPO 使用共享候选动作 MLP Actor 和 centralized critic，支持 action mask、
GAE、clipped PPO、多进程 rollout、CSV metrics 与 checkpoint。详见
[docs/mappo_design.md](docs/mappo_design.md)。训练循环已接入 `tqdm` update 级进度条，
会实时显示 reward、future mutex、outage、actor/critic loss 和 entropy。

## 并行计算说明

- `01_build_sat_state.py`：按 STK 时隙文件并行解析与写入；
- `02_build_graph_snapshots.py`：按图快照并行构建，剩余寿命反向扫描保持串行；
- `03_check_processed_data.py`：按之前设定保持串行检查；
- `04_build_traffic_pairs.py`：训练/评估源宿对生成可并行；
- `08_pack_graphs.py`：把逐时隙图打包为共享 memmap，避免多进程反复解压 npz；
- `06_build_candidates.py`：离线预计算各时隙候选路径，并自动打包为 packed candidates；
- `09_pack_candidates.py`：已有 `cand_XXXX.npz` 时单独重建候选路径 pack；
- 若关闭 `candidates.enabled`，环境运行时才会在线生成候选路径；
- `05_evaluate_mappo.py`：多个评估 episode 可用 `--workers` 并行。

预处理长循环已接入 `tqdm` 进度条；如果运行环境暂未安装 `tqdm`，代码会自动退回
普通输出。建议执行 `pip install -r requirements.txt` 以获得完整进度显示。

## 高性能机器参数

当前默认已按 512 CPU 线程、A100 80G、720G 内存调整：

- `configs/preprocess.yaml`：`parallel_workers: 128`；
- `configs/env.yaml`：`num_flows: 16`、`num_candidates: 8`、`parallel_workers: 128`、
  `future_window: 24`、`graph_backend: packed`、`candidates.backend: packed`；
- `configs/mappo.yaml`：`device: cuda`、`num_envs: 32`、`rollout_length: 256`、
  Actor/Critic 网络加宽，并带轻量启发式 action prior。

建议运行前设置 BLAS 线程为 1，避免多进程建图时每个进程再开很多内部线程：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

由于 flow 数和候选动作数已改变，请重新生成 traffic pairs 和候选路径：

```bash
python scripts/preprocess/04_build_traffic_pairs.py --config configs/env.yaml --workers 2
python scripts/preprocess/07_build_mutex_stress_traffic.py --config configs/env.yaml
python scripts/preprocess/08_pack_graphs.py --config configs/env.yaml
python scripts/preprocess/06_build_candidates.py --config configs/env.yaml --split all --workers 128
```

### GPU 使用边界

当前只有 MAPPO 的 Actor/Critic 前向、反向传播和 PPO 更新会使用 CUDA。以下部分仍
是 CPU-bound：STK 预处理、KDTree 建图、离线 SciPy/Dijkstra 候选路径预计算、
future mutex 检测、
`01_test_env_step.py` 环境冒烟测试和 proactive rule。也就是说，运行预处理或环境
测试时 `nvidia-smi` 显示 0% 是正常的；训练阶段会打印实际使用的 CUDA device。

当前训练瓶颈主要来自动态图和候选路径读取。`configs/env.yaml` 默认使用 packed
graph/candidates：图快照被打包到 `data/graphs/dmax_2000km/_packed/`，候选路径被
打包到 `data/candidates/<split>/_packed/`。多进程训练时这些 mmap 文件由 OS page
cache 共享，不会让每个 worker 各自解压一份 10GB+ 的图数据。

`03_run_proactive_rule.py` 默认只运行 20 步；它用于观察规则是否能降低未来互斥，
不需要每次跑满 721 步。若确实要完整规则基线，再显式加 `--full-episode`。

MAPPO trainer 已支持 subprocess vectorized env，通过 `num_envs` 并行收集 rollout。
GPU 仍主要用于 actor/critic 前向、反向和 PPO 更新；环境 step、future mutex 与离线
候选路径预计算仍是 CPU-bound。

## 第五阶段：Baseline 与 stress traffic 对比

```bash
python scripts/preprocess/06_build_mutex_stress_traffic.py --config configs/env.yaml
python scripts/diagnose_future_mutex.py --config configs/env.yaml --traffic data/traffic/traffic_pairs_stress.npy
python scripts/run_baselines.py --config configs/env.yaml --traffic data/traffic/traffic_pairs_stress.npy
python scripts/evaluate_all_methods.py \
  --env-config configs/env.yaml \
  --mappo-config configs/mappo.yaml \
  --checkpoint outputs/runs/<run_name>/checkpoints/latest.pt \
  --traffic data/traffic/traffic_pairs_stress.npy
```

详见 [docs/baseline_evaluation.md](docs/baseline_evaluation.md)。
