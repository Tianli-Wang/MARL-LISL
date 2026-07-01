# STK 数据预处理说明

这套流程把 STK 导出的逐时隙卫星状态转换为固定卫星编号的 NumPy 数组，以及
可直接由 RL 环境按时隙加载的 LISL 动态图快照。

## 1. 原始数据

将 721 个 `.csv`、`.tsv` 或 `.txt` 文件放入 `data/raw_stk/by_step/`。默认数据覆盖一小时，
采样间隔 5 秒（0、5、…、3600 秒），每个文件包含 6080 颗卫星。脚本按文件名
中的数字自然排序，因此 `step2` 会排在 `step10` 前。

每个文件必须包含以下列；列名两侧空白会被清理：

```text
TimeStep Time SatName X_km Y_km Z_km Vx_km_s Vy_km_s Vz_km_s Valid
```

- 位置单位为 km，速度单位为 km/s。
- `Valid` 支持 `True/False`、`1/0`、`Yes/No`、`valid/invalid`（不区分大小写）。
- 支持逗号、Tab 和空白分隔。无引号的空白格式中，`Time` 内部可以包含空格。
- 同一文件中重复的 `TimeStep + SatName` 只保留最后一条。
- 程序流式扫描所有文件，按文件和行的首次出现顺序定义全局 `sat_id`。

## 2. 运行流程

安装依赖：

```bash
pip install -r requirements.txt
```

在项目根目录依次运行：

```bash
python scripts/preprocess/01_build_sat_state.py
python scripts/preprocess/02_build_graph_snapshots.py
python scripts/preprocess/03_check_processed_data.py
python scripts/preprocess/04_build_traffic.py --config configs/env.yaml --split all
python scripts/preprocess/05_build_candidates.py --config configs/env.yaml --split both
```

三个脚本均支持显式指定 YAML 配置：

```bash
python scripts/preprocess/01_build_sat_state.py --config configs/preprocess.yaml
python scripts/preprocess/02_build_graph_snapshots.py --config configs/preprocess.yaml
python scripts/preprocess/03_check_processed_data.py --config configs/preprocess.yaml
```

前两个构建阶段默认读取 `parallel_workers` 并行执行，也可临时覆盖：

```bash
python scripts/preprocess/01_build_sat_state.py --workers 4
python scripts/preprocess/02_build_graph_snapshots.py --workers 4
```

- 状态合并的文件扫描与数值转换使用进程池；主进程按时隙写 memmap，避免并发写冲突。
- 各时隙图互相独立，由进程池并行构建并直接写入不同的 `.npz` 文件。
- 第三个检查阶段按时隙和抽样图串行执行，保证输出顺序稳定、warning 易于定位。
- 剩余寿命依赖下一时隙结果，因此必须按 `0720 → 0000` 反向串行扫描；将这一步并行会破坏连续寿命定义。
- 上述长循环均已接入 `tqdm` 进度条，包括 STK 扫描、状态转换、图快照构建、寿命反向扫描和检查阶段。
- 第六步会把每个时隙、每条业务流的 K 条候选路径预计算到 `data/candidates/`；
  训练和评估环境会直接读取这些候选路径，避免在线反复调用 NetworkX K 最短路。

建图单个 worker 会占用较多内存；当前配置按 512 CPU 线程、720G 内存机器默认使用
128 个 worker。若 IO 抖动或内存压力过大，可先降到 64；配置为 0 时程序根据 CPU
自动选择，但仍最多使用 128 个。

## 候选路径预计算格式

每个时隙保存一个文件：

```text
data/candidates/train/cand_0000.npz
data/candidates/eval/cand_0000.npz
```

文件内部包含：

- `nodes`：所有候选路径节点拼接成的一维 `int64` 数组；
- `offsets`：形状为 `(num_flows, num_candidates + 1)`，用于切分每条 flow 的候选路径。

空路径用相同的起止 offset 表示。环境通过 `CandidateStore` 懒加载最近几个时隙，
不会一次性读入全部候选文件。

建图脚本逐时隙打印边数，并采用两遍处理：先生成所有图，再反向扫描回填链路
剩余寿命。这只在内存中保留当前时隙所需的数据，适合 6080 颗卫星的规模。

## 3. 输出文件

```text
data/
├── sat_state/
│   ├── sat_state_m.npy
│   ├── valid_mask.npy
│   ├── sat_names.json
│   └── time_index.csv
└── graphs/dmax_2000km/
    ├── graph_0000.npz
    ├── ...
    └── graph_0720.npz
```

- `sat_state_m.npy`：形状 `(T, N, 6)`，依次为位置 `x/y/z`（m）和速度
  `vx/vy/vz`（m/s）。无效行全部为 `NaN`。
- `valid_mask.npy`：形状 `(T, N)` 的布尔数组。
- `sat_names.json`：`sat_names` 顺序表和 `sat_to_id` 映射。
- `time_index.csv`：内部时隙 `k` 到原始 `TimeStep`、`Time` 的映射。
- `graphs/graph_XXXX.npz`：单个时隙的压缩图，包含 `edge_index` 和 `edge_attr`。

`edge_index` 的形状为 `(2, E)`，每条无向边只保存一次，且始终满足 `i < j`。
候选边由 `scipy.spatial.cKDTree.query_pairs` 在 2000 km 内检索，再去除穿过或
接触半径 6371 km 地球的线段。

6080 颗卫星的全连接候选边约有 1848 万条，每个时隙暴力遍历会造成不可接受的
时间开销。程序使用 `cKDTree.query_pairs` 只检索距离阈值内的候选边，并以
`edge_index` 保存稀疏图。它不会创建 `6080 × 6080` dense adjacency：密集矩阵
会浪费内存、丢失边特征的自然存储方式，也不适合 721 张动态图的逐时隙加载。

## 4. 边特征

`edge_attr` 形状为 `(E, 6)`，六列依次为：

| 索引 | 名称 | 单位与含义 |
|---:|---|---|
| 0 | `distance_m` | 两星欧氏距离，m |
| 1 | `propagation_delay_s` | 距离除以光速，s |
| 2 | `setup_delay_s` | 简化 PAT 建链时延，s |
| 3 | `residual_lifetime_s` | 从当前时隙起连续存在的时长，s |
| 4 | `capacity` | 第一版统一为 `1.0` |
| 5 | `angular_rate` | 相对速度模除以距离，近似角速度，1/s |

若一条边在当前及之后两个时隙连续存在、再下一时隙消失，剩余寿命为
`3 × 5 = 15 s`。`estimate_setup_delay()` 已独立封装，便于替换为更完整的
FOU/PAT 模型。

## 5. 在 RL 环境中使用

环境初始化时加载 `sat_names.json`、`time_index.csv`，状态数组建议通过
`np.load(..., mmap_mode="r")` 映射读取。环境在时隙 `k` 加载对应的
`graph_{k:04d}.npz`，由 `edge_index` 构建邻接关系，并把 `edge_attr` 用作路径
时延、建链代价、容量与稳定性特征。执行动作后将 `k` 加一，即可切换到底层
下一张动态图，无需一次加载全部 721 张图。

最小加载示例：

```python
import numpy as np

graph = np.load("data/graphs/dmax_2000km/graph_0000.npz")
edge_index = graph["edge_index"]
edge_attr = graph["edge_attr"]
```

检查脚本只对图做随机抽样，并报告 NaN、负值、越界、自环、非规范边和重复边；
发现普通数据问题时仅发出 warning，关键文件缺失时才会终止。
