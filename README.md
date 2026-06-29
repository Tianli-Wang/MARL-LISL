# MARL-LISL

低轨巨星座星间激光通信（LISL）数据预处理项目。当前阶段只实现 STK 卫星
状态到 RL/MAPPO 底层稀疏动态图快照的转换，不包含 MAPPO 或完整 RL 环境。

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
```

前两个构建入口默认使用 `configs/preprocess.yaml` 中的 `parallel_workers: 4`，
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
