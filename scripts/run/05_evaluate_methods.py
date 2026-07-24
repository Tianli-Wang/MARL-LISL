#!/usr/bin/env python3
"""功能：统一评估 baseline、互斥诊断策略和可选的 MAPPO checkpoint。"""

from __future__ import annotations

import argparse
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def find_latest_checkpoint(run_root: Path) -> Path | None:
    """选择最新训练 run，并优先返回该 run 的 ``best.pt``。

    每次运行 MAPPO 都会在 ``run_root`` 下创建一个独立实验目录，
    因此先根据各 run 的 ``latest.pt`` 修改时间确定最新实验。选中
    run 后若固定验证产生了 ``best.pt``，则优先评估 best；旧实验
    没有 best 时才回退到 latest。不能直接把 best/latest 混在一起
    按时间排序，因为训练结束时 latest 通常比 best 更新，反而会再次
    选中已发生后期退化的参数。

    如果尚未产生任何 checkpoint，返回 ``None``，交由主程序根据
    ``--methods`` 的取值决定是报错，还是仅评估 baseline。
    """
    candidates = list(run_root.glob("*/checkpoints/latest.pt"))
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    best = latest.with_name("best.pt")
    return best if best.is_file() else latest


# 默认数据选择当前仓库已生成的文件；checkpoint 则在脚本
# 启动时动态扫描，避免每次训练后手动修改时间戳目录。命令行
# ``--checkpoint`` 仍可显式指定任意历史模型，以便复现旧实验。
DEFAULT_ENV_CONFIG = ROOT / "configs/env.yaml"
DEFAULT_MAPPO_CONFIG = ROOT / "configs/mappo.yaml"
DEFAULT_CHECKPOINT = find_latest_checkpoint(ROOT / "outputs/runs")
DEFAULT_TRAFFIC = ROOT / "data/traffic/traffic_pairs_eval.npy"
DEFAULT_OUTPUT = ROOT / "outputs/tables/method_compare.csv"

from marl_lisl.baselines.registry import build_baseline_policies
from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.evaluation import Evaluator
from marl_lisl.evaluation.result_writer import (
    print_diagnostics,
    print_results_table,
    write_per_flow_results_csv,
    write_results_csv,
)
from marl_lisl.evaluation.path_writer import write_visualization_paths
from marl_lisl.utils.runtime_config import (
    load_checkpoint_mappo_config,
    load_runtime_env_config,
    load_runtime_mappo_config,
    resolve_project_path,
)


class MAPPODeterministicAdapter:
    """把 MAPPO 的确定性动作接口适配为通用 Evaluator 所需的 ``act``。"""

    def __init__(self, trainer):
        self.trainer = trainer

    def act(self, obs, state, action_mask):
        """使用 masked logits 的 argmax 返回每个 agent 的确定性动作。"""
        actions, *_ = self.trainer.policy.act(
            obs, state, action_mask, deterministic=True
        )
        return actions


def _policies(mode: str, config: dict) -> list[tuple[str, object]]:
    """按评估模式从 baseline 注册表构造策略，主脚本不依赖具体策略类。"""
    if mode == "diagnose":
        # 哪些方法属于诊断集合由 baseline 注册信息决定。后续调整方法时，
        # 这里不需要同步维护另一份名称和构造器列表。
        return build_baseline_policies(config, diagnostic_only=True)
    if mode in ("baselines", "all"):
        return build_baseline_policies(config)
    return []


def write_results_markdown(results: list[dict], output_path: Path) -> Path:
    """把同一次评估的 baseline 与 MAPPO 指标写成便于阅读的 Markdown 表格。

    CSV 适合后续由 Excel、Pandas 或绘图脚本处理，但人工查看多个 baseline
    的权衡时，需要反复对应列名。此报告沿用控制台的两张表：第一张放总回报、
    时延、互斥与切换等核心量；第二张放分位时延、归一化率和决策耗时。这样
    既不会把单位不同的量混在同一张超宽表中，也能直接粘贴到实验记录或论文草稿。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 方法名当前均为项目内部固定名称；仍替换竖线，避免未来自定义名称破坏
    # Markdown 表格的列分隔符。
    def _method_name(row: dict) -> str:
        return str(row.get("method", "Unknown")).replace("|", "\\|")

    # 所有数值统一在写报告时格式化，而不是修改原始 results。这样 CSV 保留
    # 完整精度，Markdown 则保持紧凑、适合人工横向比较。
    def _number(row: dict, key: str, digits: int = 6) -> str:
        return f"{float(row.get(key, 0.0)):.{digits}f}"

    lines = [
        "# Baseline 与 MAPPO 统一评估对比",
        "",
        "同一 traffic、同一 episode 下的策略对比；结果按评估执行顺序列出。",
        "",
        "## 核心指标",
        "",
        "| 方法 | Total reward | Avg delay | Peak delay | Future mutex | Outage | Switch | New links |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            f"{_method_name(row)} | {_number(row, 'total_reward')} | "
            f"{_number(row, 'avg_delay')} | {_number(row, 'peak_delay')} | "
            f"{_number(row, 'future_mutex')} | {_number(row, 'outage_count', 2)} | "
            f"{_number(row, 'switch_count', 2)} | {_number(row, 'new_link_count', 2)} |"
        )

    lines.extend(
        [
            "",
            "## 时延分布与稳定性指标",
            "",
            "| 方法 | P95 delay | P99 delay | Worst-flow delay | Mutex/step | Mutex steps | Outage rate | Switch rate | Link rate | Maintain | Decision ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        lines.append(
            "| "
            f"{_method_name(row)} | {_number(row, 'delay_p95')} | "
            f"{_number(row, 'delay_p99')} | {_number(row, 'worst_flow_avg_delay')} | "
            f"{_number(row, 'future_mutex_per_step')} | "
            f"{int(row.get('future_mutex_positive_steps', 0))} | "
            f"{_number(row, 'outage_rate')} | {_number(row, 'switch_rate')} | "
            f"{_number(row, 'new_link_rate')} | {_number(row, 'maintain_ratio')} | "
            f"{_number(row, 'mean_decision_time_ms', 4)} |"
        )

    if results and all(float(row.get("future_mutex", 0.0)) == 0.0 for row in results):
        lines.extend(
            [
                "",
                "> 注：所有方法的 future mutex 均为 0；该场景不能区分主动规避未来互斥的能力，应同时查看 stress traffic 结果。",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved Markdown comparison: {output_path}")
    return output_path


def main() -> None:
    """运行所选方法，打印同口径诊断表并把结果保存为 CSV。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-config", type=Path, default=DEFAULT_ENV_CONFIG,
        help=f"环境配置，默认：{DEFAULT_ENV_CONFIG}",
    )
    parser.add_argument(
        "--mappo-config", type=Path, default=DEFAULT_MAPPO_CONFIG,
        help=f"MAPPO 配置，默认：{DEFAULT_MAPPO_CONFIG}",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=(
            "MAPPO checkpoint；默认选择最新 run 的 best.pt，"
            "若该 run 没有 best.pt 则回退到 latest.pt"
        ),
    )
    parser.add_argument(
        "--traffic", type=Path, default=DEFAULT_TRAFFIC,
        help=f"评估 traffic，默认：{DEFAULT_TRAFFIC}",
    )
    parser.add_argument(
        "--methods",
        choices=("baselines", "diagnose", "mappo", "all"),
        default="all",
        help="评估方法集合，默认：baselines",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="每种方法最多运行多少步，默认：完整 episode",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=(
            "汇总结果 CSV；同目录还会自动生成 *_per_flow.csv，"
            f"默认：{DEFAULT_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=None,
        help=(
            "可读的 Markdown 对比报告路径；默认与 --output 同目录同名，"
            "仅将扩展名替换为 .md"
        ),
    )
    parser.add_argument(
        "--paths-output", type=Path, default="outputs/visualization_paths",
        help=("可选的逐时隙路径导出根目录；输出可由 "
              "Satellate-2D-visualization/starLinkWebUI.html 直接读取"),
    )
    parser.add_argument(
        "--preload-graphs", action="store_true", default=False,
        help="评估前预加载全部图，默认关闭；开启后速度更快但占内存",
    )
    args = parser.parse_args()

    env_config = load_runtime_env_config(
        args.env_config,
        ROOT,
        traffic_path=args.traffic,
        preload_graphs=args.preload_graphs,
        train_random_start=False,
    )
    output_path = resolve_project_path(ROOT, args.output)
    paths_output = (
        None if args.paths_output is None
        else resolve_project_path(ROOT, args.paths_output)
    )
    time_index_path = ROOT / "data/sat_state/time_index.csv"
    results: list[dict] = []
    per_flow_results: list[dict] = []
    for name, policy in _policies(args.methods, env_config):
        env = LISLMultiFlowEnv(env_config)
        evaluator = Evaluator(env, policy, args.max_steps)
        episode_result = evaluator.run_episode()
        results.append({"method": name, **episode_result})
        per_flow_results.extend(
            {"method": name, **row}
            for row in evaluator.per_flow_metrics
        )
        if paths_output is not None:
            write_visualization_paths(
                name, evaluator.path_history, paths_output, time_index_path
            )

    if args.methods in ("mappo", "all"):
        if args.checkpoint is None:
            if args.methods == "mappo":
                parser.error(
                    "未在 outputs/runs 中找到 latest.pt，"
                    "请先训练或通过 --checkpoint 显式指定模型"
                )
            warnings.warn(
                "未在 outputs/runs 中找到 latest.pt，all 模式只评估 baseline。",
                stacklevel=2,
            )
        else:
            checkpoint = resolve_project_path(ROOT, args.checkpoint)
            if not checkpoint.is_file():
                if args.methods == "mappo":
                    raise FileNotFoundError(f"MAPPO checkpoint 不存在: {checkpoint}")
                warnings.warn(f"checkpoint 不存在，跳过 MAPPO: {checkpoint}", stacklevel=2)
            else:
                # 输出本次自动或手动选中的模型，使评估日志能够明确
                # 追溯到具体训练轮次，避免多个 run 并存时混淆结果。
                print(f"MAPPO checkpoint: {checkpoint}")
                # baseline-only 模式不会导入 PyTorch；只有确实评估 MAPPO 时才
                # 加载训练器，减少启动时间并避免无关 CUDA/NumPy 环境错误。
                from marl_lisl.algos.mappo import MAPPOTrainer

                yaml_config = load_runtime_mappo_config(args.mappo_config, ROOT)
                mappo_config = load_checkpoint_mappo_config(
                    checkpoint, yaml_config, num_envs=1
                )
                # MAPPOTrainer 初始化时会创建运行目录；统一方法评估只需要最终
                # CSV，因此把中间 trainer 目录放进临时目录，避免污染正式 runs。
                with TemporaryDirectory(prefix="marl_lisl_method_eval_") as temp_dir:
                    eval_mappo = deepcopy(mappo_config)
                    eval_mappo["output"] = dict(eval_mappo["output"])
                    eval_mappo["output"]["run_root"] = Path(temp_dir)
                    eval_mappo["output"]["experiment_name"] = "method_eval"
                    env = LISLMultiFlowEnv(env_config)
                    trainer = MAPPOTrainer(env, eval_mappo, env_config)
                    try:
                        trainer.load_checkpoint(checkpoint, load_optimizer=False)
                        policy = MAPPODeterministicAdapter(trainer)
                        evaluator = Evaluator(env, policy, args.max_steps)
                        episode_result = evaluator.run_episode()
                        results.append(
                            {
                                "method": "MAPPO",
                                **episode_result,
                            }
                        )
                        per_flow_results.extend(
                            {"method": "MAPPO", **row}
                            for row in evaluator.per_flow_metrics
                        )
                        if paths_output is not None:
                            write_visualization_paths(
                                "MAPPO", evaluator.path_history,
                                paths_output, time_index_path,
                            )
                    finally:
                        trainer.close()

    if not results:
        raise RuntimeError("没有可评估的方法，请检查 --methods 和 --checkpoint。")
    print_results_table(results)
    print_diagnostics(results)
    write_results_csv(results, output_path)
    # 未显式指定时，让 CSV 与 Markdown 使用同一个主文件名。例如
    # method_compare.csv 会自动配套生成 method_compare.md，便于一次评估
    # 同时满足机器读取和人工审阅两种需求。
    markdown_path = (
        resolve_project_path(ROOT, args.markdown_output)
        if args.markdown_output is not None
        else output_path.with_suffix(".md")
    )
    write_results_markdown(results, markdown_path)
    # 逐流文件与汇总文件使用相同目录和主文件名，调用者只需指定一次
    # --output。例如 method_compare.csv 会同时生成
    # method_compare_per_flow.csv。
    per_flow_suffix = output_path.suffix or ".csv"
    per_flow_path = output_path.with_name(
        f"{output_path.stem}_per_flow{per_flow_suffix}"
    )
    write_per_flow_results_csv(per_flow_results, per_flow_path)


if __name__ == "__main__":
    main()
