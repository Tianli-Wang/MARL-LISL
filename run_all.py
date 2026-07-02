#!/usr/bin/env python3
"""MARL-LISL 全流程入口：按依赖顺序完成预处理、验证、训练与评估。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 使用脚本位置确定根目录，因此从任意工作目录调用也能正确定位配置和子脚本。
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    """描述一个可单独选择的流水线步骤。"""

    name: str
    title: str


# 此顺序同时表达产物依赖关系：下游只会在上游成功后启动。
STEPS = (
    Step("sat_state", "解析 STK 卫星状态"),
    Step("graphs", "构建动态图快照"),
    Step("check_data", "检查预处理数据"),
    Step("traffic", "生成 train/eval/stress 业务流"),
    Step("pack_graphs", "打包图快照"),
    Step("candidates", "生成三套离线候选路径"),
    Step("pack_candidates", "打包三套候选路径"),
    Step("env_test", "执行环境与 future-mutex 冒烟测试"),
    Step("proactive_rule", "执行主动规则策略诊断"),
    Step("train", "训练 MAPPO"),
    Step("evaluate_mappo", "专项评估 MAPPO checkpoint"),
    Step("evaluate_methods", "统一比较 baseline 与 MAPPO"),
)
STEP_NAMES = tuple(step.name for step in STEPS)


def project_path(path: Path) -> Path:
    """把相对路径统一解释为相对于项目根目录。"""
    return path if path.is_absolute() else ROOT / path


def add_optional(command: list[str], flag: str, value: object | None) -> None:
    """仅在有值时追加选项，避免覆盖子脚本自己的配置默认值。"""
    if value is not None:
        command.extend((flag, str(value)))


def build_parser() -> argparse.ArgumentParser:
    """创建全流程参数解析器。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preprocess-config", type=Path, default=Path("configs/preprocess.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--mappo-config", type=Path, default=Path("configs/mappo.yaml"))
    parser.add_argument("--start-at", choices=STEP_NAMES, default=STEP_NAMES[0], help="从指定步骤开始")
    parser.add_argument("--stop-after", choices=STEP_NAMES, default=STEP_NAMES[-1], help="在指定步骤后停止")
    parser.add_argument("--skip", nargs="*", choices=STEP_NAMES, default=(), help="跳过已有产物的步骤")
    parser.add_argument("--preprocess-workers", type=int, default=None, help="状态、图和 traffic 的并行数")
    parser.add_argument("--candidate-workers", type=int, default=None, help="候选路径生成进程数")
    parser.add_argument("--num-envs", type=int, default=None, help="临时覆盖训练环境数")
    parser.add_argument("--rollout-length", type=int, default=None, help="临时覆盖 rollout 长度")
    parser.add_argument("--total-updates", type=int, default=None, help="临时覆盖 PPO 更新轮数")
    parser.add_argument("--test-steps", type=int, default=5, help="环境测试步数")
    parser.add_argument("--rule-steps", type=int, default=20, help="规则诊断步数")
    parser.add_argument("--eval-episodes", type=int, default=1, help="专项评估 episode 数")
    parser.add_argument("--eval-workers", type=int, default=1, help="专项评估进程数")
    parser.add_argument("--max-eval-steps", type=int, default=None, help="评估最大步数")
    parser.add_argument("--checkpoint", type=Path, default=None, help="评估模型；默认自动选最新模型")
    parser.add_argument("--force-pack", action="store_true", help="强制重建已有 pack")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令")
    return parser


def selected_steps(args: argparse.Namespace) -> list[Step]:
    """按起止位置和跳过列表生成执行序列，并检查区间。"""
    start, stop = STEP_NAMES.index(args.start_at), STEP_NAMES.index(args.stop_after)
    if start > stop:
        raise SystemExit("--start-at 不能位于 --stop-after 之后。")
    skipped = set(args.skip)
    return [step for step in STEPS[start : stop + 1] if step.name not in skipped]


def build_command(name: str, args: argparse.Namespace, checkpoint: Path | None) -> list[str]:
    """构造单步参数列表；不用 shell 字符串可避免空格路径和转义问题。"""
    py = sys.executable
    pre = str(project_path(args.preprocess_config))
    env = str(project_path(args.env_config))
    mappo = str(project_path(args.mappo_config))
    if name == "sat_state":
        cmd = [py, "scripts/preprocess/01_build_sat_state.py", "--config", pre]
        add_optional(cmd, "--workers", args.preprocess_workers)
    elif name == "graphs":
        cmd = [py, "scripts/preprocess/02_build_graph_snapshots.py", "--config", pre]
        add_optional(cmd, "--workers", args.preprocess_workers)
    elif name == "check_data":
        cmd = [py, "scripts/preprocess/03_check_processed_data.py", "--config", pre]
    elif name == "traffic":
        cmd = [py, "scripts/preprocess/04_build_traffic.py", "--config", env, "--split", "all"]
        add_optional(cmd, "--workers", args.preprocess_workers)
    elif name == "pack_graphs":
        cmd = [py, "scripts/preprocess/06_pack_data.py", "--config", env, "--target", "graphs"]
    elif name == "candidates":
        cmd = [py, "scripts/preprocess/05_build_candidates.py", "--config", env, "--split", "all"]
        add_optional(cmd, "--workers", args.candidate_workers)
    elif name == "pack_candidates":
        cmd = [
            py, "scripts/preprocess/06_pack_data.py", "--config", env,
            "--target", "candidates", "--split", "all",
        ]
    elif name == "env_test":
        cmd = [
            py, "scripts/run/01_test_env.py", "--config", env, "--mode", "all",
            "--split", "train", "--steps", str(max(0, args.test_steps)),
        ]
    elif name == "proactive_rule":
        cmd = [
            py, "scripts/run/02_run_proactive_rule.py", "--config", env,
            "--steps", str(max(0, args.rule_steps)),
        ]
    elif name == "train":
        cmd = [
            py, "scripts/run/03_train_mappo.py", "--env-config", env,
            "--mappo-config", mappo,
        ]
        add_optional(cmd, "--num-envs", args.num_envs)
        add_optional(cmd, "--rollout-length", args.rollout_length)
        add_optional(cmd, "--total-updates", args.total_updates)
    elif checkpoint is None:
        raise RuntimeError(f"步骤 {name} 缺少可用的 MAPPO checkpoint。")
    elif name == "evaluate_mappo":
        cmd = [
            py, "scripts/run/04_evaluate_mappo.py", "--env-config", env,
            "--mappo-config", mappo, "--checkpoint", str(checkpoint),
            "--episodes", str(max(1, args.eval_episodes)),
            "--workers", str(max(1, args.eval_workers)),
        ]
        add_optional(cmd, "--max-steps", args.max_eval_steps)
    elif name == "evaluate_methods":
        cmd = [
            py, "scripts/run/05_evaluate_methods.py", "--env-config", env,
            "--mappo-config", mappo, "--checkpoint", str(checkpoint), "--methods", "all",
        ]
        add_optional(cmd, "--max-steps", args.max_eval_steps)
    else:
        raise ValueError(f"未知流水线步骤: {name}")
    # 本轮若刚生成图，旧 graph pack 必须强制刷新，否则打包函数会因 meta.json
    # 已存在而直接返回。候选路径生成器自身会强制刷新 candidate pack，无需重复。
    graph_pack_is_stale = name == "pack_graphs" and args.graphs_rebuilt
    if (args.force_pack and name in ("pack_graphs", "pack_candidates")) or graph_pack_is_stale:
        cmd.append("--force")
    return cmd


def newest_checkpoint(mappo_config: Path) -> Path | None:
    """从配置的 run_root 中返回修改时间最新的 latest.pt。"""
    # PyYAML 是项目运行依赖，但延迟导入可让缺依赖环境仍能查看 --help 和 dry-run。
    import yaml

    with project_path(mappo_config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    run_root = project_path(Path(config.get("output", {}).get("run_root", "outputs/runs")))
    candidates = list(run_root.glob("*/checkpoints/latest.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def display_command(command: list[str]) -> str:
    """生成可复制的命令文本，并为含空格的参数加引号。"""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def main() -> None:
    """串行执行所选流程，任一步失败即停止全部下游步骤。"""
    args = build_parser().parse_args()
    steps = selected_steps(args)
    if not steps:
        raise SystemExit("没有需要执行的步骤，请检查 --skip 参数。")
    # 该标记来自本次实际执行序列，而不是仅看 --start-at，因而与 --skip graphs
    # 配合时也不会错误地强制重打包已有图。
    args.graphs_rebuilt = any(step.name == "graphs" for step in steps)
    checkpoint = project_path(args.checkpoint) if args.checkpoint is not None else None
    if checkpoint is not None and not checkpoint.is_file() and not args.dry_run:
        raise SystemExit(f"指定的 checkpoint 不存在: {checkpoint}")

    started = time.perf_counter()
    print(f"项目根目录: {ROOT}")
    print("执行步骤: " + " -> ".join(step.name for step in steps))
    for index, step in enumerate(steps, 1):
        # 从评估阶段起跑时复用最新模型；训练结束后也会重新扫描本轮产物。
        if step.name.startswith("evaluate_") and checkpoint is None:
            checkpoint = (
                Path("<自动选择最新 checkpoint>")
                if args.dry_run
                else newest_checkpoint(args.mappo_config)
            )
            if checkpoint is None:
                raise SystemExit("未找到 latest.pt；请先训练或通过 --checkpoint 指定模型。")
        command = build_command(step.name, args, checkpoint)
        print(f"\n[{index}/{len(steps)}] {step.title}")
        print("命令: " + display_command(command), flush=True)
        if args.dry_run:
            continue

        step_started = time.perf_counter()
        try:
            # check=True 保证失败不会被吞掉；固定 cwd 保持子脚本的相对路径行为。
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"步骤 {step.name} 执行失败，退出码: {exc.returncode}") from exc
        except KeyboardInterrupt as exc:
            raise SystemExit("用户中断，全流程已停止。") from exc
        print(f"步骤完成，用时 {time.perf_counter() - step_started:.1f} 秒")

        # 没有显式指定模型时，让两个评估入口共同使用刚训练出的 checkpoint，
        # 避免它们各自的历史默认路径导致比较对象不一致。
        if step.name == "train" and args.checkpoint is None:
            checkpoint = newest_checkpoint(args.mappo_config)
            if checkpoint is None:
                raise SystemExit("训练结束，但未在 run_root 中找到 latest.pt。")
            print(f"本轮评估 checkpoint: {checkpoint}")
    print(f"\n全部所选流程执行成功，总用时 {time.perf_counter() - started:.1f} 秒。")


if __name__ == "__main__":
    main()
