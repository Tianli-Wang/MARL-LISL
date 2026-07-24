"""Result table printing, CSV writing, and simple diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path


METRIC_COLUMNS = [
    "method",
    "total_reward",
    "reward_per_step",
    "avg_delay",
    "delay_p95",
    "delay_p99",
    "peak_delay",
    "worst_flow_avg_delay",
    "future_mutex",
    "future_mutex_per_step",
    "raw_conflict_count",
    "future_mutex_positive_steps",
    "future_mutex_positive_rate",
    "first_conflict_slot",
    "invalid_future_path_count",
    "outage_count",
    "outage_rate",
    "switch_count",
    "switch_rate",
    "new_link_count",
    "new_link_rate",
    "maintain_ratio",
    "setup_failure_count",
    "setup_failure_rate",
    "avg_hops",
    "avg_path_distance_km",
    "mean_decision_time_ms",
    "p95_decision_time_ms",
    "invalid_action_count",
    "num_flows",
    "num_steps",
]

PER_FLOW_COLUMNS = [
    "method",
    "flow_id",
    "source",
    "target",
    "avg_delay",
    "delay_p95",
    "peak_delay",
    "outage_count",
    "outage_rate",
    "switch_count",
    "switch_rate",
    "new_link_count",
    "new_link_rate",
    "maintain_ratio",
    "setup_failure_count",
    "setup_failure_rate",
    "avg_hops",
    "avg_path_distance_km",
    "avg_setup_penalty_ms",
    "num_steps",
]


def print_results_table(results: list[dict]) -> None:
    """打印总量表和归一化/分布表，兼顾熟悉口径与跨场景可比性。"""

    header = (
        f"{'Method':<24} {'TotalReward':>14} {'AvgDelay':>10} {'PeakDelay':>10} "
        f"{'FutureMutex':>13} {'Outage':>8} {'Switch':>8} {'NewLinks':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['method']:<24} "
            f"{float(row['total_reward']):>14.6f} "
            f"{float(row['avg_delay']):>10.6f} "
            f"{float(row['peak_delay']):>10.6f} "
            f"{float(row['future_mutex']):>13.6f} "
            f"{float(row['outage_count']):>8.2f} "
            f"{float(row['switch_count']):>8.2f} "
            f"{float(row['new_link_count']):>9.2f}"
        )

    print()
    normalized_header = (
        f"{'Method':<24} {'P95Delay':>10} {'P99Delay':>10} {'WorstFlow':>10} "
        f"{'Mutex/Step':>11} {'MutexSteps':>10} {'OutRate':>9} "
        f"{'SwitchRate':>11} {'LinkRate':>9} {'Maintain':>9} {'DecisionMs':>11}"
    )
    print(normalized_header)
    print("-" * len(normalized_header))
    for row in results:
        print(
            f"{row['method']:<24} "
            f"{float(row.get('delay_p95', 0.0)):>10.6f} "
            f"{float(row.get('delay_p99', 0.0)):>10.6f} "
            f"{float(row.get('worst_flow_avg_delay', 0.0)):>10.6f} "
            f"{float(row.get('future_mutex_per_step', 0.0)):>11.6f} "
            f"{int(row.get('future_mutex_positive_steps', 0)):>10d} "
            f"{float(row.get('outage_rate', 0.0)):>9.6f} "
            f"{float(row.get('switch_rate', 0.0)):>11.6f} "
            f"{float(row.get('new_link_rate', 0.0)):>9.6f} "
            f"{float(row.get('maintain_ratio', 0.0)):>9.6f} "
            f"{float(row.get('mean_decision_time_ms', 0.0)):>11.4f}"
        )


def write_results_csv(results: list[dict], output_path: str | Path) -> Path:
    """Write comparison results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})
    print(f"saved results: {path}")
    return path


def write_per_flow_results_csv(
    results: list[dict], output_path: str | Path
) -> Path:
    """写出逐流指标，使整体均值无法掩盖单条流的时延或切换退化。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_FLOW_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {key: row.get(key, "") for key in PER_FLOW_COLUMNS}
            )
    print(f"saved per-flow results: {path}")
    return path


def print_diagnostics(results: list[dict]) -> None:
    """Print simple interpretation warnings/OK messages."""
    if not results:
        return
    if all(float(row.get("future_mutex", 0.0)) == 0.0 for row in results):
        print(
            "WARNING: No future mutex pressure detected. "
            "This scenario cannot validate proactive mutex avoidance."
        )
        print("Use stress traffic or increase num_flows / future_window.")
    by_name = {row["method"]: row for row in results}
    maintain = by_name.get("MaintainUntilConflict")
    proactive = by_name.get("ProactiveRule")
    mappo = by_name.get("MAPPO")
    if maintain and proactive:
        if float(proactive["future_mutex"]) < float(maintain["future_mutex"]):
            print("OK: Proactive rule reduces future mutex conflicts.")
    if mappo and proactive:
        mappo_mutex = float(mappo["future_mutex"])
        proactive_mutex = float(proactive["future_mutex"])
        mappo_switch = float(mappo["switch_count"])
        proactive_switch = float(proactive["switch_count"])
        mappo_new = float(mappo["new_link_count"])
        proactive_new = float(proactive["new_link_count"])
        if mappo_mutex <= proactive_mutex + 1e-6 and mappo_switch <= proactive_switch and mappo_new <= proactive_new:
            print("OK: MAPPO may learn a better trade-off.")
        if mappo_switch > 1.5 * max(proactive_switch, 1.0) or mappo_new > 1.5 * max(proactive_new, 1.0):
            print(
                "WARNING: MAPPO is over-switching. Consider increasing switch_count "
                "and new_link_count reward weights."
            )
