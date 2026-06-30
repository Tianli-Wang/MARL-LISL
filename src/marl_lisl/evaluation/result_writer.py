"""Result table printing, CSV writing, and simple diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path


METRIC_COLUMNS = [
    "method",
    "total_reward",
    "avg_delay",
    "peak_delay",
    "future_mutex",
    "outage_count",
    "switch_count",
    "new_link_count",
    "invalid_action_count",
    "num_steps",
]


def print_results_table(results: list[dict]) -> None:
    """Print a compact comparison table."""
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
