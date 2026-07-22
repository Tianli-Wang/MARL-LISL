"""把评估期间的逐流路径导出为 Satellate-2D-visualization 可读取的 CSV。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


VISUALIZATION_COLUMNS = [
    "SourceGS",
    "TargetGS",
    "Path",
    "Hops",
    "Total_ISL_Distance_km",
    "Setup_Penalty_ms",
    "Setup_Failures",
    "Link_Maintained",
]


def _filename_time(value: str) -> str:
    """把 time_index.csv 的 STK 时间转换为可视化页面要求的文件名时间段。"""
    # NumPy/STK 时间包含 9 位小数，而 Python datetime 最多解析 6 位；路径展示按
    # 5 秒采样，不依赖亚微秒精度，因此先丢弃小数再生成固定的 9 位零小数字段。
    base = value.strip().split(".", 1)[0]
    parsed = datetime.strptime(base, "%d %b %Y %H:%M:%S")
    return parsed.strftime("%d_%b_%Y_%H_%M_%S") + "_000000000"


def _load_time_by_k(time_index_path: str | Path) -> dict[int, str]:
    """读取时隙编号到 HTML 文件名时间字符串的映射。"""
    result: dict[int, str] = {}
    with Path(time_index_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["k"])] = _filename_time(row["Time"])
    return result


def write_visualization_paths(
    method: str,
    path_history: list[dict],
    output_root: str | Path,
    time_index_path: str | Path,
) -> Path:
    """按“方法/流/时隙.csv”导出路径，供 starLinkWebUI.html 选择文件夹播放。

    页面使用 ``STARLINK_1`` 开始的一基编号，而仿真内部节点 ID 从 0 开始，
    因此必须在这里统一加 1；若路径为 ``None``，仍写出空 Path 以明确记录 outage。
    """
    method_dir = Path(output_root) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    time_by_k = _load_time_by_k(time_index_path)

    for snapshot in path_history:
        k = int(snapshot["k"])
        if k not in time_by_k:
            raise KeyError(f"time_index.csv 中不存在时隙 k={k}")
        details = snapshot.get("route_details")
        if details is None:
            raise ValueError("路径历史缺少 route_details，无法生成完整可视化字段")
        for detail in details:
            flow_id = int(detail["flow_id"])
            path = detail["path"]
            flow_dir = method_dir / f"flow_{flow_id:03d}"
            flow_dir.mkdir(parents=True, exist_ok=True)
            filename = f"Calculated_step{k:04d}_{time_by_k[k]}.csv"
            # 空字符串会被原页面当成无效文件并沿用上一条路径；使用显式 OUTAGE
            # 标记后，页面会在对应时隙清空高亮路径，准确表现业务中断。
            route = "OUTAGE" if path is None else " -> ".join(
                f"STARLINK_{int(node) + 1}" for node in path
            )
            row = {
                "SourceGS": f"STARLINK_{int(detail['source']) + 1}",
                "TargetGS": f"STARLINK_{int(detail['target']) + 1}",
                "Path": route,
                "Hops": int(detail["hops"]),
                "Total_ISL_Distance_km": f"{float(detail['total_isl_distance_km']):.6f}",
                "Setup_Penalty_ms": f"{float(detail['setup_penalty_ms']):.6f}",
                "Setup_Failures": int(detail["setup_failures"]),
                "Link_Maintained": str(bool(detail["link_maintained"])),
            }
            # 使用 csv 模块严格按指定列顺序写出，避免手工拼接破坏字段转义。
            with (flow_dir / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=VISUALIZATION_COLUMNS)
                writer.writeheader()
                writer.writerow(row)

    print(f"saved visualization paths: {method_dir}")
    return method_dir
