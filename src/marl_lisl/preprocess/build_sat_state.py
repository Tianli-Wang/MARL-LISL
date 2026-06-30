"""Merge per-timeslot STK exports into fixed-index satellite state arrays."""

from __future__ import annotations

import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from marl_lisl.utils.progress import progress_iter


REQUIRED_COLUMNS = [
    "TimeStep", "Time", "SatName", "X_km", "Y_km", "Z_km",
    "Vx_km_s", "Vy_km_s", "Vz_km_s", "Valid",
]
STATE_COLUMNS = ["X_km", "Y_km", "Z_km", "Vx_km_s", "Vy_km_s", "Vz_km_s"]
TRUE_VALUES = {"true", "1", "1.0", "yes", "y", "valid", "t"}
FALSE_VALUES = {"false", "0", "0.0", "no", "n", "invalid", "f", "", "nan", "none"}
_SAT_TO_ID: dict[str, int] = {}


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", path.name))


def discover_stk_files(raw_dir: Path) -> list[Path]:
    """Return naturally sorted STK tables from one directory."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw STK directory not found: {raw_dir}")
    suffixes = {".txt", ".csv", ".tsv"}
    files = [path for path in raw_dir.iterdir()
             if path.is_file() and path.suffix.lower() in suffixes]
    files.sort(key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .txt/.csv/.tsv files found in {raw_dir}")
    return files


def _read_whitespace_table(path: Path) -> pd.DataFrame:
    """Read unquoted whitespace data while preserving spaces inside Time."""
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().strip().split()
        if header != REQUIRED_COLUMNS:
            raise ValueError(f"{path}: unrecognized whitespace header")
        for line_number, line in enumerate(handle, start=2):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < len(REQUIRED_COLUMNS):
                raise ValueError(f"{path}:{line_number}: too few fields")
            rows.append([fields[0], " ".join(fields[1:-8]), *fields[-8:]])
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def read_stk_table(path: Path) -> pd.DataFrame:
    """Read a comma-, tab-, or whitespace-delimited STK table robustly."""
    errors: list[Exception] = []
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
        frame.columns = [str(column).strip() for column in frame.columns]
        if all(column in frame.columns for column in REQUIRED_COLUMNS):
            return frame[REQUIRED_COLUMNS].copy()
    except Exception as exc:  # fall through to deterministic readers
        errors.append(exc)

    try:
        frame = _read_whitespace_table(path)
    except Exception as exc:
        errors.append(exc)
        raise ValueError(f"Cannot parse {path}; errors: {errors}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    return frame[REQUIRED_COLUMNS].copy()


def normalize_valid(value: object) -> bool:
    """Normalize common STK validity representations; unknown means False."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return False


def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["SatName"] = frame["SatName"].astype(str).str.strip()
    return frame.drop_duplicates(subset=["TimeStep", "SatName"], keep="last")


def _worker_count(num_workers: int | None) -> int:
    if num_workers is None or num_workers <= 0:
        return max(1, min(128, os.cpu_count() or 1))
    return max(1, int(num_workers))


def _scan_file(path: Path) -> tuple[list[str], object, object, bool]:
    """Read one file for the mapping/time-index pass (process-pool worker)."""
    frame = _deduplicate(read_stk_table(path))
    names = list(dict.fromkeys(frame["SatName"].tolist()))
    time_steps = frame["TimeStep"].dropna().unique()
    times = frame["Time"].dropna().unique()
    metadata_warning = len(time_steps) != 1 or len(times) != 1
    time_step = time_steps[-1] if len(time_steps) else ""
    time_value = times[-1] if len(times) else ""
    return names, time_step, time_value, metadata_warning


def _init_convert_worker(sat_to_id: dict[str, int]) -> None:
    global _SAT_TO_ID
    _SAT_TO_ID = sat_to_id


def _convert_file(task: tuple[int, Path]) -> tuple[int, np.ndarray, np.ndarray, int, int]:
    """Convert one STK file into valid IDs/states (process-pool worker)."""
    k, path = task
    frame = _deduplicate(read_stk_table(path))
    ids = frame["SatName"].map(_SAT_TO_ID).astype(np.int64).to_numpy()
    values = frame[STATE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid_text = frame["Valid"].astype(str).str.strip().str.lower()
    unknown_count = int((~valid_text.isin(TRUE_VALUES | FALSE_VALUES)).sum())
    valid = frame["Valid"].map(normalize_valid).to_numpy(dtype=bool)
    finite = np.isfinite(values).all(axis=1)
    invalid_state_count = int(np.sum(valid & ~finite))
    valid &= finite
    return k, ids[valid], values[valid] * 1000.0, unknown_count, invalid_state_count


def build_sat_state(
    raw_dir: Path,
    output_dir: Path,
    expected_num_steps: int = 721,
    expected_num_sats: int = 6080,
    num_workers: int | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Build state/mask files without constructing a global pandas DataFrame."""
    files = discover_stk_files(Path(raw_dir))
    if len(files) != expected_num_steps:
        warnings.warn(f"Expected {expected_num_steps} input files, found {len(files)}")

    workers = _worker_count(num_workers)
    print(f"parallel workers={workers}")

    # First parallel pass. executor.map preserves file order, so ID assignment is stable.
    sat_names: list[str] = []
    seen: set[str] = set()
    time_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        scan_results = executor.map(_scan_file, files, chunksize=1)
        scanned = progress_iter(
            zip(files, scan_results),
            total=len(files),
            desc="01 扫描 STK 文件",
            unit="file",
        )
        for k, (path, result) in enumerate(scanned):
            names, time_step, time_value, metadata_warning = result
            for name in names:
                if name not in seen:
                    seen.add(name)
                    sat_names.append(name)
            if metadata_warning:
                warnings.warn(f"{path.name}: expected exactly one TimeStep and Time")
            time_rows.append({"k": k, "TimeStep": time_step, "Time": time_value})

    sat_to_id = {name: index for index, name in enumerate(sat_names)}
    t_count, n_count = len(files), len(sat_names)
    if n_count != expected_num_sats:
        warnings.warn(f"Expected {expected_num_sats} satellites, found {n_count}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "sat_state_m.npy"
    mask_path = output_dir / "valid_mask.npy"
    sat_state_m = open_memmap(state_path, mode="w+", dtype=np.float64,
                              shape=(t_count, n_count, 6))
    valid_mask = open_memmap(mask_path, mode="w+", dtype=np.bool_,
                             shape=(t_count, n_count))
    sat_state_m[:] = np.nan
    valid_mask[:] = False

    unknown_valid_total = 0
    tasks = list(enumerate(files))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_convert_worker,
        initargs=(sat_to_id,),
    ) as executor:
        converted = progress_iter(
            executor.map(_convert_file, tasks, chunksize=1),
            total=len(tasks),
            desc="01 转换状态数组",
            unit="slot",
        )
        for k, ids, values, unknown_count, invalid_state_count in converted:
            unknown_valid_total += unknown_count
            if invalid_state_count:
                warnings.warn(
                    f"{files[k].name}: {invalid_state_count} valid rows with "
                    "non-finite state were invalidated"
            )
            sat_state_m[k, ids, :] = values
            valid_mask[k, ids] = True

    sat_state_m.flush()
    valid_mask.flush()
    if unknown_valid_total:
        warnings.warn(f"Treated {unknown_valid_total} unknown Valid values as False")
    with (output_dir / "sat_names.json").open("w", encoding="utf-8") as handle:
        json.dump({"sat_names": sat_names, "sat_to_id": sat_to_id}, handle,
                  ensure_ascii=False, indent=2)
    pd.DataFrame(time_rows, columns=["k", "TimeStep", "Time"]).to_csv(
        output_dir / "time_index.csv", index=False
    )

    counts = np.asarray(valid_mask.sum(axis=1))
    print(f"files={len(files)}")
    print(f"T={t_count}")
    print(f"N={n_count}")
    print(f"sat_state_m.shape={sat_state_m.shape}")
    print(f"valid_mask.shape={valid_mask.shape}")
    print(f"valid counts first 5: {counts[:5].tolist()}")
    print(f"valid counts last 5: {counts[-5:].tolist()}")
    return sat_state_m.shape, valid_mask.shape
