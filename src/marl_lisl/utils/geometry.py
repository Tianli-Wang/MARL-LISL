"""Geometry primitives used by LISL preprocessing."""

from __future__ import annotations

import numpy as np


def segment_blocked_by_earth(
    p1: np.ndarray, p2: np.ndarray, r_earth: float = 6371e3
) -> bool:
    """Return True when the closed segment intersects or touches Earth."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    delta = p2 - p1
    length_sq = float(np.dot(delta, delta))
    if length_sq == 0.0:
        closest = p1
    else:
        t = float(np.clip(-np.dot(p1, delta) / length_sq, 0.0, 1.0))
        closest = p1 + t * delta
    return float(np.linalg.norm(closest)) <= r_earth


def visible_segment_mask(p1: np.ndarray, p2: np.ndarray, r_earth: float) -> np.ndarray:
    """Vectorized line-of-sight mask for equally shaped arrays of endpoints."""
    delta = p2 - p1
    length_sq = np.einsum("ij,ij->i", delta, delta)
    numerator = -np.einsum("ij,ij->i", p1, delta)
    t = np.divide(numerator, length_sq, out=np.zeros_like(numerator), where=length_sq > 0)
    np.clip(t, 0.0, 1.0, out=t)
    closest = p1 + t[:, None] * delta
    return np.linalg.norm(closest, axis=1) > r_earth
