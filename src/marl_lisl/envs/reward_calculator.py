"""Scalar cooperative reward for multi-flow routing."""

from __future__ import annotations

import numpy as np


class RewardCalculator:
    def __init__(self, reward_weights: dict):
        self.weights = reward_weights

    def compute(
        self,
        delays: np.ndarray,
        switch_count: int,
        new_link_count: int,
        outage_count: int,
        future_mutex: float = 0.0,
    ) -> tuple[float, dict]:
        delays = np.asarray(delays, dtype=np.float64)
        avg_delay = float(delays.mean()) if delays.size else 0.0
        peak_delay = float(delays.max()) if delays.size else 0.0
        reward = -(
            float(self.weights["avg_delay"]) * avg_delay
            + float(self.weights["peak_delay"]) * peak_delay
            + float(self.weights["switch_count"]) * int(switch_count)
            + float(self.weights["new_link_count"]) * int(new_link_count)
            + float(self.weights["outage"]) * int(outage_count)
            + float(self.weights.get("mutex", 0.0)) * float(future_mutex)
        )
        info = {
            "avg_delay": avg_delay,
            "peak_delay": peak_delay,
            "switch_count": int(switch_count),
            "new_link_count": int(new_link_count),
            "outage_count": int(outage_count),
            "future_mutex": float(future_mutex),
            "reward": float(reward),
        }
        return float(reward), info
