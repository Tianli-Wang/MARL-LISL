"""Episode metric accumulation."""

from __future__ import annotations

import warnings


class MetricsAccumulator:
    """Accumulate standard routing metrics over one episode."""

    def __init__(self):
        self.total_reward = 0.0
        self.avg_delay_sum = 0.0
        self.peak_delay = 0.0
        self.future_mutex = 0.0
        self.outage_count = 0.0
        self.switch_count = 0.0
        self.new_link_count = 0.0
        self.invalid_action_count = 0.0
        self.num_steps = 0
        self._warned_missing: set[str] = set()

    def _value(self, info: dict, key: str, default: float = 0.0) -> float:
        if key not in info and key not in self._warned_missing:
            warnings.warn(f"info missing '{key}', using {default}", stacklevel=2)
            self._warned_missing.add(key)
        return float(info.get(key, default))

    def update(self, reward: float, info: dict) -> None:
        """Add one env step result."""
        self.total_reward += float(reward)
        self.avg_delay_sum += self._value(info, "avg_delay")
        self.peak_delay = max(self.peak_delay, self._value(info, "peak_delay"))
        self.future_mutex += self._value(info, "future_mutex")
        self.outage_count += self._value(info, "outage_count")
        self.switch_count += self._value(info, "switch_count")
        self.new_link_count += self._value(info, "new_link_count")
        self.invalid_action_count += self._value(info, "invalid_action_count")
        self.num_steps += 1

    def summary(self) -> dict:
        """Return all standard metrics as a dict."""
        return {
            "total_reward": float(self.total_reward),
            "avg_delay": float(self.avg_delay_sum / max(self.num_steps, 1)),
            "peak_delay": float(self.peak_delay),
            "future_mutex": float(self.future_mutex),
            "outage_count": float(self.outage_count),
            "switch_count": float(self.switch_count),
            "new_link_count": float(self.new_link_count),
            "invalid_action_count": float(self.invalid_action_count),
            "num_steps": int(self.num_steps),
        }
