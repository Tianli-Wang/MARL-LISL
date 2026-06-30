"""Hand-crafted policy that prefers candidate paths avoiding future mutex conflicts."""

from __future__ import annotations

import numpy as np


class ProactiveRulePolicy:
    def __init__(
        self,
        min_b_avoid: float = 1.0,
        setup_cost_limit: float | None = None,
        prefer_low_delay_when_tie: bool = True,
    ):
        self.min_b_avoid = float(min_b_avoid)
        self.setup_cost_limit = None if setup_cost_limit is None else float(setup_cost_limit)
        self.prefer_low_delay_when_tie = bool(prefer_low_delay_when_tie)

    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        action_mask = np.asarray(action_mask)
        if obs.ndim != 3 or obs.shape[2] != 8:
            raise ValueError(f"obs must have shape (F, A, 8), got {obs.shape}")
        if action_mask.shape != obs.shape[:2]:
            raise ValueError("action_mask shape must match obs first two dimensions")
        actions = np.zeros(obs.shape[0], dtype=np.int64)
        for flow_id in range(obs.shape[0]):
            legal = np.flatnonzero(action_mask[flow_id] > 0)
            if not len(legal):
                continue
            default = 0 if 0 in legal else int(legal[0])
            candidates = legal[legal > 0]
            if self.setup_cost_limit is not None:
                candidates = candidates[
                    obs[flow_id, candidates, 1] <= self.setup_cost_limit
                ]
            candidates = candidates[
                obs[flow_id, candidates, 7] >= self.min_b_avoid
            ]
            if not len(candidates):
                actions[flow_id] = default
                continue
            avoid = obs[flow_id, candidates, 7]
            best = candidates[np.isclose(avoid, avoid.max())]
            if self.prefer_low_delay_when_tie and len(best) > 1:
                delay = obs[flow_id, best, 0] + obs[flow_id, best, 1]
                best = best[np.isclose(delay, delay.min())]
            actions[flow_id] = int(best[0])
        return actions
