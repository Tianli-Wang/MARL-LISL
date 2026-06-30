"""Proactive future-mutex avoidance baseline policy."""

from __future__ import annotations

import numpy as np


class ProactiveRuleBaseline:
    """Switch only when a legal candidate provides enough positive B_avoid."""

    def __init__(
        self,
        min_b_avoid: float = 1.0,
        setup_cost_limit: float | None = None,
        prefer_low_delay_when_tie: bool = True,
    ):
        self.min_b_avoid = float(min_b_avoid)
        self.setup_cost_limit = setup_cost_limit
        self.prefer_low_delay_when_tie = bool(prefer_low_delay_when_tie)

    @classmethod
    def from_config(cls, config: dict) -> "ProactiveRuleBaseline":
        """Create the policy from env.yaml's proactive_rule section."""
        cfg = config.get("proactive_rule", {})
        return cls(
            cfg.get("min_b_avoid", 1.0),
            cfg.get("setup_cost_limit"),
            cfg.get("prefer_low_delay_when_tie", True),
        )

    def act(self, obs, state, action_mask):  # noqa: D401 - uniform policy API
        obs = np.asarray(obs)
        action_mask = np.asarray(action_mask)
        actions = np.zeros(obs.shape[0], dtype=np.int64)
        for flow_id in range(obs.shape[0]):
            legal = np.flatnonzero(action_mask[flow_id] > 0)
            if not len(legal):
                continue
            keep_legal = bool(action_mask[flow_id, 0] > 0)
            candidate_actions = legal[legal > 0]
            if len(candidate_actions):
                if self.setup_cost_limit is not None:
                    candidate_actions = candidate_actions[
                        obs[flow_id, candidate_actions, 1] <= float(self.setup_cost_limit)
                    ]
                if len(candidate_actions):
                    b_avoid = obs[flow_id, candidate_actions, 7]
                    best_b = float(np.max(b_avoid))
                    if best_b >= self.min_b_avoid:
                        tied = candidate_actions[np.isclose(b_avoid, best_b)]
                        if self.prefer_low_delay_when_tie and len(tied) > 1:
                            delay = obs[flow_id, tied, 0] + obs[flow_id, tied, 1]
                            actions[flow_id] = int(tied[int(np.argmin(delay))])
                        else:
                            actions[flow_id] = int(tied[0])
                        continue
            if keep_legal:
                actions[flow_id] = 0
            else:
                delay = obs[flow_id, legal, 0] + obs[flow_id, legal, 1]
                actions[flow_id] = int(legal[int(np.argmin(delay))])
        return actions
