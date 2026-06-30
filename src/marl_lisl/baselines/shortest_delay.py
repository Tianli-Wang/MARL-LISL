"""Shortest-delay baseline policy."""

from __future__ import annotations

import numpy as np


class ShortestDelayPolicy:
    """Pick the legal action with minimum propagation plus setup delay."""

    def act(self, obs, state, action_mask):  # noqa: D401 - uniform policy API
        obs = np.asarray(obs)
        action_mask = np.asarray(action_mask)
        actions = np.zeros(obs.shape[0], dtype=np.int64)
        for flow_id in range(obs.shape[0]):
            legal = np.flatnonzero(action_mask[flow_id] > 0)
            if len(legal):
                scores = obs[flow_id, legal, 0] + obs[flow_id, legal, 1]
                actions[flow_id] = int(legal[int(np.argmin(scores))])
        return actions
