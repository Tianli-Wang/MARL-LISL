"""Greedy future-mutex-aware baseline policy."""

from __future__ import annotations

import numpy as np


class GreedyConflictAwarePolicy:
    """Pick the legal action with minimum A_mutex, tie-breaking by delay."""

    def act(self, obs, state, action_mask):  # noqa: D401 - uniform policy API
        obs = np.asarray(obs)
        action_mask = np.asarray(action_mask)
        actions = np.zeros(obs.shape[0], dtype=np.int64)
        for flow_id in range(obs.shape[0]):
            legal = np.flatnonzero(action_mask[flow_id] > 0)
            if not len(legal):
                continue
            a_mutex = obs[flow_id, legal, 6]
            delay = obs[flow_id, legal, 0] + obs[flow_id, legal, 1]
            order = np.lexsort((delay, a_mutex))
            actions[flow_id] = int(legal[int(order[0])])
        return actions
