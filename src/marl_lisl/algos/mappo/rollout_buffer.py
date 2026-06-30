"""Fixed-shape MAPPO rollout storage with global-reward GAE."""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        num_agents: int,
        num_actions: int,
        obs_dim: int,
        state_dim: int,
        gamma: float,
        gae_lambda: float,
    ):
        self.rollout_length = int(rollout_length)
        self.num_envs = int(num_envs)
        if self.num_envs != 1:
            raise NotImplementedError("Minimal MAPPO currently supports num_envs=1 only")
        self.num_agents = int(num_agents)
        self.num_actions = int(num_actions)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        shape = (self.rollout_length, self.num_envs)
        self.obs = np.zeros(shape + (self.num_agents, self.num_actions, self.obs_dim), np.float32)
        self.states = np.zeros(shape + (self.state_dim,), np.float32)
        self.action_masks = np.zeros(shape + (self.num_agents, self.num_actions), np.float32)
        self.actions = np.zeros(shape + (self.num_agents,), np.int64)
        self.log_probs = np.zeros(shape + (self.num_agents,), np.float32)
        self.rewards = np.zeros(shape, np.float32)
        self.dones = np.zeros(shape, np.float32)
        self.values = np.zeros(shape, np.float32)
        self.advantages = np.zeros(shape, np.float32)
        self.returns = np.zeros(shape, np.float32)
        self.infos: list[dict] = []
        self.pos = 0

    def add(
        self,
        obs,
        state,
        action_mask,
        actions,
        log_probs,
        reward,
        done,
        value,
        info=None,
    ) -> None:
        if self.pos >= self.rollout_length:
            raise RuntimeError("RolloutBuffer is full; reset it before adding")
        t = self.pos
        self.obs[t, 0] = np.asarray(obs, np.float32)
        self.states[t, 0] = np.asarray(state, np.float32)
        self.action_masks[t, 0] = np.asarray(action_mask, np.float32)
        self.actions[t, 0] = np.asarray(actions, np.int64)
        self.log_probs[t, 0] = np.asarray(log_probs, np.float32)
        self.rewards[t, 0] = float(reward)
        self.dones[t, 0] = float(done)
        self.values[t, 0] = float(value)
        self.infos.append({} if info is None else info)
        self.pos += 1

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        if self.pos == 0:
            raise RuntimeError("Cannot compute GAE for an empty rollout")
        last_advantage = np.zeros(self.num_envs, dtype=np.float32)
        for t in range(self.pos - 1, -1, -1):
            if t == self.pos - 1:
                next_value = np.full(self.num_envs, float(last_value), np.float32)
                next_nonterminal = 1.0 - float(last_done)
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_nonterminal
                - self.values[t]
            )
            last_advantage = (
                delta
                + self.gamma * self.gae_lambda * next_nonterminal * last_advantage
            )
            self.advantages[t] = last_advantage
        self.returns[: self.pos] = self.advantages[: self.pos] + self.values[: self.pos]

    def get_batches(self, minibatch_size: int, shuffle: bool = True):
        sample_count = self.pos * self.num_envs
        indices = np.arange(sample_count)
        if shuffle:
            np.random.shuffle(indices)
        flat = lambda array: array[: self.pos].reshape((sample_count,) + array.shape[2:])
        arrays = {
            "obs": flat(self.obs),
            "states": flat(self.states),
            "action_masks": flat(self.action_masks),
            "actions": flat(self.actions),
            "old_log_probs": flat(self.log_probs),
            "returns": flat(self.returns),
            "advantages": flat(self.advantages),
            "old_values": flat(self.values),
        }
        minibatch_size = max(1, int(minibatch_size))
        for start in range(0, sample_count, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            yield {key: value[batch_indices] for key, value in arrays.items()}

    def reset(self) -> None:
        self.pos = 0
        self.infos.clear()
