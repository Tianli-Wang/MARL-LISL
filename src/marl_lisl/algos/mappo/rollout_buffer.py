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
        obs_array = np.asarray(obs, np.float32)
        state_array = np.asarray(state, np.float32)
        mask_array = np.asarray(action_mask, np.float32)
        action_array = np.asarray(actions, np.int64)
        log_prob_array = np.asarray(log_probs, np.float32)
        reward_array = np.asarray(reward, np.float32).reshape(self.num_envs)
        done_array = np.asarray(done, np.float32).reshape(self.num_envs)
        value_array = np.asarray(value, np.float32).reshape(self.num_envs)

        if self.num_envs == 1 and obs_array.shape == self.obs.shape[2:]:
            self.obs[t, 0] = obs_array
            self.states[t, 0] = state_array
            self.action_masks[t, 0] = mask_array
            self.actions[t, 0] = action_array
            self.log_probs[t, 0] = log_prob_array
        else:
            self.obs[t] = obs_array
            self.states[t] = state_array
            self.action_masks[t] = mask_array
            self.actions[t] = action_array
            self.log_probs[t] = log_prob_array
        self.rewards[t] = reward_array
        self.dones[t] = done_array
        self.values[t] = value_array
        if info is None:
            self.infos.extend({} for _ in range(self.num_envs))
        elif isinstance(info, list):
            self.infos.extend(info)
        else:
            self.infos.append(info)
        self.pos += 1

    def compute_returns_and_advantages(self, last_value, last_done) -> None:
        if self.pos == 0:
            raise RuntimeError("Cannot compute GAE for an empty rollout")
        last_value = np.asarray(last_value, dtype=np.float32).reshape(self.num_envs)
        last_done = np.asarray(last_done, dtype=np.float32).reshape(self.num_envs)
        last_advantage = np.zeros(self.num_envs, dtype=np.float32)
        for t in range(self.pos - 1, -1, -1):
            if t == self.pos - 1:
                next_value = last_value
                next_nonterminal = 1.0 - last_done
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
