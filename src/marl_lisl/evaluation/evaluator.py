"""Policy-agnostic episode evaluator."""

from __future__ import annotations

from .metrics import MetricsAccumulator


class Evaluator:
    """Run one env episode for any policy exposing `act(obs, state, action_mask)`."""

    def __init__(self, env, policy, max_steps: int | None = None):
        self.env = env
        self.policy = policy
        self.max_steps = None if max_steps is None else max(0, int(max_steps))

    def run_episode(self) -> dict:
        """Execute one episode and return accumulated metrics."""
        obs, state, action_mask = self.env.reset()
        metrics = MetricsAccumulator()
        done = False
        steps = 0
        while not done:
            if self.max_steps is not None and steps >= self.max_steps:
                break
            actions = self.policy.act(obs, state, action_mask)
            obs, state, action_mask, reward, done, info = self.env.step(actions)
            metrics.update(reward, info)
            steps += 1
        return metrics.summary()
