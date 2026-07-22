"""Policy-agnostic episode evaluator."""

from __future__ import annotations

from .metrics import MetricsAccumulator


class Evaluator:
    """Run one env episode for any policy exposing `act(obs, state, action_mask)`."""

    def __init__(self, env, policy, max_steps: int | None = None):
        self.env = env
        self.policy = policy
        self.max_steps = None if max_steps is None else max(0, int(max_steps))
        # 大多数策略只需要 act 的数组输入；路径级规则策略可选择实现 bind_env，
        # 由评估器在 episode 开始前注入同一个环境实例。
        bind_env = getattr(policy, "bind_env", None)
        if callable(bind_env):
            bind_env(env)
        # 每次 run_episode 都会重新生成该列表；外部可在运行结束后将它导出，
        # 不改变原有 metrics 返回结构，避免路径数组混入结果 CSV。
        self.path_history: list[dict] = []

    def run_episode(self) -> dict:
        """Execute one episode and return accumulated metrics."""
        obs, state, action_mask = self.env.reset()
        metrics = MetricsAccumulator()
        self.path_history = []
        done = False
        steps = 0
        while not done:
            if self.max_steps is not None and steps >= self.max_steps:
                break
            actions = self.policy.act(obs, state, action_mask)
            obs, state, action_mask, reward, done, info = self.env.step(actions)
            # step 完成后 current_paths 才是本方法在时隙 info["k"] 实际采用的路径。
            # 必须逐层复制，防止后续时隙更新环境状态时改写已经记录的历史。
            self.path_history.append(
                {
                    "k": int(info["k"]),
                    "paths": [
                        None if path is None else [int(node) for node in path]
                        for path in self.env.current_paths
                    ],
                    # 环境在执行动作时已经按当前图计算好距离、建链代价和失败状态；
                    # 导出器直接消费该快照，避免评估结束后重新加载所有图文件。
                    "route_details": [dict(item) for item in info["route_details"]],
                }
            )
            metrics.update(reward, info)
            steps += 1
        return metrics.summary()
