"""Policy-agnostic episode evaluator."""

from __future__ import annotations

import time

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
        # run_episode 结束后保存逐流汇总，05 脚本会将不同方法的结果合并成
        # method_compare_per_flow.csv；不把它嵌入 episode summary，避免 CSV
        # 单元格中出现嵌套列表。
        self.per_flow_metrics: list[dict] = []

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
            # 只计量策略从 observation 到 actions 的在线决策耗时。环境 step、
            # 图读取、future mutex 和 CSV 导出均不计入，从而公平比较 MAPPO
            # 推理与规则 baseline 的在线计算开销。
            decision_start = time.perf_counter()
            actions = self.policy.act(obs, state, action_mask)
            decision_time_s = time.perf_counter() - decision_start
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
            metrics.update(reward, info, decision_time_s=decision_time_s)
            steps += 1
        self.per_flow_metrics = metrics.per_flow_summary()
        return metrics.summary()
