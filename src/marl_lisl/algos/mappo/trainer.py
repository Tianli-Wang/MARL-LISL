"""Minimal single-environment MAPPO rollout/update/checkpoint loop."""

from __future__ import annotations

import csv
import json
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .actor import Actor
from .critic import Critic
from .loss import compute_mappo_loss
from .normalization import ValueNormalizer
from .policy import MAPPOPolicy
from .rollout_buffer import RolloutBuffer
from .utils import explained_variance, set_seed
from marl_lisl.utils.progress import progress_iter


class MAPPOTrainer:
    def __init__(self, env, mappo_config: dict, env_config: dict):
        self.env = env
        self.config = mappo_config
        self.env_config = env_config
        self.num_envs = int(mappo_config["num_envs"])
        actual_num_envs = int(getattr(env, "num_envs", 1))
        if actual_num_envs != self.num_envs:
            raise ValueError(
                f"MAPPO config num_envs={self.num_envs}, but environment exposes "
                f"num_envs={actual_num_envs}"
            )
        expected_actor = (
            env.num_flows,
            env.num_candidates + 1,
            env.obs_dim,
        )
        configured = tuple(int(mappo_config[key]) for key in (
            "num_agents", "num_actions", "obs_dim", "state_dim"
        ))
        if configured[:3] != expected_actor:
            raise ValueError(
                f"MAPPO Actor dimensions {configured[:3]} do not match "
                f"environment {expected_actor}"
            )
        env_state_dim = int(env.state_dim)
        legacy_state_dim = int(getattr(env, "legacy_state_dim", 7))
        if configured[3] not in {env_state_dim, legacy_state_dim}:
            raise ValueError(
                "MAPPO Critic state_dim 与环境不兼容: "
                f"配置为 {configured[3]}，环境扩展维度为 {env_state_dim}，"
                f"允许的旧版前缀维度为 {legacy_state_dim}"
            )
        self.critic_state_dim = configured[3]
        if self.critic_state_dim != env_state_dim:
            warnings.warn(
                "正在使用仅包含 7 个全局特征的历史 Critic；新环境 state 会安全"
                "截取 legacy 前缀。该兼容模式适合评估/续跑旧 checkpoint，"
                "新训练请使用扩展 state_dim。",
                stacklevel=2,
            )

        set_seed(int(mappo_config["seed"]))
        requested_device = str(mappo_config.get("device", "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        if "torch_num_threads" in mappo_config:
            torch.set_num_threads(max(1, int(mappo_config["torch_num_threads"])))
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        actor_cfg, critic_cfg = mappo_config["actor"], mappo_config["critic"]
        actor = Actor(
            configured[2], int(actor_cfg["hidden_dim"]), int(actor_cfg["num_layers"]),
            actor_cfg.get("activation", "relu"),
            bool(actor_cfg.get("normalize_input", False)),
            actor_cfg.get("heuristic_prior"),
        )
        critic = Critic(
            configured[3], int(critic_cfg["hidden_dim"]), int(critic_cfg["num_layers"]),
            critic_cfg.get("activation", "relu"),
            critic_cfg.get("state_normalization", "layer_norm"),
            float(critic_cfg.get("state_normalization_clip", 10.0)),
            float(critic_cfg.get("normalization_epsilon", 1e-8)),
        )
        value_normalizer = ValueNormalizer(
            bool(critic_cfg.get("value_normalization", False)),
            epsilon=float(critic_cfg.get("running_statistics_epsilon", 1e-4)),
            normalization_epsilon=float(
                critic_cfg.get("normalization_epsilon", 1e-8)
            ),
        )
        self.policy = MAPPOPolicy(
            actor, critic, self.device, value_normalizer=value_normalizer
        )
        self._log_device_info()
        # Actor 与 Critic 的目标、梯度尺度和收敛速度不同。拆分 Adam 后可分别
        # 调学习率/裁剪阈值，Critic 的大误差也不会再影响 Actor 的优化器状态。
        legacy_learning_rate = float(mappo_config.get("learning_rate", 2e-4))
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=float(mappo_config.get("actor_learning_rate", legacy_learning_rate)),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=float(mappo_config.get("critic_learning_rate", legacy_learning_rate)),
        )
        self.buffer = RolloutBuffer(
            # 逐项传参而不使用星号展开：既让类型检查器能够核对参数数量，也避免
            # 后续调整 configured 元组结构时静默地把错误维度传给缓冲区。
            int(mappo_config["rollout_length"]), self.num_envs,
            configured[0], configured[1], configured[2], configured[3],
            float(mappo_config["gamma"]), float(mappo_config["gae_lambda"]),
        )
        output_cfg = mappo_config["output"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_cfg["run_root"]) / f"{timestamp}_{output_cfg['experiment_name']}"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.metrics_dir / "train_metrics.csv"
        self.validation_metrics_path = self.metrics_dir / "validation_metrics.csv"
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {"mappo": mappo_config, "env": env_config}, handle,
                indent=2, default=str,
            )
        self.obs = self.state = self.action_mask = None
        self.last_done = np.zeros(self.num_envs, dtype=bool)
        self.current_update = 0
        self._episode_reward = np.zeros(self.num_envs, dtype=np.float64)
        # 验证环境与训练环境完全隔离：训练环境可以是多进程、随机起点和 train
        # traffic；验证环境固定使用单进程、确定性策略和指定 validation traffic。
        self._validation_env = None
        self.best_validation_score: tuple[float, ...] | None = None
        self.best_validation_update: int | None = None
        self.validation_bad_count = 0

    def _select_critic_state(self, state) -> np.ndarray:
        """把环境扩展 state 转换成当前 Critic/rollout buffer 使用的前缀。"""

        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 0 or state_array.shape[-1] < self.critic_state_dim:
            raise ValueError(
                "环境 state 维度不足: "
                f"Critic 需要 {self.critic_state_dim}，实际形状 {state_array.shape}"
            )
        return state_array[..., : self.critic_state_dim]

    def _log_device_info(self) -> None:
        """Print where MAPPO neural-network computation will run."""
        param_count = sum(parameter.numel() for parameter in self.policy.parameters())
        print(f"MAPPO policy device: {self.device}")
        print(f"MAPPO parameter count: {param_count}")
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(index)
            total_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
            print(f"CUDA device: cuda:{index} {name} ({total_gb:.1f} GB)")
            print(
                "Note: LISL env / NetworkX path search / future mutex are CPU-bound; "
                "GPU is used for actor-critic forward/backward only."
            )

    def collect_rollout(self) -> dict[str, float]:
        self.buffer.reset()
        if self.obs is None:
            self.obs, self.state, self.action_mask = self.env.reset()
        completed_episode_rewards: list[float] = []
        for _ in range(self.buffer.rollout_length):
            critic_state = self._select_critic_state(self.state)
            actions, log_probs, value, _entropy = self.policy.act(
                self.obs, critic_state, self.action_mask
            )
            next_obs, next_state, next_mask, reward, done, info = self.env.step(actions)
            self.buffer.add(
                self.obs, critic_state, self.action_mask, actions, log_probs,
                reward, done, value, info,
            )
            reward_array = np.asarray(reward, dtype=np.float64).reshape(self.num_envs)
            done_array = np.asarray(done, dtype=bool).reshape(self.num_envs)
            self._episode_reward += reward_array
            self.last_done = done_array
            for env_id in np.flatnonzero(done_array):
                completed_episode_rewards.append(float(self._episode_reward[env_id]))
                self._episode_reward[env_id] = 0.0
            if self.num_envs == 1 and bool(done_array[0]):
                self.obs, self.state, self.action_mask = self.env.reset()
            else:
                self.obs, self.state, self.action_mask = next_obs, next_state, next_mask

        # 状态运行统计只在完整 rollout 后更新一次，随后在 GAE 和所有 PPO
        # epoch 内冻结。更新后统一重算所有 old values，确保同一 rollout 的
        # value 使用完全相同的归一化坐标，而不是随采样步数漂移。
        rollout_states = self.buffer.states[: self.buffer.pos].reshape(
            -1, self.critic_state_dim
        )
        last_state = self._select_critic_state(self.state)
        last_state_batch = last_state.reshape(-1, self.critic_state_dim)
        self.policy.update_state_statistics(
            np.concatenate((rollout_states, last_state_batch), axis=0)
        )
        rollout_values = np.asarray(
            self.policy.get_value(rollout_states), dtype=np.float32
        ).reshape(self.buffer.pos, self.num_envs)
        self.buffer.values[: self.buffer.pos] = rollout_values
        last_value = self.policy.get_value(last_state)
        self.buffer.compute_returns_and_advantages(last_value, self.last_done)
        infos = self.buffer.infos
        mean = lambda key: float(np.mean([float(info.get(key, 0.0)) for info in infos]))
        return {
            "mean_reward": float(self.buffer.rewards[: self.buffer.pos].mean()),
            "mean_episode_reward": float(np.mean(completed_episode_rewards))
            if completed_episode_rewards else float(self._episode_reward.mean()),
            "mean_future_mutex": mean("future_mutex"),
            "mean_avg_delay": mean("avg_delay"),
            "mean_peak_delay": mean("peak_delay"),
            "mean_outage_count": mean("outage_count"),
            "mean_switch_count": mean("switch_count"),
            "mean_new_link_count": mean("new_link_count"),
        }

    def update(self) -> dict[str, float]:
        raw_returns = self.buffer.returns[: self.buffer.pos]
        if not np.all(np.isfinite(raw_returns)):
            raise RuntimeError("rollout returns 含 NaN/Inf，拒绝更新 ValueNorm 与网络")
        # ValueNorm 每个 rollout 只吸收一次 raw returns，之后多个 minibatch 和
        # PPO epoch 均复用冻结统计，避免训练目标在一次 update 内不断移动。
        self.policy.update_value_statistics(raw_returns.reshape(-1))
        explained_variance_before = explained_variance(
            self.buffer.values[: self.buffer.pos], raw_returns
        )
        loss_records: list[dict[str, float]] = []
        target_kl = self.config.get("target_kl")
        actor_max_grad_norm = float(
            self.config.get(
                "max_actor_grad_norm", self.config.get("max_grad_norm", 0.5)
            )
        )
        critic_max_grad_norm = float(
            self.config.get(
                "max_critic_grad_norm", self.config.get("max_grad_norm", 0.5)
            )
        )
        for _ in range(int(self.config["ppo_epochs"])):
            for batch in self.buffer.get_batches(int(self.config["minibatch_size"])):
                actor_objective, critic_objective, info = compute_mappo_loss(
                    self.policy,
                    batch,
                    float(self.config["clip_ratio"]),
                    float(self.config["value_coef"]),
                    float(self.config["entropy_coef"]),
                    bool(self.config.get("normalize_advantages", True)),
                    self.config.get("value_clip_range"),
                )
                if not (
                    bool(torch.isfinite(actor_objective).item())
                    and bool(torch.isfinite(critic_objective).item())
                ):
                    warnings.warn(
                        "MAPPO Actor/Critic loss 出现 NaN/Inf；本 minibatch 不更新",
                        stacklevel=2,
                    )
                    continue

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                # 两个网络不共享参数或计算图，因此分别反向传播、分别裁剪。
                actor_objective.backward()
                critic_objective.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), actor_max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), critic_max_grad_norm
                )
                actor_grad_norm_value = float(
                    torch.as_tensor(actor_grad_norm).detach().cpu()
                )
                critic_grad_norm_value = float(
                    torch.as_tensor(critic_grad_norm).detach().cpu()
                )
                if not (
                    np.isfinite(actor_grad_norm_value)
                    and np.isfinite(critic_grad_norm_value)
                ):
                    warnings.warn(
                        "MAPPO 梯度范数出现 NaN/Inf；本 minibatch 不更新",
                        stacklevel=2,
                    )
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    continue
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                info["actor_grad_norm"] = actor_grad_norm_value
                info["critic_grad_norm"] = critic_grad_norm_value
                loss_records.append(info)
                if target_kl is not None and info["approx_kl"] > float(target_kl):
                    break
            if (
                target_kl is not None
                and loss_records
                and loss_records[-1]["approx_kl"] > float(target_kl)
            ):
                break
        if not loss_records:
            raise RuntimeError("No finite PPO minibatch update was completed")
        result = {
            key: float(np.mean([record[key] for record in loss_records]))
            for key in loss_records[0]
        }
        # 同时报告 update 前后的 raw explained variance：
        # - explained_variance 保留历史口径（采样时预测），用于观察跨 rollout
        #   泛化是否真的改善；
        # - explained_variance_after_update 说明当前 batch 是否至少能被拟合。
        # 若只有 after 很高、下一轮 before 仍接近 0，通常表示 Critic 在过拟合。
        flat_states = self.buffer.states[: self.buffer.pos].reshape(
            -1, self.critic_state_dim
        )
        updated_values = np.asarray(
            self.policy.get_value(flat_states), dtype=np.float32
        ).reshape(self.buffer.pos, self.num_envs)
        explained_variance_after = explained_variance(
            updated_values, raw_returns
        )
        result["explained_variance_before_update"] = explained_variance_before
        result["explained_variance_after_update"] = explained_variance_after
        result["explained_variance"] = explained_variance_before
        result["state_normalizer_count"] = (
            self.policy.critic.state_statistics_count
        )
        value_normalizer = self.policy.value_normalizer
        if value_normalizer.enabled:
            result["value_normalizer_count"] = float(
                value_normalizer.running.count.detach().cpu()
            )
            result["value_normalizer_mean"] = float(
                value_normalizer.running.mean.detach().cpu()
            )
            result["value_normalizer_std"] = float(
                torch.sqrt(value_normalizer.running.var).detach().cpu()
            )
        else:
            result["value_normalizer_count"] = 0.0
            result["value_normalizer_mean"] = 0.0
            result["value_normalizer_std"] = 1.0
        return result

    def _write_metrics(self, metrics: dict) -> None:
        self._write_csv_row(self.metrics_path, metrics)

    @staticmethod
    def _write_csv_row(path: Path, row: dict) -> None:
        """向独立 CSV 追加一行，并在首次写入时生成表头。"""

        exists = path.is_file()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _build_validation_env(self):
        """根据训练配置派生固定起点、独立 traffic 的验证环境。

        此处不复用训练环境，是因为 vector env 会自动 reset，且训练配置允许
        ``train_random_start=true``。如果直接调用训练环境进行验证，不仅评估
        起点会随机，还会破坏下一轮 rollout 的状态连续性。
        """

        from marl_lisl.envs.lisl_multi_flow_env import LISLMultiFlowEnv

        validation_cfg = dict(self.config.get("validation", {}))
        split = str(validation_cfg.get("traffic_split", "eval")).strip().lower()
        traffic_key = f"traffic_{split}_path"
        candidate_key = f"{split}_dir"
        if traffic_key not in self.env_config:
            raise KeyError(
                f"验证 traffic split {split!r} 缺少环境配置字段 {traffic_key!r}"
            )

        config = deepcopy(self.env_config)
        traffic_path = Path(config[traffic_key])
        if not traffic_path.is_file():
            raise FileNotFoundError(f"验证 traffic 文件不存在: {traffic_path}")
        config["traffic_path"] = traffic_path
        validation_num_flows = int(np.load(traffic_path, mmap_mode="r").shape[0])
        if validation_num_flows != int(self.config["num_agents"]):
            raise ValueError(
                "验证 traffic 流数量与 MAPPO agent 数量不一致: "
                f"traffic={validation_num_flows}, agents={self.config['num_agents']}"
            )
        config["num_flows"] = validation_num_flows

        candidates_cfg = dict(config.get("candidates", {}))
        config["candidates"] = candidates_cfg
        if bool(candidates_cfg.get("enabled", False)):
            if candidate_key not in candidates_cfg:
                raise KeyError(
                    f"验证 split {split!r} 缺少候选路径目录 {candidate_key!r}"
                )
            config["candidate_dir"] = Path(candidates_cfg[candidate_key])

        config["env"] = dict(config["env"])
        config["env"]["train_random_start"] = False
        config["env"]["episode_start"] = int(
            validation_cfg.get("episode_start", 0)
        )
        config["env"]["episode_length"] = int(
            validation_cfg.get("episode_length", config["num_steps"])
        )
        # 验证 seed 固定且不受训练 worker 数量影响，确保同一 checkpoint 重复
        # 验证得到完全相同的轨迹和模型排名。
        config["seed"] = int(validation_cfg.get("seed", self.config["seed"]))
        return LISLMultiFlowEnv(config)

    @staticmethod
    def _validation_score(metrics: dict[str, float]) -> tuple[float, ...]:
        """把多目标验证结果转换为可按字典序最小化的严格优先级。

        路由首先不能 outage，其次必须消除 future mutex；只有这些安全指标相同
        时才比较总回报、平均/峰值时延、切换和新链路数。这样不会因为总 reward
        权重偶然偏小而选中带冲突的模型。
        """

        return (
            float(metrics["outage_count"]),
            float(metrics["future_mutex"]),
            -float(metrics["total_reward"]),
            float(metrics["avg_delay"]),
            float(metrics["peak_delay"]),
            float(metrics["switch_count"]),
            float(metrics["new_link_count"]),
        )

    def _run_validation(self) -> tuple[dict[str, float], bool]:
        """执行一次固定验证、写 CSV，并在改进时更新 best.pt。"""

        if self._validation_env is None:
            self._validation_env = self._build_validation_env()
        validation_cfg = dict(self.config.get("validation", {}))
        max_steps_value = validation_cfg.get("max_steps")
        max_steps = None if max_steps_value is None else int(max_steps_value)
        metrics = self._evaluate_env(
            self._validation_env,
            num_episodes=int(validation_cfg.get("num_episodes", 1)),
            max_steps=max_steps,
        )
        self._write_csv_row(
            self.validation_metrics_path,
            {"update": self.current_update, **metrics},
        )

        score = self._validation_score(metrics)
        improved = (
            self.best_validation_score is None
            or score < self.best_validation_score
        )
        if improved:
            self.best_validation_score = score
            self.best_validation_update = self.current_update
            self.validation_bad_count = 0
            self.save_checkpoint("best.pt")
        else:
            self.validation_bad_count += 1
        return metrics, improved

    @staticmethod
    def _format_progress(metrics: dict) -> dict[str, str]:
        """Select compact metrics for tqdm postfix display."""
        keys = (
            "mean_reward",
            "mean_future_mutex",
            "mean_outage_count",
            "actor_loss",
            "critic_loss",
            "explained_variance",
            "entropy",
            "rollout_time_s",
            "update_time_s",
        )
        labels = {
            "mean_reward": "reward",
            "mean_future_mutex": "mutex",
            "mean_outage_count": "outage",
            "actor_loss": "actor",
            "critic_loss": "critic",
            "explained_variance": "ev",
            "entropy": "ent",
            "rollout_time_s": "rollout_s",
            "update_time_s": "update_s",
        }
        return {
            labels[key]: f"{float(metrics[key]):.4g}"
            for key in keys
            if key in metrics
        }

    def train(self) -> None:
        logging_cfg = self.config["logging"]
        validation_cfg = dict(self.config.get("validation", {}))
        # 新配置使用独立 validation 节；旧配置仍可通过 logging.eval_interval
        # 启用验证，从而不破坏历史 YAML。
        validation_interval = int(
            validation_cfg.get(
                "interval", logging_cfg.get("eval_interval", 0)
            )
        )
        validation_enabled = bool(
            validation_cfg.get("enabled", validation_interval > 0)
        ) and validation_interval > 0
        early_stopping_patience = max(
            0, int(validation_cfg.get("early_stopping_patience", 0))
        )
        early_stopping_min_updates = max(
            0, int(validation_cfg.get("early_stopping_min_updates", 0))
        )
        total_updates = int(self.config["total_updates"])
        updates = progress_iter(
            range(1, total_updates + 1),
            total=total_updates,
            desc="MAPPO training",
            unit="update",
        )
        for update in updates:
            self.current_update = update
            rollout_start = time.perf_counter()
            rollout_metrics = self.collect_rollout()
            rollout_time = time.perf_counter() - rollout_start
            update_start = time.perf_counter()
            update_metrics = self.update()
            update_time = time.perf_counter() - update_start
            metrics = {
                "update": update,
                **rollout_metrics,
                **update_metrics,
                "rollout_time_s": rollout_time,
                "update_time_s": update_time,
            }
            self._write_metrics(metrics)
            if hasattr(updates, "set_postfix"):
                updates.set_postfix(self._format_progress(metrics))
            if update == 1 or update % int(logging_cfg["log_interval"]) == 0:
                message = (
                    f"update={update:04d} reward={metrics['mean_reward']:.6f} "
                    f"future_mutex={metrics['mean_future_mutex']:.6f} "
                    f"outage={metrics['mean_outage_count']:.3f} "
                    f"switch={metrics['mean_switch_count']:.3f} "
                    f"actor_loss={metrics['actor_loss']:.6f} "
                    f"critic_loss={metrics['critic_loss']:.6f} "
                    f"explained_variance={metrics['explained_variance']:.6f} "
                    f"ev_after={metrics['explained_variance_after_update']:.6f} "
                    f"critic_grad_norm={metrics['critic_grad_norm']:.6f} "
                    f"entropy={metrics['entropy']:.6f} "
                    f"rollout_s={metrics['rollout_time_s']:.2f} "
                    f"update_s={metrics['update_time_s']:.2f}"
                )
                if hasattr(updates, "write"):
                    updates.write(message)
                else:
                    print(message)
            if update % int(logging_cfg["save_interval"]) == 0:
                self.save_checkpoint(f"checkpoint_{update:06d}.pt")
                self.save_checkpoint("latest.pt")
            if validation_enabled and update % validation_interval == 0:
                evaluation, improved = self._run_validation()
                status = "BEST" if improved else (
                    f"no_improve={self.validation_bad_count}/"
                    f"{early_stopping_patience}"
                )
                message = (
                    f"validation update={update:04d} "
                    f"reward={evaluation['total_reward']:.6f} "
                    f"mutex={evaluation['future_mutex']:.6f} "
                    f"outage={evaluation['outage_count']:.3f} "
                    f"switch={evaluation['switch_count']:.0f} "
                    f"status={status}"
                )
                if hasattr(updates, "write"):
                    updates.write(message)
                else:
                    print(message)
                if (
                    early_stopping_patience > 0
                    and update >= early_stopping_min_updates
                    and self.validation_bad_count >= early_stopping_patience
                ):
                    stop_message = (
                        "early stopping: "
                        f"validation 连续 {self.validation_bad_count} 次未改善；"
                        f"best_update={self.best_validation_update}"
                    )
                    if hasattr(updates, "write"):
                        updates.write(stop_message)
                    else:
                        print(stop_message)
                    break
        self.save_checkpoint("latest.pt")
        print(
            f"Training finished. Run directory: {self.run_dir}; "
            f"best update: {self.best_validation_update}"
        )

    @torch.no_grad()
    def _evaluate_env(
        self,
        env,
        *,
        num_episodes: int = 1,
        max_steps: int | None = None,
    ) -> dict[str, float]:
        """在指定单环境上执行确定性策略，不修改训练 rollout 状态。"""

        totals: list[dict[str, float]] = []
        for _ in range(int(num_episodes)):
            obs, state, mask = env.reset()
            aggregate = {
                "total_reward": 0.0, "avg_delay": 0.0, "peak_delay": 0.0,
                "future_mutex": 0.0, "outage_count": 0.0,
                "switch_count": 0.0, "new_link_count": 0.0,
            }
            steps = 0
            done = False
            while not done:
                if max_steps is not None and steps >= max(0, int(max_steps)):
                    break
                actions, *_ = self.policy.act(
                    obs, state, mask, deterministic=True
                )
                obs, state, mask, reward, done, info = env.step(actions)
                aggregate["total_reward"] += float(reward)
                for key in aggregate:
                    if key == "peak_delay":
                        aggregate[key] = max(
                            aggregate[key], float(info.get(key, 0.0))
                        )
                    elif key != "total_reward":
                        aggregate[key] += float(info.get(key, 0.0))
                steps += 1
            aggregate["avg_delay"] /= max(steps, 1)
            totals.append(aggregate)
        return {
            key: float(np.mean([item[key] for item in totals]))
            for key in totals[0]
        }

    @torch.no_grad()
    def evaluate(
        self, num_episodes: int = 1, max_steps: int | None = None
    ) -> dict[str, float]:
        """确定性评估若干 episode，可用 max_steps 执行快速短轨迹验证。"""
        if self.num_envs != 1:
            raise NotImplementedError(
                "MAPPOTrainer.evaluate 仅支持单环境；请使用 "
                "scripts/run/05_evaluate_methods.py 对 checkpoint 进行统一评估。"
            )
        return self._evaluate_env(
            self.env, num_episodes=num_episodes, max_steps=max_steps
        )

    def save_checkpoint(self, name: str) -> Path:
        path = self.checkpoint_dir / name
        torch.save({
            # 版本 2 起保存两套优化器和独立 ValueNorm；Critic 内部的状态
            # RunningMeanStd 已包含在 critic.state_dict() 中。
            "checkpoint_version": 2,
            "actor": self.policy.actor.state_dict(),
            "critic": self.policy.critic.state_dict(),
            "value_normalizer": self.policy.value_normalizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "validation_state": {
                "best_score": self.best_validation_score,
                "best_update": self.best_validation_update,
                "bad_count": self.validation_bad_count,
            },
            "config": self.config,
            "update": self.current_update,
        }, path)
        return path

    def load_checkpoint(
        self, path: str | Path, *, load_optimizer: bool = True
    ) -> dict:
        """恢复模型权重；纯评估可跳过不需要的优化器状态。"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MAPPO checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # older torch
            checkpoint = torch.load(path, map_location=self.device)
        self.policy.actor.load_state_dict(checkpoint["actor"])
        self.policy.critic.load_state_dict(checkpoint["critic"])
        # load_optimizer=False 只跳过 Adam moments；评估仍必须恢复 ValueNorm，
        # 否则同一 Critic 网络输出会被反标准化到错误的 reward 尺度。
        if "value_normalizer" in checkpoint:
            self.policy.value_normalizer.load_state_dict(
                checkpoint["value_normalizer"]
            )
        elif self.policy.value_normalizer.enabled:
            warnings.warn(
                "checkpoint 未包含 ValueNorm 统计；已使用初始统计。"
                "这通常表示 checkpoint 与配置版本不匹配。",
                stacklevel=2,
            )
        if load_optimizer:
            if (
                "actor_optimizer" in checkpoint
                and "critic_optimizer" in checkpoint
            ):
                self.actor_optimizer.load_state_dict(
                    checkpoint["actor_optimizer"]
                )
                self.critic_optimizer.load_state_dict(
                    checkpoint["critic_optimizer"]
                )
            elif "optimizer" in checkpoint:
                # 旧单 Adam 把两类参数混在一个 param group 中，无法可靠拆成两套
                # optimizer state；保留网络权重并从新优化器状态继续最安全。
                warnings.warn(
                    "检测到旧版单优化器 checkpoint；Actor/Critic 权重已恢复，"
                    "但旧 optimizer moments 无法安全拆分，已从新优化器状态继续。",
                    stacklevel=2,
                )
        validation_state = checkpoint.get("validation_state")
        if isinstance(validation_state, dict):
            best_score = validation_state.get("best_score")
            self.best_validation_score = (
                None
                if best_score is None
                else tuple(float(value) for value in best_score)
            )
            best_update = validation_state.get("best_update")
            self.best_validation_update = (
                None if best_update is None else int(best_update)
            )
            self.validation_bad_count = int(
                validation_state.get("bad_count", 0)
            )
        self.current_update = int(checkpoint.get("update", 0))
        return checkpoint

    def close(self) -> None:
        validation_close = getattr(self._validation_env, "close", None)
        if callable(validation_close):
            validation_close()
        self._validation_env = None
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
