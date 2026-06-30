"""Minimal single-environment MAPPO rollout/update/checkpoint loop."""

from __future__ import annotations

import csv
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .actor import Actor
from .critic import Critic
from .loss import compute_mappo_loss
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
        expected = (
            env.num_flows,
            env.num_candidates + 1,
            env.obs_dim,
            env.state_dim,
        )
        configured = tuple(int(mappo_config[key]) for key in (
            "num_agents", "num_actions", "obs_dim", "state_dim"
        ))
        if configured != expected:
            raise ValueError(f"MAPPO dimensions {configured} do not match environment {expected}")

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
        )
        self.policy = MAPPOPolicy(actor, critic, self.device)
        self._log_device_info()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(mappo_config["learning_rate"])
        )
        self.buffer = RolloutBuffer(
            int(mappo_config["rollout_length"]), self.num_envs, *configured[:3], configured[3],
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
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {"mappo": mappo_config, "env": env_config}, handle,
                indent=2, default=str,
            )
        self.obs = self.state = self.action_mask = None
        self.last_done = np.zeros(self.num_envs, dtype=bool)
        self.current_update = 0
        self._episode_reward = np.zeros(self.num_envs, dtype=np.float64)

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
            actions, log_probs, value, _entropy = self.policy.act(
                self.obs, self.state, self.action_mask
            )
            next_obs, next_state, next_mask, reward, done, info = self.env.step(actions)
            self.buffer.add(
                self.obs, self.state, self.action_mask, actions, log_probs,
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

        last_value = self.policy.get_value(self.state)
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
        loss_records: list[dict[str, float]] = []
        target_kl = self.config.get("target_kl")
        for _ in range(int(self.config["ppo_epochs"])):
            for batch in self.buffer.get_batches(int(self.config["minibatch_size"])):
                loss, info = compute_mappo_loss(
                    self.policy,
                    batch,
                    float(self.config["clip_ratio"]),
                    float(self.config["value_coef"]),
                    float(self.config["entropy_coef"]),
                    bool(self.config.get("normalize_advantages", True)),
                    self.config.get("value_clip_range"),
                )
                if not torch.isfinite(loss):
                    warnings.warn(
                        "Non-finite MAPPO loss; inspect rewards, advantages, log_probs, and values"
                    )
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), float(self.config["max_grad_norm"])
                )
                self.optimizer.step()
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
        result["explained_variance"] = explained_variance(
            self.buffer.values[: self.buffer.pos], self.buffer.returns[: self.buffer.pos]
        )
        return result

    def _write_metrics(self, metrics: dict) -> None:
        exists = self.metrics_path.is_file()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            if not exists:
                writer.writeheader()
            writer.writerow(metrics)

    @staticmethod
    def _format_progress(metrics: dict) -> dict[str, str]:
        """Select compact metrics for tqdm postfix display."""
        keys = (
            "mean_reward",
            "mean_future_mutex",
            "mean_outage_count",
            "actor_loss",
            "critic_loss",
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
            eval_interval = int(logging_cfg.get("eval_interval", 0))
            if eval_interval > 0 and update % eval_interval == 0:
                evaluation = self.evaluate(1)
                message = f"evaluation update={update:04d} reward={evaluation['total_reward']:.6f}"
                if hasattr(updates, "write"):
                    updates.write(message)
                else:
                    print(message)
                self.obs = self.state = self.action_mask = None
        self.save_checkpoint("latest.pt")
        print(f"Training finished. Run directory: {self.run_dir}")

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 1) -> dict[str, float]:
        if self.num_envs != 1:
            raise NotImplementedError(
                "MAPPOTrainer.evaluate expects a single env; use scripts/05_evaluate_mappo.py "
                "for checkpoints trained with vectorized rollout."
            )
        totals: list[dict[str, float]] = []
        for _ in range(int(num_episodes)):
            obs, state, mask = self.env.reset()
            aggregate = {
                "total_reward": 0.0, "avg_delay": 0.0, "peak_delay": 0.0,
                "future_mutex": 0.0, "outage_count": 0.0,
                "switch_count": 0.0, "new_link_count": 0.0,
            }
            steps = 0
            done = False
            while not done:
                actions, *_ = self.policy.act(obs, state, mask, deterministic=True)
                obs, state, mask, reward, done, info = self.env.step(actions)
                aggregate["total_reward"] += reward
                for key in aggregate:
                    if key == "peak_delay":
                        aggregate[key] = max(aggregate[key], float(info.get(key, 0.0)))
                    elif key != "total_reward":
                        aggregate[key] += float(info.get(key, 0.0))
                steps += 1
            aggregate["avg_delay"] /= max(steps, 1)
            totals.append(aggregate)
        return {key: float(np.mean([item[key] for item in totals])) for key in totals[0]}

    def save_checkpoint(self, name: str) -> Path:
        path = self.checkpoint_dir / name
        torch.save({
            "actor": self.policy.actor.state_dict(),
            "critic": self.policy.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "update": self.current_update,
        }, path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MAPPO checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # older torch
            checkpoint = torch.load(path, map_location=self.device)
        self.policy.actor.load_state_dict(checkpoint["actor"])
        self.policy.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_update = int(checkpoint.get("update", 0))
        return checkpoint

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
