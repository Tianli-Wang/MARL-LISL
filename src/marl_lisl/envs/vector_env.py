"""Small subprocess vector environment for CPU-heavy LISL rollouts."""

from __future__ import annotations

from copy import deepcopy
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Any

import numpy as np

from marl_lisl.envs.lisl_multi_flow_env import LISLMultiFlowEnv
from marl_lisl.store.packed_candidate_store import build_candidate_pack
from marl_lisl.store.packed_graph_store import build_graph_pack


def _env_worker(remote: Connection, env_config: dict, worker_id: int) -> None:
    """Own one environment instance and serve reset/step commands."""
    try:
        config = deepcopy(env_config)
        config["seed"] = int(config.get("seed", 0)) + 100_003 * int(worker_id)
        env = LISLMultiFlowEnv(config)
        while True:
            command, payload = remote.recv()
            if command == "reset":
                remote.send(env.reset())
            elif command == "step":
                obs, state, mask, reward, done, info = env.step(payload)
                if done:
                    obs, state, mask = env.reset()
                remote.send((obs, state, mask, reward, done, info))
            elif command == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown vector env command: {command}")
    except BaseException as exc:  # pragma: no cover - exercised through parent process
        try:
            remote.send(
                {
                    "__error__": True,
                    "worker_id": int(worker_id),
                    "message": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            remote.close()


class SubprocVectorEnv:
    """Parallelize fixed-shape LISL envs across subprocesses."""

    def __init__(
        self,
        env_config: dict,
        num_envs: int,
        start_method: str = "spawn",
    ):
        self.env_config = deepcopy(env_config)
        self.num_envs = max(1, int(num_envs))
        self.num_flows = int(env_config["num_flows"])
        self.num_candidates = int(env_config["num_candidates"])
        self.obs_dim = LISLMultiFlowEnv.obs_dim
        self.state_dim = LISLMultiFlowEnv.state_dim
        self._closed = False
        if (
            str(self.env_config.get("graph_backend", "lazy")).lower() == "packed"
            and bool(self.env_config.get("graph_pack_build_if_missing", True))
        ):
            build_graph_pack(
                self.env_config["graph_dir"],
                pack_dir=self.env_config.get("graph_pack_dir"),
            )
            self.env_config["graph_pack_build_if_missing"] = False
        candidates_cfg = dict(self.env_config.get("candidates", {}))
        if (
            bool(candidates_cfg.get("enabled", False))
            and str(candidates_cfg.get("backend", "npz")).lower() == "packed"
            and bool(candidates_cfg.get("pack_build_if_missing", True))
            and self.env_config.get("candidate_dir") is not None
        ):
            build_candidate_pack(self.env_config["candidate_dir"])
            candidates_cfg["pack_build_if_missing"] = False
            self.env_config["candidates"] = candidates_cfg
        context = mp.get_context(start_method)
        self._remotes: list[Connection] = []
        self._processes: list[mp.Process] = []
        for worker_id in range(self.num_envs):
            parent_remote, child_remote = context.Pipe()
            process = context.Process(
                target=_env_worker,
                args=(child_remote, self.env_config, worker_id),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._processes.append(process)

    @staticmethod
    def _check_message(message: Any) -> Any:
        if isinstance(message, dict) and message.get("__error__"):
            raise RuntimeError(
                f"Vector env worker {message['worker_id']} failed: "
                f"{message['message']}\n{message['traceback']}"
            )
        return message

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for remote in self._remotes:
            remote.send(("reset", None))
        results = [self._check_message(remote.recv()) for remote in self._remotes]
        obs, states, masks = zip(*results)
        return np.stack(obs), np.stack(states), np.stack(masks)

    def step(
        self, actions: np.ndarray | list[list[int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs, self.num_flows):
            raise ValueError(
                f"actions must have shape ({self.num_envs}, {self.num_flows}), "
                f"got {actions.shape}"
            )
        for remote, env_actions in zip(self._remotes, actions):
            remote.send(("step", env_actions))
        results = [self._check_message(remote.recv()) for remote in self._remotes]
        obs, states, masks, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(states),
            np.stack(masks),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for remote in self._remotes:
            try:
                remote.close()
            except OSError:
                pass

    def __del__(self):  # pragma: no cover - best-effort cleanup
        self.close()
