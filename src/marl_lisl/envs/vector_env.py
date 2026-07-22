"""Small subprocess vector environment for CPU-heavy LISL rollouts."""

from __future__ import annotations

from copy import deepcopy
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Any, cast
import warnings

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
        # 这些清理字段必须在任何可能失败的配置复制、pack 构建或进程创建之前
        # 初始化。这样构造函数中途抛错后，Python 调用 __del__ 也不会用二次
        # AttributeError 覆盖真正的首个异常。
        self._closed = False
        self._remotes: list[Connection] = []
        self._processes: list[mp.Process] = []
        self.env_config = deepcopy(env_config)
        self.num_envs = max(1, int(num_envs))
        self.num_flows = int(env_config["num_flows"])
        self.num_candidates = int(env_config["num_candidates"])
        self.obs_dim = LISLMultiFlowEnv.obs_dim
        self.state_dim = LISLMultiFlowEnv.state_dim
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
        requested_method = str(start_method).strip().lower()
        available_methods = tuple(mp.get_all_start_methods())
        if requested_method in ("", "auto"):
            # Linux 优先 fork，以较低启动开销共享只读 memmap；Windows 没有
            # fork，只能选择 spawn。通过运行时能力判断而不是硬编码平台名，
            # 也能兼容 macOS 和受限 Python 构建。
            selected_method = "fork" if "fork" in available_methods else "spawn"
        elif requested_method in available_methods:
            selected_method = requested_method
        else:
            # 旧配置可能在 Windows 上仍写 fork。回退到当前解释器实际支持的
            # 方法并保留警告，避免因机器迁移直接终止训练。
            selected_method = (
                "spawn" if "spawn" in available_methods else available_methods[0]
            )
            warnings.warn(
                f"multiprocessing start method {requested_method!r} is unavailable; "
                f"falling back to {selected_method!r}. Available: {available_methods}",
                stacklevel=2,
            )
        self.start_method = selected_method
        print(f"SubprocVectorEnv start method: {self.start_method}")
        # typeshed 对不同 Python 版本的 multiprocessing context 暴露不完整；
        # 运行时对象始终提供 Pipe/Process，这里用 Any 表达该跨平台工厂接口。
        context: Any = mp.get_context(self.start_method)
        try:
            for worker_id in range(self.num_envs):
                parent_remote, child_remote = context.Pipe()
                process = context.Process(
                    target=_env_worker,
                    args=(child_remote, self.env_config, worker_id),
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    # 当前进程尚未加入成员列表，需要在这里单独关闭本轮 Pipe；
                    # 已成功启动的早期 worker 则交给统一 close() 回收。
                    parent_remote.close()
                    child_remote.close()
                    raise
                child_remote.close()
                # Windows 上 Pipe() 的静态返回名为 PipeConnection，实际完整实现
                # multiprocessing.connection.Connection 协议，统一转换后再保存。
                self._remotes.append(cast(Connection, parent_remote))
                self._processes.append(process)
        except BaseException:
            self.close()
            raise

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
        if getattr(self, "_closed", True):
            return
        self._closed = True
        remotes = list(getattr(self, "_remotes", ()))
        processes = list(getattr(self, "_processes", ()))
        for remote in remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for remote in remotes:
            try:
                remote.close()
            except OSError:
                pass

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except BaseException:
            # 析构函数不能再向外抛异常，否则会产生 "Exception ignored in
            # __del__" 噪声并掩盖构造或训练阶段真正需要定位的错误。
            pass
