"""Shared MAPPO utilities."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x, device: torch.device | str, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def explained_variance(y_pred, y_true) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    variance = np.var(y_true)
    return float("nan") if variance == 0 else float(1.0 - np.var(y_true - y_pred) / variance)


def get_activation(name: str) -> type[nn.Module]:
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}; choose {sorted(activations)}") from exc


def build_mlp(
    input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, activation: str
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    activation_cls = get_activation(activation)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.extend((nn.Linear(in_dim, hidden_dim), activation_cls()))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)
