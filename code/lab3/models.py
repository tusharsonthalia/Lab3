"""Architectures exactly as frozen in predictions.md Part A.1. No BatchNorm."""
from __future__ import annotations

import gzip
import io
import os
import tempfile

import torch
import torch.nn as nn


class MLP(nn.Module):
    """784 -> 256 -> 128 -> 10, ReLU, no dropout. Params: 235,146."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class CNN(nn.Module):
    """Conv(1->16,5x5,pad2)->ReLU->MaxPool2 -> Conv(16->32,5x5)->ReLU->MaxPool2
    -> FC(800->128)->ReLU -> FC(128->10). Params: 117,066."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class GCN(nn.Module):
    """GCNConv(1433->16)->ReLU->Dropout(0.5) -> GCNConv(16->7). Params: 23,063."""

    def __init__(self, in_channels: int = 1433, hidden: int = 16, out_channels: int = 7,
                 dropout: float = 0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)


def build_model(architecture: str) -> nn.Module:
    arch = architecture.lower()
    if arch == "mlp":
        return MLP()
    if arch == "cnn":
        return CNN()
    if arch == "gcn":
        return GCN()
    raise ValueError(f"unknown architecture: {architecture}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def raw_tensor_bytes(model: nn.Module) -> int:
    """Sum of numel*element_size over every state_dict tensor (payload only,
    no container overhead)."""
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def dense_bytes(model: nn.Module) -> int:
    """ACTUAL on-disk size of a `torch.save(state_dict)` file: the state dict is
    serialized to a temporary file (no metadata payload) and its size read with
    os.path.getsize. This is the container the bytes/compressed_bytes fields
    refer to (see results/SCHEMA.md)."""
    size, _ = _state_dict_file(model)
    return size


def gzip_bytes(model: nn.Module) -> int:
    """P1: gzip-9 size of the dense state_dict file (dense_bytes() container).
    Reported for EVERY run, dense included. This is NOT the compact-export
    artifact: that is `compressed_bytes` (Part B path 2)."""
    _, path = _state_dict_file(model)
    try:
        with open(path, "rb") as f_in, open(path + ".gz", "wb") as f_out:
            with gzip.GzipFile(fileobj=f_out, mode="wb", compresslevel=9) as gz:
                for chunk in iter(lambda: f_in.read(1 << 20), b""):
                    gz.write(chunk)
        return os.path.getsize(path + ".gz")
    finally:
        os.unlink(path)
        os.unlink(path + ".gz")


def _state_dict_file(model: nn.Module) -> tuple[int, str]:
    """torch.save the bare state_dict to a temp file; returns (size, path)."""
    fd, path = tempfile.mkstemp(prefix="lab3_state_", suffix=".pt")
    os.close(fd)
    torch.save({k: v.detach() for k, v in model.state_dict().items()}, path)
    return os.path.getsize(path), path
