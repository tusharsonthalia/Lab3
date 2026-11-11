"""Test (a): tiny-overfit per architecture. Tests may use their own lr/epochs
but must not edit configs/. We train in-process with an explicit cfg dict."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lab3.models import build_model
from lab3.util import set_seed, set_threads, load_indices, verify_splits
from test_splits import _make_splits_once


def _cfg(arch: str) -> dict:
    # Deliberately NOT the frozen config: tiny-overfit tests use their own lr/epochs.
    if arch == "gcn":
        return {"lr": 0.01, "weight_decay": 5e-4, "batch_size": None, "max_epochs": 300,
                "patience": 50, "hidden": 16, "dropout": 0.5}
    return {"lr": 1e-2, "weight_decay": 0.0, "batch_size": 64, "max_epochs": 30, "patience": 30}


def test_mlp_overfit():
    set_threads()
    _make_splits_once()
    set_seed(0)
    from lab3.data import fmnist_split_tensors
    tr_x, tr_y, va_x, va_y, te_x, te_y, _, _ = fmnist_split_tensors()
    # 64 training images (frozen test spec)
    xb, yb = tr_x[:64], tr_y[:64]
    model = build_model("mlp")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(30):
        opt.zero_grad()
        out = model(xb)
        loss = nn.functional.cross_entropy(out, yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        acc = (model(xb).argmax(1) == yb).float().mean().item()
    assert acc >= 0.99, f"MLP tiny-overfit failed: acc={acc}"


def test_cnn_overfit():
    set_threads()
    _make_splits_once()
    set_seed(0)
    from lab3.data import fmnist_split_tensors
    tr_x, tr_y, va_x, va_y, te_x, te_y, _, _ = fmnist_split_tensors()
    xb, yb = tr_x[:64], tr_y[:64]
    model = build_model("cnn")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        out = model(xb)
        loss = nn.functional.cross_entropy(out, yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        acc = (model(xb).argmax(1) == yb).float().mean().item()
    assert acc >= 0.99, f"CNN tiny-overfit failed: acc={acc}"


def test_gcn_overfit():
    set_threads()
    _make_splits_once()
    set_seed(0)
    from lab3.data import load_cora
    data = load_cora()
    model = build_model("gcn")
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    tr_idx = data.lm_train
    for _ in range(300):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = nn.functional.cross_entropy(out[tr_idx], data.y[tr_idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        out = model(data.x, data.edge_index)
        acc = (out[tr_idx].argmax(1) == data.y[tr_idx]).float().mean().item()
    assert acc >= 0.99, f"GCN tiny-overfit failed: acc={acc}"
