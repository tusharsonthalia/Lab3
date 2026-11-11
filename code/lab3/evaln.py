"""Evaluation of a saved dense checkpoint (test set only, no weight updates)."""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from .data import fmnist_split_tensors, load_cora
from .models import build_model, count_params, dense_bytes
from .util import (
    CHECKPOINTS_DIR,
    base_record,
    run_wall_timer,
    set_seed,
    set_threads,
    write_run_record,
)


def load_checkpoint(architecture: str, seed: int):
    path = CHECKPOINTS_DIR / f"{architecture}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}; run train first")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(architecture)
    model.load_state_dict(payload["state_dict"])
    return model, payload["metadata"]


def eval(architecture: str, seed: int = 0) -> dict:
    """Evaluate the saved seed checkpoint on val+test. Never updates weights:
    inference_mode + eval() only. Appends a run record."""
    set_threads()
    t0 = time.perf_counter()
    rec = base_record(architecture, command="eval")
    rec["seed"] = seed
    try:
        model, meta = load_checkpoint(architecture, seed)
        rec["total_params"] = count_params(model)
        rec["nonzero_params"] = count_params(model)
        rec["dense_bytes"] = dense_bytes(model)
        rec["best_epoch"] = meta.get("best_epoch")
        rec["val_acc"] = meta.get("val_acc")
        rec["val_loss"] = meta.get("val_loss")

        if architecture == "gcn":
            data = load_cora()
            model.eval()
            with torch.inference_mode():
                out = model(data.x, data.edge_index)
                va = out[data.lm_val]
                te = out[data.lm_test]
                rec["val_loss"] = float(nn.functional.cross_entropy(va, data.y[data.lm_val]).item())
                rec["val_acc"] = float((va.argmax(dim=1) == data.y[data.lm_val]).float().mean().item())
                rec["test_acc"] = float((te.argmax(dim=1) == data.y[data.lm_test]).float().mean().item())
        else:
            tr_x, tr_y, va_x, va_y, te_x, te_y, _, _ = fmnist_split_tensors()
            model.eval()
            with torch.inference_mode():
                for x, y, kind in ((va_x, va_y, "val"), (te_x, te_y, "test")):
                    n = len(y)
                    correct = 0
                    loss_sum = 0.0
                    bs = 1024
                    for i in range(0, n, bs):
                        xb, yb = x[i:i + bs], y[i:i + bs]
                        out = model(xb)
                        loss_sum += float(nn.functional.cross_entropy(out, yb, reduction="sum").item())
                        correct += int((out.argmax(dim=1) == yb).sum().item())
                    if kind == "val":
                        rec["val_acc"] = correct / n
                        rec["val_loss"] = loss_sum / n
                    else:
                        rec["test_acc"] = correct / n

        rec = run_wall_timer(rec, t0)
        write_run_record(rec)
        return rec
    except Exception as e:
        rec = run_wall_timer(rec, t0)
        rec["status"] = "failed"
        rec["reason"] = f"{type(e).__name__}: {e}"
        write_run_record(rec)
        raise
