"""Frozen training loop with validation-selected checkpointing (Part A.1)."""
from __future__ import annotations

import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import fmnist_split_tensors, load_cora
from .models import build_model, count_params, dense_bytes
from .util import (
    CHECKPOINTS_DIR,
    CURVES_DIR,
    RESULTS_DIR,
    base_record,
    new_run_id,
    run_wall_timer,
    set_seed,
    set_threads,
    write_run_record,
)


def _eval_fmnist(model, x, y, batch_size=1024) -> tuple[float, float]:
    """Validation/test evaluation on Fashion-MNIST. Eval mode only; no weight updates."""
    model.eval()
    with torch.inference_mode():
        n = len(y)
        correct = 0
        loss_sum = 0.0
        for i in range(0, n, batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            out = model(xb)
            loss_sum += float(nn.functional.cross_entropy(out, yb, reduction="sum").item())
            correct += int((out.argmax(dim=1) == yb).sum().item())
    return correct / n, loss_sum / n


def _eval_cora(model, data, mask) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        out = model(data.x, data.edge_index)
        out = out[mask]
        y = data.y[mask]
        loss = float(nn.functional.cross_entropy(out, y).item())
        acc = float((out.argmax(dim=1) == y).float().mean().item())
    return acc, loss


def _checkpoint_path(architecture: str, seed: int):
    return CHECKPOINTS_DIR / f"{architecture}_seed{seed}.pt"


def _is_best(cand: tuple, best: tuple | None) -> bool:
    """cand=(val_acc, val_loss, epoch). Higher acc wins; tie -> lower loss; tie -> earlier epoch."""
    if best is None:
        return True
    ca, cl, ce = cand
    ba, bl, be = best
    if ca > ba:
        return True
    if ca == ba and cl < bl:
        return True
    if ca == ba and cl == bl and ce < be:
        return True
    return False


def _save_curve(run_id: str, rows: list[dict]) -> None:
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    path = CURVES_DIR / f"{run_id}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def train(architecture: str, cfg: dict, seed: int, run_id: str | None = None) -> dict:
    """Frozen training protocol. Returns the run record (also appended to runs.jsonl)."""
    set_threads()
    set_seed(seed)
    t0 = time.perf_counter()
    rec = base_record(architecture, command="train")
    rec["run_id"] = run_id or new_run_id(architecture)
    rec["seed"] = seed
    curve_rows: list[dict] = []

    try:
        model = build_model(architecture)
        rec["total_params"] = count_params(model)
        rec["nonzero_params"] = count_params(model)
        rec["dense_bytes"] = dense_bytes(model)

        if architecture == "gcn":
            data = load_cora()
            train_idx, val_idx, test_idx = data.lm_train, data.lm_val, data.lm_test
            max_epochs = int(cfg["max_epochs"])
            patience = int(cfg["patience"])
            opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                   weight_decay=float(cfg["weight_decay"]))

            best: tuple | None = None
            best_state: dict | None = None
            epochs_no_improve = 0
            epochs_trained = 0
            for epoch in range(1, max_epochs + 1):
                model.train()
                opt.zero_grad()
                out = model(data.x, data.edge_index)
                loss = nn.functional.cross_entropy(out[train_idx], data.y[train_idx])
                loss.backward()
                opt.step()
                epochs_trained = epoch

                va_acc, va_loss = _eval_cora(model, data, val_idx)
                curve_rows.append({"epoch": epoch, "train_loss": float(loss.item()),
                                   "val_acc": va_acc, "val_loss": va_loss})
                if _is_best((va_acc, va_loss, epoch), best):
                    best = (va_acc, va_loss, epoch)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            rec["train_epochs"] = epochs_trained
            rec["best_epoch"] = best[2]
            rec["val_acc"] = best[0]
            rec["val_loss"] = best[1]
            # Test set evaluated only on the selected checkpoint
            model.load_state_dict(best_state)
            te_acc, _ = _eval_cora(model, data, test_idx)
            rec["test_acc"] = te_acc
            _save_checkpoint(architecture, seed, model, rec, cfg)
        else:
            tr_x, tr_y, va_x, va_y, te_x, te_y, mean, std = fmnist_split_tensors()
            batch_size = int(cfg["batch_size"])
            max_epochs = int(cfg["max_epochs"])
            patience = int(cfg["patience"])
            opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                   weight_decay=float(cfg["weight_decay"]))
            loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch_size,
                                shuffle=True, drop_last=False)

            best = None
            best_state = None
            epochs_no_improve = 0
            epochs_trained = 0
            for epoch in range(1, max_epochs + 1):
                model.train()
                correct = 0
                loss_sum = 0.0
                n = 0
                for xb, yb in loader:
                    opt.zero_grad()
                    out = model(xb)
                    loss = nn.functional.cross_entropy(out, yb)
                    loss.backward()
                    opt.step()
                    loss_sum += float(loss.item()) * len(yb)
                    correct += int((out.argmax(dim=1) == yb).sum().item())
                    n += len(yb)
                epochs_trained = epoch
                va_acc, va_loss = _eval_fmnist(model, va_x, va_y)
                curve_rows.append({"epoch": epoch, "train_loss": loss_sum / n,
                                   "train_acc": correct / n,
                                   "val_acc": va_acc, "val_loss": va_loss})
                if _is_best((va_acc, va_loss, epoch), best):
                    best = (va_acc, va_loss, epoch)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            rec["train_epochs"] = epochs_trained
            rec["best_epoch"] = best[2]
            rec["val_acc"] = best[0]
            rec["val_loss"] = best[1]
            model.load_state_dict(best_state)
            te_acc, _ = _eval_fmnist(model, te_x, te_y)
            rec["test_acc"] = te_acc
            _save_checkpoint(architecture, seed, model, rec, cfg)

        _save_curve(rec["run_id"], curve_rows)
        rec = run_wall_timer(rec, t0)
        write_run_record(rec)
        return rec

    except Exception as e:  # record crashes; never leave runs.jsonl silent
        rec = run_wall_timer(rec, t0)
        rec["status"] = "failed"
        rec["reason"] = f"{type(e).__name__}: {e}"
        try:
            if curve_rows:
                _save_curve(rec["run_id"], curve_rows)
        except Exception:
            pass
        write_run_record(rec)
        raise


def _save_checkpoint(architecture: str, seed: int, model, rec: dict, cfg: dict) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save({
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "metadata": {
            "architecture": architecture,
            "seed": seed,
            "best_epoch": rec["best_epoch"],
            "train_epochs": rec["train_epochs"],
            "val_acc": rec["val_acc"],
            "val_loss": rec["val_loss"],
            "test_acc": rec["test_acc"],
            "config": cfg,
            "run_id": rec["run_id"],
            "total_params": rec["total_params"],
            "dense_bytes": rec["dense_bytes"],
            "code_hash": rec["code_hash"],
            "versions": rec["versions"],
            "timestamp": rec["timestamp"],
        },
    }, _checkpoint_path(architecture, seed))
