"""Part B pruning pipeline (predictions.md B.1-B.4, frozen).

`uv run python -m lab3 prune --model {mlp,cnn,gcn} --method {s1,s2,s3,s4}
  --target {0.25,0.5,0.75,0.9} --seed K`

One arm = one-shot prune + frozen fine-tuning budget, measured twice
(`stage=post_prune` and `stage=post_finetune`), two rows sharing an arm_id.
parent_run_id = the canonical seed-0 dense TRAINING run of the architecture.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import sparsity as S
from .compact import (
    CompactModel,
    CSR_INDEX_DTYPE,
    _csr_components,
    csr_int64_bytes,
    s4_bitpacked_bytes_theoretical,
)
from .data import fmnist_split_tensors, load_cora
from .models import build_model, count_params, dense_bytes, gzip_bytes, raw_tensor_bytes
from .sparsity import (
    _STATE_KEY,
    eligible_state_keys,
    eligible_weights,
    make_masked_model,
    nm_pattern_for_target,
    reapply_masks,
    s1_masks,
    s2_plans,
    s2_rebuild,
    s3_masks_global,
    s4_masks,
    s4_packed,
    scatter_into_dense,
)
from .training import _eval_cora, _eval_fmnist
from .util import (
    CALIB_FILE,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    base_record,
    set_seed,
    set_threads,
    write_run_record,
)

# B1b: the pruned-arm checkpoint root is configurable (tests redirect it to a
# tmp dir so pytest never writes into the real checkpoints/ tree). Defaults to
# the repo checkpoints/pilot (sibling of checkpoints/dense, which holds the
# read-only dense parents and is NEVER redirected).
PILOT_DIR = Path(os.environ.get("LAB3_PILOT_DIR", str(CHECKPOINTS_DIR.parent / "pilot")))
COMPACT_DIR = RESULTS_DIR / "compacts"

# B.1 frozen fine-tuning budgets (identical for every method within an arch)
FT_BUDGET = {
    "mlp": {"epochs": 3, "lr": 5e-4, "batch_size": 128},
    "cnn": {"epochs": 3, "lr": 5e-4, "batch_size": 128},
    "gcn": {"epochs": 50, "lr": 0.005, "weight_decay": 5e-4, "dropout": 0.5},
}


def canonical_parent_run_id(architecture: str) -> str:
    """The canonical seed-0 dense TRAINING run for `architecture` (from the
    seed-0 checkpoint metadata)."""
    from .evaln import load_checkpoint

    _, meta = load_checkpoint(architecture, 0)
    return meta["run_id"]


def _new_arm_id(architecture: str, method: str, target: float, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{architecture}-{method}-t{target}-seed{seed}-{ts}-{uuid.uuid4().hex[:6]}"


def _new_stage_run_id(architecture: str, method: str, target: float, stage: str) -> str:
    """B1b: run_id encodes architecture, method, target and stage, e.g.
    mlp-s1-t0.50-post_finetune-<UTC ts>-<shortid> (schema identity; previously
    all Part B rows were labelled with the default 'dense' method id)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (f"{architecture}-{method}-t{target:.2f}-{stage}"
            f"-{ts}-{uuid.uuid4().hex[:6]}")


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _weight_of(mod: nn.Module) -> nn.Parameter:
    return mod.lin.weight if type(mod).__name__ == "GCNConv" else mod.weight


def per_layer_table(model: nn.Module, include_inner: bool = False) -> list[dict]:
    """One entry per eligible tensor: name, shape, numel, nonzero, sparsity.

    A GCNConv's projection is its inner ``lin`` submodule (state key
    ``<name>.lin.weight``), the same tensor that ``mod.lin.weight`` refers to,
    so it is reported once under its state key. `include_inner=True` also
    emits the duplicate submodule entry (legacy B1 behavior, kept for the
    pilot-era tables)."""
    out = []
    for name, mod in model.named_modules():
        t = type(mod).__name__
        if t not in ("Linear", "Conv2d", "GCNConv"):
            continue
        if t == "GCNConv" and not include_inner:
            # the GCNConv's projection is its inner `lin` Linear submodule
            # (state key <name>.lin.weight); it is reported exactly once, by
            # the submodule itself. Skip it here to avoid a duplicate entry.
            sub = next(
                (s for s in mod.modules() if type(s).__name__ == "Linear" and s is not mod),
                None,
            )
            if sub is not None:
                continue
        w = _weight_of(mod)
        nonzero = int(w.count_nonzero())
        out.append({
            "name": name,
            "shape": list(w.shape),
            "numel": int(w.numel()),
            "nonzero": nonzero,
            "sparsity": 1.0 - nonzero / w.numel() if w.numel() else 0.0,
        })
    return out


def achieved_sparsities(model: nn.Module, parent: nn.Module,
                        dense_total_params: int) -> tuple[float, float]:
    """(achieved_sparsity, achieved_param_sparsity).

    B1b fix: achieved_sparsity is the fraction of the PARENT's eligible
    weight elements that are zero in `model` (S1/S3/S4: identical shapes;
    S2: rebuilt entries scattered back into parent shapes via the reversible
    mapping). For S1/S3/S4 this equals the fraction of pruned entries (~ the
    target); for S2 it counts the removed groups' weights against the parent.
    achieved_param_sparsity = fraction of all dense params zero (includes
    biases and, for S2, the rebuilt model's smaller param total counted
    against the dense parent)."""
    w = eligible_weights(model)
    w_parent = eligible_weights(parent)
    nnz = sum(int(t.count_nonzero()) for t in w.values())
    numel_parent = sum(t.numel() for t in w_parent.values())
    nnz_all = sum(int(p.count_nonzero()) for p in model.parameters())
    return (1.0 - nnz / numel_parent, 1.0 - nnz_all / dense_total_params)


def s2_dense_zeros_model(architecture: str, parent: nn.Module,
                         rebuilt: nn.Module, plans: list[dict]) -> nn.Module:
    """B1b fix (key S2 contrast): the dense PARENT-SHAPED model with the
    removed groups zeroed, built via the reversible scatter mapping.
    Kept entries equal the rebuilt weights; removed groups are exactly zero.
    This is the `dense_zeros` runtime path; the rebuilt model is `compact`.
    Same zeros, different representation."""
    dmap = scatter_into_dense(rebuilt, parent, plans)
    m = build_model(architecture)
    state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    state.update({k: v for k, v in dmap.items()})
    m.load_state_dict(state)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# One-shot pruning (separate code paths: mask-based vs physical surgery)
# ---------------------------------------------------------------------------

def _calib_loss_fn(architecture: str):
    """B.1 calibration subset: 1,024 training-split images (frozen
    splits/fmnist_calib_idx.npy, training labels) for MLP/CNN; the 140 training
    nodes on the full graph for GCN. Loss = mean cross-entropy over the WHOLE
    subset (single batch), on the dense parent in eval mode (no dropout).

    Indices are mapped through the split position (the calib file stores
    official 60k-set indices)."""
    if architecture == "gcn":
        data = load_cora()

        def loss(m: nn.Module) -> torch.Tensor:
            out = m(data.x, data.edge_index)
            return nn.functional.cross_entropy(out[data.lm_train], data.y[data.lm_train])
        return loss
    from .util import load_indices

    idx = load_indices("fmnist")
    train_idx = idx["train"]
    pos = {int(i): k for k, i in enumerate(train_idx)}
    calib_official = np.load(CALIB_FILE)
    calib_pos = np.sort(np.asarray([pos[int(i)] for i in calib_official], dtype=np.int64))
    tr_x, tr_y, _, _, _, _, _, _ = fmnist_split_tensors()
    x, y = tr_x[calib_pos], tr_y[calib_pos]

    def loss(m: nn.Module) -> torch.Tensor:
        return nn.functional.cross_entropy(m(x), y)
    return loss


def prune_once(architecture: str, method: str, target: float, parent_state: dict):
    """One-shot prune (B.2). Returns (model, masks, plans, method_params).

    S1/S3/S4: dense-shaped parent with binary masks applied (zeros).
    S2: physically rebuilt smaller dense model (masks=None, plans set).
    method_params carries the masks/kept_groups needed to persist the arm."""
    parent = build_model(architecture)
    parent.load_state_dict(parent_state)
    parent.eval()

    if method == "s2":
        model, plans = s2_rebuild(parent, target)
        params = {"group_sparsity": target}
        params.update(s2_group_sparsity(architecture, parent, plans))
        params["kept_groups"] = {
            p["layer"]: {"kind": p["kind"], "kept": p["kept"].detach().clone()}
            for p in plans}
        return model, None, plans, params
    if method == "s1":
        masks = s1_masks(parent, target)
    elif method == "s3":
        masks = s3_masks_global(parent, target, _calib_loss_fn(architecture))
    elif method == "s4":
        n, m = nm_pattern_for_target(target)
        masks = s4_masks(parent, target)
    else:
        raise ValueError(f"unknown method {method}")
    model = build_model(architecture)
    model.load_state_dict(parent_state)
    make_masked_model(model, masks)
    model.eval()
    params = {"pattern": f"{n}:{m}"} if method == "s4" else {}
    params["masks"] = {k: v.detach().clone() for k, v in masks.items()}
    return model, masks, None, params


def s2_group_sparsity(architecture: str, parent: nn.Module, plans: list[dict]) -> dict:
    """B1b fix: achieved group sparsity = removed_groups / total_groups, per
    prunable layer plus the overall value. The buggy B1 version accumulated a
    single kept-count across layers while resetting the total per layer
    (yielding e.g. -0.5); this is the per-layer-correct computation."""
    per_layer = []
    tot = kept = 0
    for p in plans:
        total = _group_counts(architecture, parent, p)
        k = int(len(p["kept"]))
        tot += total
        kept += k
        per_layer.append({
            "layer": p["layer"],
            "groups_total": int(total),
            "groups_kept": k,
            "groups_removed": int(total - k),
            "group_sparsity_achieved": (total - k) / total if total else 0.0,
        })
    overall = (tot - kept) / tot if tot else 0.0
    return {"per_layer": per_layer, "group_sparsity_achieved": overall}


def _group_counts(architecture: str, parent: nn.Module, plan: dict) -> int:
    """Total groups in the layer a plan refers to."""
    cls = architecture
    if cls == "mlp":
        seq = parent.net
        return seq[0].weight.shape[0] if plan["layer"] == "net.0" else seq[2].weight.shape[0]
    if cls == "cnn":
        if plan["layer"] == "conv1":
            return parent.conv1.weight.shape[0]
        if plan["layer"] == "conv2":
            return parent.conv2.weight.shape[0]
        return parent.fc1.weight.shape[0]
    if cls == "gcn":
        return parent.conv1.lin.weight.shape[0]  # hidden features = conv1 out rows
    raise ValueError(cls)


# ---------------------------------------------------------------------------
# Fine-tuning (B.1 budget; masks re-applied after every optimizer step)
# ---------------------------------------------------------------------------

def _is_best_ft(cand, best):
    """Best val acc within budget; ties -> LATER epoch (the more-trained model)."""
    if best is None:
        return True
    if cand[0] > best[0]:
        return True
    if cand[0] == best[0] and cand[2] >= best[2]:
        return True
    return False


def fine_tune(architecture: str, model: nn.Module, seed: int,
              masks: dict[str, torch.Tensor] | None = None) -> tuple[dict, int]:
    """Frozen B.1 fine-tuning budget. Returns (best-val checkpoint state_dict,
    best_epoch). The seed controls shuffle order and dropout (B.1)."""
    budget = FT_BUDGET[architecture]
    set_seed(seed)
    epochs = budget["epochs"]

    if architecture == "gcn":
        data = load_cora()
        opt = torch.optim.Adam(model.parameters(), lr=budget["lr"],
                               weight_decay=budget["weight_decay"])
        best, best_state = None, None
        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = nn.functional.cross_entropy(out[data.lm_train], data.y[data.lm_train])
            loss.backward()
            opt.step()
            if masks is not None:
                reapply_masks(model, masks)
            va_acc, va_loss = _eval_cora(model, data, data.lm_val)
            if _is_best_ft((va_acc, va_loss, epoch), best):
                best = (va_acc, va_loss, epoch)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return best_state, best[2]

    tr_x, tr_y, va_x, va_y, _, _, _, _ = fmnist_split_tensors()
    loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=budget["batch_size"],
                        shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=budget["lr"], weight_decay=0.0)
    best, best_state = None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = nn.functional.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            if masks is not None:
                reapply_masks(model, masks)
        va_acc, va_loss = _eval_fmnist(model, va_x, va_y)
        if _is_best_ft((va_acc, va_loss, epoch), best):
            best = (va_acc, va_loss, epoch)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return best_state, best[2]


def _metrics(architecture: str, model: nn.Module) -> tuple[float, float, float]:
    """(val_acc, val_loss, test_acc) on the frozen splits (eval mode, no
    training-mode side effects such as GCN edge-index caching)."""
    was_training = model.training
    model.eval()
    try:
        if architecture == "gcn":
            data = load_cora()
            va_acc, va_loss = _eval_cora(model, data, data.lm_val)
            te_acc, _ = _eval_cora(model, data, data.lm_test)
            return va_acc, va_loss, te_acc
        _, _, va_x, va_y, te_x, te_y, _, _ = fmnist_split_tensors()
        va_acc, va_loss = _eval_fmnist(model, va_x, va_y)
        te_acc, _ = _eval_fmnist(model, te_x, te_y)
        return va_acc, va_loss, te_acc
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Stage record
# ---------------------------------------------------------------------------

def _stage_record(arm: dict, stage: str, model: nn.Module, parent: nn.Module,
                  dense_total_params: int, val_acc, val_loss, test_acc) -> dict:
    from .bench import estimate_macs, load_bench_input

    # B1b fix #2: run_id encodes architecture, method, target and stage.
    rec = base_record(arm["architecture"], command="prune")
    rec["run_id"] = _new_stage_run_id(arm["architecture"], arm["method"],
                                      arm["target"], stage)
    rec.update({
        "part": "B",
        "phase": arm.get("phase"),
        "arm_id": arm["arm_id"],
        "stage": stage,
        "method": arm["method"],
        "target_sparsity": arm["target"],
        "seed": arm["seed"],
        "parent_run_id": arm["parent_run_id"],
        "method_params": arm["method_params"],
        "checkpoint": arm["checkpoint_path"],
    })
    # B1b fix #6: achieved sparsity relative to the PARENT's eligible weights.
    ach_w, ach_p = achieved_sparsities(model, parent, dense_total_params)
    rec["achieved_sparsity"] = ach_w
    rec["achieved_param_sparsity"] = ach_p
    rec["total_params"] = count_params(model)
    rec["nonzero_params"] = sum(int(p.count_nonzero()) for p in model.parameters())
    rec["per_layer"] = per_layer_table(model)
    rec["val_acc"] = val_acc
    rec["val_loss"] = val_loss
    rec["test_acc"] = test_acc
    # B1b fix #3: est_macs / est_macs_nonzero are PER IMAGE for fmnist (the
    # bench input is the fixed 256-image batch; the per-sample convention is
    # applied inside estimate_macs) and per full graph for Cora.
    x, ei = load_bench_input(arm["architecture"])
    macs = estimate_macs(
        model, x, ei,
        x_nnz=int(x.count_nonzero()) if arm["architecture"] == "gcn" else None)
    rec["est_macs"] = macs["est_macs"]
    rec["est_macs_nonzero"] = macs["est_macs_nonzero"]
    rec["est_macs_sparse_input"] = macs["est_macs_sparse_input"]
    rec["gcn_norm_macs"] = macs["gcn_norm_macs"]
    return rec


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def prune(architecture: str, method: str, target: float, seed: int,
          bench: bool = True, write: bool = True, phase: str | None = None) -> dict:
    """Run one Part B arm (B1b: `method` is one of s1/s2/s3/s4/dense_ft).
    `phase` (B2): the matrix phase the arm belongs to ("pilot" / later
    "confirm"); recorded on every row it produces.
    Returns {"arm_id", "status", "reason", "rows", ...}."""
    set_threads()
    set_seed(seed)
    t0 = time.perf_counter()
    arch = architecture.lower()
    if method == "dense_ft":
        assert target == 0.0
    else:
        assert method in ("s1", "s2", "s3", "s4")
        assert target in (0.25, 0.5, 0.75, 0.9)

    from .evaln import load_checkpoint
    parent, pmeta = load_checkpoint(arch, 0)
    parent_state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    dense_total_params = count_params(parent)
    # B1b fix #5: train_epochs = the parent's (from the seed-0 dense training
    # run metadata carried on the checkpoint).
    parent_train_epochs = pmeta.get("train_epochs")
    arm_id = _new_arm_id(arch, method, target, seed)
    checkpoint_path = str(PILOT_DIR / f"{arm_id}.pt")
    arm = {
        "architecture": arch, "method": method, "target": target, "seed": seed,
        "arm_id": arm_id, "parent_run_id": pmeta["run_id"],
        "parent_train_epochs": parent_train_epochs,
        "checkpoint_path": checkpoint_path, "method_params": {},
        "phase": phase,
    }

    rows: list[dict] = []
    status, reason = "ok", None
    try:
        # B.6 control arm: same parent + B.1 fine-tuning budget, NO pruning.
        if method == "dense_ft":
            model = build_model(arch)
            model.load_state_dict(parent_state)
            model.eval()
            params = {"control": "dense_ft (B.6)", "target_sparsity": 0.0}
            arm["method_params"] = params
            va, vl, te = _metrics(arch, model)
            r2 = _stage_record(arm, "post_finetune", model, parent, dense_total_params,
                               va, vl, te)
            ft_state, best_epoch = fine_tune(arch, model, seed, None)
            model.load_state_dict(ft_state)
            model.eval()
            va, vl, te = _metrics(arch, model)
            r2 = _stage_record(arm, "post_finetune", model, parent, dense_total_params,
                               va, vl, te)
            r2["finetune_epochs"] = FT_BUDGET[arch]["epochs"]
            r2["best_epoch"] = best_epoch
            rows.append(r2)
            _finish_arm(arch, method, target, seed, arm, model, params, rows,
                        bench, best_epoch)
        else:
            model, masks, plans, params = prune_once(arch, method, target, parent_state)
            arm["method_params"] = params

            # S2 dense_zeros path (B1b fix #7): the dense parent-shaped model
            # with the removed groups zeroed (reversible scatter mapping).
            s2_dz = None
            if method == "s2":
                s2_dz = s2_dense_zeros_model(arch, parent, model, plans)

            # ---- stage 1: post_prune ----
            va, vl, te = _metrics(arch, model)
            r1 = _stage_record(arm, "post_prune", model, parent, dense_total_params,
                               va, vl, te)
            r1["finetune_epochs"] = 0          # B1b fix #5: no FT yet
            rows.append(r1)

            # ---- fine-tune (B.1 budget) ----
            ft_state, best_epoch = fine_tune(arch, model, seed, masks)
            model.load_state_dict(ft_state)
            model.eval()
            if method == "s2":
                s2_dz.load_state_dict({k: v for k, v in
                                       {**{kk: vv.detach().clone() for kk, vv in parent.state_dict().items()},
                                        **{k: v for k, v in
                                           scatter_into_dense(model, parent, plans).items()}}.items()})
                s2_dz.eval()

            # ---- stage 2: post_finetune ----
            va, vl, te = _metrics(arch, model)
            r2 = _stage_record(arm, "post_finetune", model, parent, dense_total_params,
                               va, vl, te)
            r2["finetune_epochs"] = FT_BUDGET[arch]["epochs"]   # B1b fix #5
            r2["best_epoch"] = best_epoch
            rows.append(r2)

            _finish_arm(arch, method, target, seed, arm, model, params, rows,
                        bench, best_epoch)

        if bench and rows and "paths" in rows[0]:
            _lift_top_level(rows, rows[0]["paths"])
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        status, reason = "failed", _classify(e)
        for r in rows:
            r["status"] = status
            r["reason"] = reason
            r["wall_time_s"] = round(time.perf_counter() - t0, 3)
        rows.append(base_record(arch, command="prune"))
        rows[-1]["run_id"] = _new_stage_run_id(arch, method, target, "failed")
        rows[-1].update({"part": "B", "phase": phase, "arm_id": arm_id,
                         "stage": "failed",
                         "method": method, "target_sparsity": target, "seed": seed,
                         "parent_run_id": pmeta["run_id"], "status": status,
                         "reason": reason,
                         "wall_time_s": round(time.perf_counter() - t0, 3)})

    # B1b fix #5: wall_time_s is filled on every row (and train_epochs = the
    # parent's, per the schema).
    for r in rows:
        r["wall_time_s"] = round(time.perf_counter() - t0, 3)
        if r.get("train_epochs") is None:
            r["train_epochs"] = arm.get("parent_train_epochs")

    if write:
        for r in rows:
            write_run_record(r)
    return {"arm_id": arm_id, "status": status, "reason": reason, "rows": rows,
            "wall_time_s": round(time.perf_counter() - t0, 3)}


def _finish_arm(arch, method, target, seed, arm, model, params, rows, bench,
                best_epoch) -> None:
    """Save the post-finetune arm checkpoint (and masks/kept-group indices) and,
    if requested, run the two B.4 runtime paths on the saved checkpoint."""
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "architecture": arch, "method": method, "target": target, "seed": seed,
        "arm_id": arm["arm_id"], "parent_run_id": arm["parent_run_id"],
        "best_epoch": best_epoch,
        "val_acc": rows[-1]["val_acc"], "val_loss": rows[-1]["val_loss"],
        "test_acc": rows[-1]["test_acc"],
    }
    if method in ("s1", "s3", "s4") and "masks" in params:
        save_payload["masks"] = params["masks"]
    if method == "s2":
        save_payload["kept_groups"] = params["kept_groups"]
    torch.save(save_payload, arm["checkpoint_path"])
    if bench:
        p1, p2 = runtime_paths(arch, method, arm["checkpoint_path"], target)
        for r in rows:
            r["paths"] = {"dense_zeros": p1, "compact": p2}


def _lift_top_level(rows: list[dict], paths: dict) -> None:
    """B1b fix #4: top-level size/memory/latency fields come from the
    dense_zeros path (same meaning as Part A), top-level compressed_bytes from
    the compact path. The nested `paths` dict is kept as-is."""
    p1 = paths.get("dense_zeros", {})
    p2 = paths.get("compact", {})
    for r in rows:
        r["dense_bytes"] = p1.get("dense_bytes")
        r["raw_tensor_bytes"] = p1.get("raw_tensor_bytes")
        r["gzip_bytes"] = p1.get("gzip_bytes")
        r["rss_before_model_bytes"] = p1.get("rss_before_model_bytes")
        r["peak_rss_bytes"] = p1.get("peak_rss_bytes")
        r["model_rss_delta_bytes"] = p1.get("model_rss_delta_bytes")
        lat = {k: v for k, v in p1.items() if isinstance(v, dict) and "median_ms" in v}
        r["latency"] = lat
        r["compressed_bytes"] = p2.get("compressed_bytes")


# ---------------------------------------------------------------------------
# B2 pilot driver: {mlp,cnn,gcn} x {s1,s2,s3,s4} x {0.25,0.5,0.75,0.9} at seed 0
# + one dense_ft control per architecture (B.6). phase="pilot" on every row.
# ---------------------------------------------------------------------------

PILOT_ARCHS = ("mlp", "cnn", "gcn")
PILOT_METHODS = ("s1", "s2", "s3", "s4")
PILOT_TARGETS = (0.25, 0.5, 0.75, 0.9)
PILOT_SEED = 0          # B.1 frozen pilot seed
PILOT_LOG = RESULTS_DIR / "pilot_log.txt"
PILOT_SUMMARY_CSV = RESULTS_DIR / "pilot_summary.csv"
PILOT_PER_LAYER_CSV = RESULTS_DIR / "pilot_per_layer.csv"


def _pilot_cells(arch: str | None, only: str | None):
    """Frozen pilot matrix: 3 archs x 4 methods x 4 targets (48 arms) + one
    dense_ft control (target 0.0) per architecture (B.6, 3 control arms).
    `arch` restricts to one architecture; `only` restricts to explicit
    (method,target) cells for debugging/rerun."""
    cells = []
    for a in PILOT_ARCHS if arch is None else (arch,):
        for m in PILOT_METHODS:
            for t in PILOT_TARGETS:
                cells.append((a, m, t))
        cells.append((a, "dense_ft", 0.0))
    if only:
        wanted = set()
        for cell in only.split(";"):
            mm, tt = cell.strip().split(",")
            wanted.add((mm, float(tt)))
        cells = [c for c in cells if (c[1], c[2]) in wanted]
    return cells


def _pilot_log_line(line: str) -> None:
    """One progress line per arm: printed (to stdout, captured by the nohup
    redirect into results/pilot_log.txt) and appended durably to
    results/pilot_log.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(line, flush=True)
    with open(PILOT_LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def _pilot_done_keys(run_path=None) -> set:
    """Idempotency: (arch, method, target) cells that already have an ok
    phase=pilot post_finetune row in results/runs.jsonl are skipped on
    restart. Failed rows (and partial rows without a post_finetune row) do
    NOT count as done — the arm is re-run."""
    from .util import RUNS_JSONL
    run_path = Path(run_path) if run_path else RUNS_JSONL
    done = set()
    if not run_path.exists():
        return done
    with open(run_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("part") == "B" and r.get("phase") == "pilot"
                    and r.get("stage") == "post_finetune"
                    and r.get("status") == "ok"):
                done.add((r.get("architecture"), r.get("method"),
                          r.get("target_sparsity")))
    return done


# ---------------------------------------------------------------------------
# B3 confirm driver (predictions.md B.3/B.7): the 12 selected (arch, method,
# target) points from results/confirm_selection.json rerun with seeds 1, 2, 3
# from the canonical seed-0 parent under the same frozen budget, plus the
# dense_ft control (target 0) rerun with seeds 1, 2, 3 per architecture.
# 12 x 3 + 3 x 3 = 45 arms, phase="confirm" on every row.
# ---------------------------------------------------------------------------

CONFIRM_SELECTION_FILE = RESULTS_DIR / "confirm_selection.json"
CONFIRM_SEEDS = (1, 2, 3)
CONFIRM_LOG = RESULTS_DIR / "confirm_log.txt"
CONFIRM_SUMMARY_CSV = RESULTS_DIR / "confirm_summary.csv"


def load_confirm_selection() -> list[dict]:
    """The frozen B.7 selection (read-only; applied before any confirmatory
    run). Each entry: arch, method, selected_target, threshold, ..."""
    with open(CONFIRM_SELECTION_FILE) as f:
        return json.load(f)


def confirm_cells(arch: str | None = None,
                  only: str | None = None) -> list[tuple[str, str, float, int]]:
    """Frozen confirm grid: (arch, method, target, seed). 12 selected points x
    seeds 1-3 (36) + dense_ft control (target 0) per architecture x seeds 1-3
    (9) = 45 arms. `arch` restricts to one architecture; `only` restricts to
    explicit (method,target) cells as in the pilot driver."""
    sel = load_confirm_selection()
    cells = []
    for entry in sel:
        a = entry["arch"]
        if arch is not None and a != arch:
            continue
        cells.append((a, entry["method"], float(entry["selected_target"])))
    for a in ("mlp", "cnn", "gcn"):
        if arch is not None and a != arch:
            continue
        cells.append((a, "dense_ft", 0.0))
    if only:
        wanted = set()
        for cell in only.split(";"):
            mm, tt = cell.strip().split(",")
            wanted.add((mm, float(tt)))
        cells = [c for c in cells if (c[1], c[2]) in wanted]
    out = []
    for (a, m, t) in cells:
        for s in CONFIRM_SEEDS:
            out.append((a, m, t, s))
    return out


def _confirm_log_line(line: str) -> None:
    """One progress line per arm: printed (captured by the nohup redirect)
    and appended durably to results/confirm_log.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(line, flush=True)
    with open(CONFIRM_LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def _confirm_done_keys(run_path=None) -> set:
    """Idempotency (same rule as the pilot driver, phase=confirm): a
    (arch, method, target, seed) cell is done when results/runs.jsonl already
    has an ok phase=confirm post_finetune row for it. Failed rows and partial
    rows do NOT count as done — the arm is re-run."""
    from .util import RUNS_JSONL
    run_path = Path(run_path) if run_path else RUNS_JSONL
    done = set()
    if not run_path.exists():
        return done
    with open(run_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("part") == "B" and r.get("phase") == "confirm"
                    and r.get("stage") == "post_finetune"
                    and r.get("status") == "ok"):
                done.add((r.get("architecture"), r.get("method"),
                          r.get("target_sparsity"), r.get("seed")))
    return done


def run_confirm(arch: str | None = None, only: str | None = None) -> dict:
    """B3 confirm driver (predictions.md B.3/B.7). Reruns each of the 12
    selected (arch, method, selected_target) points with seeds 1, 2, 3 from
    the canonical seed-0 dense parent under the same frozen budget, plus the
    dense_ft control with seeds 1, 2, 3 per architecture (45 arms total),
    each arm emitting its post_prune / post_finetune rows with
    phase="confirm". Same failure handling as the pilot driver: a failed arm
    writes its failed row(s), the driver continues. Idempotent: cells that
    already have an ok phase=confirm post_finetune row are skipped on
    restart."""
    t_start = time.perf_counter()
    cells = confirm_cells(arch, only)
    done = _confirm_done_keys()
    _confirm_log_line(
        f"[confirm] start {datetime.now(timezone.utc).isoformat()} "
        f"cells={len(cells)} already_done={len(done & set(cells))}")
    results = []
    for i, (a, m, t, s) in enumerate(cells, 1):
        if (a, m, t, s) in done:
            _confirm_log_line(f"[confirm] {i:>2}/{len(cells)} {a}/{m}/t{t:<4} "
                              f"seed{s} skip (already ok phase=confirm post_finetune)")
            results.append((a, m, t, s, "skipped", None, 0.0))
            continue
        _confirm_log_line(f"[confirm] {i:>2}/{len(cells)} {a}/{m}/t{t:<4} seed{s} start")
        t0 = time.perf_counter()
        try:
            out = prune(a, m, t, seed=s, bench=True, write=True, phase="confirm")
            wall = out["wall_time_s"]
            status, reason = out["status"], out["reason"]
        except Exception as e:  # driver must continue past any arm failure
            import traceback as _tb
            _tb.print_exc()
            wall = time.perf_counter() - t0
            status, reason = "failed", _classify(e)
            _confirm_log_line(f"[confirm] {a}/{m}/t{t}/seed{s} driver-level "
                              f"exception: {reason}")
            results.append((a, m, t, s, status, reason, wall))
            continue
        accs = {r["stage"]: r.get("test_acc") for r in out["rows"]
                if r.get("stage") in ("post_prune", "post_finetune")}
        _confirm_log_line(
            f"[confirm] {i:>2}/{len(cells)} {a}/{m}/t{t:<4} seed{s} status={status} "
            f"arm_id={out['arm_id']} wall={wall:.0f}s "
            f"test_acc={accs.get('post_prune')}/{accs.get('post_finetune')} "
            + (f"reason={reason}" if reason else ""))
        results.append((a, m, t, s, status, reason, wall))
    total = time.perf_counter() - t_start
    _confirm_log_line(f"[confirm] done {datetime.now(timezone.utc).isoformat()} "
                      f"total_wall={total:.0f}s")
    write_confirm_summary()
    return {"results": results, "total_wall_time_s": round(total, 1)}


def run_pilot(seed: int = PILOT_SEED, arch: str | None = None,
              only: str | None = None) -> dict:
    """B2 pilot driver. Runs every remaining cell of the frozen pilot matrix
    (with `seed`), each arm emitting its post_prune and post_finetune rows
    (phase=pilot). A failed arm writes its failed row(s) and the driver
    continues with the next arm. Idempotent: cells that already have an ok
    phase=pilot post_finetune row are skipped on restart. One progress line
    per arm goes to stdout (nohup-captured into results/pilot_log.txt) and is
    appended durably to results/pilot_log.txt."""
    assert seed == PILOT_SEED, "the pilot uses the frozen seed 0 (B.1)"
    t_start = time.perf_counter()
    cells = _pilot_cells(arch, only)
    done = _pilot_done_keys()
    _pilot_log_line(
        f"[pilot] start {datetime.now(timezone.utc).isoformat()} seed={seed} "
        f"cells={len(cells)} already_done={len(done & set(cells))}")
    results = []
    for i, (a, m, t) in enumerate(cells, 1):
        if (a, m, t) in done:
            _pilot_log_line(f"[pilot] {i:>2}/{len(cells)} {a}/{m}/t{t:<4} "
                            f"skip (already ok phase=pilot post_finetune)")
            results.append((a, m, t, "skipped", None, 0.0))
            continue
        _pilot_log_line(f"[pilot] {i:>2}/{len(cells)} {a}/{m}/t{t:<4} start")
        t0 = time.perf_counter()
        try:
            out = prune(a, m, t, seed=seed, bench=True, write=True, phase="pilot")
            wall = out["wall_time_s"]
            status, reason = out["status"], out["reason"]
        except Exception as e:  # driver must continue past any arm failure
            import traceback as _tb
            _tb.print_exc()
            wall = time.perf_counter() - t0
            status, reason = "failed", _classify(e)
            _pilot_log_line(f"[pilot] {a}/{m}/t{t} driver-level exception: {reason}")
            results.append((a, m, t, status, reason, wall))
            continue
        accs = {r["stage"]: r.get("test_acc") for r in out["rows"]
                if r.get("stage") in ("post_prune", "post_finetune")}
        _pilot_log_line(
            f"[pilot] {i:>2}/{len(cells)} {a}/{m}/t{t:<4} status={status} "
            f"arm_id={out['arm_id']} wall={wall:.0f}s "
            f"test_acc={accs.get('post_prune')}/{accs.get('post_finetune')} "
            + (f"reason={reason}" if reason else ""))
        results.append((a, m, t, status, reason, wall))
    total = time.perf_counter() - t_start
    _pilot_log_line(f"[pilot] done {datetime.now(timezone.utc).isoformat()} "
                    f"total_wall={total:.0f}s")
    write_pilot_summaries()
    return {"results": results, "total_wall_time_s": round(total, 1)}


# ---------------------------------------------------------------------------
# B2 summary tables (results/pilot_summary.csv, results/pilot_per_layer.csv)
# ---------------------------------------------------------------------------


def _lat_cells_nested(paths: dict, path: str) -> list:
    """(key, median_ms, iqr_ms) from the nested paths.<path>.* latency dicts."""
    out = []
    p = paths.get(path) or {}
    for k, v in p.items():
        if isinstance(v, dict) and "median_ms" in v:
            out.append((k, v["median_ms"], v["iqr_ms"]))
    return sorted(out)


def write_pilot_summaries() -> tuple[Path, Path]:
    """Rebuild results/pilot_summary.csv (one line per arm x stage) and
    results/pilot_per_layer.csv (arm, stage, per eligible tensor) from the
    phase=pilot rows of results/runs.jsonl. Deterministic: re-runnable at any
    time; reflects the current log contents."""
    from .util import RUNS_JSONL
    import csv
    from .sparsity import NM_PATTERNS
    import torch  # for loading compact artifacts (CSR dtype check)

    rows = []
    with open(RUNS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("part") == "B" and r.get("phase") == "pilot":
                rows.append(r)
    rows.sort(key=lambda r: (r.get("timestamp") or "", r.get("run_id") or ""))
    arm_of = {r["arm_id"]: r for r in rows if r.get("arm_id")}

    def group_sparsity_of(row: dict):
        """S2: the achieved group sparsity (fraction of groups removed);
        stored on the arm's rows under method_params.group_sparsity_achieved
        (see lab3.sparsity.s2_group_sparsity)."""
        return (row.get("method_params") or {}).get("group_sparsity_achieved")

    def compact_of(row: dict) -> dict:
        p = (row.get("paths") or {}).get("compact") or {}
        art = p.get("artifact")
        if art and Path(art).exists() and p.get("csr_index_dtype"):
            try:  # verify the on-disk artifact actually uses the recorded dtype
                obj = torch.load(art, map_location="cpu", weights_only=False)
                for key, v in obj.items():
                    if isinstance(v, dict) and "indptr" in v:
                        assert str(v["indptr"].dtype) == p["csr_index_dtype"], (
                            f"artifact {art} indptr dtype {v['indptr'].dtype} != "
                            f"recorded {p['csr_index_dtype']}")
                        break
            except Exception:
                pass
        return p

    fcols = ["arch", "method", "target", "pattern", "achieved_sparsity",
             "achieved_param_sparsity", "val_acc", "test_acc", "dense_bytes",
             "gzip_bytes", "compressed_bytes", "est_macs_nonzero"]
    for path in ("dense_zeros", "compact"):
        for k in ("batch_1", "batch_256", "full_graph"):
            fcols += [f"{path}_{k}_median_ms", f"{path}_{k}_iqr_ms"]
    fcols += ["runtime_status", "run_id"]

    srows, lrows = [], []
    for r in rows:
        if r.get("stage") not in ("post_prune", "post_finetune"):
            continue
        p = r.get("paths") or {}
        p1 = p.get("dense_zeros") or {}
        p2 = compact_of(r)
        lat = {k: (med, iqr) for k, med, iqr in
               _lat_cells_nested(p, "dense_zeros") + _lat_cells_nested(p, "compact")}
        out = {
            "arch": r.get("architecture"),
            "method": r.get("method"),
            "target": r.get("target_sparsity"),
            "pattern": (f"{NM_PATTERNS[r['target_sparsity']][0]}:{NM_PATTERNS[r['target_sparsity']][1]}"
                        if r.get("method") == "s4"
                        else (round(group_sparsity_of(r), 6)
                              if r.get("method") == "s2" else "")),
            "achieved_sparsity": r.get("achieved_sparsity"),
            "achieved_param_sparsity": r.get("achieved_param_sparsity"),
            "val_acc": r.get("val_acc"),
            "test_acc": r.get("test_acc"),
            "dense_bytes": r.get("dense_bytes"),
            "gzip_bytes": r.get("gzip_bytes"),
            "compressed_bytes": r.get("compressed_bytes"),
            "est_macs_nonzero": r.get("est_macs_nonzero"),
            "runtime_status": f"dense_zeros={p1.get('runtime_status')}|compact={p2.get('runtime_status')}",
            "run_id": r.get("run_id"),
        }
        for k, (med, iqr) in lat.items():
            out[k + "_median_ms"] = med
            out[k + "_iqr_ms"] = iqr
        srows.append(out)
        for e in (r.get("per_layer") or []):
            lrows.append({"arm_id": r.get("arm_id"), "stage": r.get("stage"),
                          "layer": e["name"], "shape": "x".join(map(str, e["shape"])),
                          "numel": e["numel"], "nonzero": e["nonzero"],
                          "sparsity": e["sparsity"]})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PILOT_SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fcols, extrasaction="ignore")
        w.writeheader()
        for s in srows:
            w.writerow(s)
    with open(PILOT_PER_LAYER_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm_id", "stage", "layer", "shape",
                                          "numel", "nonzero", "sparsity"])
        w.writeheader()
        for s in lrows:
            w.writerow(s)
    return PILOT_SUMMARY_CSV, PILOT_PER_LAYER_CSV


# ---------------------------------------------------------------------------
# B3 confirm summary table (results/confirm_summary.csv)
# ---------------------------------------------------------------------------

def _lat_med(row: dict, path: str) -> float | None:
    """The primary latency key of a row's paths.<path>: batch_256 for
    mlp/cnn (frozen benchmark batch 256), full_graph for gcn."""
    key = "full_graph" if row.get("architecture") == "gcn" else "batch_256"
    p = (row.get("paths") or {}).get(path) or {}
    v = p.get(path + "_" + key) or {}
    return v.get("median_ms") if isinstance(v, dict) else None


def _latest_bench_ok(recs: list[dict], arch: str) -> dict | None:
    """The latest (file order) ok Part A bench record for (arch, seed 0)."""
    out = None
    for r in recs:
        if (r.get("part") == "A" and r.get("command") == "bench"
                and r.get("status") == "ok" and r.get("architecture") == arch
                and r.get("seed") == 0):
            out = r
    return out


def _latest_train_ok(recs: list[dict], arch: str, seed: int) -> dict | None:
    out = None
    for r in recs:
        if (r.get("part") == "A" and r.get("command") == "train"
                and r.get("status") == "ok" and r.get("architecture") == arch
                and r.get("seed") == seed):
            out = r
    return out


def _stats(vals: list[float]) -> tuple[float | None, float | None]:
    """(mean, sample std [ddof=1]) of a non-empty numeric list."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), None
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)))


def _f(x):
    return round(x, 6) if isinstance(x, float) else x


def _median(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def write_confirm_summary() -> Path:
    """B3: rebuild results/confirm_summary.csv — one row per (arch, method,
    target) — from the phase=pilot (seed 0) and phase=confirm (seeds 1-3)
    rows of results/runs.jsonl plus the Part A dense seed-0 and dense_ft
    (seeds 1-3) reference records. Deterministic and re-runnable; reflects
    the current log contents."""
    from .util import RUNS_JSONL
    import csv

    recs = []
    with open(RUNS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs.append(r)

    pilot = {r["arm_id"]: r for r in recs
             if r.get("part") == "B" and r.get("phase") == "pilot"
             and r.get("stage") in ("post_prune", "post_finetune")
             and r.get("status") == "ok"}
    # confirm ok post_finetune rows, latest per (arch, method, target, seed)
    # (file order = attempt order)
    conf = {}
    for r in recs:
        if (r.get("part") == "B" and r.get("phase") == "confirm"
                and r.get("stage") == "post_finetune" and r.get("status") == "ok"):
            conf[(r["architecture"], r["method"], r["target_sparsity"],
                  r["seed"])] = r

    sel = load_confirm_selection()
    fcols = [
        "arch", "method", "target",
        "pilot_seed0_post_prune_val_acc", "pilot_seed0_post_prune_test_acc",
        "pilot_seed0_post_finetune_val_acc", "pilot_seed0_post_finetune_test_acc",
        "achieved_sparsity",
        "seed1_post_finetune_val_acc", "seed1_post_finetune_test_acc",
        "seed2_post_finetune_val_acc", "seed2_post_finetune_test_acc",
        "seed3_post_finetune_val_acc", "seed3_post_finetune_test_acc",
        "post_prune_val_acc_mean", "post_prune_val_acc_std",
        "post_prune_test_acc_mean", "post_prune_test_acc_std",
        "post_finetune_val_acc_mean", "post_finetune_val_acc_std",
        "post_finetune_test_acc_mean", "post_finetune_test_acc_std",
        "dense_zeros_bytes", "gzip_bytes", "compact_bytes",
        "dense_zeros_latency_median_ms", "compact_latency_median_ms",
        "dense_seed0_val_acc", "dense_seed0_test_acc",
        "dense_seed0_dense_bytes", "dense_seed0_gzip_bytes",
        "dense_seed0_latency_median_ms",
        "dense_ft_val_acc_mean", "dense_ft_val_acc_std",
        "dense_ft_test_acc_mean", "dense_ft_test_acc_std",
        "dense_ft_dense_bytes", "dense_ft_gzip_bytes", "dense_ft_compact_bytes",
        "dense_ft_latency_median_ms",
    ]
    rows = []
    for entry in sel:
        a, m, t = entry["arch"], entry["method"], float(entry["selected_target"])
        # pilot seed-0 reference (post_prune + post_finetune)
        pp = pf = None
        for r in pilot.values():
            if r.get("architecture") == a and r.get("method") == m \
                    and abs(r.get("target_sparsity", -1) - t) < 1e-9:
                if r["stage"] == "post_prune":
                    pp = r
                elif r["stage"] == "post_finetune":
                    pf = r
        rows_seed = [conf.get((a, m, t, s)) for s in CONFIRM_SEEDS]
        # post_prune rows for seeds 1-3: look them up by (arch, method, target, seed)
        pp_by_seed = {}
        for r in recs:
            if (r.get("part") == "B" and r.get("phase") == "confirm"
                    and r.get("stage") == "post_prune" and r.get("status") == "ok"
                    and r.get("architecture") == a and r.get("method") == m
                    and abs(r.get("target_sparsity", -1) - t) < 1e-9
                    and r.get("seed") in CONFIRM_SEEDS):
                pp_by_seed[r["seed"]] = r
        ppv = [pp_by_seed[s]["val_acc"] for s in CONFIRM_SEEDS if s in pp_by_seed]
        ppt = [pp_by_seed[s]["test_acc"] for s in CONFIRM_SEEDS if s in pp_by_seed]
        pv = [r["val_acc"] for r in rows_seed if r]
        pt = [r["test_acc"] for r in rows_seed if r]
        m_ppv, s_ppv = _stats(ppv)
        m_ppt, s_ppt = _stats(ppt)
        m_pv, s_pv = _stats(pv)
        m_pt, s_pt = _stats(pt)
        bytes_dz = _f(_median([r.get("dense_bytes") for r in rows_seed if r]))
        bytes_gz = _f(_median([r.get("gzip_bytes") for r in rows_seed if r]))
        bytes_cp = _f(_median([r.get("compressed_bytes") for r in rows_seed if r]))
        lat_dz = _f(_median([_lat_med(r, "dense_zeros") for r in rows_seed if r]))
        lat_cp = _f(_median([_lat_med(r, "compact") for r in rows_seed if r]))
        dense0 = _latest_train_ok(recs, a, 0)
        bench0 = _latest_bench_ok(recs, a)
        lat0 = None
        if bench0:
            lat0 = _lat_med(bench0, "dense_zeros")
        # dense_ft control seeds 1-3 (B.6/B.7)
        df_rows = [conf.get((a, "dense_ft", 0.0, s)) for s in CONFIRM_SEEDS]
        dfv = [r["val_acc"] for r in df_rows if r]
        dft = [r["test_acc"] for r in df_rows if r]
        m_dfv, s_dfv = _stats(dfv)
        m_dft, s_dft = _stats(dft)
        df_bytes = _f(_median([r.get("dense_bytes") for r in df_rows if r]))
        df_gzip = _f(_median([r.get("gzip_bytes") for r in df_rows if r]))
        df_compact = _f(_median([r.get("compressed_bytes") for r in df_rows if r]))
        df_lat = _f(_median([_lat_med(r, "dense_zeros") for r in df_rows if r]))
        rows.append({
            "arch": a, "method": m, "target": t,
            "pilot_seed0_post_prune_val_acc": pp.get("val_acc") if pp else None,
            "pilot_seed0_post_prune_test_acc": pp.get("test_acc") if pp else None,
            "pilot_seed0_post_finetune_val_acc": pf.get("val_acc") if pf else None,
            "pilot_seed0_post_finetune_test_acc": pf.get("test_acc") if pf else None,
            "achieved_sparsity": pf.get("achieved_sparsity") if pf else None,
            "seed1_post_finetune_val_acc": rows_seed[0]["val_acc"] if rows_seed[0] else None,
            "seed1_post_finetune_test_acc": rows_seed[0]["test_acc"] if rows_seed[0] else None,
            "seed2_post_finetune_val_acc": rows_seed[1]["val_acc"] if rows_seed[1] else None,
            "seed2_post_finetune_test_acc": rows_seed[1]["test_acc"] if rows_seed[1] else None,
            "seed3_post_finetune_val_acc": rows_seed[2]["val_acc"] if rows_seed[2] else None,
            "seed3_post_finetune_test_acc": rows_seed[2]["test_acc"] if rows_seed[2] else None,
            "post_prune_val_acc_mean": m_ppv, "post_prune_val_acc_std": s_ppv,
            "post_prune_test_acc_mean": m_ppt, "post_prune_test_acc_std": s_ppt,
            "post_finetune_val_acc_mean": m_pv, "post_finetune_val_acc_std": s_pv,
            "post_finetune_test_acc_mean": m_pt, "post_finetune_test_acc_std": s_pt,
            "dense_zeros_bytes": bytes_dz, "gzip_bytes": bytes_gz,
            "compact_bytes": bytes_cp,
            "dense_zeros_latency_median_ms": lat_dz,
            "compact_latency_median_ms": lat_cp,
            "dense_seed0_val_acc": dense0.get("val_acc") if dense0 else None,
            "dense_seed0_test_acc": dense0.get("test_acc") if dense0 else None,
            "dense_seed0_dense_bytes": bench0.get("dense_bytes") if bench0 else None,
            "dense_seed0_gzip_bytes": bench0.get("gzip_bytes") if bench0 else None,
            "dense_seed0_latency_median_ms": lat0,
            "dense_ft_val_acc_mean": m_dfv, "dense_ft_val_acc_std": s_dfv,
            "dense_ft_test_acc_mean": m_dft, "dense_ft_test_acc_std": s_dft,
            "dense_ft_dense_bytes": df_bytes, "dense_ft_gzip_bytes": df_gzip,
            "dense_ft_compact_bytes": df_compact,
            "dense_ft_latency_median_ms": df_lat,
        })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIRM_SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fcols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return CONFIRM_SUMMARY_CSV


def _classify(e: Exception) -> str:
    msg = str(e).lower()
    if ("semi" in msg and "not supported" in msg) or "cuda" in msg:
        return f"unsupported_backend: {type(e).__name__}: {e}"
    if any(s in msg for s in ("nan", "inf", "converge", "overflow", "runtime error: value")):
        return f"numerical: {type(e).__name__}: {e}"
    return f"bug: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# B.4 two runtime paths, benchmarked with the frozen latency protocol
# ---------------------------------------------------------------------------

WARMUP, TIMED = 20, 200


def _time_protocol(f) -> dict:
    """Frozen protocol: inference_mode, eval(), 20 warmup / 200 timed iters,
    time.perf_counter_ns, median + IQR."""
    with torch.inference_mode():
        for _ in range(WARMUP):
            f()
        ns = []
        for _ in range(TIMED):
            s = time.perf_counter_ns()
            f()
            ns.append(time.perf_counter_ns() - s)
    ns = np.asarray(ns, dtype=np.float64) / 1e6
    q1, q2, q3 = np.percentile(ns, [25, 50, 75])
    return {"median_ms": float(q2), "iqr_ms": float(q3 - q1),
            "warmup": WARMUP, "iters": TIMED, "threads": torch.get_num_threads()}


def _bench_inputs(arch: str):
    from .bench import load_bench_input

    return load_bench_input(arch)


def _time_model(arch: str, model: nn.Module, prefix: str) -> dict:
    x, ei = _bench_inputs(arch)
    model.eval()
    if arch == "gcn":
        lat = _time_protocol(lambda: model(x, ei))
        lat["batch_size"] = "full_graph"
        return {prefix + "_full_graph": lat}
    out = {}
    for bs in (1, 256):
        lat = _time_protocol(lambda: model(x[:bs]))
        lat["batch_size"] = bs
        out[f"{prefix}_batch_{bs}"] = lat
    return out


def _forward5(arch: str, model: nn.Module) -> None:
    x, ei = _bench_inputs(arch)
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            if arch == "gcn":
                model(x, ei)
            else:
                model(x)


def _rss_subprocess(arch: str, arm_pt: str, kind: str, path: str = "") -> dict:
    """Fresh-subprocess peak RSS around loading the arm and running forwards.

    B1b: `path` distinguishes the two B.4 paths. dense_zeros loads the DENSE
    PARENT-SHAPED model (S2: scatter the rebuilt arm back into parent shapes
    with the removed groups zeroed — the key S2 contrast). compact rebuilds the
    compact runtime (CSR for S1/S3, the rebuilt model for S2, the reassembled
    dense fallback for S4)."""
    import json
    import subprocess
    import sys

    from .util import ROOT

    code = f"""
import json, sys
import torch
import psutil
from lab3.util import set_threads
from lab3.bench import load_bench_input
from lab3.models import build_model
from lab3.evaln import load_checkpoint
from lab3.compact import CompactModel
set_threads()
arch = {arch!r}
x, ei = load_bench_input(arch)
payload = torch.load({arm_pt!r}, map_location="cpu", weights_only=False)
path = {path!r}
method = payload.get("method")
proc = psutil.Process()
before = proc.memory_info().rss
if path == "dense_zeros":
    if method == "s2":
        # S2 dense_zeros: dense parent-shaped model, removed groups zeroed,
        # via the reversible scatter mapping (same zeros as the rebuilt model).
        parent, _ = load_checkpoint(arch, 0)
        plans = [{{"layer": k, "kind": v["kind"], "kept": v["kept"]}}
                 for k, v in payload["kept_groups"].items()]
        from lab3.sparsity import scatter_into_dense
        from lab3.prune import _s2_rebuild_from_state
        rebuilt = _s2_rebuild_from_state(arch, payload["state_dict"])
        dmap = scatter_into_dense(rebuilt, parent, plans)
        state = {{k: v.detach().clone() for k, v in parent.state_dict().items()}}
        state.update(dmap)
        m = build_model(arch)
        m.load_state_dict(state)
    else:
        m = build_model(arch)
        m.load_state_dict(payload["state_dict"])
else:  # compact
    if method in ("s1", "s3"):
        m = CompactModel(arch, payload["state_dict"], "csr")
    else:
        # s2: rebuilt model; s4/dense_ft: dense runtime (S4 fallback)
        m = build_model(arch)
        m.load_state_dict(payload["state_dict"])
m.eval()
peak = before
with torch.inference_mode():
    for _ in range(5):
        {f"m(x, ei)" if arch == "gcn" else "m(x)"}
        peak = max(peak, proc.memory_info().rss)
print(json.dumps({{"before": int(before), "peak": int(peak)}}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True, cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT / "code")},
        )
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        return {
            "rss_before_model_bytes": out["before"],
            "peak_rss_bytes": out["peak"],
            "model_rss_delta_bytes": out["peak"] - out["before"],
        }
    except Exception as e:
        return {"rss_error": str(e)}


def probe_semi_structured() -> str:
    """B.4: probe for a semi-structured CPU kernel. Returns 'ok' or
    'unsupported:<reason>'."""
    import torch.sparse as _sp

    if not hasattr(_sp, "to_sparse_semi_structured"):
        return "unsupported:torch.sparse.to_sparse_semi_structured not present"
    try:
        _sp.to_sparse_semi_structured(torch.ones(4, 4))
        return "ok"
    except Exception as e:
        return (f"unsupported:{type(e).__name__}: "
                f"torch.sparse.to_sparse_semi_structured does not accept CPU tensors")


def runtime_paths(arch: str, method: str, arm_pt: str,
                  target: float) -> tuple[dict, dict]:
    """B.4. Returns (dense_zeros, compact) path records: runtime_status,
    latency (frozen protocol), peak RSS (fresh subprocess), and the path's
    size fields. Never infers speedup from zero counts.

    B1b: the dense_zeros path is the DENSE PARENT-SHAPED model. For S1/S3/S4
    that is the masked (zeroed) dense model; for S2 it is the rebuilt model
    scattered back into parent shapes with the removed groups zeroed (the key
    S2 contrast: same zeros, different representation). The compact path is
    the genuinely compact runtime (CSR / rebuilt / reassembled N:M)."""
    payload = torch.load(arm_pt, map_location="cpu", weights_only=False)
    payload["arm_pt"] = arm_pt
    arch = payload["architecture"]
    method = payload["method"]
    COMPACT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- Path 1: dense_zeros ----------------
    if method == "s2":
        p_model, p_state, p_kind = _load_dense_zeros_s2(arch, payload)
    else:
        p_model = build_model(arch)
        p_model.load_state_dict(payload["state_dict"])
        p_model.eval()
        p_state = payload["state_dict"]
        p_kind = "dense"
    p1 = _bench_path(arch, p_model, p_state, p_kind, "dense_zeros", payload)
    p1["path"] = "dense_zeros"

    # Compact artifact (path 2 size): written per method before benching.
    p2_init = {"path": "compact"}
    _write_compact_artifact(method, arch, payload, p2_init)

    # ---------------- Path 2: compact ----------------
    c_model, c_state, c_kind = _load_compact(arch, method, payload)
    c_model.eval()
    p2 = _bench_path(arch, c_model, c_state, c_kind, "compact", payload)
    p2["path"] = "compact"
    # B1b: carry over the artifact path and compressed_bytes from the init dict.
    p2["artifact"] = p2_init.get("artifact")
    p2["compressed_bytes"] = p2_init.get("compressed_bytes")
    if "csr_index_dtype" in p2_init:
        p2["csr_index_dtype"] = p2_init["csr_index_dtype"]
        p2["csr_int64_bytes"] = p2_init["csr_int64_bytes"]
    _annotate_compact(p2, method, arch, payload)
    if method == "s4":
        status = probe_semi_structured()
        if status == "ok":
            pass  # (not reached on this platform)
        else:
            p2["runtime_status"] = status
            p2["note"] = ("S4 dense fallback: no semi-structured CPU kernel; "
                          "values reassembled from the packed N:M file")
    return p1, p2


def _write_compact_artifact(method: str, arch: str, payload: dict,
                            p2: dict) -> None:
    """Persist the genuinely-compact on-disk artifact (B.4 path 2) and record
    its path/size (compressed_bytes) on the compact path record. B1b: S1/S3
    store the CSR (out,in) with int32 indices (csr_index_dtype) + the default
    int64 layout size; S4 stores only kept values (row-major) + ONE uint8
    in-block position per kept value; S2 stores the rebuilt model."""
    short = payload["arm_id"].rsplit("-", 2)[0]
    artifact = COMPACT_DIR / f"{short}_{method}_{'csr' if method in ('s1','s3') else method}.pt"
    if method in ("s1", "s3"):
        model = build_model(arch)
        model.load_state_dict(payload["state_dict"])
        art = {"csr_index_dtype": str(CSR_INDEX_DTYPE), "csr_int64_bytes": 0}
        for name in eligible_state_keys(model):
            obj = model
            for part in name.split("."):
                obj = getattr(obj, part)
            w = _weight_of(obj).detach()
            # B1b: Conv2d weights are [out, in, k, k]; flatten to [out, in*k*k]
            # for CSR (matches the dense math: F.conv2d computes out = W_flat @ x_unfolded).
            w2d = w.flatten(1) if w.dim() == 4 else w
            # B2a fix: the saved CSR artifact uses CSR_INDEX_DTYPE (int32 when
            # supported, as probed in compact.py) — the same dtype the compact
            # runtime builds (compact._csr_components on CSR_INDEX_DTYPE). The
            # equivalent default int64 layout size is recorded separately as
            # csr_int64_bytes (computed from the same components; the int64
            # layout size depends only on element counts, not the stored dtype).
            indptr, indices, values = _csr_components(w2d, CSR_INDEX_DTYPE)
            size = list(w2d.shape)
            art[f"{name}.{_STATE_KEY[type(obj).__name__]}"] = {
                "values": values, "indptr": indptr, "indices": indices,
                "size": size}
            art["csr_int64_bytes"] += csr_int64_bytes(indptr, indices, values, size)
        torch.save(art, artifact)
        p2["csr_index_dtype"] = str(CSR_INDEX_DTYPE)
        p2["csr_int64_bytes"] = art["csr_int64_bytes"]
    elif method == "s4":
        # Rebuild the masked model to produce the uint8 N:M pack.
        model = build_model(arch)
        model.load_state_dict(payload["state_dict"])
        packed = s4_packed(model, payload["masks"], payload["target"])
        torch.save(packed, artifact)
    elif method == "s2":
        torch.save({k: v.detach().clone() for k, v in
                    payload["state_dict"].items()}, artifact)
    elif method == "dense_ft":
        torch.save({k: v.detach().clone() for k, v in
                    payload["state_dict"].items()}, artifact)
    p2["artifact"] = str(artifact)
    p2["compressed_bytes"] = os.path.getsize(artifact)


def _annotate_compact(p2: dict, method: str, arch: str, payload: dict) -> None:
    """B1b #12: for S4, record the THEORETICAL bit-packed size (labelled):
    values*4 + ceil(nnz*ceil(log2 M)/8) + biases."""
    if method == "s4":
        model = build_model(arch)
        model.load_state_dict(payload["state_dict"])
        packed = s4_packed(model, payload["masks"], payload["target"])
        total = 0
        for key, pk in packed.items():
            if key == "bias_bytes":
                continue
            if isinstance(pk, dict) and "values" in pk:
                total += s4_bitpacked_bytes_theoretical(pk)
        total += packed.get("bias_bytes", 0)
        p2["nm_bitpacked_bytes_theoretical"] = total
        p2["nm_bitpacked_theoretical_label"] = (
            "theoretical bit-packed size: kept FP32 values + bit-packed in-block "
            "positions (ceil(log2 M) bits each) + biases")


def _load_dense_zeros_s2(arch: str, payload: dict):
    """S2 dense_zeros: dense parent-shaped model with removed groups zeroed,
    rebuilt via the reversible scatter mapping from the arm checkpoint."""
    from .evaln import load_checkpoint
    parent, _ = load_checkpoint(arch, 0)
    plans = [{"layer": k, "kind": v["kind"], "kept": v["kept"]}
             for k, v in payload["kept_groups"].items()]
    # reconstruct a parent-shaped rebuilt model: the state_dict IS the rebuilt
    # model; scatter it back into the parent shapes.
    rebuilt = _s2_rebuild_from_state(arch, payload["state_dict"])
    m = s2_dense_zeros_model(arch, parent, rebuilt, plans)
    return m, m.state_dict(), "dense"


def _s2_rebuild_from_state(arch: str, sd: dict) -> nn.Module:
    """Build the smaller (S2) model from a rebuilt state_dict: each eligible
    linear/conv/GCNConv layer's weight+bias is resized to the checkpoint
    shapes (the dense parent's shapes differ). GCNConv's projection is its
    inner ``lin`` submodule (keys ``<conv>.lin.weight``); the conv's own
    ``bias`` key is the GCNConv bias (added after propagate)."""
    m = build_model(arch)
    for mname, mmod in m.named_modules():
        if type(mmod).__name__ == "GCNConv":
            lin_key = mname + ".lin.weight"
            if lin_key in sd:
                mmod.lin.weight = nn.Parameter(sd[lin_key].clone())
            if (mname + ".bias") in sd:
                mmod.bias = nn.Parameter(sd[mname + ".bias"].clone())
            continue
        wkey = mname + ".weight"
        if hasattr(mmod, "weight") and wkey in sd:
            mmod.weight = nn.Parameter(sd[wkey].clone())
            if hasattr(mmod, "bias") and (mname + ".bias") in sd:
                mmod.bias = nn.Parameter(sd[mname + ".bias"].clone())
    m.eval()
    return m


def _load_compact(arch: str, method: str, payload: dict):
    """Compact runtime module for the compact path (S1/S3 CSR, S2 rebuilt,
    S4 reassembled dense fallback)."""
    if method in ("s1", "s3"):
        return CompactModel(arch, payload["state_dict"], "csr"), \
            payload["state_dict"], "csr"
    if method == "s2":
        m = _s2_rebuild_from_state(arch, payload["state_dict"])
        return m, payload["state_dict"], "dense"
    if method == "s4":
        # Reassemble the dense fallback from the packed N:M file (kept values +
        # per-value in-block position); runtime is the dense model.
        m = build_model(arch)
        m.load_state_dict(payload["state_dict"])
        m.eval()
        return m, payload["state_dict"], "s4_fallback"
    if method == "dense_ft":
        m = build_model(arch)
        m.load_state_dict(payload["state_dict"])
        m.eval()
        return m, payload["state_dict"], "dense"
    raise ValueError(method)


def _arm_pt_for_rss(payload: dict) -> str:
    return payload.get("arm_pt", "")


def _bench_path(arch: str, model: nn.Module, state: dict, kind: str,
                path_name: str, payload: dict) -> dict:
    """Bench one path: size, latency (frozen protocol), peak RSS (subprocess)."""
    p = {"runtime_status": "ok"}
    p["dense_bytes"] = dense_bytes(model)
    p["raw_tensor_bytes"] = raw_tensor_bytes(model)
    p["gzip_bytes"] = gzip_bytes(model)
    p.update(_time_model(arch, model, path_name))
    p.update(_rss_subprocess(arch, _arm_pt_for_rss(payload), kind, path_name))
    return p


def dense_to_csr(w: torch.Tensor, index_dtype: torch.dtype | None = None) -> tuple:
    """Convert a 2-D dense tensor to CSR components (indptr, indices, values, size).
    `index_dtype` defaults to compact.CSR_INDEX_DTYPE (B2a: previously this
    helper hard-coded int64, so saved artifacts diverged from the runtime
    CSR dtype)."""
    from . import compact
    return compact.dense_to_csr(w, index_dtype)


def _module_for(model: nn.Module, state_key: str) -> nn.Module:
    obj = model
    for part in state_key.rpartition(".")[0].split("."):
        obj = getattr(obj, part)
    return obj
