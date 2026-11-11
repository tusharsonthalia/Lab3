"""Core sparsity machinery for Part B (predictions.md B.1-B.4, frozen).

Four methods, two families (separate code paths by construction):

* **Mask-based** (S1, S3, S4): produce a per-tensor *binary mask* (1 = keep,
  0 = pruned) over the DENSE-shaped parent. The model keeps the parent's
  shapes; pruned entries are exactly zero and the mask is re-applied after
  every fine-tuning optimizer step (B.1).
* **Structured surgery** (S2): physically rebuild a *smaller dense* model by
  copying surviving rows/columns/bias entries (B.2). Kept-group indices per
  layer are stored so the rebuilt model can be scattered back into dense
  parent shapes (zero elsewhere).

All methods are one-shot (no iterative pruning). Calibration subset (S3) is
the frozen splits/fmnist_calib_idx.npy (training-split images, training
labels) or, for GCN, the 140 training nodes on the full graph.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .models import build_model
from .util import CALIB_FILE, SPLITS_DIR

# ---------------------------------------------------------------------------
# Eligible tensors (B.1): the weight of every Linear, Conv2d and GCNConv layer,
# including the final classifier. Biases are never pruned.
# ---------------------------------------------------------------------------

_STATE_KEY = {
    "Linear": "weight",
    "Conv2d": "weight",
    "GCNConv": "lin.weight",  # torch_geometric holds the projection in lin
}


def eligible_state_keys(model: nn.Module) -> dict[str, str]:
    """{module name: state_dict key} for every eligible weight tensor."""
    from torch_geometric.nn import GCNConv

    out: dict[str, str] = {}
    for name, mod in model.named_modules():
        t = type(mod).__name__
        if t in _STATE_KEY:
            out[name] = f"{name}.{_STATE_KEY[t]}"
    return out


def eligible_weights(model: nn.Module) -> dict[str, nn.Parameter]:
    """{state_dict key: weight parameter} for every eligible tensor."""
    out = {}
    for name, key in eligible_state_keys(model).items():
        # walk the name to reach the parameter object
        obj = model
        for part in name.split("."):
            obj = getattr(obj, part)
        parts = _STATE_KEY[type(obj).__name__].split(".")
        w = obj
        for part in parts:
            w = getattr(w, part)
        out[key] = w
    return out


def dense_parent_model(architecture: str, state: dict) -> nn.Module:
    """Build the dense parent of `architecture` and load `state` into it.

    For S1/S3/S4 the pruned model IS the dense parent shape; for S2 callers
    use this to obtain the parent into which the rebuilt model is scattered."""
    model = build_model(architecture)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# S1 — global unstructured magnitude
# ---------------------------------------------------------------------------

def s1_masks(model: nn.Module, target: float) -> dict[str, torch.Tensor]:
    """Pool all eligible weights and zero exactly the target fraction with the
    smallest |w| (ties broken deterministically so the count is exact)."""
    w = eligible_weights(model)
    keys = sorted(w)
    all_abs = torch.cat([w[k].abs().flatten() for k in keys])
    numel = all_abs.numel()
    want_zero = int(round(target * numel))
    order = torch.argsort(all_abs, stable=True)          # ascending magnitude
    pruned = order[:want_zero]
    masks = {}
    pos = 0
    for k in keys:
        t = w[k]
        n = t.numel()
        in_range = (pruned >= pos) & (pruned < pos + n)
        m = torch.ones(n, dtype=torch.int8)
        m[pruned[in_range] - pos] = 0
        masks[k] = m.view_as(t)
        pos += n
    return masks


def _apply_mask(param: nn.Parameter, mask: torch.Tensor) -> None:
    with torch.no_grad():
        param.mul_(mask.to(param.dtype))


def make_masked_model(model: nn.Module, masks: dict[str, torch.Tensor]) -> nn.Module:
    """Apply binary masks in place (pruned entries become exactly zero).

    Implemented as zero-then-scatter so the result depends ONLY on the mask
    and the kept weight values: an exactly-zero kept entry is written back as
    exactly zero (no multiply rounding)."""
    for name, mod in model.named_modules():
        key = f"{name}.lin.weight" if type(mod).__name__ == "GCNConv" else f"{name}.weight"
        if key in masks:
            _apply_mask(mod.lin.weight if type(mod).__name__ == "GCNConv" else mod.weight,
                        masks[key])
    return model


def reapply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """B.1: re-apply the binary masks after every optimizer step."""
    make_masked_model(model, masks)


# ---------------------------------------------------------------------------
# S2 — structured group pruning (L1), physical rebuild
# ---------------------------------------------------------------------------

def _keep_count(n: int, s: float) -> int:
    """B.2: k = max(1, round((1 - s) * n))."""
    return max(1, int(round((1.0 - s) * n)))


def s2_plans(model: nn.Module, target: float) -> list[dict]:
    """Layer-wise group plans: {layer, groups, kept (indices), kind}.

    kinds:
      mlp_unit:    keep rows of an incoming Linear weight (rows of W, W: [out,in])
      cnn_filter:  keep output channels of a Conv2d (rows of weight [out,in,k,k])
      gcn_feature: keep columns of an incoming GCNConv weight (cols of W, W: [out,in])
    Each plan also carries `downstream` = list of (module name, axis) where the
    removed groups must be dropped from the *input* side (e.g. removing a conv2
    output channel also removes the matching FC1 input column).
    """
    from torch_geometric.nn import GCNConv

    plans = []
    if model.__class__.__name__ == "MLP":
        seq = model.net
        for name, lin in (("net.0", seq[0]), ("net.2", seq[2])):
            w = lin.weight  # [out, in]
            norms = w.abs().sum(dim=1)  # incoming weight row per hidden unit
            keep = _keep_count(w.shape[0], target)
            idx = torch.topk(norms, keep).indices.sort().values
            plans.append({"layer": name, "kind": "mlp_unit", "kept": idx})
    elif model.__class__.__name__ == "CNN":
        w1 = model.conv1.weight  # [16, 1, 5, 5]
        n1 = _keep_count(w1.shape[0], target)
        idx1 = torch.topk(w1.flatten(1).abs().sum(dim=1), n1).indices.sort().values
        plans.append({"layer": "conv1", "kind": "cnn_filter", "kept": idx1})
        w2 = model.conv2.weight  # [32, 16, 5, 5]
        n2 = _keep_count(w2.shape[0], target)
        idx2 = torch.topk(w2.flatten(1).abs().sum(dim=1), n2).indices.sort().values
        plans.append({"layer": "conv2", "kind": "cnn_filter", "kept": idx2})
        # removing conv2 channels removes the matching FC1 input columns
        plans[-1]["downstream"] = [("fc1", idx2)]
        wfc = model.fc1.weight  # [128, 800]
        nfc = _keep_count(wfc.shape[0], target)
        idxfc = torch.topk(wfc.abs().sum(dim=1), nfc).indices.sort().values
        plans.append({"layer": "fc1", "kind": "mlp_unit", "kept": idxfc})
    elif model.__class__.__name__ == "GCN":
        w1 = model.conv1.lin.weight  # [16, 1433]: hidden features = ROWS
        n1 = _keep_count(w1.shape[0], target)
        idx1 = torch.topk(w1.abs().sum(dim=1), n1).indices.sort().values
        plans.append({"layer": "conv1", "kind": "gcn_feature", "kept": idx1})
    else:
        raise ValueError(f"s2_plans: unsupported architecture {model.__class__.__name__}")
    return plans


def s2_rebuild(model: nn.Module, target: float):
    """Physically rebuild a smaller dense model. Returns (new_model, plans).

    Kept-group indices per layer are in `plans` — the reversible mapping
    (scatter_into_dense) uses them to map the rebuilt model back into dense
    parent shapes."""
    plans = s2_plans(model, target)
    by_layer = {p["layer"]: p for p in plans}
    from torch_geometric.nn import GCNConv

    def unit_keep(plan):
        return plan["kept"]

    new = build_model(model.__class__.__name__)
    if model.__class__.__name__ == "MLP":
        keep0 = unit_keep(by_layer["net.0"])
        keep2 = unit_keep(by_layer["net.2"])
        new.net[0] = nn.Linear(model.net[0].in_features, len(keep0))
        new.net[2] = nn.Linear(len(keep0), len(keep2))
        new.net[4] = nn.Linear(len(keep2), model.net[4].out_features)
        with torch.no_grad():
            new.net[0].weight.copy_(model.net[0].weight[keep0])
            new.net[0].bias.copy_(model.net[0].bias[keep0])
            new.net[2].weight.copy_(model.net[2].weight[keep2][:, keep0])
            new.net[2].bias.copy_(model.net[2].bias[keep2])
            new.net[4].weight.copy_(model.net[4].weight[:, keep2])
            new.net[4].bias.copy_(model.net[4].bias)
    elif model.__class__.__name__ == "CNN":
        k1 = unit_keep(by_layer["conv1"])
        k2 = unit_keep(by_layer["conv2"])
        kfc = unit_keep(by_layer["fc1"])
        new.conv1 = nn.Conv2d(model.conv1.in_channels, len(k1), model.conv1.kernel_size,
                              padding=model.conv1.padding)
        new.conv2 = nn.Conv2d(len(k1), len(k2), model.conv2.kernel_size,
                              padding=model.conv2.padding)
        # conv1 (k5,pad2): 28 -> pool -> 14; conv2 (k5): 14 -> 10 -> pool -> 5.
        # parent fc1 input = flattened conv2 output (C, H, W) row-major with C
        # fastest: col = (y * W2 + x) * C2 + c.
        W2 = 5
        C2 = model.conv2.out_channels
        # flattened conv2 output (C, h2, h2) row-major: col = (c * h2 + y) * h2 + x
        cols = [(int(c) * W2 + int(y)) * W2 + int(x)
                for c in k2 for y in range(W2) for x in range(W2)]
        new.fc1 = nn.Linear(len(cols), len(kfc))
        new.fc2 = nn.Linear(len(kfc), model.fc2.out_features)
        with torch.no_grad():
            new.conv1.weight.copy_(model.conv1.weight[k1])
            new.conv1.bias.copy_(model.conv1.bias[k1])
            new.conv2.weight.copy_(model.conv2.weight[k2][:, k1])
            new.conv2.bias.copy_(model.conv2.bias[k2])
            new.fc1.weight.copy_(model.fc1.weight[kfc][:, cols])
            new.fc1.bias.copy_(model.fc1.bias[kfc])
            new.fc2.weight.copy_(model.fc2.weight[:, kfc])
            new.fc2.bias.copy_(model.fc2.bias)
    elif model.__class__.__name__ == "GCN":
        kf = by_layer["conv1"]["kept"]
        # GCNConv(in, out).lin.weight is [out, in]. Hidden features are the
        # OUTPUT rows of conv1 (= input cols of conv2).
        new.conv1 = GCNConv(model.conv1.in_channels, int(len(kf)))
        new.conv2 = GCNConv(int(len(kf)), model.conv2.out_channels)
        with torch.no_grad():
            # conv1: hidden features are its OUTPUT dim -> keep rows kf.
            new.conv1.lin.weight.data = model.conv1.lin.weight[kf].clone()
            new.conv1.bias.data = model.conv1.bias[kf].clone()
            # conv2: hidden features are its INPUT dim -> keep cols kf.
            new.conv2.lin.weight.data = model.conv2.lin.weight[:, kf].clone()
            new.conv2.bias.data = model.conv2.bias.clone()
    else:
        raise ValueError("s2_rebuild: unsupported architecture")
    new.eval()
    return new, plans


def scatter_into_dense(rebuilt: nn.Module, parent: nn.Module,
                       plans: list[dict]) -> dict[str, torch.Tensor]:
    """Reversible mapping (B.4/impl 4): map the rebuilt (S2) model back into
    dense parent shapes — kept entries equal the rebuilt weights, zero elsewhere.
    Returns {parent state_dict key: dense tensor} for every *weight* tensor
    touched by the plans (biases are never pruned and are copied unchanged by
    the caller when needed)."""
    by_layer = {p["layer"]: p for p in plans}
    out: dict[str, torch.Tensor] = {}

    def dense_like(key: str):
        t = parent.state_dict()[key]
        return torch.zeros_like(t)

    def scatter_rows_cols(dst_full, src, rows, cols):
        """src (R x C) -> dst_full[rows, cols] (row-major scatter via advanced indexing)."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        cols = torch.as_tensor(cols, dtype=torch.long)
        dst_full[rows[:, None], cols[None, :]] = src

    cls = parent.__class__.__name__
    if cls == "MLP":
        keep0 = by_layer["net.0"]["kept"]
        keep2 = by_layer["net.2"]["kept"]
        w0 = dense_like("net.0.weight"); w0[keep0] = rebuilt.net[0].weight
        out["net.0.weight"] = w0
        b0 = torch.zeros_like(parent.state_dict()["net.0.bias"]); b0[keep0] = rebuilt.net[0].bias
        out["net.0.bias"] = b0
        w2 = dense_like("net.2.weight"); scatter_rows_cols(w2, rebuilt.net[2].weight, keep2, keep0)
        out["net.2.weight"] = w2
        b2 = torch.zeros_like(parent.state_dict()["net.2.bias"]); b2[keep2] = rebuilt.net[2].bias
        out["net.2.bias"] = b2
        w4 = dense_like("net.4.weight"); w4[torch.arange(w4.shape[0])[:, None], keep2[None, :]] = rebuilt.net[4].weight
        out["net.4.weight"] = w4
        b4 = torch.zeros_like(parent.state_dict()["net.4.bias"])
        b4.copy_(parent.state_dict()["net.4.bias"])  # classifier bias never pruned
        out["net.4.bias"] = b4
    elif cls == "CNN":
        k1 = by_layer["conv1"]["kept"]
        k2 = by_layer["conv2"]["kept"]
        kfc = by_layer["fc1"]["kept"]
        w1 = dense_like("conv1.weight"); w1[k1] = rebuilt.conv1.weight
        out["conv1.weight"] = w1
        b1 = torch.zeros_like(parent.state_dict()["conv1.bias"]); b1[k1] = rebuilt.conv1.bias
        out["conv1.bias"] = b1
        h2 = 5  # post-pool conv2 spatial (28 -> conv1,pad2 -> 28 -> pool -> 14 -> conv2 k5 -> 10 -> pool -> 5)
        C2 = parent.conv2.out_channels
        # flattened conv2 output (C, h2, h2) row-major: col = (c * h2 + y) * h2 + x
        cols = [(int(c) * h2 + int(y)) * h2 + int(x)
                for c in k2 for y in range(h2) for x in range(h2)]
        # conv2 weight [32, 16, 5, 5]: keep output channels k2 (dim 0) and input
        # channels k1 (dim 1). Row-major flatten of (c1, kh, kw).
        w2 = dense_like("conv2.weight")
        flat_w2 = w2.reshape(parent.conv2.weight.shape[0], -1)  # [32, 16*25]
        cols2 = [int(c) * 25 + int(kh) * 5 + int(kw)
                 for c in k1 for kh in range(5) for kw in range(5)]
        src2 = rebuilt.conv2.weight.reshape(len(k2), len(k1), -1).reshape(len(k2), -1)
        scatter_rows_cols(flat_w2, src2, k2, cols2)
        out["conv2.weight"] = w2
        b2 = torch.zeros_like(parent.state_dict()["conv2.bias"]); b2[k2] = rebuilt.conv2.bias
        out["conv2.bias"] = b2
        wfc = dense_like("fc1.weight"); scatter_rows_cols(wfc, rebuilt.fc1.weight, kfc, cols)
        out["fc1.weight"] = wfc
        bfc = torch.zeros_like(parent.state_dict()["fc1.bias"]); bfc[kfc] = rebuilt.fc1.bias
        out["fc1.bias"] = bfc
        wfc2 = dense_like("fc2.weight"); wfc2[torch.arange(wfc2.shape[0])[:, None], kfc[None, :]] = rebuilt.fc2.weight
        out["fc2.weight"] = wfc2
        bfc2 = torch.zeros_like(parent.state_dict()["fc2.bias"])
        bfc2.copy_(parent.state_dict()["fc2.bias"])  # classifier bias never pruned
        out["fc2.bias"] = bfc2
    elif cls == "GCN":
        kf = by_layer["conv1"]["kept"]
        # conv1: hidden features = its OUTPUT dim (rows); conv2: INPUT dim (cols).
        w1 = dense_like("conv1.lin.weight"); w1[kf] = rebuilt.conv1.lin.weight
        out["conv1.lin.weight"] = w1
        w2 = dense_like("conv2.lin.weight"); w2[torch.arange(w2.shape[0])[:, None], kf[None, :]] = rebuilt.conv2.lin.weight
        out["conv2.lin.weight"] = w2
        b2 = torch.zeros_like(parent.state_dict()["conv2.bias"])
        b2.copy_(parent.state_dict()["conv2.bias"])  # classifier bias never pruned
        out["conv2.bias"] = b2
    else:
        raise ValueError("scatter_into_dense: unsupported architecture")
    return out


# ---------------------------------------------------------------------------
# S3 — first-order saliency (|w * grad|), global, calibration subset
# ---------------------------------------------------------------------------

def s3_gradients(model: nn.Module, loss_fn: Callable[[nn.Module], torch.Tensor]) -> dict:
    """One gradient of the mean cross-entropy loss on the whole calibration
    subset, on the DENSE parent in eval mode (no dropout). Returns
    {state key: dL/dw} for every eligible weight."""
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    # GCN message passing caches the normalized edge index on the module;
    # a parent that saw a train-mode forward would carry a STALE cache into
    # the calibration gradient (and later evals) -> clear it.
    for mod in model.modules():
        for attr in list(vars(mod)):
            if attr.startswith("_cached"):
                setattr(mod, attr, None)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model)
    loss.backward()
    if was_training:
        model.train()
    grads = {}
    missing = []
    seen = set()
    for name, mod in model.named_modules():
        t = type(mod).__name__
        if t not in _STATE_KEY:
            continue
        key = f"{name}.{_STATE_KEY[t]}"
        # PyG registers some sub-modules under two parent paths, so
        # named_modules() can yield the same state key twice.
        if key in seen:
            continue
        seen.add(key)
        w = mod
        for part in _STATE_KEY[t].split("."):
            w = getattr(w, part)
        if w.grad is None:
            missing.append(key)
        else:
            grads[key] = w.grad.detach().clone()
            w.grad = None
    if missing:
        raise RuntimeError(f"S3: no gradient for {missing}; "
                           "every eligible tensor must receive gradient "
                           "(check the loss covers the full model output)")
    return grads


def s3_masks(model: nn.Module, grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pool |w * grad| across all eligible weights; zero the lowest-scoring
    fraction (target applied by the caller via _select_fraction_mask)."""
    w = eligible_weights(model)
    scores = torch.cat([(w[k] * grads[k]).abs().flatten() for k in w])
    return {"__pool__": scores}


def _select_fraction_mask(w: dict[str, nn.Parameter], pool_scores: torch.Tensor,
                          target: float,
                          pool_order: list[str]) -> dict[str, torch.Tensor]:
    """Zero the `target` fraction with the lowest pooled scores; ties broken
    deterministically so the global count is exact (within one weight)."""
    all_n = sum(t.numel() for t in w.values())
    k_keep = int(round((1.0 - target) * all_n))
    # rank by score ascending; the first (all_n - k_keep) are pruned.
    order = torch.argsort(pool_scores, stable=True)
    n_prune = all_n - k_keep
    pruned = order[:n_prune]
    # map flat indices back to per-tensor
    masks = {}
    pos = 0
    for key in pool_order:
        t = w[key]
        n = t.numel()
        in_range = (pruned >= pos) & (pruned < pos + n)
        local = torch.ones(n, dtype=torch.int8)
        local[pruned[in_range] - pos] = 0
        masks[key] = local.view_as(t)
        pos += n
    return masks


def s3_masks_global(model: nn.Module, target: float,
                    loss_fn: Callable[[nn.Module], torch.Tensor]) -> dict[str, torch.Tensor]:
    """Full S3: calibration gradient + pooled |w*grad| ranking -> binary masks."""
    grads = s3_gradients(model, loss_fn)
    w = eligible_weights(model)
    order = sorted(w)
    scores = torch.cat([(w[k] * grads[k]).abs().flatten() for k in order])
    return _select_fraction_mask(w, scores, target, order)


# ---------------------------------------------------------------------------
# S4 — N:M semi-structured (blocks along the reduction/input axis)
# ---------------------------------------------------------------------------

NM_PATTERNS = {0.25: (3, 4), 0.5: (2, 4), 0.75: (1, 4), 0.9: (1, 10)}


def nm_pattern_for_target(target: float) -> tuple[int, int]:
    for t, p in NM_PATTERNS.items():
        if abs(target - t) < 1e-9:
            return p
    raise ValueError(f"target {target} not in the frozen S4 patterns {NM_PATTERNS}")


def _nm_keep_mask_row(row_abs: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Keep mask for one row (length L) under N:M with the frozen partial-block
    rule: full blocks keep the N largest; a final partial block of length r
    keeps ceil(N*r/M).

    B1b: the kept set within each block is the N LARGEST-magnitude positions
    (the original S4 semantics), i.e. ``topk(n_keep).indices`` — not the first
    ``n_keep`` columns. This is frozen; the compact N:M reassembly must
    therefore store the actual in-block column of each kept value (one uint8
    per kept value), which is what ``s4_packed`` does and ``reassemble_packed``
    inverts."""
    L = row_abs.numel()
    keep = torch.zeros(L, dtype=torch.int8)
    n_full, r = divmod(L, m)
    for b in range(n_full + (1 if r else 0)):
        s = b * m if b < n_full else n_full * m
        blen = m if b < n_full else r
        blk = row_abs[s:s + blen]
        n_keep = n if b < n_full else math.ceil(n * blen / m)
        if n_keep <= 0:
            continue
        if n_keep >= blen:
            keep[s:s + blen] = 1
            continue
        idx = blk.topk(n_keep).indices
        keep[s:s + blen].index_fill_(0, idx, 1)
    return keep


def s4_masks(model: nn.Module, target: float) -> dict[str, torch.Tensor]:
    """N:M masks along each row's reduction (input) dimension.

    Layouts: Linear/GCNConv weights are [out, in] (in = reduction axis = dim 1);
    Conv2d weights are flattened to [out, in*k*k] (dim 1 is the reduction axis).
    Masks are returned in the NATIVE weight shape (Conv2d: [out,in,k,k])."""
    n, m = nm_pattern_for_target(target)
    masks = {}
    for key, t in eligible_weights(model).items():
        mod_name, _, param = key.rpartition(".")
        # find the module to know its type
        obj = model
        for part in mod_name.split("."):
            obj = getattr(obj, part)
        tname = type(obj).__name__
        if tname == "Conv2d":
            wt = t.flatten(1)  # [out, in*k*k]
            flat = torch.stack([_nm_keep_mask_row(wt[i].abs(), n, m) for i in range(wt.shape[0])])
            masks[key] = flat.view_as(t)
        else:
            wt = t  # [out, in]
            flat = torch.stack([_nm_keep_mask_row(wt[i].abs(), n, m) for i in range(wt.shape[0])])
            masks[key] = flat
    return {k: v.to(torch.int8) for k, v in masks.items()}


def nm_block_check(mask_row: torch.Tensor, n: int, m: int) -> list[int]:
    """Nonzero count per block of a mask row (full blocks + final partial)."""
    L = mask_row.numel()
    n_full, r = divmod(L, m)
    counts = [int(mask_row[b * m:(b + 1) * m].sum()) for b in range(n_full)]
    if r:
        counts.append(int(mask_row[n_full * m:].sum()))
    return counts


def s4_packed(model: nn.Module, masks: dict[str, torch.Tensor],
              target: float) -> dict[str, dict]:
    """Packed N:M artifact (B.4 path 2 for S4), B1b format: per weight tensor,
    only the kept values (row-major) plus ONE uint8 in-block position per kept
    value. Row and block are implied by the fixed N:M pattern and the frozen
    partial-block ceil rule (per-row keep counts). Returns per-tensor packs
    (kept for testing) plus a ``bias_bytes`` entry (sum of state-dict bias
    tensor payload bytes, used for the theoretical size)."""
    n, m = nm_pattern_for_target(target)
    out = {}
    bias_bytes = 0
    for name, mod in model.named_modules():
        tname = type(mod).__name__
        if tname in ("Linear", "Conv2d"):
            if mod.bias is not None:
                bias_bytes += int(mod.bias.numel() * mod.bias.element_size())
        elif tname == "GCNConv":
            if mod.bias is not None:
                bias_bytes += int(mod.bias.numel() * mod.bias.element_size())
    out["bias_bytes"] = bias_bytes
    for key, t in eligible_weights(model).items():
        mask = masks[key]
        obj = model
        for part in key.rpartition(".")[0].split("."):
            obj = getattr(obj, part)
        tname = type(obj).__name__
        wt = t.flatten(1) if tname == "Conv2d" else t
        mt = mask.flatten(1)
        R, L = wt.shape
        r = L % m
        n_full = L // m
        n_blocks = n_full + (1 if r else 0)
        keep_per_block = [n] * n_full
        if r:
            keep_per_block.append(math.ceil(n * r / m))
        per_row = sum(keep_per_block)
        vals, in_blk = [], []
        for row in range(R):
            for b in range(n_blocks):
                s = b * m if b < n_full else n_full * m
                blen = m if b < n_full else r
                nz = torch.nonzero(mt[row, s:s + blen]).flatten()
                # B1b: store the ACTUAL in-block column (j) of each kept value,
                # NOT its rank within the block. The kept set is the N
                # largest-magnitude positions (arbitrary columns within the
                # block), so the column cannot be recovered from the rank alone
                # — it must be stored explicitly (one uint8 per kept value).
                for j in nz:
                    vals.append(float(wt[row, s + int(j)].item()))
                    in_blk.append(int(j))
        out[key] = {
            "values": torch.tensor(vals, dtype=wt.dtype),
            "in_block": torch.tensor(in_blk, dtype=torch.uint8),
            "n": n, "m": m,
            "row_len": int(L),
            "shape": list(t.shape),
        }
    return out
