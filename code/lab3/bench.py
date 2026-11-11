"""Frozen latency protocol (Part A.1) + analytical MAC estimates.

Latency: torch.inference_mode(), model in eval(). 20 warm-up and 200 timed
iterations, time.perf_counter_ns. Report median and IQR (q3 - q1).
The fixed benchmark input is the first 256 TEST images (inputs only, saved once
to a small file, results/bench/fmnist_bench_input.pt) — NOT training images.
Batch sizes 1 and 256 for Fashion-MNIST; one full-graph forward for Cora.

Memory: the RSS subprocess does not load the 55k training tensor. It loads the
small fixed bench input and the checkpoint, and reports
rss_before_model_bytes (after imports + input load), peak_rss_bytes and
model_rss_delta_bytes = peak - before.

MAC counting (computed from the ACTUAL model's layer shapes — works for any
Linear / Conv2d / GCNConv sizes; nothing is hard-coded). We trace one forward
pass with hooks, and for every weight-bearing module (Linear, Conv2d, GCNConv)
count the MACs its weight performs per single sample:

- Linear(in, out):   each weight element multiplies one input per token;
  MACs = numel(W) * tokens (tokens = 1 here, i.e. per image / per node).
- Conv2d(cin, cout, k): each weight element is applied at every output-pixel
  position; MACs = numel(W) * H_out * W_out per image.
- GCNConv(in, out):   GCNConv computes X @ W ONCE per node (not per edge) and
  then aggregates over the augmented adjacency A' = A + I (self-loops). Per
  layer:
      projection   = N * in * out       (the dense-X weight product)
      aggregation  = (E + N) * out      (weighted sum over the E+N A' entries)
  where N = node count, E = directed edges (A' adds N self-loops).
  The degree-normalization factors D^{1/2} are elementwise scales and add
  ~2 * out * (E+N) multiplications; these are reported separately under
  `gcn_norm_macs` and are NOT part of est_macs.

Two MAC fields are emitted for every model:
- est_macs          : MACs assuming ALL weights are nonzero (dense convention).
                      For GCN this is the dense-X total above.
- est_macs_nonzero  : the same count but only for weight elements that are
                      actually nonzero in the loaded checkpoint (this is what
                      matters once Part B makes the model physically sparse).
GCN additionally reports:
- est_macs_sparse_input : est_macs with the layer-1 projection counted as
                      nnz(X)*out instead of N*in*out, because Cora's features
                      are ~1.3% dense (sparse-input convention).

For the dense seed-0 baselines every weight is nonzero, so
est_macs == est_macs_nonzero.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .data import fmnist_split_tensors, load_cora
from .evaln import load_checkpoint
from .models import count_params, dense_bytes, gzip_bytes, raw_tensor_bytes
from .util import (
    BENCH_DIR,
    ROOT,
    base_record,
    prior_bench_run_ids,
    run_wall_timer,
    set_threads,
    write_run_record,
)

WARMUP = 20
TIMED = 200
BATCH_SIZES = [1, 256]
FMNIST_BENCH_INPUT = BENCH_DIR / "fmnist_bench_input.pt"
CORAL_BENCH_INPUT = BENCH_DIR / "cora_bench_input.pt"


# --------------------------------------------------------------------------
# Fixed benchmark inputs (inputs only). Saved once, reused by the main process
# and the RSS subprocess so neither ever loads the 55k training tensor.
# --------------------------------------------------------------------------

def _ensure_fmnist_bench_input() -> torch.Tensor:
    """First 256 TEST images, inputs only (no labels). Saved once."""
    if FMNIST_BENCH_INPUT.exists():
        return torch.load(FMNIST_BENCH_INPUT, map_location="cpu", weights_only=True)
    _, _, _, _, te_x, _, _, _ = fmnist_split_tensors()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(te_x[:256].contiguous(), FMNIST_BENCH_INPUT)
    return torch.load(FMNIST_BENCH_INPUT, map_location="cpu", weights_only=True)


def _ensure_cora_bench_input() -> dict:
    """Full-graph Cora features + edge_index (inputs only). Saved once."""
    if CORAL_BENCH_INPUT.exists():
        return torch.load(CORAL_BENCH_INPUT, map_location="cpu", weights_only=False)
    data = load_cora()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"x": data.x.contiguous(), "edge_index": data.edge_index.contiguous()}
    torch.save(payload, CORAL_BENCH_INPUT)
    return torch.load(CORAL_BENCH_INPUT, map_location="cpu", weights_only=False)


def load_bench_input(architecture: str):
    """Return the fixed benchmark input: (x, edge_index). edge_index is None for
    fmnist; x is the (256,1,28,28) test-image batch for fmnist and the full
    (N,in) Cora feature matrix for gcn."""
    if architecture == "gcn":
        d = _ensure_cora_bench_input()
        return d["x"], d["edge_index"]
    return _ensure_fmnist_bench_input(), None


# --------------------------------------------------------------------------
# MAC estimation from actual layer shapes (no hard-coded numbers).
# --------------------------------------------------------------------------

def _weight_of(mod):
    """The parameter a module multiplies the input by (GCNConv holds it in a
    nested Linear as `lin.weight`; Linear/Conv2d hold it as `.weight`)."""
    w = getattr(mod, "weight", None)
    if w is None and hasattr(mod, "lin"):
        w = mod.lin.weight
    return w


def _mac_entries(model, x, edge_index=None):
    """Run one forward pass, recording (weight numel, weight nnz, out_ch,
    tokens) for each Linear / Conv2d / GCNConv in forward order.

    B1b: `tokens` is always the SINGLE-SAMPLE convention (1 image / 1 node
    spatial position), independent of the batch size of `x` — callers pass
    the actual forward input for shape discovery, but per-image MACs are what
    `est_macs`/`est_macs_nonzero` report (full graph for GCN, N = node count)."""
    from torch_geometric.nn import GCNConv

    model.eval()
    entries = []
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) or type(mod).__name__ == "GCNConv":
            def _hook(m, inp, out, _mod=mod, _name=name):
                w = _weight_of(_mod)
                if w.dim() == 4:  # Conv2d: per image = 1 per output pixel
                    tokens = int(out.shape[-2] * out.shape[-1])
                elif type(_mod).__name__ == "GCNConv":
                    # GCN: per full graph = N nodes (forward is one graph)
                    tokens = int(out.shape[0])
                else:            # Linear: per image = 1 token
                    tokens = 1
                entries.append({
                    "name": _name,
                    "type": type(_mod).__name__,
                    "numel": int(w.numel()),
                    "nnz": int(w.count_nonzero()),
                    "out_ch": int(out.shape[-1]),
                    "tokens": tokens,
                })
            hooks.append(mod.register_forward_hook(_hook))
    try:
        with torch.inference_mode():
            if edge_index is not None:
                model(x, edge_index)
            else:
                model(x[:1])
    finally:
        for h in hooks:
            h.remove()
    return entries


def estimate_macs(model, x, edge_index=None, x_nnz: int | None = None) -> dict:
    """Analytical MAC estimate computed from the model's actual layer shapes.

    Returns dict with est_macs, est_macs_nonzero, gcn_norm_macs and
    est_macs_sparse_input (GCN only, else None).
    """
    entries = _mac_entries(model, x, edge_index)

    is_gcn = edge_index is not None
    if is_gcn:
        N = int(x.shape[0])
        E = int(edge_index.shape[1])          # directed edges (no self-loops yet)
        E_plus_N = E + N                      # A' = A + I
        norm_macs = 0
    else:
        N = E_plus_N = None
        norm_macs = 0

    est = 0
    est_nz = 0
    est_sparse = None  # computed on first GCNConv layer, else None (non-GCN)
    for e in entries:
        if e["type"] == "GCNConv":
            proj = e["numel"] * e["tokens"]              # N*in*out (dense-X)
            proj_nz = e["nnz"] * e["tokens"]             # nonzero weights only
            agg = E_plus_N * e["out_ch"]
            if est_sparse is None and x_nnz is not None:
                est_sparse = (x_nnz * e["out_ch"]) + agg
            else:
                est_sparse += proj + agg
            est += proj + agg
            est_nz += proj_nz + agg
            norm_macs += 2 * e["out_ch"] * E_plus_N
        else:  # Linear / Conv2d
            est += e["numel"] * e["tokens"]
            est_nz += e["nnz"] * e["tokens"]
            if est_sparse is not None:
                est_sparse += e["numel"] * e["tokens"]

    return {
        "est_macs": est,
        "est_macs_nonzero": est_nz,
        "est_macs_sparse_input": (est_sparse if is_gcn and est_sparse is not None else None),
        "gcn_norm_macs": norm_macs,
    }


# --------------------------------------------------------------------------
# Timing (frozen protocol) and peak-RSS subprocess.
# --------------------------------------------------------------------------

def _time_batch(model, x, batch_size: int) -> dict:
    """Frozen protocol: inference_mode, eval(), 20 warm-up / 200 timed iterations.
    A pre-allocated (batch_size, ...) view of the fixed bench input is used each
    iteration so the timed region is the model forward pass only."""
    model.eval()
    with torch.inference_mode():
        xb = x[:batch_size] if len(x) >= batch_size else x
        for _ in range(WARMUP):
            model(xb)
        ns = []
        for _ in range(TIMED):
            t0 = time.perf_counter_ns()
            model(xb)
            ns.append(time.perf_counter_ns() - t0)
    ns = np.asarray(ns, dtype=np.float64) / 1e6
    q1, q2, q3 = np.percentile(ns, [25, 50, 75])
    return {"median_ms": float(q2), "iqr_ms": float(q3 - q1)}


def _bench_peak_rss(architecture: str, seed: int) -> dict:
    """RSS measured in a fresh subprocess that (1) imports + loads the small
    fixed bench input, (2) records rss_before_model_bytes, (3) loads the
    checkpoint and runs a few forwards, recording peak_rss_bytes. Returns
    (rss_before, peak)."""
    code = f"""
import json, psutil, sys
import torch
from lab3.util import set_threads
from lab3.bench import load_bench_input
from lab3.evaln import load_checkpoint
set_threads()
x, ei = load_bench_input("{architecture}")
proc = psutil.Process()
before = proc.memory_info().rss
m, _ = load_checkpoint("{architecture}", {seed})
m.eval()
peak = before
with torch.inference_mode():
    for _ in range(5):
        if "{architecture}" == "gcn":
            m(x, ei)
        else:
            m(x[:256])
        peak = max(peak, proc.memory_info().rss)
print(json.dumps({{"before": int(before), "peak": int(peak)}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "code")},
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


def bench(architecture: str, seed: int = 0) -> dict:
    set_threads()
    t0 = time.perf_counter()
    rec = base_record(architecture, command="bench")
    rec["seed"] = seed
    try:
        model, meta = load_checkpoint(architecture, seed)
        # Linkage: parent run is the training run that produced this checkpoint.
        rec["parent_run_id"] = meta.get("run_id")
        rec["supersedes"] = prior_bench_run_ids(architecture, seed)

        rec["total_params"] = count_params(model)
        rec["nonzero_params"] = count_params(model)
        rec["dense_bytes"] = dense_bytes(model)
        rec["raw_tensor_bytes"] = raw_tensor_bytes(model)
        # P1: gzip_bytes = gzip-9 size of the dense state_dict file, reported for
        # EVERY run (dense included). compressed_bytes is reserved for the
        # compact-export artifact (Part B path 2); null for dense.
        rec["gzip_bytes"] = gzip_bytes(model)
        rec["compressed_bytes"] = None
        rec["best_epoch"] = meta.get("best_epoch")
        rec["val_acc"] = meta.get("val_acc")

        x, ei = load_bench_input(architecture)
        model.eval()

        if architecture == "gcn":
            with torch.inference_mode():
                for _ in range(WARMUP):
                    model(x, ei)
                ns = []
                for _ in range(TIMED):
                    t_start = time.perf_counter_ns()
                    model(x, ei)
                    ns.append(time.perf_counter_ns() - t_start)
            ns = np.asarray(ns, dtype=np.float64) / 1e6
            q1, q2, q3 = np.percentile(ns, [25, 50, 75])
            rec["latency"] = {
                "full_graph": {
                    "median_ms": float(q2),
                    "iqr_ms": float(q3 - q1),
                    "warmup": WARMUP,
                    "iters": TIMED,
                    "threads": torch.get_num_threads(),
                    "batch_size": "full_graph",
                }
            }
            macs = estimate_macs(model, x, ei, x_nnz=int(x.count_nonzero()))
        else:
            lat = {}
            for bs in BATCH_SIZES:
                lat[f"batch_{bs}"] = _time_batch(model, x, bs)
                lat[f"batch_{bs}"].update({"warmup": WARMUP, "iters": TIMED,
                                           "threads": torch.get_num_threads(),
                                           "batch_size": bs})
            rec["latency"] = lat
            macs = estimate_macs(model, x[:1], None)

        rec["est_macs"] = macs["est_macs"]
        rec["est_macs_nonzero"] = macs["est_macs_nonzero"]
        rec["est_macs_sparse_input"] = macs["est_macs_sparse_input"]
        rec["gcn_norm_macs"] = macs["gcn_norm_macs"]

        rss = _bench_peak_rss(architecture, seed)
        rec["rss_before_model_bytes"] = rss["before"]
        rec["peak_rss_bytes"] = rss["peak"]
        rec["model_rss_delta_bytes"] = rss["peak"] - rss["before"]

        rec = run_wall_timer(rec, t0)
        write_run_record(rec)
        return rec
    except Exception as e:
        rec = run_wall_timer(rec, t0)
        rec["status"] = "failed"
        rec["reason"] = f"{type(e).__name__}: {e}"
        write_run_record(rec)
        raise
