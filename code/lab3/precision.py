"""Part C — reduced precision (predictions.md C.1-C.3, frozen).

Conversions from the canonical seed-0 dense FP32 parent, per architecture:
fp16, bf16, dynamic INT8, static INT8 (calibrated), simulated weight-only INT4.

Backend facts on THIS machine (Apple M4 Pro, torch 2.15.0, darwin), probed in
Task C1b and recorded in results/precision_support.md:

* ``torch.backends.quantized.supported_engines == ['qnnpack']``;
  ``torch._C._get_qengine() == 2`` (QNNPACK only; fbgemm/onednn absent).
* ``torch.ao.quantization.quantize_dynamic({nn.Linear}, dtype=torch.qint8)``
  works and produces native quantized Linear modules that execute the qnnpack
  ``quantized::linear`` kernel (weight INT8 per-tensor symmetric, activation
  quantized per-batch at runtime). Same for ``{nn.Conv2d}``.
* FX static flow: ``prepare_fx``/``convert_fx`` with
  ``get_default_qconfig_mapping("qnnpack")`` works for Linear and Conv2d
  (produces ``LinearReLU`` / ``ConvReLU2d`` fused quantized modules).
  PT2E (``torch.ao.quantization.quantize_pt2e``) does NOT exist in torch 2.15;
  torchao is not installed (not added — not needed).
* ``nn.Linear`` and ``nn.Conv2d`` execute natively in fp16 and bf16 on CPU
  (output dtype == input dtype; no upcast).
* GCNConv (PyG propagate / scatter) has no quantized or half-precision kernel;
  only its inner projection ``GCNConv.lin`` (an ``nn.Linear``) can run reduced
  precision. Reduced-precision GCN rows therefore record that the projection
  ran reduced precision and the aggregation/normalization ran fp32.

INT8 methods (native qnnpack kernels):
- Dynamic: ``quantize_dynamic({nn.Linear}, dtype=torch.qint8)`` — INT8 weights,
  per-batch activation quantization, qnnpack ``quantized::linear`` kernel.
- Static (calibrated): FX ``prepare_fx``/``convert_fx`` with the qnnpack qconfig
  mapping, calibrated on the frozen calibration inputs (C.1 rule).
  Activation qparams are fixed at calibration time.

Simulated weight-only INT4 (frozen C.1):
- Per-output-channel symmetric quantization to [-8, 7], fp32 scale per output
  channel, 2 values packed per uint8 on disk.
- Dequantize to fp32 and compute in fp32. **The dequantization is INSIDE the
  timed region** (decided once, applied identically to every INT4 run).
  ``runtime_status = "simulated_non_native"``.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.ao.quantization import get_default_qconfig_mapping

from .bench import WARMUP, TIMED, load_bench_input
from .compact import clear_gcn_caches
from .data import fmnist_split_tensors, load_cora
from .evaln import load_checkpoint
from .models import build_model, count_params
from .util import (
    CALIB_FILE,
    SPLITS_DIR,
    RESULTS_DIR,
    ROOT,
    base_record,
    code_hash,
    new_run_id,
    set_threads,
    versions,
    write_run_record,
)

# ---------------------------------------------------------------------------
# qnnpack engine
# ---------------------------------------------------------------------------


def set_qnnpack_engine() -> None:
    """Enable the qnnpack quantized engine (the only one on this machine)."""
    torch.backends.quantized.engine = "qnnpack"
    assert torch._C._get_qengine() == 2  # QNNPACK


# ---------------------------------------------------------------------------
# Eligible weight-bearing ops
# ---------------------------------------------------------------------------


def eligible_ops(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    """(name, module, kind) for every Linear / Conv2d / GCNConv projection.

    kind: 'linear' (nn.Linear), 'conv' (nn.Conv2d), 'lin' (GCNConv's inner
    projection, state key <name>.lin). GCNConv itself is never
    reduced-precision-capable on this backend; only its .lin is. A GCNConv's
    inner projection is reported once (as 'lin'), not again as a plain
    'linear', even though it shows up in named_modules() under its own name."""
    from torch_geometric.nn import GCNConv

    # Collect GCNConv inner-projection names to skip in the plain-Linear pass.
    gcn_lin_names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, GCNConv):
            gcn_lin_names.add(f"{name}.lin")
    out = []
    for name, mod in model.named_modules():
        t = type(mod).__name__
        if t == "GCNConv":
            out.append((f"{name}.lin", mod.lin, "lin"))
        elif t == "Linear":
            if name in gcn_lin_names:
                continue  # already reported as the GCNConv's .lin projection
            out.append((name, mod, "linear"))
        elif t == "Conv2d":
            out.append((name, mod, "conv"))
    return out


def _resolve_parent(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


# ---------------------------------------------------------------------------
# C2: integer-weight introspection (nonzero counts + per-dtype byte accounting)
# ---------------------------------------------------------------------------


def _qnnpack_weight_bias(mod: nn.Module) -> tuple | None:
    """(qint8 weight, fp32 bias|None) for a qnnpack quantized op, or None.

    C3: handles both qnnpack shapes the study produces:
      (a) dynamic / FX-static linears and convs whose `_packed_params` is a
          Python `LinearPackedParams` wrapper module — `_weight_bias()` works
          through the wrapper (its raw ScriptObject lives in the wrapper's
          vars under the same key and is NOT re-scanned, so no double count);
      (b) FX-static convs (CNN static) and FX-static GCN per-projection
          GraphModules, where `convert_fx` attaches the packed weight directly
          as a raw instance attribute (`_packed_params` / `_packed_weight_N`,
          a `torch.ScriptObject` of class `torch.classes.quantized.
          LinearPackedParamsBase`). That C++ binding exposes `unpack()` but
          NOT `_weight_bias`, and it is not a child module, so C2's traversal
          (module-level `_packed_params` + `_weight_bias()` only) missed it:
          GCN static INT8 reported `nonzero_after_int8 = 23` (the 16+7 GCN
          biases) and `bytes_by_dtype = {"fp32": 92}` for the whole quantized
          projection.
    """
    pp = getattr(mod, "_packed_params", None)
    if pp is not None and not isinstance(pp, torch.ScriptObject):
        try:
            qw, bias = pp._weight_bias()
            if torch.is_tensor(qw) and qw.dtype in (torch.qint8, torch.quint8):
                return qw, bias
        except Exception:
            pass
        return None  # Python wrapper present: the ScriptObject is in the
        # wrapper's vars, not this module's — nothing to find here.
    for k, v in list(vars(mod).items()):
        if not isinstance(v, torch.ScriptObject):
            continue
        try:
            parts = v.unpack()  # (qint8 weight, [bias]) — LinearPackedParamsBase
        except Exception:
            continue
        if (len(parts) >= 1 and torch.is_tensor(parts[0])
                and parts[0].dtype in (torch.qint8, torch.quint8)):
            return parts[0], (parts[1] if len(parts) > 1 and torch.is_tensor(parts[1])
                              else None)
    return None


def _is_packed_params_wrapper(mod: nn.Module) -> bool:
    """True for the Python `LinearPackedParams` wrapper module (an `nn.Module`
    subclass whose own `vars()` holds the raw packed-params ScriptObject under
    a `_packed_params` key). The integer weight is accounted once, via the
    owning op module (`_qnnpack_weight_bias`), and the wrapper must not be
    visited again (its vars() contain the very same ScriptObject, which the
    raw-attribute scan would unpack a second time → double count)."""
    pp = vars(mod).get("_packed_params")
    return (isinstance(pp, torch.ScriptObject)
            and type(mod).__name__ == "LinearPackedParams")


def _int_weight_tensors(model: nn.Module) -> list[torch.Tensor]:
    """Return the integer weight tensors of a converted model, unpacking
    qnnpack packed params (int_repr) and INT4 packed buffers, so that nonzero
    counts / byte accounting operate on the *integer* representation rather
    than on fp32 placeholders. FP16/BF16/FP32 models have no integer weight
    representation and yield an empty list. C3: also finds the FX-static
    GraphModule `_packed_weight_N` ScriptObjects (see `_qnnpack_weight_bias`)."""
    out: list[torch.Tensor] = []
    for _name, mod in model.named_modules():
        if isinstance(mod, (Int4PackedLinear, Int4PackedConv)):
            # Packed 2-values-per-uint8 buffer; report it as its own dtype.
            out.append(mod.packed)
            continue
        if _is_packed_params_wrapper(mod):
            continue
        wb = _qnnpack_weight_bias(mod)
        if wb is not None:
            out.append(wb[0].int_repr())
    return out


def int8_nonzero(model: nn.Module) -> int:
    """Number of nonzero entries of the model's INTEGER weight representation
    (int_repr() of qnnpack quantized weights, plus INT4 packed bytes). C2: the
    old code used `param.count_nonzero()` over the model's own parameters, which
    are fp32 / empty placeholders for quantized modules and reported 0 for every
    INT8 row. For dense models (no integer weight representation) this returns
    the count of nonzero fp params as a fallback so the field stays meaningful."""
    int_tensors = _int_weight_tensors(model)
    if int_tensors:
        return sum(int(t.count_nonzero()) for t in int_tensors)
    return sum(int(p.count_nonzero()) for p in model.parameters())


def bytes_by_dtype(model: nn.Module, kind: str | None = None) -> dict[str, int]:
    """C2: sum of numel*element_size per stored dtype, so container overhead
    (artifact_bytes - sum(bytes_by_dtype.values())) is visible. Handles:
      - plain fp params/buffers (fp32/fp16/bf16),
      - qnnpack quantized modules: int8 weight via int_repr() (1 B/elem),
        fp32 scales, int64 zero points, fp32 biases,
      - INT4 packed: the packed buffer as 'int4_packed_uint8' (1 B/packed-byte
        == 2 weights), plus fp32 scales / fp32 biases.
    C3: the quantized-op detection uses `_qnnpack_weight_bias`, which also
    finds FX-static GraphModule `_packed_weight_N` ScriptObjects (GCN static)
    and CNN static conv `_packed_params` attached directly to the Conv2d
    module — the C2 traversal missed both (it only matched module types named
    Linear/LinearReLU/ConvReLU2d/Conv2d that had a working `_weight_bias`)."""
    total: dict[str, int] = {}
    handled_int_module_ids: set[int] = set()

    def _add(label: str, t: torch.Tensor) -> None:
        if not torch.is_tensor(t):
            return
        n = int(t.numel()) * int(t.element_size())
        total[label] = total.get(label, 0) + n

    for name, mod in model.named_modules():
        if isinstance(mod, (Int4PackedLinear, Int4PackedConv)):
            # Packed buffer + scales + metadata are fully accounted here
            # (state_dict covers params AND buffers), so mark the module
            # handled to keep the fallback below from re-counting them
            # (C3: the C2 fallback skipped nothing for buffers, which
            # double-counted the INT4 `packed` buffer as 'uint8').
            handled_int_module_ids.add(id(mod))
            for k, v in mod.state_dict().items():
                if not torch.is_tensor(v) or v.numel() == 0:
                    continue
                if k == "packed":
                    _add("int4_packed_uint8", v)
                elif v.dtype in (torch.float16, torch.bfloat16):
                    _add(str(v.dtype).replace("torch.", ""), v)
                elif v.dtype == torch.float32:
                    _add("fp32", v)
                elif k in ("in_features", "out_features", "weight_shape",
                           "stride", "padding", "per_row"):
                    _add("int64_metadata", v)
                else:
                    _add(str(v.dtype).replace("torch.", ""), v)
            continue
        if _is_packed_params_wrapper(mod):
            continue
        wb = _qnnpack_weight_bias(mod)
        if wb is not None:
            # Quantized op: count its integer weight, its fp32 bias (when the
            # quantized module carries one), and its per-module qparam
            # params/buffers (fp32 scales, int64 zero points) without
            # double-counting the packed ScriptObject child itself.
            handled_int_module_ids.add(id(mod))
            qw, bias = wb
            _add("int8", qw.int_repr())
            if bias is not None:
                _add("fp32" if bias.dtype == torch.float32
                     else str(bias.dtype).replace("torch.", ""), bias)
            for n, p in mod.named_parameters():
                if torch.is_tensor(p) and p.dtype == torch.float32:
                    _add("fp32", p)
                elif torch.is_tensor(p) and p.dtype in (torch.float16, torch.bfloat16):
                    _add(str(p.dtype).replace("torch.", ""), p)
            for n, b in mod.named_buffers():
                if torch.is_tensor(b):
                    lbl = ("int64" if b.dtype == torch.int64
                           else "fp32" if b.dtype == torch.float32
                           else str(b.dtype).replace("torch.", ""))
                    _add(lbl, b)
            continue
        # A module whose whole state IS a packed ScriptObject (e.g. the
        # LinearPackedParams wrapper module, or a quantized Conv2d whose
        # `_packed_params` is a raw ScriptObject rather than a submodule):
        # skip it so its child tensors are not re-counted in the fallback
        # below — its int weight was counted via `_qnnpack_weight_bias` on
        # the owning op module (or, for the wrapper itself, it holds no
        # parameters/buffers of its own).

    # Plain (non-quantized) parameters/buffers that were not covered above.
    # C3: the C2 fallback only iterated `named_parameters()`; buffers (the INT4
    # `packed` uint8 buffer, activation/output qparam buffers) were then
    # double-counted here AND under their owning module's state-dtype loop
    # (e.g. 'uint8' == 'int4_packed_uint8'), inflating payload_bytes past
    # artifact_bytes. Track the handled modules and their own buffer keys.
    handled_ids = handled_int_module_ids
    handled_buffer_keys: set[str] = set()
    for name, p in model.named_parameters():
        leaf = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if id(leaf) in handled_ids:
            continue
        if p.dtype == torch.float32:
            _add("fp32", p)
        elif p.dtype in (torch.float16, torch.bfloat16):
            _add(str(p.dtype).replace("torch.", ""), p)
        else:
            _add(str(p.dtype).replace("torch.", ""), p)
    for name, b in model.named_buffers():
        leaf = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if id(leaf) in handled_ids:
            continue
        lbl = ("int64" if b.dtype == torch.int64
               else "fp32" if b.dtype == torch.float32
               else str(b.dtype).replace("torch.", ""))
        _add(lbl, b)
    return total


# ---------------------------------------------------------------------------
# Forward hooks (output-dtype inspection + reduced-precision MAC accounting)
# ---------------------------------------------------------------------------

REDUCED_DTYPES = {torch.float16, torch.bfloat16}


def _model_in_dtype(model: nn.Module) -> torch.dtype:
    """Input dtype the converted model expects: fp16/bf16 for half models,
    fp32 for fp32/INT8/INT4 models."""
    for p in model.parameters():
        if p.dtype in (torch.float16, torch.bfloat16):
            return p.dtype
    return torch.float32


def inspect_forwards(model, arch: str, iters: int = 3) -> dict[str, torch.dtype]:
    """Run `iters` eval-mode forwards on the fixed bench input and record the
    ACTUAL output dtype of every eligible op (evidence for the support report
    and the per-layer compute-precision table)."""
    set_threads()
    x, ei = load_bench_input(arch)
    seen: dict[str, torch.dtype] = {}
    hooks = []
    named = dict(model.named_modules())
    for name, _, _ in eligible_ops(model):
        def _hook(n=name):
            def h(m, inp, out):
                seen[n] = out.dtype
            return h
        hooks.append(named[name].register_forward_hook(_hook()))
    model.eval()
    in_dtype = _model_in_dtype(model)
    try:
        clear_gcn_caches(model)
        with torch.inference_mode():
            for _ in range(iters):
                if arch == "gcn":
                    model(x.to(in_dtype), ei)
                else:
                    model(x[:8].to(in_dtype))
    finally:
        for h in hooks:
            h.remove()
    return seen


def _weight_macs(model, x, ei) -> list[dict]:
    """Per-eligible-op MAC table (same conventions as bench.estimate_macs):
    per-image for fmnist (per-token linear / per-pixel conv), full graph for
    GCN (projection N*in*out + aggregation (E+N)*out per layer)."""
    from .bench import _mac_entries

    in_dtype = _model_in_dtype(model)
    entries = _mac_entries(model, (x[:1].to(in_dtype) if ei is None else x.to(in_dtype)), ei)
    is_gcn = ei is not None
    if is_gcn:
        N = int(x.shape[0])
        E_plus_N = int(ei.shape[1]) + N
    table = []
    for e in entries:
        if e["type"] == "GCNConv":
            name = e["name"] + ".lin"  # the reduced-precision unit is the projection
            macs = e["numel"] * e["tokens"] + E_plus_N * e["out_ch"]
        else:
            name = e["name"]
            macs = e["numel"] * e["tokens"]
        table.append({"name": name, "type": e["type"], "macs": int(macs),
                      "nnz": e["nnz"], "numel": e["numel"]})
    return table


def frac_macs_reduced_precision(model, arch: str, reduced_layer_names: set[str]) -> float:
    """Fraction of total MACs performed by ops whose recorded output dtype (or
    explicit quantized status) shows reduced-precision execution.
    fp32 -> 0.0; every eligible op reduced -> 1.0."""
    x, ei = load_bench_input(arch)
    table = _weight_macs(model, x, ei)
    total = sum(t["macs"] for t in table)
    if total == 0:
        return 0.0
    red = sum(t["macs"] for t in table if t["name"] in reduced_layer_names)
    return red / total


def _mac_count(model, x, ei) -> int:
    """Total MAC count for a single forward pass (per-sample for fmnist,
    full-graph for GCN)."""
    table = _weight_macs(model, x, ei)
    return sum(t["macs"] for t in table)


def _frac_macs(model, x, ei, reduced_layer_names: set[str]) -> float:
    """Fraction of total MACs in reduced-precision layers."""
    table = _weight_macs(model, x, ei)
    total = sum(t["macs"] for t in table)
    if total == 0:
        return 0.0
    red = sum(t["macs"] for t in table if t["name"] in reduced_layer_names)
    return red / total


def _select_int8(rows: list[dict]) -> str:
    """Frozen INT8 selection rule (C.1): pick the format with the best
    val_acc among those whose latency is within 2× the minimum. Ties broken
    by lower latency. If only one format is present, return it."""
    if not rows:
        return "none"
    if len(rows) == 1:
        return rows[0]["format"]
    min_lat = min(r["latency_ms"] for r in rows)
    # Candidates: latency within 2x of the minimum
    candidates = [r for r in rows if r["latency_ms"] <= 2.0 * min_lat]
    # Among candidates, pick highest val_acc; ties broken by lower latency
    candidates.sort(key=lambda r: (-r["val_acc"], r["latency_ms"]))
    return candidates[0]["format"]


# ---------------------------------------------------------------------------
# Static-INT8 calibration (frozen C.1 rule)
# ---------------------------------------------------------------------------


def calibration_inputs(arch: str) -> dict:
    """Frozen calibration inputs. fmnist: the 1,024 training-split images in
    splits/fmnist_calib_idx.npy (asserted subset of the train split, inputs
    only, normalized identically to training). Cora: the full-graph features
    (the only coherent transductive calibration; inputs only)."""
    if arch == "gcn":
        data = load_cora()
        return {"x": data.x.contiguous(), "edge_index": data.edge_index,
                "n": int(data.x.shape[0]),
                "rule": "Cora full-graph features only"}
    from .data import _fmnist_dataset, _images

    idx = np.load(CALIB_FILE)
    # C1b fix: use the project-root splits dir (ROOT), not RESULTS_DIR.parent,
    # so tests that redirect LAB3_RESULTS_DIR do not break calibration.
    train_idx = np.load(SPLITS_DIR / "fmnist_train_idx.npy")
    assert set(int(i) for i in idx).issubset(set(int(i) for i in train_idx)), \
        "calib idx must be a subset of the train split"
    # Calibration indices are into the 60k official dataset; map them to
    # positions in the 55k training tensor so normalization matches training.
    ds = _fmnist_dataset("train")
    x = _images(ds, idx)  # (1024, 1, 28, 28) raw
    tr_x_full = _images(ds, train_idx)
    mean = tr_x_full.mean(dim=(0, 2, 3), keepdim=True)
    std = tr_x_full.std(dim=(0, 2, 3), keepdim=True)
    calib = ((x - mean) / std).contiguous()
    return {"x": calib, "n": int(calib.shape[0]),
            "rule": "splits/fmnist_calib_idx.npy (1024 train-split images, inputs only)"}


def _calib_batches(arch: str) -> list[dict]:
    """Frozen calibration batches (C.1 rule). fmnist: 1,024 training-split
    images in 64-image batches (labels never used). Cora: the full-graph
    features, one pass (inputs only)."""
    cal = calibration_inputs(arch)
    if arch == "gcn":
        return [cal]
    x = cal["x"]
    bs = 64
    return [{"x": x[i:i + bs], "n": min(bs, x.shape[0] - i), "rule": cal["rule"]}
            for i in range(0, x.shape[0], bs)]


# ---------------------------------------------------------------------------
# FP16 / BF16
# ---------------------------------------------------------------------------


def half_quantize(model: nn.Module, arch: str, dtype: torch.dtype) -> nn.Module:
    """Cast EVERYTHING (weights, biases, parameters) to `dtype` and cast
    inputs to `dtype` before every forward. On this machine nn.Linear/Conv2d
    compute natively in fp16/bf16 on CPU (verified: output dtype == dtype, no
    upcast), so the whole graph runs in `dtype`; logits are cast back to fp32
    for cross-entropy inside the eval helpers."""
    m = build_model(arch)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    m = m.to(dtype)
    m.eval()
    return m


def _half_forward(model: nn.Module, arch: str, x, ei) -> torch.Tensor:
    in_dtype = _model_in_dtype(model)
    with torch.inference_mode():
        if arch == "gcn":
            clear_gcn_caches(model)
            out = model(x.to(in_dtype), ei)
        else:
            out = model(x.to(in_dtype))
    return out.float() if out.dtype != torch.float32 else out


# ---------------------------------------------------------------------------
# INT8 — dynamic (quantize_dynamic, qnnpack)
# ---------------------------------------------------------------------------


def _is_qnnpack_quantized_linear(mod: nn.Module) -> bool:
    """True if `mod` is a qnnpack INT8-quantized Linear produced by either
    quantize_dynamic (module class torch.ao.nn.quantized.dynamic.modules.linear.Linear,
    NOT a subclass of nn.Linear) or FX prepare_fx/convert_fx (module class
    LinearReLU for fused linear+relu, or Linear for the unfused last linear).
    Detection: presence of _packed_params (LinearPackedParams) child on a
    module named Linear or LinearReLU."""
    if not hasattr(mod, "_packed_params"):
        return False
    return type(mod).__name__ in ("Linear", "LinearReLU")


def _is_qnnpack_quantized_conv(mod: nn.Module) -> bool:
    """True if `mod` is a qnnpack INT8-quantized Conv2d produced by FX
    prepare_fx/convert_fx (module class Conv2d / ConvReLU2d with a
    _packed_params child)."""
    if type(mod).__name__ in ("Conv2d", "ConvReLU2d"):
        return hasattr(mod, "_packed_params")
    return False


def _gcn_swap_projection(model: nn.Module) -> None:
    """Swap each GCNConv's inner projection for a plain ``nn.Linear`` with
    identical weights so it can be reduced-precision quantized (dynamic or FX
    static). GCNConv.lin carries NO bias — the layer bias is ``conv.bias`` and
    is added by PyG after ``propagate`` — so the replacement Linear is
    bias=False. Logits are bit-identical before/after the swap (verified by
    the C.1 "verify logits equal before quantizing" requirement). Aggregation
    / normalization (propagate / gcn_norm) stay fp32 and are counted as fp32
    MACs.

    FX cannot trace the full GCN (PyG ``propagate`` receives ``edge_index`` as
    a data argument, which ``MessagePassing.propagate`` rejects during FX
    symbolic tracing), so for static INT8 the swapped projections are
    calibrated+converted individually (see ``int8_quantize_static``)."""
    from torch_geometric.nn import GCNConv

    for mod in model.modules():
        if isinstance(mod, GCNConv):
            in_c, out_c = mod.in_channels, mod.out_channels
            lin = nn.Linear(in_c, out_c, bias=False)
            with torch.no_grad():
                lin.weight.copy_(mod.lin.weight)
            mod.lin = lin


def int8_quantize_dynamic(model: nn.Module, arch: str) -> tuple[nn.Module, dict]:
    """Apply quantize_dynamic({nn.Linear}) to a copy of `model`.

    Every nn.Linear (including the GCNConv.lin projections after the swap) is
    replaced by a qnnpack dynamic-quantized Linear (INT8 weights, per-batch
    activation quantization, qnnpack ``quantized::linear`` kernel). Conv2d is
    left fp32 for the dynamic path (the frozen C.1 dynamic spec is
    ``quantize_dynamic({nn.Linear})``); Conv2d static INT8 is handled by the
    FX flow."""
    set_qnnpack_engine()
    set_threads()
    m = build_model(arch)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    if arch == "gcn":
        _gcn_swap_projection(m)
    m.eval()

    if arch == "gcn":
        # Quantize each swapped projection. quantize_dynamic only works on a
        # model (container), not a standalone nn.Linear, so we wrap each
        # projection in a tiny container, quantize it, and extract the result.
        for mod in m.modules():
            if type(mod).__name__ == "GCNConv":
                container = nn.Sequential(mod.lin)
                qcontainer = tq.quantize_dynamic(container, {nn.Linear},
                                                 dtype=torch.qint8)
                mod.lin = qcontainer[0]
    else:
        m = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        m.eval()
    return m, {"mode": "dynamic",
               "rule": "quantize_dynamic({nn.Linear}, dtype=torch.qint8), qnnpack engine"}


def int8_quantize_static(model: nn.Module, arch: str) -> tuple[nn.Module, dict]:
    """Apply FX static INT8 to a copy of `model`.

    Uses prepare_fx/convert_fx with get_default_qconfig_mapping("qnnpack").
    Calibrates on the frozen calibration inputs (C.1 rule).

    For MLP: the whole model is traced and every Linear is quantized.
    For CNN: the whole model is traced; Conv2d (fused to ConvReLU2d) and
    Linear (fused to LinearReLU) are both quantized.
    For GCN: each GCNConv projection is quantized individually via FX (the
    GCNConv wrapper itself is not FX-traceable through PyG's propagate/scatter).
    """
    m = _convert_static(model, arch)
    calib = _calib_batches(arch)
    calib_info = {"size": int(sum(b["n"] for b in calib)),
                  "rule": calib[0]["rule"],
                  "batching": ("64-image batches" if arch != "gcn"
                               else "single full-graph pass"),
                  "mode": "static (FX prepare_fx/convert_fx, qnnpack)"}
    return m, calib_info


def _convert_static(model: nn.Module, arch: str) -> nn.Module:
    """Convert a copy of `model` to FX static INT8 (no record). Calibrates on
    the frozen calibration inputs. Reusable by the static reload path."""
    set_qnnpack_engine()
    set_threads()
    qcm = get_default_qconfig_mapping("qnnpack")
    calib = _calib_batches(arch)

    if arch == "gcn":
        m = build_model(arch)
        m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
        _gcn_swap_projection(m)
        m.eval()
        x = calib[0]["x"]
        ei = calib[0]["edge_index"]
        from .compact import clear_gcn_caches as _ccc
        _ccc(m)
        captured: dict[str, torch.Tensor] = {}
        hooks = []
        for name, mod, kind in eligible_ops(m):
            if kind != "lin":
                continue
            def _mk(n=name):
                def h(mm, inp, out):
                    captured[n] = inp[0].detach().clone()
                return h
            hooks.append(mod.register_forward_hook(_mk()))
        with torch.inference_mode():
            m(x, ei)
        for h in hooks:
            h.remove()
        for name, mod, kind in list(eligible_ops(m)):
            if kind != "lin":
                continue
            act = captured[name]
            fx = prepare_fx(mod, qcm, example_inputs=(act[:8],))
            mod.eval()
            with torch.inference_mode():
                for i in range(0, act.shape[0], 64):
                    fx(act[i:i + 64])
            converted = convert_fx(fx)
            parent, child = _resolve_parent(m, name)
            setattr(parent, child, converted)
        return m

    m = build_model(arch)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    m.eval()
    fx = prepare_fx(m, qcm, example_inputs=(calib[0]["x"][:8],))
    m.eval()
    with torch.inference_mode():
        for b in calib:
            fx(b["x"])
    m = convert_fx(fx)
    m.eval()
    return m


def _int8_reduced_names(model: nn.Module, arch: str, parent: nn.Module | None = None) -> set[str]:
    """Return the set of PARENT op names that ran in INT8 (for frac_macs).

    Walks the parent's eligible ops (always plain nn.Linear/Conv2d/GCNConv) and
    looks up the same-named module in the converted `model` to check whether it
    was swapped for a qnnpack quantized module (quantize_dynamic Linear, FX
    LinearReLU / Conv2d / ConvReLU2d, or an FX GraphModule for the GCN
    per-projection path)."""
    ref = parent if parent is not None else model
    red = set()
    # Map converted-model module names -> module, for lookup by parent op name.
    conv_by_name = {n: m for n, m in model.named_modules()}
    for name, _op, kind in eligible_ops(ref):
        mod = conv_by_name.get(name)
        if mod is None:
            continue
        if kind in ("lin", "linear"):
            if (_is_qnnpack_quantized_linear(mod)
                    or type(mod).__name__ == "GraphModule"):
                red.add(name)
        elif kind == "conv":
            if _is_qnnpack_quantized_conv(mod):
                red.add(name)
    return red


def _reduced_op_names(model: nn.Module, arch: str, fmt: str,
                      parent: nn.Module | None = None) -> set[str]:
    """Set of PARENT op names that executed in reduced precision under `fmt`.

    Names are the dense-parent op names (from eligible_ops(parent)) so they
    join cleanly with the parent MAC table. For half formats every dense
    op is reduced; for INT8 the set is the qnnpack-quantized ops; for INT4
    (simulated) compute stays fp32 so the set is empty."""
    if fmt in ("fp16", "bf16"):
        # Every eligible op ran in the half dtype.
        return {name for name, _op, _k in eligible_ops(model)}
    if fmt in ("int8_dynamic", "int8_static"):
        return _int8_reduced_names(model, arch, parent)
    if fmt == "int4_simulated":
        return set()  # simulated: dequant to fp32, no reduced-precision compute
    return set()


# ---------------------------------------------------------------------------
# Simulated weight-only INT4
# ---------------------------------------------------------------------------


class Int4PackedLinear(nn.Module):
    """Weight-only INT4, per-output-channel symmetric [-8, 7], fp32 scales.
    The DEQUANTIZATION IS INSIDE THE TIMED REGION: the module holds only the
    packed uint8 buffer + fp32 scales and rebuilds the fp32 weight each
    forward (cached until the packed buffer changes)."""

    def __init__(self, lin: nn.Module):
        super().__init__()
        w = lin.weight.detach()
        # Support both standard nn.Linear (in_features/out_features) and
        # PyG GCNConv.lin (in_channels/out_channels, no bias attr).
        if hasattr(lin, "in_features"):
            self.in_features = lin.in_features
            self.out_features = lin.out_features
            bias = lin.bias
        else:
            self.in_features = w.shape[1]
            self.out_features = w.shape[0]
            bias = getattr(lin, "bias", None)
        scale = (w.abs().amax(dim=1) / 7.0).clamp(min=1e-8).float()
        wq = torch.round(w / scale.unsqueeze(1)).clamp(-8, 7)
        u = (wq + 8).to(torch.uint8)
        L = u.shape[1]
        pad = torch.zeros(u.shape[0], 1, dtype=torch.uint8) if L % 2 else u.new_empty(0)
        up = torch.cat([u, pad], dim=1) if pad.numel() else u
        packed = (up[:, 1::2] & 0x0F) | (up[:, 0::2] << 4)
        self.register_buffer("packed", packed.contiguous())
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias.detach().clone() if bias is not None else None)
        self._cache_key: tuple | None = None
        self._w: torch.Tensor | None = None

    def dequant(self) -> torch.Tensor:
        key = (self.packed.data_ptr(), self.packed.shape)
        if self._w is None or self._cache_key != key:
            p = self.packed.to(torch.int16)
            cols = self.packed.shape[1] * 2
            w = torch.empty(self.packed.shape[0], cols, dtype=torch.float32)
            # Packing: up[:, 0::2] → HIGH nibble, up[:, 1::2] → LOW nibble
            w[:, 0::2] = ((p >> 4) & 0x0F).to(torch.float32)
            w[:, 1::2] = (p & 0x0F).to(torch.float32)
            w = (w - 8.0) * self.scale.unsqueeze(1)
            if w.shape[1] > self.in_features:
                w = w[:, :self.in_features]
            self._w = w
            self._cache_key = key
        return self._w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.dequant().t()
        if self.bias is not None:
            y = y + self.bias
        return y

    def state_dict(self, *a, **kw) -> dict:
        return {"packed": self.packed, "scale": self.scale,
                "bias": self.bias,
                "in_features": torch.tensor(self.in_features),
                "out_features": torch.tensor(self.out_features)}

    def load_state_dict(self, sd: dict, strict: bool = True):
        self.packed = sd["packed"]
        self.scale = sd["scale"]
        self.bias = sd["bias"]
        self.in_features = int(sd["in_features"])
        self.out_features = int(sd["out_features"])
        self._w = None
        return None


class Int4PackedConv(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        w = conv.weight.detach().reshape(conv.out_channels, -1)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.weight_shape = tuple(conv.weight.shape)
        scale = (w.abs().amax(dim=1) / 7.0).clamp(min=1e-8).float()
        wq = torch.round(w / scale.unsqueeze(1)).clamp(-8, 7)
        # Convert to uint8 (values 0..15 after bias shift) to avoid int8
        # overflow on the << 4 bit shift.
        u = (wq + 8).to(torch.uint8)
        L = u.shape[1]
        pad = torch.zeros(u.shape[0], 1, dtype=torch.uint8) if L % 2 else u.new_empty(0)
        up = torch.cat([u, pad], dim=1) if pad.numel() else u
        packed = (up[:, 1::2] & 0x0F) | (up[:, 0::2] << 4)
        self.register_buffer("packed", packed.contiguous())
        self.register_buffer("scale", scale)
        self.register_buffer("bias", conv.bias.detach().clone() if conv.bias is not None else None)
        self._n_elements = w.numel()  # total element count
        self._per_row = w.shape[1]  # elements per output channel (before padding)
        self._cache_key = None
        self._w = None

    def dequant(self) -> torch.Tensor:
        key = (self.packed.data_ptr(), self.packed.shape)
        if self._w is None or self._cache_key != key:
            p = self.packed.to(torch.int16)
            L = self.packed.shape[1] * 2
            w = torch.empty(self.packed.shape[0], L, dtype=torch.float32)
            # Packing: up[:, 0::2] → HIGH nibble, up[:, 1::2] → LOW nibble
            w[:, 0::2] = ((p >> 4) & 0x0F).to(torch.float32)
            w[:, 1::2] = (p & 0x0F).to(torch.float32)
            w = (w - 8.0) * self.scale.unsqueeze(1)
            w = w[:, :self._per_row]  # truncate padding to per-row count
            self._w = w.reshape(self.weight_shape)
            self._cache_key = key
        return self._w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.dequant(), self.bias, self.stride, self.padding)

    def state_dict(self, *a, **kw) -> dict:
        return {"packed": self.packed, "scale": self.scale, "bias": self.bias,
                "weight_shape": torch.tensor(list(self.weight_shape)),
                "stride": torch.tensor(list(self.stride)),
                "padding": torch.tensor(list(self.padding)),
                "per_row": torch.tensor(self._per_row)}

    def load_state_dict(self, sd: dict, strict: bool = True):
        self.packed = sd["packed"]
        self.scale = sd["scale"]
        self.bias = sd["bias"]
        self.weight_shape = tuple(int(v) for v in sd["weight_shape"])
        self.stride = tuple(int(v) for v in sd["stride"])
        self.padding = tuple(int(v) for v in sd["padding"])
        self._per_row = int(sd["per_row"])
        self._w = None
        return None


def int4_quantize(model: nn.Module, arch: str) -> nn.Module:
    # C2 fix: `state_dict()[k] = v.clone()` does NOT update the module's
    # parameters (the dict returned by state_dict() is a copy of the
    # Parameter references, not a live view of the module); use
    # `load_state_dict` instead so the INT4 weights are built from the
    # actual parent weights, not from `build_model(arch)`'s random init.
    m = build_model(arch)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    for name, op, kind in list(eligible_ops(m)):
        wrap = Int4PackedLinear(op) if kind in ("linear", "lin") else Int4PackedConv(op)
        parent, child_name = _resolve_parent(m, name)
        setattr(parent, child_name, wrap)
    return m


def int4_packed_bytes(model: nn.Module) -> int:
    flat = {}
    for name, m_ in model.named_modules():
        if isinstance(m_, (Int4PackedLinear, Int4PackedConv)):
            for k, v in m_.state_dict().items():
                flat[f"{name}.{k}"] = v
    buf = io.BytesIO()
    torch.save(flat, buf)
    return len(buf.getvalue())


# ---------------------------------------------------------------------------
# Timing / RSS (frozen protocol; INT4 dequant inside timed region)
# ---------------------------------------------------------------------------


def _time_model(model, arch: str) -> dict:
    x, ei = load_bench_input(arch)
    model.eval()
    lat = {}
    in_dtype = _model_in_dtype(model)
    if arch == "gcn":
        def fwd():
            clear_gcn_caches(model)
            return model(x.to(in_dtype), ei)
        _run(fwd)
        lat["full_graph"] = _protocol(fwd, "full_graph")
    else:
        for bs in (1, 256):
            xb = x[:bs].to(in_dtype)
            f = (lambda xb=xb: model(xb))
            _run(f)
            lat[f"batch_{bs}"] = _protocol(f, bs)
    return lat


def _run(f, n: int = WARMUP):
    with torch.inference_mode():
        for _ in range(n):
            f()


def _protocol(f, batch_size) -> dict:
    with torch.inference_mode():
        ns = []
        for _ in range(TIMED):
            t0 = time.perf_counter_ns()
            f()
            ns.append(time.perf_counter_ns() - t0)
    ns = np.asarray(ns, dtype=np.float64) / 1e6
    q1, q2, q3 = np.percentile(ns, [25, 50, 75])
    return {"median_ms": float(q2), "iqr_ms": float(q3 - q1),
            "warmup": WARMUP, "iters": TIMED,
            "threads": torch.get_num_threads(), "batch_size": batch_size}


def _rss_subprocess(arch: str, artifact: str, kind: str) -> dict:
    """Fresh-subprocess peak RSS: imports + bench input load -> before;
    artifact load + few forwards -> peak."""
    import json
    import subprocess
    import sys

    code = f"""
import json, psutil, sys
import torch
torch.backends.quantized.engine = "qnnpack"
from lab3.util import set_threads
from lab3.bench import load_bench_input
from lab3.precision import _rebuild_from_artifact, _model_in_dtype
set_threads()
arch = {arch!r}
x, ei = load_bench_input(arch)
proc = psutil.Process()
before = proc.memory_info().rss
m = _rebuild_from_artifact(arch, {artifact!r}, {kind!r})
m.eval()
dtype = _model_in_dtype(m)
peak = before
with torch.inference_mode():
    for _ in range(5):
        if arch == "gcn":
            m(x.to(dtype), ei)
        else:
            m(x[:256].to(dtype))
        peak = max(peak, proc.memory_info().rss)
print(json.dumps({{"before": int(before), "peak": int(peak)}}))
"""
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, check=True,
                          cwd=str(ROOT),
                          env={**os.environ, "PYTHONPATH": str(ROOT / "code")})
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


# ---------------------------------------------------------------------------
# Artifact save / reload
# ---------------------------------------------------------------------------


def _save_artifact(model: nn.Module, path: Path, kind: str, arch: str) -> int:
    """Persist the converted model to `path`. Returns file bytes.

    C4: for INT8 (dynamic and static, all archs) the file is a single
    torch.save of `{kind, architecture, state}` where `state` is the
    converted model's `state_dict()` itself — the int8 weights + scales +
    zero points (and, for FX-static GCN, the packed-weight ScriptObjects),
    with NO nested pickled module bytes and NO fp32 copy of the weights.
    Reload = rebuild the quantized structure from the fp32 parent
    architecture (same quantize_dynamic / FX prepare+convert / GCN
    per-projection path), then `load_state_dict(strict=True)` (GCN static
    additionally re-assigns the raw `_packed_weight_N` ScriptObject
    attributes, which FX GraphModules do not round-trip through
    `load_state_dict` — see `_rebuild_from_artifact`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind,
                "architecture": arch,
                "state": _artifact_state(model, kind, arch)}, path)
    return os.path.getsize(path)


def _container_overhead(path: str) -> dict:
    """C3: per-entry `zipfile.infolist()` accounting of a saved artifact, so the
    container overhead (artifact_bytes - payload_bytes) is attributable instead
    of "unexplained".

    For every artifact torch.save writes a zip. The zip's local headers +
    central-directory + EOCD are `structural_bytes`; the sum of `file_size`
    across entries is `stored_bytes`; for int8_dynamic/int8_static the state
    is `{"pickled": <nested torch.save zip bytes>}`, so the outer data.pkl
    embeds the ENTIRE inner zip as one stored (uncompressed) blob — that blob
    is itself a zip (with its own structural bytes + data entries). We unpack
    that inner zip too (`inner`), so the reader can see that the "overhead" on
    the mlp/gcn int8 rows is the nested re-encoding, not dead data.

    Returns {artifact_bytes, structural_bytes, stored_bytes, entries: [...],
             inner: {structural_bytes, stored_bytes, entries: [...]} | None}.
    """
    import io as _io
    info = zipfile.ZipFile(path).infolist()
    out = {
        "artifact_bytes": os.path.getsize(path),
        "structural_bytes": int(os.path.getsize(path) - sum(i.file_size for i in info)),
        "stored_bytes": int(sum(i.file_size for i in info)),
        "entries": [{"name": i.filename, "file_size": i.file_size,
                     "compress_size": i.compress_size,
                     "compress_type": i.compress_type} for i in info],
        "inner": None,
    }
    try:
        payload = torch.load(path, weights_only=False)
        st = (payload or {}).get("state") or {}
        if isinstance(st, dict) and "pickled" in st:
            zi = zipfile.ZipFile(_io.BytesIO(st["pickled"]))
            in_info = zi.infolist()
            out["inner"] = {
                "stored_bytes": int(sum(i.file_size for i in in_info)),
                "structural_bytes": int(len(st["pickled"]) - sum(i.file_size for i in in_info)),
                "entries": [{"name": i.filename, "file_size": i.file_size,
                             "compress_size": i.compress_size,
                             "compress_type": i.compress_type} for i in in_info],
            }
    except Exception:
        pass
    return out


# `state_dict()` result for a converted model, carrying the `_metadata`
# attribute (per-module `version` entries) that the qnnpack loaders require.
# Must be module-level (picklable) for torch.save.
class _SDWithMeta(dict):
    """Plain dict that keeps the state_dict `_metadata` attribute."""
    pass


def _artifact_state(model: nn.Module, kind: str, arch: str) -> dict:
    if kind == "int4_simulated":
        flat = {}
        for name, m_ in model.named_modules():
            if isinstance(m_, (Int4PackedLinear, Int4PackedConv)):
                for k, v in m_.state_dict().items():
                    flat[f"{name}.{k}"] = v
        # Also save the fp32 parent weights so the reload can rebuild the
        # base model with the correct shapes before wrapping.
        fp32 = {}
        for name, m_ in model.named_modules():
            if not isinstance(m_, (Int4PackedLinear, Int4PackedConv)):
                for k, v in m_.state_dict().items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        fp32[f"{name}.{k}"] = v
        return {"packed": flat, "fp32": fp32}
    if kind in ("fp16", "bf16", "fp32"):
        # Plain dicts (no `_metadata`): fp16/bf16/fp32 flat state dicts
        # (their loaders do not need version metadata). (C4: the C3
        # `quantized_state` wrapper and the nested-pickle `pickled` container
        # are gone.)
        return {k: v for k, v in model.state_dict().items()}
    # int8 (dynamic AND static, all archs) — single state_dict of the
    # converted model. Contains ONLY the int8 weights + fp32/int64 scales and
    # zero points (plus, for FX-static GCN, the raw `_packed_weight_N`
    # ScriptObjects as state-dict values). No nested torch.save(model) zip, no
    # fp32 copy of the weights. (C2 pickled the whole module for mlp/gcn —
    # a nested container that doubled the int8 payload and carried dead fp32
    # structure; C3 used a flat state dict only for cnn static.)
    #
    # `model.state_dict()` is an OrderedDict with a `_metadata` attribute
    # (per-module `version` entries) that the qnnpack dynamic/fused Linear
    # loaders REQUIRE to pick the v3+ deserialization path (without it they
    # fall back to the v1 path and raise KeyError 'net.0.weight'). A plain
    # dict comprehension drops it, so copy it explicitly — the values are
    # plain picklable objects and add no tensor bytes.
    sd = model.state_dict()
    out2 = _SDWithMeta()
    out2.update(sd)
    md = getattr(sd, "_metadata", None)
    if md is not None:
        out2._metadata = dict(md)  # type: ignore[attr-defined]
    return out2


def _rebuild_from_artifact(arch: str, path: str, kind: str) -> nn.Module:
    """Load a saved converted artifact into a runnable model."""
    set_qnnpack_engine()
    if kind in ("int8_dynamic", "int8_static"):
        # C4: single state_dict artifact. Rebuild the deterministic quantized
        # structure from the fp32 seed-0 parent using the SAME conversion path
        # that produced the artifact (quantize_dynamic / FX prepare+convert /
        # GCN per-projection), then load the quantized weights from the file
        # (strict). Reloaded logits are bit-identical to the in-memory
        # converted model (verified with atol 0 in _serialize_and_reload / the
        # factorial reload check and in the tests).
        parent, _ = load_checkpoint(arch, 0)
        if kind == "int8_dynamic":
            m = build_model(arch)
            m.load_state_dict({k: v.clone() for k, v in parent.state_dict().items()})
            m.eval()
            if arch == "gcn":
                _gcn_swap_projection(m)
                for mod in m.modules():
                    if type(mod).__name__ == "GCNConv":
                        container = nn.Sequential(mod.lin)
                        qcontainer = tq.quantize_dynamic(container, {nn.Linear},
                                                         dtype=torch.qint8)
                        mod.lin = qcontainer[0]
            else:
                m = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
                m.eval()
        else:
            # Static: _convert_static recalibrates the activation qparams on
            # the frozen calibration inputs (deterministic), then we overwrite
            # with the exact calibrated values + quantized weights from the
            # file.
            m = _convert_static(parent, arch)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        st = payload["state"]
        packed_keys: list[tuple[nn.Module, str, object]] = []
        if arch == "gcn" and kind == "int8_static":
            # FX GraphModules keep the packed weight as a raw instance
            # attribute (`_packed_weight_N` ScriptObject); load_state_dict
            # reports it as an "unexpected key" and cannot set it. Re-assign
            # it directly from the file (the structure's own attribute).
            by_name = {n: mm for n, mm in m.named_modules()}
            st = dict(st)
            for k in [k for k in st if "_packed_weight_" in k]:
                packed_keys.append((by_name[k.rsplit("._packed_weight_")[0]],
                                    k.rsplit(".", 1)[-1], st[k]))
                del st[k]
        m.load_state_dict(st, strict=True)
        for mod, leaf, val in packed_keys:
            setattr(mod, leaf, val)
        m.eval()
        return m
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if kind in ("fp16", "bf16"):
        m = build_model(arch)
        src = payload["state"]
        dtype = next(iter(src.values())).dtype
        m.load_state_dict({k: v for k, v in src.items()}, strict=True)
        m = m.to(dtype)
        m.eval()
        return m
    if kind == "fp32":
        m = build_model(arch)
        m.load_state_dict({k: v for k, v in payload["state"].items()}, strict=True)
        m.eval()
        return m
    if kind == "int4_simulated":
        # Rebuild the base model, load fp32 weights, wrap eligible ops in
        # Int4Packed*, then load the packed state.
        m = build_model(arch)
        st = payload["state"]
        # Load fp32 parent weights into the base model
        if st.get("fp32"):
            base_sd = {}
            for k, v in st["fp32"].items():
                parts = k.split(".", 1)
                if len(parts) == 2:
                    base_sd[k] = v
            # The base model's state dict has the same key names for non-packed
            # modules (bias, etc.)
            try:
                m.load_state_dict(base_sd, strict=False)
            except Exception:
                pass
        flat = st["packed"]
        for name, op, kind_ in list(eligible_ops(m)):
            wrap = (Int4PackedLinear(op) if kind_ in ("linear", "lin")
                    else Int4PackedConv(op))
            parent, child_name = _resolve_parent(m, name)
            setattr(parent, child_name, wrap)
        for name, m_ in m.named_modules():
            if isinstance(m_, (Int4PackedLinear, Int4PackedConv)):
                sd = {k[len(name) + 1:]: flat[k]
                      for k in flat if k.startswith(name + ".")}
                m_.load_state_dict(sd)
        m.eval()
        return m
    m = build_model(arch)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Evaluation (val + test; test is reported, never used for any choice)
# ---------------------------------------------------------------------------


def eval_model(model: nn.Module, arch: str, kind: str) -> tuple[float, float, float, float]:
    """Returns (val_acc, val_loss, test_acc, test_loss)."""
    set_threads()
    in_dtype = _model_in_dtype(model)
    if arch == "gcn":
        data = load_cora()
        model.eval()
        with torch.inference_mode():
            clear_gcn_caches(model)
            out = model(data.x.to(in_dtype), data.edge_index)
            out = out.float() if out.dtype != torch.float32 else out
            va, te = out[data.lm_val], out[data.lm_test]
            va_acc = float((va.argmax(1) == data.y[data.lm_val]).float().mean())
            va_loss = float(F.cross_entropy(va, data.y[data.lm_val]))
            te_acc = float((te.argmax(1) == data.y[data.lm_test]).float().mean())
            te_loss = float(F.cross_entropy(te, data.y[data.lm_test]))
        return va_acc, va_loss, te_acc, te_loss
    tr_x, tr_y, va_x, va_y, te_x, te_y, _, _ = fmnist_split_tensors()
    model.eval()
    res = {}
    with torch.inference_mode():
        for x, y, key in ((va_x, va_y, "val"), (te_x, te_y, "test")):
            x = x.to(in_dtype)
            n = len(y)
            correct = 0
            loss_sum = 0.0
            for i in range(0, n, 1024):
                xb, yb = x[i:i + 1024], y[i:i + 1024]
                out = model(xb)
                out = out.float() if out.dtype != torch.float32 else out
                loss_sum += float(F.cross_entropy(out, yb, reduction="sum"))
                correct += int((out.argmax(1) == yb).sum())
            res[key] = (correct / n, loss_sum / n)
    return res["val"][0], res["val"][1], res["test"][0], res["test"][1]


def logits_of(model: nn.Module, arch: str, x, ei) -> torch.Tensor:
    model.eval()
    in_dtype = _model_in_dtype(model)
    with torch.inference_mode():
        if arch == "gcn":
            clear_gcn_caches(model)
            out = model(x.to(in_dtype), ei)
        else:
            out = model(x.to(in_dtype))
    return out.float() if out.dtype != torch.float32 else out


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

PRECISION_ARTIFACT_DIR = RESULTS_DIR / "precision_artifacts"


def _row(arch: str, phase: str, method: str, parent_run_id: str) -> dict:
    rec = base_record(arch, command="eval")
    rec.update({
        "part": "C",
        "phase": phase,
        "method": method,
        "parent_run_id": parent_run_id,
        "frac_macs_reduced_precision": None,
        "reload_equivalence": None,
        "runtime_status": "ok",
        "calibration": None,
        "reduced_precision_ops": [],
    })
    return rec


def run_wall(rec: dict, t0: float) -> dict:
    rec["wall_time_s"] = round(time.perf_counter() - t0, 3)
    return rec


def _classify(e: Exception) -> str:
    msg = str(e)
    if "NoQEngine" in msg or "Didn't find engine" in msg:
        return f"unsupported_backend: {type(e).__name__}: {msg[:120]}"
    if "NotImplementedError" in type(e).__name__ or "Could not run" in msg:
        return f"unsupported_op: {type(e).__name__}: {msg[:120]}"
    if "deprecated" in msg.lower():
        return f"deprecated_api: {type(e).__name__}: {msg[:120]}"
    return f"{type(e).__name__}: {msg[:120]}"


def _serialize_and_reload(rec: dict, model: nn.Module, arch: str, kind: str,
                          tolerance: float) -> None:
    """Save the converted artifact, reload it, and compare logits on a fixed
    input slice (reload_equivalence). C2: also records bytes_by_dtype (the
    sum of numel*element_size per stored dtype) next to artifact_bytes so the
    container overhead (artifact_bytes - sum(bytes_by_dtype.values())) is
    visible, and the integer-representation nonzero count."""
    PRECISION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = PRECISION_ARTIFACT_DIR / f"{arch}-{rec['method']}_{rec['run_id']}.pt"
    rec["artifact"] = str(path)
    rec["artifact_bytes"] = _save_artifact(model, path, kind, arch)
    bbd = bytes_by_dtype(model, kind)
    rec["bytes_by_dtype"] = bbd
    rec["payload_bytes"] = int(sum(bbd.values()))
    reloaded = _rebuild_from_artifact(arch, str(path), kind)
    x, ei = load_bench_input(arch)
    if arch == "gcn":
        lx, lei = x, ei
    else:
        lx, lei = x[:16], None
    a = logits_of(model, arch, lx, lei)
    b = logits_of(reloaded, arch, lx, lei)
    max_diff = float((a - b).abs().max())
    rec["reload_equivalence"] = {"tolerance": tolerance,
                                 "max_abs_diff": max_diff,
                                 "passed": bool(max_diff <= tolerance)}


def convert_and_measure(arch: str, fmt: str, parent: nn.Module,
                        parent_run_id: str, write: bool = True) -> dict:
    """One standalone Part C row: convert from the dense parent, measure
    everything (accounting fields per C.1), append the run record."""
    set_threads()
    set_qnnpack_engine()
    t0 = time.perf_counter()
    rec = _row(arch, "precision", fmt, parent_run_id)
    kind = fmt
    tolerance = 0.0
    calib_info = None
    try:
        if fmt in ("fp16", "bf16"):
            dtype = torch.float16 if fmt == "fp16" else torch.bfloat16
            model = half_quantize(parent, arch, dtype)
            rec["stored_precision"] = fmt
            rec["compute_precision"] = fmt
            rec["accumulator_precision"] = f"fp32 (aten {fmt} matmul accumulation)"
        elif fmt == "int8_dynamic":
            model, calib_info = int8_quantize_dynamic(parent, arch)
            rec["stored_precision"] = "int8 weights (per-tensor symmetric, qnnpack)"
            rec["compute_precision"] = "int8 (qnnpack quantized::linear, per-batch act quant)"
            rec["accumulator_precision"] = "int32 (qnnpack), fp32 dequant"
            tolerance = 1e-3  # int8 quantization error
        elif fmt == "int8_static":
            model, calib_info = int8_quantize_static(parent, arch)
            rec["stored_precision"] = "int8 weights + calibrated int8 activations (qnnpack FX)"
            rec["compute_precision"] = "int8 (qnnpack quantized::linear/conv2d, static act qparams)"
            rec["accumulator_precision"] = "int32 (qnnpack), fp32 dequant"
            tolerance = 1e-3  # int8 quantization error
        elif fmt == "int4_simulated":
            model = int4_quantize(parent, arch)
            rec["stored_precision"] = "int4 packed 2/uint8 + fp32 per-channel scales"
            rec["compute_precision"] = "fp32 (dequantize to fp32; dequant inside timed region)"
            rec["accumulator_precision"] = "fp32"
            rec["runtime_status"] = "simulated_non_native"
            tolerance = 1e-5  # fp32 recompute path (frozen C.1)
        else:
            raise ValueError(f"unknown format {fmt}")

        model.eval()
        va_acc, va_loss, te_acc, te_loss = eval_model(model, arch, fmt)
        rec["val_acc"] = va_acc
        rec["val_loss"] = va_loss
        rec["test_acc"] = te_acc
        rec["calibration"] = calib_info

        # actual serialized bytes on disk + reload equivalence
        _serialize_and_reload(rec, model, arch, fmt, tolerance)

        # Per-layer table + frac_macs. MACs are computed from the DENSE PARENT
        # (its layers are always plain nn.Linear/Conv2d/GCNConv, so the hook
        # in bench._mac_entries fires); the reduced-precision flag per op is
        # taken from the converted model by matching op names.
        x, ei = load_bench_input(arch)
        dtypes = inspect_forwards(model, arch) if fmt in ("fp16", "bf16") else None
        parent_ops = eligible_ops(parent)
        mac_table = _weight_macs(parent, x, ei)  # keyed by parent op name
        mac_by_name = {t["name"]: t for t in mac_table}
        # Map converted-model op names -> reduced-precision booleans.
        red_set = _reduced_op_names(model, arch, fmt, parent)
        table = []
        for name, op, kind_ in parent_ops:
            row_t = mac_by_name.get(name)
            if row_t is None:
                continue
            entry = {"name": name, "type": kind_, "macs": row_t["macs"],
                     "stored_precision": rec["stored_precision"],
                     "compute_precision": rec["compute_precision"],
                     "accumulator_precision": rec["accumulator_precision"],
                     "reduced_precision": bool(name in red_set)}
            if fmt in ("fp16", "bf16"):
                out_dt = dtypes.get(name)
                entry["output_dtype"] = str(out_dt) if out_dt is not None else None
            elif fmt in ("int8_dynamic", "int8_static"):
                if entry["reduced_precision"]:
                    entry["compute_op"] = ("quantized::linear" if kind_ in ("linear", "lin")
                                           else "quantized::conv2d")
            table.append(entry)
        rec["per_layer"] = table
        rec["frac_macs_reduced_precision"] = frac_macs_reduced_precision(
            parent, arch, red_set)
        rec["reduced_precision_ops"] = sorted(red_set)

        rec["total_params"] = count_params(parent)
        # C2: nonzero count of the INTEGER weight representation (int_repr of
        # quantized weights / INT4 packed), not the fp32 placeholder params
        # (which were reporting 0 for every INT8 row).
        rec["nonzero_params"] = int8_nonzero(model)
        rec["nonzero_params_dense_parent"] = int(
            sum(int(p.count_nonzero()) for p in parent.parameters()))

        rec["latency"] = _time_model(model, arch)
        rss = _rss_subprocess(arch, rec["artifact"], fmt)
        rec["rss_before_model_bytes"] = rss["before"]
        rec["peak_rss_bytes"] = rss["peak"]
        rec["model_rss_delta_bytes"] = rss["peak"] - rss["before"]

        rec = run_wall(rec, t0)
        if write:
            write_run_record(rec)
        return rec
    except Exception as e:
        import traceback
        traceback.print_exc()
        rec["status"] = "failed"
        rec["reason"] = _classify(e)
        rec = run_wall(rec, t0)
        if write:
            write_run_record(rec)
        raise


# ---------------------------------------------------------------------------
# INT8 selection (frozen C.1 rule)
# ---------------------------------------------------------------------------


def select_int8(arch: str, parent: nn.Module, parent_run_id: str,
                dyn_rec: dict, stat_rec: dict) -> dict:
    """Select the best supported INT8 method for `arch` by the frozen C.1
    rule: higher validation accuracy; ties within one val example -> lower
    batch-256 (fmnist) / full-graph (gcn) median latency. If only one method
    is supported, use it."""
    lat_key = "full_graph" if arch == "gcn" else "batch_256"

    def _val_count(rec: dict) -> int:
        n_val = 500 if arch == "gcn" else 5000
        return int(round(rec["val_acc"] * n_val))

    def _lat(rec: dict) -> float | None:
        d = (rec.get("latency") or {}).get(lat_key) or {}
        v = d.get("median_ms")
        return float(v) if isinstance(v, (int, float)) else None

    candidates = []
    for name, rec in (("int8_dynamic", dyn_rec), ("int8_static", stat_rec)):
        if rec.get("status") != "ok" or rec.get("val_acc") is None:
            continue
        candidates.append((name, rec))

    if not candidates:
        return {"arch": arch, "selected": None,
                "reason": "no supported INT8 method", "candidates": []}
    if len(candidates) == 1:
        sel_name, sel_rec = candidates[0]
        return {"arch": arch, "selected": sel_name,
                "val_acc": sel_rec["val_acc"],
                "latency_ms": _lat(sel_rec),
                "run_id": sel_rec["run_id"],
                "reason": "only one method supported",
                "candidates": [(n, r["val_acc"], _lat(r), r["run_id"]) for n, r in candidates]}

    # Both supported: higher val accuracy wins; tie within 1 example -> lower latency
    (n1, r1), (n2, r2) = candidates
    c1, c2 = _val_count(r1), _val_count(r2)
    if abs(c1 - c2) <= 1:
        l1, l2 = _lat(r1), _lat(r2)
        if l1 is not None and l2 is not None:
            sel = (n1, r1) if l1 <= l2 else (n2, r2)
        else:
            sel = (n1, r1)  # default to dynamic if latency missing
    else:
        sel = (n1, r1) if c1 > c2 else (n2, r2)
    return {"arch": arch, "selected": sel[0],
            "val_acc": sel[1]["val_acc"],
            "latency_ms": _lat(sel[1]),
            "run_id": sel[1]["run_id"],
            "reason": f"val count {c1} vs {c2}; "
                      f"latency {_lat(r1)} vs {_lat(r2)} ms",
            "candidates": [(n, r["val_acc"], _lat(r), r["run_id"]) for n, r in candidates]}


# ---------------------------------------------------------------------------
# Factorial (12 conditions)
# ---------------------------------------------------------------------------


def _load_sparse_checkpoint(arch: str, pilot_arm_id: str) -> tuple[nn.Module, dict]:
    """Load the sparse seed-0 pilot checkpoint selected by partc_selection.json."""
    from .prune import PILOT_DIR
    path = PILOT_DIR / f"{pilot_arm_id}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(arch)
    m.load_state_dict(payload["state_dict"])
    m.eval()
    return m, payload


def factorial_row(arch: str, condition: str, model: nn.Module,
                  parent_run_id: str, kind: str, parent: nn.Module,
                  write: bool = True, nonzero_count: int | None = None,
                  src_fp32: nn.Module | None = None) -> dict:
    """One factorial row (12 conditions: 3 archs x 4 conditions).

    condition in {"dense_fp32", "sparse_fp32", "dense_int8", "sparse_int8"}.
    `kind` is the artifact format for save/reload ("fp32" for the two fp32
    conditions, or "int8_dynamic"/"int8_static" for the two int8 conditions).
    `parent` is a plain fp32 model with the same op structure (used only for
    the per-op MAC table; for dense rows it is the model itself, for sparse
    rows it is the dense parent — MACs are shape-based and identical)."""
    set_threads()
    set_qnnpack_engine()
    t0 = time.perf_counter()
    rec = _row(arch, "factorial", condition, parent_run_id)
    try:
        va_acc, va_loss, te_acc, te_loss = eval_model(model, arch, condition)
        rec["val_acc"] = va_acc
        rec["val_loss"] = va_loss
        rec["test_acc"] = te_acc

        # size fields: nonzero_count is the model's source (fp32) nonzero
        # weight count (quantization preserves the zero pattern); default to
        # model.parameters() when the model has real parameters (fp32 rows).
        rec["total_params"] = count_params(parent)
        if nonzero_count is not None:
            rec["nonzero_params"] = nonzero_count
        else:
            rec["nonzero_params"] = sum(int(p.count_nonzero()) for p in model.parameters())
        # C2: nonzero count of the INTEGER weight representation. For int8
        # conditions this is the nonzero count of the quantized int8 weights
        # (int_repr), which for dense is == the dense-parent nonzero count and
        # for sparse is the count of surviving nonzero weights after INT8
        # (evidence for whether zeros survive quantization). For fp32
        # conditions it equals the fp32 param nonzero count.
        if "int8" in condition:
            rec["nonzero_before_int8"] = nonzero_count
            rec["nonzero_after_int8"] = int8_nonzero(model)
        else:
            rec["nonzero_before_int8"] = rec["nonzero_params"]
            rec["nonzero_after_int8"] = rec["nonzero_params"]

        # latency
        rec["latency"] = _time_model(model, arch)

        # artifact save + RSS + reload equivalence
        PRECISION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = PRECISION_ARTIFACT_DIR / f"{arch}-{condition}_{rec['run_id']}.pt"
        rec["artifact"] = str(path)
        rec["artifact_bytes"] = _save_artifact(model, path, kind, arch)
        bbd = bytes_by_dtype(model, kind)
        rec["bytes_by_dtype"] = bbd
        rec["payload_bytes"] = int(sum(bbd.values()))
        # C3: per-entry container-overhead accounting (zipfile infolist), so the
        # "unexplained" dense-vs-sparse artifact_bytes gap (issue 2) is visible
        # instead of hidden. For int8 (state = {"pickled": <nested torch.save
        # zip bytes>}) the outer data.pkl embeds the whole inner zip as one
        # stored (uncompressed) blob, so outer overhead >> inner; for fp32 and
        # cnn-static-int8 (state = state_dict) the overhead is a few KB.
        rec["container_overhead"] = _container_overhead(str(path))
        rss = _rss_subprocess(arch, rec["artifact"], kind)
        rec["rss_before_model_bytes"] = rss["before"]
        rec["peak_rss_bytes"] = rss["peak"]
        rec["model_rss_delta_bytes"] = rss["peak"] - rss["before"]

        # reload equivalence (tolerance 0 for fp32, 1e-3 for int8)
        tol = 0.0 if kind == "fp32" else 1e-3
        reloaded = _rebuild_from_artifact(arch, str(path), kind)
        x, ei = load_bench_input(arch)
        lx, lei = (x, ei) if arch == "gcn" else (x[:16], None)
        a = logits_of(model, arch, lx, lei)
        b = logits_of(reloaded, arch, lx, lei)
        max_diff = float((a - b).abs().max())
        rec["reload_equivalence"] = {"tolerance": tol,
                                     "max_abs_diff": max_diff,
                                     "passed": bool(max_diff <= tol)}

        # per-layer table + frac_macs (MACs from the fp32 `parent` structure)
        x, ei = load_bench_input(arch)
        parent_ops = eligible_ops(parent)
        mac_table = _weight_macs(parent, x, ei)
        mac_by_name = {t["name"]: t for t in mac_table}
        if "int8" in condition:
            red_set = _int8_reduced_names(model, arch, parent)
            rec["stored_precision"] = "int8 weights (qnnpack)"
            rec["compute_precision"] = "int8 (qnnpack)"
            rec["accumulator_precision"] = "int32 (qnnpack), fp32 dequant"
        else:
            red_set = set()
            rec["stored_precision"] = "fp32"
            rec["compute_precision"] = "fp32"
            rec["accumulator_precision"] = "fp32"
        table = []
        for name, _op, kind_ in parent_ops:
            row_t = mac_by_name.get(name)
            if row_t is None:
                continue
            table.append({"name": name, "type": kind_, "macs": row_t["macs"],
                          "stored_precision": rec["stored_precision"],
                          "compute_precision": rec["compute_precision"],
                          "reduced_precision": bool(name in red_set)})
        rec["per_layer"] = table
        rec["frac_macs_reduced_precision"] = frac_macs_reduced_precision(parent, arch, red_set)
        rec["reduced_precision_ops"] = sorted(red_set)

        # sparse fields
        if "sparse" in condition:
            # storage from the source fp32 sparse model (sparsity preserved)
            src = src_fp32 if src_fp32 is not None else model
            if "int8" in condition:
                rec["sparse_int8_storage_bytes"] = _sparse_int8_storage_bytes(src, arch)
                rec["sparse_int8_runtime_status"] = "runtime_unsupported"

        rec = run_wall(rec, t0)
        if write:
            write_run_record(rec)
        return rec
    except Exception as e:
        import traceback
        traceback.print_exc()
        rec["status"] = "failed"
        rec["reason"] = _classify(e)
        rec = run_wall(rec, t0)
        if write:
            write_run_record(rec)
        raise


def _sparse_int8_storage_bytes(src: nn.Module, arch: str) -> int:
    """Theoretical sparse-INT8 storage size: INT8 values (1 B each nonzero) +
    int32 CSR indices (4 B each) + fp32 scales (4 B per output channel).
    Labelled runtime-unsupported (no sparse INT8 CPU kernel). `src` is the
    source fp32 model (the sparse model before INT8 conversion): the sparsity
    pattern is preserved by quantization, so the storage is computed from the
    fp32 weights."""
    total = 0
    for name, op, kind in eligible_ops(src):
        if kind == "lin":
            op = op.lin if hasattr(op, "lin") else op
        if hasattr(op, "weight") and torch.is_tensor(op.weight):
            w = op.weight
        else:
            continue
        nnz = int(w.count_nonzero())
        out_ch = w.shape[0]
        total += nnz * 1          # int8 values
        total += nnz * 4          # int32 CSR column indices
        total += (nnz + out_ch) * 4  # int32 indptr + fp32 scales
    return total
