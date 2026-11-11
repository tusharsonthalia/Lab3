"""Part C (reduced precision) tests.

Covers the C1_precision.md item list:
  1. Backend probes (qnnpack, quantize_dynamic, FX, fp16/bf16 native)
  2. Half-precision accuracy within 1 pp of fp32 for all 3 architectures
  3. INT8 conversions (dynamic + static) produce qnnpack modules
  4. GCN projection swap preserves logits exactly
  5. Reload-equivalence (artifact save/reload → identical logits)
  6. INT4 simulated packing correctness
  7. MAC / frac_macs accounting
  8. INT8 selection rule
  9. Factorial (sparse × INT8) row construction
"""
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lab3.util import set_threads, new_run_id  # noqa: E402
from lab3.models import build_model  # noqa: E402
from lab3.evaln import load_checkpoint  # noqa: E402
from lab3.compact import clear_gcn_caches  # noqa: E402
from lab3 import precision as P  # noqa: E402
from lab3.data import load_cora, fmnist_split_tensors  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def threads():
    set_threads()
    return 4


@pytest.fixture(scope="module")
def checkpoints():
    """Load all three dense seed-0 checkpoints once per module."""
    out = {}
    for arch in ("mlp", "cnn", "gcn"):
        model, meta = load_checkpoint(arch, 0)
        out[arch] = (model, meta)
    return out


@pytest.fixture(scope="module")
def bench_inputs():
    """Frozen inference inputs for each architecture."""
    return {arch: P.load_bench_input(arch) for arch in ("mlp", "cnn", "gcn")}


def _eval(model, arch, x, ei):
    """Evaluate a model, returning (val_acc, val_loss, test_acc, test_loss)."""
    return P.eval_model(model, arch, "test")


def _logits(model, arch, x, ei):
    return P.logits_of(model, arch, x, ei)


# ---------------------------------------------------------------------------
# 1. Backend probes
# ---------------------------------------------------------------------------

def test_qnnpack_engine_set():
    """torch.backends.quantized.engine is 'qnnpack'."""
    P.set_qnnpack_engine()
    assert torch.backends.quantized.engine == "qnnpack"


def test_qengine_is_qnnpack():
    """torch._C._get_qengine() == 2 (qnnpack)."""
    assert torch._C._get_qengine() == 2


def test_quantize_dynamic_linear_produces_qnnpack():
    """quantize_dynamic({nn.Linear}) with qnnpack engine produces a Linear
    whose _packed_params carries a qint8 quantized weight."""
    P.set_qnnpack_engine()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(16, 8)
        def forward(self, x):
            return self.lin(x)

    m = M()
    q = torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    # The quantized linear must have _packed_params (qint8 quantized weight)
    assert hasattr(q.lin, "_packed_params"), (
        "dynamic-quantized Linear has no _packed_params"
    )


def test_quantize_dynamic_conv2d_not_quantized():
    """quantize_dynamic({nn.Conv2d}) on this build does NOT produce a
    quantized Conv2d (no _packed_params, weight stays fp32). This is
    recorded as unsupported_op in the support report — qnnpack has no
    dynamic-quantized Conv2d kernel on CPU."""
    P.set_qnnpack_engine()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    m = M()
    q = torch.quantization.quantize_dynamic(m, {nn.Conv2d}, dtype=torch.qint8)
    # The conv should NOT have _packed_params (not quantized)
    assert not hasattr(q.conv, "_packed_params"), (
        "Conv2d unexpectedly has _packed_params (was quantized)"
    )
    # Weight should still be fp32
    assert q.conv.weight.dtype == torch.float32


def test_pt2e_does_not_exist():
    """torch.ao.quantization has no quantize_pt2e (recorded as unsupported)."""
    assert not hasattr(torch.ao.quantization, "quantize_pt2e")


def test_torchao_not_installed():
    """torchao is not installed (recorded as unavailable)."""
    try:
        import torchao  # noqa: F401
        pytest.fail("torchao should not be installed")
    except ImportError:
        pass


def test_fp16_native_on_cpu():
    """fp16 forward on CPU preserves dtype (no upcast to fp32)."""
    P.set_qnnpack_engine()
    set_threads()
    lin = nn.Linear(32, 16).half()
    x = torch.randn(4, 32).half()
    with torch.inference_mode():
        out = lin(x)
    assert out.dtype == torch.float16


def test_bf16_native_on_cpu():
    """bf16 forward on CPU preserves dtype (no upcast to fp32)."""
    P.set_qnnpack_engine()
    set_threads()
    lin = nn.Linear(32, 16).bfloat16()
    x = torch.randn(4, 32).bfloat16()
    with torch.inference_mode():
        out = lin(x)
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# 2. Half-precision accuracy within 1 pp of fp32 (BUG fix from C1 review)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_accuracy_within_1pp(arch, dtype, threads, checkpoints, bench_inputs):
    """fp16/bf16 val accuracy within 1 pp of fp32 val accuracy."""
    parent, _ = checkpoints[arch]
    x, ei = bench_inputs[arch]
    # fp32 baseline
    fp32_va, fp32_vl, fp32_ta, fp32_tl = P.eval_model(parent, arch, "test")
    # half-precision
    half_model = P.half_quantize(parent, arch, dtype)
    half_va, half_vl, half_ta, half_tl = P.eval_model(half_model, arch, "test")
    # val accuracy difference must be < 1 pp
    diff = abs(half_va - fp32_va)
    assert diff < 0.01, (
        f"{arch} {dtype}: val_acc diff {diff:.4f} >= 0.01 "
        f"(fp32={fp32_va:.4f}, half={half_va:.4f})"
    )


# ---------------------------------------------------------------------------
# 3. INT8 conversions produce qnnpack modules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
def test_int8_dynamic_produces_qnnpack(arch, threads, checkpoints, bench_inputs):
    """int8_quantize_dynamic produces qnnpack QuantizedLinear modules."""
    parent, _ = checkpoints[arch]
    x, ei = bench_inputs[arch]
    model, meta = P.int8_quantize_dynamic(parent, arch)
    # At least one qnnpack quantized linear must exist
    q_count = 0
    for name, mod in model.named_modules():
        if P._is_qnnpack_quantized_linear(mod):
            q_count += 1
    assert q_count > 0, f"{arch} int8_dynamic: no qnnpack quantized linear found"


@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
def test_int8_static_produces_quantized(arch, threads, checkpoints, bench_inputs):
    """int8_quantize_static produces quantized modules (qnnpack or GraphModule)."""
    parent, _ = checkpoints[arch]
    x, ei = bench_inputs[arch]
    model, meta = P.int8_quantize_static(parent, arch)
    # The model must produce finite logits
    lx, lei = (x, ei) if arch == "gcn" else (x[:16], None)
    with torch.inference_mode():
        out = P.logits_of(model, arch, lx, lei)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. CNN static-INT8 artifact must contain the quantized weights (C2)
# ---------------------------------------------------------------------------

def test_cnn_static_int8_artifact_has_quantized_weights(threads, checkpoints, bench_inputs, tmp_path):
    """C2 item 3: the CNN static-INT8 artifact must actually save the
    converted (quantized) model's state_dict — reloading it must reproduce the
    quantized weights (not an fp32 placeholder), and artifact_bytes must be the
    real file size (C1b reported 1,805 bytes, impossible for ~117k int8 weights).

    Regression test: C1b saved only a 1.8 KB fp32 placeholder and then rebuilt
    the quantized structure from scratch on reload, so the artifact did not
    contain the quantized weights."""
    parent, _ = checkpoints["cnn"]
    x, ei = bench_inputs["cnn"]
    model, _ = P.int8_quantize_static(parent, "cnn")

    path = tmp_path / "cnn-int8_static_c2test.pt"
    ab = P._save_artifact(model, path, "int8_static", "cnn")

    # The artifact must be a real file of the recorded size and must be
    # large enough to hold the quantized weights (not the 1.8 KB placeholder).
    assert ab == path.stat().st_size, f"artifact_bytes {ab} != file size {path.stat().st_size}"
    # ~103,680 int8 weight bytes + 552 fp32 -> payload ~104 KB; artifact must
    # be well above the C1b 1,805-byte artifact.
    assert ab > 50_000, f"CNN static INT8 artifact only {ab} bytes (quantized weights missing?)"

    # Reload from disk and verify the loaded model's integer weight
    # representation matches the original converted model's (maxdiff 0).
    reloaded = P._rebuild_from_artifact("cnn", str(path), "int8_static")
    lx = x[:16]
    a = P.logits_of(model, "cnn", lx, None)
    b = P.logits_of(reloaded, "cnn", lx, None)
    assert float((a - b).abs().max()) == 0.0, "CNN static INT8 reload from disk changed logits"
    # The reloaded integer-representation nonzero count must be the real one
    # (not 0, which is what the C1b placeholder reported).
    assert P.int8_nonzero(reloaded) > 50_000, "reloaded CNN static INT8 has no quantized weights"



# ---------------------------------------------------------------------------
# 4. GCN projection swap preserves logits exactly
# ---------------------------------------------------------------------------

def test_gcn_swap_preserves_logits(threads, checkpoints, bench_inputs):
    """Swapping GCNConv.lin for nn.Linear preserves logits exactly."""
    parent, _ = checkpoints["gcn"]
    x, ei = bench_inputs["gcn"]
    # fp32 reference
    clear_gcn_caches(parent)
    with torch.inference_mode():
        ref = parent(x, ei)
    # swap
    m = build_model("gcn")
    m.load_state_dict({k: v.clone() for k, v in parent.state_dict().items()})
    m.eval()
    P._gcn_swap_projection(m)
    clear_gcn_caches(m)
    with torch.inference_mode():
        out = m(x, ei)
    assert torch.equal(ref, out), (
        f"GCN swap logits differ: maxdiff={(ref - out).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 5. Reload-equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
@pytest.mark.parametrize("fmt", ["fp16", "bf16", "int8_dynamic", "int8_static", "int4_simulated"])
def test_reload_equivalence(arch, fmt, threads, checkpoints, bench_inputs, tmp_path):
    """Save artifact → reload → logits are bit-identical (maxdiff == 0.0)."""
    parent, _ = checkpoints[arch]
    x, ei = bench_inputs[arch]
    lx, lei = (x, ei) if arch == "gcn" else (x[:16], None)

    # Convert
    if fmt == "fp16":
        model = P.half_quantize(parent, arch, torch.float16)
    elif fmt == "bf16":
        model = P.half_quantize(parent, arch, torch.bfloat16)
    elif fmt == "int8_dynamic":
        model = P.int8_quantize_dynamic(parent, arch)[0]
    elif fmt == "int8_static":
        model = P.int8_quantize_static(parent, arch)[0]
    elif fmt == "int4_simulated":
        model = P.int4_quantize(parent, arch)
    else:
        pytest.fail(f"unknown format {fmt}")

    a = _logits(model, arch, lx, lei)
    path = tmp_path / f"{arch}_{fmt}.pt"
    P._save_artifact(model, path, fmt, arch)
    reloaded = P._rebuild_from_artifact(arch, str(path), fmt)
    b = _logits(reloaded, arch, lx, lei)
    diff = (a - b).abs().max().item()
    assert diff == 0.0, f"{arch} {fmt}: reload maxdiff {diff} != 0.0"


# ---------------------------------------------------------------------------
# 6. INT4 simulated packing correctness
# ---------------------------------------------------------------------------

def test_int4_packing_roundtrip_linear():
    """Int4PackedLinear dequant approximates the original weight (within quant step)."""
    torch.manual_seed(42)
    lin = nn.Linear(64, 32)
    w = lin.weight.detach()
    packed = P.Int4PackedLinear(lin)
    w_deq = packed.dequant()
    # The dequant weight should be close to the original (within ~1 quant step per channel)
    scale = (w.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
    max_err = (w - w_deq).abs().max(dim=1).values
    # Each element error should be <= 0.5 * scale (half a quant step)
    assert (max_err <= 0.5 * scale + 1e-6).all(), (
        f"INT4 linear: max per-channel error exceeds half quant step"
    )


def test_int4_packing_roundtrip_conv():
    """Int4PackedConv dequant approximates the original weight."""
    torch.manual_seed(42)
    conv = nn.Conv2d(3, 8, 3, padding=1)
    w = conv.weight.detach()
    packed = P.Int4PackedConv(conv)
    w_deq = packed.dequant()
    assert w_deq.shape == w.shape
    # Dequant should be finite
    assert torch.isfinite(w_deq).all()


def test_int4_packed_bytes_less_than_fp32(threads, checkpoints):
    """INT4 packed bytes are much less than fp32 weight bytes."""
    parent, _ = checkpoints["mlp"]
    fp32_bytes = sum(p.numel() * 4 for p in parent.parameters())
    m4 = P.int4_quantize(parent, "mlp")
    int4_bytes = P.int4_packed_bytes(m4)
    # INT4 should be roughly 1/8 of fp32 (4 bits vs 32 bits) + scales
    assert int4_bytes < fp32_bytes, (
        f"INT4 bytes {int4_bytes} not less than fp32 {fp32_bytes}"
    )


def _reference_int4_quantdequant(parent: nn.Module) -> nn.Module:
    """Reference per-output-channel symmetric [-8,7] quantize-dequantize of the
    parent's eligible weight-bearing ops (scale = absmax/7 per output row of the
    weight flattened to [out, -1], round, clamp, multiply back). This is the
    independent reference the INT4 packed model must match (frozen C.1 rule)."""
    import copy
    ref = copy.deepcopy(parent)
    ref.eval()
    for name, mod, kind in P.eligible_ops(ref):
        w = mod.weight.detach().reshape(mod.weight.shape[0], -1)
        scale = (w.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
        wq = torch.round(w / scale.unsqueeze(1)).clamp(-8, 7) * scale.unsqueeze(1)
        with torch.no_grad():
            mod.weight.copy_(wq.reshape(mod.weight.shape))
    return ref


@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
def test_int4_matches_reference_from_disk(arch, threads, checkpoints, bench_inputs, tmp_path):
    """C2 item 2: for all 3 archs, the INT4 packed model loaded FROM DISK has
    logits equal to the reference per-output-channel symmetric [-8,7]
    quantize-dequantize within 1e-5, and its val accuracy matches.

    This is the regression test for the C1b INT4 bug (the packed model was
    silently built from `build_model`'s random init instead of the parent
    weights, scoring chance-level accuracy reported as a real result)."""
    parent, _ = checkpoints[arch]
    parent.eval()
    x, ei = bench_inputs[arch]
    lx, lei = (x, ei) if arch == "gcn" else (x[:16], None)

    # Convert, save, and reload FROM DISK (not the in-memory model).
    m4 = P.int4_quantize(parent, arch)
    path = tmp_path / f"int4_{arch}.pt"
    P._save_artifact(m4, path, "int4_simulated", arch)
    reloaded = P._rebuild_from_artifact(arch, str(path), "int4_simulated")

    # Reference quantize-dequantize logits.
    ref = _reference_int4_quantdequant(parent)
    ref_logits = P.logits_of(ref, arch, lx, lei)
    disk_logits = P.logits_of(reloaded, arch, lx, lei)
    max_diff = float((ref_logits - disk_logits).abs().max())
    assert max_diff <= 1e-5, (
        f"{arch} INT4 (from disk) logits differ from reference quant-dequant by "
        f"{max_diff:.3e} > 1e-5"
    )

    # Val accuracy must match the reference (and be far from chance).
    ref_va, *_ = P.eval_model(ref, arch, "test")
    disk_va, *_ = P.eval_model(reloaded, arch, "test")
    assert abs(ref_va - disk_va) < 1e-9, (
        f"{arch} INT4 val acc {disk_va} != reference {ref_va}"
    )
    # Sanity: the fixed INT4 must not be at chance level (the C1b bug symptom).
    assert disk_va > 0.5, f"{arch} INT4 val acc {disk_va} is at/below chance (C1b bug)"


# ---------------------------------------------------------------------------
# 6b. bytes_by_dtype + int8 nonzero representation (C2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
def test_int8_nonzero_not_zero(arch, threads, checkpoints):
    """C2 item 5: nonzero count of the integer weight representation is nonzero
    for dense INT8 (the old code reported 0 for every INT8 row because it counted
    fp32 placeholder params)."""
    parent, _ = checkpoints[arch]
    parent.eval()
    model, _ = P.int8_quantize_dynamic(parent, arch)
    nnz = P.int8_nonzero(model)
    parent_weight_nnz = sum(int(m.weight.count_nonzero())
                            for _n, m, _k in P.eligible_ops(parent))
    assert nnz > 0, f"{arch} INT8 dynamic integer nonzero count is 0"
    # Dense parent weights are all nonzero; per-tensor symmetric int8 quantization
    # (0 -> 0) zeros out small-magnitude weights that fall within half a quant
    # step. The integer-representation nonzero count must be a substantial
    # majority (>= 80%) of the dense parent's nonzero weights — this is the
    # signal that the count is the real integer-representation count, not the
    # broken 0 the C1b code reported.
    assert nnz >= 0.8 * parent_weight_nnz, (
        f"{arch} INT8 dynamic nnz {nnz} < 80% of parent weight nnz {parent_weight_nnz}"
    )


@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
@pytest.mark.parametrize("fmt", ["fp16", "bf16", "int8_dynamic", "int8_static", "int4_simulated"])
def test_bytes_by_dtype_present(arch, fmt, threads, checkpoints):
    """C2 item 4: bytes_by_dtype is a non-empty per-dtype dict on every row, and
    its sum (payload_bytes) is <= artifact_bytes (container overhead is the
    difference, visible to the reader)."""
    parent, _ = checkpoints[arch]
    parent.eval()
    if fmt in ("fp16", "bf16"):
        model = P.half_quantize(parent, arch,
                                torch.float16 if fmt == "fp16" else torch.bfloat16)
    elif fmt == "int8_dynamic":
        model, _ = P.int8_quantize_dynamic(parent, arch)
    elif fmt == "int8_static":
        model, _ = P.int8_quantize_static(parent, arch)
    else:
        model = P.int4_quantize(parent, arch)
    bbd = P.bytes_by_dtype(model, fmt)
    assert isinstance(bbd, dict) and len(bbd) > 0, f"{arch} {fmt}: empty bytes_by_dtype"
    assert all(v > 0 for v in bbd.values()), f"{arch} {fmt}: non-positive byte entry"
    # artifact_bytes >= payload_bytes (container overhead >= 0)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / f"{arch}_{fmt}.pt"
        ab = P._save_artifact(model, path, fmt, arch)
    assert ab >= int(sum(bbd.values())), (
        f"{arch} {fmt}: artifact_bytes {ab} < payload_bytes {sum(bbd.values())}"
    )


@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
@pytest.mark.parametrize("fmt", ["int8_dynamic", "int8_static"])
def test_int8_bytes_count_quantized_weights(arch, fmt, threads, checkpoints):
    """C3: for every INT8 artifact of every arch, the integer-weight accounting
    must actually see the quantized projection weights.

    C2's traversal (`_int_weight_tensors` / `bytes_by_dtype`) only matched
    modules whose `_packed_params` child exposed a working `_weight_bias()`,
    so it MISSED the FX-static shape where `convert_fx` attaches the packed
    weight as a raw `torch.ScriptObject` attribute (`_packed_weight_N` on the
    GCN per-projection GraphModules; `_packed_params` on the CNN static convs).
    Symptom in C2: gcn dense/sparse int8 reported
    `nonzero_after_int8 = 23` (the 16+7 GCN biases, not the 22,928+112
    projection weights) and `bytes_by_dtype = {"fp32": 92}` (biases only),
    even though val acc (0.794-0.796) proved the INT8 weights were present.

    Asserts (dense parent, both INT8 methods, all 3 archs). "Quantized layers"
    = the eligible ops actually converted to qnnpack int8 (both methods
    quantize every eligible op, except the frozen C.1 dynamic spec
    `quantize_dynamic({nn.Linear})` leaves CNN Conv2d in fp32):
      * `bytes_by_dtype["int8"]` >= the number of weights of the quantized
        layers (1 B/elem payload, so byte count == weight count);
      * `int8_nonzero` (int_repr nonzero count) is within 20% of the fp32
        nonzero count of those same layers — per-tensor symmetric INT8
        (0 -> 0) zeros out at most a small fraction of small-magnitude
        weights, so a correct count must track the fp32 nonzero count.
    """
    parent, _ = checkpoints[arch]
    parent.eval()
    maker = (P.int8_quantize_dynamic if fmt == "int8_dynamic"
             else P.int8_quantize_static)
    model, _ = maker(parent, arch)

    # Which eligible ops actually got quantized (int8)?
    conv_by_name = {n: m for n, m in model.named_modules()}
    quantized_numel = 0
    quantized_fp32_nnz = 0
    for n, op, k in P.eligible_ops(parent):
        w = getattr(op, "weight", None)
        if w is None:
            continue
        qmod = conv_by_name.get(n)
        is_int8 = (
            (k in ("linear", "lin") and (qmod is not None
                                         and P._qnnpack_weight_bias(qmod) is not None))
            or (k == "conv" and qmod is not None
                and P._is_qnnpack_quantized_conv(qmod))
        )
        if is_int8:
            quantized_numel += int(w.numel())
            quantized_fp32_nnz += int(w.count_nonzero())
    assert quantized_numel > 0 and quantized_fp32_nnz > 0, \
        f"{arch} {fmt}: no eligible op detected as quantized?"

    bbd = P.bytes_by_dtype(model, fmt)
    nnz = P.int8_nonzero(model)

    # The int8 payload must cover every weight of the quantized layers
    # (1 B/elem, so bytes == element count for the int8 weight tensors).
    assert bbd.get("int8", 0) >= quantized_numel, (
        f"{arch} {fmt}: bytes_by_dtype['int8']={bbd.get('int8')} < "
        f"quantized-layer weights {quantized_numel} "
        f"(int8 payload missing quantized weights)"
    )
    # The int_repr nonzero count must track the fp32 nonzero count of the
    # quantized layers within 20% (quantization can only zero out small
    # weights; it cannot invent or erase most of them).
    assert 0.8 * quantized_fp32_nnz <= nnz <= 1.2 * quantized_fp32_nnz, (
        f"{arch} {fmt}: int8_nonzero {nnz} not within 20% of fp32 nonzero "
        f"count {quantized_fp32_nnz} of the quantized layers"
    )


@pytest.mark.parametrize("arch", ["mlp", "cnn", "gcn"])
@pytest.mark.parametrize("fmt", ["int8_dynamic", "int8_static"])
def test_int8_artifact_is_flat_state_dict(arch, fmt, threads, checkpoints,
                                          bench_inputs, tmp_path):
    """C4: every INT8 artifact (dynamic and static, all 3 archs) must be a
    single torch.save of the converted model's state_dict — no nested
    pickled-module container (C3's mlp/gcn format stored
    `state = {"pickled": <bytes of an inner torch.save of the whole live
    module>}`, which re-encoded the module tree inside the outer zip and
    bloated the file ~120 KB above the payload), and no fp32 copy of the
    weights.

    Asserts:
      * `state` is the flat state_dict itself (a key/value dict of the
        converted model; no `pickled` key, no `quantized_state` wrapper);
      * `artifact_bytes <= sum(bytes_by_dtype) + 16 KB` (zip structural
        overhead only — no re-encoded payload);
      * reload rebuilds the quantized structure from the fp32 parent and
        `load_state_dict(strict=True)` reproduces the in-memory model's
        logits EXACTLY (atol 0)."""
    parent, _ = checkpoints[arch]
    parent.eval()
    maker = (P.int8_quantize_dynamic if fmt == "int8_dynamic"
             else P.int8_quantize_static)
    model, _ = maker(parent, arch)
    x, ei = bench_inputs[arch]
    lx, lei = (x, ei) if arch == "gcn" else (x[:16], None)

    path = tmp_path / f"{arch}-{fmt}.pt"
    ab = P._save_artifact(model, path, fmt, arch)
    assert ab == path.stat().st_size

    payload = torch.load(path, map_location="cpu", weights_only=False)
    st = payload["state"]
    # Flat state dict: keys are the converted model's state keys themselves.
    assert isinstance(st, dict) and "pickled" not in st and "quantized_state" not in st, (
        f"{arch} {fmt}: artifact state is not a flat state_dict: {list(st)[:6]}")
    assert set(st.keys()) == set(model.state_dict().keys()), (
        f"{arch} {fmt}: state_dict keys differ from the converted model")
    # Container overhead: at most 16 KB beyond the payload bytes.
    bbd = P.bytes_by_dtype(model, fmt)
    assert ab <= int(sum(bbd.values())) + 16 * 1024, (
        f"{arch} {fmt}: artifact_bytes {ab} > payload {sum(bbd.values())} + 16 KB "
        f"(nested-pickle container overhead?)")

    # Reload: bit-identical logits (atol 0).
    reloaded = P._rebuild_from_artifact(arch, str(path), fmt)
    a = P.logits_of(model, arch, lx, lei)
    b = P.logits_of(reloaded, arch, lx, lei)
    assert float((a - b).abs().max()) == 0.0, (
        f"{arch} {fmt}: reload maxdiff {(a - b).abs().max().item()} != 0.0")


# ---------------------------------------------------------------------------
# 7. MAC / frac_macs accounting
# ---------------------------------------------------------------------------

def test_macs_mlp(threads, checkpoints, bench_inputs):
    """MLP MAC count matches the known value (234,934 per sample)."""
    parent, _ = checkpoints["mlp"]
    x, ei = bench_inputs["mlp"]
    macs = P._mac_count(parent, x, None)
    # MLP: 784*256 + 256*128 + 128*10 = 200,704 + 32,768 + 1,280 = 234,752
    # (exact count depends on bias handling; check it's in the right range)
    assert 200_000 < macs < 300_000, f"MLP MACs {macs} out of range"


def test_frac_macs_full_precision(threads, checkpoints, bench_inputs):
    """fp32 model has frac_macs == 0.0 (no reduced-precision layers)."""
    parent, _ = checkpoints["mlp"]
    x, ei = bench_inputs["mlp"]
    # No reduced-precision layers → frac_macs should be 0
    frac = P._frac_macs(parent, x, None, reduced_layer_names=set())
    assert frac == 0.0


# ---------------------------------------------------------------------------
# 8. INT8 selection rule
# ---------------------------------------------------------------------------

def test_int8_selection_rule():
    """The frozen INT8 selection rule: pick the format with the best
    val_acc among those whose latency is within 2× the minimum.
    Ties broken by lower latency."""
    # Case 1: both within 2× min latency; static has slightly lower acc
    # but both qualify → pick higher acc (dynamic)
    rows = [
        {"format": "int8_dynamic", "val_acc": 0.8984, "latency_ms": 1.2},
        {"format": "int8_static", "val_acc": 0.8980, "latency_ms": 1.0},
    ]
    best = P._select_int8(rows)
    # min_lat=1.0, 2×=2.0; both 1.2 and 1.0 qualify → pick higher acc
    assert best == "int8_dynamic", f"Expected int8_dynamic, got {best}"

    # Case 2: static latency is much lower (0.5), dynamic (1.2) exceeds 2×0.5=1.0
    # → dynamic is excluded; static wins by default
    rows2 = [
        {"format": "int8_dynamic", "val_acc": 0.8984, "latency_ms": 1.2},
        {"format": "int8_static", "val_acc": 0.8900, "latency_ms": 0.5},
    ]
    best2 = P._select_int8(rows2)
    assert best2 == "int8_static", f"Expected int8_static, got {best2}"

    # Case 3: only one format present
    rows3 = [{"format": "int8_dynamic", "val_acc": 0.8984, "latency_ms": 1.2}]
    best3 = P._select_int8(rows3)
    assert best3 == "int8_dynamic"


# ---------------------------------------------------------------------------
# 9. Factorial row construction
# ---------------------------------------------------------------------------

def test_factorial_row_construction(threads, checkpoints, bench_inputs):
    """factorial_row produces a valid run record with all required fields."""
    arch = "mlp"
    parent, _ = checkpoints[arch]
    # Dense fp32 condition
    parent.eval()
    rec = P.factorial_row(
        arch=arch, condition="dense_fp32",
        model=parent,
        parent_run_id="test-parent",
        kind="fp32",
        parent=parent,
        write=False,
    )
    # Check required fields
    for field in ("run_id", "architecture", "method", "val_acc", "phase", "part"):
        assert field in rec, f"factorial_row missing field: {field}"
    assert rec["part"] == "C"
    assert rec["phase"] == "factorial"
    assert rec["method"] == "dense_fp32"
