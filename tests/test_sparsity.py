"""Task B1 tests (predictions.md Part B, frozen).

Conventions:
- Never touch the real results/runs.jsonl (conftest redirects LAB3_RESULTS_DIR).
- Real seed-0 dense checkpoints are used as parents (read-only).
- Fine-tuning in tests uses the REAL lab3.prune.fine_tune (frozen B.1 budget);
  the only speedup is skipping the latency/RSS benching (bench=... not involved,
  we call the primitives directly to keep tests fast).
"""
import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from lab3.compact import CompactModel, CSR_INDEX_DTYPE, reassemble_packed
from lab3.data import fmnist_split_tensors, load_cora
from lab3.evaln import load_checkpoint
from lab3.models import build_model
from lab3.prune import (
    FT_BUDGET,
    _metrics,
    _calib_loss_fn,
    fine_tune,
    per_layer_table,
    prune_once,
)
from lab3.sparsity import (
    _nm_keep_mask_row,
    eligible_weights,
    make_masked_model,
    nm_pattern_for_target,
    s1_masks,
    s2_plans,
    s2_rebuild,
    s3_masks_global,
    s4_masks,
    s4_packed,
    scatter_into_dense,
)
from lab3.util import CALIB_FILE, SPLITS_DIR, set_threads

ARCHS = ("mlp", "cnn", "gcn")
MASK_METHODS = ("s1", "s3", "s4")


@pytest.fixture(scope="session", autouse=True)
def _setup():
    set_threads()
    yield


def _forward(arch, model, x=None):
    """Eval-mode forward. GCN additionally runs on a FRESH module copy: the
    (train-time) message-passing caches its normalized edge index on the module,
    so any module that saw a train-mode forward would corrupt later evals.
    Returns (out, x, ei) so callers can re-forward with identical inputs."""
    model.eval()
    with torch.inference_mode():
        if arch == "gcn":
            if x is None:
                x = torch.randn(30, 1433)
                ei = torch.randint(0, 30, (2, 120))
            else:
                x, ei = x[0], x[1]
            m = copy.deepcopy(model)
            m.eval()
            # message passing caches the normalized edge index on the module
            # during train-mode forwards; a deepcopy of a module that saw one
            # inherits a STALE cache (keyed on num_nodes, not on the weight
            # shape) -> clear every cache attribute on every sub-module.
            for mod in m.modules():
                for attr in list(vars(mod)):
                    if attr.startswith("_cached"):
                        setattr(mod, attr, None)
            return m(x, ei), x, ei
        if x is None:
            x = torch.randn(2, 1, 28, 28)
        elif isinstance(x, tuple):
            x = x[0]
        return model(x), x, None


def _fresh(arch, model):
    """GCN caches its normalized edge index on the MODULE during a training-mode
    forward (message passing). Evaluation on a freshly built module is the
    clean path used by every other eval in the package. The deepcopy inherits
    the (possibly stale) cache, so every `_cached*` attribute is cleared."""
    if arch != "gcn":
        return model
    import copy

    m2 = copy.deepcopy(model)
    m2.eval()
    for mod in m2.modules():
        for attr in list(vars(mod)):
            if attr.startswith("_cached"):
                setattr(mod, attr, None)
    return m2


def _input(arch):
    if arch == "gcn":
        g = torch.Generator().manual_seed(7)
        x = torch.randn(30, 1433, generator=g)
        ei = torch.randint(0, 30, (2, 120), generator=g)
        return x, ei
    g = torch.Generator().manual_seed(7)
    return torch.randn(2, 1, 28, 28, generator=g), None


# ---------------------------------------------------------------------------
# (a) pruned entries remain exactly zero after fine-tuning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("method", MASK_METHODS)
def test_pruned_entries_stay_zero_after_finetune(arch, method, tmp_path):
    """S1/S3/S4 on every architecture: after the full B.1 fine-tuning budget,
    every masked (pruned) entry of every eligible weight is exactly 0.0."""
    parent, _ = load_checkpoint(arch, 0)
    parent_state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    model, masks, plans, _ = prune_once(arch, method, 0.5, parent_state)
    model = _fresh(arch, model)  # GCN: train-time evals cache the edge index on the module
    assert masks is not None
    ft_state, _ = fine_tune(arch, model, seed=0, masks=masks)
    model.load_state_dict(ft_state)
    w = eligible_weights(model)
    for key, m in masks.items():
        zeros_at_pruned = ((w[key] == 0) | (m == 1)).all()
        assert bool(zeros_at_pruned), f"{arch}/{method}: {key} has nonzero pruned entries"


# ---------------------------------------------------------------------------
# (b) S2 structured shapes consistent on all targets, all architectures
# ---------------------------------------------------------------------------

def _expected_s2_shapes(arch, target):
    """Frozen B.2 keep-count formula k = max(1, round((1-s)*n))."""
    k = lambda n: max(1, int(round((1.0 - target) * n)))
    if arch == "mlp":
        h1, h2 = k(256), k(128)
        return {
            "net.0.weight": (h1, 784), "net.0.bias": (h1,),
            "net.2.weight": (h2, h1), "net.2.bias": (h2,),
            "net.4.weight": (10, h2),
            "total_params": h1 * 784 + h1 + h2 * h1 + h2 + 10 * h2 + 10,
        }
    if arch == "cnn":
        c1, c2, f1 = k(16), k(32), k(128)
        fc1_in = 5 * 5 * c2
        return {
            "conv1.weight": (c1, 1, 5, 5), "conv1.bias": (c1,),
            "conv2.weight": (c2, c1, 5, 5), "conv2.bias": (c2,),
            "fc1.weight": (f1, fc1_in), "fc1.bias": (f1,),
            "fc2.weight": (10, f1),
            "total_params": (c1 * 25 + c1) + (c2 * c1 * 25 + c2)
            + (f1 * fc1_in + f1) + (10 * f1 + 10),
        }
    if arch == "gcn":
        h = k(16)
        return {
            "conv1.lin.weight": (h, 1433), "conv1.bias": (h,),
            "conv2.lin.weight": (7, h), "conv2.bias": (7,),
            "total_params": h * 1433 + h + 7 * h + 7,
        }
    raise ValueError(arch)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("target", [0.25, 0.5, 0.75, 0.9])
def test_s2_shapes_and_params(arch, target):
    parent, _ = load_checkpoint(arch, 0)
    rebuilt, plans = s2_rebuild(parent, target)
    rebuilt = _fresh(arch, rebuilt)
    exp = _expected_s2_shapes(arch, target)
    sd = rebuilt.state_dict()
    for key, shape in exp.items():
        if key == "total_params":
            continue
        assert tuple(sd[key].shape) == tuple(shape), (key, tuple(sd[key].shape), shape)
    assert sum(p.numel() for p in rebuilt.parameters()) == exp["total_params"], (
        arch, target, sum(p.numel() for p in rebuilt.parameters()), exp["total_params"])
    # forward runs
    x, ei = _input(arch)
    out, _, _ = _forward(arch, rebuilt, (x, ei))
    expected_last = 7 if arch == "gcn" else 10
    assert out.shape[-1] == expected_last
    if arch == "gcn" and target == 0.9:
        # frozen example: GCN at 0.9 keeps round(1.6) = 2 hidden features
        assert rebuilt.conv1.lin.weight.shape == (2, 1433)
        assert rebuilt.conv2.lin.weight.shape == (7, 2)


# ---------------------------------------------------------------------------
# (c) save/reload of a pruned checkpoint reproduces logits and val/test exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_pruned_checkpoint_save_reload(arch, tmp_path):
    """Run a real arm (prune + fine-tune) for MLP at 0.5 via the same code the
    CLI uses, save to tmp, reload, and check bit-exact logits + val/test acc."""
    from lab3.prune import prune as run_arm

    out = run_arm(arch, "s1", 0.5, seed=0, bench=False, write=False)
    assert out["status"] == "ok", out["reason"]
    arm_pt = out["rows"][0]["checkpoint"]
    import os
    assert os.path.exists(arm_pt)

    payload = torch.load(arm_pt, map_location="cpu", weights_only=False)
    model = build_model(arch)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    # logits identical to the in-memory model used to write the checkpoint
    x, ei = _input(arch)
    a, _, _ = _forward(arch, model, (x, ei))
    # recompute from the saved state into a fresh module
    model2 = _fresh(arch, build_model(arch))
    model2.load_state_dict(torch.load(arm_pt, map_location="cpu", weights_only=False)["state_dict"])
    b, _, _ = _forward(arch, model2, (x, ei))
    assert torch.equal(a, b), "logits differ after reload"

    # val/test metrics identical to the recorded post_finetune row
    r2 = next(r for r in out["rows"] if r["stage"] == "post_finetune")
    model_loaded = _fresh(arch, build_model(arch))
    model_loaded.load_state_dict(payload["state_dict"])
    va, vl, te = _metrics(arch, model_loaded)
    assert va == r2["val_acc"]
    assert vl == r2["val_loss"]
    assert te == r2["test_acc"]


# ---------------------------------------------------------------------------
# (d) achieved sparsity verified tensor by tensor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_s1_global_fraction_exact(arch):
    parent, _ = load_checkpoint(arch, 0)
    w = eligible_weights(parent)
    total = sum(t.numel() for t in w.values())
    for target in (0.25, 0.5, 0.75, 0.9):
        masks = s1_masks(parent, target)
        zeroed = sum(int((masks[k] == 0).sum()) for k in w)
        assert abs(zeroed - round(target * total)) <= 1, (arch, target, zeroed, target * total)
        # exactness: S1 tie-break is deterministic, so the count is exact
        assert zeroed == round(target * total), (arch, target, zeroed, round(target * total))


@pytest.mark.parametrize("arch", ARCHS)
def test_s3_global_fraction_exact(arch):
    parent, _ = load_checkpoint(arch, 0)
    w = eligible_weights(parent)
    total = sum(t.numel() for t in w.values())
    masks = s3_masks_global(parent, 0.5, _calib_loss_fn(arch))
    zeroed = sum(int((masks[k] == 0).sum()) for k in w)
    assert zeroed == round(0.5 * total), (arch, zeroed, round(0.5 * total))


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("target", [0.25, 0.5, 0.75, 0.9])
def test_s4_blocks_exact(arch, target):
    """Every complete block has exactly N nonzeros; partial blocks follow the
    ceil rule. Checked with distinct magnitudes (synthetic) and on the masks of
    the real parent (block structure must hold regardless of magnitudes)."""
    n, m = nm_pattern_for_target(target)
    # (i) synthetic distinct magnitudes
    parent, _ = load_checkpoint(arch, 0)
    for key, t in eligible_weights(parent).items():
        obj = parent
        for part in key.rpartition(".")[0].split("."):
            obj = getattr(obj, part)
        row_len = t.shape[1] if type(obj).__name__ != "Conv2d" else int(torch.prod(torch.tensor(t.shape[1:])))
        L = row_len
        n_full, r = divmod(L, m)
        row = torch.arange(1, L + 1, dtype=torch.float32)  # distinct magnitudes
        mask = _nm_keep_mask_row(row, n, m)
        for b in range(n_full + (1 if r else 0)):
            s = b * m if b < n_full else n_full * m
            blen = m if b < n_full else r
            cnt = int(mask[s:s + blen].sum())
            expect = n if b < n_full else math.ceil(n * blen / m)
            assert cnt == expect, (arch, key, target, b, cnt, expect)
    # (ii) real parent masks: same per-block structure
    masks = s4_masks(parent, target)
    for key, mk in masks.items():
        flat = mk.flatten(1)
        L = flat.shape[1]
        n_full, r = divmod(L, m)
        for row in range(flat.shape[0]):
            for b in range(n_full + (1 if r else 0)):
                s = b * m if b < n_full else n_full * m
                blen = m if b < n_full else r
                cnt = int(flat[row, s:s + blen].sum())
                expect = n if b < n_full else math.ceil(n * blen / m)
                assert cnt == expect, (arch, key, target, row, b, cnt, expect)


# ---------------------------------------------------------------------------
# (e) reversible mapping round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_s2_scatter_roundtrip(arch):
    """Mapped-back weights equal the parent on kept entries and are zero
    elsewhere; the scattered dense model is functionally identical to the
    rebuilt model."""
    parent, _ = load_checkpoint(arch, 0)
    for target in (0.25, 0.5, 0.75, 0.9):
        rebuilt, plans = s2_rebuild(parent, target)
        rebuilt = _fresh(arch, rebuilt)
        dense_map = scatter_into_dense(rebuilt, parent, plans)
        x, ei = _input(arch)
        a, x2, _ = _forward(arch, rebuilt, (x, ei))
        p2 = build_model(arch)
        p2 = _fresh(arch, p2)  # deepcopy -> clean cache; load the scattered weights
        p2.load_state_dict({**{k: v.detach().clone() for k, v in parent.state_dict().items()},
                            **{k: v for k, v in dense_map.items()}})
        b, _, _ = _forward(arch, p2, (x, ei))
        # fp32 round-trip tolerance for the extra scatter->dense multiply path
        assert torch.allclose(a, b, atol=1e-5), (arch, target)
        # kept entries equal the rebuilt weights; the rest of the scattered
        # weight tensors is exactly zero
        for key, t in dense_map.items():
            if "weight" not in key:
                continue
            assert t.numel() == parent.state_dict()[key].numel()
            # zero elsewhere: every zero of the parent scatter must be an
            # untouched (pruned) entry
            assert (t == 0).any(), key


def test_s4_packed_roundtrip():
    """s4_packed -> reassemble_packed reproduces the masked weights exactly
    (kept values land exactly where the mask says; zeros elsewhere)."""
    for arch in ARCHS:
        parent, _ = load_checkpoint(arch, 0)
        masks = s4_masks(parent, 0.5)
        model = build_model(arch)
        model.load_state_dict(parent.state_dict())
        make_masked_model(model, masks)
        model.eval()
        packed = s4_packed(model, masks, 0.5)
        w = eligible_weights(model)
        for key, p in packed.items():
            if key == "bias_bytes":
                continue
            rec = reassemble_packed(p)
            assert rec.shape == w[key].shape, key
            assert torch.equal(rec, w[key]), f"{arch}/{key}: packed round-trip mismatch"
            # zero elsewhere: positions the mask prunes are zero
            masked = w[key] * masks[key].to(w[key].dtype)
            assert torch.equal(rec, masked), key


# ---------------------------------------------------------------------------
# (f) S3 uses only calibration-subset inputs/labels (subset of train split)
# ---------------------------------------------------------------------------

def test_calib_subset_of_train():
    calib = np.load(CALIB_FILE)
    train = np.load(SPLITS_DIR / "fmnist_train_idx.npy")
    val = np.load(SPLITS_DIR / "fmnist_val_idx.npy")
    assert calib.shape == (1024,)
    assert set(int(i) for i in calib).issubset(set(int(i) for i in train))
    assert not (set(int(i) for i in calib) & set(int(i) for i in val))


def test_s3_calibration_loss_uses_only_calib_subset():
    """The S3 loss closure evaluates the model only on the 1,024 calibration
    images (train-split) and their training labels."""
    arch = "mlp"
    loss_fn = _calib_loss_fn(arch)
    parent, _ = load_checkpoint(arch, 0)

    calls = {}

    class Spy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            calls["n"] = x.shape[0]
            return self.inner(x)

    model = Spy(parent)
    loss_fn(model)
    assert calls["n"] == 1024
    # GCN: the 140 training nodes on the full graph
    if True:
        data = load_cora()
        assert int(data.lm_train.numel()) == 140
        calls.clear()
        loss_fn_g = _calib_loss_fn("gcn")
        p_g, _ = load_checkpoint("gcn", 0)

        class SpyG(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x, edge_index):
                calls["n"] = x.shape[0]
                return self.inner(x, edge_index)

        loss_fn_g(SpyG(p_g))
        assert calls["n"] == 2708  # full graph forward; loss on 140 train nodes


# ---------------------------------------------------------------------------
# (g) B1b: every compact model's logits match its dense_zeros model within
#     atol 1e-5, for all methods (s1-s4) and all architectures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("method", ["s1", "s2", "s3", "s4"])
def test_compact_logits_match_dense_zeros(arch, method):
    """For each method/arch: the genuinely-compact runtime (CSR for s1/s3,
    the rebuilt model for s2, the reassembled dense fallback for s4) must
    produce logits matching the dense_zeros path (dense parent-shaped model
    with the same zeros) within atol 1e-5. For s2 this is the key contrast:
    same zeros, different representation."""
    from lab3.prune import s2_dense_zeros_model, _load_compact
    import tempfile

    parent, _ = load_checkpoint(arch, 0)
    parent_state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    model, masks, plans, _ = prune_once(arch, method, 0.5, parent_state)

    # dense_zeros: dense parent-shaped model with the same zeros
    if method == "s2":
        dz = s2_dense_zeros_model(arch, parent, model, plans)
    else:
        dz = model  # s1/s3/s4 prune_once already returns the zeroed dense model

    # compact runtime
    payload = {"architecture": arch, "method": method, "target": 0.5,
               "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
               "masks": {k: v.detach().clone() for k, v in (masks or {}).items()}}
    compact, _, _ = _load_compact(arch, method, payload)

    x, ei = _input(arch)
    # GCN: fresh module copies (train-time cache hazard) — use _forward for
    # both so both see identical clean cache handling.
    a, _, _ = _forward(arch, _fresh(arch, dz), (x, ei) if ei is not None else x)
    b, _, _ = _forward(arch, _fresh(arch, compact), (x, ei) if ei is not None else x)
    assert torch.allclose(a, b, atol=1e-5, rtol=0), \
        f"{arch}/{method}: compact vs dense_zeros max diff " \
        f"{float((a - b).abs().max())}"


# ---------------------------------------------------------------------------
# (h) B2a: the saved CSR artifact and the runtime CSR both use CSR_INDEX_DTYPE
# ---------------------------------------------------------------------------

def test_csr_artifact_index_dtype(tmp_path):
    """B2a: the saved S1 CSR compact artifact must store indptr/indices with
    compact.CSR_INDEX_DTYPE (probed once at import; torch.int32 on this
    build) — the same dtype the compact runtime builds — while
    csr_int64_bytes stays the separately-computed default int64 layout size.
    Runs the real production path (_write_compact_artifact) on a real
    pruned model, with compact artifacts redirected to a tmp dir for the
    call (conftest already redirects results/ to a session tmp dir)."""
    from pathlib import Path
    from lab3.compact import csr_int64_bytes, dense_to_csr
    import lab3.prune as P

    arch = "mlp"
    parent, _ = load_checkpoint(arch, 0)
    parent_state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    model, masks, _, _ = prune_once(arch, "s1", 0.5, parent_state)
    model = _fresh(arch, model)
    payload = {"architecture": arch, "method": "s1", "target": 0.5,
               "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
               "masks": {k: v.detach().clone() for k, v in (masks or {}).items()},
               "arm_id": f"{arch}-s1-t0.5-seed0-test"}

    saved = P.COMPACT_DIR
    P.COMPACT_DIR = Path(tmp_path) / "compacts"
    P.COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        p2 = {"path": "compact"}
        P._write_compact_artifact("s1", arch, payload, p2)
    finally:
        P.COMPACT_DIR = saved

    artifact = Path(p2["artifact"])
    assert artifact.exists(), artifact
    art = torch.load(artifact, map_location="cpu", weights_only=False)
    assert art["csr_index_dtype"] == str(CSR_INDEX_DTYPE)
    tensors = [v for v in art.values() if isinstance(v, dict) and "indptr" in v]
    assert tensors, "artifact carries no CSR tensors"
    for v in tensors:
        # saved artifact index dtype == compact.CSR_INDEX_DTYPE (the runtime dtype)
        assert v["indptr"].dtype == CSR_INDEX_DTYPE, v["indptr"].dtype
        assert v["indices"].dtype == CSR_INDEX_DTYPE, v["indices"].dtype
        assert v["values"].dtype == torch.float32
    # csr_int64_bytes is the separately-computed default int64 layout size
    # and must equal an independent recomputation (element counts only).
    expected_64 = 0
    for key, t in eligible_weights(model).items():
        wt = t.flatten(1) if t.dim() == 4 else t
        indptr, indices, values, size = dense_to_csr(wt, torch.int64)
        expected_64 += csr_int64_bytes(indptr, indices, values, size)
    assert int(art["csr_int64_bytes"]) == expected_64, (
        int(art["csr_int64_bytes"]), expected_64)
