"""Test (c): the REAL lab3 evaluation and bench timing code never update
weights. We load the actual saved seed-0 checkpoint, snapshot its
state_dict (bit-exact), requires_grad flags and grad fields, then call the
genuine ``lab3.evaln.eval`` and ``lab3.bench.bench`` functions on it and assert
the checkpoint parameters are bit-identical with grad still None. Covers all
three architectures."""
import pytest
import torch
import torch.nn as nn

from lab3.bench import bench
from lab3.data import fmnist_split_tensors, load_cora
from lab3.evaln import eval as do_eval, load_checkpoint
from lab3.models import build_model
from lab3.util import set_threads
from test_splits import _make_splits_once

ARCHS = ("mlp", "cnn", "gcn")


def _snapshot(model: nn.Module) -> dict:
    """Bit-exact parameter bytes plus requires_grad flags and grad fields."""
    snap = {
        "params": {k: v.view(torch.uint8).clone()
                   for k, v in model.state_dict().items()},
        "requires_grad": {n: bool(p.requires_grad)
                          for n, p in model.named_parameters()},
        "grad": {n: (None if p.grad is None else p.grad.view(torch.uint8).clone())
                 for n, p in model.named_parameters()},
    }
    return snap


def _assert_unchanged(before: dict, model: nn.Module) -> None:
    after_sd = model.state_dict()
    assert set(before["params"]) == set(after_sd), "state_dict keys changed"
    for k, b in before["params"].items():
        a = after_sd[k]
        assert a.dtype == b.view(dtype=a.dtype).dtype
        assert torch.equal(b, a.view(torch.uint8)), f"tensor {k} changed bytes"
    for n, p in model.named_parameters():
        assert bool(p.requires_grad) == before["requires_grad"][n], \
            f"requires_grad changed for {n}"
        gb = before["grad"][n]
        ga = p.grad
        if gb is None:
            assert ga is None, f"grad materialized for {n}"
        else:
            assert ga is not None
            assert torch.equal(gb, ga.view(torch.uint8)), f"grad changed for {n}"


@pytest.mark.parametrize("arch", ARCHS)
def test_real_eval_never_updates(arch):
    """The real lab3.evaln.eval on the loaded checkpoint leaves weights bit-identical
    and grad/requires_grad untouched."""
    set_threads()
    _make_splits_once()
    # Preload data so eval does the same work as the production call.
    if arch == "gcn":
        load_cora()
    else:
        fmnist_split_tensors()
    model, _ = load_checkpoint(arch, seed=0)
    before = _snapshot(model)
    do_eval(arch, seed=0)  # real evaluation function; appends a run record
    _assert_unchanged(before, model)


@pytest.mark.parametrize("arch", ARCHS)
def test_real_bench_never_updates(arch):
    """The real lab3.bench.bench timing/protocol on the loaded checkpoint leaves
    weights bit-identical and grad/requires_grad untouched."""
    set_threads()
    _make_splits_once()
    model, _ = load_checkpoint(arch, seed=0)
    before = _snapshot(model)
    bench(arch, seed=0)  # real bench function; appends a run record
    _assert_unchanged(before, model)
