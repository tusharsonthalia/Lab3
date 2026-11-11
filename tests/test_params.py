"""Test (b): frozen parameter counts and the analytical MAC formula."""
import torch

from lab3.bench import estimate_macs
from lab3.models import build_model, count_params
from lab3.util import set_threads
from test_splits import _make_splits_once


def test_param_counts():
    assert count_params(build_model("mlp")) == 235146
    assert count_params(build_model("cnn")) == 117066
    assert count_params(build_model("gcn")) == 23063


def test_est_macs_mlp_and_cnn():
    """MACs must be computed from the actual layer shapes and match the frozen
    per-image totals: MLP=234,752, CNN=1,697,280 (conv2 output is 10x10, not
    12x12: 28 -> conv1 pad2 -> 28 -> pool -> 14 -> conv2 5x5 no pad -> 10)."""
    set_threads()
    x1 = torch.randn(1, 1, 28, 28)
    mlp = estimate_macs(build_model("mlp"), x1, None)
    cnn = estimate_macs(build_model("cnn"), x1, None)
    assert mlp["est_macs"] == 784 * 256 + 256 * 128 + 128 * 10 == 234752
    # conv2 output is 10x10 (28 -> conv1 pad2 -> 28 -> pool -> 14 -> conv2 5x5
    # no pad -> 10 -> pool -> 5), so conv2 = 16*32*25*10*10 per image.
    assert cnn["est_macs"] == (1 * 16 * 25 * 28 * 28 + 16 * 32 * 25 * 10 * 10
                               + 800 * 128 + 128 * 10) == 1697280
    assert cnn["est_macs"] == 1697280
    # Dense (freshly initialized) models have all weights nonzero.
    assert mlp["est_macs_nonzero"] == mlp["est_macs"]
    assert cnn["est_macs_nonzero"] == cnn["est_macs"]


def test_est_macs_gcn_formula():
    """GCN MACs on Cora: per layer projection = N*in*out (dense-X), aggregation
    = (E+N)*out; the weight is applied once per node, NOT per edge."""
    set_threads()
    _make_splits_once()
    from lab3.bench import load_bench_input
    x, ei = load_bench_input("gcn")
    N = x.shape[0]               # 2708
    E = ei.shape[1]              # 10556 directed edges
    x_nnz = int(x.count_nonzero())
    model = build_model("gcn")
    r = estimate_macs(model, x, ei, x_nnz=x_nnz)
    # dense-X convention: layer-1 projection N*1433*16 (~62M) + aggregation.
    expected = ((N * 1433 * 16) + ((E + N) * 16)
                + (N * 16 * 7) + ((E + N) * 7))
    assert r["est_macs"] == expected
    assert 62_000_000 < r["est_macs"] < 63_000_000
    # normalization scales reported separately, not in est_macs.
    assert r["gcn_norm_macs"] == 2 * (16 + 7) * (E + N)
    # sparse-input convention: layer-1 projection counted as nnz(X)*16.
    assert r["est_macs_sparse_input"] == (x_nnz * 16) + ((E + N) * 16) \
        + (N * 16 * 7) + ((E + N) * 7)
    assert r["est_macs_sparse_input"] < r["est_macs"]
