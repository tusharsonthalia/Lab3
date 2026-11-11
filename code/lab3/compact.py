"""Part B path 2 — genuinely compact runtime modules (predictions.md B.4).

S1/S3: dense-shaped math executed as CSR sparse x dense matmul.

B1b review (frozen design, corrected implementation):

* CSR orientation: every weight W [out, in] is stored as a CSR matrix with
  shape (out, in). The optimized CPU path is ``torch.sparse.mm`` in the
  **sparse x dense** orientation, so each layer computes
  ``y = (W_csr @ x^T)^T`` (never ``x @ W_csr^T``, which dispatches dense x
  sparse and is not the optimized path).
* Index dtype: CSR indices use int32 when ``torch.sparse.mm`` supports it on
  CPU (probed once at import; ``csr_index_dtype`` is recorded per artifact).
  The equivalent default int64 layout size is recorded as
  ``csr_int64_bytes``.
* CSRConv2d: no Python loop over the batch. ``F.unfold`` produces columns
  (B, in*k*k, L); reshaping to (in*k*k, B*L) and one sparse mm computes the
  whole batch (pixel order of F.unfold is (y, x) inside each image, which is
  exactly Conv2d's output order).
* CSRGCNConv: a GCNConv subclass in which ONLY the per-edge projection
  differs (``message`` does the sparse x dense projection); the
  gcn_norm edge-weight computation, propagation and aggregation are exactly
  the PyG GCNConv code (same ``propagate``/``gcn_norm``/aggr_module).
* S4: the packed file stores only the kept values (row-major) plus ONE uint8
  in-block position per kept value; row and block are implied by the fixed
  N:M pattern and the frozen partial-block ceil rule.
  ``nm_bitpacked_bytes_theoretical`` = values*4 + ceil(nnz*ceil(log2 M)/8)
  + biases is recorded and labelled theoretical.
"""
from __future__ import annotations

import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*Sparse CSR.*")
warnings.filterwarnings("ignore", message=".*Sparse invariant.*")

from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# CSR construction (orientation (out, in); sparse x dense via torch.sparse.mm)
# ---------------------------------------------------------------------------

def _csr_components(w: torch.Tensor, index_dtype: torch.dtype):
    """Dense [out, in] -> (crow, col, values) row-major CSR components."""
    rows, cols = w.nonzero().t()
    crow = torch.zeros(w.shape[0] + 1, dtype=index_dtype)
    if rows.numel():
        cnt = torch.bincount(rows, minlength=w.shape[0]).to(index_dtype)
        crow[1:] = cnt.cumsum(0)
    values = w[rows, cols].contiguous()
    return crow, cols.to(index_dtype).contiguous(), values


def dense_to_csr(w: torch.Tensor, index_dtype: torch.dtype | None = None):
    """Dense [out, in] -> CSR (out, in) components (indptr, indices, values, size)."""
    if index_dtype is None:
        index_dtype = CSR_INDEX_DTYPE
    indptr, indices, values = _csr_components(w, index_dtype)
    return indptr, indices, values, list(w.shape)


def _probe_csr_index_dtype() -> torch.dtype:
    """Probe whether torch supports int32 CSR indices for sparse.mm on CPU.

    B1b: on this torch build the CSR tensor object itself is NOT deepcopy
    -able (the CSR impl forbids storage access; ``copy.deepcopy`` on a CSR
    tensor raises). Since the eval protocol (tests + GCN cache clearing)
    deepcopies the compact model, we additionally require int32 support to
    fall back cleanly: prefer int32 (smaller artifact, supported); the
    default int64 layout size is recorded separately as csr_int64_bytes."""
    try:
        w = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        w[0, 3] = 0.0
        crow, col, vals = _csr_components(w, torch.int32)
        csr = torch.sparse_csr_tensor(crow, col, vals, size=list(w.shape), dtype=w.dtype)
        x = torch.randn(4, 2)
        y = torch.sparse.mm(csr, x)
        assert torch.allclose(y, w @ x, atol=1e-5)
        return torch.int32
    except Exception:
        return torch.int64


# CSR_INDEX_DTYPE is the index dtype used for the compact runtime modules and
# recorded as `csr_index_dtype` on the artifact.
CSR_INDEX_DTYPE = _probe_csr_index_dtype()


def csr_tensor(crow, col, values, size) -> torch.Tensor:
    return torch.sparse_csr_tensor(crow, col, values, size=size, dtype=values.dtype)


def csr_from_components(d: dict) -> torch.Tensor:
    return csr_tensor(d["indptr"], d["indices"], d["values"], size=d["size"])


def csr_int64_bytes(crow, col, values, size) -> int:
    """Size of the equivalent DEFAULT int64 CSR layout (values + int64
    indptr/indices), payload only."""
    return (values.numel() * values.element_size()
            + crow.numel() * 8 + col.numel() * 8)


def s4_bitpacked_bytes_theoretical(pack: dict) -> int:
    """THEORETICAL (labelled) bit-packed size of one S4 weight tensor: kept
    values as FP32 (4 B) + ONE bit-packed in-block position per kept value
    (ceil(log2 M) bits, packed, ceiling to whole bytes) — row/block implied
    by the fixed pattern + partial-block ceil rule. Biases are accounted for
    separately by the caller via pack['bias_bytes'] (sum over the model)."""
    m = pack["m"]
    nnz = pack["values"].numel()
    value_bytes = nnz * 4
    if nnz == 0:
        pos_bytes = 0
    else:
        bits = int(math.ceil(math.log2(m))) if m > 1 else 1
        pos_bytes = int(math.ceil(nnz * bits / 8))
    return value_bytes + pos_bytes


def save_csr(w: torch.Tensor, path: str) -> int:
    """Save the CSR of dense [out, in] weight `w` to disk; returns file size."""
    indptr, indices, values, size = dense_to_csr(w)
    torch.save({"values": values, "indptr": indptr, "indices": indices,
                "size": size, "csr_index_dtype": str(CSR_INDEX_DTYPE),
                "csr_int64_bytes": csr_int64_bytes(indptr, indices, values, size)},
               path)
    return os.path.getsize(path)


# ---------------------------------------------------------------------------
# Compact runtime modules
# ---------------------------------------------------------------------------

class CSRLinear(nn.Module):
    """y = (W_csr @ x^T)^T + b, W_csr shape (out, in); sparse x dense.

    B1b: the CSR tensor is rebuilt from its dense weight on the first forward
    (and after a train-mode forward) rather than kept as a buffer: the CSR
    tensor object is not deepcopy-able in this torch build (the CSR impl
    forbids storage access), and the eval protocol deepcopies the model.
    The weight is therefore held as a dense buffer (the same zeros as the
    dense_zeros path) and converted on demand; the conversion is cached per
    (data_ptr, nnz) so repeated forwards are cheap."""

    def __init__(self, w_dense: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("w_dense", w_dense.detach().clone())
        self.register_buffer("bias", bias)
        self.out_features = w_dense.size(0)
        self.in_features = w_dense.size(1)
        self._w_csr: torch.Tensor | None = None
        self._w_key: tuple | None = None

    def _csr(self) -> torch.Tensor:
        key = (self.w_dense.data_ptr(), int(self.w_dense.count_nonzero()))
        if self._w_csr is None or self._w_key != key:
            crow, col, vals = _csr_components(self.w_dense, CSR_INDEX_DTYPE)
            self._w_csr = torch.sparse_csr_tensor(
                crow, col, vals, size=list(self.w_dense.shape),
                dtype=self.w_dense.dtype)
            self._w_key = key
        return self._w_csr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sparse.mm(self._csr(), x.t()).t()
        if self.bias is not None:
            y = y + self.bias
        return y


class CSRConv2d(nn.Module):
    """CSR matmul conv, no Python loop over the batch: unfold once, reshape
    the columns to (in*k*k, B*L) and run ONE sparse mm (sparse x dense).
    B1b: holds the dense (zeroed) weight and rebuilds the CSR on demand (the
    CSR tensor object is not deepcopy-able in this torch build)."""

    def __init__(self, w_dense_flat: torch.Tensor, bias: torch.Tensor,
                 kernel_size: int, padding: int, out_spatial: tuple[int, int] | None = None):
        super().__init__()
        self.register_buffer("w_dense", w_dense_flat.detach().clone())
        self.register_buffer("bias", bias)
        self.k = kernel_size
        self.pad = padding
        self.out_spatial = out_spatial
        self._w_csr: torch.Tensor | None = None
        self._w_key: tuple | None = None

    def _csr(self) -> torch.Tensor:
        key = (self.w_dense.data_ptr(), int(self.w_dense.count_nonzero()))
        if self._w_csr is None or self._w_key != key:
            crow, col, vals = _csr_components(self.w_dense, CSR_INDEX_DTYPE)
            self._w_csr = torch.sparse_csr_tensor(
                crow, col, vals, size=list(self.w_dense.shape),
                dtype=self.w_dense.dtype)
            self._w_key = key
        return self._w_csr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, cin, h, wd = x.shape
        kk = self.k
        cols = F.unfold(x, (kk, kk), padding=self.pad)      # (B, in*k*k, L)
        cols = cols.permute(1, 0, 2).reshape(self.w_dense.size(1), -1)
        y = torch.sparse.mm(self._csr(), cols)              # (out, B*L)
        if self.out_spatial is not None:
            oh, ow = self.out_spatial
        else:
            oh = ow = int(math.isqrt(y.size(1) // b))
        # F.unfold pixel order: p = (y * ow + x) inside each image
        y = y.reshape(self.w_dense.size(0), b, oh, ow).permute(1, 0, 2, 3)
        return y + self.bias.view(1, -1, 1, 1)


class CSRGCNConv(GCNConv):
    """GCNConv in which ONLY the projection differs from the dense path.

    The dense GCNConv computes ``x = self.lin(x)`` (x @ W^T + lin bias-free)
    and then runs ``propagate`` (gcn_norm normalization + message +
    aggregation). Here ``self.lin`` is replaced by a CSR projection that
    computes ``x @ W^T = (W_csr @ x^T)^T`` via the optimized sparse x dense
    ``torch.sparse.mm`` path. ``propagate`` (normalization, message and
    aggregation) is the untouched PyG GCNConv code, so the latency contrast
    isolates the sparse projection."""

    def __init__(self, w_dense: torch.Tensor, bias: torch.Tensor,
                 in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, add_self_loops=True,
                         normalize=True)
        with torch.no_grad():
            self.bias.data = bias.detach().clone()
        # Replace the dense projection ``lin`` (bias=False in GCNConv) with the
        # CSR projection. Only the projection differs; message/aggregation are
        # the PyG identity/scatter code.
        self.lin = CSRLinear(w_dense, self.lin.bias)
        # B1b: GCNConv.__init__ registers a `cached` flag; PyG's message
        # passing reads it via attribute access (not in __dict__), so it is
        # harmless; nothing to drop.
        self.w_dense_ref = None  # not used; kept for clarity

    def forward(self, x, edge_index, edge_weight=None):
        # Inherited PyG GCNConv.forward: x = self.lin(x) [CSR projection],
        # gcn_norm normalization + propagate + bias. No override needed.
        return GCNConv.forward(self, x, edge_index, edge_weight)

    def forward(self, x, edge_index, edge_weight=None):
        # Inherited PyG GCNConv.forward: x = self.lin(x) [CSR projection],
        # gcn_norm normalization + propagate + bias. No override needed.
        return GCNConv.forward(self, x, edge_index, edge_weight)


def clear_gcn_caches(model: nn.Module) -> None:
    """Clear PyG message-passing ``_cached*`` attributes (GCN caches the
    normalized edge index on the module during train-mode forwards)."""
    for mod in model.modules():
        for attr in list(vars(mod)):
            if attr.startswith("_cached"):
                setattr(mod, attr, None)


def reassemble_packed(pack: dict) -> torch.Tensor:
    """Invert s4_packed (bit-exact). The packed file stores the kept values
    in row-major order plus ONE uint8 in-block position per kept value; the
    row and block of each value are implied by the fixed N:M pattern and the
    frozen partial-block ceil rule (per-row keep counts)."""
    n, m, L, shape = pack["n"], pack["m"], pack["row_len"], tuple(pack["shape"])
    R = shape[0]
    r = L % m
    n_full = L // m
    n_blocks = n_full + (1 if r else 0)
    keep_per_block = [n] * n_full
    if r:
        keep_per_block.append(math.ceil(n * r / m))
    per_row = sum(keep_per_block)

    # per-block start offsets, in row-major block order
    starts = []
    pos = 0
    for kb in keep_per_block:
        starts.append(pos)
        pos += kb
    block_starts = torch.tensor(starts, dtype=torch.long)

    k = torch.arange(pack["n_kept"]) if "n_kept" in pack else torch.arange(pack["values"].numel())
    row = k // per_row
    block_in_row = k % per_row
    # B1b: kept values are emitted in BLOCK-MAJOR order (per row: block 0's
    # kept values in increasing in-block column, then block 1's, ...). The
    # in-block column is the ACTUAL column within the block (stored as one
    # uint8 per kept value), so column = block_flat_start + in_block_column.
    # (Using the rank within the block would be wrong because the kept set is
    # the N largest-magnitude positions, which are NOT the first N columns.)
    # B1b fix: use side="right" so that bir == block_starts[b] maps to block b
    # (not b+1). The default side="left" gives the first index where
    # block_starts[i] >= bir, which for bir == block_starts[b] returns b,
    # but for bir == block_starts[b] + 1 (the second value of block b) it
    # returns b+1, which is wrong. side="right" gives the first index where
    # block_starts[i] > bir, which for bir in [block_starts[b], block_starts[b+1])
    # returns b+1, so we subtract 1 to get b.
    block = torch.searchsorted(block_starts, block_in_row, side="right") - 1
    block = torch.clamp(block, max=len(keep_per_block) - 1, min=0)
    in_block = pack["in_block"].to(torch.long)
    if r:
        block_pos = torch.where(block < n_full,
                                block * m,
                                n_full * m + (block - n_full) * r)
    else:
        block_pos = block * m
    flat_idx = block_pos + in_block
    w = torch.zeros(shape, dtype=pack["values"].dtype)
    flat = w.view(w.shape[0], -1)
    flat[row, flat_idx] = pack["values"]
    return w


class DenseS4Module(nn.Module):
    """S4 dense fallback: weight reassembled from the packed N:M file
    (kept values + per-value in-block position) into zeros."""

    def __init__(self, dense_w: torch.Tensor, bias: torch.Tensor | None, kind: str):
        super().__init__()
        self.register_buffer("w", dense_w.detach().clone())
        self.register_buffer("bias", bias)
        self.kind = kind

    def forward(self, x, edge_index=None):
        raise NotImplementedError


class CompactModel(nn.Module):
    """Wraps a dense parent's state into the compact runtime form.

    `kind`: 'csr' (S1/S3) or 's4' (S4 dense fallback)."""

    def __init__(self, architecture: str, state: dict, kind: str):
        super().__init__()
        from .models import build_model
        self.arch = architecture
        self.kind = kind
        parent = build_model(architecture)
        parent.load_state_dict(state)
        self._build(parent)

    def _build(self, parent: nn.Module) -> None:
        from torch_geometric.nn import GCNConv

        cls = parent.__class__.__name__

        def lin(w_dense, bias):
            # stored [out, in]; the CSR projection computes (W_csr @ x^T)^T.
            # B1b: pass the dense (zeroed) weight; the CSR tensor is built on
            # demand (CSR tensors are not deepcopy-able in this torch build).
            return CSRLinear(w_dense.detach().clone(), bias)

        if cls == "MLP":
            self.m = nn.Sequential(
                lin(parent.net[0].weight, parent.net[0].bias),
                nn.ReLU(),
                lin(parent.net[2].weight, parent.net[2].bias),
                nn.ReLU(),
                lin(parent.net[4].weight, parent.net[4].bias),
            )
        elif cls == "CNN":
            # conv weight (out, in, k, k) -> CSR (out, in*k*k)
            c1, c2 = parent.conv1, parent.conv2
            self.m = nn.Sequential(
                CSRConv2d(c1.weight.detach().clone().flatten(1),
                          c1.bias, 5, 2, out_spatial=(28, 28)),
                nn.ReLU(),
                nn.MaxPool2d(2),
                CSRConv2d(c2.weight.detach().clone().flatten(1),
                          c2.bias, 5, 0, out_spatial=(10, 10)),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.flatten = nn.Flatten(1)
            self.m2 = nn.Sequential(
                lin(parent.fc1.weight, parent.fc1.bias),
                nn.ReLU(),
                lin(parent.fc2.weight, parent.fc2.bias),
            )
        elif cls == "GCN":
            # GCNConv.lin.weight is stored [out, in]; the CSR projection uses
            # (W_csr @ x^T)^T so pass the stored [out, in] matrix directly.
            c1, c2 = parent.conv1, parent.conv2
            self.m = nn.Sequential(
                CSRGCNConv(c1.lin.weight.detach().clone(),
                           c1.bias, c1.in_channels, c1.out_channels),
                nn.ReLU(),
                nn.Dropout(0.5),
                CSRGCNConv(c2.lin.weight.detach().clone(),
                           c2.bias, c2.in_channels, c2.out_channels),
            )
        else:
            raise ValueError(f"CompactModel: unsupported {cls}")

    def forward(self, x, edge_index=None):
        if self.arch == "gcn":
            return self._gcn(x, edge_index)
        if self.arch == "mlp":
            return self.m(x.flatten(1))
        x = self.m(x)
        return self.m2(self.flatten(x))

    def _gcn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # PyG message passing caches the normalized edge index on the module
        # (base GCNConv) during train-mode forwards; evaluation must use a
        # clean cache, exactly as the dense path's _fresh/clear protocol does.
        clear_gcn_caches(self)
        x = self.m[0](x, edge_index)
        x = self.m[1](x)
        x = self.m[2](x)
        x = self.m[3](x, edge_index)
        return x
