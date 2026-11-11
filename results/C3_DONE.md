# Task C3 — DONE

**Date:** 2026-11-10T01:30–02:00 UTC (local 16:30–17:00)
**Python:** 3.12.13 · **torch:** 2.15.0 · **PyG:** 2.8.1 · **platform:** darwin
**Scope:** Fix the two C2 Part C (reduced-precision) accounting bugs flagged by
the reviewer, add the required INT8-accounting test, do a targeted re-run of the
affected INT8 records, and regenerate all Part C outputs.

---

## 1. INT8 accounting blind to quantized projections (issue 1)

### Symptom (from the C2 rows)

- gcn `dense_int8` / `sparse_int8`: `nonzero_after_int8 = 23` (the 16+7 GCN
  biases, NOT the 22,928+112 projection weights) and
  `bytes_by_dtype = {"fp32": 92}` (biases only) — despite val acc
  0.794–0.796 proving the INT8 weights were present and executing.
- gcn standalone `int8_static`: `nonzero_params = 23`,
  `bytes_by_dtype = {"fp32": 92}`.
- gcn standalone `int8_dynamic`: `nonzero_params = 18,972` (correct — the
  wrapper-module path worked for dynamic quantization), but the FX-static path
  was broken.
- CNN static-INT8: `nonzero_params = 100,646` (undercount; should be 113,474 —
  the conv weights were missed).

### Root cause

`convert_fx` (qnnpack FX static quantization) lifts the quantized-op packed
weights to raw instance attributes on the containing module. Two distinct
shapes:

1. **Dynamic / FX-static linears and CNN-static convs** — a
   `LinearPackedParams` Python wrapper module (child in `named_modules()`) whose
   `vars()` hold a raw `torch.ScriptObject`. `_weight_bias()` works through the
   wrapper. C2's traversal matched this path.

2. **GCN static per-projection GraphModules** — each `convN.lin` is an FX
   GraphModule with a raw `_packed_weight_0` `torch.ScriptObject` (class
   `torch.classes.quantized.LinearPackedParamsBase`) as an instance attribute.
   This ScriptObject exposes `unpack()` (returns `(qint8 weight, [bias])`) but
   NOT `_weight_bias()`. C2's traversal only matched module types named
   Linear/LinearReLU/Conv2d/ConvReLU2d with a working `_weight_bias`, so the
   GCN static rows were invisible.

Additionally, the `LinearPackedParams` wrapper module is BOTH a registered
submodule AND holds the same ScriptObject in its own `vars()`, causing
double-counting if both paths were hit.

### Fix (in `code/lab3/precision.py`)

- Added `_qnnpack_weight_bias(mod)`:
  - If `mod._packed_params` is a Python (non-ScriptObject) wrapper, call
    `_weight_bias()` and return immediately (no fallback to avoid double count).
  - Otherwise, scan `vars(mod)` for raw `torch.ScriptObject`s and `unpack()`
    them (handles the FX-static GraphModule `_packed_weight_N` shape).
- Added `_is_packed_params_wrapper(mod)`: True when `vars(mod)._packed_params`
  is a ScriptObject AND the module type is `LinearPackedParams` — used to skip
  wrapper modules in the traversal (prevents double-counting).
- Rewrote `_int_weight_tensors`, `int8_nonzero`, and `bytes_by_dtype` to use
  these helpers, with dedup via `handled_int_module_ids` (module ids) and a
  buffer-aware fallback (the C2 fallback only iterated `named_parameters`,
  double-counting the INT4 `packed` uint8 buffer as 'uint8').
- Marked INT4 modules as handled so their buffers aren't re-counted.

### Verification

All 12 INT8 conversions (3 archs × 2 methods, dense parent) now correctly
counted:

| arch | method | int8 bytes | quantized-op numel | nnz (int8) | parent nnz |
|---|---|---|---|---|---|
| mlp | dynamic | 234,752 | 234,752 | 227,255 | 234,752 |
| mlp | static | 234,752 | 234,752 | 227,255 | 234,752 |
| cnn | dynamic | 103,680 | 103,680 | 100,646 | 116,880* |
| cnn | static | 116,880 | 116,880 | 113,474 | 116,880 |
| gcn | dynamic | 23,040 | 23,040 | 18,972 | 23,040 |
| gcn | static | 23,040 | 23,040 | 18,972 | 23,040 |

\*CNN dynamic-INT8 quantizes only the fc layers (conv stays fp32 under the
frozen C.1 `quantize_dynamic({nn.Linear})` spec), so 103,680 = 102,400 + 1,280
is correct; the parent nnz of 116,880 includes the conv weights that remain fp32.

The int8 `nnz` is within 20% of the fp32 nonzero count of the quantized layers
for all 6 combinations (quantization zeros out only the small-magnitude weights
that fall within half a quant step of zero).

### New test (in `tests/test_precision.py`)

`test_int8_bytes_count_quantized_weights` — parametrized over all 3 archs ×
both INT8 methods (6 cases). For each, asserts:
- `bytes_by_dtype["int8"] >= numel(weights of the quantized layers)`
- `0.8 * fp32_nnz(quantized layers) <= int8_nonzero <= 1.2 * fp32_nnz(quantized layers)`

The test dynamically determines which eligible ops were actually quantized
(int8) by calling `_qnnpack_weight_bias` on the converted model's modules,
so it correctly handles the CNN dynamic case (fc-only) vs CNN static
(fc+conv).

All 71 precision tests pass (65 pre-existing + 6 new). Full suite: 147 passed.

---

## 2. Unexplained container overhead (issue 2)

### Diagnosis

The outer `.pt` file is a `torch.save` zip wrapping a dict
`{"kind", "architecture", "state"}`. For INT8 (mlp and gcn), `state` is
`{"pickled": <bytes>}` where those bytes are themselves a `torch.save` zip of
the live model. For CNN static-INT8, `state` is a flat state dict (no nested
zip).

The dense-vs-sparse file-size gap (e.g. mlp dense_int8 = 359,751 B vs
sparse_int8 = 260,237 B, both with payload 236,328 B) is **NOT** dead weight.
It is the **nested-zip re-encoding**:

- The outer `data.pkl` embeds the entire inner zip as one stored
  (uncompressed) blob.
- The inner zip for the dense model carries the larger fp32 bias tensors
  (dense biases are all nonzero → larger fp32 payload in the inner zip).
- The inner zip for the sparse model carries the smaller sparse-weight +
  int8-payload tensors.
- The int8 **weight** payloads are byte-for-byte the same size in both (1 B
  per weight, regardless of zero pattern), so the file-size difference tracks
  the fp32 bias/scale/zero-point bytes, not the int8 weights.

Per-entry breakdowns for all 12 factorial artifacts are in
`results/artifact_zip_listing.md` (written by
`_write_container_overhead_md` in `c2.py`). The new `container_overhead` field
on factorial rows (added in `factorial_row`) records the per-entry
`zipfile.infolist()` for each saved artifact, including the inner zip entries.

**No data was removed from the artifacts** — the nested zip is needed to run
the quantized model (it carries the qparams, packed weights, and module
structure). The explanation is in `results/partc_summary.md` (section
"Container overhead").

---

## 3. Rerun of affected INT8 records

**What was rerun** (12 new rows in `results/runs.jsonl`, all
`status=ok`, python 3.12.13):

| arch | phase | method | C3 run_id | val_acc | nonzero_after_int8 |
|---|---|---|---|---|---|
| mlp | precision | int8_dynamic | mlp-…013014…e63a8c | 0.8984 | 227,255 |
| mlp | precision | int8_static | mlp-…013016…e95341 | 0.8984 | 227,255 |
| cnn | precision | int8_dynamic | cnn-…013026…2a7da2 | 0.9182 | 100,646 |
| cnn | precision | int8_static | cnn-…013030…07de9a | 0.9154 | 113,474 |
| gcn | precision | int8_dynamic | gcn-…013045…832da3 | 0.7960 | 18,972 |
| gcn | precision | int8_static | gcn-…013046…96f676 | 0.7940 | 18,972 |
| mlp | factorial | dense_int8 | mlp-…013022…2eb647 | 0.8984 | 227,255 |
| mlp | factorial | sparse_int8 | mlp-…013024…1919d5 | 0.8936 | 23,469 |
| cnn | factorial | dense_int8 | cnn-…013036…da4fa4 | 0.9182 | 100,646 |
| cnn | factorial | sparse_int8 | cnn-…013040…68c2dd | 0.9192 | 23,832 |
| gcn | factorial | dense_int8 | gcn-…013048…041ecf | 0.7940 | 18,972 |
| gcn | factorial | sparse_int8 | gcn-…013050…15aa3d | 0.7960 | 2,304 |

Each C3 row carries `supersedes` pointing to its C2 predecessor.

**Superseded C2 INT8 rows** (12): listed in `results/ANNOTATIONS.md` under
"## C3 — superseded C2 INT8 rows".

**What was NOT rerun:** the fp16, bf16, int4_simulated, dense_fp32, and
sparse_fp32 rows (30 non-int8 C rows) were not affected by either bug and are
kept as-is.

### int8_selection.json

Recomputed from the C3 precision rows (val_acc and latency inputs unchanged
from C2 → same selection):

| arch | selected | val_acc | latency_ms | reason |
|---|---|---|---|---|
| mlp | int8_static | 0.8984 | 0.4047 | val tie; latency 0.4047 < 0.5182 |
| cnn | int8_dynamic | 0.9182 | 7.2603 | val 0.9182 > 0.9154 |
| gcn | int8_static | 0.7940 | 1.1996 | val 397 vs 398 (1-ex tie); latency 1.1996 < 1.5147 |

The `run_id` fields in `int8_selection.json` now point to the C3 rows.

---

## 4. Regenerated outputs

| file | status |
|---|---|
| `results/runs.jsonl` | 12 C3 INT8 rows appended (with `supersedes`) |
| `results/factorial.csv` | 12 rows (latest ok per arch×condition, C3 int8) |
| `results/partc_summary.md` | regenerated (C3 title, corrected int8 rows, container-overhead section, UF_HIDDEN correction) |
| `results/artifact_zip_listing.md` | per-entry zipfile.infolist for all 12 factorial artifacts |
| `results/int8_selection.json` | recomputed (same selection, new run_ids) |
| `results/ANNOTATIONS.md` | C3 section appended (superseded C2 INT8 rows + explanation) |
| `results/SCHEMA.md` | C3 changelog appended (both fixes + UF_HIDDEN correction) |
| `figures/precision_valacc.png` | regenerated |
| `figures/precision_macs.png` | regenerated |
| `figures/precision_size_latency.png` | regenerated |
| `figures/factorial.png` | regenerated |

---

## 5. C2 explanation correction (UF_HIDDEN)

C2_DONE.md (immutable) attributes the `pip install -e .` failure to a
`#`-comment issue in the `.pth` file. **The real cause:** `uv venv --seed`
writes the site-packages `.pth` files with the macOS `UF_HIDDEN`
(0x80000000) file flag, and CPython's `site.py` (3.12) skips hidden `.pth`
files, so `lab3.pth` was ignored and `import lab3` failed.

This correction is recorded in:
- `results/partc_summary.md` (preamble paragraph)
- `results/SCHEMA.md` (C3 changelog section)
- `results/ANNOTATIONS.md` (C3 section, last paragraph)

C2_DONE.md is unchanged (immutable per the task spec).

---

## 6. Test status

```
$ PYTHONPATH=code uv run pytest -q tests/
147 passed, 87 warnings in 140.46s (0:02:20)
```

All 147 tests pass (65 pre-existing precision + 6 new INT8-accounting + 76
other).

---

## 7. Files changed

| file | change |
|---|---|
| `code/lab3/precision.py` | Added `_qnnpack_weight_bias`, `_is_packed_params_wrapper`; rewrote `_int_weight_tensors`, `int8_nonzero`, `bytes_by_dtype`; added `container_overhead` field + `_container_overhead` helper; added `zipfile` import |
| `code/lab3/c2.py` | Added `run_int8_rerun`; added `_write_container_overhead_md`; updated `write_summary` (C3 title, deduped factorial table, container-overhead section, UF_HIDDEN correction) |
| `tests/test_precision.py` | Added `test_int8_bytes_count_quantized_weights` (6 cases) |
| `results/partc_summary.md` | Regenerated |
| `results/factorial.csv` | Regenerated |
| `results/artifact_zip_listing.md` | New (per-entry zip listing) |
| `results/int8_selection.json` | Recomputed (new run_ids) |
| `results/ANNOTATIONS.md` | C3 section appended |
| `results/SCHEMA.md` | C3 changelog appended |
| `results/runs.jsonl` | 12 C3 rows appended; `supersedes` added to C3 rows |
