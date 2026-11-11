# C2 — Part C rerun (Python 3.12, all 8 issues fixed)

C2 supersedes the entire C1b Part C study. C1b ran on **Python 3.13.14** (env
mismatch) and had the INT4/bytes/nonzero/factorial bugs the reviewer flagged.
C2 reran the whole Part C study on the pinned **Python 3.12.13** environment
(torch 2.15.0, PyG 2.8.1, numpy 2.5.4) after fixing all 8 issues.

**Environment confirmed:** Python 3.12.13, torch 2.15.0, PyG 2.8.1, numpy 2.5.4
(verified at run time; every C2 row carries `versions.python=3.12.13`).
All work was done with `uv run` only.

## The 8 issues, fixed

### 1. Environment (Python 3.13 → 3.12)
C1b ran on Python 3.13.14. C2 reran everything on 3.12.13. All 45 C1b
`part="C"` rows (2026-11-09T23:09–23:30) are marked **superseded** in
`results/ANNOTATIONS.md` (new section "C1b — superseded by C2"). `runs.jsonl`
was **never edited**.

### 2. INT4 scored chance (val 0.0824/0.0942/0.1260)
**Root cause:** `int4_quantize` built the INT4 packed model from
`build_model(arch)`'s **random-init** weights instead of the parent's trained
weights. The bug: `state_dict()[k] = v.clone()` on a freshly-constructed module
does **not** update its parameters (the returned dict is a copy of Parameter
references, not a live view), so the packed model silently used random weights
and wrong scales (e.g. 0.0051 instead of 0.0436), scoring chance.
**Fix:** `int4_quantize` now uses `m.load_state_dict({k: v.clone() for ...})`
to load the parent weights, then wraps eligible ops with the INT4 packed
modules.
**New test:** `test_int4_matches_reference_from_disk` (all 3 archs) — the INT4
packed model **loaded from disk** has logits equal to the reference
per-output-channel symmetric [-8,7] quantize-dequantize within 1e-5, and its
val accuracy matches the reference and is far from chance.
**C2 results (match the reviewer's reference exactly):** MLP 0.8914,
CNN 0.9006, GCN 0.7860.

### 3. CNN static INT8 artifact was 1,805 bytes
**Root cause:** the saved artifact was an fp32 placeholder
(`reconvert=True`); reload rebuilt the FX-quantized structure from scratch, so
the file never contained the quantized weights (1,805 bytes, impossible for
~117k int8 weights).
**Fix:** `_artifact_state` now returns `{"quantized_state": model.state_dict()}`
for CNN static-INT8 (the converted model's quantized state);
`_rebuild_from_artifact` rebuilds the FX-converted structure via
`_convert_static(parent, arch)` and `load_state_dict(..., strict=True)` from
file. `artifact_bytes` is the **actual file size**. Reload is now verified for
**every** format (max_abs_diff = 0.0 on all 15 C2 precision rows).
**New test:** `test_cnn_static_int8_artifact_has_quantized_weights` — the CNN
static-INT8 artifact is a real file of the recorded size (>50 KB, not 1.8 KB),
and reloading it from disk reproduces the quantized weights (logits maxdiff
0.0, nonzero count preserved).
**C2 result:** CNN static-INT8 artifact = **127,291 bytes** (was 1,805).

### 4. Byte breakdown next to artifact_bytes
**Fix:** `bytes_by_dtype(model, kind)` records `numel * element_size` per stored
dtype, written next to `artifact_bytes` on every row (`bytes_by_dtype` and
`payload_bytes = sum(...)`). Container overhead = `artifact_bytes -
payload_bytes` is now visible. For INT4 the packing layout is explicit
(`int4_packed_uint8` = ceil(n/2) bytes for 2 values/byte, + per-output-channel
fp32 scales, + int64 metadata). New test `test_bytes_by_dtype_present` (5
formats × 3 archs) checks it is non-empty and that `artifact_bytes >=
payload_bytes`.

### 5. nonzero_params = 0 for all INT8 rows
**Root cause:** the old code counted `p.count_nonzero()` over `model.parameters()`,
which for qnnpack-quantized models are fp32 placeholder params (zero).
**Fix:** `int8_nonzero(model)` counts nonzero entries of the **integer weight
representation** — `qw.int_repr()` for qnnpack quantized linear/conv weights,
and the packed bytes for INT4 — unpacking packed params. `rec["nonzero_params"]`
now uses this; `rec["nonzero_params_dense_parent"]` is recorded for reference.
For the factorial, `nonzero_before_int8` / `nonzero_after_int8` report the
sparse source nonzero count and the surviving integer-representation nonzero
count after INT8. New test `test_int8_nonzero_not_zero` (3 archs) checks the
integer-representation nonzero count is a substantial majority of the dense
parent's nonzero weights (not the broken 0).

### 6. Factorial: exactly 12 conditions, no failures/duplicates
C1b had a failed `mlp sparse_int8` row plus duplicate rows. C2 produces exactly
**12 conditions** (3 archs × 4: dense_fp32, sparse_fp32, dense_int8,
sparse_int8), all `status=ok`. The INT8 method applied is the one selected by
the frozen C.1 rule (MLP/GCN → int8_static, CNN → int8_dynamic). A failed
condition would be written as a classified `status=failed` row (none occurred).

### 7. Package hygiene
- Driver moved from standalone `code/lab3_precision.py` (deleted) into the
  package as `code/lab3/c2.py`, wired into the package CLI:
  `uv run python -m lab3 precision` and `uv run python -m lab3 factorial`
  (also `c2summary` for the report).
  - Note: the repo path contains `#` ("...Q4/#5. Deep Learning Systems..."),
    which the `.pth`/site-packages loader treats as a comment, so a plain
    site-packages path shim cannot be installed into the venv. The reliable
    invocation is `PYTHONPATH=code uv run python -m lab3 precision` /
    `factorial` / `c2summary` (the driver is in the package; `PYTHONPATH=code`
    is the standard mechanism the pytest config already uses via
    `pythonpath=["code"]`).
- `results/_quarantine/` (an empty `c1_debug` dir) was removed; the real
  quarantined artifacts already live in top-level `_quarantine/c1_debug/`.

### 8. Figures
Two new figures added (legends render; all panels asserted non-empty by the
`_shared_legend` non-empty-text assertion and verified with a non-white-pixel
check):
- `figures/precision_size_latency.png` — per arch: artifact bytes (log) vs
  batch-256 / full-graph median latency (log), one point per format, C2 (3.12)
  rows.
- `figures/factorial.png` — 3×3 grid (rows = val_acc / artifact bytes / median
  latency; columns = MLP/CNN/GCN), one bar per condition.
Kept the existing `partC_precision_valacc.png`, `partC_precision_latency.png`,
`partC_precision_frac_macs.png`.

## C2 run results (summary)

See `results/partc_summary.md` for the full tables with `artifact_bytes`,
`bytes_by_dtype` (+ `payload_bytes`), `nonzero_params` before/after, latency
(batch 1 + batch 256/full graph, median + IQR), RSS, frac_macs,
stored/compute/accumulator precision, reload_equivalence, and run_ids.

Key C2 numbers (val accuracy):

| arch | fp16 | bf16 | int8 dyn | int8 static | int4 sim |
|---|---|---|---|---|---|
| mlp | 0.8982 | 0.8988 | 0.8984 | 0.8984 | **0.8914** |
| cnn | 0.9182 | 0.9180 | 0.9182 | 0.9154 | **0.9006** |
| gcn | 0.8000 | 0.8000 | 0.7960 | 0.7940 | **0.7860** |

INT4 now matches the reviewer's reference exactly (was chance in C1b).

INT8 selection (frozen C.1 rule, C2 run, `results/int8_selection.json`):
MLP → int8_static (val tie 4492/4492, faster 0.40 ms), CNN → int8_dynamic
(higher val 4591/4577), GCN → int8_static (val tie 398/397 within 1 example,
faster 1.19 ms).

12-condition factorial (all `status=ok`, C2): see `results/factorial.csv` and
the summary. Notable: GCN static-INT8 quantization collapses the projection
weights to 23 nonzero int8 entries (per-tensor symmetric quant of small GCN
weights) — now visible via `nonzero_after_int8` / `bytes_by_dtype`.

## Tests

`uv run pytest -q` → **141 passed** (was 119; +22 new tests: INT4-from-disk
reference ×3, CNN static artifact regression, int8 nonzero ×3, bytes_by_dtype
×15, etc.). No test was weakened. No selection used the test set (INT8
selection uses val accuracy + latency only; val/test are frozen 80/20 splits).
