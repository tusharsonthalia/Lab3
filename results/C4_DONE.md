# Task C4 — DONE

**Date:** 2026-11-09T20:20–20:30 UTC (local 20:20–20:30 CST; runs.jsonl timestamps
2026-11-10T02:24 UTC)
**Python:** 3.12.13 · **torch:** 2.15.0 · **PyG:** 2.8.1 · **platform:** darwin
**Scope:** Fix the INT8 artifact FORMAT (C3 issue 2, corrected): the C3 mlp/gcn
INT8 artifacts used a nested-pickle container (`state = {"pickled": <bytes of an
inner torch.save of the whole live module>}`). C4 makes every INT8 artifact a
single `torch.save` of the converted model's `state_dict()`, re-runs the 12
affected rows as new records, regenerates the Part C outputs, and adds the
required artifact-size test.

---

## 1. Root cause (correcting the C3 explanation)

C3's explanation ("the dense-vs-sparse gap tracks the larger fp32 bias tensors
in the nested inner zip") was **wrong** — the MLP's biases total ~2 KB, not
~99 KB. The real problem was the artifact FORMAT: for INT8 (mlp and gcn, both
dynamic and static) the saved file was a dict whose `state` was
`{"pickled": <bytes of an inner torch.save of the whole live module>}` — a
**nested container**: the outer .pt zip embedded the entire inner zip (a
torch.save of the live module, re-encoding the whole module tree: quantized
weights plus the module's own structure/state, plus a full second zip of
headers) as one stored (uncompressed) blob. The C3 static-INT8 CNN row already
used a flat `state_dict` (via the `quantized_state` wrapper), which is why
C3's per-arch bytes were inconsistent.

Measured on this machine (as reported by the reviewer, confirmed): a plain
`torch.save(q.state_dict())` of the converted dense MLP is **241,351 bytes**,
and the **same 241,351 bytes** for the S1-90% sparse MLP (zeros cost 1 byte in
int8, as expected) — versus the C3 nested-pickle artifacts' 359,751 (dense) /
260,237 (sparse).

## 2. Fix (code)

### `code/lab3/precision.py`

- `_artifact_state` (INT8, dynamic AND static, all 3 archs): returns the
  converted model's `state_dict()` itself — int8 weights + fp32/int64 scales
  and zero points only (plus, for FX-static GCN, the raw `_packed_weight_N`
  `torch.ScriptObject`s as state-dict values). No nested pickled bytes, no
  fp32 copy of the weights. The dict is a module-level `_SDWithMeta(dict)`
  subclass that carries the state_dict `_metadata` (per-module `version`
  entries) — required by the qnnpack dynamic/fused-Linear loaders (without it
  they fall back to the v1 deserialization path and raise
  `KeyError: 'net.0.weight'`). A plain dict comprehension drops `_metadata`,
  which is why the first implementation failed only in the reload path.
- `_rebuild_from_artifact` (INT8): rebuilds the same quantized structure from
  the fp32 seed-0 parent via the SAME conversion path that produced the
  artifact (`quantize_dynamic({nn.Linear})` with the GCN per-projection
  container path, or FX `prepare_fx`+calibration+`convert_fx` — the
  calibration-free dummy parameters are re-derived deterministically), then
  `load_state_dict(torch.load(file), strict=True)`. For FX-static GCN the
  `_packed_weight_N` ScriptObject keys are popped from the state dict and
  re-assigned directly on the rebuilt GraphModules (FX GraphModules keep the
  packed weight as a raw instance attribute; `load_state_dict` cannot set it
  and would report it as an unexpected key).
- `_save_artifact` unchanged in shape (still returns `os.path.getsize(file)`);
  the saved file is now flat for every format.
- `_container_overhead` kept (the `inner` branch is simply never triggered
  now).

**Other formats checked:** fp16/bf16/fp32 already stored a flat state dict;
int4_simulated stores a packed-buffer dict (`{packed, fp32}`) — no
nested-pickle container anywhere else. Those 30 non-INT8 rows were **left
unchanged and NOT re-run**.

### `code/lab3/c2.py`

- New driver `run_c4_artifact_rerun` (CLI: `uv run python -m lab3 c4rerun`):
  re-runs the 12 INT8 rows as new records (phase `precision`: int8_dynamic +
  int8_static per arch; phase `factorial`: dense_int8 + sparse_int8 per arch,
  using the selected INT8 method from int8_selection.json), each with
  `supersedes` → the C3 row; recomputes and rewrites
  `results/int8_selection.json`.
- `write_summary` / `_write_container_overhead_md`: C4 title, corrected
  container-overhead section (replaces the wrong container-overhead
  explanation with the nested-pickle cause), zip listing now outer-only
  (no inner-zip entries).

### `tests/test_precision.py`

- New test `test_int8_artifact_is_flat_state_dict` (3 archs × 2 INT8 methods =
  6 cases): for each INT8 artifact asserts (a) the saved `state` is a flat
  state_dict (no `pickled` / `quantized_state` wrapper, keys == the converted
  model's state_dict keys), (b) **`artifact_bytes <= sum(bytes_by_dtype) +
  16 KB`**, (c) reload reproduces the in-memory model's logits exactly
  (max_diff == 0.0).

## 3. Rerun: before/after bytes per artifact

All 12 rows `status=ok`, `reload_equivalence.max_abs_diff = 0.0` (atol 0).

| arch | phase | method | C3 run_id (short) | C3 bytes | C4 run_id (short) | C4 bytes | payload |
|---|---|---|---|---|---|---|---|
| mlp | precision | int8_dynamic | e63a8c | 357,075 | 7df957 | 242,481 | 236,328 |
| mlp | precision | int8_static | e95341 | 359,757 | 25d6f7 | 243,001 | 236,340 |
| mlp | factorial | dense_int8 | 2eb647 | 359,751 | e2d198 | 242,981 | 236,340 |
| mlp | factorial | sparse_int8 | 1919d5 | 260,301 | 5ad684 | 243,001 | 236,340 |
| cnn | precision | int8_dynamic | 2a7da2 | 245,907 | 9e1d6e | 163,249 | 157,224 |
| cnn | precision | int8_static | 07de9a | 127,291 | f082d1 | 127,291 | 117,672 |
| cnn | factorial | dense_int8 | da4fa4 | 245,895 | 3976fe | 163,213 | 157,224 |
| cnn | factorial | sparse_int8 | 68c2dd | 192,077 | 5e3be7 | 163,231 | 157,224 |
| gcn | precision | int8_dynamic | 832da3 | 43,155 | d73a16 | 28,885 | 23,224 |
| gcn | precision | int8_static | 96f676 | 49,805 | 312d78 | 30,009 | 23,272 |
| gcn | factorial | dense_int8 | 041ecf | 49,799 | 5b95aa | 29,989 | 23,272 |
| gcn | factorial | sparse_int8 | 15aa3d | 41,549 | 24dba8 | 30,009 | 23,272 |

Every C4 artifact satisfies `artifact_bytes <= sum(bytes_by_dtype) + 16 KB`
(overhead is now 4.6–9.9 KB of zip structural bytes, identical in shape for
dense and sparse; the mlp dense/sparse gap is now 20 bytes).

Note: cnn-precision-int8_static's C3 and C4 byte counts coincide (127,291) —
C3's cnn-static artifact was already the flat `quantized_state` state dict
(the C3 code special-cased `arch == "cnn"`), so the C4 change only removed the
wrapper key there.

val_acc / nonzero_after_int8 / frac_macs are unchanged from C3 (the artifact
format does not affect the model, only its serialization).

## 4. int8_selection.json

**Recomputed — the selection did NOT change:** mlp→int8_static,
cnn→int8_dynamic, gcn→int8_static (val_acc identical to C3; latencies within
normal run-to-run noise: mlp 0.4044 vs 0.4047 ms, cnn 7.2922 vs 7.2603 ms,
gcn 1.2745 vs 1.1996 ms). Only the `run_id`/`latency_ms` fields now point to
the C4 rows and the `code` hash updated
(442b647a8cc1c7eae06a979e20853f6e391f73eac00ef4209f4623013bfcd6ae).

## 5. Regenerated outputs

| file | status |
|---|---|
| `results/runs.jsonl` | 12 C4 INT8 rows appended (with `supersedes` → C3) |
| `results/factorial.csv` | 12 rows (latest ok per arch×condition: C4 int8 + C3 fp32) |
| `results/partc_summary.md` | regenerated (C4 title; container-overhead section now explains the nested-pickle cause and the wrong C3 bias explanation) |
| `results/artifact_zip_listing.md` | regenerated (outer-only entries; no inner zip) |
| `results/int8_selection.json` | recomputed (same selection, new run_ids) |
| `results/ANNOTATIONS.md` | C4 section appended ("superseded by C4: nested-pickle artifact container" for the 12 C3 INT8 rows) |
| `results/SCHEMA.md` | C4 changelog appended |
| `figures/precision_size_latency.png` | regenerated |
| `figures/factorial.png` | regenerated |

Not touched: C2_DONE.md / C3_DONE.md (immutable), the 30 non-INT8 Part C rows
(fp16/bf16/int4/dense_fp32/sparse_fp32), and their artifacts.

## 6. Test status

```
$ PYTHONPATH=code uv run pytest -q tests/
153 passed, 133 warnings in 153.92s (0:02:33)
```

All 153 tests pass (71 pre-existing precision tests incl. C3's
`test_int8_bytes_count_quantized_weights` + 6 new C4
`test_int8_artifact_is_flat_state_dict` cases + 76 other). No test was
weakened; the 6 new assertions are strict (flat state_dict, ≤ payload+16 KB,
reload atol 0).

## 7. Files changed

| file | change |
|---|---|
| `code/lab3/precision.py` | `_artifact_state`: INT8 → flat state_dict (with `_SDWithMeta` carrying `_metadata`); removed nested-pickle and `quantized_state` paths. `_rebuild_from_artifact`: INT8 → rebuild-from-parent + `load_state_dict(strict=True)` + GCN `_packed_weight_N` re-assignment. |
| `code/lab3/c2.py` | `run_c4_artifact_rerun` driver; `write_summary`/`_write_container_overhead_md` C4 title + corrected explanation |
| `code/lab3/cli.py` | `c4rerun` subcommand |
| `tests/test_precision.py` | `test_int8_artifact_is_flat_state_dict` (6 cases) |
| `results/*` + `figures/*` | regenerated as listed in §5 |
