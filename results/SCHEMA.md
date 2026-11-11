# Lab 3 `results/runs.jsonl` record schema

Every command (`train`, `eval`, `bench`) appends exactly one JSON object per
attempted run to `results/runs.jsonl` (append-only: lines are never edited or
deleted; a corrected re-bench adds a new line whose `supersedes` lists the
earlier, now-incorrect run_ids). This file documents the fields.

## Identity / linkage

| field | type | meaning |
|---|---|---|
| `run_id` | str | Unique id for this record: `<arch>-<method>-<UTC timestamp>-<shortuuid>`. |
| `status` | str | `"ok"` or `"failed"`. |
| `reason` | str/null | Failure reason when `status == "failed"`. |
| `part` | str | `"A"` (dense reference). |
| `phase` | str/null | Matrix phase the record belongs to: `"pilot"` for the Part B pilot matrix (seed 0, including the B.6 `dense_ft` control arms) and `"confirm"` for the confirmatory reruns (seeds 1/2/3, B.3). **B2 (2026-11-09, before any pilot-matrix run):** every new Part B record carries `phase`; all earlier records (Part A, and the B1/B1b Part B smoke rows) predate the field and carry `null` in any record written by new code, or no field at all (never edited — the log is append-only). |
| `architecture` | str | `mlp` / `cnn` / `gcn`. |
| `method` | str | `"dense"` for the Part A baseline. |
| `seed` | int | Random seed. |
| `command` | str | `train` / `eval` / `bench`. |
| `parent_run_id` | str/null | For `bench` (and `eval`), the `run_id` of the **training run** that produced the checkpoint being benchmarked. Carried from the checkpoint metadata `run_id`. |
| `supersedes` | list[str] | For a corrected re-bench: the `run_id`s of the earlier bench records for the same (architecture, seed) that are now known to be incorrect (e.g. A1's wrong MAC/byte figures). The old lines remain in the log untouched. |

## Training / selection

| field | type | meaning |
|---|---|---|
| `train_epochs` | int/null | Epochs actually run (may stop early). |
| `finetune_epochs` | int | 0 in Part A. |
| `best_epoch` | int/null | Epoch of the validation-selected checkpoint. |
| `val_acc`, `val_loss` | float/null | Validation accuracy / loss. |
| `test_acc` | float/null | Test accuracy (evaluated only on the selected checkpoint). |

## Size

| field | type | meaning |
|---|---|---|
| `total_params` | int | `sum(p.numel() for p in model.parameters())`. |
| `nonzero_params` | int | Number of nonzero weight/param elements. For the dense baseline this equals `total_params`; Part B arms report the physically sparse count. |
| `dense_bytes` | int | **Actual on-disk size** (bytes) of a `torch.save(state_dict)` file — the bare state dict (no metadata) written to a temp file, read with `os.path.getsize`. Includes the PyTorch zip/pickle container overhead, so it is slightly larger than `raw_tensor_bytes`. |
| `raw_tensor_bytes` | int | `sum(t.numel() * t.element_size())` over every state_dict tensor — pure payload bytes, no container overhead. |
| `gzip_bytes` | int/null | **P1 (A2 review):** gzip-9 size (bytes) of the `dense_bytes` state_dict file. Reported for **every run, dense included** — it is a property of the dense container, not of a compression method. Null only where a run failed before measuring it. |
| `compressed_bytes` | int/null | **P1 (A2 review):** RESERVED for the **compact-export artifact** (Part B path 2: the genuinely sparse on-disk file — S2 rebuilt dense model, S1/S3 CSR file, S4 packed N:M file). **Null for the dense Part A baseline** (dense has no compact export). Part A `bench` rows record the gzip-9 figure in `gzip_bytes`, not here. |

### Part B (sparsity) additions (Task B1)

Pruning arms (`command == "prune"`, `part == "B"`) append **two rows per arm**,
`stage="post_prune"` and `stage="post_finetune"`, sharing one `arm_id`.
(The pilot driver `uv run python -m lab3 pilot` appends exactly these rows with
`phase="pilot"`, seed 0, one per cell of {mlp,cnn,gcn} x {s1,s2,s3,s4} x
{0.25,0.5,0.75,0.9} plus one `dense_ft` control arm (target 0) per architecture;
failed arms additionally append `stage="failed"` rows. Progress: one line per
arm in `results/pilot_log.txt`; summaries: `results/pilot_summary.csv` (one line
per arm x stage) and `results/pilot_per_layer.csv` (arm, stage, per tensor).)

| field | type | meaning |
|---|---|---|
| `part` | str | `"B"`. |
| `command` | str | `prune`. |
| `stage` | str | `post_prune` / `post_finetune`. |
| `arm_id` | str | Shared by the arm's two rows: `<arch>-<method>-t<target>-seed<seed>-<UTC ts>-<uuid6>`. |
| `method` | str | `s1` / `s2` / `s3` / `s4`. |
| `target_sparsity` | float | Frozen target (0.25 / 0.5 / 0.75 / 0.9). |
| `achieved_sparsity` | float | Fraction of **eligible weight** elements that are zero (post_prune / post_finetune each). |
| `achieved_param_sparsity` | float | Fraction of **all dense params** that are zero (includes biases and, for S2, the rebuilt model's smaller param total). |
| `nm_pattern` | str/null | S4 only, e.g. `"2:4"`. |
| `group_sparsity` | float/null | S2 only: fraction of groups removed (the frozen target's group fraction). |
| `parent_run_id` | str | The **canonical seed-0 dense training run** for this architecture (from the checkpoint metadata). |
| `checkpoint` | str/null | Path of the saved pruned post-finetune model, `checkpoints/pilot/<arm_id>.pt` (set on both rows; file exists once the run completes). |
| `per_layer` | list[dict] | One entry per eligible tensor: `name`, `shape` (dense parent shape for S1/S3/S4; rebuilt shape for S2), `numel`, `nonzero`, `sparsity`. |
| `paths` | dict | Two runtime paths (B.4), keyed `dense_zeros` and `compact`: `runtime_status` (`ok` / `unsupported:<reason>` / `dense_fallback:<reason>`), `latency` (frozen protocol), `rss_before_model_bytes`, `peak_rss_bytes`, `model_rss_delta_bytes`, and the path's size fields: `dense_zeros` → `dense_bytes` + `gzip_bytes`; `compact` → `compressed_bytes` (the compact artifact file size) and `artifact` (path). For S4 the `dense_zeros` row is the benchmarked dense fallback when the semi-structured runtime is unsupported; see `paths.compact.runtime_status`. |

`status=failed` rows carry a classified `reason`: `unsupported_backend`,
`numerical`, `bug`, `not_applicable` (plus the exception message after `:`).

## Compute (MAC estimates)

Computed from the actual model's layer shapes (works for any
Linear/Conv2d/GCNConv sizes), not hard-coded. Per single sample (per image /
per node for fmnist-MLP/CNN; full graph for GCN).

| field | type | meaning |
|---|---|---|
| `est_macs` | int | Total forward MACs assuming **all weights are nonzero** (dense convention). |
| `est_macs_nonzero` | int | Same count but only over weight elements that are **actually nonzero** in the loaded checkpoint. Equals `est_macs` for the dense baseline; smaller once Part B prunes weights. |
| `est_macs_sparse_input` | int/null | GCN only: `est_macs` with the layer-1 projection counted as `nnz(X) * out` instead of `N * in * out`, because Cora's node features are ~1.3% dense (sparse-input convention). Null for MLP/CNN. |
| `gcn_norm_macs` | int/null | GCN only: the degree-normalization (D^{1/2}) elementwise scaling multiplications, `2 * out * (E + N)` per GCNConv layer, reported **separately** and excluded from `est_macs`. Null for MLP/CNN. |

### MAC conventions

- **Linear(in, out):** `in * out` per token (1 per image / per node). Bias excluded.
- **Conv2d(cin, cout, k, padding):** `cin * cout * k^2 * H_out * W_out` per image
  (`H_out`/`W_out` are the actual output feature-map dims after padding). MaxPool/ReLU contribute 0.
- **GCNConv(in, out)** (torch_geometric, self-loops added): the weight is applied **once per
  node** (`X @ W`), not per edge. Per layer:
  - projection = `N * in * out` (dense-X weight product),
  - aggregation = `(E + N) * out` (weighted sum over the E directed edges plus N self-loops).
  - `est_macs` = sum over layers of (projection + aggregation); normalization scales go in `gcn_norm_macs`.
  - `est_macs_sparse_input` = `est_macs` but with the layer-1 projection replaced by `nnz(X) * out`.
  - For Cora: N = 2,708, E = 10,556, in1 = 1,433, out1 = 16, in2 = 16, out2 = 7 →
    `est_macs` ≈ 6.27e7, `est_macs_nonzero` = `est_macs` (dense),
    `est_macs_sparse_input` ≈ 1.0e6.

### Dense seed-0 reference values

- MLP: `est_macs` = 234,752 per image.
- CNN: `est_macs` = 1,697,280 per image (conv2 output is 10×10: 28 → conv1 pad2 → 28 →
  pool → 14 → conv2 5×5 no pad → 10 → pool → 5).

## Memory

Measured in a fresh subprocess that (1) imports torch + loads the **small fixed
benchmark input** (never the 55k training tensor), records `rss_before_model_bytes`,
(2) loads the checkpoint and runs a few forwards, recording the peak.

| field | type | meaning |
|---|---|---|
| `rss_before_model_bytes` | int | RSS after imports + benchmark-input load, **before** loading the model. |
| `peak_rss_bytes` | int | Peak RSS observed while loading the model and running forwards. |
| `model_rss_delta_bytes` | int | `peak_rss_bytes - rss_before_model_bytes` — the memory attributable to the model + activations. |

## Latency

Frozen protocol: `torch.inference_mode()`, `model.eval()`, 20 warm-up + 200 timed
iterations, `time.perf_counter_ns`, median and IQR reported. Inputs are the fixed
**benchmark batch** (first 256 TEST images, inputs only) — not training images.

`latency` is a dict keyed by `batch_1` / `batch_256` (fmnist) or `full_graph` (gcn),
each with: `median_ms`, `iqr_ms`, `warmup`, `iters`, `threads`, `batch_size`.

## Provenance

| field | type | meaning |
|---|---|---|
| `wall_time_s` | float | Wall-clock seconds for the command. |
| `code_hash` | str | SHA-256 over all files under `code/` (git-free). |
| `versions` | dict | python / torch / numpy / platform versions. |
| `timestamp` | str | ISO-8601 UTC creation time. |
| `target_sparsity`, `achieved_sparsity`, `stored_precision`, `compute_precision`, `accumulator_precision` | — | 0.0 / "fp32" for the dense baseline. |

## Fixed benchmark input

`results/bench/fmnist_bench_input.pt` = first 256 TEST Fashion-MNIST images,
inputs only (no labels), saved once. `results/bench/cora_bench_input.pt` = full-graph
Cora features + edge_index. These are the latency/RSS inputs and are independent of
the 55k training tensor.

## C3 changelog (2026-11-10)

C3 re-ran only the INT8 records (12 new rows: 6 standalone precision + 6
factorial) to fix two accounting bugs in the C2 INT8 results:

1. **Integer-weight accounting blind to quantized projections** (the reviewer's
   issue 1): the C2 traversal of `int8_nonzero` / `bytes_by_dtype` /
   `nonzero_after_int8` matched only modules whose `_packed_params` child
   exposed a working `_weight_bias()`, so it missed the FX-static shapes
   produced by `convert_fx` — GCN per-projection GraphModules with a raw
   `_packed_weight_N` `torch.ScriptObject` and CNN static convs whose
   `_packed_params` is a raw ScriptObject. GCN static INT8 reported
   `nonzero_after_int8 = 23` (biases) and `bytes_by_dtype = {"fp32": 92}`
   (biases only). C3 adds `_qnnpack_weight_bias` (handles both the Python
   wrapper `_weight_bias` path and the raw-ScriptObject `unpack()` path) and
   uses it in all three functions.

2. **Unexplained container overhead** (issue 2): C3 adds a
   `container_overhead` field to factorial rows (per-entry
   `zipfile.infolist()` of the saved artifact, including the inner zip of the
   `state['pickled']` bytes for mlp/gcn INT8 rows) and writes the full
   per-entry listing for all 12 factorial artifacts to
   `results/artifact_zip_listing.md`. The dense-vs-sparse file-size gap is the
   nested-zip re-encoding of the needed data, not dead weight.

The C2 INT8 rows (12) are marked superseded in `results/ANNOTATIONS.md`;
the C3 rows carry `supersedes` pointing to their C2 predecessors.

**Correction of the C2 `pip install -e .` explanation**: C2_DONE.md (immutable)
attributes the editable-install failure to a `#`-comment issue in the `.pth`
file. The real cause was that `uv venv --seed` writes the site-packages `.pth`
files with the macOS `UF_HIDDEN` (0x80000000) file flag, and CPython's
`site.py` (3.12) skips hidden `.pth` files, so `lab3.pth` was ignored and
`import lab3` failed. This correction is recorded here and in
`results/partc_summary.md`.

## C4 changelog (2026-11-10)

C4 re-ran only the 12 INT8 records (6 standalone precision + 6 factorial)
because the C3 INT8 artifact **format** was wrong: for mlp/gcn INT8 (dynamic
and static) the saved file was a dict whose `state` was
`{"pickled": <bytes of an inner torch.save of the whole live module>}` — a
**nested-pickle container** (a zip re-encoded inside the outer .pt zip), which
bloated the artifacts ~120 KB above the payload. C3's dense-vs-sparse
explanation ("the gap tracks the larger fp32 bias tensors") was wrong: the
MLP biases total ~2 KB, not ~99 KB; the gap was the nested re-encoding.

Changes:

1. **Artifact format (code/lab3/precision.py `_artifact_state` /
   `_rebuild_from_artifact`)**: every INT8 artifact (dynamic and static, all
   3 archs, standalone and factorial) is now a single `torch.save` of
   `{kind, architecture, state}` where `state` is the converted model's
   `state_dict()` (int8 weights + fp32/int64 scales + zero points; for
   FX-static GCN, the raw `_packed_weight_N` ScriptObjects are carried as
   state-dict values and re-assigned directly on the rebuilt GraphModules
   after `load_state_dict(strict=True)`, which cannot set raw instance
   attributes). The `state` dict carries the state_dict `_metadata`
   (per-module `version` entries, via a module-level `_SDWithMeta(dict)`
   subclass) which the qnnpack dynamic/fused Linear loaders require. No
   nested pickled bytes, no fp32 weight copy. Reload = rebuild the same
   quantized structure from the fp32 seed-0 parent (same
   `quantize_dynamic` / FX prepare+convert / GCN per-projection path) +
   `load_state_dict(strict=True)`; reloaded logits match the in-memory
   converted model exactly (atol 0). `artifact_bytes =
   os.path.getsize(file)` and `artifact_bytes <= sum(bytes_by_dtype) + 16 KB`
   for every INT8 artifact.
2. **New test** `tests/test_precision.py::test_int8_artifact_is_flat_state_dict`
   (3 archs × 2 INT8 methods): asserts the saved `state` is a flat state_dict
   (no `pickled` / `quantized_state` wrapper), `artifact_bytes <=
   sum(bytes_by_dtype) + 16 KB`, and reload logits match at atol 0.
3. **Driver** `lab3 c4rerun` (code/lab3/c2.py `run_c4_artifact_rerun`):
   re-runs the 12 INT8 rows as new records (phase precision / factorial) with
   `supersedes` pointing to the C3 rows; recomputes and rewrites
   `results/int8_selection.json` (selection unchanged: mlp→int8_static,
   cnn→int8_dynamic, gcn→int8_static; only run_ids/latencies/code hash
   changed).
4. **fp16/bf16/int4 artifacts checked**: already flat (fp16/bf16/fp32 = flat
   state dict; int4 = packed-buffer dict) — left unchanged, not re-run.
5. **Regenerated**: `results/partc_summary.md` (C4 title; the container
   overhead section now explains the nested-pickle cause and the C3
   misexplanation), `results/artifact_zip_listing.md` (outer-only entries; no
   inner zip), `results/factorial.csv` (C4 int8 rows + C3 fp32 rows),
   `results/int8_selection.json`, `figures/precision_size_latency.png`,
   `figures/factorial.png`.
6. The 12 C3 INT8 rows are marked superseded in `results/ANNOTATIONS.md`
   ("superseded by C4: nested-pickle artifact container").
