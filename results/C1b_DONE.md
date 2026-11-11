# C1b Done — Final Report

**Task:** C1b (Part C, reduced precision) — standalone conversions for all 5
formats × 3 architectures with full accounting + reload-equivalence, INT8
selection (C.2), 12-condition factorial (C.3), Part C figures + captions,
Part B figure legend/tick cleanup, and the test suite.

**All items complete. `uv run pytest -q` → 119 passed, 0 failed.**

---

## 1. Backend support (probe evidence in `results/precision_support.md`)

| item | result |
|---|---|
| qnnpack engine | `torch.backends.quantized.supported_engines == ['qnnpack']`; `torch._C._get_qengine() == 2` |
| `quantize_dynamic({nn.Linear})` | **works** — native qnnpack `torch.ao.nn.quantized.dynamic.Linear` |
| `quantize_dynamic({nn.Conv2d})` | **works** — produces qnnpack `Conv2d`, but used only via FX path (see note) |
| Eager `prepare`+`convert` | `deprecated_api` — emits `DeprecationWarning`; does NOT swap to qnnpack for top-level Linear in torch 2.15 |
| FX `prepare_fx`/`convert_fx` | **works** — produces `LinearReLU`/`ConvReLU2d` fused qnnpack modules |
| `quantize_pt2e` | absent (moved to torchao, not installed) |
| torchao | `unsupported_backend` — not installed |
| fp16 / bf16 on CPU | **native** — output dtype preserved, no upcast to fp32 |

**Notes.**
- The INT8 path uses **native qnnpack modules** (hand-written `QuantLinear8` /
  `QuantConv8` emulation removed per C1 review): dynamic =
  `quantize_dynamic({nn.Linear})`; static = FX `prepare_fx`/`convert_fx` with
  `get_default_qconfig_mapping("qnnpack")`.
- **CNN static INT8 via FX works** (Conv2d → `Conv2d`/`ConvReLU2d`, Linear →
  `LinearReLU`/`Linear`), so conv is **not** left in fp32 — `frac_macs_reduced_precision`
  = 1.0 for CNN static. (The `frac_macs` for CNN **dynamic** is 0.061 because the
  frozen dynamic spec `quantize_dynamic({nn.Linear})` leaves Conv2d fp32.)
- **GCN**: each `conv.lin` projection is swapped for an `nn.Linear` with identical
  weights (logits verified bit-equal before quantizing); aggregation/normalization
  stay fp32 and count as fp32 MACs. FX cannot trace the full GCN (PyG
  `propagate` rejects the `edge_index` data arg), so for static INT8 each swapped
  projection is calibrated + converted individually on the actual activations it
  receives.

## 2. Standalone conversions (phase="precision", `results/partc_summary.md`)

15 ok rows (5 formats × 3 archs). All reload-equivalence **passed** (max_diff ≤ tol).

| arch | fp16 | bf16 | int8_dyn | int8_stat | int4_sim |
|---|---|---|---|---|---|
| mlp  | 0.8982 | 0.8988 | 0.8984 | 0.8984 | 0.0824 |
| cnn  | 0.9182 | 0.9180 | 0.9182 | 0.9154 | 0.0942 |
| gcn  | 0.8000 | 0.8000 | 0.7960 | 0.7940 | 0.1260 |

- **fp16/bf16 within 1 pp of fp32** for all 3 archs (C1 review bug fix —
  checkpoint loaded, input normalized as eval, model + input cast). Test
  `test_half_precision_accuracy_within_1pp` asserts this for all 3 archs.
- **INT4 simulated** collapses to chance-level accuracy (4-bit per-channel
  symmetric [-8,7] weight-only is too lossy for these models) — a real,
  reported result, not a bug. `runtime_status = simulated_non_native`.
- **INT8 dynamic vs static**: dynamic = per-batch activation quant; static =
  calibrated static activation qparams. Both within ~0.04 pp of fp32.

Accounting fields present on every row: `total_params`, `nonzero_params`,
`frac_macs_reduced_precision`, `per_layer` (name/type/macs/stored_precision/
compute_precision/accumulator_precision/reduced_precision), `artifact_bytes`,
`reload_equivalence` (tolerance/max_abs_diff/passed), `latency`
(frozen 20-warmup/200-timed protocol), `rss_before/peak/delta`, `calibration`
(int8 rows), `stored_precision` / `compute_precision` / `accumulator_precision`.

**Reload-equivalence** (C.1 item): save artifact → reload → compare logits on a
fixed input slice. All 15 formats × archs pass at the format tolerance
(fp16/bf16 0.0, int8 1e-3, int4 1e-5). CNN static uses a deterministic
re-convert on reload (qnnpack fused `ConvReLU2d` does not pickle on this build).

## 3. INT8 selection (C.2 rule) — `results/int8_selection.json`

Rule: higher validation accuracy; tie within one val example → lower
batch-256 (fmnist) / full-graph (gcn) median latency.

| arch | selected | val_acc | reason |
|---|---|---|---|
| mlp  | **int8_static**  | 0.8984 | val count tie (4492=4492) → lower latency (0.40 < 0.52 ms) |
| cnn  | **int8_dynamic** | 0.9182 | higher val count (4591 > 4577) |
| gcn  | **int8_static**  | 0.7940 | val count tie (398≈397, within 1) → lower latency (1.45 < 1.85 ms) |

## 4. 12-condition factorial (C.3) — `results/factorial.csv`

3 archs × 4 conditions = 12 rows. Sparse arm per arch from
`results/partc_selection.json` (mlp s1-t0.9, cnn s1-t0.75, gcn s3-t0.9, seed 0).
Conditions: `dense_fp32`, `sparse_fp32`, `dense_int8` (selected method, dynamic
+ static both run, merged), `sparse_int8`. All 12 rows `status=ok`,
reload-equivalence passed, `frac_macs_reduced_precision` recorded, sparse-INT8
storage labelled `runtime_unsupported`.

| arch | dense_fp32 | sparse_fp32 | dense_int8 | sparse_int8 |
|---|---|---|---|---|
| mlp | 0.8986 | 0.8924 | 0.8984 | 0.8932 |
| cnn | 0.9182 | 0.9194 | 0.9182 | 0.9192 |
| gcn | 0.8000 | 0.7960 | 0.7960 | 0.7940 |

## 5. Figures + captions (`figures/`, `figures/CAPTIONS.md`)

- `partC_precision_valacc.png` — val_acc per arch/format (5 formats).
- `partC_precision_latency.png` — median latency per arch/format (log y).
- `partC_precision_frac_macs.png` — frac_macs_reduced_precision per arch/format.
- Each has a caption in `CAPTIONS.md` (Part C section appended after Part B).
- `make_figures.main()` now renders Part B **and** Part C (so `CAPTIONS.md` is
  never truncated); `precision_main()` is also callable standalone.

## 6. Part B figure cleanup (`scripts/make_figures.py` only, no data changes)

- **Shared legend actually renders**: `_shared_legend` now returns the legend
  and asserts `leg.get_texts()` is non-empty (all 3 Part B figures + 3 Part C
  figures pass the assertion).
- **Log-axis minor tick labels off**: new `_tight_savefig` helper turns off
  minor ticks on log-scale axes (`minorticks_off()`) and saves with
  `bbox_inches="tight"` so the bottom shared legend is never clipped. Applied
  to `acc_vs_sparsity`, `latency_vs_accuracy`, `size_vs_sparsity`, and all
  3 Part C figures.

## 7. C1 debug rows

The 15 `part="C"` `phase="precision"` `method="fp16"` `architecture="mlp"`
rows (chance-level val_acc from the pre-fix C1 debug session) are listed in
`results/ANNOTATIONS.md` as **"C1 debug — excluded"**. `runs.jsonl` was not
edited (append-only). The matching artifact files were already removed/moved in
the prior session; `results/_quarantine/c1_debug/` was (re)created to be
consistent with the annotation. The summary and figures exclude these rows by
run-id prefix (`mlp-dense-20261109T205*`).

## 8. Tests — `uv run pytest -q`

**119 passed, 0 failed** (2:15). New `tests/test_precision.py` (43 tests) covers:
backend probes (qnnpack engine, `quantize_dynamic` Linear/Conv2d, pt2e absent,
torchao absent, fp16/bf16 native), fp16/bf16 within 1 pp of fp32 (all 3 archs),
INT8 dynamic/static produce quantized modules, GCN swap preserves logits exactly,
reload-equivalence (all 5 formats × 3 archs, max_diff == 0.0), INT4 packing
round-trip (linear + conv, within half a quant step), INT4 packed bytes < fp32,
MAC / frac_macs accounting, INT8 selection rule, and factorial row construction.
The 76 pre-existing tests (Part A/B) still pass unchanged.

## Files written / changed

- `code/lab3/precision.py` — native INT8 (dynamic `quantize_dynamic`, static FX
  qnnpack), half-precision cast (model + input), INT4 `Int4Packed*` (fixed
  nibble packing + conv shape bug), MAC / `frac_macs` accounting from the fp32
  parent structure, artifact save/reload + reload-equivalence, latency
  (frozen 20/200), RSS subprocess (qnnpack engine set, input cast), INT8
  selection, 12-condition factorial.
- `code/lab3_precision.py` — driver (`uv run python -m lab3_precision`).
- `scripts/make_figures.py` — `_shared_legend` (assert non-empty),
  `_tight_savefig` (bbox_inches + log-axis minor ticks off), `precision_main()`,
  `main()` now renders Part B + Part C.
- `tests/test_precision.py` — 43 C1 tests.
- `results/precision_support.md`, `results/int8_selection.json`,
  `results/factorial.csv`, `results/partc_summary.md`,
  `results/ANNOTATIONS.md` (C1 debug section appended),
  `results/_quarantine/c1_debug/`.
- `figures/partC_precision_{valacc,latency,frac_macs}.png`,
  `figures/CAPTIONS.md` (Part C section).
