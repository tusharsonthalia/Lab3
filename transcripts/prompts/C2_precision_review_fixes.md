Task C2 — review corrections to Part C, then a clean rerun of the whole Part C study. Fresh session.
Read predictions.md Part C (FROZEN), results/C1b_DONE.md (the previous report), results/partc_summary.md,
results/SCHEMA.md, code/lab3/precision.py, code/lab3_precision.py, tests/test_precision.py.
uv only (`uv run ...`; the package imports normally now). Never weaken tests. Never select using test.
Because the endpoint sometimes cuts replies off, keep replies short and act through tool calls.

I reviewed C1b and verified the following myself:

1. ENVIRONMENT (my mistake, now fixed). All 45 Part C rows from C1b ran on Python 3.13.14, while Parts A
   and B ran on Python 3.12.13. I pinned the project (.python-version = 3.12, requires-python
   >=3.12,<3.13) and rebuilt the venv: now Python 3.12.13, torch 2.15.0, PyG 2.8.1, numpy 2.5.4 (same as
   A/B); 119 tests pass. Therefore ALL of Part C must be rerun after the fixes below. Mark every existing
   part="C" row from C1b (timestamps 2026-11-09T22:4x onward) in results/ANNOTATIONS.md as "C1b —
   superseded by C2 (Python 3.13 env mismatch + INT4/bytes/nonzero bugs)". Never edit runs.jsonl.

2. INT4 BUG. Your Int4Packed* models score chance (val 0.0824 / 0.0942 / 0.1260), which you reported as a
   real result. It is not. A reference per-output-channel symmetric [-8,7] quantize-dequantize of the
   seed-0 parents (scale = absmax/7 per output row of the weight flattened to [out, -1], round, clamp,
   multiply back) gives val MLP 0.8914, CNN 0.9006, GCN 0.7860. Find the bug (likely nibble packing
   order or sign extension on unpack, a transposed shape, or scale broadcasting) and add a test: for all
   3 archs, logits of the INT4 packed model loaded FROM DISK equal those of this reference
   quantize-dequantize within 1e-5, and val accuracy matches.

3. CNN static INT8 artifact is 1,805 bytes — impossible for ~117k int8 weights. The "deterministic
   re-convert on reload" means the saved file does not contain the quantized weights. Save the converted
   model's quantized state_dict (weights + scales + zero points + packed params), reload by rebuilding
   the FX-converted structure and load_state_dict from the FILE, then compare logits. artifact_bytes must
   be the size of that file. Apply the same "reload from disk" requirement to every format.

4. Byte breakdown. For every Part C artifact record bytes_by_dtype (sum of numel*element_size per stored
   dtype: int8/quint8/int4-packed-uint8/fp16/bf16/fp32 scales/fp32 biases) next to artifact_bytes, so
   container overhead is visible. MLP INT8 at 357 KB vs ~235k int8 weights needs this explanation.

5. nonzero_params is 0 for every INT8 row. Count nonzero entries of the integer weight representation
   (int_repr() of quantized weights, unpacking packed params if needed). For sparse+INT8 this is the
   evidence for whether zeros survive quantization: report nonzero before and after INT8.

6. Factorial: MLP sparse_int8 failed and the table has duplicate rows from reruns. After the fixes,
   produce exactly 12 conditions, all from the C2 run. If a condition fails, a classified failed row.

7. Package hygiene: move the driver from code/lab3_precision.py into the package as CLI subcommands
   `uv run python -m lab3 precision` and `uv run python -m lab3 factorial`. Move results/_quarantine/
   contents to the top-level _quarantine/ and delete the empty results/_quarantine directory.

8. Figures: C1 item 7 asked for figures/precision_size_latency.png (per arch: artifact bytes vs batch-256 /
   full-graph latency, one point per format, log axes, unsupported formats listed in the caption) and
   figures/factorial.png (per arch: accuracy, bytes, latency for the 4 conditions). Add both; keep the
   valacc/latency/frac_macs figures. Legends must render; assert non-empty panels.

Then rerun everything in Part C on the 3.12 env with phase "precision" / "factorial" (new records):
standalone 5 formats x 3 archs, the INT8 selection by the frozen C.1 rule (recomputed from the new
rows -> results/int8_selection.json), the 12-condition factorial, figures, results/partc_summary.md (must
include artifact_bytes, bytes_by_dtype, nonzero before/after, latency bs1 + bs256/full graph with IQR, rss,
frac_macs, stored/compute/accumulator precision, run_ids). Run `uv run pytest -q` last.
When everything is complete and tests pass, write results/C2_DONE.md containing your full final report
(support matrix, standalone table, int8 selection, factorial table, test output, anomalies; no
interpretation), then stop.
