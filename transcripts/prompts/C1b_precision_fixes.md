Task C1b — finish Part C. Fresh session; the previous session (C1) ended mid-task on a malformed tool
call and produced no deliverables. Project root is ~/cmsc35200/lab3 (it is now a symlink to the course
folder; keep using this short path). The project is now an installed uv package: `uv run python -m lab3`
and `uv run pytest` work without PYTHONPATH. uv only. Read predictions.md Part C (C.1–C.3, FROZEN),
results/SCHEMA.md, results/partc_selection.json and the existing code/lab3/precision.py first.

My review of C1 (I verified the backend myself on this machine; reproduce and record these probes):
- engines = ['qnnpack']. torch.ao.quantization.quantize_dynamic works and yields native
  torch.ao.nn.quantized.dynamic Linear modules. FX static (prepare_fx/convert_fx with
  get_default_qconfig_mapping("qnnpack"), calibrate, convert) works and yields native LinearReLU /
  quantized modules. torch.ao.quantization.quantize_pt2e does NOT exist; torchao is not installed.
  nn.Linear runs natively in float16 and bfloat16 on CPU (output dtype stays fp16/bf16).
- OVERRULED: QuantLinear8 / QuantConv8 in precision.py compute INT8 with hand-written int64 matmuls in
  Python. That is an emulation, not the backend's INT8 kernel, so its latency is meaningless as "INT8 on
  this laptop". Remove them from the INT8 path. INT8 rows must use the native modules:
  dynamic = quantize_dynamic({nn.Linear}, qint8); static = FX prepare_fx/convert_fx with the qnnpack
  qconfig mapping, calibrated per C.1. Conv2d static INT8 via FX on qnnpack: try it; if the backend
  rejects it, record a classified unsupported row and leave conv in fp32 (counted in frac_macs).
- GCN: GCNConv/propagate is not FX-quantizable. Quantize only the projection by swapping each
  conv.lin for an nn.Linear with identical weights (verify logits equal before quantizing), then apply
  dynamic / static to those Linears; aggregation stays fp32 and is counted as fp32 MACs. If that fails,
  classified unsupported row.
- BUG: the C1 fp16 rows score 5-15% (chance). A correct fp16 cast loses < 0.5 pp. Find and fix the cause
  (check the checkpoint is loaded, the input is normalized exactly as in eval, and both model and input
  are cast). Add a test: fp16 and bf16 val accuracy within 1 pp of fp32 for all three architectures.
- BUG: precision.py loads splits via RESULTS_DIR.parent / "splits", which breaks when tests redirect
  LAB3_RESULTS_DIR. Use the project root, as the rest of the package does.
- The 15 part="C" rows already in runs.jsonl are C1 development/debug rows: list their run_ids in
  results/ANNOTATIONS.md as "C1 debug — excluded"; move the matching files in results/precision_artifacts/
  into _quarantine/c1_debug/. Never edit runs.jsonl.
- INT4 simulated weight-only (Int4Packed*) is fine to keep as long as it is labelled
  simulated_non_native and follows C.1 (per-output-channel symmetric [-8,7], fp32 scales, 2 per uint8).

Then complete the original C1 task items 1-9 exactly as written in transcripts/prompts/C1_precision.md
(support report with probe evidence, all conversions for all 3 archs with the full accounting fields,
reload-equivalence, tests, INT8 selection by the frozen rule, the 12-condition factorial, figures,
partc_summary.md). Write real records with phase "precision" / "factorial".

Final item — Part B figure cleanup (scripts/make_figures.py only, no data changes): the three Part B
figures render with an EMPTY band where the shared legend should be (fig.legend is not drawing — likely
no handles/labels collected, or drawn outside the saved bbox). Make the legend actually render (verify by
checking len(legend.get_texts()) > 0 and use bbox_inches="tight"), and fix overlapping minor tick
labels on the CNN latency axis (turn off minor tick labels on log axes).

Run `uv run pytest -q` last. Report: support matrix, standalone table, int8 selection, factorial table,
test output, anomalies. Do not interpret.
