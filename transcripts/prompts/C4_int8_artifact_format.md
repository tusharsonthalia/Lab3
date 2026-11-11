Task C4 — fix the INT8 artifact format (narrow). Fresh session. Read results/C3_DONE.md section 2,
results/artifact_zip_listing.md, and the artifact save/reload code in code/lab3/precision.py
(_save_artifact, _artifact_state, _rebuild_from_artifact). Use `PYTHONPATH=code uv run ...`.
Keep replies short; act through tool calls. Never weaken tests.

Review of C3 issue 2: your explanation ("the dense-vs-sparse gap tracks the larger fp32 bias tensors") is
wrong — the MLP's biases total ~2 KB, not ~99 KB. The real problem is the artifact FORMAT: for INT8 the
saved file is a dict whose "state" is {"pickled": <bytes of an inner torch.save of the whole live
module>}, i.e. a nested container. I measured the standard format myself on this machine:
  q = quantize_dynamic(copy.deepcopy(mlp_seed0), {nn.Linear}, dtype=torch.qint8)
  torch.save(q.state_dict(), f)  ->  241,351 bytes (entries: 200,704 + 32,768 int8 storages, data.pkl 1,473)
and the SAME 241,351 bytes for the S1-90% sparse MLP (zeros cost 1 byte in int8, as expected), versus
your artifacts 359,751 (dense) / 260,237 (sparse).

Fix: every INT8 artifact (dynamic and static, all 3 archs, standalone and factorial) must be a single
torch.save of the converted model's state_dict() — no nested pickled bytes, no fp32 copy of the
weights. Reload = rebuild the same quantized structure from the fp32 parent architecture (same
quantize_dynamic / FX prepare+convert / GCN per-projection path, with any calibration-free dummy
parameters), then load_state_dict(torch.load(file), strict=True); logits must match the in-memory
converted model exactly (atol 0). artifact_bytes = os.path.getsize(file). Check the other formats too:
if any fp16/bf16/int4 artifact also uses the nested-pickle container, fix it the same way (state_dict
or packed-buffer dict) and rerun those rows as well; otherwise leave them.
Add a test: for each INT8 artifact, artifact_bytes <= sum(bytes_by_dtype) + 16 KB.

Then rerun the affected rows as new records (phase precision / factorial), mark the superseded rows in
results/ANNOTATIONS.md ("superseded by C4: nested-pickle artifact container"), regenerate
results/artifact_zip_listing.md, results/factorial.csv, results/partc_summary.md (replace the wrong
container-overhead explanation with the correct one) and the Part C figures. int8_selection.json:
recompute and state whether it changed. `uv run pytest -q` (with PYTHONPATH=code).
When everything is complete and tests pass, write results/C4_DONE.md with your report (before/after
bytes per artifact, test output), then stop.
