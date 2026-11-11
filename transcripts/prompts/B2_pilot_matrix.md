Task B2 — one small fix, then run the complete Part B pilot matrix. Fresh session: read predictions.md
Part B (B.1–B.6), results/SCHEMA.md, results/ANNOTATIONS.md and code/lab3/{prune,compact,cli}.py first.
uv only. Never weaken tests. Do not change any frozen design choice.

Review notes from B1b (I verified these myself):
- The CSR artifacts are still written with int64 indices (e.g. results of the B1b CNN smoke:
  conv1/conv2/fc1/fc2 indices and indptr are torch.int64) even though compact.CSR_INDEX_DTYPE probes to
  torch.int32. prune.py's dense_to_csr builds its own int64 tensors. Fix: the saved CSR artifact and the
  runtime CSR must both use CSR_INDEX_DTYPE; keep csr_int64_bytes as the separately-computed alternative.
  Add a test asserting the saved artifact's index dtype == CSR_INDEX_DTYPE.
- The B1b smoke records were written to /tmp/lab3_smoke_final; I archived them to
  results/smoke_b1b/runs.jsonl. From now on every real run writes to results/runs.jsonl.

Then build and run the pilot:
1. Add a `phase` field to every new record: "pilot" here (later "confirm"). Document it in SCHEMA.md.
2. Pilot driver: `uv run python -m lab3 pilot [--arch ...]` running, with seed 0, every cell of
   {mlp,cnn,gcn} x {s1,s2,s3,s4} x {0.25,0.50,0.75,0.90} (48 arms; S4 uses the frozen patterns 3:4,
   2:4, 1:4, 1:10) plus dense_ft for each architecture (3 control arms). Each pruning arm emits its
   post_prune and post_finetune rows. A failed arm must write a failed row with a classified reason and
   the driver continues with the next arm. Idempotent: on restart, skip arms that already have an ok
   phase=pilot post_finetune row. Print one progress line per arm to results/pilot_log.txt.
3. Run it for real with nohup (e.g. `nohup uv run python -m lab3 pilot > results/pilot_log.txt 2>&1 &`)
   and poll with sleep (e.g. `sleep 120`) until it finishes. Rerun only failed arms once if the failure is
   a code bug you then fix (record the fix; the failed row stays). Do not rerun arms that succeeded.
4. Write results/pilot_summary.csv (one line per arm and stage: arch, method, target, pattern,
   achieved_sparsity, achieved_param_sparsity, val_acc, test_acc, dense_bytes, gzip_bytes,
   compressed_bytes, est_macs_nonzero, latency medians/IQR for both paths, runtime_status, run_id) and
   results/pilot_per_layer.csv (arm_id, stage, layer, shape, numel, nonzero, sparsity).
5. Do NOT select confirmatory points and do NOT run confirmatory seeds — I will apply the frozen
   selection rule myself.

Report: test output, how many arms ok/failed (with reasons), total wall time, and the path to the
summary files. Point out any anomalous records (e.g. accuracy above dense_ft, achieved sparsity far from
target, latency IQR > 20% of median), but do not interpret them.
