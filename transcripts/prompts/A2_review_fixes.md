Task A2 — review corrections to Task A1. I reviewed code/lab3/bench.py, models.py and tests/. The
baselines themselves are accepted (do NOT retrain). Fix the following, then re-run tests and re-bench.

1. CNN MAC estimate is wrong. conv2 output is 10x10, not 12x12 (28 -> conv1 pad2 -> 28 -> pool -> 14
   -> conv2 5x5 no pad -> 10 -> pool -> 5). Correct total = 1,697,280 MACs/image.
2. GCN MAC convention is wrong. GCNConv computes X @ W once per node, then aggregates over edges; the
   weight is not applied per edge. Per layer: projection N*in*out + aggregation (E+N)*out (+ the
   normalization scales, reported separately). Report two fields for Cora: est_macs (dense-X convention,
   N*in*out for layer 1, ~62.7M total) and est_macs_sparse_input (layer-1 projection counted as
   nnz(X)*out, since Cora features are ~1.3% dense). Document both conventions.
3. MACs must not be hard-coded. Compute them from the actual model's layer shapes (works for any
   Linear/Conv2d/GCNConv sizes) so that Part B's physically smaller models are counted correctly.
   Emit est_macs (all weights) and est_macs_nonzero (only MACs whose weight is nonzero). Add a unit
   test asserting MLP=234,752, CNN=1,697,280 per image and the GCN formula on Cora.
4. Bytes. dense_bytes must be the ACTUAL on-disk size: torch.save(state_dict only, no metadata) to a
   temp file and os.path.getsize. Also record raw_tensor_bytes (sum of numel*element_size) and
   gzip_bytes (that same file gzip level 9). Keep compressed_bytes for the "compact export" path used
   in Part B (null for dense). Document the fields in results/SCHEMA.md.
5. Memory. The RSS subprocess must not load the 55k training tensor. Save a fixed benchmark input
   (first 256 TEST images, inputs only, no labels needed) to a small file once, and in the subprocess
   record rss_before_model_bytes (after imports + input load), peak_rss_bytes, and
   model_rss_delta_bytes = peak - before.
6. Latency inputs: use the fixed TEST-image benchmark batch above (inputs only), not training images.
7. Linkage: checkpoint metadata must carry the training run_id, and each bench record must carry
   parent_run_id (the training run) and supersedes (list of run_ids of the earlier, now-incorrect
   bench records for that checkpoint). Do not edit or delete old runs.jsonl lines.
8. test_no_update.py replicates the eval loop instead of exercising the real code. Rewrite it to call
   the actual lab3 evaluation function and the actual bench timing function on a loaded checkpoint,
   then assert bit-identical state_dict and that requires_grad flags/grad fields are unchanged (grad
   stays None). Cover all three architectures.

Then: `uv run pytest -q`, re-bench the three seed-0 checkpoints, regenerate results/baselines.md and
figures/learning_curves.png. Report the new table and the test output. Same rules as before: don't
weaken tests, don't change frozen hyperparameters, use uv.
