Task B1b — review corrections to B1 (fresh session; B1's session ended on an endpoint connection
error right after the smoke runs worked). Read predictions.md Part B including the new B.6 amendment,
results/SCHEMA.md, code/lab3/{sparsity,prune,compact}.py and tests/test_sparsity.py first.
I reviewed the B1 code and smoke records. Fix ALL of the following; uv only; never weaken tests;
do not change any frozen design choice.

Test hygiene
1. Tests currently write real files into checkpoints/pilot/ and results/compacts/ (my own pytest run
   created checkpoints/pilot/*20261109T1742*). Route every artifact path (runs, checkpoints, compacts,
   bench inputs written by tests) through the configurable results root so tests use tmp dirs only.
   Move every file in checkpoints/pilot/ and results/compacts/ that is not referenced by an ok record in
   results/runs.jsonl into _quarantine/test_artifacts/ (do not delete; not part of the submission).

Records / schema
2. run_id must encode architecture, method, target and stage, e.g. mlp-s1-t0.50-post_finetune-<ts>-<id>.
   Currently pruned runs are labelled "mlp-dense-...".
3. est_macs / est_macs_nonzero must be PER IMAGE for fmnist (as in Part A: MLP dense = 234,752), and per
   full graph for Cora. The B1 rows show 60,096,512 = 256x the per-image figure.
4. Top-level dense_bytes, gzip_bytes, raw_tensor_bytes, latency, peak_rss_bytes, rss_before_model_bytes,
   model_rss_delta_bytes must be populated from the dense_zeros path (same meaning as Part A);
   top-level compressed_bytes from the compact path. Keep the nested paths dict too.
5. finetune_epochs = 0 for post_prune rows and = the budget (3 or 50) for post_finetune rows;
   train_epochs = the parent's; wall_time_s filled.
6. S2: achieved_sparsity must be the eligible-weight sparsity relative to the PARENT's eligible weights
   (MLP s=0.5 should be ~0.535, not 0.0). group_sparsity_achieved is buggy (-0.5): it must be
   removed_groups / total_groups per prunable layer (store per layer + overall).
7. S2 dense_zeros path must be the dense PARENT-SHAPED model with the removed groups zeroed (built via
   your reversible scatter mapping), not the rebuilt model. The compact path is the rebuilt model.
   This is the key S2 contrast (same zeros, different representation).

Compact path correctness (current numbers are artifacts of the implementation, not the backend)
8. CSR orientation: store W as CSR with shape (out, in) and compute y = (W_csr @ x^T)^T with
   torch.sparse.mm (sparse x dense), which is the optimized path. x @ W_csr (dense x sparse) is not.
9. CSRConv2d: no Python loop over the batch. Unfold once, reshape cols to (in*k*k, B*L), one sparse mm.
10. CSRGCNConv: only the projection may differ from the dense path. Use the same aggregation as the dense
    GCNConv (call the same propagate/normalization code PyG uses), so the latency contrast isolates the
    sparse projection. Verify the compact GCN logits match the dense-zeros GCN within atol 1e-5.
11. Index dtype: use int32 CSR indices if torch supports them for sparse.mm on CPU (record
    csr_index_dtype); also record csr_int64_bytes for the default int64 layout.
12. N:M packed file: store only the kept values (row-major) plus ONE uint8 in-block position per kept
    value; row/block are implied by the fixed pattern + the frozen partial-block ceil rule. Also record
    nm_bitpacked_bytes_theoretical = values*4 + ceil(nnz*ceil(log2 M)/8) + biases, labelled theoretical.
    Keep the reassembly test (bit-exact).
13. Add a test that every compact model's logits match its dense_zeros model within atol 1e-5 for all
    methods and architectures.

Control arm (B.6)
14. Implement method dense_ft: same parent, same B.1 fine-tuning budget, no pruning; one row
    (post_finetune) per architecture; CLI `prune --method dense_ft`.

Annotation
15. Append to results/ANNOTATIONS.md: every part-B row currently in runs.jsonl is "B1 smoke/debug —
    superseded, excluded from analysis", listing their run_ids. Never edit runs.jsonl lines.

Then: `uv run pytest -q`; smoke runs (real records) MLP S1/S2/S3/S4 at 0.5, plus CNN S1 0.5 and GCN S1 0.5
(to exercise conv and GCN CSR paths), plus dense_ft for all three. STOP and report: test output, a compact
table of those rows (val/test pre/post, achieved sparsity, dense_zeros vs compact bytes and bs1/bs256 or
full-graph latency, runtime_status), and anything you could not do.
