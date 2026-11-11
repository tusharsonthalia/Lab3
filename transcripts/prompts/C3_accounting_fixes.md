Task C3 — two accounting bugs in the C2 results, then a targeted rerun. Fresh session. Read
results/C2_DONE.md, results/factorial.csv, results/partc_summary.md and code/lab3/precision.py
(int8_nonzero, _int_weight_tensors, bytes_by_dtype, _save_artifact/_artifact_state) first.
Invoke the package as `PYTHONPATH=code uv run python -m lab3 ...` (see note at the end). Never weaken
tests. Keep replies short and act through tool calls.

Review of C2 (verified by me from results/factorial.csv):

1. GCN INT8 accounting is blind to the quantized projections. For gcn dense_int8 and sparse_int8:
   nonzero_after_int8 = 23 (= the GCN's 16+7 BIASES) and bytes_by_dtype = {"fp32": 92} (biases only),
   yet val acc stays 0.794-0.796, so the INT8 projection weights obviously exist. _int_weight_tensors /
   bytes_by_dtype do not find the quantized modules that replaced conv1.lin / conv2.lin (static flow
   quantizes each swapped projection individually). Fix the traversal so the integer weights of those
   modules are counted (int_repr nonzero, 1 B/elem payload, fp32 scales/zero points). Also check the
   standalone gcn int8_dynamic / int8_static rows. C2_DONE.md called the "23 nonzero" a real effect of
   per-tensor quantization; it is an accounting bug. Add a test: for every INT8 artifact of every arch,
   bytes_by_dtype["int8"] >= the number of eligible weights of the quantized layers, and the
   int_repr nonzero count is within 20% of the fp32 nonzero count of those layers.

2. Unexplained container overhead. Same int8 payload, very different file sizes:
   mlp dense_int8 artifact 359,751 B vs payload 236,328 (overhead ~123 KB); mlp sparse_int8 260,237 B
   vs the same payload (overhead ~24 KB); cnn dense_int8 245,895 vs payload 157,224; cnn sparse_int8
   192,077. torch.save zip entries are stored uncompressed, so payload-equal files should be near-equal.
   Diagnose with zipfile.ZipFile(artifact).infolist() (entry names, file_size, compress_size,
   compress_type) and by listing every tensor in the saved object (dtype, shape, numel*element_size).
   Find what extra data the dense INT8 files carry (e.g. duplicated packed params, an fp32 copy of the
   weights, observer state, or compression). If the file contains anything that is not needed to run the
   quantized model, remove it from the artifact and rerun; if the difference is real, explain exactly
   what it is in results/partc_summary.md. Write the per-entry listing for all 12 factorial artifacts to
   results/artifact_zip_listing.md.

After the fixes: rerun ONLY the affected records (all INT8 rows: standalone int8_dynamic/int8_static x 3
archs and the 6 *_int8 factorial conditions) as new records with phase "precision"/"factorial"; mark the
superseded C2 INT8 rows in results/ANNOTATIONS.md ("C2 INT8 rows — superseded by C3: accounting bug /
artifact content"). Recompute results/int8_selection.json only if a val/latency input changed (state
explicitly whether it did). Regenerate factorial.csv, partc_summary.md and the Part C figures from the
latest records. `uv run pytest -q`.

Note: your C2 report says the editable install fails because '#' in the path is read as a .pth comment.
That is not the cause. uv writes the .pth files with the macOS UF_HIDDEN flag, and Python 3.12.13 skips
hidden .pth files. PYTHONPATH=code is the correct workaround; correct this explanation wherever you
wrote it (C2_DONE.md is immutable; correct it in partc_summary.md / SCHEMA.md if repeated there).
When done, write results/C3_DONE.md with your full report (what was wrong, what changed, before/after
numbers, test output), then stop.
