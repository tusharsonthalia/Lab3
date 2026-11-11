# Lab 3 — Predictions and Frozen Design Decisions

Each section is dated and was written **before** the runs it governs. Sections are
append-only: later corrections are added as dated amendments, never by editing a frozen
section.

---

## Part A — Dense reference models (frozen 2026-11-09, before any training)

### A.1 Frozen design

**Hardware / software.** Apple M4 Pro (12 CPU cores), 24 GB RAM, macOS. Python 3.12,
PyTorch 2.15.0, torchvision 0.29.2, torch_geometric 2.8.1, NumPy 2.5.4. Package
management with `uv` (`pyproject.toml` + `uv.lock`). **CPU only for every arm**; no MPS.

**Threads.** `torch.set_num_threads(4)` and `torch.set_num_interop_threads(1)` for all
training, evaluation, and benchmarking.

**Data splits (created once, saved to `splits/`, reused by every arm).**
- Fashion-MNIST: official 60k train / 10k test. From the 60k train set, a stratified
  5,000-image validation set (500 per class) is drawn once with split seed 12345; the
  remaining 55,000 images are the training set. The 10k official test set is the test set.
  Saved as index files plus a SHA-256 of the index arrays.
- Cora: the standard Planetoid public split (140 train / 500 val / 1000 test nodes), saved
  as node-index files with a SHA-256. The graph (2,708 nodes, 10,556 directed edges) is
  never modified by any compression arm.
- Inputs: Fashion-MNIST normalized with mean/std computed on the 55k training split only.
  Cora features row-normalized (standard GCN preprocessing).

**Architectures.**
| Model | Definition | Params (computed) |
|---|---|---|
| MLP | 784→256→128→10, ReLU, no dropout | 235,146 |
| CNN | Conv(1→16, 5×5, pad 2)→ReLU→MaxPool2 → Conv(16→32, 5×5)→ReLU→MaxPool2 → FC(800→128)→ReLU → FC(128→10) | 117,066 |
| GCN | GCNConv(1433→16)→ReLU→Dropout(0.5) → GCNConv(16→7) | 23,063 |

No BatchNorm anywhere (keeps INT8 conversion and structured surgery unambiguous).

**Training.**
- MLP and CNN: Adam, lr 1e-3, weight decay 0, batch size 128, cross-entropy, max **30
  epochs**.
- GCN: Adam, lr 0.01, weight decay 5e-4, full-batch, cross-entropy on the 140 train nodes,
  max **300 epochs**.

**Early stopping / checkpoint selection (validation only).** After each epoch, evaluate
validation accuracy. Keep the checkpoint with the highest validation accuracy; ties go to
the lower validation loss, then to the earlier epoch. Stop when validation accuracy has not
improved for **5 epochs** (MLP, CNN) or **50 epochs** (GCN), or at the max epoch count. The
test set is evaluated only on the selected checkpoint and never feeds back into any choice.

**Seeds.** Each run sets Python `random`, NumPy, and PyTorch seeds to the run seed and logs
it.
- Pilot seed: **0**. Produces the canonical dense checkpoint per architecture. Every
  compression arm starts from it, and the full Part B pilot matrix uses it.
- Confirmatory seeds: **1, 2, 3**. Used (a) to train three replica dense baselines per
  architecture, which report seed variance only and are never used as compression parents,
  and (b) in Part B to repeat the selected operating points. There the seed controls
  fine-tuning shuffle order and dropout, and the parent is always the canonical seed-0
  checkpoint.
- The split seed 12345 is separate from these and is never changed.

**Sanity targets (test accuracy).** MLP ≥ 82%, CNN ≥ 87%, GCN ≥ 65%. A miss is diagnosed,
and the cause is recorded. The target is not lowered.

**Latency protocol.** `torch.inference_mode()`, model in `eval()`. 20 warm-up and 200
timed iterations, timed with `time.perf_counter_ns`. Report the median and IQR. Batch sizes
are **1** and **256** for Fashion-MNIST, and one full-graph forward pass for Cora. Peak RSS
is measured in a fresh subprocess per benchmark.

**Stopping rule for agent work.** If a baseline misses its target after one diagnosed fix,
or any required test fails twice after a fix, work stops and the issue is reviewed before
continuing. Tests are never weakened to pass.

### A.2 Predictions

**Parameter count: MLP (235k) > CNN (117k) > GCN (23k).**
*Mechanism:* the MLP's first layer alone (784×256 ≈ 201k) dominates its count. The CNN
shares 5×5 kernels across positions, so its convolutions are tiny (≈13k), and most of its
parameters sit in FC(800→128) (≈103k). The GCN has a 16-wide hidden layer, so nearly all
of its parameters are the 1433×16 input projection.

**Dense checkpoint size: MLP (~0.94 MB) > CNN (~0.47 MB) > GCN (~0.09 MB).**
*Mechanism:* FP32 state dicts are about 4 bytes × params, plus a few KB of pickle/zip
overhead. The overhead is proportionally largest for the GCN, so its ratio to the others
will be slightly above the raw parameter ratio. The Cora graph is data, not checkpoint.

**Training time to the selected checkpoint: CNN > MLP > GCN.**
*Mechanism:* per image, the CNN does about 1.7M MACs (conv2 alone is about 1.28M), against
0.235M for the MLP. That is about 7× more compute over the same 55k images per epoch,
partly offset by better-optimized conv kernels. The GCN trains full-batch on 2,708 nodes.
Each epoch is one sparse aggregation plus a 2708×1433×16 projection, so 300 epochs should
take seconds, against minutes for the image models.

**Sensitivity to compression (most to least): GCN > CNN > MLP.**
*Mechanism:*
- *GCN:* it has a 16-dimensional bottleneck and only 140 labelled nodes for fine-tuning.
  Every remaining weight carries a large share of the representation, and recovery data is
  scarce. Most of its input-projection weights belong to rare bag-of-words features, so
  moderate unstructured sparsity should be tolerated. At high sparsity, and for structured
  removal of hidden features, I expect the sharpest collapse.
- *CNN:* convolution weights are reused at every spatial position, so pruning one conv
  weight perturbs many activations. Global magnitude pruning should mostly remove FC
  weights, which buffers it, but structured channel pruning of a 16/32-channel network
  should hurt.
- *MLP:* it is heavily over-parameterized for Fashion-MNIST. Many first-layer weights
  attach to border pixels with near-constant input, so they carry little signal and should
  prune almost for free up to high sparsity.
- *Precision:* I predict INT8 costs every model < 0.5 pp of accuracy. Simulated INT4 should
  hurt the GCN and CNN more than the MLP, for the same bottleneck and weight-reuse reasons.

**Expected baseline test accuracy:** MLP about 88–89%, CNN about 90–91%, GCN about 80–81%.

---

## Part B — Four forms of sparsity (frozen 2026-11-09, before any pruning run)

### B.1 Common protocol (all four methods)

- **Parent:** every arm starts from the canonical seed-0 dense checkpoint of its
  architecture (`checkpoints/dense/<arch>_seed0.pt`).
- **One-shot pruning,** followed by a fixed fine-tuning budget. There is no iterative
  pruning.
- **Eligible tensors:** the weight matrix/kernel of every Linear, Conv2d, and GCNConv layer,
  including the final classifier. Biases are never pruned, and there are no normalization
  layers.
- **Fine-tuning budget (identical for every method within an architecture):**
  - MLP/CNN: 3 epochs, Adam lr 5e-4, batch 128, frozen 55k training split.
  - GCN: 50 epochs, Adam lr 0.005, weight decay 5e-4, dropout 0.5, 140 training nodes.
  - The fine-tuning checkpoint is the epoch with the best validation accuracy inside that
    budget. Ties go to the later epoch, which is the more-trained model.
- **Masks:** for S1/S3/S4 a binary mask is re-applied after every optimizer step, so pruned
  entries stay exactly zero. Adam state for masked entries is irrelevant because the mask
  is applied to the weights.
- **Measurement stages:** every arm is measured twice: immediately after pruning, before
  any update (`stage=post_prune`), and after fine-tuning (`stage=post_finetune`). Both rows
  carry val/test accuracy, per-layer sparsity, bytes, MACs, latency, and memory.
- **Pilot:** seed 0 for the full 3 × 4 × 4 matrix. In fine-tuning, the seed controls data
  order and dropout.
- **Calibration subset (S3 now, reused in Part C):**
  - Fashion-MNIST: 1,024 images from the **training** split, stratified 102–103 per class,
    drawn once with seed 12345 and saved to `splits/fmnist_calib_idx.npy`. The labels are
    training labels.
  - Cora: the 140 training nodes on the full graph.
  - No validation or test labels are ever used for calibration.

### B.2 Method specifications

**S1 Global unstructured magnitude.** Pool all eligible weights across layers, then zero
the fraction *s* with the smallest |w|. Targets: 25/50/75/90%. Achieved sparsity is
reported globally and per tensor.

**S2 Structured group pruning (L1).** Groups and their L1 norms:
- MLP: hidden units of both hidden layers. The norm is the unit's incoming weight row.
- CNN: output channels of conv1 and conv2 (L1 of each filter) and FC1 hidden units.
  Removing conv2 channels also removes the matching FC1 input columns.
- GCN: the 16 hidden features. The norm is the incoming weight column of layer 1.

Ranking is **layer-wise**: each prunable layer removes the same fraction, keeping
k = max(1, round((1 − s)·n)) groups. The classifier output and the input dimension are
never pruned. The model is **physically rebuilt** as a smaller dense model by copying the
surviving rows/columns/bias entries. The target is the fraction of groups removed, and the
achieved value is reported as both group sparsity and parameter sparsity relative to the
dense model. These differ. For example, at s = 90%, GCN keeps round(1.6) = 2 features, so
87.5% of groups are removed.

**S3 First-order saliency (unstructured, global; this choice is fixed).** The score is
|w ⊙ ∂L/∂w|, where L is the mean cross-entropy over the whole calibration subset (one
gradient computed on the dense parent in eval mode, so no dropout). Pool all eligible
weights across layers and zero the lowest-scoring fraction *s*.

**S4 N:M semi-structured.**
- Blocks are M consecutive weights along each row's **reduction (input) dimension**. Conv
  kernels are flattened to [out, in·k·k]. In each block, keep the N largest |w|.
- If the row length is not divisible by M, the final partial block of length r keeps
  ⌈N·r/M⌉ weights.
- **Patterns, chosen before running:** 3:4 (25%), 2:4 (50%), 1:4 (75%), and 1:10 (90%,
  following the instructor's guidance). Every pilot cell is filled by a real run.
- Partial blocks make achieved sparsity slightly below the nominal value for rows with
  remainders: conv1 rows have 25 weights, and GCN layer 1 rows have 1,433. Achieved
  sparsity is reported.
- No cell is filled by copying another run.

### B.3 Confirmatory point selection rule (validation only; fixed before any test result is seen)

For each architecture × method, take the pilot (seed 0) post-fine-tune rows:

1. **Eligible levels:** post-fine-tune validation accuracy ≥ dense seed-0 validation
   accuracy − τ. τ = 1.0 pp for MLP and CNN. τ = 2.0 pp for GCN, whose 500-node validation
   set gives one node = 0.2 pp and a binomial SE of ~1.8 pp.
2. **Choose** the eligible level with the **highest target sparsity**.
3. **Fallback:** if no level is eligible, choose the level with the highest post-fine-tune
   validation accuracy. Ties go to the lower sparsity.
4. **Rerun** the chosen point with seeds 1, 2, 3, using the same parent and budget.

Test accuracy is logged but never read by this rule.

### B.4 Two runtime paths

- **Path 1:** dense PyTorch modules holding zero-valued weights. Run this for every arm.
- **Path 2: genuinely compact export.**
  - S2: the physically rebuilt dense model.
  - S1/S3: `torch.sparse_csr` weights executed with sparse matmul (sparse × dense), plus
    a CSR file on disk.
  - S4: a packed N:M file (kept values + a small index per kept value) for size.
- If the CPU backend has no semi-structured kernel, S4 runtime is recorded as unsupported
  or dense-fallback. It is never inferred from the zero count.

### B.5 Predictions (per method)

**S1 global magnitude.**
- *(i) Largest loss at 75%:* **GCN**, about −1 pp after fine-tuning. It has the smallest
  network and the least labelled data for recovery. Before fine-tuning I expect MLP ≈ −1,
  CNN ≈ −2, and GCN ≈ −3 pp.
- *(ii) Recovery:* fine-tuning recovers almost all loss up to 75%. At 90% it recovers most,
  leaving MLP/CNN about 1–2 pp and GCN about 3–5 pp below dense.
- *(iii) Size and runtime:*
  - The dense-path file size and latency will **not** change, because zeros are stored and
    multiplied as FP32. Gzip will shrink the file roughly with sparsity.
  - CSR export becomes smaller than dense only above roughly 60–67% sparsity, since each
    nonzero costs 4 B of value plus an 8 B int64 column index.
  - Sparse matmul will be **slower** than dense at every level for these small layers,
    including 90%.

**S2 structured.**
- *(i) Largest loss at 75%:* **CNN before fine-tuning.** It drops to 4 and 8 channels and
  each removed channel loses a whole feature map, so I expect a drop of tens of pp.
  **GCN after fine-tuning:** 4 hidden features for 7 classes is a bottleneck narrower than
  the class count, and I expect −5 to −10 pp.
- *(ii) Recovery:* fine-tuning recovers most of the loss for MLP/CNN, leaving them 1–3 pp
  below dense at 75%. It recovers less for GCN.
- *(iii) Size and runtime:* the only method with **real** dense-path reductions.
  - File size shrinks roughly with parameters.
  - At batch 256, MLP/CNN latency falls roughly with MACs.
  - At batch 1, latency is dominated by framework overhead (~10 µs), so the speedup is
    small.
  - GCN latency falls modestly because the X·W projection shrinks, while aggregation and
    overhead remain.

**S3 saliency (unstructured).**
- *(i) Largest loss at 75%:* **GCN**. Its post-prune accuracy should be slightly better than
  S1 at 50–75% because the score accounts for the gradient. At 90%, a gradient taken at a
  near-converged dense point is noisy, so I expect it to be no better than S1.
- *(ii) Recovery:* similar to S1.
- *(iii) Size and runtime:* same as S1, since it produces the same unstructured format.

**S4 N:M.**
- *(i) Largest loss at 75% (1:4):* **CNN** before fine-tuning. The local per-block
  constraint is harsher than global ranking, and conv1 has only 25 weights per filter.
  After fine-tuning, the GCN loses the most.
- *(ii) Recovery:* 3:4 and 2:4 recover to within about 0.5 pp. 1:4 and 1:10 lose more than
  S1 at the same sparsity, because the pattern can't concentrate surviving weights in the
  important rows.
- *(iii) Size and runtime:*
  - I predict **no runtime gain** on this laptop, because PyTorch has no semi-structured
    sparse CPU kernel on ARM. Expect a dense fallback.
  - The packed N:M file will be smaller than dense at 2:4 and sparser levels.
  - Dense-path file size is unchanged.

**Cross-method expectation:** for deployment on this laptop, only S2 will improve both size
and measured latency. S1/S3/S4 will mainly show that more zeros ≠ faster.

### B.6 Amendment (2026-11-09, after the B1 smoke runs, before any pilot-matrix run)

The B1 smoke runs showed post-fine-tune validation accuracy *above* the dense parent. For
example, MLP S1 at 50% reached val 0.9044 against the dense parent's 0.8986. The extra 3
epochs therefore confound "pruning + fine-tuning". **Added control:** one *dense
fine-tuned* arm per architecture (`method=dense_ft`, sparsity 0) that receives exactly the
B.1 fine-tuning budget from the same parent. It is reported alongside the 48 pilot arms and
is not counted among them.

The B.3 selection rule is **unchanged**. It still references the dense seed-0 parent's
validation accuracy. The dense_ft arm is used only to interpret how much of any post-FT
gain comes from the extra epochs.

This amendment changes no method definition, budget, level, or rule. The smoke rows are
debug runs and are excluded from all analysis.

### B.7 Confirmatory points (applied 2026-11-09 after the pilot, before any confirmatory run)

The frozen B.3 rule was applied mechanically by `scripts/select_confirm.py`, which reads
only `val_acc`. The output is in `results/confirm_selection.json`.

**Implementation note.** Eligibility is compared as **integer counts of correct
validation examples**, not as float accuracies. `val_acc` is float32, so GCN S2 at 50%
(390/500 correct = 0.78) is stored as 0.77999997, and a float `>=` would wrongly reject it
against the 0.78 threshold. Counting correct examples implements the rule as written.
Only GCN S2 is affected: float comparison gives 0.25, while the rule as written gives
0.50.

| Arch | S1 | S2 (groups) | S3 | S4 |
|---|---|---|---|---|
| MLP | 0.90 | 0.50 | 0.90 | 0.75 (1:4) |
| CNN | 0.75 | 0.25 | 0.75 | 0.50 (2:4) |
| GCN | 0.90 | 0.50 | 0.90 | 0.25 (3:4) |

Each point is rerun with seeds 1, 2, 3 from the canonical seed-0 parent under the same
budget. The dense_ft control is also rerun with seeds 1–3 for interpretation.

**Transparency.** The agent's pilot report quoted a few test accuracies when it flagged
anomalies. The rule was frozen before the pilot and reads no test field, so those numbers
did not and could not change the selection.

**Pilot observations recorded before confirmation (interpretation deferred):**
- Structured pruning and N:M collapse on the GCN above 50%: S2 val 0.434 at 75%, and S4
  0.484 at 1:4. Unstructured S1/S3 hold about 0.79 even at 90%. This contradicts my B.5
  S1 prediction that GCN would be clearly the most damaged at 90%.

---

## Part C — Reduced precision (frozen 2026-11-09, before any Part C run)

### C.1 Frozen design

**Parents.** The standalone precision study starts from the canonical seed-0 dense FP32
checkpoint of each architecture. Conversion is post-training only; there is no QAT.

**Formats.** Each one is run only where the laptop backend supports it. An unsupported
format gets a classified failed/unsupported row, never a missing cell.
1. **FP16 and BF16.** Weights and inputs are cast. I will report whether each op actually
   executes in that dtype on CPU, and any upcasting, by inspecting output dtypes per layer
   and checking kernel support.
2. **Dynamic INT8.** INT8 weights with per-batch activation quantization, applied to
   Linear-type layers, on whichever quantized engine is available (`qnnpack` was the only
   engine at setup). GCN projections are included if the backend can quantize them,
   otherwise they are classified as unsupported.
3. **Static INT8 (calibrated).** INT8 weights and activations with observers calibrated on
   the subset below. It uses whichever supported flow works, and records which flow was used
   (eager, FX, or PT2E). Ops that cannot be quantized stay FP32, and are counted as such.
4. **Simulated weight-only INT4.**
   - Per-output-channel symmetric quantization to [−8, 7], with an FP32 scale per output
     channel.
   - Two values are packed per uint8 on disk.
   - At runtime the weights are dequantized to FP32 and computed in FP32. This is
     **labelled simulated/non-native**, and the dequantization cost is either inside or
     outside the timed region. That choice is recorded and applied identically everywhere.

**Required accounting per row.**
- Stored, compute, and accumulator precision.
- Actual serialized bytes on disk.
- The **fraction of MACs that truly executed in reduced precision**, computed from the
  per-layer MAC table and which layers actually ran quantized/half kernels.
- Reload equivalence: logits of the reloaded artifact vs the in-memory converted model,
  within an explicit tolerance. The tolerance is 0 for exact formats, and 1e-5 relative for
  FP32 recompute paths.

**Calibration rule (frozen).**
- Fashion-MNIST: the same 1,024-image calibration subset as S3. It comes from the
  **training** split, stratified 102–103 per class, seed 12345, saved as
  `splits/fmnist_calib_idx.npy`. Only inputs are used, no labels.
- Cora: the single full-graph forward pass over all node features with the fixed graph.
  This is inputs only. No labels of any split are used, and it is the only coherent
  calibration for a transductive full-graph model.
- No validation or test labels are used anywhere in calibration.

**Best supported INT8 method per architecture (frozen rule).** Among dynamic and static
INT8 conversions of the dense parent that are supported for that architecture, choose the
one with the **higher validation accuracy**. Ties within one validation example go to the
lower batch-256 / full-graph median latency. If only one method is supported, use it.

**Sparse operating point per architecture (frozen rule, validation-based; applied after
the Part B confirmations, before any Part C run).**
- Consider the four confirmed Part B points (S1–S4) for that architecture.
- *Eligible:* mean post-fine-tune validation accuracy over confirmatory seeds 1–3 must be
  ≥ dense seed-0 validation − τ, counted in correct validation examples, with τ as in B.3.
- Among the eligible points, choose the one with the **smallest measured serialized size of
  its best compact artifact** (CSR for S1/S3, the rebuilt model for S2, the N:M pack for
  S4). This is the smallest deployable model that keeps validation accuracy. Size is a
  property of the artifact, not of test data.
- Ties within 1% of bytes go to the higher mean validation accuracy.
- If none is eligible, choose the point with the highest mean validation accuracy.
- The chosen point's **seed-0 pilot checkpoint** is the sparse FP32 parent for the
  factorial.

**Factorial (12 conditions).**
- For each architecture: dense FP32, sparse FP32, dense INT8, and sparse+INT8 (the best
  INT8 method above, applied to the sparse model).
- For mask-based sparsity I report whether zeros survive quantization (the nonzero count
  after INT8), the INT8 file size, and, where implementable, a sparse-INT8 storage size.
  The latter is labelled as runtime-unsupported if there is no sparse INT8 kernel.

### C.2 Predictions

**Format that minimizes file size: simulated INT4.** It stores about 0.5 B/weight plus one
FP32 scale per output channel, so files will be about 13–14% of FP32 for MLP/CNN. The GCN
will be slightly worse because its container overhead is proportionally larger. INT8 comes
second at about 26–28% of FP32, and FP16/BF16 at about 50%.

**Format that minimizes latency: none of the reduced formats beats FP32 at batch 1 on any
model.**
- At batch 256 I expect at most a modest INT8 gain on the MLP (dynamic) and possibly the CNN
  (static, if conv runs on qnnpack).
- FP16/BF16 will be *slower* than or equal to FP32 on this CPU, because of weak half-precision
  CPU kernels and casts.
- Simulated INT4 will be the slowest, because it pays dequantization and still computes in
  FP32.
- For the GCN, the full-graph sparse aggregation is not quantized, and the X·W projection
  stays FP32 or gets converted at a cost. I expect no reduced-precision speedup.

**Why the size winner and the latency winner need not be the same.** File size is set only
by the bits stored per weight. Latency is set by whether the backend has a **native kernel**
that computes in that dtype, plus conversion overheads: quantizing and dequantizing
activations, casts, and layout packing. For models this small, fixed dispatch overhead
dominates arithmetic. INT4 has the smallest footprint but no native compute here, so it is
the opposite of fastest.

**Accuracy.**
- FP16/BF16: < 0.1 pp change; BF16 possibly up to 0.3 pp.
- Dynamic INT8: < 0.3 pp.
- Static INT8: < 0.5 pp.
- INT4: MLP −0.5 to −1 pp, CNN −1 to −2 pp, GCN −1 to −3 pp.
- Ranking of sensitivity to precision: GCN ≥ CNN > MLP.

**Composition.**
- *Size:* sizes compose only when the sparse format carries INT8 values. With int32 indices,
  index overhead dominates, so sparse+INT8 storage beats dense INT8 only at high sparsity
  (about ≥ 80%).
- *Latency:* latency will **not** compose. There is no sparse INT8 CPU kernel, so
  sparse+INT8 runs as dense INT8 with zeros, at the same latency as dense INT8.
- *Accuracy:* the loss will be roughly additive, and small.
- *Zeros:* zeros survive symmetric INT8 weight quantization exactly, because 0 maps to 0.

### C.3 Sparse operating points (applied 2026-11-09 after the Part B confirmations, before any Part C run)

The frozen C.1 rule was applied by `scripts/select_partc.py`; the output is in
`results/partc_selection.json`. Eligibility uses mean confirmatory validation counts over
seeds 1–3. The choice among eligible points uses the measured compact bytes of the seed-0
pilot checkpoint. All 12 confirmed points were eligible.

| Arch | Selected | Compact bytes | Mean val (s1–3) | Note |
|---|---|---|---|---|
| MLP | **S1, 90%** | 193,288 | 0.8935 | S3 90% within 1% of bytes (193,416), lower val (0.8889) |
| CNN | **S1, 75%** | 239,235 | 0.9203 | S3 75% within 1% (239,363), lower val (0.9157) |
| GCN | **S3, 90%** | 21,793 | 0.7947 | S1 90% identical bytes, lower val (0.7913) |

**Consequence noted in advance.** All three operating points are *unstructured* masks.
Their measured speed advantage on this laptop is zero (CSR was 20–44× slower than dense in
Part B). So the factorial will test composition on **size** with the INT8 *dense* runtime
carrying the zeros. This follows from the frozen size-based rule; it was not chosen to
produce a particular result.
