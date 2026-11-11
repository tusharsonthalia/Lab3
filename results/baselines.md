# Lab 3 Part A — Dense baselines (frozen design, predictions.md A.1)

Apple M4 Pro, CPU only, `torch.set_num_threads(4)`, interop threads 1.
Python 3.12 / PyTorch 2.15.0 / torchvision 0.29.2 / torch_geometric 2.8.1 / NumPy 2.5.4.
Split seed 12345 (55k/5k/10k stratified fmnist; Cora 140/500/1000 public split).
Seed-0 rows are the canonical dense checkpoints for the compression arms.

**A2 corrections (supersede the A1 bench rows, which are retained in `runs.jsonl`
untouched and cross-linked via `supersedes` / `parent_run_id`):** the CNN MAC count
(conv2 output is 10×10, not 12×12), the GCN MAC convention (weight applied once per
node, not per edge), and the byte/RSS accounting. All MACs are now computed from the
actual model layer shapes (no hard-coded numbers). See `results/SCHEMA.md` for the
field definitions.

## Sanity targets (test acc, seed 0): all met

| model | target | seed 0 | seed mean (0-3) |
|---|---|---|---|
| MLP | ≥ 82% | **89.29%** | 89.14% |
| CNN | ≥ 87% | **90.87%** | 91.30% |
| GCN | ≥ 65% | **80.80%** | 81.40% |

## Baseline table (seed-0 canonical checkpoints)

| model | val acc (s0) | test acc (s0) | test acc seeds 1-3 | params | dense bytes (on disk) | raw tensor bytes | est. MACs | latency median (IQR) | model RSS Δ |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.8986 | 0.8929 | 0.8935 / 0.8873 / 0.8918 | 235,146 | 943,589 B (0.90 MB) | 940,584 B | 234,752 (per image) | bs1 0.0128 ms (0.0005); bs256 0.1319 ms (0.0046) | 2.38 MB |
| CNN | 0.9182 | 0.9087 | 0.9150 / 0.9134 / 0.9147 | 117,066 | 471,797 B (0.45 MB) | 468,264 B | 1,697,280 (per image) | bs1 0.0930 ms (0.0053); bs256 7.0787 ms (0.1140) | 53.9 MB |
| GCN | 0.8000 | 0.8080 | 0.8070 / 0.8220 / 0.8200 | 23,063 | 94,741 B (0.09 MB) | 92,252 B | 62,697,392 (full graph, dense-X); 1,395,824 sparse-input | full-graph 1.0529 ms (0.0401) | 57.8 MB |

- `dense_bytes` = actual on-disk size of `torch.save(state_dict)` (no metadata),
  via `os.path.getsize`; `raw_tensor_bytes` = `sum(numel × element_size)`; the
  difference is the PyTorch zip/pickle container overhead. `compressed_bytes`
  (Part B "compact export", gzip-9 of the same file) is **null** for the dense baseline.
- MACs computed from the actual layer shapes. Dense baselines have every weight
  nonzero, so `est_macs == est_macs_nonzero` here.
  - MLP: 784·256 + 256·128 + 128·10 = **234,752** per image.
  - CNN: 1·16·25·28·28 + 16·32·25·10·10 + 800·128 + 128·10 = **1,697,280** per image
    (conv2 output is 10×10: 28 → conv1 pad2 → 28 → pool → 14 → conv2 5×5 no pad → 10 → pool → 5).
  - GCN (Cora, N=2708, E=10556): per layer projection `N·in·out` + aggregation
    `(E+N)·out`; weight applied **once per node, not per edge**.
    `est_macs` = 2708·1433·16 + (E+N)·16 + 2708·16·7 + (E+N)·7 = **62,697,392**;
    `est_macs_sparse_input` (layer-1 projection as nnz(X)·16, Cora features ~1.3% dense)
    = **1,395,824**; degree-normalization scales `gcn_norm_macs` = 2·(16+7)·(E+N) = **610,144**
    (reported separately, not in `est_macs`).
- Latency protocol: `torch.inference_mode()`, `eval()`, 20 warm-up / 200 timed iterations,
  `time.perf_counter_ns`, median + IQR. Fixed benchmark batch = first 256 **TEST** images
  (inputs only, saved once to `results/bench/fmnist_bench_input.pt`), not training images;
  batch 1 and 256 for fmnist, one full-graph forward for Cora.
- Memory (RSS) measured in a fresh subprocess that loads the small fixed benchmark
  input + the checkpoint (never the 55k training tensor). `model_rss_delta_bytes` =
  `peak_rss_bytes − rss_before_model_bytes` (RSS after imports+input load, before model).
  (Peak RSS figures in the A1 table included the training-tensor load and are not comparable.)
- Linkage: each bench record carries `parent_run_id` (the training run whose
  `run_id` is stored in the checkpoint) and `supersedes` (the earlier, now-incorrect
  bench run_ids for that checkpoint). Old `runs.jsonl` lines are unedited/undeleted.

## Training (unchanged)

- Training (frozen): MLP/CNN Adam lr 1e-3, wd 0, bs 128, max 30 epochs, patience 5;
  GCN Adam lr 0.01, wd 5e-4, full-batch, max 300 epochs, patience 50.
  Checkpoint = best val acc (ties: lower val loss, then earlier epoch); test evaluated
  only on the selected checkpoint.
- Seed variance (test acc): MLP 88.73-89.35, CNN 90.87-91.50, GCN 80.70-82.20.

## Training stops

| model | best epoch | epochs run | stop reason |
|---|---|---|---|
| MLP | 16 | 21 | patience (no val acc improvement for 5 epochs) |
| CNN | 20 | 25 | patience |
| GCN | 93 | 143 | patience |

## Notes / anomalies

- predictions.md expected test acc ≈ MLP 88-89%, CNN 90-91%, GCN 80-81%: observed
  seed-0 values 89.29 / 90.87 / 80.80 — all inside or above the predicted ranges.
- A1's GCN `est_macs` (306,212,704) counted the weight per edge; the corrected
  dense-X value is 62,697,392 (weight once per node). A1's CNN `est_macs`
  (2,260,480) used a 12×12 conv2 output; corrected to 1,697,280 (10×10).
- Every field is documented in `results/SCHEMA.md`. `results/runs.jsonl` is
  append-only; the corrected seed-0 bench records supersede the A1 ones.
