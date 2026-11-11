# Failure accounting — Part B (smoke, pilot, confirm)

Every Part B arm/row ever attempted, with status. The run log is append-only: failed rows are never edited or deleted. 'Excluded from analysis' means the row (or its arm) is not used by any summary, figure, or selection rule in this lab.

## 1. B1 / B1b smoke arms (phase=null, pre-pilot)

The B1 smoke matrix (`mlp s1/s2/s3/s4 t0.5` + `dense_ft` controls and the `cnn`/`gcn` s1 t0.5 checks) ran with the pre-B2 code, i.e. without the `phase` field (recorded as `null`). Its ok rows are duplicated in `results/smoke_b1b/runs.jsonl` (frozen copy) for traceability. Per predictions.md B.6, **all smoke rows are debug runs and are excluded from all analysis**; the B.6 amendment added the dense_ft control because these smoke runs showed post-fine-tune val above the dense parent.

| arch | method | target | seed | stage | status | run_id | reason | excluded from analysis? |
|---|---|---|---|---|---|---|---|---|
| mlp | s1 | 0.5 | 0 | post_prune | failed | mlp-dense-20261109T160519110783Z-1c9aca | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_finetune | failed | mlp-dense-20261109T160521441665Z-89f07a | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| mlp | s1 | 0.5 | 0 | failed | failed | mlp-dense-20261109T160521740875Z-ae3ea9 | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_prune | failed | mlp-dense-20261109T160601905413Z-5013ba | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_finetune | failed | mlp-dense-20261109T160604200403Z-a568a6 | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| mlp | s1 | 0.5 | 0 | failed | failed | mlp-dense-20261109T160604466977Z-984e81 | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T160645034917Z-d88756 | — | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T160647320575Z-ab8b31 | — | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T160753505574Z-a59eb0 | — | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T160755794465Z-4d181c | — | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T160826285138Z-75209c | — | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T160828591605Z-920cb7 | — | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T160937524019Z-5f2919 | — | yes (smoke) |
| mlp | s1 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T160939839990Z-d67c06 | — | yes (smoke) |
| mlp | s2 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T161005724178Z-cbceb5 | — | yes (smoke) |
| mlp | s2 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T161007817258Z-a4b845 | — | yes (smoke) |
| mlp | s3 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T161034329533Z-6e3919 | — | yes (smoke) |
| mlp | s3 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T161036600093Z-ed1c4c | — | yes (smoke) |
| mlp | s4 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T161102808322Z-8f3ac7 | — | yes (smoke) |
| mlp | s4 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T161105152582Z-e9c4f5 | — | yes (smoke) |
| mlp | s2 | 0.5 | 0 | post_prune | failed | mlp-dense-20261109T161143430537Z-6e1d87 | bug: NameError: name 'name' is not defined | yes (smoke) |
| mlp | s2 | 0.5 | 0 | post_finetune | failed | mlp-dense-20261109T161145489915Z-5cc21c | bug: NameError: name 'name' is not defined | yes (smoke) |
| mlp | s2 | 0.5 | 0 | failed | failed | mlp-dense-20261109T161145525672Z-19b1f7 | bug: NameError: name 'name' is not defined | yes (smoke) |
| mlp | s2 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T161206732404Z-a52ffb | — | yes (smoke) |
| mlp | s2 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T161208803384Z-6ea8be | — | yes (smoke) |
| mlp | s4 | 0.5 | 0 | post_prune | ok | mlp-dense-20261109T161746722796Z-9ba3ca | — | yes (smoke) |
| mlp | s4 | 0.5 | 0 | post_finetune | ok | mlp-dense-20261109T161802348905Z-8fff22 | — | yes (smoke) |

Smoke failures and their classified reasons (bug fixes are recorded in the B1/B1b history; all three were fixed before the pilot and every smoke row above is superseded by the pilot runs):

- `mlp-dense-20261109T160519110783Z-1c9aca` (post_prune): bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr'
- `mlp-dense-20261109T160521441665Z-89f07a` (post_finetune): bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr'
- `mlp-dense-20261109T160521740875Z-ae3ea9` (failed): bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr'
- `mlp-dense-20261109T160601905413Z-5013ba` (post_prune): bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr'
- `mlp-dense-20261109T160604200403Z-a568a6` (post_finetune): bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr'
- `mlp-dense-20261109T160604466977Z-984e81` (failed): bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr'
- `mlp-dense-20261109T161143430537Z-6e1d87` (post_prune): bug: NameError: name 'name' is not defined
- `mlp-dense-20261109T161145489915Z-5cc21c` (post_finetune): bug: NameError: name 'name' is not defined
- `mlp-dense-20261109T161145525672Z-19b1f7` (failed): bug: NameError: name 'name' is not defined

## 2. Pilot matrix (phase=pilot, seed 0): 48 cells + 3 dense_ft

The pilot is a full 3 × 4 × 4 matrix; **every one of the 48 cells has an ok phase=pilot post_finetune row** (see verification below), so no cell is missing, failed, or 'not applicable'. The three B.6 dense_ft controls are likewise ok and reported alongside (not counted among the 48).

Verification of the 48-cell matrix:

| arch | method | t0.25 | t0.5 | t0.75 | t0.9 |
|---|---|---|---|---|---|
| mlp | s1 | ok | ok | ok | ok |
| mlp | s2 | ok | ok | ok | ok |
| mlp | s3 | ok | ok | ok | ok |
| mlp | s4 | ok | ok | ok | ok |
| cnn | s1 | ok | ok | ok | ok |
| cnn | s2 | ok | ok | ok | ok |
| cnn | s3 | ok | ok | ok | ok |
| cnn | s4 | ok | ok | ok | ok |
| gcn | s1 | ok | ok | ok | ok |
| gcn | s2 | ok | ok | ok | ok |
| gcn | s3 | ok | ok | ok | ok |
| gcn | s4 | ok | ok | ok | ok |

Cells marked not applicable: **0** of 48. As required, none — every cell is filled by a real ok run (predictions.md B.2: no cell is filled by copying another run).

No failed phase=pilot rows in the log.

## 3. Confirmatory reruns (phase=confirm, seeds 1, 2, 3)

Expected arms: 12 selected points × 3 seeds + 3 dense_ft × 3 seeds = 45. Ok arms with a post_finetune row: 45. Failed phase=confirm rows: 0.


Confirmatory arms are NOT excluded from analysis: they are the confirmatory evidence (results/confirm_summary.csv, figures) for the B.3-selected points. The dense_ft seed 1–3 arms are reference/interpretation controls (B.6/B.7), not part of the 12 selected points.

## 4. Every failed record with classified reason

`reason` is the classified failure (prefix `unsupported_backend`, `numerical`, `bug`, or `not_applicable`, followed by the exception message). Part A failures (pre-Part-B, listed for completeness) are also excluded from Part B analysis.

| part | phase | arch | method | target | seed | stage | run_id | reason (classified) | excluded from analysis? |
|---|---|---|---|---|---|---|---|---|---|
| B | null | mlp | s1 | 0.5 | 0 | post_prune | mlp-dense-20261109T160519110783Z-1c9aca | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| B | null | mlp | s1 | 0.5 | 0 | post_finetune | mlp-dense-20261109T160521441665Z-89f07a | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| B | null | mlp | s1 | 0.5 | 0 | failed | mlp-dense-20261109T160521740875Z-ae3ea9 | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| B | null | mlp | s1 | 0.5 | 0 | post_prune | mlp-dense-20261109T160601905413Z-5013ba | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| B | null | mlp | s1 | 0.5 | 0 | post_finetune | mlp-dense-20261109T160604200403Z-a568a6 | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| B | null | mlp | s1 | 0.5 | 0 | failed | mlp-dense-20261109T160604466977Z-984e81 | bug: AttributeError: 'Tensor' object has no attribute 'csr_indptr' | yes (smoke) |
| B | null | mlp | s2 | 0.5 | 0 | post_prune | mlp-dense-20261109T161143430537Z-6e1d87 | bug: NameError: name 'name' is not defined | yes (smoke) |
| B | null | mlp | s2 | 0.5 | 0 | post_finetune | mlp-dense-20261109T161145489915Z-5cc21c | bug: NameError: name 'name' is not defined | yes (smoke) |
| B | null | mlp | s2 | 0.5 | 0 | failed | mlp-dense-20261109T161145525672Z-19b1f7 | bug: NameError: name 'name' is not defined | yes (smoke) |
| A | null | gcn | dense | 0.0 | 0 | bench/eval/train | gcn-dense-20261109T141830610386Z-e1c5c3 | UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL torch.torch_version.TorchVersion was not an allowed global by default. Please use `torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])` or the `torch.serialization.safe_globals([torch.torch_version.TorchVersion])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html. | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | mlp | dense | 0.0 | 0 | bench/eval/train | mlp-dense-20261109T143026842230Z-5e3485 | RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (220x200704 and 784x256) | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | cnn | dense | 0.0 | 0 | bench/eval/train | cnn-dense-20261109T143028707047Z-79bf4a | RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 1, 1, 28, 28] | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | mlp | dense | 0.0 | 0 | bench/eval/train | mlp-dense-20261109T143047065147Z-a85827 | RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (256x200704 and 784x256) | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | cnn | dense | 0.0 | 0 | bench/eval/train | cnn-dense-20261109T143048935411Z-8bdf50 | RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 1, 1, 28, 28] | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | mlp | dense | 0.0 | 0 | bench/eval/train | mlp-dense-20261109T143059128310Z-75492b | RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (256x200704 and 784x256) | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | mlp | dense | 0.0 | 0 | bench/eval/train | mlp-dense-20261109T144827741481Z-2e1909 | CalledProcessError: Command '['/Users/tusharsonthalia/cmsc35200/lab3/.venv/bin/python', '-c', '\nimport psutil, sys\nimport torch\nfrom lab3.util import set_threads\nfrom lab3.bench import load_bench_input\nfrom lab3.evaln import load_checkpoint\nset_threads()\nx, ei = load_bench_input("mlp")\nproc = psutil.Process()\nbefore = proc.memory_info().rss\nm, _ = load_checkpoint("mlp", 0)\nm.eval()\npeak = before\nwith torch.inference_mode():\n    for _ in range(5):\n        if "mlp" == "gcn":\n            m(x, ei)\n        else:\n            m(x[:256])\n        peak = max(peak, proc.memory_info().rss)\nprint(json.dumps({"before": int(before), "peak": int(peak)}))\n']' returned non-zero exit status 1. | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | cnn | dense | 0.0 | 0 | bench/eval/train | cnn-dense-20261109T144829489104Z-793893 | CalledProcessError: Command '['/Users/tusharsonthalia/cmsc35200/lab3/.venv/bin/python', '-c', '\nimport psutil, sys\nimport torch\nfrom lab3.util import set_threads\nfrom lab3.bench import load_bench_input\nfrom lab3.evaln import load_checkpoint\nset_threads()\nx, ei = load_bench_input("cnn")\nproc = psutil.Process()\nbefore = proc.memory_info().rss\nm, _ = load_checkpoint("cnn", 0)\nm.eval()\npeak = before\nwith torch.inference_mode():\n    for _ in range(5):\n        if "cnn" == "gcn":\n            m(x, ei)\n        else:\n            m(x[:256])\n        peak = max(peak, proc.memory_info().rss)\nprint(json.dumps({"before": int(before), "peak": int(peak)}))\n']' returned non-zero exit status 1. | yes (Part A pre-B2; superseded / not part of Part B) |
| A | null | gcn | dense | 0.0 | 0 | bench/eval/train | gcn-dense-20261109T144832141680Z-d33cd2 | CalledProcessError: Command '['/Users/tusharsonthalia/cmsc35200/lab3/.venv/bin/python', '-c', '\nimport psutil, sys\nimport torch\nfrom lab3.util import set_threads\nfrom lab3.bench import load_bench_input\nfrom lab3.evaln import load_checkpoint\nset_threads()\nx, ei = load_bench_input("gcn")\nproc = psutil.Process()\nbefore = proc.memory_info().rss\nm, _ = load_checkpoint("gcn", 0)\nm.eval()\npeak = before\nwith torch.inference_mode():\n    for _ in range(5):\n        if "gcn" == "gcn":\n            m(x, ei)\n        else:\n            m(x[:256])\n        peak = max(peak, proc.memory_info().rss)\nprint(json.dumps({"before": int(before), "peak": int(peak)}))\n']' returned non-zero exit status 1. | yes (Part A pre-B2; superseded / not part of Part B) |

## 5. Records excluded from analysis (summary)

- **All smoke rows** (phase=null, Part B): debug runs per B.6 — excluded.
- **All failed rows** (Part A and Part B, any phase): failed rows carry no measured values; the arms they belonged to were re-run and the ok rows supersede them (append-only log: both remain on file).
- **Everything else is included**: 48 pilot arms (48 post_prune + 48 post_finetune ok rows), 3 dense_ft pilot controls, and the confirmatory arms once their ok rows are present.
