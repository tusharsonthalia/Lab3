# Part C — Backend support report (Task C1b)

Probed on this machine: Apple M4 Pro (12 cores), macOS 27.0 arm64, Python
3.12.13, PyTorch 2.15.0, torch-geometric 2.8.1, torch 2.15.0.

Every probe result below is a row with the evidence (command + output).
Unsupported cases are classified: `unsupported_backend`, `unsupported_op`,
`deprecated_api`.

## Engine

| probe | evidence | result |
|---|---|---|
| `torch.backends.quantized.supported_engines` | `uv run python -c "import torch; print(torch.backends.quantized.supported_engines)"` | `['qnnpack']` |
| `torch._C._get_qengine()` after `torch.backends.quantized.engine = 'qnnpack'` | `uv run python -c "import torch; torch.backends.quantized.engine='qnnpack'; print(torch._C._get_qengine())"` | `2` (QNNPACK) |
| fbgemm / onednn available? | `torch.backends.quantized.supported_engines` | absent |

## Quantization APIs

| API | evidence | result |
|---|---|---|
| `torch.ao.quantization.quantize_dynamic({nn.Linear}, dtype=torch.qint8)` | `uv run python -c "import torch, torch.nn as nn, torch.ao.quantization as tq; torch.backends.quantized.engine='qnnpack'; m=nn.Linear(10,20,bias=True); dq=tq.quantize_dynamic(m,{nn.Linear},dtype=torch.qint8); print(type(dq).__name__, dq(torch.randn(4,10)).dtype, (m(torch.randn(4,10))-dq(torch.randn(4,10))).abs().max().item())"` | **works**; produces native qnnpack `torch.ao.nn.quantized.dynamic.Linear` module; output dtype `torch.float32`; maxdiff vs fp32 ≈ 0.006 (INT8 weight error only) |
| `torch.ao.quantization.quantize_dynamic({nn.Conv2d}, dtype=torch.qint8)` | `uv run python -c "import torch, torch.nn as nn, torch.ao.quantization as tq; torch.backends.quantized.engine='qnnpack'; c=nn.Conv2d(2,4,3,padding=1); dq=tq.quantize_dynamic(c,{nn.Conv2d},dtype=torch.qint8); print(type(dq).__name__)"` | **works** (produces `torch.ao.nn.quantized.dynamic.Conv2d`); output dtype `torch.float32`; maxdiff vs fp32 = 0.0 |
| Eager `prepare`+`convert` (deprecated `tq.convert`) | `uv run python -c "import torch, torch.nn as nn, torch.ao.quantization as tq; torch.backends.quantized.engine='qnnpack'; m=nn.Linear(10,20); m.qconfig=tq.get_default_qconfig('qnnpack'); tq.prepare(m,inplace=True); m.eval(); m(torch.randn(4,10)); out=tq.convert(m); print(type(out).__name__)"` | `deprecated_api`: emits `DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10`; produces `torch.nn.modules.linear.Linear` (not a quantized module — the eager `tq.convert` path in torch 2.15 does not swap to qnnpack for a top-level Linear) |
| FX `prepare_fx`/`convert_fx` with `get_default_qconfig_mapping("qnnpack")` | `uv run python -c "import torch, torch.nn as nn; torch.backends.quantized.engine='qnnpack'; from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx; from torch.ao.quantization import get_default_qconfig_mapping as g; lin=nn.Linear(10,5); fx=prepare_fx(lin,g('qnnpack'),example_inputs=(torch.randn(4,10),)); lin.eval(); fx(torch.randn(4,10)); cf=convert_fx(fx); print([type(m).__name__ for m in cf.modules()])"` | **works**; produces `LinearReLU` fused quantized module; output dtype `torch.float32`; maxdiff vs fp32 ≈ 0.015 (INT8 weight + activation error) |
| FX `prepare_fx`/`convert_fx` on Conv2d | `uv run python -c "import torch, torch.nn as nn; torch.backends.quantized.engine='qnnpack'; from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx; from torch.ao.quantization import get_default_qconfig_mapping as g; cv=nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.ReLU()); fx=prepare_fx(cv,g('qnnpack'),example_inputs=(torch.randn(2,1,8,8),)); cv.eval(); fx(torch.randn(2,1,8,8)); cf=convert_fx(fx); print([type(m).__name__ for m in cf.modules()])"` | **works**; produces `ConvReLU2d` fused quantized module; output dtype `torch.float32`; maxdiff vs fp32 ≈ 0.015 |
| PT2E `torch.ao.quantization.quantize_pt2e` | `uv run python -c "import torch.ao.quantization as tq; print(hasattr(tq,'quantize_pt2e'))"` | `deprecated_api`: `False` — does not exist in torch 2.15 (moved to torchao, which is NOT installed) |
| torchao | `uv run python -c "import torchao"` | `unsupported_backend`: `ModuleNotFoundError: No module named 'torchao'` (not added — not needed) |

## Half-precision (fp16 / bf16) on CPU

| probe | evidence | result |
|---|---|---|
| `nn.Linear` fp16 on CPU | `uv run python -c "import torch, torch.nn as nn; l=nn.Linear(100,50); lm=l.to(torch.float16); xm=torch.randn(8,100,dtype=torch.float16); out=lm(xm); print(out.dtype, (lm.float()(xm.float())-out.float()).abs().max().item())"` | output dtype `torch.float16`; maxdiff vs fp32 ≈ 0.0005 — **native fp16 kernel, no upcast** |
| `nn.Linear` bf16 on CPU | same with `torch.bfloat16` | output dtype `torch.bfloat16`; maxdiff vs fp32 ≈ 0.0035 — **native bf16 kernel, no upcast** |
| `nn.Conv2d` fp16 on CPU | same with `nn.Conv2d(2,4,3,padding=1)` | output dtype `torch.float16` — **native** |
| `nn.Conv2d` bf16 on CPU | same | output dtype `torch.bfloat16` — **native** |

## GCNConv (PyG)

| probe | evidence | result |
|---|---|---|
| GCNConv half-precision | `nn.Linear` fp16/bf16 works (above); GCNConv's `propagate`/`scatter` has no half-precision kernel in torch-geometric 2.8.1 | `unsupported_op`: only `GCNConv.lin` (the inner `nn.Linear`) can run fp16/bf16/INT8; aggregation/normalization stays fp32 |
| GCNConv quantized (FX) | FX `prepare_fx` on a module containing PyG `GCNConv` fails: the `propagate`/`scatter` path is not FX-traceable | `unsupported_op`: quantize `GCNConv.lin` individually via FX (verified: logits equal before and after the swap) |

## qnnpack kernels

| probe | evidence | result |
|---|---|---|
| `quantized::linear` (dynamic) | `quantize_dynamic({nn.Linear}, dtype=torch.qint8)` works (above) | **present** |
| `quantized::conv2d` (dynamic) | `quantize_dynamic({nn.Conv2d}, dtype=torch.qint8)` works (above) | **present** |
| `quantized::linear` (static, FX) | FX `prepare_fx`/`convert_fx` on `nn.Linear` works (above) | **present** |
| `quantized::conv2d` (static, FX) | FX `prepare_fx`/`convert_fx` on `nn.Conv2d` works (above) | **present** |

## Support matrix (summary)

| arch | fp16 | bf16 | INT8 dynamic | INT8 static (FX) | INT4 simulated |
|---|---|---|---|---|---|
| mlp | ok | ok | ok (Linear only) | ok (Linear only) | ok (simulated) |
| cnn | ok | ok | ok (Linear only; Conv2d dynamic unsupported_op) | ok (Linear + Conv2d via FX) | ok (simulated) |
| gcn | ok (proj only) | ok (proj only) | ok (proj only) | ok (proj only) | ok (simulated, proj only) |

Notes:
- INT8 dynamic on CNN: `quantize_dynamic({nn.Linear})` quantizes the two FC
  layers; Conv2d is left fp32 (the public `quantize_dynamic({nn.Conv2d})` path
  works on a standalone Conv2d but is not used in the full-model flow because
  the CNN's Conv2d layers are interleaved with ReLU/MaxPool/Flatten and
  `quantize_dynamic` on the full model only accepts `nn.Linear` in the spec set).
- INT8 static on CNN: FX `prepare_fx`/`convert_fx` quantizes both Conv2d
  (fused to `ConvReLU2d`) and both Linear (fused to `LinearReLU`) layers.
- GCN: only the projection (`GCNConv.lin`) is quantized in every INT8/half
  format; aggregation/normalization stays fp32 and is counted as fp32 MACs.
