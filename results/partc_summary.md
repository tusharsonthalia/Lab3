# Part C (C4) summary

Reduced precision: fp16, bf16, dynamic-INT8 (qnnpack), static-INT8 (qnnpack FX), simulated-INT4, 5 formats x 3 architectures, run on the pinned Python 3.12 environment (torch 2.15.0). This supersedes the C1b run (Python 3.13 env mismatch) and the C2 run (2026-11-09/10); the C2 INT8 rows are superseded by C3 because the C2 integer-weight accounting (`nonzero_after_int8`, `bytes_by_dtype`) was blind to the quantized projection weights (it only matched modules with a working `_weight_bias`, missing the FX-static `_packed_weight_N` / raw-ScriptObject shapes — GCN static int8 reported nonzero_after_int8=23 and bytes_by_dtype={fp32:92}); the C3 INT8 rows are superseded by C4 because the C3 INT8 artifacts for mlp/gcn used a nested-pickle container (`state = {"pickled": <nested torch.save(model) zip bytes>}`), which re-encoded the whole module tree inside the outer zip instead of storing a plain state dict. All 45 C1b, the C2 part="C" rows, and the 12 C3 INT8 rows are marked superseded in results/ANNOTATIONS.md. Fields: artifact_bytes + bytes_by_dtype (per-stored-dtype payload, container overhead = artifact_bytes - payload_bytes; per-entry zip breakdown in results/artifact_zip_listing.md), nonzero_params (integer-representation nonzero count), frac_macs_reduced_precision, latency (batch 1 + batch 256 / full graph, median + IQR), rss, stored/compute/accumulator precision, reload_equivalence, run_id.

C3 also corrects the C2 explanation of the `pip install -e .` failure: the real cause was that `uv venv --seed` writes the site-packages `.pth` files with the macOS `UF_HIDDEN` (0x80000000) file flag, and CPython's `site.py` (3.12) skips hidden `.pth` files, so `lab3.pth` was ignored and `import lab3` failed — NOT the `#`-comment issue C2_DONE.md described. (C2_DONE.md is immutable; the correction lives here and in SCHEMA.md's C3 changelog.)

## Standalone conversions (phase="precision")

| arch | format | val_acc | total_params | nonzero_params | nonzero(dense parent) | frac_macs | stored | compute | accumulator | artifact_bytes | payload_bytes | bytes_by_dtype | reload max_diff | status | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | fp16 | 0.8982 | 235146 | 235146 | 235146 | 1.0 | fp16 | fp16 | fp32 (aten fp16 matmul accumulation) | 473857 | 470292 | {'float16': 470292} | 0.0 | ok | mlp-dense-20261110T003252343274Z-981f86 |
| mlp | bf16 | 0.8988 | 235146 | 235146 | 235146 | 1.0 | bf16 | bf16 | fp32 (aten bf16 matmul accumulation) | 473857 | 470292 | {'bfloat16': 470292} | 0.0 | ok | mlp-dense-20261110T003254614482Z-ba7522 |
| mlp | int8_dynamic | 0.8984 | 235146 | 227255 | 235146 | 1.0 | int8 weights (per-tensor symmetric, qnnpack) | int8 (qnnpack quantized::linear, per-batch act quant) | int32 (qnnpack), fp32 dequant | 242481 | 236328 | {'int8': 234752, 'fp32': 1576} | 0.0 | ok | mlp-dense-20261110T022418193278Z-7df957 |
| mlp | int8_static | 0.8984 | 235146 | 227255 | 235146 | 1.0 | int8 weights + calibrated int8 activations (qnnpack FX) | int8 (qnnpack quantized::linear/conv2d, static act qparams) | int32 (qnnpack), fp32 dequant | 243001 | 236340 | {'int8': 234752, 'fp32': 1580, 'int64': 8} | 0.0 | ok | mlp-dense-20261110T022420586492Z-25d6f7 |
| mlp | int4_simulated | 0.8914 | 235146 | 117376 | 235146 | 0.0 | int4 packed 2/uint8 + fp32 per-channel scales | fp32 (dequantize to fp32; dequant inside timed region) | fp32 | 127141 | 120576 | {'int4_packed_uint8': 117376, 'fp32': 3152, 'int64_metadata': 48} | 0.0 | ok | mlp-dense-20261110T003302398071Z-143754 |
| cnn | fp16 | 0.9182 | 117066 | 117066 | 117066 | 1.0 | fp16 | fp16 | fp32 (aten fp16 matmul accumulation) | 238347 | 234132 | {'float16': 234132} | 0.0 | ok | cnn-dense-20261110T003304515902Z-34979c |
| cnn | bf16 | 0.918 | 117066 | 117066 | 117066 | 1.0 | bf16 | bf16 | fp32 (aten bf16 matmul accumulation) | 238347 | 234132 | {'bfloat16': 234132} | 0.0 | ok | cnn-dense-20261110T003311865366Z-12ace5 |
| cnn | int8_dynamic | 0.9182 | 117066 | 100646 | 117066 | 0.06108597285067873 | int8 weights (per-tensor symmetric, qnnpack) | int8 (qnnpack quantized::linear, per-batch act quant) | int32 (qnnpack), fp32 dequant | 163249 | 157224 | {'int8': 103680, 'fp32': 53544} | 0.0 | ok | cnn-dense-20261110T022434224834Z-9e1d6e |
| cnn | int8_static | 0.9154 | 117066 | 113474 | 117066 | 1.0 | int8 weights + calibrated int8 activations (qnnpack FX) | int8 (qnnpack quantized::linear/conv2d, static act qparams) | int32 (qnnpack), fp32 dequant | 127291 | 117672 | {'int8': 116880, 'fp32': 760, 'int64': 32} | 0.0 | ok | cnn-dense-20261110T022438533056Z-f082d1 |
| cnn | int4_simulated | 0.9006 | 117066 | 58448 | 117066 | 0.0 | int4 packed 2/uint8 + fp32 per-channel scales | fp32 (dequantize to fp32; dequant inside timed region) | fp32 | 69781 | 60112 | {'int4_packed_uint8': 58448, 'fp32': 1488, 'int64_metadata': 176} | 0.0 | ok | cnn-dense-20261110T003329693361Z-002cb7 |
| gcn | fp16 | 0.800000011920929 | 23063 | 23063 | 23063 | 1.0 | fp16 | fp16 | fp32 (aten fp16 matmul accumulation) | 49143 | 46126 | {'float16': 46126} | 0.0 | ok | gcn-dense-20261110T003333895347Z-4f8355 |
| gcn | bf16 | 0.800000011920929 | 23063 | 23063 | 23063 | 1.0 | bf16 | bf16 | fp32 (aten bf16 matmul accumulation) | 49207 | 46126 | {'bfloat16': 46126} | 0.0 | ok | gcn-dense-20261110T003335721577Z-458226 |
| gcn | int8_dynamic | 0.7960000038146973 | 23063 | 18972 | 23063 | 1.0 | int8 weights (per-tensor symmetric, qnnpack) | int8 (qnnpack quantized::linear, per-batch act quant) | int32 (qnnpack), fp32 dequant | 28885 | 23224 | {'int8': 23040, 'fp32': 184} | 0.0 | ok | gcn-dense-20261110T022453024493Z-d73a16 |
| gcn | int8_static | 0.7940000295639038 | 23063 | 18972 | 23063 | 1.0 | int8 weights + calibrated int8 activations (qnnpack FX) | int8 (qnnpack quantized::linear/conv2d, static act qparams) | int32 (qnnpack), fp32 dequant | 30009 | 23272 | {'int8': 23040, 'fp32': 200, 'int64': 32} | 0.0 | ok | gcn-dense-20261110T022454895631Z-312d78 |
| gcn | int4_simulated | 0.7860000133514404 | 23063 | 11528 | 23063 | 0.0 | int4 packed 2/uint8 + fp32 per-channel scales | fp32 (dequantize to fp32; dequant inside timed region) | fp32 | 17077 | 11744 | {'int4_packed_uint8': 11528, 'fp32': 184, 'int64_metadata': 32} | 0.0 | ok | gcn-dense-20261110T003341199763Z-2f7015 |

## Latency (median_ms / IQR_ms) and RSS (delta bytes)

| arch | format | bs1 med | bs1 IQR | bs256/full med | bs256/full IQR | rss_delta |
|---|---|---|---|---|---|---|
| mlp | fp16 | 0.0520415 | 0.004228999999999997 | 0.623729 | 0.01857299999999995 | 737280 |
| mlp | bf16 | 0.0537085 | 0.0032397499999999996 | 0.8513335 | 0.013500000000000068 | 2228224 |
| mlp | int8_dynamic | 0.09635450000000001 | 0.0014160000000000006 | 0.519479 | 0.013781749999999926 | 3637248 |
| mlp | int8_static | 0.099333 | 0.006865249999999989 | 0.404396 | 0.019885499999999945 | 303398912 |
| mlp | int4_simulated | 0.016083 | 0.0005002500000000007 | 0.14335399999999998 | 0.0062500000000000056 | 1884160 |
| cnn | fp16 | 0.327542 | 0.009447499999999998 | 18.246771000000003 | 0.19404150000000087 | 41795584 |
| cnn | bf16 | 0.342354 | 0.013062750000000012 | 19.6513125 | 0.453479999999999 | 42598400 |
| cnn | int8_dynamic | 0.1689795 | 0.03025 | 7.292208499999999 | 0.13926025000000042 | 57294848 |
| cnn | int8_static | 0.182333 | 0.008520500000000014 | 3.3536665 | 0.05177150000000008 | 347570176 |
| cnn | int4_simulated | 0.09702050000000001 | 0.0030937500000000062 | 6.9281875 | 0.14516699999999982 | 56688640 |
| gcn | fp16 | None | None | 1.6842920000000001 | 0.026281499999999847 | 11616256 |
| gcn | bf16 | None | None | 1.8189375 | 0.025520250000000022 | 10174464 |
| gcn | int8_dynamic | None | None | 1.5353539999999999 | 0.07479200000000019 | 10551296 |
| gcn | int8_static | None | None | 1.2744585000000002 | 0.14050000000000007 | 63176704 |
| gcn | int4_simulated | None | None | 1.0673124999999999 | 0.04137449999999987 | 5177344 |

## INT8 selection (C.2 rule, C4 run)

| arch | selected | val_acc | latency_ms | reason | run_id |
|---|---|---|---|---|---|
| mlp | int8_static | 0.8984 | 0.404396 | val count 4492 vs 4492; latency 0.519479 vs 0.404396 ms | mlp-dense-20261110T022420586492Z-25d6f7 |
| cnn | int8_dynamic | 0.9182 | 7.292208499999999 | val count 4591 vs 4577; latency 7.292208499999999 vs 3.3536665 ms | cnn-dense-20261110T022434224834Z-9e1d6e |
| gcn | int8_static | 0.7940000295639038 | 1.2744585000000002 | val count 398 vs 397; latency 1.5353539999999999 vs 1.2744585000000002 ms | gcn-dense-20261110T022454895631Z-312d78 |

## 12-condition factorial (phase="factorial", C4 run)

| arch | condition | val_acc | total_params | nonzero | nonzero_before_int8 | nonzero_after_int8 | frac_macs | artifact_bytes | payload_bytes | container_overhead_bytes | reload max_diff | status | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | dense_fp32 | 0.8986 | 235146 | 235146 | 235146 | 235146 | 0.0 | 944329 | 940584 |  | 0.0 | ok | mlp-dense-20261110T003408250547Z-844c70 |
| mlp | sparse_fp32 | 0.8924 | 235146 | 23869 | 23869 | 23869 | 0.0 | 944341 | 940584 |  | 0.0 | ok | mlp-dense-20261110T003410399942Z-ae0892 |
| mlp | dense_int8 | 0.8984 | 235146 | 235146 | 235146 | 227255 | 1.0 | 242981 | 236340 | 6641 | 0.0 | ok | mlp-dense-20261110T022427244417Z-e2d198 |
| mlp | sparse_int8 | 0.8936 | 235146 | 23869 | 23869 | 23469 | 1.0 | 243001 | 236340 | 6661 | 0.0 | ok | mlp-dense-20261110T022430764410Z-5ad684 |
| cnn | dense_fp32 | 0.9182 | 117066 | 117066 | 117066 | 117066 | 0.0 | 472607 | 468264 |  | 0.0 | ok | cnn-dense-20261110T003417224154Z-185cdf |
| cnn | sparse_fp32 | 0.9194 | 117066 | 29406 | 29406 | 29406 | 0.0 | 472621 | 468264 |  | 0.0 | ok | cnn-dense-20261110T003421385863Z-08af14 |
| cnn | dense_int8 | 0.9182 | 117066 | 117066 | 117066 | 100646 | 0.06108597285067873 | 163213 | 157224 | 5989 | 0.0 | ok | cnn-dense-20261110T022444213993Z-3976fe |
| cnn | sparse_int8 | 0.9192 | 117066 | 29406 | 29406 | 23832 | 0.06108597285067873 | 163231 | 157224 | 6007 | 0.0 | ok | cnn-dense-20261110T022448687117Z-5e3be7 |
| gcn | dense_fp32 | 0.800000011920929 | 23063 | 23063 | 23063 | 23063 | 0.0 | 95411 | 92252 |  | 0.0 | ok | gcn-dense-20261110T003434070429Z-bcec0f |
| gcn | sparse_fp32 | 0.7960000038146973 | 23063 | 2327 | 2327 | 2327 | 0.0 | 95421 | 92252 |  | 0.0 | ok | gcn-dense-20261110T003435777955Z-0ef243 |
| gcn | dense_int8 | 0.7940000295639038 | 23063 | 23063 | 23063 | 18972 | 1.0 | 29989 | 23272 | 6717 | 0.0 | ok | gcn-dense-20261110T022457149013Z-5b95aa |
| gcn | sparse_int8 | 0.7960000038146973 | 23063 | 2327 | 2327 | 2304 | 1.0 | 30009 | 23272 | 6737 | 0.0 | ok | gcn-dense-20261110T022459147223Z-24dba8 |

## Container overhead: why the C3 int8 files were bloated (C4 fix)

`container_overhead_bytes = artifact_bytes - payload_bytes`. Per-entry breakdowns (zipfile.infolist) are in results/artifact_zip_listing.md.

The C3 explanation of the dense-vs-sparse INT8 file-size gap was WRONG: it blamed the gap on larger fp32 bias tensors in the nested inner zip, but the MLP biases total only ~2 KB, not ~99 KB. The real cause was the C3 INT8 artifact FORMAT: for mlp/gcn INT8 (dynamic and static) the saved file was a dict whose `state` was `{"pickled": <bytes of an inner torch.save of the whole live module>}` — a nested container. The inner zip re-encoded the ENTIRE module tree: every quantized weight storage appeared twice (once as the quantized weight, once via the module's fp32 placeholder/structure re-pickled inside the module's own state), plus duplicated module-structure pickles and a full second zip of headers. A plain `torch.save(q.state_dict())` of the same converted model is 241,351 bytes for the dense AND the S1-90% sparse MLP (zeros cost 1 byte in int8, as expected), versus 359,751 (dense) / 260,237 (sparse) for the nested-pickle C3 artifacts.

C4 fix: every INT8 artifact (dynamic and static, all 3 archs, standalone and factorial) is now a single torch.save of the converted model's `state_dict()` — no nested pickled bytes, no fp32 copy of the weights. Reload rebuilds the same quantized structure from the fp32 parent architecture (same quantize_dynamic / FX prepare+convert / GCN per-projection path) and calls `load_state_dict(strict=True)`; reloaded logits match the in-memory converted model exactly (atol 0). The C4 overhead is a few KB (zip structural bytes), identical in shape for all conditions; the dense-vs-sparse int8 gap is now only the scale/zero-point bytes (a few dozen bytes). fp16/bf16/int4 artifacts were already flat (state dict / packed-buffer dict) and were left unchanged.

