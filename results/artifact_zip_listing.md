# Factorial artifact zip listings (C4)

Per-entry `zipfile.ZipFile(artifact).infolist()` for the 12 latest-ok

3.12 factorial artifacts. `file_size` = stored (uncompressed) bytes;

`compress_size` = bytes on disk; `compress_type` 0 = stored (no

compression). All entries are the outer .pt zip (torch.save of

`{kind, architecture, state}`). Since C4 every `state` is a flat

state dict (int8: the converted model's state_dict with int8

weights + scales + zero points; fp32: the fp32 state dict), so

there is NO inner/nested zip. See C4_DONE.md for the fix of the

C3 nested-pickle artifact container.

## mlp dense_fp32  (mlp-dense-20261110T003408250547Z-844c70)

artifact_bytes = 944329; payload_bytes = 940584; outer structural = 3094; outer stored = 941235

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/data.pkl | 600 | 600 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/.format_version | 1 | 1 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/.storage_alignment | 2 | 2 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/byteorder | 6 | 6 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/data/0 | 802816 | 802816 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/data/1 | 1024 | 1024 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/data/2 | 131072 | 131072 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/data/3 | 512 | 512 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/data/4 | 5120 | 5120 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/data/5 | 40 | 40 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/version | 2 | 2 | 0 |
| OUT mlp-dense_fp32_mlp-dense-20261110T003408250547Z-844c70/.data/serialization_id | 40 | 40 | 0 |

## mlp sparse_fp32  (mlp-dense-20261110T003410399942Z-ae0892)

artifact_bytes = 944341; payload_bytes = 940584; outer structural = 3106; outer stored = 941235

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/data.pkl | 600 | 600 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/.format_version | 1 | 1 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/.storage_alignment | 2 | 2 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/byteorder | 6 | 6 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/data/0 | 802816 | 802816 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/data/1 | 1024 | 1024 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/data/2 | 131072 | 131072 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/data/3 | 512 | 512 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/data/4 | 5120 | 5120 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/data/5 | 40 | 40 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/version | 2 | 2 | 0 |
| OUT mlp-sparse_fp32_mlp-dense-20261110T003410399942Z-ae0892/.data/serialization_id | 40 | 40 | 0 |

## mlp dense_int8  (mlp-dense-20261110T022427244417Z-e2d198)

artifact_bytes = 242981; payload_bytes = 236340; outer structural = 4902; outer stored = 238079

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data.pkl | 1652 | 1652 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/.format_version | 1 | 1 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/.storage_alignment | 2 | 2 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/byteorder | 6 | 6 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/0 | 4 | 4 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/1 | 8 | 8 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/2 | 4 | 4 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/3 | 8 | 8 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/4 | 200704 | 200704 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/5 | 1024 | 1024 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/6 | 4 | 4 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/7 | 8 | 8 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/8 | 32768 | 32768 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/9 | 512 | 512 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/10 | 4 | 4 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/11 | 8 | 8 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/12 | 1280 | 1280 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/data/13 | 40 | 40 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/version | 2 | 2 | 0 |
| OUT mlp-dense_int8_mlp-dense-20261110T022427244417Z-e2d198/.data/serialization_id | 40 | 40 | 0 |

## mlp sparse_int8  (mlp-dense-20261110T022430764410Z-5ad684)

artifact_bytes = 243001; payload_bytes = 236340; outer structural = 4922; outer stored = 238079

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data.pkl | 1652 | 1652 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/.format_version | 1 | 1 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/.storage_alignment | 2 | 2 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/byteorder | 6 | 6 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/0 | 4 | 4 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/1 | 8 | 8 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/2 | 4 | 4 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/3 | 8 | 8 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/4 | 200704 | 200704 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/5 | 1024 | 1024 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/6 | 4 | 4 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/7 | 8 | 8 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/8 | 32768 | 32768 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/9 | 512 | 512 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/10 | 4 | 4 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/11 | 8 | 8 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/12 | 1280 | 1280 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/data/13 | 40 | 40 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/version | 2 | 2 | 0 |
| OUT mlp-sparse_int8_mlp-dense-20261110T022430764410Z-5ad684/.data/serialization_id | 40 | 40 | 0 |

## cnn dense_fp32  (cnn-dense-20261110T003417224154Z-185cdf)

artifact_bytes = 472607; payload_bytes = 468264; outer structural = 3545; outer stored = 469062

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data.pkl | 747 | 747 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/.format_version | 1 | 1 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/.storage_alignment | 2 | 2 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/byteorder | 6 | 6 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/0 | 1600 | 1600 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/1 | 64 | 64 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/2 | 51200 | 51200 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/3 | 128 | 128 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/4 | 409600 | 409600 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/5 | 512 | 512 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/6 | 5120 | 5120 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/data/7 | 40 | 40 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/version | 2 | 2 | 0 |
| OUT cnn-dense_fp32_cnn-dense-20261110T003417224154Z-185cdf/.data/serialization_id | 40 | 40 | 0 |

## cnn sparse_fp32  (cnn-dense-20261110T003421385863Z-08af14)

artifact_bytes = 472621; payload_bytes = 468264; outer structural = 3559; outer stored = 469062

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data.pkl | 747 | 747 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/.format_version | 1 | 1 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/.storage_alignment | 2 | 2 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/byteorder | 6 | 6 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/0 | 1600 | 1600 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/1 | 64 | 64 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/2 | 51200 | 51200 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/3 | 128 | 128 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/4 | 409600 | 409600 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/5 | 512 | 512 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/6 | 5120 | 5120 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/data/7 | 40 | 40 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/version | 2 | 2 | 0 |
| OUT cnn-sparse_fp32_cnn-dense-20261110T003421385863Z-08af14/.data/serialization_id | 40 | 40 | 0 |

## cnn dense_int8  (cnn-dense-20261110T022444213993Z-3976fe)

artifact_bytes = 163213; payload_bytes = 157224; outer structural = 4446; outer stored = 158767

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data.pkl | 1468 | 1468 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/.format_version | 1 | 1 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/.storage_alignment | 2 | 2 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/byteorder | 6 | 6 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/0 | 1600 | 1600 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/1 | 64 | 64 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/2 | 51200 | 51200 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/3 | 128 | 128 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/4 | 4 | 4 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/5 | 8 | 8 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/6 | 102400 | 102400 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/7 | 512 | 512 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/8 | 4 | 4 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/9 | 8 | 8 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/10 | 1280 | 1280 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/data/11 | 40 | 40 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/version | 2 | 2 | 0 |
| OUT cnn-dense_int8_cnn-dense-20261110T022444213993Z-3976fe/.data/serialization_id | 40 | 40 | 0 |

## cnn sparse_int8  (cnn-dense-20261110T022448687117Z-5e3be7)

artifact_bytes = 163231; payload_bytes = 157224; outer structural = 4464; outer stored = 158767

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data.pkl | 1468 | 1468 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/.format_version | 1 | 1 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/.storage_alignment | 2 | 2 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/byteorder | 6 | 6 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/0 | 1600 | 1600 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/1 | 64 | 64 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/2 | 51200 | 51200 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/3 | 128 | 128 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/4 | 4 | 4 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/5 | 8 | 8 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/6 | 102400 | 102400 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/7 | 512 | 512 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/8 | 4 | 4 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/9 | 8 | 8 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/10 | 1280 | 1280 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/data/11 | 40 | 40 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/version | 2 | 2 | 0 |
| OUT cnn-sparse_int8_cnn-dense-20261110T022448687117Z-5e3be7/.data/serialization_id | 40 | 40 | 0 |

## gcn dense_fp32  (gcn-dense-20261110T003434070429Z-bcec0f)

artifact_bytes = 95411; payload_bytes = 92252; outer structural = 2647; outer stored = 92764

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/data.pkl | 461 | 461 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/.format_version | 1 | 1 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/.storage_alignment | 2 | 2 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/byteorder | 6 | 6 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/data/0 | 64 | 64 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/data/1 | 91712 | 91712 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/data/2 | 28 | 28 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/data/3 | 448 | 448 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/version | 2 | 2 | 0 |
| OUT gcn-dense_fp32_gcn-dense-20261110T003434070429Z-bcec0f/.data/serialization_id | 40 | 40 | 0 |

## gcn sparse_fp32  (gcn-dense-20261110T003435777955Z-0ef243)

artifact_bytes = 95421; payload_bytes = 92252; outer structural = 2657; outer stored = 92764

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/data.pkl | 461 | 461 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/.format_version | 1 | 1 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/.storage_alignment | 2 | 2 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/byteorder | 6 | 6 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/data/0 | 64 | 64 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/data/1 | 91712 | 91712 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/data/2 | 28 | 28 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/data/3 | 448 | 448 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/version | 2 | 2 | 0 |
| OUT gcn-sparse_fp32_gcn-dense-20261110T003435777955Z-0ef243/.data/serialization_id | 40 | 40 | 0 |

## gcn dense_int8  (gcn-dense-20261110T022457149013Z-5b95aa)

artifact_bytes = 29989; payload_bytes = 23272; outer structural = 4947; outer stored = 25042

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data.pkl | 1719 | 1719 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/.format_version | 1 | 1 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/.storage_alignment | 2 | 2 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/byteorder | 6 | 6 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/0 | 64 | 64 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/1 | 4 | 4 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/2 | 8 | 8 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/3 | 4 | 4 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/4 | 8 | 8 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/5 | 22928 | 22928 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/6 | 64 | 64 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/7 | 28 | 28 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/8 | 4 | 4 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/9 | 8 | 8 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/10 | 4 | 4 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/11 | 8 | 8 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/12 | 112 | 112 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/data/13 | 28 | 28 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/version | 2 | 2 | 0 |
| OUT gcn-dense_int8_gcn-dense-20261110T022457149013Z-5b95aa/.data/serialization_id | 40 | 40 | 0 |

## gcn sparse_int8  (gcn-dense-20261110T022459147223Z-24dba8)

artifact_bytes = 30009; payload_bytes = 23272; outer structural = 4967; outer stored = 25042

| entry | file_size | compress_size | compress_type |
|---|---|---|---|
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data.pkl | 1719 | 1719 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/.format_version | 1 | 1 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/.storage_alignment | 2 | 2 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/byteorder | 6 | 6 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/0 | 64 | 64 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/1 | 4 | 4 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/2 | 8 | 8 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/3 | 4 | 4 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/4 | 8 | 8 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/5 | 22928 | 22928 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/6 | 64 | 64 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/7 | 28 | 28 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/8 | 4 | 4 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/9 | 8 | 8 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/10 | 4 | 4 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/11 | 8 | 8 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/12 | 112 | 112 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/data/13 | 28 | 28 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/version | 2 | 2 | 0 |
| OUT gcn-sparse_int8_gcn-dense-20261110T022459147223Z-24dba8/.data/serialization_id | 40 | 40 | 0 |

