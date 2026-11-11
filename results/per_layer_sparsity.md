# Per-layer sparsity — Part B pilot, post_finetune (seed 0)

One table per pilot arm (phase=pilot, stage=post_finetune, status=ok), grouped by architecture and method. Columns: layer (state key of the eligible weight tensor), shape (the layer's actual weight shape in the post-fine-tune checkpoint — for S2 this is the physically REBUILT model's smaller shape, recorded on the arm row), numel, nonzero, sparsity = 1 - nonzero/numel. Achieved sparsity (eligible-weight fraction zero, relative to the dense parent) is shown per arm. Source: results/runs.jsonl (phase=pilot rows, `per_layer` field); dense parent shapes below each method section are shown for reference.

## MLP

Dense parent shapes: net.0 256x784, net.2 128x256, net.4 10x128

### S1

#### target 0.25 — achieved eligible sparsity 0.2500, arm mlp-s1-t0.25-seed0-20261109T194311826265Z-f2ade5

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 148948 | 0.2579 |
| net.2 | 128x256 | 32768 | 25991 | 0.2068 |
| net.4 | 10x128 | 1280 | 1125 | 0.1211 |
#### target 0.50 — achieved eligible sparsity 0.5000, arm mlp-s1-t0.5-seed0-20261109T194320824874Z-a4cdcf

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 97679 | 0.5133 |
| net.2 | 128x256 | 32768 | 18749 | 0.4278 |
| net.4 | 10x128 | 1280 | 948 | 0.2594 |
#### target 0.75 — achieved eligible sparsity 0.7500, arm mlp-s1-t0.75-seed0-20261109T194328954979Z-6c1e19

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 47870 | 0.7615 |
| net.2 | 128x256 | 32768 | 10199 | 0.6888 |
| net.4 | 10x128 | 1280 | 619 | 0.5164 |
#### target 0.90 — achieved eligible sparsity 0.9000, arm mlp-s1-t0.9-seed0-20261109T194336132291Z-0c4979

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 18596 | 0.9073 |
| net.2 | 128x256 | 32768 | 4521 | 0.8620 |
| net.4 | 10x128 | 1280 | 358 | 0.7203 |
### S2

#### target 0.25 (group sparsity achieved 0.2500) — achieved eligible sparsity 0.2762, arm mlp-s2-t0.25-seed0-20261109T194342732028Z-552f0c

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 192x784 | 150528 | 150528 | 0.0000 |
| net.2 | 96x192 | 18432 | 18432 | 0.0000 |
| net.4 | 10x96 | 960 | 960 | 0.0000 |


#### target 0.50 (group sparsity achieved 0.5000) — achieved eligible sparsity 0.5349, arm mlp-s2-t0.5-seed0-20261109T194348267526Z-519783

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 128x784 | 100352 | 100352 | 0.0000 |
| net.2 | 64x128 | 8192 | 8192 | 0.0000 |
| net.4 | 10x64 | 640 | 640 | 0.0000 |


#### target 0.75 (group sparsity achieved 0.7500) — achieved eligible sparsity 0.7762, arm mlp-s2-t0.75-seed0-20261109T194353758914Z-eb6230

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 64x784 | 50176 | 50176 | 0.0000 |
| net.2 | 32x64 | 2048 | 2048 | 0.0000 |
| net.4 | 10x32 | 320 | 320 | 0.0000 |


#### target 0.90 (group sparsity achieved 0.8984) — achieved eligible sparsity 0.9112, arm mlp-s2-t0.9-seed0-20261109T194359149726Z-9e1a38

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 26x784 | 20384 | 20384 | 0.0000 |
| net.2 | 13x26 | 338 | 338 | 0.0000 |
| net.4 | 10x13 | 130 | 130 | 0.0000 |


### S3

#### target 0.25 — achieved eligible sparsity 0.2500, arm mlp-s3-t0.25-seed0-20261109T194404349098Z-857178

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 151507 | 0.2451 |
| net.2 | 128x256 | 32768 | 23464 | 0.2839 |
| net.4 | 10x128 | 1280 | 1093 | 0.1461 |
#### target 0.50 — achieved eligible sparsity 0.5000, arm mlp-s3-t0.5-seed0-20261109T194414106711Z-e4131a

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 99061 | 0.5064 |
| net.2 | 128x256 | 32768 | 17313 | 0.4716 |
| net.4 | 10x128 | 1280 | 1002 | 0.2172 |
#### target 0.75 — achieved eligible sparsity 0.7500, arm mlp-s3-t0.75-seed0-20261109T194422899794Z-c58896

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 47296 | 0.7643 |
| net.2 | 128x256 | 32768 | 10499 | 0.6796 |
| net.4 | 10x128 | 1280 | 893 | 0.3023 |
#### target 0.90 — achieved eligible sparsity 0.9000, arm mlp-s3-t0.9-seed0-20261109T194430732342Z-aa3d55

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 17333 | 0.9136 |
| net.2 | 128x256 | 32768 | 5373 | 0.8360 |
| net.4 | 10x128 | 1280 | 769 | 0.3992 |
### S4

#### target 0.25 (pattern 3:4) — achieved eligible sparsity 0.2500, arm mlp-s4-t0.25-seed0-20261109T194437915425Z-863482

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 150528 | 0.2500 |
| net.2 | 128x256 | 32768 | 24576 | 0.2500 |
| net.4 | 10x128 | 1280 | 960 | 0.2500 |
#### target 0.50 (pattern 2:4) — achieved eligible sparsity 0.5000, arm mlp-s4-t0.5-seed0-20261109T194444973838Z-68db81

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 100352 | 0.5000 |
| net.2 | 128x256 | 32768 | 16384 | 0.5000 |
| net.4 | 10x128 | 1280 | 640 | 0.5000 |
#### target 0.75 (pattern 1:4) — achieved eligible sparsity 0.7500, arm mlp-s4-t0.75-seed0-20261109T194452178959Z-03d5f9

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 50176 | 0.7500 |
| net.2 | 128x256 | 32768 | 8192 | 0.7500 |
| net.4 | 10x128 | 1280 | 320 | 0.7500 |
#### target 0.90 (pattern 1:10) — achieved eligible sparsity 0.8991, arm mlp-s4-t0.9-seed0-20261109T194459199805Z-c6da2b

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| net.0 | 256x784 | 200704 | 20224 | 0.8992 |
| net.2 | 128x256 | 32768 | 3328 | 0.8984 |
| net.4 | 10x128 | 1280 | 130 | 0.8984 |
## CNN

Dense parent shapes: conv1 16x1x5x5, conv2 32x16x5x5, fc1 128x800, fc2 10x128

### S1

#### target 0.25 — achieved eligible sparsity 0.2500, arm cnn-s1-t0.25-seed0-20261109T194511782324Z-9e8a4d

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 371 | 0.0725 |
| conv2 | 32x16x5x5 | 12800 | 11003 | 0.1404 |
| fc1 | 128x800 | 102400 | 75154 | 0.2661 |
| fc2 | 10x128 | 1280 | 1132 | 0.1156 |
#### target 0.50 — achieved eligible sparsity 0.5000, arm cnn-s1-t0.5-seed0-20261109T194552805006Z-b14d14

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 344 | 0.1400 |
| conv2 | 32x16x5x5 | 12800 | 8861 | 0.3077 |
| fc1 | 128x800 | 102400 | 48289 | 0.5284 |
| fc2 | 10x128 | 1280 | 946 | 0.2609 |
#### target 0.75 — achieved eligible sparsity 0.7500, arm cnn-s1-t0.75-seed0-20261109T194630448460Z-a6e08a

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 274 | 0.3150 |
| conv2 | 32x16x5x5 | 12800 | 5114 | 0.6005 |
| fc1 | 128x800 | 102400 | 23279 | 0.7727 |
| fc2 | 10x128 | 1280 | 553 | 0.5680 |
#### target 0.90 — achieved eligible sparsity 0.9000, arm cnn-s1-t0.9-seed0-20261109T194707204396Z-92fdac

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 207 | 0.4825 |
| conv2 | 32x16x5x5 | 12800 | 2364 | 0.8153 |
| fc1 | 128x800 | 102400 | 8797 | 0.9141 |
| fc2 | 10x128 | 1280 | 320 | 0.7500 |
### S2

#### target 0.25 (group sparsity achieved 0.2500) — achieved eligible sparsity 0.4348, arm cnn-s2-t0.25-seed0-20261109T194739447276Z-c2ec2d

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 12x1x5x5 | 300 | 300 | 0.0000 |
| conv2 | 24x12x5x5 | 7200 | 7200 | 0.0000 |
| fc1 | 96x600 | 57600 | 57600 | 0.0000 |
| fc2 | 10x96 | 960 | 960 | 0.0000 |


#### target 0.50 (group sparsity achieved 0.5000) — achieved eligible sparsity 0.7464, arm cnn-s2-t0.5-seed0-20261109T194807018760Z-3ba00b

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 8x1x5x5 | 200 | 200 | 0.0000 |
| conv2 | 16x8x5x5 | 3200 | 3200 | 0.0000 |
| fc1 | 64x400 | 25600 | 25600 | 0.0000 |
| fc2 | 10x64 | 640 | 640 | 0.0000 |


#### target 0.75 (group sparsity achieved 0.7500) — achieved eligible sparsity 0.9348, arm cnn-s2-t0.75-seed0-20261109T194831289551Z-bcbe4a

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 4x1x5x5 | 100 | 100 | 0.0000 |
| conv2 | 8x4x5x5 | 800 | 800 | 0.0000 |
| fc1 | 32x200 | 6400 | 6400 | 0.0000 |
| fc2 | 10x32 | 320 | 320 | 0.0000 |


#### target 0.90 (group sparsity achieved 0.8977) — achieved eligible sparsity 0.9888, arm cnn-s2-t0.9-seed0-20261109T194850175788Z-fab4bc

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 2x1x5x5 | 50 | 50 | 0.0000 |
| conv2 | 3x2x5x5 | 150 | 150 | 0.0000 |
| fc1 | 13x75 | 975 | 975 | 0.0000 |
| fc2 | 10x13 | 130 | 130 | 0.0000 |


### S3

#### target 0.25 — achieved eligible sparsity 0.2500, arm cnn-s3-t0.25-seed0-20261109T194909461552Z-351acd

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 400 | 0.0000 |
| conv2 | 32x16x5x5 | 12800 | 12400 | 0.0312 |
| fc1 | 128x800 | 102400 | 73580 | 0.2814 |
| fc2 | 10x128 | 1280 | 1280 | 0.0000 |
#### target 0.50 — achieved eligible sparsity 0.5000, arm cnn-s3-t0.5-seed0-20261109T194951243363Z-ab9d87

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 392 | 0.0200 |
| conv2 | 32x16x5x5 | 12800 | 11283 | 0.1185 |
| fc1 | 128x800 | 102400 | 46009 | 0.5507 |
| fc2 | 10x128 | 1280 | 756 | 0.4094 |
#### target 0.75 — achieved eligible sparsity 0.7500, arm cnn-s3-t0.75-seed0-20261109T195032057038Z-7209d3

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 380 | 0.0500 |
| conv2 | 32x16x5x5 | 12800 | 8520 | 0.3344 |
| fc1 | 128x800 | 102400 | 19732 | 0.8073 |
| fc2 | 10x128 | 1280 | 588 | 0.5406 |
#### target 0.90 — achieved eligible sparsity 0.9000, arm cnn-s3-t0.9-seed0-20261109T195111659601Z-0b516f

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 349 | 0.1275 |
| conv2 | 32x16x5x5 | 12800 | 5027 | 0.6073 |
| fc1 | 128x800 | 102400 | 5862 | 0.9428 |
| fc2 | 10x128 | 1280 | 450 | 0.6484 |
### S4

#### target 0.25 (pattern 3:4) — achieved eligible sparsity 0.2500, arm cnn-s4-t0.25-seed0-20261109T195149127430Z-9dda38

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 304 | 0.2400 |
| conv2 | 32x16x5x5 | 12800 | 9600 | 0.2500 |
| fc1 | 128x800 | 102400 | 76800 | 0.2500 |
| fc2 | 10x128 | 1280 | 960 | 0.2500 |
#### target 0.50 (pattern 2:4) — achieved eligible sparsity 0.4999, arm cnn-s4-t0.5-seed0-20261109T195220479628Z-4f6927

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 208 | 0.4800 |
| conv2 | 32x16x5x5 | 12800 | 6400 | 0.5000 |
| fc1 | 128x800 | 102400 | 51200 | 0.5000 |
| fc2 | 10x128 | 1280 | 640 | 0.5000 |
#### target 0.75 (pattern 1:4) — achieved eligible sparsity 0.7499, arm cnn-s4-t0.75-seed0-20261109T195249685037Z-c7602f

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 112 | 0.7200 |
| conv2 | 32x16x5x5 | 12800 | 3200 | 0.7500 |
| fc1 | 128x800 | 102400 | 25600 | 0.7500 |
| fc2 | 10x128 | 1280 | 320 | 0.7500 |
#### target 0.90 (pattern 1:10) — achieved eligible sparsity 0.8999, arm cnn-s4-t0.9-seed0-20261109T195320711932Z-a2e317

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1x5x5 | 400 | 48 | 0.8800 |
| conv2 | 32x16x5x5 | 12800 | 1280 | 0.9000 |
| fc1 | 128x800 | 102400 | 10240 | 0.9000 |
| fc2 | 10x128 | 1280 | 130 | 0.8984 |
## GCN

Dense parent shapes: conv1.lin 16x1433, conv2.lin 7x16

### S1

#### target 0.25 — achieved eligible sparsity 0.2500, arm gcn-s1-t0.25-seed0-20261109T195421719956Z-6345ff

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 17182 | 0.2506 |
| conv1.lin | 16x1433 | 22928 | 17182 | 0.2506 |
| conv2 | 7x16 | 112 | 98 | 0.1250 |
| conv2.lin | 7x16 | 112 | 98 | 0.1250 |
#### target 0.50 — achieved eligible sparsity 0.5000, arm gcn-s1-t0.5-seed0-20261109T195430968359Z-bdea8b

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 11422 | 0.5018 |
| conv1.lin | 16x1433 | 22928 | 11422 | 0.5018 |
| conv2 | 7x16 | 112 | 98 | 0.1250 |
| conv2.lin | 7x16 | 112 | 98 | 0.1250 |
#### target 0.75 — achieved eligible sparsity 0.7500, arm gcn-s1-t0.75-seed0-20261109T195438285989Z-38ebfe

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 5664 | 0.7530 |
| conv1.lin | 16x1433 | 22928 | 5664 | 0.7530 |
| conv2 | 7x16 | 112 | 96 | 0.1429 |
| conv2.lin | 7x16 | 112 | 96 | 0.1429 |
#### target 0.90 — achieved eligible sparsity 0.9000, arm gcn-s1-t0.9-seed0-20261109T195443888861Z-5888b1

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 2209 | 0.9037 |
| conv1.lin | 16x1433 | 22928 | 2209 | 0.9037 |
| conv2 | 7x16 | 112 | 95 | 0.1518 |
| conv2.lin | 7x16 | 112 | 95 | 0.1518 |
### S2

#### target 0.25 (group sparsity achieved 0.2500) — achieved eligible sparsity 0.2500, arm gcn-s2-t0.25-seed0-20261109T195448350709Z-2776bb

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 12x1433 | 17196 | 17196 | 0.0000 |
| conv1.lin | 12x1433 | 17196 | 17196 | 0.0000 |
| conv2 | 7x12 | 84 | 84 | 0.0000 |
| conv2.lin | 7x12 | 84 | 84 | 0.0000 |


#### target 0.50 (group sparsity achieved 0.5000) — achieved eligible sparsity 0.5000, arm gcn-s2-t0.5-seed0-20261109T195451832421Z-d2da8d

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 8x1433 | 11464 | 11464 | 0.0000 |
| conv1.lin | 8x1433 | 11464 | 11464 | 0.0000 |
| conv2 | 7x8 | 56 | 56 | 0.0000 |
| conv2.lin | 7x8 | 56 | 56 | 0.0000 |


#### target 0.75 (group sparsity achieved 0.7500) — achieved eligible sparsity 0.7500, arm gcn-s2-t0.75-seed0-20261109T195455294713Z-3f7059

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 4x1433 | 5732 | 5732 | 0.0000 |
| conv1.lin | 4x1433 | 5732 | 5732 | 0.0000 |
| conv2 | 7x4 | 28 | 28 | 0.0000 |
| conv2.lin | 7x4 | 28 | 28 | 0.0000 |


#### target 0.90 (group sparsity achieved 0.8750) — achieved eligible sparsity 0.8750, arm gcn-s2-t0.9-seed0-20261109T195458756975Z-bfe41c

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 2x1433 | 2866 | 2866 | 0.0000 |
| conv1.lin | 2x1433 | 2866 | 2866 | 0.0000 |
| conv2 | 7x2 | 14 | 14 | 0.0000 |
| conv2.lin | 7x2 | 14 | 14 | 0.0000 |


### S3

#### target 0.25 — achieved eligible sparsity 0.2500, arm gcn-s3-t0.25-seed0-20261109T195502225525Z-bc0bc2

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 17182 | 0.2506 |
| conv1.lin | 16x1433 | 22928 | 17182 | 0.2506 |
| conv2 | 7x16 | 112 | 98 | 0.1250 |
| conv2.lin | 7x16 | 112 | 98 | 0.1250 |
#### target 0.50 — achieved eligible sparsity 0.5000, arm gcn-s3-t0.5-seed0-20261109T195511370535Z-585487

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 11422 | 0.5018 |
| conv1.lin | 16x1433 | 22928 | 11422 | 0.5018 |
| conv2 | 7x16 | 112 | 98 | 0.1250 |
| conv2.lin | 7x16 | 112 | 98 | 0.1250 |
#### target 0.75 — achieved eligible sparsity 0.7500, arm gcn-s3-t0.75-seed0-20261109T195518667586Z-5a3f20

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 5662 | 0.7531 |
| conv1.lin | 16x1433 | 22928 | 5662 | 0.7531 |
| conv2 | 7x16 | 112 | 98 | 0.1250 |
| conv2.lin | 7x16 | 112 | 98 | 0.1250 |
#### target 0.90 — achieved eligible sparsity 0.9000, arm gcn-s3-t0.9-seed0-20261109T195524206001Z-8ba903

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 2208 | 0.9037 |
| conv1.lin | 16x1433 | 22928 | 2208 | 0.9037 |
| conv2 | 7x16 | 112 | 96 | 0.1429 |
| conv2.lin | 7x16 | 112 | 96 | 0.1429 |
### S4

#### target 0.25 (pattern 3:4) — achieved eligible sparsity 0.2498, arm gcn-s4-t0.25-seed0-20261109T195528623568Z-b212b0

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 17200 | 0.2498 |
| conv1.lin | 16x1433 | 22928 | 17200 | 0.2498 |
| conv2 | 7x16 | 112 | 84 | 0.2500 |
| conv2.lin | 7x16 | 112 | 84 | 0.2500 |
#### target 0.50 (pattern 2:4) — achieved eligible sparsity 0.4997, arm gcn-s4-t0.5-seed0-20261109T195532318881Z-114b84

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 11472 | 0.4997 |
| conv1.lin | 16x1433 | 22928 | 11472 | 0.4997 |
| conv2 | 7x16 | 112 | 56 | 0.5000 |
| conv2.lin | 7x16 | 112 | 56 | 0.5000 |
#### target 0.75 (pattern 1:4) — achieved eligible sparsity 0.7495, arm gcn-s4-t0.75-seed0-20261109T195536011282Z-ef3f1c

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 5744 | 0.7495 |
| conv1.lin | 16x1433 | 22928 | 5744 | 0.7495 |
| conv2 | 7x16 | 112 | 28 | 0.7500 |
| conv2.lin | 7x16 | 112 | 28 | 0.7500 |
#### target 0.90 (pattern 1:10) — achieved eligible sparsity 0.8994, arm gcn-s4-t0.9-seed0-20261109T195539699156Z-c4417b

| layer | shape | numel | nonzero | sparsity |
|---|---|---|---|---|
| conv1 | 16x1433 | 22928 | 2304 | 0.8995 |
| conv1.lin | 16x1433 | 22928 | 2304 | 0.8995 |
| conv2 | 7x16 | 112 | 14 | 0.8750 |
| conv2.lin | 7x16 | 112 | 14 | 0.8750 |