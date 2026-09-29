
### Task 0 — data and preprocessing configuration

|                    | value                                              |
|:-------------------|:---------------------------------------------------|
| dataset            | CIFAR-10                                           |
| train_images       | 50000                                              |
| test_images        | 10000                                              |
| input_resolution   | 32x32x3                                            |
| train_batch_size   | 128                                                |
| eval_batch_size    | 256                                                |
| profile_batch_size | 128                                                |
| normalization_mean | (0.4914, 0.4822, 0.4465)                           |
| normalization_std  | (0.2471, 0.2435, 0.2616)                           |
| train_augmentation | RandomCrop(32, padding=4) + RandomHorizontalFlip() |
| eval_augmentation  | none (ToTensor + Normalize only)                   |
| dataloader_workers | 8                                                  |

### Task 0 — dense FP32 baseline

|                                   |   Dense FP32 baseline |
|:----------------------------------|----------------------:|
| Train Top-1 (%)                   |            99.87      |
| Test Top-1 (%)                    |            93.07      |
| Test Top-5 (%)                    |            99.74      |
| Test macro-F1                     |             0.9307    |
| Parameters                        |             1.117e+07 |
| Model size (MB)                   |            42.7       |
| MACs / image                      |             1.402e+08 |
| FLOPs / image                     |             2.804e+08 |
| Latency (ms / batch of 128)       |             6.641     |
| Latency (ms / image)              |             0.05188   |
| Peak GPU memory, activations (MB) |            64         |
| Avg GPU memory, activations (MB)  |            13.46      |
| Model weights resident (MB)       |            42.66      |
| GPU energy (J / 50 batches)       |           440.3       |
| CPU energy (J / 50 batches)       |           133.8       |
| Energy (mJ / image)               |             8.97      |

### Task 0 — per-layer parameter and MAC breakdown

| layer                 | type   | shape            |   params |   % of prunable params |   MACs / image |
|:----------------------|:-------|:-----------------|---------:|-----------------------:|---------------:|
| conv1                 | Conv2d | (64, 3, 3, 3)    |     1728 |                0.01548 |      1.769e+06 |
| layer1.0.conv1        | Conv2d | (64, 64, 3, 3)   |    36864 |                0.3302  |      9.437e+06 |
| layer1.0.conv2        | Conv2d | (64, 64, 3, 3)   |    36864 |                0.3302  |      9.437e+06 |
| layer1.1.conv1        | Conv2d | (64, 64, 3, 3)   |    36864 |                0.3302  |      9.437e+06 |
| layer1.1.conv2        | Conv2d | (64, 64, 3, 3)   |    36864 |                0.3302  |      9.437e+06 |
| layer2.0.conv1        | Conv2d | (128, 64, 3, 3)  |    73728 |                0.6604  |      4.719e+06 |
| layer2.0.conv2        | Conv2d | (128, 128, 3, 3) |   147456 |                1.321   |      9.437e+06 |
| layer2.0.downsample.0 | Conv2d | (128, 64, 1, 1)  |     8192 |                0.07338 |      5.243e+05 |
| layer2.1.conv1        | Conv2d | (128, 128, 3, 3) |   147456 |                1.321   |      9.437e+06 |
| layer2.1.conv2        | Conv2d | (128, 128, 3, 3) |   147456 |                1.321   |      9.437e+06 |
| layer3.0.conv1        | Conv2d | (256, 128, 3, 3) |   294912 |                2.642   |      4.719e+06 |
| layer3.0.conv2        | Conv2d | (256, 256, 3, 3) |   589824 |                5.283   |      9.437e+06 |
| layer3.0.downsample.0 | Conv2d | (256, 128, 1, 1) |    32768 |                0.2935  |      5.243e+05 |
| layer3.1.conv1        | Conv2d | (256, 256, 3, 3) |   589824 |                5.283   |      9.437e+06 |
| layer3.1.conv2        | Conv2d | (256, 256, 3, 3) |   589824 |                5.283   |      9.437e+06 |
| layer4.0.conv1        | Conv2d | (512, 256, 3, 3) |  1179648 |               10.57    |      4.719e+06 |
| layer4.0.conv2        | Conv2d | (512, 512, 3, 3) |  2359296 |               21.13    |      9.437e+06 |
| layer4.0.downsample.0 | Conv2d | (512, 256, 1, 1) |   131072 |                1.174   |      5.243e+05 |
| layer4.1.conv1        | Conv2d | (512, 512, 3, 3) |  2359296 |               21.13    |      9.437e+06 |
| layer4.1.conv2        | Conv2d | (512, 512, 3, 3) |  2359296 |               21.13    |      9.437e+06 |
| fc                    | Linear | (10, 512)        |     5120 |                0.04586 |   5120         |

### Task 1.1 — layer-wise sensitivity: Top-1 (%) vs per-layer sparsity, no fine-tuning

| layer                 |   0.0 |   0.1 |   0.2 |   0.5 |   0.7 |   0.8 |   0.9 |
|:----------------------|------:|------:|------:|------:|------:|------:|------:|
| conv1                 | 93.65 | 93.65 | 93.65 | 92.25 | 78.65 | 43.00 | 22.90 |
| layer1.0.conv1        | 93.65 | 93.65 | 93.65 | 93.50 | 92.95 | 91.80 | 88.70 |
| layer1.0.conv2        | 93.65 | 93.65 | 93.65 | 93.40 | 92.90 | 92.50 | 89.25 |
| layer1.1.conv1        | 93.65 | 93.65 | 93.75 | 93.60 | 92.05 | 91.00 | 89.55 |
| layer1.1.conv2        | 93.65 | 93.65 | 93.70 | 93.65 | 93.35 | 92.85 | 91.30 |
| layer2.0.conv1        | 93.65 | 93.70 | 93.85 | 93.55 | 92.15 | 87.85 | 61.70 |
| layer2.0.conv2        | 93.65 | 93.70 | 93.65 | 93.50 | 93.30 | 92.75 | 88.80 |
| layer2.0.downsample.0 | 93.65 | 93.65 | 93.70 | 93.50 | 93.00 | 92.90 | 92.10 |
| layer2.1.conv1        | 93.65 | 93.75 | 93.65 | 93.55 | 91.50 | 90.65 | 82.85 |
| layer2.1.conv2        | 93.65 | 93.70 | 93.80 | 93.65 | 93.20 | 92.45 | 90.30 |
| layer3.0.conv1        | 93.65 | 93.65 | 93.65 | 93.75 | 93.30 | 92.85 | 91.00 |
| layer3.0.conv2        | 93.65 | 93.65 | 93.70 | 93.85 | 93.15 | 93.20 | 91.30 |
| layer3.0.downsample.0 | 93.65 | 93.65 | 93.65 | 93.70 | 93.70 | 93.60 | 93.65 |
| layer3.1.conv1        | 93.65 | 93.65 | 93.75 | 93.80 | 93.60 | 93.55 | 93.15 |
| layer3.1.conv2        | 93.65 | 93.65 | 93.65 | 93.60 | 93.45 | 93.40 | 91.85 |
| layer4.0.conv1        | 93.65 | 93.65 | 93.65 | 93.70 | 93.75 | 93.75 | 93.75 |
| layer4.0.conv2        | 93.65 | 93.65 | 93.65 | 93.65 | 93.70 | 93.70 | 93.75 |
| layer4.0.downsample.0 | 93.65 | 93.65 | 93.65 | 93.70 | 93.85 | 93.80 | 93.70 |
| layer4.1.conv1        | 93.65 | 93.65 | 93.65 | 93.65 | 93.65 | 93.70 | 93.65 |
| layer4.1.conv2        | 93.65 | 93.65 | 93.65 | 93.65 | 93.65 | 93.70 | 93.75 |
| fc                    | 93.65 | 93.65 | 93.65 | 93.70 | 93.70 | 93.75 | 93.75 |

### Task 1.1 — accuracy drop (pp) at 90% single-layer sparsity, most sensitive first

| layer                 |   Top-1 drop (pp) |
|:----------------------|------------------:|
| conv1                 |             70.75 |
| layer2.0.conv1        |             31.95 |
| layer2.1.conv1        |             10.80 |
| layer1.0.conv1        |              4.95 |
| layer2.0.conv2        |              4.85 |
| layer1.0.conv2        |              4.40 |
| layer1.1.conv1        |              4.10 |
| layer2.1.conv2        |              3.35 |
| layer3.0.conv1        |              2.65 |
| layer1.1.conv2        |              2.35 |
| layer3.0.conv2        |              2.35 |
| layer3.1.conv2        |              1.80 |
| layer2.0.downsample.0 |              1.55 |
| layer3.1.conv1        |              0.50 |
| layer3.0.downsample.0 |              0.00 |
| layer4.1.conv1        |              0.00 |
| layer4.0.downsample.0 |             -0.05 |
| layer4.0.conv1        |             -0.10 |
| layer4.0.conv2        |             -0.10 |
| layer4.1.conv2        |             -0.10 |
| fc                    |             -0.10 |

### Task 1.2 — per-layer sparsity chosen for local pruning

|                       |       params |   tolerable (sweep) |   local ratio r_l |   weights pruned |
|:----------------------|-------------:|--------------------:|------------------:|-----------------:|
| conv1                 | 1728         |                 0.5 |            0.454  |      784         |
| layer1.0.conv1        |    3.686e+04 |                 0.8 |            0.7264 |        2.678e+04 |
| layer1.0.conv2        |    3.686e+04 |                 0.8 |            0.7264 |        2.678e+04 |
| layer1.1.conv1        |    3.686e+04 |                 0.7 |            0.6356 |        2.343e+04 |
| layer1.1.conv2        |    3.686e+04 |                 0.8 |            0.7264 |        2.678e+04 |
| layer2.0.conv1        |    7.373e+04 |                 0.7 |            0.6356 |        4.686e+04 |
| layer2.0.conv2        |    1.475e+05 |                 0.8 |            0.7264 |        1.071e+05 |
| layer2.0.downsample.0 | 8192         |                 0.9 |            0.8171 |     6694         |
| layer2.1.conv1        |    1.475e+05 |                 0.5 |            0.454  |        6.694e+04 |
| layer2.1.conv2        |    1.475e+05 |                 0.8 |            0.7264 |        1.071e+05 |
| layer3.0.conv1        |    2.949e+05 |                 0.8 |            0.7264 |        2.142e+05 |
| layer3.0.conv2        |    5.898e+05 |                 0.8 |            0.7264 |        4.284e+05 |
| layer3.0.downsample.0 |    3.277e+04 |                 0.9 |            0.8171 |        2.678e+04 |
| layer3.1.conv1        |    5.898e+05 |                 0.9 |            0.8171 |        4.82e+05  |
| layer3.1.conv2        |    5.898e+05 |                 0.9 |            0.8171 |        4.82e+05  |
| layer4.0.conv1        |    1.18e+06  |                 0.9 |            0.8171 |        9.639e+05 |
| layer4.0.conv2        |    2.359e+06 |                 0.9 |            0.8171 |        1.928e+06 |
| layer4.0.downsample.0 |    1.311e+05 |                 0.9 |            0.8171 |        1.071e+05 |
| layer4.1.conv1        |    2.359e+06 |                 0.9 |            0.8171 |        1.928e+06 |
| layer4.1.conv2        |    2.359e+06 |                 0.9 |            0.8171 |        1.928e+06 |
| fc                    | 5120         |                 0.9 |            0.8171 |     4184         |

### Task 1.3 — sparsity allocation at the same overall sparsity (80.00%): local (chosen) vs global (emergent)

|                       |       params |   local sparsity |   global sparsity |   tolerable (sweep) |   global - local (pp) |
|:----------------------|-------------:|-----------------:|------------------:|--------------------:|----------------------:|
| conv1                 | 1728         |           0.454  |          0.1701   |                 0.5 |               -28.38  |
| layer1.0.conv1        |    3.686e+04 |           0.7264 |          0.2692   |                 0.8 |               -45.72  |
| layer1.0.conv2        |    3.686e+04 |           0.7264 |          0.1076   |                 0.8 |               -61.87  |
| layer1.1.conv1        |    3.686e+04 |           0.6356 |          0.118    |                 0.7 |               -51.76  |
| layer1.1.conv2        |    3.686e+04 |           0.7264 |          0.1368   |                 0.8 |               -58.95  |
| layer2.0.conv1        |    7.373e+04 |           0.6356 |          0.115    |                 0.7 |               -52.05  |
| layer2.0.conv2        |    1.475e+05 |           0.7264 |          0.1233   |                 0.8 |               -60.3   |
| layer2.0.downsample.0 | 8192         |           0.8171 |          0.07935  |                 0.9 |               -73.78  |
| layer2.1.conv1        |    1.475e+05 |           0.454  |          0.1358   |                 0.5 |               -31.81  |
| layer2.1.conv2        |    1.475e+05 |           0.7264 |          0.1746   |                 0.8 |               -55.18  |
| layer3.0.conv1        |    2.949e+05 |           0.7264 |          0.204    |                 0.8 |               -52.24  |
| layer3.0.conv2        |    5.898e+05 |           0.7264 |          0.3134   |                 0.8 |               -41.3   |
| layer3.0.downsample.0 |    3.277e+04 |           0.8171 |          0.2119   |                 0.9 |               -60.52  |
| layer3.1.conv1        |    5.898e+05 |           0.8171 |          0.4888   |                 0.9 |               -32.84  |
| layer3.1.conv2        |    5.898e+05 |           0.8171 |          0.6666   |                 0.9 |               -15.06  |
| layer4.0.conv1        |    1.18e+06  |           0.8171 |          0.9032   |                 0.9 |                 8.603 |
| layer4.0.conv2        |    2.359e+06 |           0.8171 |          0.9335   |                 0.9 |                11.64  |
| layer4.0.downsample.0 |    1.311e+05 |           0.8171 |          0.3151   |                 0.9 |               -50.21  |
| layer4.1.conv1        |    2.359e+06 |           0.8171 |          0.9941   |                 0.9 |                17.7   |
| layer4.1.conv2        |    2.359e+06 |           0.8171 |          0.9523   |                 0.9 |                13.52  |
| fc                    | 5120         |           0.8171 |          0.001563 |                 0.9 |               -81.56  |

### Task 1.5 — mask verification after fine-tuning (mask-derived vs measured sparsity)

| layer                 |   local: mask sparsity |   local: measured |   local: |delta| |   global: mask sparsity |   global: measured |   global: |delta| |
|:----------------------|-----------------------:|------------------:|-----------------:|------------------------:|-------------------:|------------------:|
| conv1                 |               0.453704 |          0.453704 |         0.000000 |                0.170139 |           0.170139 |          0.000000 |
| layer1.0.conv1        |               0.726345 |          0.726345 |         0.000000 |                0.269151 |           0.269151 |          0.000000 |
| layer1.0.conv2        |               0.726345 |          0.726345 |         0.000000 |                0.107612 |           0.107612 |          0.000000 |
| layer1.1.conv1        |               0.635552 |          0.635552 |         0.000000 |                0.118001 |           0.118001 |          0.000000 |
| layer1.1.conv2        |               0.726345 |          0.726345 |         0.000000 |                0.136827 |           0.136827 |          0.000000 |
| layer2.0.conv1        |               0.635552 |          0.635552 |         0.000000 |                0.115044 |           0.115044 |          0.000000 |
| layer2.0.conv2        |               0.726352 |          0.726352 |         0.000000 |                0.123311 |           0.123311 |          0.000000 |
| layer2.0.downsample.0 |               0.817139 |          0.817139 |         0.000000 |                0.079346 |           0.079346 |          0.000000 |
| layer2.1.conv1        |               0.453966 |          0.453966 |         0.000000 |                0.135844 |           0.135844 |          0.000000 |
| layer2.1.conv2        |               0.726352 |          0.726352 |         0.000000 |                0.174581 |           0.174581 |          0.000000 |
| layer3.0.conv1        |               0.726352 |          0.726352 |         0.000000 |                0.203959 |           0.203959 |          0.000000 |
| layer3.0.conv2        |               0.726351 |          0.726351 |         0.000000 |                0.313360 |           0.313360 |          0.000000 |
| layer3.0.downsample.0 |               0.817139 |          0.817139 |         0.000000 |                0.211945 |           0.211945 |          0.000000 |
| layer3.1.conv1        |               0.817144 |          0.817144 |         0.000000 |                0.488773 |           0.488773 |          0.000000 |
| layer3.1.conv2        |               0.817144 |          0.817144 |         0.000000 |                0.666584 |           0.666584 |          0.000000 |
| layer4.0.conv1        |               0.817145 |          0.817145 |         0.000000 |                0.903175 |           0.903175 |          0.000000 |
| layer4.0.conv2        |               0.817145 |          0.817145 |         0.000000 |                0.933518 |           0.933518 |          0.000000 |
| layer4.0.downsample.0 |               0.817146 |          0.817146 |         0.000000 |                0.315063 |           0.315063 |          0.000000 |
| layer4.1.conv1        |               0.817145 |          0.817145 |         0.000000 |                0.994107 |           0.994107 |          0.000000 |
| layer4.1.conv2        |               0.817145 |          0.817145 |         0.000000 |                0.952345 |           0.952345 |          0.000000 |
| fc                    |               0.817187 |          0.817187 |         0.000000 |                0.001563 |           0.001563 |          0.000000 |

### Task 1.6 — storage: dense vs COO sparse checkpoints

|                            |   local + FT |   global + FT |
|:---------------------------|-------------:|--------------:|
| dense .pt (MB)             |   42.7       |    42.7       |
| dense .pt gzip (MB)        |   10.6       |     9.901     |
| COO int64 + masks (MB)     |   87.41      |    87.35      |
| COO int64 no masks (MB)    |   76.76      |    76.7       |
| COO int32 no masks (MB)    |   42.7       |    42.67      |
| theory: dense payload (MB) |   42.59      |    42.59      |
| theory: COO int64 (MB)     |   76.65      |    76.58      |
| theory: COO int32 (MB)     |   42.58      |    42.55      |
| theory: COO int16 (MB)     |   25.55      |    25.53      |
| nonzeros                   |    2.233e+06 |     2.233e+06 |

### Task 1.6 — when does COO actually beat dense fp32 storage?

|   tensor rank |   bytes/nonzero (int64 COO) |   break-even sparsity |   bytes/nonzero (int32 COO) |   break-even sparsity (int32) |
|--------------:|----------------------------:|----------------------:|----------------------------:|------------------------------:|
|             2 |                          20 |                0.8    |                          12 |                        0.6667 |
|             4 |                          36 |                0.8889 |                          20 |                        0.8    |

### Task 1 — master comparison at 80.00% overall sparsity

|                                |   Sparsity |   Train Top-1 |   Test Top-1 |   Train Top-5 |   Test Top-5 |   Macro-F1 |      Params |   Params (nonzero) |   Dense size (MB) |   Sparse size (MB) |        MACs |   MACs (nonzero) |       FLOPs |   Latency (ms/batch) |   Latency (ms/img) |   Peak GPU mem (MB) |   Avg GPU mem (MB) |   GPU energy (J) |   CPU energy (J) |   Energy (mJ/img) |
|:-------------------------------|-----------:|--------------:|-------------:|--------------:|-------------:|-----------:|------------:|-------------------:|------------------:|-------------------:|------------:|-----------------:|------------:|---------------------:|-------------------:|--------------------:|-------------------:|-----------------:|-----------------:|------------------:|
| Dense FP32 baseline            |      nan   |         99.87 |        93.07 |        100    |        99.74 |     0.9307 |   1.117e+07 |          1.117e+07 |              42.7 |             nan    |   1.402e+08 |        1.402e+08 |   2.804e+08 |                6.641 |            0.05188 |                  64 |              13.46 |            440.3 |            133.8 |             8.97  |
| Local prune 80% (no FT)        |        0.8 |         87.21 |        82.42 |         99.57 |        98.66 |     0.8274 | nan         |        nan         |             nan   |             nan    | nan         |      nan         | nan         |              nan     |          nan       |                 nan |             nan    |            nan   |            nan   |           nan     |
| Global prune 80% (no FT)       |        0.8 |         99.85 |        93.11 |        100    |        99.75 |     0.9312 | nan         |        nan         |             nan   |             nan    | nan         |      nan         | nan         |              nan     |          nan       |                 nan |             nan    |            nan   |            nan   |           nan     |
| Local 80% + FT (dense masked)  |        0.8 |         99.46 |        92.8  |        100    |        99.78 |     0.9279 |   1.117e+07 |          2.242e+06 |              42.7 |              42.7  |   1.402e+08 |        3.784e+07 |   2.804e+08 |                6.652 |            0.05197 |                  64 |              13.82 |            424.4 |            113.4 |             8.403 |
| Global 80% + FT (dense masked) |        0.8 |         99.5  |        92.64 |         99.99 |        99.74 |     0.9263 |   1.117e+07 |          2.242e+06 |              42.7 |              42.67 |   1.402e+08 |        8.271e+07 |   2.804e+08 |                6.666 |            0.05208 |                  64 |              12.66 |            426.1 |            108.8 |             8.357 |
| Local 80% + FT (COO)           |        0.8 |         99.46 |        92.8  |        100    |        99.78 |     0.9279 |   1.117e+07 |          2.242e+06 |              42.7 |              42.7  |   1.402e+08 |        3.784e+07 |   2.804e+08 |                7.377 |            0.05764 |                  64 |              14.11 |            460.2 |            125.3 |             9.148 |
| Global 80% + FT (COO)          |        0.8 |         99.5  |        92.64 |         99.99 |        99.74 |     0.9263 |   1.117e+07 |          2.242e+06 |              42.7 |              42.67 |   1.402e+08 |        8.271e+07 |   2.804e+08 |                7.32  |            0.05719 |                  64 |              13.35 |            466.6 |            130.2 |             9.325 |

### Task 1 — accuracy vs deployment cost

|                                |   Overall sparsity |   Test Top-1 (%) |   Delta vs dense (pp) |   Macro-F1 |   Latency (ms/batch) |   Latency vs dense |   MACs (nonzero) / dense |   Dense size (MB) |   Sparse size (MB) |
|:-------------------------------|-------------------:|-----------------:|----------------------:|-----------:|---------------------:|-------------------:|-------------------------:|------------------:|-------------------:|
| Dense FP32 baseline            |              nan   |            93.07 |                  0    |     0.9307 |                6.641 |              1     |                     1    |              42.7 |             nan    |
| Local prune 80% (no FT)        |                0.8 |            82.42 |                -10.65 |     0.8274 |              nan     |            nan     |                   nan    |             nan   |             nan    |
| Global prune 80% (no FT)       |                0.8 |            93.11 |                  0.04 |     0.9312 |              nan     |            nan     |                   nan    |             nan   |             nan    |
| Local 80% + FT (dense masked)  |                0.8 |            92.8  |                 -0.27 |     0.9279 |                6.652 |              1.002 |                     0.27 |              42.7 |              42.7  |
| Global 80% + FT (dense masked) |                0.8 |            92.64 |                 -0.43 |     0.9263 |                6.666 |              1.004 |                     0.59 |              42.7 |              42.67 |
| Local 80% + FT (COO)           |                0.8 |            92.8  |                 -0.27 |     0.9279 |                7.377 |              1.111 |                     0.27 |              42.7 |              42.7  |
| Global 80% + FT (COO)          |                0.8 |            92.64 |                 -0.43 |     0.9263 |                7.32  |              1.102 |                     0.59 |              42.7 |              42.67 |

### Task 2.1 — GraSP calibration set

|                        | value                                                                                                                    |
|:-----------------------|:-------------------------------------------------------------------------------------------------------------------------|
| calibration_samples    | 100                                                                                                                      |
| samples_per_class      | 10                                                                                                                       |
| calibration_batch_size | 50                                                                                                                       |
| number_of_batches      | 2                                                                                                                        |
| gradient_combination   | averaged over batches (sum / n_batches) for both g and Hg                                                                |
| class_counts           | {'plane': 10, 'car': 10, 'bird': 10, 'cat': 10, 'deer': 10, 'dog': 10, 'frog': 10, 'horse': 10, 'ship': 10, 'truck': 10} |
| source                 | CIFAR-10 training set, evaluation transforms (no augmentation)                                                           |
| logit_temperature      | 200.0                                                                                                                    |

### Task 2.3 — sorting-direction check at 80% (dense ||grad L||^2 = 7.8679e-05)

|                                           |   ||grad L||^2 after pruning |   fraction of dense gradient flow |
|:------------------------------------------|-----------------------------:|----------------------------------:|
| prune high S (keep -S largest) [paper]    |                    0.005099  |                            64.81  |
| prune low S  (keep +S largest) [inverted] |                    0.0002458 |                             3.125 |

### Task 2 — pruning stages: accuracy before/after, sparsity, criterion cost

|                            |   iteration |   epoch |   target_sparsity |   achieved_sparsity |   weights_pruned_this_stage |   acc_before_prune |   acc_after_prune |   acc_drop |   criterion_seconds |   min_layer_density |
|:---------------------------|------------:|--------:|------------------:|--------------------:|----------------------------:|-------------------:|------------------:|-----------:|--------------------:|--------------------:|
| ('iterative_magnitude', 1) |         100 |  0.2558 |               0.4 |                 0.4 |                   4.466e+06 |              20.85 |             20.7  |       0.15 |            0.002392 |              0.5559 |
| ('iterative_magnitude', 2) |         350 |  0.8951 |               0.6 |                 0.6 |                   2.233e+06 |              44.3  |             43.1  |       1.2  |            0.002157 |              0.3419 |
| ('iterative_magnitude', 3) |         800 |  2.046  |               0.8 |                 0.8 |                   2.233e+06 |              61.7  |             60    |       1.7  |            0.001925 |              0.1382 |
| ('iterative_grasp', 1)     |         100 |  0.2558 |               0.4 |                 0.4 |                   4.466e+06 |              20.85 |              9.7  |      11.15 |            0.2358   |              0.4974 |
| ('iterative_grasp', 2)     |         400 |  1.023  |               0.6 |                 0.6 |                   2.233e+06 |              42.65 |              9.9  |      32.75 |            0.2353   |              0.228  |
| ('iterative_grasp', 3)     |        1500 |  3.836  |               0.8 |                 0.8 |                   2.233e+06 |              60.3  |             10.35 |      49.95 |            0.2267   |              0.1175 |

### Task 2 — cost of computing the pruning criterion

|                            |   iterative_magnitude |   iterative_grasp |
|:---------------------------|----------------------:|------------------:|
| criterion calls            |              3        |            3      |
| total criterion time (s)   |              0.006475 |            0.6978 |
| mean per call (s)          |              0.002158 |            0.2326 |
| training wall time (min)   |              7.441    |            7.464  |
| criterion as % of training |              0.00145  |            0.1558 |

### Task 2 — layer-wise remaining-weight ratio

|                       |       params |   iter. magnitude: remaining |   iter. GraSP: remaining |   GraSP - magnitude (pp) |
|:----------------------|-------------:|-----------------------------:|-------------------------:|-------------------------:|
| conv1                 | 1728         |                       0.7992 |                   0.1175 |                  -68.17  |
| layer1.0.conv1        |    3.686e+04 |                       0.5879 |                   0.1293 |                  -45.86  |
| layer1.0.conv2        |    3.686e+04 |                       0.5791 |                   0.1242 |                  -45.49  |
| layer1.1.conv1        |    3.686e+04 |                       0.5833 |                   0.1281 |                  -45.52  |
| layer1.1.conv2        |    3.686e+04 |                       0.571  |                   0.1207 |                  -45.03  |
| layer2.0.conv1        |    7.373e+04 |                       0.4757 |                   0.1276 |                  -34.82  |
| layer2.0.conv2        |    1.475e+05 |                       0.4583 |                   0.1305 |                  -32.78  |
| layer2.0.downsample.0 | 8192         |                       0.7507 |                   0.1228 |                  -62.79  |
| layer2.1.conv1        |    1.475e+05 |                       0.4585 |                   0.1308 |                  -32.78  |
| layer2.1.conv2        |    1.475e+05 |                       0.4583 |                   0.1264 |                  -33.19  |
| layer3.0.conv1        |    2.949e+05 |                       0.3188 |                   0.135  |                  -18.38  |
| layer3.0.conv2        |    5.898e+05 |                       0.3072 |                   0.1412 |                  -16.59  |
| layer3.0.downsample.0 |    3.277e+04 |                       0.6918 |                   0.1231 |                  -56.87  |
| layer3.1.conv1        |    5.898e+05 |                       0.3024 |                   0.1486 |                  -15.38  |
| layer3.1.conv2        |    5.898e+05 |                       0.3024 |                   0.1415 |                  -16.09  |
| layer4.0.conv1        |    1.18e+06  |                       0.1437 |                   0.1924 |                    4.871 |
| layer4.0.conv2        |    2.359e+06 |                       0.1414 |                   0.2332 |                    9.177 |
| layer4.0.downsample.0 |    1.311e+05 |                       0.61   |                   0.1236 |                  -48.64  |
| layer4.1.conv1        |    2.359e+06 |                       0.1382 |                   0.2221 |                    8.391 |
| layer4.1.conv2        |    2.359e+06 |                       0.1421 |                   0.2243 |                    8.224 |
| fc                    | 5120         |                       0.5457 |                   0.3066 |                  -23.91  |

### Task 2 — storage: dense vs COO for the iteratively pruned models

|                            |   iterative_magnitude |   iterative_grasp |
|:---------------------------|----------------------:|------------------:|
| dense .pt (MB)             |            42.7       |        42.7       |
| dense .pt gzip (MB)        |            12         |        12.19      |
| COO int64 + masks (MB)     |            87.39      |        87.41      |
| COO int64 no masks (MB)    |            76.74      |        76.75      |
| COO int32 no masks (MB)    |            42.69      |        42.69      |
| theory: dense payload (MB) |            42.59      |        42.59      |
| theory: COO int64 (MB)     |            76.62      |        76.64      |
| theory: COO int32 (MB)     |            42.57      |        42.58      |
| theory: COO int16 (MB)     |            25.54      |        25.55      |
| nonzeros                   |             2.233e+06 |         2.233e+06 |

### Task 2 — master comparison at 80% overall sparsity

|                                        |   Sparsity |   Train Top-1 |   Test Top-1 |   Train Top-5 |   Test Top-5 |   Macro-F1 |    Params |   Params (nonzero) |   Dense size (MB) |   Sparse size (MB) |      MACs |   MACs (nonzero) |     FLOPs |   Latency (ms/batch) |   Latency (ms/img) |   Peak GPU mem (MB) |   Avg GPU mem (MB) |   GPU energy (J) |   CPU energy (J) |   Energy (mJ/img) |
|:---------------------------------------|-----------:|--------------:|-------------:|--------------:|-------------:|-----------:|----------:|-------------------:|------------------:|-------------------:|----------:|-----------------:|----------:|---------------------:|-------------------:|--------------------:|-------------------:|-----------------:|-----------------:|------------------:|
| Dense FP32 baseline                    |      nan   |         99.87 |        93.07 |        100    |        99.74 |     0.9307 | 1.117e+07 |          1.117e+07 |              42.7 |             nan    | 1.402e+08 |        1.402e+08 | 2.804e+08 |                6.641 |            0.05188 |                  64 |              13.46 |            440.3 |            133.8 |             8.97  |
| Local 80% + FT (dense masked)          |        0.8 |         99.46 |        92.8  |        100    |        99.78 |     0.9279 | 1.117e+07 |          2.242e+06 |              42.7 |              42.7  | 1.402e+08 |        3.784e+07 | 2.804e+08 |                6.652 |            0.05197 |                  64 |              13.82 |            424.4 |            113.4 |             8.403 |
| Global 80% + FT (dense masked)         |        0.8 |         99.5  |        92.64 |         99.99 |        99.74 |     0.9263 | 1.117e+07 |          2.242e+06 |              42.7 |              42.67 | 1.402e+08 |        8.271e+07 | 2.804e+08 |                6.666 |            0.05208 |                  64 |              12.66 |            426.1 |            108.8 |             8.357 |
| Iterative magnitude 80% (dense masked) |        0.8 |         98.34 |        91.12 |         99.99 |        99.75 |     0.9112 | 1.117e+07 |          2.242e+06 |              42.7 |              42.69 | 1.402e+08 |        5.439e+07 | 2.804e+08 |                6.663 |            0.05206 |                  64 |              12.24 |            439.8 |            109.9 |             8.589 |
| Iterative GraSP 80% (dense masked)     |        0.8 |         95.03 |        88    |         99.94 |        99.57 |     0.8795 | 1.117e+07 |          2.242e+06 |              42.7 |              42.69 | 1.402e+08 |        2.143e+07 | 2.804e+08 |                6.717 |            0.05247 |                  64 |              12.22 |            426.8 |            108.8 |             8.369 |
| Iterative magnitude 80% (COO)          |        0.8 |         98.34 |        91.12 |         99.99 |        99.75 |     0.9112 | 1.117e+07 |          2.242e+06 |              42.7 |              42.69 | 1.402e+08 |        5.439e+07 | 2.804e+08 |                7.433 |            0.05807 |                  64 |              12.04 |            463.7 |            120.9 |             9.134 |
| Iterative GraSP 80% (COO)              |        0.8 |         95.03 |        88    |         99.94 |        99.57 |     0.8795 | 1.117e+07 |          2.242e+06 |              42.7 |              42.69 | 1.402e+08 |        2.143e+07 | 2.804e+08 |                7.433 |            0.05807 |                  64 |              12.56 |            454.8 |            120.8 |             8.994 |

### Task 2 — all pruning methods side by side

|                        | Start point   | Pruning              |   Sparsity |   Test Top-1 (%) |   Train Top-1 (%) |   Macro-F1 |   Effective MACs (M) |   Latency (ms) |   Criterion time (s) |
|:-----------------------|:--------------|:---------------------|-----------:|-----------------:|------------------:|-----------:|---------------------:|---------------:|---------------------:|
| Dense baseline         | pretrained    | none (dense)         |        0   |            93.07 |             99.87 |     0.9307 |               140.2  |          6.641 |           nan        |
| Task 1 local           | pretrained    | one-shot local + FT  |        0.8 |            92.8  |             99.46 |     0.9279 |                37.84 |          6.652 |           nan        |
| Task 1 global          | pretrained    | one-shot global + FT |        0.8 |            92.64 |             99.5  |     0.9263 |                82.71 |          6.666 |           nan        |
| Task 2 iter. magnitude | random init   | iterative magnitude  |        0.8 |            91.12 |             98.34 |     0.9112 |                54.39 |          6.663 |             0.006475 |
| Task 2 iter. GraSP     | random init   | iterative GraSP      |        0.8 |            88    |             95.03 |     0.8795 |                21.43 |          6.717 |             0.6978   |

### Task 3.5 — per-block channel budget (alpha=1.4551, predicted overall sparsity 80.11%)

|          |   original channels |   kept channels c0 |   keep ratio |   tolerable sparsity (Task 1) |
|:---------|--------------------:|-------------------:|-------------:|------------------------------:|
| layer1.1 |                  64 |                 28 |       0.4375 |                           0.7 |
| layer2.0 |                 128 |                 56 |       0.4375 |                           0.7 |
| layer2.1 |                 128 |                 93 |       0.7266 |                           0.5 |
| layer3.0 |                 256 |                 74 |       0.2891 |                           0.8 |
| layer3.1 |                 256 |                 37 |       0.1445 |                           0.9 |
| layer4.0 |                 512 |                 74 |       0.1445 |                           0.9 |
| layer4.1 |                 512 |                 74 |       0.1445 |                           0.9 |

### Task 3.6 — sequential channel pruning, block by block

| block    |   channels_before |   channels_kept |   keep_ratio |   lasso_alpha |   lasso_nonzero |   recon_rel_residual |   subset_top1_after |   params_after |   sparsity_after |   t_collect_s |   t_lasso_s |   t_lstsq_s |
|:---------|------------------:|----------------:|-------------:|--------------:|----------------:|---------------------:|--------------------:|---------------:|-----------------:|--------------:|------------:|------------:|
| layer1.1 |                64 |              28 |       0.4375 |     0.00382   |              28 |            0.09908   |               92.45 |      1.113e+07 |         0.003718 |        0.1401 |     0.04024 |      0.1399 |
| layer2.0 |               128 |              56 |       0.4375 |     0.004089  |              56 |            0.069     |               90.2  |      1.101e+07 |         0.01487  |        0.1469 |     0.02087 |      0.4818 |
| layer2.1 |               128 |              93 |       0.7266 |     0.0005729 |              93 |            0.02916   |               89.55 |      1.093e+07 |         0.02209  |        0.1398 |     0.0172  |      1.136  |
| layer3.0 |               256 |              74 |       0.2891 |     0.0003498 |              74 |            0.07232   |               88.5  |      1.03e+07  |         0.07841  |        0.1452 |     0.05361 |      1.044  |
| layer3.1 |               256 |              37 |       0.1445 |     0.0003846 |              37 |            0.01865   |               88.05 |      9.288e+06 |         0.1688   |        0.1419 |     0.1175  |      0.379  |
| layer4.0 |               512 |              74 |       0.1445 |     1.762e-05 |              74 |            0.0001816 |               87.95 |      6.26e+06  |         0.4398   |        0.14   |     7.575   |      1.295  |
| layer4.1 |               512 |              74 |       0.1445 |     9.207e-06 |              74 |            3.822e-05 |               88    |      2.222e+06 |         0.8011   |        0.1309 |     7.118   |      1.31   |

### All tasks — full deliverables table

|                                        |   Sparsity |   Train Top-1 |   Test Top-1 |   Train Top-5 |   Test Top-5 |   Macro-F1 |    Params |   Params (nonzero) |   Dense size (MB) |   Sparse size (MB) |      MACs |   MACs (nonzero) |     FLOPs |   Latency (ms/batch) |   Latency (ms/img) |   Peak GPU mem (MB) |   Avg GPU mem (MB) |   GPU energy (J) |   CPU energy (J) |   Energy (mJ/img) |
|:---------------------------------------|-----------:|--------------:|-------------:|--------------:|-------------:|-----------:|----------:|-------------------:|------------------:|-------------------:|----------:|-----------------:|----------:|---------------------:|-------------------:|--------------------:|-------------------:|-----------------:|-----------------:|------------------:|
| Dense FP32 baseline                    |   nan      |         99.87 |        93.07 |        100    |        99.74 |     0.9307 | 1.117e+07 |          1.117e+07 |            42.7   |             nan    | 1.402e+08 |        1.402e+08 | 2.804e+08 |                6.641 |            0.05188 |                  64 |              13.46 |            440.3 |            133.8 |             8.97  |
| Local 80% + FT (dense masked)          |     0.8    |         99.46 |        92.8  |        100    |        99.78 |     0.9279 | 1.117e+07 |          2.242e+06 |            42.7   |              42.7  | 1.402e+08 |        3.784e+07 | 2.804e+08 |                6.652 |            0.05197 |                  64 |              13.82 |            424.4 |            113.4 |             8.403 |
| Global 80% + FT (dense masked)         |     0.8    |         99.5  |        92.64 |         99.99 |        99.74 |     0.9263 | 1.117e+07 |          2.242e+06 |            42.7   |              42.67 | 1.402e+08 |        8.271e+07 | 2.804e+08 |                6.666 |            0.05208 |                  64 |              12.66 |            426.1 |            108.8 |             8.357 |
| Iterative magnitude 80% (dense masked) |     0.8    |         98.34 |        91.12 |         99.99 |        99.75 |     0.9112 | 1.117e+07 |          2.242e+06 |            42.7   |              42.69 | 1.402e+08 |        5.439e+07 | 2.804e+08 |                6.663 |            0.05206 |                  64 |              12.24 |            439.8 |            109.9 |             8.589 |
| Iterative GraSP 80% (dense masked)     |     0.8    |         95.03 |        88    |         99.94 |        99.57 |     0.8795 | 1.117e+07 |          2.242e+06 |            42.7   |              42.69 | 1.402e+08 |        2.143e+07 | 2.804e+08 |                6.717 |            0.05247 |                  64 |              12.22 |            426.8 |            108.8 |             8.369 |
| Structured channel prune + FT          |     0.8011 |         89.92 |        86.45 |         99.66 |        99.42 |     0.8638 | 2.222e+06 |          2.222e+06 |             8.541 |             nan    | 6.198e+07 |        6.198e+07 | 1.24e+08  |                5.592 |            0.04369 |                  64 |              13.03 |            326.6 |            112.8 |             6.865 |

### All tasks — accuracy vs real deployment cost

|                                             |   Sparsity |    Params |   Test Top-1 (%) |   Delta (pp) |   Macro-F1 |   Size (MB) |   MACs (M) |   Eff. MACs (M) |   Latency (ms) |   Speedup |   Peak act. mem (MB) |   Energy (mJ/img) |
|:--------------------------------------------|-----------:|----------:|-----------------:|-------------:|-----------:|------------:|-----------:|----------------:|---------------:|----------:|---------------------:|------------------:|
| ('0', 'dense FP32 baseline')                |     0      | 1.117e+07 |            93.07 |         0    |     0.9307 |      42.7   |     140.2  |          140.2  |          6.641 |    1      |                   64 |             8.97  |
| ('1', 'unstructured local magnitude + FT')  |     0.8    | 1.117e+07 |            92.8  |        -0.27 |     0.9279 |      42.7   |     140.2  |           37.84 |          6.652 |    0.9983 |                   64 |             8.403 |
| ('1', 'unstructured global magnitude + FT') |     0.8    | 1.117e+07 |            92.64 |        -0.43 |     0.9263 |      42.7   |     140.2  |           82.71 |          6.666 |    0.9962 |                   64 |             8.357 |
| ('2', 'iterative magnitude (from scratch)') |     0.8    | 1.117e+07 |            91.12 |        -1.95 |     0.9112 |      42.7   |     140.2  |           54.39 |          6.663 |    0.9966 |                   64 |             8.589 |
| ('2', 'iterative GraSP (from scratch)')     |     0.8    | 1.117e+07 |            88    |        -5.07 |     0.8795 |      42.7   |     140.2  |           21.43 |          6.717 |    0.9887 |                   64 |             8.369 |
| ('3', 'structured channel prune + FT')      |     0.8011 | 2.222e+06 |            86.45 |        -6.62 |     0.8638 |       8.541 |      61.98 |           61.98 |          5.592 |    1.188  |                   64 |             6.865 |
