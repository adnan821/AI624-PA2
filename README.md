# AI624 — Assignment 2: Pruning (ResNet-18 on CIFAR-10)

Muhammad Adnan · 25280067

Everything is in one notebook, `PA2_25280067.ipynb`. It covers Task 0 (baseline), Task 1
(unstructured post training pruning), Task 2 (GraSP iterative pruning) and Task 3
(structured channel pruning). The submitted copy has all outputs saved.

## Dependencies

Tested with Python 3.12 on an NVIDIA RTX 3060 (driver 535, CUDA 12.2).

| Package | Version used |
|---|---|
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| numpy | 2.5.2 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 (LASSO in Task 3, macro-F1 cross check) |
| thop | 0.1.1 (MAC cross check) |
| nvidia-ml-py | 13.615.71 (GPU memory and energy via NVML) |
| tqdm, ipykernel | any recent version |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib scikit-learn thop nvidia-ml-py tqdm ipykernel
```

The first code cell checks these packages and prints the `pip` command for anything
missing.

**CPU energy (optional).** CPU energy is read from Intel RAPL, which only root can
read by default. To enable it, run this once per boot:

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl*
```

Without it, CPU energy is reported as unavailable and everything else still runs.

## How to run

1. Open `PA2_25280067.ipynb` with the kernel from the environment above.
2. Run all cells (*Restart & Run All*). A full run takes about 28 minutes on an
   RTX 3060 (the submitted outputs come from one such run).
3. The first run downloads CIFAR-10 to `data/` and the pretrained checkpoint to
   `checkpoints/resnet18.pt` (a 933 MB archive from the
   [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) Google
   Drive link; about 15–20 minutes at Drive's speed). If the download fails, the cell
   prints manual instructions.

Two expensive steps are cached and reused if their files exist. Delete them to
recompute:

- `artifacts/task1_sensitivity.csv` — the Task 1 layerwise sensitivity sweep
- `checkpoints/task2_warmup.pt` — the Task 2 warm up from random initialization

## Code outline

| Notebook section | What it does |
|---|---|
| Environment & configuration | package check, all hyperparameters, seeding, results store |
| Task 0 | model definition (`CifarResNet`), checkpoint download, dataloaders, `evaluate`, MAC/size counting, `profile_model` (latency, memory, energy), baseline results |
| Task 1 | sensitivity sweep, local and global magnitude pruning, `MaskEnforcer`, fine tuning, mask checks, COO storage, `SparseCOOConv2d` sparse inference, results and discussion |
| Task 2 | calibration set, `grasp_scores` (Hessian gradient product, masked to surviving weights), sign check, cumulative masks, warm up and its curves, `run_iterative_pruning` for GraSP and magnitude with a BatchNorm recomputation check at every step, per stage and per layer results, discussion |
| Task 3 | activation collection with `F.unfold`, LASSO channel selection (Gram form), least squares reconstruction, block surgery, channel budgets, sequential pruning, fine tuning, profiling, discussion |

## Outputs

Sizes and memory are reported in MiB (1024² bytes).

| Path | Contents |
|---|---|
| `artifacts/results.json` | every metric for every model |
| `artifacts/report_tables.md` | all result tables in markdown |
| `artifacts/task*_discussion_numbers.json` | numbers quoted in each discussion |
| `artifacts/task*_history.json`, `task1_finetune_history.json` | training curves and per stage records |
| `artifacts/figures/*.png` | all figures |
| `artifacts/lr0.01_runs/` | the two earlier fine tuning runs at lr 0.01 cited in the report (Task 1 and Task 3) |
| `checkpoints/` | pruned models, masks and COO checkpoints |
