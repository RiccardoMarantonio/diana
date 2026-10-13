# diana

Unsupervised **diffusion-based anomaly detection** on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(hazelnut, 64x64): train a DDPM on defect-free images only; at test time an
image is noised to an intermediate level (`t_start`) and re-imagined by the
model. The working hypothesis is that reconstruction suppresses unseen defects;
the squared reconstruction residual is the anomaly score.

Small, fully typed, CLI-only PyTorch codebase (no `train.py --argparse` soup,
no config blobs): every setting lives in `src/diana/config.py` and can be
supplied as a TOML file, a CLI flag, or both (CLI > TOML > defaults). HPC
features from the spec: AMP, `cudnn.benchmark`, CUDA graphs, `torch.profiler`
harness, tuned dataloaders. Dev/iterate on Apple Silicon (MPS), run the real
training on a Colab T4.

## Layout

```
src/diana/
  config.py            single Config dataclass + TOML/CLI merge
  diffusion/           DDPM (q_sample, eps-loss, strided sampler, SDEdit)
  models/              UNet, EMA shadow weights
  data/                MVTec downloader (HF mirror, sha256-verified),
                       dataset (+eval masks), dataloader factory
  train.py             training loop, RNG-aware resume, checkpoint cadence
  sample.py            inference: anomaly maps, image+pixel AUROC, grids
  profiler.py          HPC benchmark harness (train/inference/sweep)
  utils/               device, AMP ordering, CUDA graphs
configs/hazelnut.toml  canonical single-category config
tests/                 unit, integration, and accelerator regression tests
```

## Quick start

```bash
uv sync                                   # or: pip install -e .
uv run pytest tests/ -q                   # accelerator tests skip when unavailable
uvx ruff check src tests && uvx ty check src

# smoke run on random tensors (MPS on Apple Silicon)
uv run python -m diana.train --config configs/hazelnut.toml \
    --data_path synthetic --epochs 2

# real data
uv run python -m diana.data.download --categories hazelnut
uv run python -m diana.train --config configs/hazelnut.toml \
    --epochs 1 --batch_size 8 --num_workers 2

# resume from the latest run dir (restores weights, AMP scaler, EMA, RNG)
uv run python -m diana.train --config configs/hazelnut.toml \
    --resume_from runs/hazelnut-<timestamp>

# infer + score against the checkpoint
uv run python -m diana.sample --run runs/hazelnut-<timestamp> --pixel
```

## One-shot run (Colab T4 or local)

`t4.sh` is device-agnostic (`device = "auto"` picks CUDA on Colab, MPS on
Apple Silicon). It benchmarks and trains the selected category, evaluates
image- and pixel-level AUROC across `t_start`, then writes a CSV and SVG chart.
Pass `all` as the config to run every category config in `configs/`:

```bash
git clone https://github.com/RiccardoMarantonio/diana.git && cd diana
bash t4.sh              # 100 total epochs, at most 200 unique test images
bash t4.sh 3 32         # local smoke: 3 epochs, 32-sample eval
bash t4.sh 100 200 configs/bottle.toml
bash t4.sh 100 200 all  # train/evaluate all configs, one after another
./run_all_configs.sh 100 200  # explicit runner over configs/*.toml
```

`run_all_configs.sh` takes the same epoch and image-limit arguments as `t4.sh`
and runs the full workflow once per TOML config. It stops if any category
fails. `FRESH=1` and `T_STARTS` are forwarded to each run.

`data/` is gitignored, so the script re-downloads it on a fresh machine. If a
session dies mid-training, rerunning resumes the latest category checkpoint.
Set `FRESH=1` to start a new model instead. The script verifies existing data
and repairs incomplete downloads on every run. It does not update the Git checkout.
The config snapshot lives in the run directory:

```bash
bash t4.sh 100 200      # or, manually:
uv run python -m diana.train --resume_from runs/hazelnut-<ts> --epochs 100
```

Manual equivalents (defaults shown):

```bash
uv run python -m diana.data.download --categories hazelnut
uv run python -m diana.profiler sweep --config configs/hazelnut.toml --steps 50
uv run python -m diana.train --config configs/hazelnut.toml \
    --use_amp --cudnn_benchmark --epochs 100
uv run python -m diana.sample --run runs/hazelnut-<ts> --pixel --limit 200
```

Each run stores its canonical result in `eval.json` and `evaluations/final.json`,
using the checkpoint's configured `eval_t_start`. The CSV and SVG under
`evaluations/` contain **exploratory** per-`t_start` reports. The script never
promotes the highest test AUROC into the canonical result.
Set `T_STARTS="50 100 150 200"` to change the exploratory sweep.

## Evaluation protocol

- Fix `eval_t_start` before final testing, or choose it on a separate validation
  split. Existing configurations were explored on test data in earlier runs;
  retaining them does not retrospectively make those results unbiased.
- Evaluation disables dropout and uses EMA weights by default. `--limit` caps
  the number of unique images; it never repeats images to fill the limit.
- Pixel AUROC now includes **all selected test images**, including normal images.
  Reports record `pixel_scope`. Older reports used defective images only and
  must not be compared directly without recomputing them.
- The seed controls reconstruction noise. Report variability across independent
  seeds and preserve the checkpoint, configuration, code revision and sample count.
- Center cropping can remove defects; inspect transformed masks for each category.
- `sample_timesteps` specifies the grid across the full diffusion chain. The
  reconstruction uses only grid points below `t_start`, plus `t_start` itself;
  250 grid points at `t_start=150` means 39 denoiser calls, not 250.

## Training, recovery and profiling

- Training shuffles samples by epoch. Eager training includes the final small
  batch and correctly normalizes incomplete accumulation groups. Fixed-shape
  CUDA graph training drops the shuffled tail and rejects an empty epoch.
- Augmentation and synthetic data depend on seed, epoch and sample index,
  allowing worker restarts without losing data reproducibility. This changes
  trajectories relative to older versions. Device kernels may still be
  nondeterministic, particularly with cuDNN benchmarking enabled.
- Checkpoints are published atomically and include CPU, CUDA and MPS RNG state
  when available. On resume, precedence is CLI > TOML > checkpoint > defaults;
  incompatible model/schedule changes are rejected. Legacy checkpoints without
  MPS state cannot reproduce the original MPS stream exactly.
- `--use_cuda_graphs` captures training forward/backward. The same option on
  `diana.sample` or `diana.profiler inference` captures the denoiser forward at
  each reverse step. Noise remains outside inference capture. CPU/MPS use eager
  execution. Capture failure falls back to eager execution when recoverable;
  replay failures propagate. PyTorch graph-safe dropout RNG advances on replay;
  use zero dropout for fixed-input training parity comparisons.
- Profiling `--steps` counts measured optimizer updates for training and batches
  for inference. Training excludes two warmup updates; inference excludes one
  warmup reconstruction. CUDA and MPS are synchronized at timing boundaries.
  Reports include images/s and effective graph/backend settings. CUDA memory is
  the interval peak allocation, MPS memory is the current allocation, and CPU RSS
  is the process-lifetime peak; `memory_scope` identifies the measurement.
- On the default linear schedule, alpha-bar is approximately 0.895 at t=100
  and 0.0778 at t=500. Increasing `t_start` removes more input structure; select
  its value through the evaluation protocol above.

GPU execution and final anomaly-detection performance require a real accelerator
run. Passing CPU tests alone is not evidence of CUDA speedups or model accuracy.
