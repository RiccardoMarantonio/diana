# diana

Unsupervised **diffusion-based anomaly detection** on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(hazelnut, 64x64): train a DDPM on defect-free images only; at test time an
image is noised to an intermediate level (`t_start`) and re-imagined by the
model — an unseen defect is not in the training distribution, so it is healed
away and the squared reconstruction residual is the anomaly score.

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
tests/                 90+ tests; run with pytest, ruff and ty (both clean)
```

## Quick start

```bash
uv sync                                   # or: pip install -e .
uv run pytest tests/ -q                   # 92 passed, 3 skipped
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
Apple Silicon) and self-contained — benchmark, train, then sweep `t_start`
and write the pixel-level evaluation:

```bash
git clone https://github.com/RiccardoMarantonio/diana.git && cd diana
bash t4.sh              # 100 epochs, 200-sample eval (Colab)
bash t4.sh 3 32         # local smoke: 3 epochs, 32-sample eval
```

`data/` is gitignored, so the script re-downloads it on a fresh machine. If a
session dies mid-training, resume (the config snapshot lives in the run dir):

```bash
bash t4.sh 100 200      # or, manually:
uv run python -m diana.train --config configs/hazelnut.toml --resume_from runs/hazelnut-<ts>
```

Manual equivalents (defaults shown):

```bash
uv run python -m diana.data.download --categories hazelnut
uv run python -m diana.profiler sweep --config configs/hazelnut.toml --steps 50
uv run python -m diana.train --config configs/hazelnut.toml \
    --use_amp --cudnn_benchmark --epochs 100
uv run python -m diana.sample --run runs/hazelnut-<ts> --pixel --limit 200
```

Notes:

- **CUDA graphs** (`--use_cuda_graphs`) capture forward+backward as one graph;
  dropout masks freeze at capture, so use a `dropout = 0` config for parity
  with eager training.
- **`t_start`** is the healing dial. On the 1000-step linear schedule the
  useful range is ~t in [50, 200] (ᾱ ≈ 0.37 at t=100, ≈ 0.007 at t=500);
  `configs/hazelnut.toml` pins 150 pending the T4 run.