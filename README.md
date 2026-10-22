# DIANA — (Diffusion-based Industrial Anomaly Analysis)

DIANA is a PyTorch project for detecting defects in industrial images from
[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). A
Denoising Diffusion Probabilistic Model (DDPM) is trained on defect-free images
only. At test time, noise is added to an image and the model reconstructs it.
The squared difference between the input and the reconstruction provides:

- an **anomaly map**, with one score per pixel;
- an **image score**, obtained by averaging the anomaly map.

The working hypothesis is that reconstruction changes defects more than healthy
regions. It can also preserve a defect or alter a healthy region, so performance
is evaluated separately at image and pixel level using AUROC.

The experiments in the [project report](relazione.md) cover **bottle, cable,
capsule, carpet and leather**, with a separate model for each category. The
[report materials](relazione_materiali/README.md) contain the evaluation and
profiling reports used in the comparison. The [PDF version](relazione.pdf) is
the submission copy; `relazione.md` remains the editable source. The PDF uses
absolute GitHub links to code and supporting materials.

## Requirements and installation

Use **Python 3.12 or later** and [uv](https://docs.astral.sh/uv/). The dependency
versions are declared in `pyproject.toml`; `uv.lock` records the resolved versions.
CUDA is supported for NVIDIA GPUs, MPS for Apple Silicon, and CPU for development
and small checks. Full training is substantially more demanding than a smoke run.

```bash
git clone https://github.com/RiccardoMarantonio/diana.git
cd diana
uv sync
```

Alternatively, install the package with `pip install -e .`. The commands below
use `uv run` to select the project environment.

## Quick start

Run a short training check on synthetic images without downloading MVTec AD:

```bash
uv run python -m diana.train --config configs/bottle.toml \
    --data_path synthetic --epochs 2
```

To train and evaluate on real images:

```bash
uv run python -m diana.data.download --categories bottle
uv run python -m diana.train --config configs/bottle.toml
```

Training prints the directory created under `runs/`. Substitute that directory
for `runs/bottle-<timestamp>` in subsequent commands:

```bash
uv run python -m diana.sample --run 'runs/bottle-<timestamp>' --pixel
```

`--pixel` enables pixel AUROC against the defect masks; image AUROC is always
computed. Evaluation uses `best.pt`, EMA weights, and the checkpoint's configured
seed, `eval_t_start` and `sample_timesteps` unless explicitly overridden.

## Demo: two defective and two healthy images

For a complete demo with the existing checkpoints and dataset, run:

```bash
bash demo.sh
```

The script selects the most recently modified `best.pt` under `runs/bottle-*/`,
selects the first two defective and first two healthy images in sorted path order,
and calibrates on up to 10 other healthy images from `test/good`. All four inputs,
including byte-identical copies, are excluded from calibration. Selection happens
before inference and is independent of scores. These references are reserved for this demo's
calibration; this workflow is **not an independent test-set evaluation**. The
default is deliberately small for a quick demo, not a statistically reliable
threshold estimate. It does not train a model or download data.

Results are written to `demo/bottle/`: `threshold.json`, PNG and JSON files for
`defective_1`, `defective_2`, `healthy_1` and `healthy_2`, plus `summary.png` (all
four visualizations) and `summary.json`. The terminal compares expected and
predicted labels, scores, and the number of correct predictions. Misclassifications
are reported as such; the threshold is not adjusted to make these examples pass.
All four predictions use the same threshold. Four examples are not an accuracy
estimate for the dataset. The saved threshold is reused on subsequent runs. To choose
a single image instead, another category, or recalibrate:

```bash
bash demo.sh --image data/mvtec/bottle/test/good/000.png --recalibrate
bash demo.sh --category capsule
bash demo.sh --recalibrate --device cpu
```

The prediction images (including byte-identical copies) are excluded when
calibrating. If any target was already used by the saved calibration, the script
automatically recalibrates excluding all current targets. After changing the
model or device, use `--recalibrate`.
For a specific model and a separate, known healthy reference set:

```bash
bash demo.sh --run 'runs/bottle-<timestamp>' \
    --healthy-dir path/to/held_out_healthy_bottles \
    --reference-limit 40 --image path/to/bottle.png
```

`--healthy-dir` and `--reference-limit` force recalibration. Other options include
`--out-dir`, `--device` and `--help`. Relative paths are resolved from the project
directory, even if the script is launched elsewhere. Missing data or checkpoints
produce an error with instructions.

All demo heatmaps use a **shared absolute color scale**, from zero to the maximum
pixel error across the selected images, with the same numerical legend on each
panel. The same color therefore represents the same squared error in all four
images. This range is only for visualization; classification still compares the
original mean error with the calibrated image threshold. Reports record the
range under `heatmap_scale`. Each prediction also saves an `.npz` file containing
the full-precision input crop, reconstruction and anomaly map, allowing the
summary to be redrawn without rerunning inference.

`diana.predict` accepts a single image of the checkpoint's category and prints
**SANA / DIFETTOSA**, its anomaly score and the threshold used. It saves a labelled
PNG with **input center crop → reconstruction → anomaly map**, plus a JSON report.
No dataset or ground-truth mask is required for prediction.

A binary decision requires a threshold. Calibrate it once on a directory of
**known healthy images held out from training and final testing**, with matching
category, framing and acquisition conditions:

```bash
uv run python -m diana.calibrate \
    --run 'runs/bottle-<timestamp>' \
    --images path/to/held_out_healthy_bottles \
    --out demo/bottle-threshold.json
```

The directory is searched recursively for PNG/JPEG/BMP images. The default
threshold is the empirical 95th percentile of healthy reconstruction scores
(`--quantile 0.95`, with the higher observation used for interpolation).
At least two distinct files are required; use a larger, representative reference
set (preferably dozens of images). This percentile does **not** guarantee a 5%
false-alarm rate on future images, and calibration on training images is biased.
Optional `--limit N` caps the references and repeatable `--exclude-image FILE`
options exclude prediction images and byte-identical copies before selecting that subset.

Then classify a separate image:

```bash
uv run python -m diana.predict \
    --run 'runs/bottle-<timestamp>' \
    --image data/mvtec/bottle/test/broken_small/000.png \
    --calibration demo/bottle-threshold.json \
    --out demo/bottle-prediction.png \
    --report demo/bottle-prediction.json
```

An image is classified as defective when `score > threshold`; equality is
classified as healthy. The score is the same mean squared reconstruction error
used by evaluation, **not a probability**. To supply an existing threshold
manually, replace `--calibration ...` with `--threshold YOUR_NUMERIC_THRESHOLD`.
One of these two options is required; the command never invents a cutoff from
the image being classified or from AUROC.

Both commands accept `--device auto|cpu|mps|cuda`, `--tag best|last`, `--seed`,
`--t_start`, `--num_steps` and `--no_ema`. By default they use the checkpoint's
seed, noise level, sampling grid and EMA weights, and select an available device
on the current machine. Calibration records the checkpoint SHA-256 and scoring
settings; prediction rejects incompatible settings and images whose file hash
appears in the reference set. Use matching options in both commands, or
recalibrate after changing the checkpoint, device or sampling settings. Each
image uses batch size 1 and a fresh reset to the same seed, so its score does not
depend on reference order. Device kernels can still introduce nondeterminism.

The verdict concerns the **center crop actually seen by the model** (64 × 64
with the supplied configurations); defects outside it cannot be detected.
Standalone predictions show an absolute zero-to-maximum scale for that image;
the demo redraws them with the range shared across its inputs. This is a thresholded model
prediction and can produce both missed defects and false alarms.

## Complete workflow

`t4.sh` downloads or verifies the data, profiles synthetic training, trains the
model, evaluates it at the configured noise level, runs an exploratory noise-level
sweep, profiles inference, and generates a CSV table and SVG chart.

Its arguments are the **total number of epochs**, the **maximum number of test
images**, and the **configuration file**. Pass the configuration explicitly to
select the category:

```bash
bash t4.sh 100 200 configs/bottle.toml
bash t4.sh 3 32 configs/bottle.toml  # short workflow check
```

To run exactly the five categories used in the report:

```bash
for category in bottle cable capsule carpet leather; do
    bash t4.sh 100 200 "configs/$category.toml" || break
done
```

To process every TOML configuration present in `configs/`, use either command:

```bash
bash t4.sh 100 200 all
bash run_all_configs.sh 100 200
```

Both stop on the first failed category. These commands follow the contents of
`configs/`, which may include categories outside the report.

### Device and execution options

| Environment variable | Default | Effect |
|---|---|---|
| `DEVICE` | `auto` | Select `auto`, `cuda`, `mps` or `cpu` |
| `FRESH` | `0` | Set to `1` to start a new training run instead of resuming |
| `T_STARTS` | `100 150 200` | Noise levels for the exploratory evaluation sweep |
| `CUDA_GRAPHS` | `0` | Set to `1` to request CUDA graphs during training |

With `auto`, device selection prefers CUDA, then MPS, then CPU. An explicit
`DEVICE=cuda` request fails before downloading data if CUDA is unavailable.
For example, after selecting a GPU runtime in Colab:

```bash
DEVICE=cuda bash t4.sh 100 200 configs/bottle.toml
DEVICE=cuda CUDA_GRAPHS=1 bash t4.sh 100 200 configs/bottle.toml
FRESH=1 T_STARTS="50 100 150 200" bash t4.sh 100 200 configs/bottle.toml
```

The script requests AMP and cuDNN benchmarking for training. AMP uses FP16 only
on CUDA; CPU and MPS use the regular execution path. When `DEVICE=cuda` is
explicitly set, the workflow also benchmarks inference with CUDA graphs in
addition to eager inference. With `DEVICE=auto`, it only runs the eager inference
benchmark, even if CUDA is selected automatically.

`data/` is ignored by Git. The downloader verifies existing files and repairs
incomplete downloads. The scripts operate on the current checkout and do not
pull updates from GitHub.

## Configuration and training recovery

Settings are defined in [src/diana/config.py](src/diana/config.py). TOML keys
match the configuration fields. For a new run, precedence is **CLI > TOML >
defaults**; when resuming, it is **CLI > TOML > checkpoint > defaults**.

The five category configurations share these main settings:

| Setting | Value |
|---|---|
| Input | RGB, 64 × 64 pixels after preprocessing |
| U-Net | Base width 64; channel multipliers 1, 2, 4, 8 |
| Residual blocks | 2 per encoder level, 3 per decoder level |
| Attention | 16 × 16 resolution, 4 heads |
| Diffusion | 1000 timesteps; linear beta schedule from 0.0001 to 0.02 |
| Training | 100 epochs; batch size 16; AdamW |
| Learning rate | 0.0001; linear warm-up over 1000 optimizer steps |
| Weight decay / gradient clipping | 0.000001 / maximum gradient norm 1.0 |
| Dropout / EMA decay | 0.1 / 0.9999 |
| Evaluation | `eval_t_start=150`; `sample_timesteps=250`; seed 42 |

A training run saves `config.json` and model checkpoints:

- **`best.pt`** stores the epoch with the lowest mean training loss. It is the
  default for evaluation; it is not selected by test AUROC.
- **`last.pt`** is saved every 25 epochs and at the end, for recovery.

Checkpoints include model and optimizer state, AMP scaler, EMA weights,
training counters, and available CPU/CUDA/MPS random-generator states. Writes
are atomic. Resume accepts a run directory or checkpoint path and rejects
incompatible changes to the category, model or diffusion schedule:

```bash
uv run python -m diana.train \
    --resume_from 'runs/bottle-<timestamp>' --epochs 100
```

`--epochs 100` means 100 total epochs, including those already completed.
The workflow script automatically finds the latest `last.pt` for the category
unless `FRESH=1` is set. Manual resume without a TOML file retains the saved
configuration apart from explicit CLI overrides.

## Evaluation protocol

Images are converted to RGB, resized to 128 × 128 pixels, center-cropped to
64 × 64, and normalized to [-1, 1]. Training additionally uses random horizontal
flips with probability 0.5. Masks receive the same geometric transformation,
with nearest-neighbor interpolation and binary output. Training uses only
`train/good`; test labels and masks are used only for evaluation.

The anomaly map is the squared reconstruction error averaged over the three
RGB channels. The image score is its spatial mean. **Image AUROC** compares
image scores with healthy/defective labels; **pixel AUROC** compares map values
with masks across all selected images, including healthy images. An AUROC of
0.5 corresponds to random ordering and 1.0 to perfect ordering. Neither metric
specifies an operating threshold or its false-alarm rate.

For reproducible comparisons:

- Choose `eval_t_start` before final testing, preferably on a separate validation
  split. Existing configurations were explored on test data in earlier runs;
  keeping their values does not make those results an independent evaluation.
  This limitation also applies to the configured results in the project report;
  the exploratory sweep is presented separately and does not select the final
  result by its highest test AUROC.
- Preserve the checkpoint, configuration, code revision, sample count, device,
  EMA setting and seed. Report variability across independent seeds. Evaluation
  disables dropout; the seed controls reconstruction noise. Device kernels may
  still be nondeterministic, especially with cuDNN benchmarking enabled.
- Inspect transformed masks. Center cropping can remove a defect while the
  image retains its original defective label.
- Compare matching pixel scopes. Older reports used defective images only;
  current reports record `pixel_scope="all_test_images"`.

`--limit` caps the number of unique test images without repeating samples.
All five reported categories have fewer than 200 test images, so the complete
workflow evaluates their full test sets with the default limit.

`sample_timesteps` defines a grid across the full 1000-step diffusion chain.
Reconstruction starts at `t_start` and uses only the grid points below it.
With 250 grid points and `t_start=150`, this requires **39 denoiser calls**.
Increasing `t_start` allows larger changes but can also distort healthy regions.

## Output files

The complete workflow produces:

| Location | Contents |
|---|---|
| `runs/<category>-<timestamp>/config.json` | Configuration snapshot |
| `runs/<category>-<timestamp>/best.pt`, `last.pt` | Training checkpoints |
| `runs/<category>-<timestamp>/eval.json` | Evaluation at the configured `eval_t_start` |
| `runs/<category>-<timestamp>/evaluations/final.json` | Copy of that evaluation |
| `runs/<category>-<timestamp>/evaluations/eval_t*.json` | Exploratory sweep reports |
| `runs/<category>-<timestamp>/evaluations/evaluation.csv`, `evaluation.svg` | Sweep table and chart |
| `runs/<category>-<timestamp>/anomaly_best_t*.png` | Input, reconstruction and heatmap grids |
| `runs/<category>-<timestamp>/profiling/inference_eager.json` | Eager inference benchmark |
| `runs/<category>-<timestamp>/profiling/inference_cuda_graphs.json` | Inference graph benchmark when explicitly running with `DEVICE=cuda` |
| `runs/<category>-prof-<timestamp>.json` | Synthetic training sweep |

The sweep does not replace `eval.json` or `evaluations/final.json` with the
highest test AUROC. Heatmap colors are normalized separately per image, so
colors cannot be used to compare absolute anomaly scores across images.

## Profiling and CUDA graphs

Run profiling independently with:

```bash
uv run python -m diana.profiler sweep --config configs/bottle.toml --steps 50
uv run python -m diana.profiler inference --config configs/bottle.toml \
    --run 'runs/bottle-<timestamp>' --steps 5 --out runs/bottle-inference.json
```

The training sweep compares FP32/AMP with cuDNN benchmarking off/on on CUDA;
it measures one regular variant on CPU or MPS. It uses synthetic data and
excludes two warm-up optimizer updates. `--steps` counts measured optimizer
updates for training and batches for inference. Inference excludes one warm-up
reconstruction. Neither measurement is the total training time on MVTec AD.

Reports include throughput and memory scope. CUDA memory is the peak allocation
in the measured interval; MPS memory is the current allocation; CPU RSS is the
process-lifetime peak. CUDA and MPS are synchronized at timing boundaries.
The individual `train` profiling command also supports operator tables from
`torch.profiler`; use `--no_profiler` to disable them. Inference benchmarks use
synthetic input tensors, optionally with trained checkpoint weights, and do not
measure dataset loading or AUROC.

`--use_cuda_graphs` captures training forward/backward or the denoiser forward
in inference. Inference noise generation stays outside capture. CPU and MPS
use eager execution. Recoverable capture failures fall back to eager; replay
failures propagate. Fixed-shape graph training drops the final incomplete
batch, while eager training includes it. Use zero dropout for fixed-input
training parity checks because dropout randomness advances during graph replay.

## Project layout and development checks

```text
src/diana/
  config.py       Configuration, TOML and CLI parsing
  data/           Download verification, datasets, transforms and data loaders
  diffusion/      Noise schedules, DDPM loss and reconstruction
  models/         U-Net and EMA weights
  train.py        Training, checkpoint saving and recovery
  sample.py       Anomaly maps, image/pixel AUROC and image grids
  prediction.py   Shared single-image scoring and labelled visualization
  demo_summary.py Demo comparison with shared heatmap scale and numeric legends
  calibrate.py    Healthy-reference threshold calibration
  predict.py      Single-image healthy/defective demo
  plot_eval.py    CSV tables and SVG charts from evaluation reports
  profiler.py     Training and inference benchmarks
  utils/          Device selection, AMP and CUDA graphs
configs/          Per-category TOML configurations
tests/            Unit, integration and accelerator regression tests
relazione.md      Project report in Italian
relazione_materiali/  Reports and charts cited in the project report
```

```bash
uv run pytest tests/ -q
uvx ruff check src tests
uvx ty check src
```

Accelerator-specific tests skip when the required device is unavailable.
Augmentation and synthetic data depend on seed, epoch and sample index so
restarting workers preserves their reproducibility. Older checkpoints without
MPS random-generator state cannot reproduce the original MPS stream exactly.
