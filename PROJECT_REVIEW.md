# DIANA project review — 2026-10-13

Original verdict: request changes before relying on the reported evaluation and HPC comparisons.

## Remediation status

The fourteen numbered findings below are retained as the original review record.
Their code fixes and behavioral regressions have now been implemented: evaluation
mode/limits/grids, graph gradient lifetime and fallback, complete accumulation,
shuffled sample coverage, effective benchmark settings and step counts, platform
memory units, MPS RNG checkpoints, download size checks/recovery, and separation
of canonical evaluation from exploratory test sweeps.

Additional changes include atomic checkpoint publication, resume configuration
loading, stateless epoch/sample augmentation, accelerator timing synchronization,
images/s reporting, fewer training host synchronizations, inference denoiser CUDA
graphs, all-test-image pixel AUROC, and corrected README protocol/schedule values.

CUDA/MPS execution and new scientific results remain to be validated on suitable
hardware. Existing evaluation artifacts were not rewritten. The user-owned
`relazione.md` draft is intentionally not included in these implementation commits;
its author/course details and final experimental results cannot be invented.

Post-fix verification: **265 passed, 7 skipped** in the complete suite, including
the persistent-worker and shell integration regressions. Ruff, ty and shell
syntax checks pass. The suite ran with shared-memory access because the sandbox
blocks PyTorch's worker shared-memory manager. CUDA-dependent regressions remain
skipped; no CUDA performance claim is made.

The verification numbers below describe the original review, before remediation.

Scope: the current working tree, all Python source and tests, six TOML configurations, the shell workflow, packaging, README, project instructions, and the draft report. This is a whole-project review, so findings include existing defects, not only recent changes. No implementation files were changed.

## Verification and limits

- Existing suite: **234 passed, 3 skipped in 42.72 seconds** with `.venv/bin/python -m pytest tests -q`.
- `bash -n t4.sh` passed.
- Focused, temporary reproductions checked inference dropout, repeated evaluation indices, partial grids, incomplete accumulation groups, capture-failure control flow, benchmark flags, truncated downloads, profiler step limits, and gradient-buffer lifetime across epochs.
- Runtime: PyTorch 2.13.0; CUDA and MPS both unavailable to this process. CUDA findings below distinguish code/control-flow evidence from actual accelerator execution.
- Inspected the locally available hazelnut dataset: 391 training images, 110 test images, 70 defective test images. None of those 70 masks became completely empty after the current crop.
- No full training run, accelerator benchmark, clean dependency installation, live remote-download integration test, or dependency vulnerability audit was performed. Ruff and ty were not installed in the project environment; their README claims were not revalidated.
- The Code Review plugin exposes a PR CI-diagnostics tool; no PR was supplied, so this local project review used repository inspection and executable checks.

P1 means high priority; P2 means a concrete defect or methodological issue that should be fixed in its affected workflow.

## Findings

### 1. P1 — Inference leaves dropout enabled

Location: `src/diana/sample.py:137–141`.

The loaded model remains in training mode. Neither `run_inference` nor `DDPM.reconstruct` calls `eval()`. All supplied configurations have dropout 0.1, so each denoising step randomly drops activations while producing evaluation scores. `no_grad()` does not disable dropout. This also makes the evaluation path differ from the profiler, which explicitly calls `model.eval()`.

Evidence: hooked every Dropout forward during a complete temporary inference run; every observed module had `training=True`.

Fix: set the model to evaluation mode before reconstructing, and add an inference-mode regression test. Recompute evaluation results afterward.

### 2. P1 — New epochs detach gradients from CUDA graph replay

Location: `src/diana/train.py:186`.

`train_epoch` always calls `optimizer.zero_grad(set_to_none=True)` at entry. The captured graph survives across epochs and writes to its captured gradient buffers. At epoch two, parameters no longer reference those buffers; the later `set_to_none=False` calls cannot restore references that are already `None`. The optimizer therefore sees missing gradients, even though graph replay continues. With AMP, the missing gradients can also lead to scaler assertions.

Evidence: a CPU stand-in that writes to fixed captured gradient buffers updated the parameter in epoch one, but epoch two left `param.grad=None` and the parameter unchanged. Actual CUDA replay was unavailable.

Fix: preserve gradient storage for the lifetime of an active graph, and test at least two complete epochs on CUDA with and without AMP.

### 3. P2 — CUDA capture failure does not fall back

Location: `src/diana/utils/cudagraphs.py:84–87` and `118–123`.

`_capture` catches failures, disables capture, clears buffers, and returns. Its caller immediately asserts that the graph and buffers exist. Thus every caught capture error terminates the run with `AssertionError`, despite the documented fallback.

Evidence: simulated a failed capture by disabling the wrapper in `_capture`; `step` raised `AssertionError`.

Fix: after capture, branch on success; reset partial gradients and execute a fresh eager forward/backward when recovery is possible. Test this failure path independently of CUDA availability.

### 4. P2 — Evaluation limits duplicate and unevenly weight images

Location: `src/diana/sample.py:149`.

`linspace(..., limit).astype(int)` always returns `limit` entries, even when the dataset is smaller. The default shell run requests 200 images from the locally available 110-image hazelnut test split. Images are reconstructed repeatedly with different noise, some images receive more repeats than others, and the report presents the repeated count as the test-image count. This changes the metric rather than merely limiting work.

Evidence: a four-image fixture evaluated with limit eight returned `n=8`. Its selected indices repeat images unevenly.

Fix: validate a positive limit and cap it at the dataset length. If repeated reconstructions are desired, expose an explicit repeat count and aggregate per original image before computing image AUROC.

### 5. P2 — Partial visualization grids abort otherwise valid evaluation

Location: `src/diana/sample.py:121–123`.

The first concatenation creates columns of up to four samples. For five, six, or seven shown samples, the last column is shorter, and the final horizontal concatenation fails. For example, `--limit 6` can compute the scores and then fail before writing the report.

Evidence: six sample triplets raise NumPy `ValueError` because column heights are 32 and 16.

Fix: pad the final column or construct a fixed-size canvas. Cover five, six, and seven samples.

### 6. P2 — Incomplete gradient accumulation groups are discarded

Location: `src/diana/train.py:220–240`.

An optimizer step occurs only when `micro % accum == 0`. Remaining gradients are never applied at the end of an epoch and are cleared at the next epoch's entry. If the epoch contains fewer microbatches than the accumulation factor, no learning occurs and EMA remains uninitialized. The reported mean loss nevertheless includes those unused batches.

Evidence: three microbatches with accumulation two yielded one optimizer step and an unapplied gradient of 0.5.

Fix: flush the final partial group with correct normalization, or explicitly reject incompatible settings. Test both a remainder and an epoch shorter than the accumulation factor.

### 7. P2 — The same training images are excluded every epoch

Location: `src/diana/data/loader.py:55–66`.

The loader uses `drop_last=True` without shuffling. MVTec paths are sorted, so the identical tail is permanently excluded from training. At the supplied batch size of 16, the local 391-image hazelnut split always omits seven images. A dataset smaller than one batch silently produces an empty epoch and a zero loss.

Fix: shuffle training samples with a reproducible sampler; retain partial batches where compatible, and reject an empty loader. For fixed-shape CUDA graphs, shuffling at least rotates the dropped tail.

### 8. P2 — The cuDNN benchmark sweep never changes cuDNN

Location: `src/diana/profiler.py:210–214`.

The sweep changes `cfg.cudnn_benchmark`, but neither the loop nor `profile_training` applies that flag. `main` configures the global backend once before entering the sweep. Consequently rows labeled benchmark-on and benchmark-off execute with the same backend setting, invalidating that comparison.

Evidence: a stubbed training callback observed requested/actual flags `(False, False), (True, False), (False, False), (True, False)`.

Fix: configure the backend for every variant, report effective settings, and restore prior settings afterward.

### 9. P2 — Profiling silently stops before the requested step count

Location: `src/diana/profiler.py:96–98`.

`islice(loader, steps)` can only consume one pass through the 512-image synthetic dataset. The supplied batch size 16 gives 32 microbatches, so the shell's `--steps 50` does not run 50. With accumulation, the mismatch with the CLI description of optimizer steps is larger. Short runs also give startup overhead disproportionate weight.

Evidence: requested 50 steps at batch size 32 and accumulation one; the report contained only 16.

Fix: define whether the argument counts microbatches or optimizer updates, iterate enough data to satisfy it, and exclude a deliberate warmup interval from measurement.

### 10. P2 — macOS CPU memory is overstated by 1024×

Location: `src/diana/profiler.py:60–61`.

The conversion guesses units from the magnitude of `ru_maxrss`. On macOS it is bytes, so normal process sizes below 1 TiB take the wrong branch and divide by 1024 rather than 1024². The platform comment is reversed as well.

Evidence on this macOS host: raw RSS 297,287,680 bytes, approximately 283.52 MiB, was reported as 290,320 MiB.

Fix: select conversion by platform. Also label CPU RSS as a process-lifetime peak rather than implying a resettable per-variant measurement.

### 11. P2 — MPS random state is missing from checkpoints

Location: `src/diana/train.py:103–110` and `129–133`.

The comment claims CPU RNG state covers MPS, but the installed PyTorch implementation exposes a separate default MPS generator through `torch.mps.get_rng_state()` and `set_rng_state()`. Training draws noise and timesteps on the model's device. Restoring CPU and CUDA state therefore does not resume the MPS random stream.

Evidence: inspected the installed PyTorch MPS RNG implementation. No MPS execution was possible.

Fix: save and restore MPS RNG state when relevant, with backward-compatible loading, and compare interrupted versus uninterrupted MPS training.

### 12. P2 — Downloaded sizes are never verified before publication

Location: `src/diana/data/download.py:120–129`.

`expected_size` is used only to decide whether an existing file needs downloading. Newly downloaded bytes are published without checking size. When no SHA256 exists in the listing, or `--no-verify` is used, a short response is accepted as complete and reported as successful.

Evidence: a one-byte mocked response with expected size 100 and no digest was saved at the final path and returned `ok`.

Fix: count received bytes and compare with the expected size before `os.replace`, independently of hash verification.

### 13. P2 — An interrupted download is skipped by the one-shot workflow

Location: `t4.sh:31–33`.

The shell considers a category complete if its directory exists. The downloader creates that directory before it finishes, so rerunning after interruption skips all missing files. Training can then silently use a partial split, and evaluation can fail or use incomplete data. The downloader already supports checking existing files, but this guard prevents it from doing so.

Fix: invoke the resumable downloader each time, or use a completion manifest written only after verification of the full plan.

### 14. P2 — Test labels select the reported best hyperparameter

Location: `t4.sh:59–62`.

The workflow maximizes image AUROC across `t_start` values measured on the test set and copies the winner to the canonical `eval.json`. If that file is treated as final test performance, the result is selected using the same labels it purports to evaluate. Training remains normal-only, but model selection is supervised by test anomalies.

Fix: select the operating point on a separate validation split or freeze it before final testing. Preserve the full sweep as exploratory results and explicitly identify the selection protocol in the project report.

## Additional design and coverage observations

- **Pixel metric scope:** pixel AUROC deliberately excludes good images (`sample.py:169–173`, also documented by CLI help). This measures localization conditional on an anomalous image and does not penalize false-positive maps on normal images. Name it accordingly, or additionally report an all-test-image pixel metric; do not present the two as interchangeable.
- **Resume reproducibility with workers:** default configurations use four persistent workers and random flips. Checkpoints do not preserve worker RNG streams, and resumed workers receive fresh seeds. The existing equivalence test uses `num_workers=0`, so it cannot establish exact resume equivalence for the supplied configurations. Stateless per-sample/per-epoch augmentation is one possible solution.
- **Checkpoint durability:** `torch.save` writes directly over `best.pt` and `last.pt`. Interruption during a save can destroy the previous checkpoint. Write a temporary sibling and replace atomically, especially for the stated ephemeral-session workflow.
- **Resume workflow documentation:** rerunning `bash t4.sh` starts a fresh training run; it never passes `--resume_from`. The README presents it as a way to resume. Implement checkpoint discovery or remove that claim. Model configuration is also supplied anew on manual resume rather than restored wholesale from the saved configuration.
- **Acceleration requirement:** the project instructions request CUDA graphs during sampling. The implementation only captures training forward/backward; sampling has no CUDA graph path. Clarify the accepted requirement or implement and measure inference capture.
- **Timing methodology:** inference synchronizes CUDA but not MPS. Training timing includes initialization and worker startup, with no explicit final device synchronization. Add warmup and device-specific synchronization before drawing HPC conclusions. Training reports updates/s rather than the images/s requested by the project instructions.
- **Avoidable host synchronization:** every training microbatch converts its loss to `float`, and every update converts the gradient norm to `float` even though the caller discards it. On accelerators these force host synchronization and limit the CPU-overhead benefit of graphs. Accumulate device-side metrics and transfer only at logging boundaries.
- **Noise-schedule documentation:** the stated linear-schedule alpha-bar at t=100 (~0.37) is incorrect for beta 0.0001→0.02 over 1000 steps; it is approximately 0.895. At t=500 it is approximately 0.0778, not 0.007. Correct README and sampling explanations before using them to justify the operating range.
- **Draft report:** `relazione.md` contains explicit placeholders, malformed inline math delimiters, and describes 250 sampling steps without distinguishing the configured full-chain grid from the shorter reconstruction chain at t_start=150. Finish it only after the evaluation and benchmark corrections above.
- **Coverage gaps:** all existing tests pass, but they omit inference dropout state, partial grid counts, limit greater than dataset size, accumulation remainders, multi-epoch CUDA graphs, capture failure, effective sweep settings, requested profile length, and download-size mismatch. The new plotting module has no dedicated tests. Add behavioral regressions alongside fixes.

## Strengths

The code has useful boundaries between configuration, data, model, diffusion, training, and profiling. Schedule buffers travel with the model and checkpoint. The respaced sampler has focused algebraic tests, EMA has explicit bias correction, checkpoint loading uses the installed PyTorch default safe-loading behavior, and remote download paths are constrained to the destination with hash checks before publication when hashes are available. These are useful foundations; the primary weaknesses are lifecycle edge cases and the validity of measurement workflows.

## Suggested repair order

1. Fix inference mode, evaluation sampling, and the CUDA graph gradient lifecycle.
2. Fix graph fallback, accumulation tails, and training sample coverage.
3. Correct benchmark settings, step counts, memory units, and timing boundaries.
4. Harden checkpoint and download recovery; add accelerator and worker resume checks.
5. Freeze a validation/test protocol, regenerate metrics, and update the README and report.
