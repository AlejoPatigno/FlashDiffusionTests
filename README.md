# FlashDiffusionTests

Independent dense DMAP versus FlashDiffusion benchmark. Repository: `AlejoPatigno/FlashDiffusionTests`.

## Install

Use Python 3.11 or newer. Install PyTorch appropriate for your hardware, then:

```bash
python -m pip install "flashdiffusion @ git+https://github.com/sparsetrace/FlashDiffusion.git@f5c2d1abb7e418829651f7e2fedbb8b323fd56f9" matplotlib psutil
```

The upstream commit is pinned; this repository does not modify FlashDiffusion.
For CPU only, PyTorch can be installed with:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Execute

Small validation:

```bash
python benchmarks/benchmark_dmap_scaling.py --device cpu --sizes 100 1000 --repeats 3 --output smoke_results
```

Full experiment (supported NVIDIA GPU, CUDA-enabled PyTorch and nvcc required):

```bash
python benchmarks/benchmark_dmap_scaling.py --device cuda --sizes 100 1000 10000 100000 1000000 --repeats 3
```

Outputs: timings.pdf, timings.png, timings.csv, results.json and per-trial logs.
Three repetitions are summarized with the median and min-max envelope. Both
methods share ARPACK and its settings. The timed region includes construction,
normalization, transfers, eigenpairs and embedding; imports/JIT, data generation,
small warm-up and validation are excluded. CUDA synchronization brackets timing.
This measures dense versus matrix-free operators with a common CPU eigensolver,
not the separate native GPU Lanczos path. Precision differs between backends;
residuals and small-case agreement checks are recorded.

A dense float32 matrix at N=1,000,000 alone needs 4 TB. A conservative workspace
budget screens infeasible cases; no timing is fabricated for omitted, timed-out,
failed or numerically invalid trials. Matrix-free computation remains quadratic
in arithmetic. See the Python module docstring for full protocol details.

## GitHub Actions and Modal T4

`.github/workflows/benchmark-dmap-scaling.yml` runs the CPU tests first and then
launches Modal from an ordinary `ubuntu-latest` runner. The remote function in
`benchmarks/modal_t4.py` requests exactly one **T4**, asserts its actual GPU name
and SM75 capability, and never substitutes another GPU or silently falls back
to CPU. No self-hosted GPU runner is required.

Create the following repository **Actions secrets** before running the GPU job:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Use Settings > Secrets and variables > Actions. Credentials never belong in
source files, workflow inputs, logs or commits. Pushes to main affecting benchmark
files launch the pipeline. Alternatively select Actions > Dense DMAP versus
FlashDiffusion scaling > Run workflow, enable run_gpu, and set repetitions and
trial timeout. Missing authentication causes an explicit failure, not a success.

Modal first validates N=100 and 1000 using both GPU operators. Only if all four
preflight trials pass does it attempt N=100, 1000, 10000, 100000, 1000000 and
10000000. Defaults are three repetitions and 120 seconds per subprocess; inputs
are limited to 1-3 repetitions and 1-300 seconds. The deadline includes process
startup/warm-up, whereas reported timings exclude setup. A timeout is not a
measurement and produces no point on the timing curve. Modal tasks consume the
configured account's GPU resources; retries are disabled.

The T4 is SM75. The pinned upstream JIT builder does not contain an SM75 entry;
this project's Modal image adds that architecture mapping to compile the SAME
upstream WMMA kernel for sm_75. The CUDA source and mathematical operator are
unchanged. The patch is checked against the expected source layout, and real
GPU preflight checks are required before scaling. Compilation or numerical
failure stops large-N work and preserves the diagnostic files.

The image pins CUDA 12.4.1, PyTorch 2.5.1/cu124 and the upstream commit. Artifacts
include GPU identification, package versions, the CUDA source hash, benchmark
commit, trial statuses and output figures. GPU artifacts are uploaded as
`dmap-scaling-modal-t4`. CPU validation artifacts retain their separate name.

N=10,000,000 is an experimental attempt, not a demonstrated supported size.
One dense FP32 matrix at this size alone requires 400 TB. FlashDiffusion avoids
that matrix but still evaluates quadratic interactions, so it may exceed the
configured deadline on a T4.

## Provenance and validation

This is a new benchmark project. No files or results from the inaccessible
`sparsetrace/FlashDiffusionTests` repository were used. The dependency is the
public `sparsetrace/FlashDiffusion` library at the commit pinned above.

The YAML itself runs 12 numerical CPU trials (N=100 and 1000, both methods,
three repetitions) followed by a memory-screening check. Consult the actual
Actions run and Modal artifacts for their outcomes. CPU checks are not GPU
measurements; adding a T4 build configuration is not evidence that it has run.
