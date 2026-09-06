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

## GitHub Actions

Pushes to main or benchmark/dense-vs-flash-scaling run CPU smoke tests and upload
their results. For all five sizes, select Actions > Dense DMAP versus
FlashDiffusion scaling > Run workflow and enable run_gpu. Register a runner with
labels self-hosted, linux, x64, gpu, and install the NVIDIA driver, matching CUDA
toolkit and CUDA-enabled PyTorch before launching it. Without this runner the
GPU job will wait in the queue. GPU execution is opt-in; ordinary hosted CPU
runners only execute the small validation.

## Provenance and validation

This is a new benchmark project. No files or results from the inaccessible
`sparsetrace/FlashDiffusionTests` repository were used. The dependency is the
public `sparsetrace/FlashDiffusion` library at the commit pinned above.

The YAML itself runs 12 numerical CPU trials (N=100 and 1000, both methods,
three repetitions) followed by a memory-screening check. It asserts the trial
statuses and preserves JSON, CSV, PDF, PNG, logs and environment metadata as
GitHub Actions artifacts. Consult the actual Actions run for its outcome.
Local preparation results are not presented as GitHub or GPU results.

Full GPU scaling remains a separate workflow job, enabled with run_gpu.
Missing points due to memory screening or timeout are explicitly reported.
