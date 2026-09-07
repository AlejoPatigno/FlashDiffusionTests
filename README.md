# FlashDiffusionTests

Exact all-pairs DMAP benchmark on Modal T4. Public upstream FlashDiffusion is
pinned to f5c2d1abb7e418829651f7e2fedbb8b323fd56f9; the inaccessible original
FlashDiffusionTests repository supplied no code or claimed results.

## Dense reference by blocks

The default is `--dense-mode batched --block-size 2048`. Dense Gaussian blocks
are materialized, reduced and discarded. All N samples interact with all N
samples: there is no subsampling, kNN sparsification or Nystrom approximation.
Two reductions compute Coifman-Lafon normalization; each ARPACK matvec then
recomputes the blocks. Operator storage is O(N + block_size^2), while the solver
also stores O(N*krylov_size) vectors. Work remains quadratic and can be slow.

The plot explicitly labels this method **DMAP dense (batched)**. Its timing is
not interchangeable with the old **dense (full)** baseline, which stored the
entire matrix. `--dense-mode full` remains available for small reference checks.
CPU uses float64, CUDA dense blocks use float32, and Flash uses its upstream
mixed-precision kernel. Both methods share data, normalization and ARPACK settings.

## Timing and deadlines

`--timeout` applies only to dense workers (120 seconds in the workflow).
`--flash-timeout 0` disables the Flash subprocess deadline; it no longer stops
at 120 seconds. ARPACK convergence/iteration criteria remain in effect.
Modal still imposes a 24-hour limit on the complete remote invocation, including
preflight and all requested trials. Unlimited compute is not claimed.
N=10,000,000 is an attempt, not a demonstrated supported completion size.

## Workflow

Create repository Actions secrets MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
Never commit credentials. The YAML runs CPU numerical validation and the full
matrix memory guard, deploys the Modal function, then submits it asynchronously.
Exactly one T4 is requested and checked (SM75). The pinned upstream WMMA source
is unchanged; a checked JIT architecture entry compiles it for SM75.

The remote function gates large work on GPU preflight at N=100 and 1000.
It then attempts 100, 1000, 10000, 100000, 1000000 and 10000000, three repetitions
by default. No subprocess deadline is applied to Flash. Failures and unfinished
trials never receive invented times.

GitHub Actions finishes after submission, not after GPU completion. Its artifact
`dmap-scaling-modal-t4` initially contains submission.json with the Modal call ID
and run directory. A green submission job does NOT mean numerical GPU success.
The remote worker periodically commits partial files to the Modal volume
`flashdiffusion-t4-results`, under that unique run directory. Completed outputs
include JSON/CSV, PDF/PNG, logs, hardware details, provenance and a ZIP.

To retrieve results later, run the same workflow manually with collect_call_id
set to the recorded call ID. It checks once and downloads the finished ZIP, or
reports pending without waiting or launching another GPU run. Completed call
results are retained by Modal for a limited time; the volume preserves the files.
If the 24-hour platform limit interrupts the invocation, retrieve partial files
from its run directory in the volume; there may be no completed ZIP.

Pushes affecting benchmark files on main submit a new GPU run after CPU checks.
The benchmark branch only performs CPU checks. Manual main runs with run_gpu
also submit. Modal runs incur the account's GPU usage; max_containers=1 and
retries=0 prevent parallel GPU workers and automatic computation retries.
