#!/usr/bin/env python3
"""Dense DMAP versus FlashDiffusion on a reproducible Swiss roll.

Run: python benchmarks/benchmark_dmap_scaling.py --device cuda
Requires numpy, scipy, torch, matplotlib, psutil and the upstream FlashDiffusion package (see README.md).
CUDA additionally requires a supported NVIDIA GPU, CUDA toolkit/nvcc and ninja.

Both methods use K_ij=exp(-beta*||x_i-x_j||^2), alpha normalisation,
and the SAME CPU ARPACK solver, initial vector, k, tolerance and iteration cap.
This isolates dense versus matrix-free operators; it does not benchmark the
package's separate GPU Lanczos implementation. CUDA dense uses float32;
FlashDiffusion uses its native mixed precision. CPU paths use package defaults.
Time includes centering, transfers, kernel/normalisation, eigenpairs and diffusion
coordinates. Data generation, imports/JIT, small warm-up, validation and plotting
are excluded. Synchronisation brackets each measurement. Each trial is isolated
in a subprocess. Flash has no subprocess deadline by default (--flash-timeout 0). No sparse approximation, subsampling or extrapolated times.

Dense defaults to exact all-pairs block reductions, recomputed per matvec.
Use --dense-mode full to retain a complete matrix. Memory screening estimates
3*block_size**2 + 32*N elements for batched mode or 3*N*N for full mode,
NOT a measured peak or a proof of OOM. Omitted/failed trials have no timing point.
The default sizes are 100, 1000, 10000, 100000, 1000000, 10000000. Large matrix-free runs
still require quadratic arithmetic and can time out. Partial outputs are saved
after every trial. CPU mode is intended for small correctness/smoke tests.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes', nargs='+', type=int, default=[10,100, 1000, 10000, 100000, 1000000, 10000000])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--components', type=int, default=8, help='Includes the trivial eigenpair')
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--maxiter', type=int, default=1000)
    p.add_argument('--validation-tol', type=float, default=5e-3)
    p.add_argument('--timeout', type=int, default=0, help='Dense subprocess deadline in seconds; 0 disables it')
    p.add_argument('--flash-timeout', type=int, default=0, help='Flash subprocess deadline; 0 disables it')
    p.add_argument('--dense-mode', choices=['batched', 'full'], default='batched')
    p.add_argument('--block-size', type=int, default=2048)
    p.add_argument('--memory-fraction', type=float, default=0.6)
    p.add_argument('--dense-max-gib', type=float, default=0, help='Additional memory estimate cap; 0 uses available memory')
    p.add_argument('--output', type=Path, default=Path('benchmark_results'))
    p.add_argument('--worker', choices=['dense', 'flash'], help=argparse.SUPPRESS)
    p.add_argument('--trial', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if (min(a.sizes) <= a.components or a.components < 2 or a.repeats < 1
            or a.timeout < 0 or a.maxiter < 1 or a.beta <= 0 or a.tol <= 0
            or a.validation_tol <= 0 or not 0 < a.memory_fraction < 1
            or a.dense_max_gib < 0 or a.flash_timeout < 0 or a.block_size < 1):
        p.error('Invalid sizes, components, repetitions, tolerance, timeout or memory budget')
    return a



def batched_dense_operator(x, a):
    """Exact all-pairs Gaussian reductions; O(N + block_size**2) operator storage.

    Materializes dense blocks, recomputing them for every reduction/matvec.
    This changes the storage/computation tradeoff, not the DMAP operator.
    """
    import numpy as np
    gpu = a.device == 'cuda'
    if gpu:
        import torch
    z = torch.as_tensor(x, dtype=torch.float32, device='cuda') if gpu else np.asarray(x, dtype=np.float64)
    n, block = len(z), a.block_size
    sq = (z*z).sum(axis=1)
    def zeros():
        return torch.zeros(n, device='cuda', dtype=torch.float32) if gpu else np.zeros(n)
    def reduce(weights=None):
        out = zeros()
        for i in range(0, n, block):
            end_i = min(i+block, n)
            for j in range(0, n, block):
                end_j = min(j+block, n)
                k = z[i:end_i] @ z[j:end_j].T
                k *= -2
                k += sq[i:end_i, None]
                k += sq[None, j:end_j]
                if gpu:
                    k.clamp_(min=0).mul_(-a.beta).exp_()
                else:
                    np.maximum(k, 0, out=k)
                    k *= -a.beta
                    np.exp(k, out=k)
                out[i:end_i] += k.sum(axis=1) if weights is None else k @ weights[j:end_j]
        return out
    q = reduce()
    w = q ** (-a.alpha)
    degree = w * reduce(w)
    scale = w * degree ** (-0.5)
    def mv(v):
        v = torch.as_tensor(v, device='cuda', dtype=torch.float32) if gpu else np.asarray(v, dtype=np.float64)
        out = scale * reduce(scale*v)
        return out.cpu().numpy().astype(np.float64) if gpu else out
    degree_numpy = degree.cpu().numpy().astype(np.float64) if gpu else degree
    return mv, degree_numpy, ('torch-dense-batched-fp32' if gpu else 'numpy-dense-batched-fp64')


def worker(a):
    import platform
    import numpy as np
    import psutil
    import scipy
    import torch
    from scipy.sparse.linalg import LinearOperator, eigsh

    # Hide CUDA before import in CPU mode, so the package cannot select a GPU.
    from flashdiffusion.dmap import DiffusionMap
    n = a.sizes[0]
    row = dict(method=a.worker, n=n, trial=a.trial, status='error', seconds=None)
    def sync():
        if a.device == 'cuda':
            torch.cuda.synchronize()

    def data(count):
        r = np.random.default_rng(a.seed)
        u = r.random((count, 3))
        t = 1.5 * np.pi + 3 * np.pi * u[:, 0]
        x = np.column_stack([t*np.cos(t), 10*u[:, 1], t*np.sin(t)])
        return np.ascontiguousarray((x-x.mean(0))/x.std(), dtype=np.float64)

    def operator(x, method):
        if method == 'flash':
            dm = DiffusionMap(beta=a.beta, alpha=a.alpha, n_components=a.components)
            dm.fit(x)
            if a.device == 'cuda' and dm._cuda_state is None:
                raise RuntimeError('CUDA backend unavailable; refusing silent CPU fallback')
            backend = type(dm._cuda_state).__name__ if dm._cuda_state is not None else 'numpy-tiled'
            return dm.matvec, dm.D_alpha_, backend
        x = x - x.mean(0)
        if a.dense_mode == 'batched':
            return batched_dense_operator(x, a)
        if a.device == 'cuda':
            z = torch.as_tensor(x, dtype=torch.float32, device='cuda')
            sq = (z*z).sum(1)
            m = z @ z.T
            m.mul_(-2).add_(sq[:, None]).add_(sq[None, :]).clamp_(min=0)
            m.mul_(-a.beta).exp_()
            w = m.sum(1).pow(-a.alpha)
            m.mul_(w[:, None]).mul_(w[None, :])
            degree = m.sum(1)
            inv = degree.rsqrt()
            m.mul_(inv[:, None]).mul_(inv[None, :])
            def mv(v):
                return (m @ torch.as_tensor(v, dtype=torch.float32, device='cuda')).cpu().numpy().astype(np.float64)
            return mv, degree.cpu().numpy().astype(np.float64), 'torch-dense-fp32'
        sq = (x*x).sum(1)
        m = x @ x.T
        m *= -2
        m += sq[:, None]
        m += sq[None, :]
        np.maximum(m, 0, out=m)
        m *= -a.beta
        np.exp(m, out=m)
        w = m.sum(1)**(-a.alpha)
        m *= w[:, None]
        m *= w[None, :]
        degree = m.sum(1)
        inv = degree**(-0.5)
        m *= inv[:, None]
        m *= inv[None, :]
        return lambda v: m @ v, degree, 'numpy-dense-fp64'

    def solve(x, method):
        mv, degree, backend = operator(x, method)
        op = LinearOperator((len(x), len(x)), matvec=mv, dtype=np.float64)
        v0 = np.random.default_rng(a.seed+1).standard_normal(len(x))
        vals, vecs = eigsh(op, k=a.components, which='LA', tol=a.tol,
                           maxiter=a.maxiter, ncv=min(len(x), max(2*a.components+1, 20)), v0=v0)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        # P = D_alpha^-1 K_alpha: right eigenvectors = D_alpha^-1/2 u.
        psi = vecs / np.sqrt(degree[:, None])
        psi /= np.linalg.norm(psi, axis=0, keepdims=True)
        embedding = psi[:, 1:] * vals[None, 1:]
        return mv, vals, vecs, embedding, backend

    try:
        if a.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but no CUDA device is available')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        row['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda,
            device=torch.cuda.get_device_name() if a.device == 'cuda' else platform.processor(),
            platform=platform.platform())
        available = torch.cuda.mem_get_info()[0] if a.device == 'cuda' else psutil.virtual_memory().available
        budget = available * a.memory_fraction
        if a.dense_max_gib:
            budget = min(budget, a.dense_max_gib * 2**30)
        itemsize = 4 if a.device == 'cuda' else 8
        estimate = (3*n*n if a.dense_mode == 'full' else 3*min(n, a.block_size)**2 + 32*n) * itemsize
        row.update(dense_mode=a.dense_mode, block_size=a.block_size)
        row.update(dense_estimated_bytes=estimate, dense_budget_bytes=int(budget))
        if a.worker == 'dense' and estimate > budget:
            row.update(status='memory_limit', reason='Conservative dense workspace estimate exceeds configured budget')
            return row
        # A full small solve warms BLAS, CUDA kernels and ARPACK, outside timing.
        warm = solve(data(max(64, a.components+2)), a.worker)
        del warm
        gc.collect()
        if a.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        x = data(n)
        sync()
        start = time.perf_counter()
        mv, vals, vecs, embedding, backend = solve(x, a.worker)
        sync()
        elapsed = time.perf_counter() - start
        residual = max(float(np.linalg.norm(mv(vecs[:, j])-vals[j]*vecs[:, j])) for j in range(a.components))
        row.update(backend=backend, eigenvalues=vals.tolist(), residual=residual,
                   measured_seconds=elapsed, peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None)
        if n <= 1000:
            probe = np.random.default_rng(a.seed+2).standard_normal(n)
            probe /= np.linalg.norm(probe)
            row['operator_probe'] = mv(probe).tolist()
        if (not np.isfinite(embedding).all() or not np.isfinite(residual)
                or residual > a.validation_tol or abs(vals[0]-1) > a.validation_tol):
            row.update(status='invalid', reason='Eigenpair residual, leading eigenvalue or finite-value check failed')
        else:
            row.update(status='ok', seconds=elapsed)
    except (MemoryError, torch.cuda.OutOfMemoryError) as e:
        row.update(status='oom', reason=str(e))
    except Exception as e:
        row.update(status='error', reason=f'{type(e).__name__}: {e}')
    return row


def save(a, rows):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'results.json').write_text(json.dumps(dict(config={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}, trials=rows), indent=2))
    fields = ['method', 'n', 'trial', 'status', 'seconds', 'measured_seconds', 'backend', 'residual', 'dense_estimated_bytes', 'dense_budget_bytes', 'peak_cuda_bytes', 'reason']
    with (a.output/'timings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    table = []
    for method, label, color in [('dense', 'DMAP dense (batched)' if a.dense_mode == 'batched' else 'DMAP dense (full)', '#235789'), ('flash', 'FlashDiffusion', '#c44b28')]:
        med, lo, hi, status = [], [], [], []
        for n in a.sizes:
            selected = [r for r in rows if r['method']==method and r['n']==n]
            times = [r['seconds'] for r in selected if r['status']=='ok']
            med.append(np.median(times) if times else np.nan)
            lo.append(min(times) if times else np.nan)
            hi.append(max(times) if times else np.nan)
            status.append(f'{len(times)}/{a.repeats} valid' if times else ', '.join(sorted({r['status'] for r in selected})) or 'pending')
        ax.plot(a.sizes, med, 'o-', label=label, color=color)
        ax.fill_between(a.sizes, lo, hi, alpha=.15, color=color)
        table.append(status)
    ax.set(xscale='log', yscale='log', xlabel='Number of samples N', ylabel='Execution time (s)')
    ax.set_xlim(min(a.sizes) / 1.3, max(a.sizes) * 1.3)
    if not any(r['status'] == 'ok' for r in rows):
        ax.set_ylim(1e-3, 1)
    ax.set_xticks(a.sizes, [f'$10^{{{int(np.log10(n))}}}$' if 10**int(np.log10(n))==n else str(n) for n in a.sizes])
    ax.grid(True, which='both', alpha=.2)
    ax.legend()
    status_table = ax.table(cellText=table, rowLabels=['Dense', 'Flash'], colLabels=[f'{n:,}' for n in a.sizes], cellLoc='center', bbox=[0, -.43, 1, .24])
    status_table.auto_set_font_size(False)
    status_table.set_fontsize(8)
    fig.text(.13, .03, f'Median and min–max of valid trials; {a.device}; common ARPACK solver.\nMissing points are unmeasured; memory_limit denotes a conservative preflight skip.', fontsize=9)
    fig.subplots_adjust(left=.13, right=.97, top=.95, bottom=.37)
    fig.savefig(a.output/'timings.png', dpi=200)
    fig.savefig(a.output/'timings.pdf')
    plt.close(fig)


def main(a):
    if a.worker:
        a.result.write_text(json.dumps(worker(a)))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    save(a, rows)
    for n in a.sizes:
        for trial in range(a.repeats):
            # Alternate order to reduce systematic order effects.
            for method in (['dense', 'flash'] if trial % 2 == 0 else ['flash', 'dense']):
                result = a.output/f'{method}_{n}_{trial}.json'
                result.unlink(missing_ok=True)
                cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', method, '--sizes', str(n), '--trial', str(trial), '--result', str(result.resolve())]
                for key in ['device', 'seed', 'beta', 'alpha', 'components', 'tol', 'maxiter', 'validation_tol', 'memory_fraction', 'dense_max_gib', 'dense_mode', 'block_size']:
                    cmd += ['--'+key.replace('_', '-'), str(getattr(a, key))]
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                if a.device == 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                row = dict(method=method, n=n, trial=trial, status='error', seconds=None)
                deadline = (a.flash_timeout or None) if method == 'flash' else (a.timeout or None)
                with (a.output/f'{method}_{n}_{trial}.log').open('w') as log:
                    try:
                        completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=deadline)
                        if completed.returncode == 0 and result.exists():
                            row = json.loads(result.read_text())
                        else:
                            row['reason'] = f'Worker exit {completed.returncode}; inspect log (signal termination is not assumed to be OOM)'
                    except subprocess.TimeoutExpired:
                        row.update(status='timeout', reason=f'Worker exceeded {deadline}s including setup/warm-up')
                rows.append(row)
                # Check agreement on small paired cases, outside measured regions.
                pairs = [r for r in rows if r['n']==n and r['trial']==trial and r['status']=='ok']
                if len(pairs)==2 and n <= 1000:
                    import numpy as np
                    eig_error = float(np.max(np.abs(np.array(pairs[0]['eigenvalues'])-pairs[1]['eigenvalues'])))
                    probe_error = float(np.linalg.norm(np.array(pairs[0]['operator_probe'])-pairs[1]['operator_probe']))
                    for r in pairs:
                        r.update(paired_eigenvalue_error=eig_error, paired_operator_error=probe_error)
                        if max(eig_error, probe_error) > a.validation_tol:
                            r.update(status='invalid', seconds=None, reason='Dense/Flash small-case agreement check failed')
                print(f'{method} N={n} trial={trial}: {row["status"]}, seconds={row["seconds"]}', flush=True)
                save(a, rows)
    # Expected resource omissions do not fail the run; correctness/backend errors do.
    if any(r['status'] in {'error', 'invalid'} for r in rows) or not any(r['status']=='ok' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main(arguments())
