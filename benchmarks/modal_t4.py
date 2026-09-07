"""Run the repository benchmark on exactly one Modal T4; never select another GPU.

Credentials are read by Modal from MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
No token belongs in this file. The Actions workflow supplies repository secrets.
The SM75 adaptation only adds an architecture entry to upstream's JIT builder;
the pinned WMMA CUDA source is unchanged. A small GPU validation gates scaling.
"""
from pathlib import Path
import io
import json
import zipfile
import modal

UPSTREAM = 'f5c2d1abb7e418829651f7e2fedbb8b323fd56f9'
HERE = Path(__file__).resolve().parent
app = modal.App('flashdiffusion-tests-t4')
volume = modal.Volume.from_name('flashdiffusion-t4-results', create_if_missing=True)


def configure_sm75():
    """Build-time, checked patch; do not import the GPU package to locate it."""
    import importlib.util
    from pathlib import Path
    root = Path(importlib.util.find_spec('flashdiffusion').origin).parent
    path = root / 'kernel_jit.py'
    source = path.read_text()
    anchor = '    _SOURCES = {\n'
    if source.count(anchor) != 1:
        raise RuntimeError('Pinned upstream JIT layout changed; refusing an unchecked patch')
    entry = "        75:  ('flash_diffusion_sm80.cu', 'sm_75', False, 'flash_diffusion_cuda'),\n"
    path.write_text(source.replace(anchor, anchor + entry))


image = (
    modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.11')
    .apt_install('git', 'g++')
    .pip_install('torch==2.5.1', index_url='https://download.pytorch.org/whl/cu124')
    .pip_install('numpy==2.1.3', 'scipy==1.14.1', 'matplotlib==3.9.2', 'psutil==6.1.0', 'ninja==1.11.1.1',
                 f'flashdiffusion @ git+https://github.com/sparsetrace/FlashDiffusion.git@{UPSTREAM}')
    .run_function(configure_sm75)
    .env({'MAX_JOBS': '2', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'MKL_NUM_THREADS': '1', 'TORCH_CUDA_ARCH_LIST': '7.5'})
    .add_local_file(HERE / 'benchmark_dmap_scaling.py', '/bench/benchmark_dmap_scaling.py')
)


@app.function(image=image, gpu='T4', cpu=4, memory=16384, timeout=86400, volumes={'/results': volume},
              retries=0, max_containers=1)
def run_benchmark(sizes: list[int], repeats: int, trial_timeout: int, source_commit: str, run_id: str):
    import hashlib
    import importlib.util
    import subprocess
    import sys
    import torch
    from pathlib import Path

    root = Path('/results') / run_id
    root.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if 'T4' not in gpu or capability != (7, 5):
        raise RuntimeError(f'Required T4 SM75, received {gpu} {capability}')
    (root/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))
    (root/'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    package = Path(importlib.util.find_spec('flashdiffusion').origin).parent
    source = package/'csrc/flash_diffusion_sm80.cu'
    (root/'provenance.json').write_text(json.dumps(dict(
        benchmark_commit=source_commit, upstream_commit=UPSTREAM, gpu=gpu,
        capability=capability, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        adaptation='JIT architecture mapping SM75; upstream CUDA source unchanged',
        requested_sizes=sizes, repeats=repeats, dense_timeout_seconds=trial_timeout or None, flash_timeout_seconds=None,
        modal_function_limit_seconds=86400, dense_mode="batched", block_size=2048,
    ), indent=2))

    def execute(name, requested, count, deadline):
        cmd = [sys.executable, '/bench/benchmark_dmap_scaling.py', '--device', 'cuda',
               '--sizes', *map(str, requested), '--repeats', str(count),
               '--timeout', str(deadline), '--flash-timeout', '0', '--dense-mode', 'batched', '--output', str(root/name)]
        with (root/f'{name}.log').open('w') as log:
            import time
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                volume.commit()  # Persist partial JSON/CSV/figures during long trials.
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            volume.commit()
            return process.returncode

    # Includes real CUDA compilation and both dense/Flash agreement checks.
    code = execute('preflight', [100, 1000], 1, 600)
    preflight_path = root/'preflight/results.json'
    preflight_ok = False
    if code == 0 and preflight_path.exists():
        trials = json.loads(preflight_path.read_text())['trials']
        preflight_ok = len(trials) == 4 and all(r['status'] == 'ok' for r in trials)
    if preflight_ok:
        code = execute('scaling', sizes, repeats, trial_timeout)
    else:
        code = 1
        (root/'scaling_not_run.txt').write_text('GPU preflight failed; no large-N measurements were attempted. Inspect preflight logs.\n')
    (root/'exit_status.json').write_text(json.dumps(dict(exit_code=code, preflight_ok=preflight_ok)))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    (root/'modal-t4-results.zip').write_bytes(buffer.getvalue())
    volume.commit()
    return code, buffer.getvalue()


@app.local_entrypoint()
def main(sizes: str = '100,1000,10000,100000,1000000,10000000', repeats: int = 3,
         trial_timeout: int = 0, output: str = 'modal_results', source_commit: str = ''):
    requested = [int(n) for n in sizes.split(',')]
    allowed = {100, 1000, 10000, 100000, 1000000, 10000000}
    if not requested or not set(requested) <= allowed or len(set(requested)) != len(requested):
        raise ValueError('Sizes must be distinct powers of ten from 100 to 10000000')
    if not 1 <= repeats <= 3 or not 0 <= trial_timeout <= 300:
        raise ValueError('Use 1-3 repetitions and 0-300 seconds for dense (0 disables deadline)')
    if not source_commit:
        import subprocess
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    code, payload = run_benchmark.remote(requested, repeats, trial_timeout, source_commit, __import__('uuid').uuid4().hex)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'modal-t4-results.zip').write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # Only consume this function's expected relative artifact paths.
        for item in archive.infolist():
            if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts:
                raise ValueError('Invalid result archive path')
        archive.extractall(folder)
    print(f'T4 artifacts saved to {folder}; exit_code={code}')
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    import argparse
    import uuid
    p = argparse.ArgumentParser()
    p.add_argument('--submit', action='store_true')
    p.add_argument('--collect', default='')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--trial-timeout', type=int, default=0, help='Dense deadline; 0 disables it')
    p.add_argument('--source-commit', default='')
    p.add_argument('--output', default='modal_results')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.submit:
        if not 1 <= args.repeats <= 3 or not 0 <= args.trial_timeout <= 300:
            p.error('Use 1-3 repeats and a dense deadline of 0-300 seconds (0 disables it)')
        run_id = uuid.uuid4().hex
        fn = modal.Function.from_name('flashdiffusion-tests-t4', 'run_benchmark')
        call = fn.spawn([100,1000,10000,100000,1000000,10000000], args.repeats,
                        args.trial_timeout, args.source_commit, run_id)
        receipt = dict(call_id=call.object_id, volume='flashdiffusion-t4-results',
                       run_directory=run_id, status='submitted', source_commit=args.source_commit)
        (out/'submission.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt))
    elif args.collect:
        try:
            code, payload = modal.FunctionCall.from_id(args.collect).get(timeout=0)
        except TimeoutError:
            (out/'pending.json').write_text(json.dumps(dict(call_id=args.collect, status='pending')))
            print('Still running; no wait performed. Partial results are in the Modal volume.')
        else:
            (out/'modal-t4-results.zip').write_bytes(payload)
            print(f'Collected result; exit_code={code}')
            raise SystemExit(code)
    else:
        p.error('Use --submit or --collect CALL_ID')
