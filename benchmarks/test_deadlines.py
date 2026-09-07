"""Regression: disabled process limits must map to None, never timeout=0."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import benchmark_dmap_scaling as benchmark

for dense_limit, flash_limit in [(0, 0), (120, 0), (0, 200)]:
    with tempfile.TemporaryDirectory() as folder:
        args = ['benchmark', '--sizes', '10000', '--repeats', '1', '--device', 'cpu',
                '--output', folder, '--timeout', str(dense_limit), '--flash-timeout', str(flash_limit)]
        with patch.object(sys, 'argv', args):
            config = benchmark.arguments()
        calls = []
        def fake_run(command, **kwargs):
            method = command[command.index('--worker')+1]
            expected = dense_limit if method == 'dense' else flash_limit
            assert kwargs['timeout'] == (expected or None)
            path = Path(command[command.index('--result')+1])
            path.write_text(json.dumps(dict(method=method, n=10000, trial=0, status='ok', seconds=1.)))
            calls.append(method)
            return SimpleNamespace(returncode=0)
        with patch.object(benchmark.subprocess, 'run', side_effect=fake_run), patch.object(benchmark, 'save'):
            benchmark.main(config)
        assert calls == ['dense', 'flash']
print('PASS: both deadlines disabled; optional deadlines remain independent')
