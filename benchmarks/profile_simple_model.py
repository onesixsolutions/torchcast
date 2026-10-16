"""
Profile a simple model (no mixture components etc.) -- the model and data from ``docs/quick_start.py`` -- to check that
changes to the internals don't slow down the common case.

Usage::

    # profile the current working tree:
    python benchmarks/profile_simple_model.py

    # compare the working tree against one or more git refs (each is exported to a temp dir and run in a subprocess):
    python benchmarks/profile_simple_model.py --compare main
    python benchmarks/profile_simple_model.py --compare origin/main develop --repeats 60

Each version runs in a subprocess, in alternating rounds (so background load affects them equally). Timings are the
median (and interquartile range) across runs, after warmup. The 'loss' column is a sanity
check that the versions being compared compute the same thing.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build(seed: int = 0):
    import numpy as np
    import torch
    from torchcast.kalman_filter import KalmanFilter
    from torchcast.process import LocalTrend, Season
    from torchcast.utils.data import TimeSeriesDataset
    from torchcast.utils.datasets import load_air_quality_data

    df_aq = load_air_quality_data('weekly')
    df_aq['PM2p5_log10'] = np.log10(df_aq['PM2p5'])
    df_aq['PM10_log10'] = np.log10(df_aq['PM10'])
    dataset = TimeSeriesDataset.from_dataframe(
        dataframe=df_aq,
        dt_unit='W',
        measure_colnames=['PM2p5_log10', 'PM10_log10'],
        group_colname='station',
        time_colname='week'
    )
    dataset, _ = dataset.train_val_split(dt=np.datetime64('2016-02-22'))

    torch.manual_seed(seed)
    processes = []
    for m in dataset.measures[0]:
        processes.extend([
            LocalTrend(id=f'{m}_trend', measure=m),
            Season(id=f'{m}_day_in_year', period=365.25 / 7, dt_unit='W', K=3, measure=m, fixed=True)
        ])
    kf = KalmanFilter(measures=dataset.measures[0], processes=processes)
    return kf, dataset.tensors[0], dataset.start_datetimes


def profile(repeats: int, warmup: int) -> dict:
    import torch
    import torchcast

    torch.set_num_threads(1)  # less noisy
    kf, y, start_offsets = build()

    def forward_no_grad():
        with torch.no_grad():
            return kf(y, start_offsets=start_offsets).log_prob(y).mean()

    def train_step(n_step: int = 1):
        kf.zero_grad()
        loss = -kf(y, start_offsets=start_offsets, n_step=n_step).log_prob(y).mean()
        loss.backward()
        return loss

    scenarios = {
        'forward (no grad)': forward_no_grad,
        'forward+backward': train_step,
        'forward+backward, n_step=4': lambda: train_step(n_step=4),
    }
    out = {'torchcast': torchcast.__file__, 'shape': list(y.shape), 'results': {}}
    for name, fun in scenarios.items():
        for _ in range(warmup):
            fun()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            loss = fun()
            times.append(time.perf_counter() - start)
        out['results'][name] = {'times': times, 'loss': abs(loss.item())}
    return out


def _run_subprocess(pythonpath: str, repeats: int, warmup: int) -> dict:
    env = {**os.environ, 'PYTHONPATH': pythonpath}
    cmd = [sys.executable, __file__, '--json', '--repeats', str(repeats), '--warmup', str(warmup)]
    res = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, cwd=pythonpath)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    if not Path(out['torchcast']).resolve().is_relative_to(Path(pythonpath).resolve()):
        raise RuntimeError(f"Expected to import torchcast from {pythonpath}, got {out['torchcast']}")
    return out


def _merge(runs: list[dict]) -> dict:
    out = {**runs[0], 'results': {}}
    for scenario in runs[0]['results']:
        times = sorted(t for run in runs for t in run['results'][scenario]['times'])
        out['results'][scenario] = {
            'median': times[len(times) // 2],
            'q1': times[len(times) // 4],
            'q3': times[(3 * len(times)) // 4],
            'loss': runs[0]['results'][scenario]['loss'],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--compare', nargs='*', default=[], help="Git refs to compare against the working tree.")
    parser.add_argument('--repeats', type=int, default=30, help="Total timed runs per scenario, per version.")
    parser.add_argument('--rounds', type=int, default=5,
                        help="Versions are run in alternating rounds, so that background load affects them equally.")
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--json', action='store_true', help=argparse.SUPPRESS)  # used internally
    args = parser.parse_args()

    if args.json:
        print(json.dumps(profile(args.repeats, args.warmup)))
        return

    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for ref in args.compare:
            paths[ref] = os.path.join(tmp, f'ref{len(paths)}')
            os.makedirs(paths[ref])
            archive = subprocess.run(['git', 'archive', ref, 'torchcast'], cwd=REPO_ROOT, check=True,
                                     capture_output=True)
            subprocess.run(['tar', '-x', '-C', paths[ref]], input=archive.stdout, check=True)
        paths['working tree'] = str(REPO_ROOT)

        runs = {name: [] for name in paths}
        per_round = max(1, args.repeats // args.rounds)
        for i in range(args.rounds):
            for name, path in paths.items():
                print(f"Round {i + 1}/{args.rounds}: {name}...", file=sys.stderr)
                runs[name].append(_run_subprocess(path, per_round, args.warmup))
        versions = {name: _merge(r) for name, r in runs.items()}
        # per-round medians, to show any drift in background load over the course of the benchmark:
        drift = {
            name: [sorted(run['results']['forward+backward']['times'])[per_round // 2] for run in r]
            for name, r in runs.items()
        }

    shape = next(iter(versions.values()))['shape']
    print(f"\nquick_start model, y.shape={tuple(shape)}, median of {per_round * args.rounds} runs (IQR)\n")
    baseline = next(iter(versions))
    scenarios = list(versions[baseline]['results'])
    width = max(len(v) for v in versions) + 2
    for scenario in scenarios:
        print(scenario)
        base = versions[baseline]['results'][scenario]['median']
        for name, out in versions.items():
            r = out['results'][scenario]
            ratio = '' if name == baseline else f"  {r['median'] / base:5.2f}x vs {baseline}"
            print(f"  {name:<{width}} {1000 * r['median']:8.1f}ms  ({1000 * r['q1']:.1f}-{1000 * r['q3']:.1f})"
                  f"  loss={r['loss']:.6f}{ratio}")
        print()

    print("per-round medians, forward+backward (ms) -- these should be flat; if they drift, rerun on a quieter machine")
    for name, medians in drift.items():
        print(f"  {name:<{width}} " + ' '.join(f"{1000 * m:6.1f}" for m in medians))


if __name__ == '__main__':
    main()
