"""
Compare the sampling methods behind torchcast's sample-based prediction outputs: randomized quasi-monte-carlo
(scrambled Sobol, ``sampling='qmc'``, the default) vs. plain monte-carlo (``sampling='mc'``).

On synthetic data, for a few realistic outputs, at equal sample counts: the error vs. a high-sample reference, for each
group and timestep (``row``), and for the sum over groups at each timestep (``aggregate``) -- which matters because the
draws are shared across groups, so their monte-carlo error doesn't average out in a sum.

The outputs (``model.quantity``):

- ``binomial.visit_mean``: a binary measure's mean, ``E[sigmoid(state)]`` (continuous in the draws).
- ``binomial.visit_upper``: the upper bound of its 90% interval (a quantile of the draws).
- ``binomial.capped_spend_mean``: ``derived`` spend capped at 30, ``min(exp(log_spend), 30)``, with observation noise
  and the mixture's regime (with latent classes) drawn per sample: continuous in the state and noise, discontinuous in
  the regime. (Uncapped, it would be exact: ``derived`` uses ``exp(log_spend)``'s exact mean as a control variate.)
- ``binomial.weekly_spend_mean``: ``derived`` ``visit * spend``, where ``visit`` is a binomial draw (always monte-carlo,
  independent across rows).
- ``sigmoid.log_mean``: a log-transformed measure with a nonlinear (sigmoid) measured-mean and a mixture component: the
  back-transformed mean, ``E[exp(Y)]`` (continuous: the regimes are mixed exactly, only the state is sampled).
- ``sigmoid.log_upper``: the upper bound of its 90% interval (with a regime drawn per sample: discontinuous).

Also ``qmc-mc-regime``: QMC, but with the regime drawn from monte-carlo uniforms -- to see what QMC gains on the
discrete draws.

Usage::

    python benchmarks/output_sampling.py               # the defaults below (~10 minutes)
    python benchmarks/output_sampling.py --reps 5      # quicker, noisier

Prints a table of RMSEs (over replicate seeds, and over rows or timesteps) and the ratio MC / QMC. The aggregate RMSE
is a percentage of the (absolute) sum over groups.

Results (torch 2.5.1, the defaults: 20 seeds; reference 4 x 2 ** 16 QMC samples), RMSE ratio MC / QMC per row
(aggregate):

=================================  ==============  ==============  ==============
output                             256             1024            4096
=================================  ==============  ==============  ==============
binomial.visit_mean                53 (65)         174 (214)       330 (411)
binomial.visit_upper               8.8 (8.8)       13 (13)         25 (25)
binomial.capped_spend_mean         3.2 (3.3)       5.7 (7.2)       9.6 (15)
binomial.weekly_spend_mean         1.08 (4.6)      1.06 (4.2)      1.05 (3.9)
sigmoid.log_mean                   30 (30)         87 (87)         163 (163)
sigmoid.log_upper                  3.1 (3.5)       4.5 (5.7)       5.0 (7.8)
=================================  ==============  ==============  ==============

With monte-carlo uniforms for the regime (``qmc-mc-regime``), QMC's RMSE was 2-6x larger on the outputs with regime
draws (e.g. ``capped_spend_mean`` at 1024: 0.0208 vs. 0.0053 per row): so the regime uses the Sobol points too.
``weekly_spend_mean``'s per-row error is dominated by its binomial draws (always monte-carlo); since those are
independent across rows, they average out in the aggregate, where the shared draws' gain shows.
"""
import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # (this repo's torchcast, not an installed one)
from torchcast.kalman_filter import BinomialFilter, KalmanFilter
from torchcast.mixture import MixtureModel, MixtureComponent, LatentClass
from torchcast.process import LocalLevel
from torchcast.state_space import Predictions
from torchcast.state_space.transforms import LogTransform

NUM_GROUPS = 200
NUM_OBSERVED, NUM_FORECAST = 26, 13


def make_binomial_predictions(num_groups: int = NUM_GROUPS) -> Predictions:
    """
    A binary measure (``visit``) and a log-scale measure (``log_spend``, only observed with a visit) with a 'quick'
    mixture component (a low-spend regime) and two latent classes with different rates of it. Forecasts, so that the
    state's uncertainty grows.
    """
    torch.manual_seed(2026)
    group_logit = torch.randn(num_groups, 1) * 1.5
    visit = (torch.rand(num_groups, NUM_OBSERVED) < torch.sigmoid(group_logit)).float()
    log_spend = 3. + torch.randn(num_groups, 1) * .5 + torch.randn(num_groups, NUM_OBSERVED).cumsum(1) * .1
    quick = torch.rand(num_groups, NUM_OBSERVED) < torch.where(torch.rand(num_groups, 1) < .3, .4, .05)
    log_spend = torch.where(quick, log_spend - 1.5, log_spend)
    log_spend[visit == 0] = float('nan')
    y = torch.stack([visit, log_spend], -1)

    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'log_spend']],
        measures=['visit', 'log_spend'],
        binary_measures=['visit'],
        mixture=MixtureModel(
            [MixtureComponent(measure='log_spend', offset_init=-1.5, prob_init=.1, id='quick')],
            classes=[LatentClass('steady'), LatentClass('quick_prone', relative_rate_init={'log_spend.quick': 4.})],
        ),
    )
    with torch.no_grad():
        bf.process_covariance.cholesky_log_diag.copy_(torch.log(torch.tensor([.3, .1])))
        bf.measure_covariance.cholesky_log_diag.fill_(math.log(.3))
        bf.mixture.components[0]._log_extra_std.fill_(math.log(.3))
    bf.mc_sampling = 100
    with torch.no_grad():
        return bf(y, out_timesteps=NUM_OBSERVED + NUM_FORECAST)


def make_sigmoid_predictions(num_groups: int = NUM_GROUPS) -> Predictions:
    """
    A measure with a nonlinear (sigmoid) measured-mean, gaussian noise, and a 'low' mixture component; modeled on the
    log scale (so it's back-transformed with ``LogTransform``).
    """
    torch.manual_seed(2027)
    y = torch.sigmoid(torch.randn(num_groups, 1) + torch.randn(num_groups, NUM_OBSERVED).cumsum(1) * .2)
    y = y + torch.randn(y.shape) * .05
    y[torch.rand(y.shape) < .15] = .1
    kf = KalmanFilter(
        processes=[LocalLevel(id='level')],
        measures=['y'],
        measure_funs={'y': 'sigmoid'},
        mixture=[MixtureComponent(measure='y', offset_init=-.4, prob_init=.15, id='low')],
    )
    with torch.no_grad():
        kf.measure_covariance.cholesky_log_diag.fill_(math.log(.05))
        kf.process_covariance.cholesky_log_diag.fill_(math.log(.5))
        kf.mixture.components[0]._log_extra_std.fill_(math.log(.03))
    kf.mc_sampling = 100
    with torch.no_grad():
        return kf(y.unsqueeze(-1), out_timesteps=NUM_OBSERVED + NUM_FORECAST)


def get_outputs(binomial: Predictions, sigmoid: Predictions, num_samples: int) -> dict[str, np.ndarray]:
    """Each output as a ``(num_groups, num_timesteps)`` array, using the predictions' current seed and sampling."""

    def pivot(df: pd.DataFrame, measure: str, col: str) -> np.ndarray:
        return df.query(f"measure == '{measure}'").pivot(index='group', columns='time', values=col).values

    df = binomial.to_dataframe(
        conf=.9,
        use_map=False,
        transform={'log_spend': LogTransform()},
        derived={
            'capped_spend': lambda s: s['log_spend'].clamp(max=30.),
            'weekly_spend': lambda s: s['visit'] * s['log_spend'],
        },
        num_samples=num_samples,
    )
    df_sig = sigmoid.to_dataframe(conf=.9, use_map=False, transform=LogTransform(), num_samples=num_samples)
    return {
        'binomial.visit_mean': pivot(df, 'visit', 'mean'),
        'binomial.visit_upper': pivot(df, 'visit', 'upper'),
        'binomial.capped_spend_mean': pivot(df, 'capped_spend', 'mean'),
        'binomial.weekly_spend_mean': pivot(df, 'weekly_spend', 'mean'),
        'sigmoid.log_mean': pivot(df_sig, 'y', 'mean'),
        'sigmoid.log_upper': pivot(df_sig, 'y', 'upper'),
    }


@contextlib.contextmanager
def monte_carlo_regimes():
    """QMC, but with monte-carlo uniforms for the regime (to isolate what QMC does for the discrete draws)."""
    original = Predictions._output_draws_given_state

    def patched(self, num_samples):
        draws = original(self, num_samples)
        generator = torch.Generator().manual_seed(self._get_output_seed() + 1)
        draws['regime'] = torch.rand(num_samples, generator=generator, dtype=draws['regime'].dtype)
        return draws

    with mock.patch.object(Predictions, '_output_draws_given_state', patched):
        yield


def get_outputs_for(binomial: Predictions,
                    sigmoid: Predictions,
                    method: str,
                    num_samples: int,
                    seed: int) -> dict[str, np.ndarray]:
    """``get_outputs`` with ``method`` ('qmc', 'mc', or 'qmc-mc-regime') and output-seed ``seed``."""
    for pred in (binomial, sigmoid):
        pred._output_seed = seed
        pred.output_sampling = 'mc' if method == 'mc' else 'qmc'
    with monte_carlo_regimes() if method == 'qmc-mc-regime' else contextlib.nullcontext():
        return get_outputs(binomial, sigmoid, num_samples)


def main(reps: int, sample_sizes: list[int], reference_samples: int, reference_reps: int) -> pd.DataFrame:
    torch.set_grad_enabled(False)
    binomial, sigmoid = make_binomial_predictions(), make_sigmoid_predictions()

    started = time.time()
    references = [
        get_outputs_for(binomial, sigmoid, 'qmc', reference_samples, seed=10_000 + i) for i in range(reference_reps)
    ]
    reference = {k: np.mean([r[k] for r in references], 0) for k in references[0]}
    print(f"reference: {reference_reps} x {reference_samples} QMC samples ({time.time() - started:.0f}s)")

    rows = []
    for num_samples in sample_sizes:
        for method in ['mc', 'qmc', 'qmc-mc-regime']:
            started = time.time()
            errors = {k: [] for k in reference}
            for rep in range(reps):
                outputs = get_outputs_for(binomial, sigmoid, method, num_samples, seed=rep)
                for k in reference:
                    errors[k].append(outputs[k] - reference[k])
            for k, errs in errors.items():
                errs = np.stack(errs)  # (reps, groups, timesteps)
                total = np.abs(reference[k].sum(0)).mean()
                rows.append({
                    'output': k,
                    'num_samples': num_samples,
                    'method': method,
                    'row_rmse': np.sqrt(np.mean(errs ** 2)),
                    'aggregate_rmse_pct': 100 * np.sqrt(np.mean(errs.sum(1) ** 2)) / total,
                })
            print(f"{method} x {num_samples}: {time.time() - started:.0f}s")

    df = pd.DataFrame(rows)
    # (the regime-variant only matters for outputs with regime draws)
    with_regimes = ['binomial.capped_spend_mean', 'binomial.weekly_spend_mean', 'sigmoid.log_upper']
    df = df.query("method != 'qmc-mc-regime' or output in @with_regimes")
    wide = df.pivot_table(index=['output', 'num_samples'], columns='method', values=['row_rmse', 'aggregate_rmse_pct'])
    for metric in ['row_rmse', 'aggregate_rmse_pct']:
        wide[(metric, 'mc / qmc')] = wide[(metric, 'mc')] / wide[(metric, 'qmc')]
    return wide.sort_index(axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reps', type=int, default=20, help="Replicate seeds per method and sample size.")
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[256, 1024, 4096])
    parser.add_argument('--reference-samples', type=int, default=2 ** 16)
    parser.add_argument('--reference-reps', type=int, default=4, help="QMC seeds averaged for the reference.")
    args = parser.parse_args()
    with pd.option_context('display.width', 250, 'display.max_columns', 20, 'display.float_format', '{:.3g}'.format):
        print(main(args.reps, args.sample_sizes, args.reference_samples, args.reference_reps))
