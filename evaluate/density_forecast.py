from __future__ import annotations
import warnings
from typing import NamedTuple
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import anderson

def compute_crps(forecast_samples: np.ndarray, realized: np.ndarray) -> np.ndarray:
    x = np.sort(forecast_samples.ravel().astype(np.float64))
    M = len(x)
    y = realized.ravel().astype(np.float64)
    mae_term = np.mean(np.abs(y[:, None] - x[None, :]), axis=1)
    k = np.arange(1, M + 1, dtype=np.float64)
    spread = float(np.sum(x * (2 * k - M - 1)) / M ** 2)
    return mae_term - spread

def mean_crps(forecast_samples: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    crps_vals = compute_crps(forecast_samples, realized)
    return (float(crps_vals.mean()), float(crps_vals.std() / np.sqrt(len(crps_vals))))

def compute_log_score(forecast_samples: np.ndarray, realized: np.ndarray, bw_method: str='silverman') -> np.ndarray:
    x = forecast_samples.ravel().astype(np.float64)
    if x.std() < 1e-12:
        return np.full(len(realized), -np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kde = sp_stats.gaussian_kde(x, bw_method=bw_method)
    y = realized.ravel().astype(np.float64)
    log_dens = np.log(np.maximum(kde(y), 1e-300))
    return log_dens

def mean_log_score(forecast_samples: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    ls = compute_log_score(forecast_samples, realized)
    return (float(ls.mean()), float(ls.std() / np.sqrt(len(ls))))

def compute_pit(forecast_samples: np.ndarray, realized: np.ndarray) -> np.ndarray:
    x = np.sort(forecast_samples.ravel().astype(np.float64))
    M = len(x)
    y = realized.ravel().astype(np.float64)
    ranks = np.searchsorted(x, y, side='right')
    return ranks.astype(np.float64) / (M + 1)

class TestResult(NamedTuple):
    statistic: float
    p_value: float

def ks_test_uniform(pit_values: np.ndarray) -> TestResult:
    u = pit_values.ravel()
    (stat, p) = sp_stats.kstest(u, 'uniform')
    return TestResult(float(stat), float(p))

def ad_test_uniform(pit_values: np.ndarray) -> TestResult:
    u = pit_values.ravel()
    u = np.clip(u, 1e-10, 1 - 1e-10)
    exp_vals = -np.log(u)
    res = anderson(exp_vals, dist='expon')
    sig_levels = np.array([0.15, 0.1, 0.05, 0.025, 0.01])
    crits = res.critical_values
    stat = float(res.statistic)
    if stat < crits[0]:
        p_approx = 0.2
    elif stat >= crits[-1]:
        p_approx = 0.005
    else:
        idx = np.searchsorted(crits, stat)
        (lo, hi) = (sig_levels[idx], sig_levels[idx - 1])
        p_approx = float(np.interp(stat, [crits[idx - 1], crits[idx]], [hi, lo]))
    return TestResult(stat, p_approx)

def dm_test_crps(crps_a: np.ndarray, crps_b: np.ndarray) -> TestResult:
    d = (crps_a - crps_b).astype(np.float64)
    n = len(d)
    if n < 2:
        return TestResult(float('nan'), float('nan'))
    d_bar = float(d.mean())
    lag = max(1, int(n ** (1 / 3)))
    gamma0 = float(np.var(d, ddof=0))
    gamma = gamma0
    for k in range(1, lag + 1):
        gamma += 2 * (1 - k / (lag + 1)) * float(np.cov(d[k:], d[:-k])[0, 1])
    gamma = max(gamma, 1e-15)
    t_dm = d_bar / np.sqrt(gamma / n)
    p = float(2 * sp_stats.t.sf(abs(t_dm), df=n - 1))
    return TestResult(float(t_dm), p)

def extract_h_day_returns(paths: np.ndarray, h: int, rolling: bool=True) -> np.ndarray:
    if paths.ndim == 1:
        r = paths.astype(np.float64)
        if rolling:
            n = len(r) - h + 1
            return np.array([r[i:i + h].sum() for i in range(n)])
        else:
            n_win = len(r) // h
            return np.array([r[i * h:(i + 1) * h].sum() for i in range(n_win)])
    (n_paths, seq_len) = paths.shape
    if not rolling:
        return paths[:, :h].sum(axis=1).astype(np.float64)
    n_windows = seq_len - h + 1
    result = []
    for i in range(n_windows):
        result.append(paths[:, i:i + h].sum(axis=1))
    return np.concatenate(result).astype(np.float64)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--generated", required=True, type=str)
    p.add_argument("--test", required=True, type=str)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 22])
    p.add_argument("--rolling", action="store_true")
    args = p.parse_args()

    gen = np.load(args.generated).astype(np.float64)
    test = np.load(args.test).astype(np.float64)
    if test.ndim == 1:
        test = test.reshape(1, -1)
    if gen.ndim == 1:
        gen = gen.reshape(1, -1)

    print(f"{'h':>3}  {'CRPS':>10}  {'SE':>10}  {'KS p':>10}  {'AD p':>10}  {'PIT mean':>10}  {'PIT std':>10}")
    print("-" * 80)
    for h in args.horizons:
        fc_h = extract_h_day_returns(gen, h, rolling=args.rolling)
        te_h = extract_h_day_returns(test, h, rolling=False)
        mc, se = mean_crps(fc_h, te_h)
        pit = compute_pit(fc_h, te_h)
        ks = ks_test_uniform(pit)
        ad = ad_test_uniform(pit)
        print(f"{h:>3}  {mc:>10.6f}  {se:>10.6f}  {ks.p_value:>10.4g}  {ad.p_value:>10.4g}  "
              f"{pit.mean():>10.4f}  {pit.std():>10.4f}")


if __name__ == "__main__":
    main()
