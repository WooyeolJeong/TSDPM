import warnings
from typing import NamedTuple
import numpy as np
from scipy import stats as sp_stats

def compute_var_es(returns: np.ndarray, alpha: float, horizon: int=1, n_paths: int | None=None) -> tuple[float, float]:
    if returns.ndim == 2:
        if n_paths is not None and n_paths < returns.shape[0]:
            idx = np.random.choice(returns.shape[0], n_paths, replace=False)
            returns = returns[idx]
        h_returns = returns[:, :horizon].sum(axis=1).astype(np.float64)
    else:
        h_returns = returns.ravel().astype(np.float64)
    q = float(np.percentile(h_returns, alpha * 100))
    var = -q
    tail = h_returns[h_returns <= q]
    es = -float(tail.mean()) if len(tail) > 0 else var
    return (var, es)

def violation_series(actual_returns: np.ndarray, var: float, horizon: int=1) -> np.ndarray:
    r = actual_returns.ravel().astype(np.float64)
    if horizon > 1:
        n = len(r) - horizon + 1
        agg = np.array([r[i:i + horizon].sum() for i in range(n)])
    else:
        agg = r
    return (agg <= -var).astype(int)

class TestResult(NamedTuple):
    statistic: float
    p_value: float

def kupiec_test(violations: np.ndarray, alpha: float) -> TestResult:
    n = len(violations)
    x = int(violations.sum())
    if x == 0 or x == n:
        return TestResult(float('nan'), float('nan'))
    pi_hat = x / n
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr = -2.0 * (x * np.log(alpha / pi_hat) + (n - x) * np.log((1 - alpha) / (1 - pi_hat)))
    p = float(sp_stats.chi2.sf(lr, df=1))
    return TestResult(float(lr), p)

def christoffersen_test(violations: np.ndarray, alpha: float) -> TestResult:
    v = violations.astype(int)
    n = len(v)
    x = int(v.sum())
    if x == 0 or x == n:
        return TestResult(float('nan'), float('nan'))
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    pi_hat = x / n
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr_ind = -2.0 * ((n00 + n10) * np.log(1 - pi_hat) + (n01 + n11) * np.log(pi_hat) - (n00 * (np.log(1 - pi01) if pi01 < 1 else 0) + n01 * (np.log(pi01) if pi01 > 0 else 0)) - (n10 * (np.log(1 - pi11) if pi11 < 1 else 0) + n11 * (np.log(pi11) if pi11 > 0 else 0)))
    if np.isnan(lr_ind) or lr_ind < 0:
        lr_ind = 0.0
    lr_pof = kupiec_test(violations, alpha).statistic
    if np.isnan(lr_pof):
        return TestResult(float('nan'), float('nan'))
    lr_cc = lr_pof + lr_ind
    p = float(sp_stats.chi2.sf(lr_cc, df=2))
    return TestResult(float(lr_cc), p)

def es_ratio(actual_returns: np.ndarray, var: float, es: float, horizon: int=1) -> float:
    r = actual_returns.ravel().astype(np.float64)
    if horizon > 1:
        n = len(r) - horizon + 1
        agg = np.array([r[i:i + horizon].sum() for i in range(n)])
    else:
        agg = r
    tail = agg[agg <= -var]
    if len(tail) == 0:
        return float('nan')
    avg_loss = -float(tail.mean())
    return avg_loss / es if es > 0 else float('nan')

def acerbi_szekely_z2(actual_returns: np.ndarray, var: float, es: float, alpha: float, horizon: int=1, n_simulations: int=10000, seed: int=0) -> TestResult:
    r = actual_returns.ravel().astype(np.float64)
    if horizon > 1:
        n_raw = len(r) - horizon + 1
        r = np.array([r[i:i + horizon].sum() for i in range(n_raw)])
    n = len(r)
    I = (r <= -var).astype(float)
    z2 = float((I * (r / es + 1)).sum() / (alpha * n))
    rng = np.random.default_rng(seed)
    z2_null = np.empty(n_simulations)
    for i in range(n_simulations):
        r_boot = rng.choice(r, size=n, replace=True)
        q_boot = np.percentile(r_boot, alpha * 100)
        tail = r_boot[r_boot <= q_boot]
        es_boot = -float(tail.mean()) if len(tail) > 0 else es
        I_b = (r_boot <= -var).astype(float)
        z2_null[i] = float((I_b * (r_boot / es_boot + 1)).sum() / (alpha * n))
    p = float(np.mean(z2_null <= z2))
    return TestResult(float(z2), p)

def lopez_loss(actual_returns: np.ndarray, var: float, alpha: float, horizon: int=1) -> np.ndarray:
    r = actual_returns.ravel().astype(np.float64)
    if horizon > 1:
        n_raw = len(r) - horizon + 1
        r = np.array([r[i:i + horizon].sum() for i in range(n_raw)])
    viol = (r <= -var).astype(float)
    return (viol - alpha) ** 2

def tick_loss(actual_returns: np.ndarray, var: float, alpha: float, horizon: int=1) -> np.ndarray:
    r = actual_returns.ravel().astype(np.float64)
    if horizon > 1:
        n_raw = len(r) - horizon + 1
        r = np.array([r[i:i + horizon].sum() for i in range(n_raw)])
    u = r + var
    viol = (r < -var).astype(float)
    return (alpha - viol) * u

def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> TestResult:
    d = (loss_a - loss_b).astype(np.float64)
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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--generated", required=True, type=str)
    p.add_argument("--real", required=True, type=str)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args()

    gen = np.load(args.generated).astype(np.float64).ravel()
    real = np.load(args.real).astype(np.float64).ravel()

    var, es = compute_var_es(gen, args.alpha, args.horizon)
    viols = violation_series(real, var, args.horizon)
    viol_rate = float(viols.mean())
    kupiec = kupiec_test(viols, args.alpha)
    chris = christoffersen_test(viols, args.alpha)
    es_r = es_ratio(real, var, es, args.horizon)
    lopez = float(lopez_loss(real, var, args.alpha, args.horizon).mean())
    tick = float(tick_loss(real, var, args.alpha, args.horizon).mean())

    print(f"alpha           : {args.alpha}")
    print(f"horizon         : {args.horizon}")
    print(f"VaR             : {var:.6f}")
    print(f"ES              : {es:.6f}")
    print(f"violation rate  : {viol_rate*100:.2f}%  (nominal {args.alpha*100:.1f}%)")
    print(f"Kupiec  stat/p  : {kupiec.statistic:.4f} / {kupiec.p_value:.4g}")
    print(f"Christoffersen  : {chris.statistic:.4f} / {chris.p_value:.4g}")
    print(f"ES ratio        : {es_r:.4f}")
    print(f"Lopez loss mean : {lopez:.6f}")
    print(f"Tick loss  mean : {tick:.6f}")


if __name__ == "__main__":
    main()
