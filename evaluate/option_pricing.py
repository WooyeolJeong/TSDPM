from __future__ import annotations
import warnings
from typing import Literal
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal['call', 'put']) -> float:
    if sigma <= 0 or T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
        return float(intrinsic)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * sp_stats.norm.cdf(d1) - K * np.exp(-r * T) * sp_stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * sp_stats.norm.cdf(-d2) - S * sp_stats.norm.cdf(-d1)
    return float(max(price, 0.0))

def bs_implied_vol(price: float, S: float, K: float, T: float, r: float, option_type: Literal['call', 'put'], tol: float=1e-06, sigma_lo: float=0.0001, sigma_hi: float=10.0) -> float:
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == 'call' else max(K * np.exp(-r * T) - S, 0.0)
    if price <= intrinsic + 1e-08:
        return float('nan')

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price
    try:
        f_lo = objective(sigma_lo)
        f_hi = objective(sigma_hi)
        if f_lo * f_hi > 0:
            return float('nan')
        iv = brentq(objective, sigma_lo, sigma_hi, xtol=tol, maxiter=200)
        return float(iv)
    except Exception:
        return float('nan')

def _heston_char_fn(u: np.ndarray, S0: float, K: float, T: float, r: float, kappa: float, theta: float, sigma_v: float, rho: float, v0: float) -> np.ndarray:
    i = 1j
    xi = kappa - rho * sigma_v * i * u
    d = np.sqrt(xi ** 2 + sigma_v ** 2 * u * (u + i))
    g = (xi - d) / (xi + d)
    C = r * i * u * T + kappa * theta / sigma_v ** 2 * ((xi - d) * T - 2.0 * np.log((1.0 - g * np.exp(-d * T)) / (1.0 - g)))
    D = (xi - d) / sigma_v ** 2 * ((1.0 - np.exp(-d * T)) / (1.0 - g * np.exp(-d * T)))
    return np.exp(C + D * v0 + i * u * np.log(S0 * np.exp(r * T)))

def heston_price_carr_madan(S0: float, K: float, T: float, r: float, kappa: float, theta: float, sigma_v: float, rho: float, v0: float, option_type: Literal['call', 'put'], N: int=4096, alpha: float=1.5, eta: float=0.25) -> float:
    lam = 2.0 * np.pi / (N * eta)
    b = N * lam / 2.0
    k_grid = -b + lam * np.arange(N)
    v_grid = eta * np.arange(N)
    w = np.ones(N)
    w[0] = 1.0 / 3.0
    w[-1] = 1.0 / 3.0
    w[1:-1:2] = 4.0 / 3.0
    w[2:-2:2] = 2.0 / 3.0
    psi = np.exp(-r * T) * _heston_char_fn(v_grid - (alpha + 1) * 1j, S0, K, T, r, kappa, theta, sigma_v, rho, v0) / (alpha ** 2 + alpha - v_grid ** 2 + 1j * (2 * alpha + 1) * v_grid)
    x = np.exp(1j * v_grid * b) * psi * w * eta
    fft_val = np.fft.fft(x).real
    log_K = np.log(K)
    prices = np.exp(-alpha * k_grid) / np.pi * fft_val
    idx = np.searchsorted(k_grid, log_K)
    idx = int(np.clip(idx, 1, N - 2))
    t = (log_K - k_grid[idx - 1]) / (k_grid[idx] - k_grid[idx - 1])
    call_price = float((1 - t) * prices[idx - 1] + t * prices[idx])
    call_price = max(call_price, max(S0 - K * np.exp(-r * T), 0.0))
    if option_type == 'call':
        return float(call_price)
    else:
        put_price = call_price - S0 + K * np.exp(-r * T)
        return float(max(put_price, max(K * np.exp(-r * T) - S0, 0.0)))

def mc_price_european(samples: np.ndarray, S0: float, K: float, T_mat: int, option_type: Literal['call', 'put'], r: float=0.0, payoff_clip_pct: float | None=99.9) -> tuple[float, float]:
    if T_mat > samples.shape[1]:
        raise ValueError(f'T_mat={T_mat} exceeds seq_len={samples.shape[1]}')
    log_cumsum = np.clip(samples[:, :T_mat].sum(axis=1).astype(np.float64), -500, 500)
    S_T = S0 * np.exp(log_cumsum)
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    if payoff_clip_pct is not None:
        cap = float(np.percentile(payoffs, payoff_clip_pct))
        payoffs = np.minimum(payoffs, cap)
    discount = np.exp(-r * T_mat)
    price = discount * float(payoffs.mean())
    mc_se = discount * float(payoffs.std() / np.sqrt(len(payoffs)))
    return (price, mc_se)

def compute_pricing_metrics(mc_prices: np.ndarray, bm_prices: np.ndarray) -> dict:
    mc = np.asarray(mc_prices, dtype=np.float64)
    bm = np.asarray(bm_prices, dtype=np.float64)
    mask = np.isfinite(mc) & np.isfinite(bm)
    (mc, bm) = (mc[mask], bm[mask])
    errors = mc - bm
    abs_err = np.abs(errors)
    rel_err = abs_err / np.maximum(bm, 1e-06)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mse = float(np.mean(errors.astype(np.float64) ** 2))
        rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float('inf')
    return {'rmse': rmse, 'mape': float(np.mean(rel_err) * 100), 'max_abs_error': float(abs_err.max()), 'bias': float(errors.mean()), 'n': int(mask.sum())}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--generated", required=True, type=str)
    p.add_argument("--simulator", required=True, choices=["gbm", "heston"])
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--moneyness", type=float, nargs="+",
                   default=[0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    p.add_argument("--maturities", type=int, nargs="+", default=[21, 63, 126, 252])
    p.add_argument("--option-type", choices=["call", "put"], default="call")
    p.add_argument("--gbm-sigma", type=float, default=0.20)
    p.add_argument("--heston-kappa", type=float, default=2.0)
    p.add_argument("--heston-theta", type=float, default=0.04)
    p.add_argument("--heston-xi", type=float, default=0.3)
    p.add_argument("--heston-rho", type=float, default=-0.7)
    p.add_argument("--heston-v0", type=float, default=0.04)
    args = p.parse_args()

    samples = np.load(args.generated)
    mc_prices, bm_prices = [], []
    for T_mat in args.maturities:
        for m in args.moneyness:
            K = m * args.S0
            T_yr = T_mat / 252.0
            mc_p, mc_se = mc_price_european(samples, args.S0, K, T_mat,
                                            args.option_type, args.r)
            if args.simulator == "gbm":
                bm_p = bs_price(args.S0, K, T_yr, args.r, args.gbm_sigma, args.option_type)
            else:
                bm_p = heston_price_carr_madan(
                    args.S0, K, T_yr, args.r,
                    args.heston_kappa, args.heston_theta, args.heston_xi,
                    args.heston_rho, args.heston_v0, args.option_type,
                )
            mc_prices.append(mc_p)
            bm_prices.append(bm_p)
            print(f"  K/S0={m:.2f}  T={T_mat:>3}d  mc={mc_p:>10.4f}  bm={bm_p:>10.4f}  |err|={abs(mc_p-bm_p):>10.4f}")

    metrics = compute_pricing_metrics(np.array(mc_prices), np.array(bm_prices))
    print(f"\nRMSE          : {metrics['rmse']:.4f}")
    print(f"MAPE          : {metrics['mape']:.2f}%")
    print(f"max abs error : {metrics['max_abs_error']:.4f}")
    print(f"bias          : {metrics['bias']:+.4f}")
    print(f"n contracts   : {metrics['n']}")


if __name__ == "__main__":
    main()
