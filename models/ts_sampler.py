from functools import lru_cache
import numpy as np
from scipy.special import gamma as _gamma

@lru_cache(maxsize=None)
def normalisation_c(beta_idx: float, lam: float) -> float:
    if not 0.0 < beta_idx < 1.0:
        raise ValueError(f'beta_idx must be in (0, 1), got {beta_idx}')
    if lam <= 0.0:
        raise ValueError(f'lam must be > 0, got {lam}')
    return float(lam ** (1.0 - beta_idx) / _gamma(1.0 - beta_idx))

@lru_cache(maxsize=None)
def _proposal_scale(beta_idx: float, lam: float) -> float:
    return float((lam ** (1.0 - beta_idx) / beta_idx) ** (1.0 / beta_idx))

def predicted_acceptance_rate(beta_idx: float, lam: float) -> float:
    return float(np.exp(-lam / beta_idx))

def sample_positive_stable_CMS(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha must be in (0, 1) for one-sided stable, got {alpha}')
    W = rng.exponential(size=n)
    V = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=n)
    xi = np.pi / 2.0
    return np.sin(alpha * (V + xi)) / np.cos(V) ** (1.0 / alpha) * (np.cos(V - alpha * (V + xi)) / W) ** ((1.0 - alpha) / alpha)

def sample_TS_plus_E1(n: int, beta_idx: float, lam: float, rng: np.random.Generator, max_iter: int=100, return_diagnostics: bool=False) -> np.ndarray | tuple[np.ndarray, dict]:
    if n <= 0:
        raise ValueError(f'n must be > 0, got {n}')
    a = _proposal_scale(beta_idx, lam)
    out = np.empty(n, dtype=np.float64)
    todo = np.arange(n)
    total_proposals = 0
    n_iter = 0
    while todo.size > 0 and n_iter < max_iter:
        m = todo.size
        Y = sample_positive_stable_CMS(m, beta_idx, rng)
        Yp = a * Y
        U = rng.uniform(size=m)
        keep = U <= np.exp(-lam * Yp)
        accepted_idx = todo[keep]
        out[accepted_idx] = Yp[keep]
        todo = todo[~keep]
        total_proposals += m
        n_iter += 1
    if todo.size > 0:
        raise RuntimeError(f'BM rejection did not converge: {todo.size}/{n} unfilled after {max_iter} iterations.  Predicted acceptance ≈ exp(-lam/beta_idx) = {predicted_acceptance_rate(beta_idx, lam):.4e}.  Reduce lam, increase beta_idx, or raise max_iter.')
    if return_diagnostics:
        diag = {'acceptance_rate': n / total_proposals, 'n_iter': n_iter, 'total_proposals': total_proposals, 'predicted_rate': predicted_acceptance_rate(beta_idx, lam)}
        return (out, diag)
    return out
