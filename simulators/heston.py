import warnings
import numpy as np

def simulate_heston(n_paths: int, n_steps: int, mu: float=0.05, kappa: float=2.0, theta: float=0.04, xi: float=0.3, rho: float=-0.7, v0: float | None=None, dt: float=1 / 252, seed: int | None=None, return_variance: bool=False) -> 'np.ndarray | tuple[np.ndarray, np.ndarray]':
    if v0 is None:
        v0 = theta
    feller = 2.0 * kappa * theta
    if feller <= xi ** 2:
        warnings.warn(f'Feller condition violated: 2·κ·θ = {feller:.4f} ≤ ξ² = {xi ** 2:.4f}. Variance will touch zero more frequently; simulation remains valid.', RuntimeWarning, stacklevel=2)
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    sqrt_1m_rho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))
    W = rng.standard_normal((n_steps, n_paths, 2))
    Z_s = W[:, :, 0]
    Z_v = rho * W[:, :, 0] + sqrt_1m_rho2 * W[:, :, 1]
    log_returns = np.empty((n_paths, n_steps), dtype=np.float64)
    if return_variance:
        var_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        var_paths[:, 0] = max(v0, 0.0)
    v = np.full(n_paths, v0, dtype=np.float64)
    for t in range(n_steps):
        v_plus = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_plus)
        log_returns[:, t] = (mu - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * Z_s[t]
        v = v + kappa * (theta - v_plus) * dt + xi * sqrt_v * sqrt_dt * Z_v[t]
        if return_variance:
            var_paths[:, t + 1] = np.maximum(v, 0.0)
    if return_variance:
        return (log_returns, var_paths)
    return log_returns
