import numpy as np

def simulate_gbm(n_paths: int, n_steps: int, mu: float=0.05, sigma: float=0.2, dt: float=1 / 252, seed: int | None=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = rng.standard_normal((n_paths, n_steps))
    return drift + diffusion * Z

def returns_to_prices(log_returns: np.ndarray, S0: float=100.0) -> np.ndarray:
    cum = np.concatenate([np.zeros((log_returns.shape[0], 1)), np.cumsum(log_returns, axis=1)], axis=1)
    return S0 * np.exp(cum)
