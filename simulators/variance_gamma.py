import numpy as np

def simulate_variance_gamma(n_paths: int, n_steps: int, mu: float=0.05, sigma: float=0.2, theta: float=-0.15, nu: float=0.2, dt: float=1 / 252, seed: int | None=None, apply_martingale_correction: bool=True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if apply_martingale_correction:
        arg = 1.0 - theta * nu - 0.5 * sigma ** 2 * nu
        if arg <= 0:
            raise ValueError(f'Martingale correction undefined: 1 - theta*nu - 0.5*sigma^2*nu = {arg:.6f} <= 0. Reduce |theta|, sigma, or nu.')
        omega = 1.0 / nu * np.log(arg)
    else:
        omega = 0.0
    Gdt = rng.gamma(shape=dt / nu, scale=nu, size=(n_paths, n_steps))
    Z = rng.standard_normal((n_paths, n_steps))
    return (mu + omega) * dt + theta * Gdt + sigma * np.sqrt(Gdt) * Z
