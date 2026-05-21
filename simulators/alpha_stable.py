import numpy as np
from scipy.stats import levy_stable

def simulate_alpha_stable(n_paths: int, n_steps: int, mu: float=0.05, alpha: float=1.7, beta: float=0.0, scale: float=0.01, dt: float=1 / 252, seed: int | None=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    step_scale = scale * dt ** (1.0 / alpha)
    step_loc = mu * dt
    return levy_stable.rvs(alpha=alpha, beta=beta, loc=step_loc, scale=step_scale, size=(n_paths, n_steps), random_state=rng)
