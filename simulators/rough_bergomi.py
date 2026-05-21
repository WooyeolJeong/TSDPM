import numpy as np

def _g(x: float, a: float) -> float:
    return x ** a

def _b(k: int, a: float) -> float:
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)

def _cov_matrix(a: float, dt: float) -> np.ndarray:
    c = np.empty((2, 2))
    c[0, 0] = dt
    c[0, 1] = dt ** (a + 1) / (a + 1)
    c[1, 0] = c[0, 1]
    c[1, 1] = dt ** (2 * a + 1) / (2 * a + 1)
    return c

def _build_G(n_steps: int, a: float, dt: float) -> np.ndarray:
    G = np.zeros(n_steps + 1)
    for k in range(2, n_steps + 1):
        G[k] = _g(_b(k, a) * dt, a)
    return G

def _Y_loop(dW1: np.ndarray, G: np.ndarray, n_steps: int, H: float) -> np.ndarray:
    n_paths = dW1.shape[0]
    Y1 = np.zeros((n_paths, n_steps + 1))
    for i in range(1, n_steps + 1):
        Y1[:, i] = dW1[:, i - 1, 1]
    X = dW1[:, :, 0]
    Y2 = np.zeros((n_paths, n_steps + 1))
    for i in range(n_paths):
        conv = np.convolve(G, X[i, :])
        Y2[i, :] = conv[:n_steps + 1]
    return np.sqrt(2.0 * H) * (Y1 + Y2)

def _Y_fft(dW1: np.ndarray, G: np.ndarray, n_steps: int, H: float) -> np.ndarray:
    n_paths = dW1.shape[0]
    Y1 = np.zeros((n_paths, n_steps + 1))
    Y1[:, 1:] = dW1[:, :, 1]
    n_fft = 1 << int(np.ceil(np.log2(2 * n_steps + 1)))
    X = dW1[:, :, 0]
    G_fft = np.fft.rfft(G, n=n_fft)
    X_fft = np.fft.rfft(X, n=n_fft, axis=1)
    GX = np.fft.irfft(G_fft[None, :] * X_fft, n=n_fft, axis=1)[:, :n_steps + 1]
    return np.sqrt(2.0 * H) * (Y1 + GX)

def simulate_rough_bergomi(n_paths: int, n_steps: int, mu: float=0.05, H: float=0.1, eta: float=1.5, rho: float=-0.7, xi_0: float=0.04, dt: float=1 / 252, seed: int | None=None, return_variance: bool=False) -> 'np.ndarray | tuple[np.ndarray, np.ndarray]':
    rng = np.random.default_rng(seed)
    a = H - 0.5
    cov_mat = _cov_matrix(a, dt)
    dW1 = rng.multivariate_normal(mean=np.zeros(2), cov=cov_mat, size=(n_paths, n_steps))
    dW2 = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    G = _build_G(n_steps, a, dt)
    Y = _Y_fft(dW1, G, n_steps, H)
    t_grid = np.arange(n_steps + 1, dtype=np.float64) * dt
    V = xi_0 * np.exp(eta * Y - 0.5 * eta ** 2 * t_grid[None, :] ** (2.0 * H))
    sqrt_1m_rho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))
    dB = rho * dW1[:, :, 0] + sqrt_1m_rho2 * dW2
    log_returns = (mu - 0.5 * V[:, :-1]) * dt + np.sqrt(V[:, :-1]) * dB
    if return_variance:
        return (log_returns, V)
    return log_returns
