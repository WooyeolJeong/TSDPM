import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import levy_stable
_NORM_CACHE: dict[float, float] = {}

def _get_norm_constant(alpha: float, n_calib: int=100000, clip: float=1000000.0) -> float:
    if alpha not in _NORM_CACHE:
        rng = np.random.default_rng(0)
        A_raw = levy_stable.rvs(alpha / 2, beta=1, loc=0, scale=1, size=n_calib, random_state=rng)
        A_raw = np.clip(A_raw, 1e-10, clip).astype(np.float64)
        _NORM_CACHE[alpha] = float(np.median(A_raw ** (1.0 / alpha)))
    return _NORM_CACHE[alpha]

class DLPM:

    def __init__(self, n_timesteps: int=1000, beta_schedule: str='linear', beta_start: float=0.0001, beta_end: float=0.02, alpha: float=1.7, A_clip: float=1000000.0, use_median_norm: bool=False) -> None:
        self.T = n_timesteps
        self.alpha = alpha
        self.A_clip = A_clip
        self.use_median_norm = use_median_norm
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float64)
        alphas_ = 1.0 - betas
        alpha_bars = np.cumprod(alphas_)
        gammas = alphas_ ** (1.0 / alpha)
        bar_gammas = np.cumprod(gammas)
        bar_sigmas = (1.0 - bar_gammas ** alpha) ** (1.0 / alpha)
        bs_alpha = bar_sigmas ** alpha
        gm_alpha = gammas ** alpha
        ss_alpha = np.empty_like(bar_sigmas)
        ss_alpha[0] = bs_alpha[0]
        ss_alpha[1:] = np.clip(bs_alpha[1:] - gm_alpha[1:] * bs_alpha[:-1], 0.0, None)
        step_sigmas = ss_alpha ** (1.0 / alpha)
        self.betas = torch.from_numpy(betas.astype(np.float32))
        self.alphas = torch.from_numpy(alphas_.astype(np.float32))
        self.alpha_bars = torch.from_numpy(alpha_bars.astype(np.float32))
        self.bar_gammas = torch.from_numpy(bar_gammas.astype(np.float32))
        self.bar_sigmas = torch.from_numpy(bar_sigmas.astype(np.float32))
        self.gammas = torch.from_numpy(gammas.astype(np.float32))
        self.step_sigmas = torch.from_numpy(step_sigmas.astype(np.float32))
        self.sqrt_ab = self.bar_gammas
        self.sqrt_1mab = self.bar_sigmas
        self._C = 1.0 if alpha >= 2.0 - 1e-06 or not use_median_norm else _get_norm_constant(alpha)

    def sample_A(self, shape: tuple, device='cpu') -> torch.Tensor:
        if self.alpha >= 2.0 - 1e-06:
            return torch.ones(shape, device=device)
        A_raw = levy_stable.rvs(self.alpha / 2, beta=1, loc=0, scale=1, size=shape)
        A_raw = np.clip(A_raw, 1e-10, self.A_clip).astype(np.float32)
        if self.use_median_norm:
            A_raw = A_raw / float(self._C ** self.alpha)
        return torch.from_numpy(A_raw).to(device)

    def sample_A_sequence(self, n_timesteps: int, batch_size: int, device='cpu') -> torch.Tensor:
        return self.sample_A((n_timesteps, batch_size), device=device)

    def compute_Sigmas(self, A_seq: torch.Tensor) -> torch.Tensor:
        device = A_seq.device
        g = self.gammas.to(device)
        ss = self.step_sigmas.to(device)
        T = A_seq.shape[0]
        Sigmas = torch.empty_like(A_seq)
        Sigmas[0] = ss[0] ** 2 * A_seq[0]
        for t in range(1, T):
            Sigmas[t] = g[t] ** 2 * Sigmas[t - 1] + ss[t] ** 2 * A_seq[t]
        return Sigmas

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, A: torch.Tensor, noise_G: torch.Tensor | None=None) -> torch.Tensor:
        if noise_G is None:
            noise_G = torch.randn_like(x_0)
        dev = x_0.device
        bg = self.bar_gammas.to(dev)[t].view(-1, 1, 1)
        bs = self.bar_sigmas.to(dev)[t].view(-1, 1, 1)
        As = A.to(dev).pow(1.0 / self.alpha).view(-1, 1, 1)
        return bg * x_0 + bs * As * noise_G

    def loss(self, score_net: torch.nn.Module, x_0: torch.Tensor, target_clip: float=100.0, n_A_samples: int=1) -> torch.Tensor:
        batch = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.T, (batch,), device=device)
        losses = []
        for _ in range(n_A_samples):
            A = self.sample_A((batch,), device=device)
            G = torch.randn_like(x_0)
            eps_target = A.pow(1.0 / self.alpha).view(-1, 1, 1) * G
            eps_target = eps_target.clamp(-target_clip, target_clip)
            x_t = self.q_sample(x_0, t, A, G)
            losses.append(F.mse_loss(score_net(x_t, t), eps_target))
        return torch.stack(losses).mean()
