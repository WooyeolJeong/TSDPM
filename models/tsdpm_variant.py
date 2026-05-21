import warnings
import numpy as np
import torch
import torch.nn.functional as F
from .ts_sampler import normalisation_c, predicted_acceptance_rate, sample_TS_plus_E1

class TSDPMVariant:

    def __init__(self, n_timesteps: int=1000, beta_schedule: str='linear', beta_start: float=0.0001, beta_end: float=0.02, beta_idx: float=0.7, lam: float=1.0, T_max_clip: float=10.0) -> None:
        if beta_schedule != 'linear':
            raise ValueError(f"only 'linear' supported, got {beta_schedule!r}")
        if not 0.0 < beta_idx < 1.0:
            raise ValueError(f'beta_idx must be in (0, 1), got {beta_idx}')
        if lam <= 0.0:
            raise ValueError(f'lam must be > 0, got {lam}')
        if T_max_clip <= 0.0:
            raise ValueError(f'T_max_clip must be > 0, got {T_max_clip}')
        if beta_idx < 0.3 or beta_idx > 0.95:
            warnings.warn(f'beta_idx={beta_idx} outside recommended [0.3, 0.95]; sampler may be inefficient or noisy.', stacklevel=2)
        if lam / beta_idx > 4.0:
            warnings.warn(f'lam/beta_idx = {lam / beta_idx:.2f} > 4 — BM acceptance rate ≈ {predicted_acceptance_rate(beta_idx, lam):.2e} is impractical.', stacklevel=2)
        self.T = n_timesteps
        self.beta_idx = beta_idx
        self.lam = lam
        self.T_max_clip = T_max_clip
        self.c = normalisation_c(beta_idx, lam)
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float64)
        alphas_ = 1.0 - betas
        alpha_bars = np.cumprod(alphas_)
        self.betas = torch.from_numpy(betas.astype(np.float32))
        self.alphas = torch.from_numpy(alphas_.astype(np.float32))
        self.alpha_bars = torch.from_numpy(alpha_bars.astype(np.float32))
        self._last_eps_target_abs_max: float = 0.0
        self._last_T_max: float = 0.0
        self._last_T_clip_frac: float = 0.0

    def sample_T(self, batch_size: int, device: str='cpu', rng: np.random.Generator | None=None, max_iter: int=1500) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        raw = sample_TS_plus_E1(n=batch_size, beta_idx=self.beta_idx, lam=self.lam, rng=rng, max_iter=max_iter).astype(np.float32)
        n_clipped = int(np.sum(raw > self.T_max_clip))
        self._last_T_clip_frac = n_clipped / max(batch_size, 1)
        raw = np.clip(raw, 1e-06, self.T_max_clip)
        self._last_T_max = float(raw.max())
        return torch.from_numpy(raw).to(device)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, T: torch.Tensor, noise: torch.Tensor | None=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        device = x_0.device
        sab = self.alpha_bars.to(device)[t].sqrt().view(-1, 1, 1)
        s1mab = (1.0 - self.alpha_bars.to(device)[t]).sqrt().view(-1, 1, 1)
        sT = T.to(device).sqrt().view(-1, 1, 1)
        return sab * x_0 + s1mab * sT * noise

    def loss(self, score_net: torch.nn.Module, x_0: torch.Tensor, rng: np.random.Generator | None=None, target_clip: float=0.0) -> torch.Tensor:
        batch = x_0.shape[0]
        device = x_0.device
        if rng is None:
            rng = np.random.default_rng()
        t = torch.randint(0, self.T, (batch,), device=device)
        T = self.sample_T(batch, device=device, rng=rng)
        G = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, T, noise=G)
        self._last_eps_target_abs_max = float(G.detach().abs().max().item())
        log_T = torch.log(T.clamp(min=1e-12))
        eps_pred = score_net(x_t, t, extra_cond=log_T)
        return F.mse_loss(eps_pred, G)
