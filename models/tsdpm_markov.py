import warnings
import numpy as np
import torch
import torch.nn.functional as F
from .ts_sampler import normalisation_c, predicted_acceptance_rate, sample_TS_plus_E1

class TSDPM:

    def __init__(self, n_timesteps: int=1000, beta_schedule: str='linear', beta_start: float=0.0001, beta_end: float=0.02, beta_idx: float=0.7, lam: float=1.0) -> None:
        if beta_schedule != 'linear':
            raise ValueError(f"only 'linear' supported for now, got {beta_schedule}")
        if not 0.0 < beta_idx < 1.0:
            raise ValueError(f'beta_idx must be in (0, 1), got {beta_idx}')
        if lam <= 0.0:
            raise ValueError(f'lam must be > 0, got {lam}')
        if beta_idx < 0.3 or beta_idx > 0.95:
            warnings.warn(f'beta_idx={beta_idx} outside recommended [0.3, 0.95]; sampler may be inefficient or noisy.', stacklevel=2)
        if lam / beta_idx > 4.0:
            warnings.warn(f'lam/beta_idx = {lam / beta_idx:.2f} > 4 — BM acceptance rate ≈ {predicted_acceptance_rate(beta_idx, lam):.2e} is impractical.', stacklevel=2)
        self.T = n_timesteps
        self.beta_idx = beta_idx
        self.lam = lam
        self.c = normalisation_c(beta_idx, lam)
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float64)
        alphas_ = 1.0 - betas
        alpha_bars = np.cumprod(alphas_)
        self.betas = torch.from_numpy(betas.astype(np.float32))
        self.alphas = torch.from_numpy(alphas_.astype(np.float32))
        self.alpha_bars = torch.from_numpy(alpha_bars.astype(np.float32))

    def sample_T_sequence(self, n_samples: int, t_max: int | None=None, device: str='cpu', rng: np.random.Generator | None=None, max_iter: int=1500) -> torch.Tensor:
        if t_max is None:
            t_max = self.T
        if rng is None:
            rng = np.random.default_rng()
        flat = sample_TS_plus_E1(n=t_max * n_samples, beta_idx=self.beta_idx, lam=self.lam, rng=rng, max_iter=max_iter)
        T = torch.from_numpy(flat.astype(np.float32)).reshape(t_max, n_samples)
        return T.to(device)

    def compute_S_sequence(self, T_seq: torch.Tensor) -> torch.Tensor:
        device = T_seq.device
        t_max = T_seq.shape[0]
        alphas = self.alphas.to(device)[:t_max]
        betas = self.betas.to(device)[:t_max]
        S = torch.empty_like(T_seq)
        S[0] = betas[0] * T_seq[0]
        for t in range(1, t_max):
            S[t] = alphas[t] * S[t - 1] + betas[t] * T_seq[t]
        return S

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, S_t: torch.Tensor, noise: torch.Tensor | None=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        device = x_0.device
        sab = self.alpha_bars.to(device)[t].sqrt().view(-1, 1, 1)
        sS = S_t.to(device).sqrt().view(-1, 1, 1)
        return sab * x_0 + sS * noise

    def loss(self, score_net: torch.nn.Module, x_0: torch.Tensor, rng: np.random.Generator | None=None) -> torch.Tensor:
        batch = x_0.shape[0]
        device = x_0.device
        if rng is None:
            rng = np.random.default_rng()
        t = torch.randint(0, self.T, (batch,), device=device)
        T_seq = self.sample_T_sequence(batch, t_max=self.T, device=device, rng=rng)
        S_seq = self.compute_S_sequence(T_seq)
        S_t = S_seq.gather(0, t.view(1, -1)).squeeze(0)
        g = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, S_t, noise=g)
        eps_target = g
        self._last_eps_target_abs_max = float(eps_target.detach().abs().max().item())
        log_S_t = torch.log(S_t.clamp(min=1e-12))
        eps_pred = score_net(x_t, t, extra_cond=log_S_t)
        return F.mse_loss(eps_pred, eps_target)
