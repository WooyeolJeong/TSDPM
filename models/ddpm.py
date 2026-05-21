import numpy as np
import torch
import torch.nn.functional as F

class DDPM:

    def __init__(self, n_timesteps: int=1000, beta_schedule: str='linear', beta_start: float=0.0001, beta_end: float=0.02) -> None:
        self.T = n_timesteps
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)
        alphas = (1.0 - betas).astype(np.float32)
        alpha_bars = np.cumprod(alphas).astype(np.float32)
        self.betas = torch.from_numpy(betas)
        self.alphas = torch.from_numpy(alphas)
        self.alpha_bars = torch.from_numpy(alpha_bars)
        self.sqrt_ab = torch.sqrt(self.alpha_bars)
        self.sqrt_1mab = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        dev = x_0.device
        sqrt_ab = self.sqrt_ab.to(dev)[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_1mab.to(dev)[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1mab * noise

    def loss(self, score_net: torch.nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        batch = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.T, (batch,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        return F.mse_loss(score_net(x_t, t), noise)
