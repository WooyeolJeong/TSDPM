import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1))
    emb = t.float()[:, None] * freqs[None]
    return torch.cat([emb.cos(), emb.sin()], dim=-1)

class _AttentionBlock(nn.Module):

    def __init__(self, channels: int, num_heads: int=8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x).transpose(1, 2)
        (h, _) = self.attn(h, h, h)
        return h.transpose(1, 2) + residual

def _continuous_sinusoidal_emb(s: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=s.device) / max(half - 1, 1))
    emb = s.float()[:, None] * freqs[None]
    return torch.cat([emb.cos(), emb.sin()], dim=-1)

class _ResBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        n_groups = min(8, out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(n_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.act = nn.SiLU()
        self.t_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t_emb).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class UNet1D(nn.Module):

    def __init__(self, channels: tuple[int, ...]=(64, 128, 256), time_emb_dim: int=128, seq_len: int=252, use_attn: bool=False, use_extra_cond: bool=False) -> None:
        super().__init__()
        self.seq_len = seq_len
        n_levels = len(channels)
        multiple_of = 2 ** n_levels
        self.pad_to = math.ceil(seq_len / multiple_of) * multiple_of
        self._t_dim = time_emb_dim
        self.use_extra_cond = use_extra_cond
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        if use_extra_cond:
            self.extra_cond_mlp = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        else:
            self.extra_cond_mlp = None
        self.init_conv = nn.Conv1d(1, channels[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.enc_blocks.append(nn.ModuleList([_ResBlock(in_ch, out_ch, time_emb_dim), _ResBlock(out_ch, out_ch, time_emb_dim)]))
            self.downsample.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch
        self.mid = nn.ModuleList([_ResBlock(in_ch, in_ch, time_emb_dim), _ResBlock(in_ch, in_ch, time_emb_dim)])
        self.mid_attn = _AttentionBlock(in_ch, num_heads=8) if use_attn else None
        self.dec_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for out_ch in reversed(channels):
            self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(in_ch, out_ch, 3, padding=1)))
            self.dec_blocks.append(nn.ModuleList([_ResBlock(out_ch * 2, out_ch, time_emb_dim), _ResBlock(out_ch, out_ch, time_emb_dim)]))
            in_ch = out_ch
        self.final_conv = nn.Conv1d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, extra_cond: torch.Tensor | None=None) -> torch.Tensor:
        pad = self.pad_to - self.seq_len
        if pad > 0:
            x = F.pad(x, (0, pad))
        t_emb = _sinusoidal_emb(t, self._t_dim)
        t_emb = self.time_mlp(t_emb)
        if extra_cond is not None:
            if self.extra_cond_mlp is None:
                raise RuntimeError('extra_cond passed but UNet1D was built with use_extra_cond=False')
            c_emb = _continuous_sinusoidal_emb(extra_cond, self._t_dim)
            t_emb = t_emb + self.extra_cond_mlp(c_emb)
        h = self.init_conv(x)
        skips = []
        for (blocks, down) in zip(self.enc_blocks, self.downsample):
            for blk in blocks:
                h = blk(h, t_emb)
            skips.append(h)
            h = down(h)
        h = self.mid[0](h, t_emb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid[1](h, t_emb)
        for (blocks, up, skip) in zip(self.dec_blocks, self.upsample, reversed(skips)):
            h = up(h)
            if h.shape[-1] > skip.shape[-1]:
                h = h[..., :skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, t_emb)
        return self.final_conv(h)[..., :self.seq_len]
