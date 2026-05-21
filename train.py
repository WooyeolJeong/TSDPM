import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from models.ddpm import DDPM
from models.dlpm import DLPM
from models.tsdpm_markov import TSDPM
from models.tsdpm_variant import TSDPMVariant
from models.unet import UNet1D
from simulators import (
    simulate_gbm,
    simulate_heston,
    simulate_rough_bergomi,
    simulate_variance_gamma,
    simulate_alpha_stable,
)


SIMULATOR_DEFAULTS = {
    "gbm":          {"mu": 0.05, "sigma": 0.20},
    "heston":       {"mu": 0.05, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7},
    "rbergomi":     {"mu": 0.05, "H": 0.1, "eta": 1.5, "rho": -0.7, "xi_0": 0.04},
    "vg":           {"mu": 0.05, "sigma": 0.20, "theta": -0.15, "nu": 0.20},
    "alpha_stable": {"mu": 0.05, "alpha": 1.7, "beta": 0.0, "scale": 0.01},
}

SIMULATOR_FNS = {
    "gbm":          simulate_gbm,
    "heston":       simulate_heston,
    "rbergomi":     simulate_rough_bergomi,
    "vg":           simulate_variance_gamma,
    "alpha_stable": simulate_alpha_stable,
}


def load_simulator_data(name, n_paths, seq_len, seed):
    fn = SIMULATOR_FNS[name]
    params = SIMULATOR_DEFAULTS[name]
    result = fn(n_paths=n_paths, n_steps=seq_len, seed=seed, **params)
    if isinstance(result, tuple):
        result = result[0]
    return result.astype(np.float32)


def load_sp500_data(seq_len, cache_path):
    import pandas as pd
    cache_path = Path(cache_path)
    if cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)["Close"]
    else:
        import yfinance as yf
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices = yf.download("^GSPC", start="2000-01-01", end="2019-12-31",
                             progress=False, auto_adjust=False)["Close"]
        prices.to_csv(cache_path)
    logret = np.log(prices.values).flatten()
    logret = np.diff(logret)
    n = len(logret) - seq_len + 1
    paths = np.stack([logret[i:i + seq_len] for i in range(n)])
    return paths.astype(np.float32)


def build_model(name, n_timesteps, beta_idx, lam, alpha):
    if name == "ddpm":
        return DDPM(n_timesteps=n_timesteps)
    if name == "dlpm":
        return DLPM(n_timesteps=n_timesteps, alpha=alpha)
    if name == "markov":
        return TSDPM(n_timesteps=n_timesteps, beta_idx=beta_idx, lam=lam)
    if name == "variant":
        return TSDPMVariant(n_timesteps=n_timesteps, beta_idx=beta_idx, lam=lam)
    raise ValueError(f"unknown model: {name}")


def train_loop(model, score_net, data, n_epochs, batch_size, lr, grad_clip,
               device, warmup_steps, target_clip, log_every, seed,
               ckpt_dir, ckpt_tag, ckpt_every):
    if seed is not None:
        torch.manual_seed(seed)
    x_all = torch.from_numpy(data.astype(np.float32)).unsqueeze(1)
    loader = DataLoader(TensorDataset(x_all), batch_size=batch_size, shuffle=True)
    score_net = score_net.to(device)
    optimizer = optim.Adam(score_net.parameters(), lr=lr)
    lr_start = lr / 40.0 if warmup_steps > 0 else lr

    def _lr_lambda(step):
        if warmup_steps <= 0 or step >= warmup_steps:
            return 1.0
        return lr_start / lr + (1.0 - lr_start / lr) * step / warmup_steps
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    is_dlpm = hasattr(model, "compute_Sigmas")
    is_tsdpm = hasattr(model, "compute_S_sequence")
    is_tsdpm_variant = hasattr(model, "T_max_clip")
    ts_rng = np.random.default_rng(seed if seed is not None else 0) if (is_tsdpm or is_tsdpm_variant) else None

    history = {"loss": []}
    global_step = 0
    score_net.train()
    t_start = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        for batch_idx, (x_0,) in enumerate(loader):
            x_0 = x_0.to(device)
            if is_tsdpm or is_tsdpm_variant:
                loss = model.loss(score_net, x_0, rng=ts_rng)
            elif is_dlpm:
                loss = model.loss(score_net, x_0, target_clip=target_clip)
            else:
                loss = model.loss(score_net, x_0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
            optimizer.zero_grad()
            loss.backward()
            gn = 0.0
            if grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(score_net.parameters(), grad_clip).item()
            optimizer.step()
            if global_step < warmup_steps:
                scheduler.step()
            epoch_loss += loss.item()
            epoch_grad_norm = max(epoch_grad_norm, gn)
            global_step += 1
        avg = epoch_loss / len(loader)
        history["loss"].append(avg)
        if (epoch + 1) % log_every == 0:
            elapsed = time.time() - t_start
            print(f"Epoch {epoch + 1:>4}/{n_epochs}  loss={avg:.6f}  grad={epoch_grad_norm:.4f}  elapsed={elapsed:.1f}s")
        if ckpt_dir is not None and ckpt_tag and (epoch + 1) % ckpt_every == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_p = ckpt_dir / f"{ckpt_tag}_ep{epoch + 1}.pt"
            torch.save(score_net.state_dict(), ckpt_p)
    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["ddpm", "dlpm", "markov", "variant"])
    p.add_argument("--simulator", required=True,
                   choices=["gbm", "heston", "rbergomi", "vg", "alpha_stable", "sp500"])
    p.add_argument("--n-paths", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=252)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--target-clip", type=float, default=10.0)
    p.add_argument("--T", type=int, default=200, dest="n_timesteps")
    p.add_argument("--beta-idx", type=float, default=0.7)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.95)
    p.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--use-attn", action="store_true", default=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--sp500-cache", default="data/sp500_cache.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.simulator == "sp500":
        data = load_sp500_data(args.seq_len, args.sp500_cache)
    else:
        data = load_simulator_data(args.simulator, args.n_paths, args.seq_len, args.seed)
    print(f"data: {data.shape}")

    use_extra_cond = args.model in ("markov", "variant")
    score_net = UNet1D(
        channels=tuple(args.channels),
        seq_len=args.seq_len,
        use_attn=args.use_attn,
        use_extra_cond=use_extra_cond,
    )
    n_params = sum(p.numel() for p in score_net.parameters())
    print(f"UNet1D: {n_params/1e6:.2f}M params")

    model = build_model(args.model, args.n_timesteps, args.beta_idx, args.lam, args.alpha)
    ckpt_tag = f"{args.simulator}_{args.model}_T{args.n_timesteps}_ep{args.epochs}_seed{args.seed}"
    ckpt_dir = Path(args.ckpt_dir)

    history = train_loop(
        model=model,
        score_net=score_net,
        data=data,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        device=args.device,
        warmup_steps=args.warmup_steps,
        target_clip=args.target_clip,
        log_every=args.log_every,
        seed=args.seed,
        ckpt_dir=ckpt_dir,
        ckpt_tag=ckpt_tag,
        ckpt_every=args.ckpt_every,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / f"{ckpt_tag}.pt"
    torch.save({
        "score_net": score_net.state_dict(),
        "config": vars(args),
        "history": history,
    }, final_path)
    print(f"saved: {final_path}")


if __name__ == "__main__":
    main()
