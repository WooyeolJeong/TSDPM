# TSDPM

Reference implementation for the paper "Tempered Stable Denoising Probabilistic Models for Heavy-Tailed Financial Time Series". DLPM baseline is reproduced based on Shariatian et al. (2024).

## Requirements

```
pip install -r requirements.txt
```

## Usage

```
python train.py --model variant --simulator heston --epochs 200
python evaluate/var_backtest.py --generated samples.npy --real test.npy --alpha 0.01
python evaluate/option_pricing.py --generated samples.npy --simulator heston
python evaluate/density_forecast.py --generated samples.npy --test test.npy --horizons 1 5 10 22
```
