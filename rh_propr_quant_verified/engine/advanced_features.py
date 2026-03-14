from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(window)


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = np.sum(x**2)
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1 : i + 1]
        y = y - y.mean()
        out[i] = float(np.sum(x * y) / denom)
    return pd.Series(out, index=series.index)


def rolling_entropy(returns: pd.Series, window: int, bins: int = 8) -> pd.Series:
    vals = returns.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        sample = vals[i - window + 1 : i + 1]
        hist, _ = np.histogram(sample, bins=bins, density=True)
        hist = hist[hist > 0]
        out[i] = float(-(hist * np.log(hist)).sum())
    return pd.Series(out, index=returns.index)


def fractal_dimension_proxy(series: pd.Series, window: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        sample = vals[i - window + 1 : i + 1]
        path = np.sum(np.abs(np.diff(sample)))
        displacement = abs(sample[-1] - sample[0]) + 1e-9
        out[i] = float(np.log(path + 1e-9) / np.log(displacement + 1.0000001))
    return pd.Series(out, index=series.index)


def hurst_proxy(series: pd.Series, window: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        sample = vals[i - window + 1 : i + 1]
        dev = sample - sample.mean()
        cumdev = np.cumsum(dev)
        rs = (cumdev.max() - cumdev.min()) / (sample.std() + 1e-9)
        out[i] = float(np.log(rs + 1e-9) / np.log(window))
    return pd.Series(out, index=series.index)


def bollinger_width(series: pd.Series, window: int, n_std: float = 2.0) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mean + n_std * std
    lower = mean - n_std * std
    return (upper - lower) / mean.replace(0, np.nan)


def cusum_events(returns: pd.Series, threshold: float) -> pd.Series:
    pos, neg = 0.0, 0.0
    out = []
    for r in returns.fillna(0.0):
        pos = max(0.0, pos + r)
        neg = min(0.0, neg + r)
        event = 1.0 if pos > threshold else -1.0 if neg < -threshold else 0.0
        if event != 0.0:
            pos, neg = 0.0, 0.0
        out.append(event)
    return pd.Series(out, index=returns.index)


def add_propr_features(df: pd.DataFrame, fast_ema: int, slow_ema: int, mr_window: int, vol_window: int, breakout_window: int) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["ema_fast"] = ema(out["close"], fast_ema)
    out["ema_slow"] = ema(out["close"], slow_ema)
    out["trend_strength"] = (out["ema_fast"] / out["ema_slow"] - 1.0).fillna(0.0)
    out["rv"] = realized_vol(out["ret"], vol_window).bfill().fillna(0.0)
    out["rv_slope"] = out["rv"].diff().fillna(0.0)
    out["z_mr"] = zscore(out["close"], mr_window).fillna(0.0)
    out["atr"] = atr(out, vol_window).bfill().fillna(0.0)
    out["atr_pct"] = (out["atr"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["range_high"] = out["high"].rolling(breakout_window).max().shift(1)
    out["range_low"] = out["low"].rolling(breakout_window).min().shift(1)
    out["compression"] = (
        out["close"].rolling(mr_window).std() / out["close"].rolling(breakout_window).std()
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["boll_width"] = bollinger_width(out["close"], mr_window).bfill().fillna(0.0)
    out["entropy"] = rolling_entropy(out["ret"], vol_window).bfill().fillna(0.0)
    out["fractal_dim"] = fractal_dimension_proxy(out["close"], mr_window).bfill().fillna(1.0)
    out["hurst"] = hurst_proxy(out["close"], mr_window).bfill().fillna(0.5)
    out["slope_short"] = rolling_slope(out["close"], max(8, mr_window // 2)).bfill().fillna(0.0)
    out["slope_long"] = rolling_slope(out["close"], breakout_window).bfill().fillna(0.0)
    out["vwap_proxy"] = (out["close"] * out["volume"]).rolling(mr_window).sum() / out["volume"].rolling(mr_window).sum().replace(0, np.nan)
    out["vwap_dev"] = ((out["close"] / out["vwap_proxy"]) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["volume_z"] = zscore(out["volume"].replace(0, np.nan).ffill().fillna(1.0), mr_window).fillna(0.0)
    out["volume_burst"] = (out["volume_z"] > 1.5).astype(float)
    out["price_accel"] = out["close"].diff().diff().fillna(0.0)
    out["momentum_exhaustion"] = ((out["z_mr"].abs() > 1.8) & (out["volume_z"] < 0)).astype(float)
    out["cusum_break"] = cusum_events(out["ret"], max(0.003, float(out["ret"].std() * 2.5 if len(out) else 0.003)))
    out["breakout_up"] = (out["close"] > out["range_high"]).astype(float)
    out["breakout_dn"] = (out["close"] < out["range_low"]).astype(float)
    out["liq_gap"] = ((out["high"] - out["low"]) / out["close"].replace(0, np.nan)).fillna(0.0)
    out["stress_flag"] = ((out["rv"] > out["rv"].rolling(vol_window).mean().bfill().fillna(0.0) * 1.4) | (out["liq_gap"] > out["atr_pct"] * 1.5)).astype(float)
    return out.bfill().fillna(0.0)


@dataclass
class CrossAssetSnapshot:
    relative_momentum: Dict[str, float]
    lead_lag_beta: Dict[str, float]
    dominance_score: Dict[str, float]


def build_cross_asset_features(panel: Dict[str, pd.DataFrame], lookback: int = 24) -> CrossAssetSnapshot:
    symbols = list(panel.keys())
    rets = {s: panel[s]["close"].pct_change().fillna(0.0) for s in symbols}
    momentum = {}
    beta = {}
    dom = {}
    basket = pd.concat(rets, axis=1).mean(axis=1)
    for s in symbols:
        series = rets[s]
        momentum[s] = float(panel[s]["close"].iloc[-1] / panel[s]["close"].iloc[-lookback] - 1.0) if len(panel[s]) > lookback else 0.0
        shifted = basket.shift(1).rolling(lookback).corr(series).iloc[-1] if len(series) > lookback + 1 else 0.0
        beta[s] = float(0.0 if pd.isna(shifted) else shifted)
        dom[s] = float((series.tail(lookback).mean() - basket.tail(lookback).mean()) / (basket.tail(lookback).std() + 1e-9)) if len(series) > lookback else 0.0
    vals = np.array(list(momentum.values()), dtype=float)
    mu = vals.mean() if len(vals) else 0.0
    sd = vals.std() if len(vals) else 1.0
    if sd == 0:
        sd = 1.0
    rel = {k: float((v - mu) / sd) for k, v in momentum.items()}
    return CrossAssetSnapshot(rel, beta, dom)
