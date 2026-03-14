from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PatternMatch:
    similarity: float
    directional_edge: float
    volatility_edge: float
    label: str


class PatternLibrary:
    """Simple price-pattern embedding library using normalized rolling windows."""

    def __init__(self, window: int = 40):
        self.window = window

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = values.astype(float)
        std = values.std() + 1e-9
        return (values - values.mean()) / std

    def _windows(self, closes: pd.Series, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
        vals = closes.to_numpy(dtype=float)
        patterns, future = [], []
        for i in range(self.window, len(vals) - horizon):
            patterns.append(self._normalize(vals[i - self.window : i]))
            fut_ret = vals[i + horizon] / vals[i] - 1.0
            future.append(fut_ret)
        if not patterns:
            return np.empty((0, self.window)), np.empty((0,))
        return np.vstack(patterns), np.array(future)

    def match(self, df: pd.DataFrame, horizon: int = 5) -> PatternMatch:
        patterns, future = self._windows(df["close"], horizon=horizon)
        if len(patterns) < 10:
            return PatternMatch(0.0, 0.0, 0.0, "insufficient_history")
        current = self._normalize(df["close"].iloc[-self.window :].to_numpy())
        sims = patterns @ current / (np.linalg.norm(patterns, axis=1) * np.linalg.norm(current) + 1e-9)
        idx = np.argsort(sims)[-8:]
        top_sims = sims[idx]
        top_future = future[idx]
        similarity = float(np.mean(top_sims))
        directional_edge = float(np.mean(top_future))
        volatility_edge = float(np.std(top_future))
        label = "bullish_analogue" if directional_edge > 0 else "bearish_analogue"
        return PatternMatch(similarity, directional_edge, volatility_edge, label)

    def panel_snapshot(self, panel: Dict[str, pd.DataFrame]) -> Dict[str, PatternMatch]:
        return {symbol: self.match(df) for symbol, df in panel.items()}
