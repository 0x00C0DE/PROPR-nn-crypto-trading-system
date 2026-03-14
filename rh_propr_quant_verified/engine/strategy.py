from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .advanced_features import add_propr_features, build_cross_asset_features
from .meta import MetaSignalAllocator
from .patterns import PatternLibrary
from .regime import RegimeDetector
from .propr_nn import AdaptiveNeighborPolicy


@dataclass
class SignalSnapshot:
    symbol: str
    regime: str
    regime_confidence: float
    trend_score: float
    mr_score: float
    micro_score: float
    rotation_score: float
    pattern_score: float
    memory_score: float
    ensemble_score: float
    target_weight: float
    pattern_similarity: float
    stress_flag: float
    memory_confidence: float
    memory_neighbors: int


class PROPRStrategy:
    def __init__(self, config):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.patterns = PatternLibrary(window=config.pattern_window)
        self.meta = MetaSignalAllocator()
        self.step_id = 0
        self.feature_names = [
            "trend_strength", "z_mr", "rv", "rv_slope", "atr_pct", "compression", "boll_width",
            "entropy", "fractal_dim", "hurst", "slope_short", "slope_long", "vwap_dev",
            "volume_z", "price_accel", "momentum_exhaustion", "cusum_break", "liq_gap",
            "cross_relative_momentum", "cross_dominance", "cross_beta", "spread_bps",
            "regime_confidence", "pattern_similarity", "stress_flag"
        ]
        self.memory = AdaptiveNeighborPolicy(
            feature_names=self.feature_names,
            k=getattr(config, "neighbor_k", 15),
            max_cases=getattr(config, "neighbor_max_cases", 1200),
            prototype_threshold=getattr(config, "neighbor_prototype_threshold", 0.82),
            exploration=getattr(config, "neighbor_exploration", 0.18),
            recency_halflife=getattr(config, "neighbor_recency_halflife", 240),
            memory_path=getattr(config, "neighbor_memory_path", None),
            session_warm_start=getattr(config, "neighbor_session_warm_start", True),
        )

    def reset(self):
        self.step_id = 0
        self.memory.reset()

    def _recent_engine_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        future = df["ret"].shift(-1).fillna(0.0)
        quality = {
            "trend": float((np.sign(df["trend_strength"].tail(self.config.meta_lookback)) * future.tail(self.config.meta_lookback)).mean()),
            "mean_reversion": float((np.sign(-df["z_mr"].tail(self.config.meta_lookback)) * future.tail(self.config.meta_lookback)).mean()),
            "microstructure": float((np.sign(df["volume_burst"].tail(self.config.meta_lookback) - 0.5) * future.tail(self.config.meta_lookback)).mean()),
            "rotation": float((np.sign(df["slope_long"].tail(self.config.meta_lookback)) * future.tail(self.config.meta_lookback)).mean()),
            "pattern": float((np.sign(df["breakout_up"].tail(self.config.meta_lookback) - df["breakout_dn"].tail(self.config.meta_lookback)) * future.tail(self.config.meta_lookback)).mean()),
        }
        return quality

    def _resolve_action(self, ensemble: float, memory_signal: float) -> str:
        composite = 0.65 * ensemble + 0.35 * memory_signal
        if composite > 0.06:
            return "long"
        if composite < -0.06:
            return "short"
        if abs(composite) < 0.02:
            return "flat"
        return "reduce"

    def _encode_state(self, row: pd.Series, cross, symbol: str, spread: float, regime_conf: float, pattern_similarity: float) -> np.ndarray:
        values = [
            float(row["trend_strength"]),
            float(row["z_mr"]),
            float(row["rv"]),
            float(row["rv_slope"]),
            float(row["atr_pct"]),
            float(row["compression"]),
            float(row["boll_width"]),
            float(row["entropy"]),
            float(row["fractal_dim"]),
            float(row["hurst"]),
            float(row["slope_short"] / max(abs(row["close"]), 1e-9)),
            float(row["slope_long"] / max(abs(row["close"]), 1e-9)),
            float(row["vwap_dev"]),
            float(row["volume_z"]),
            float(row["price_accel"] / max(abs(row["atr"]), 1e-9)),
            float(row["momentum_exhaustion"]),
            float(row["cusum_break"]),
            float(row["liq_gap"]),
            float(cross.relative_momentum.get(symbol, 0.0)),
            float(cross.dominance_score.get(symbol, 0.0)),
            float(cross.lead_lag_beta.get(symbol, 0.0)),
            float(spread / 100.0),
            float(regime_conf),
            float(pattern_similarity),
            float(row["stress_flag"]),
        ]
        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def generate(self, panel: Dict[str, pd.DataFrame], spread_bps: Dict[str, float] | None = None) -> Dict[str, SignalSnapshot]:
        spread_bps = spread_bps or {symbol: 10.0 for symbol in panel}
        feats = {
            symbol: add_propr_features(
                df,
                self.config.fast_ema,
                self.config.slow_ema,
                self.config.mr_window,
                self.config.vol_window,
                self.config.breakout_window,
            )
            for symbol, df in panel.items()
        }
        pattern_matches = self.patterns.panel_snapshot(feats)
        cross = build_cross_asset_features(feats, lookback=self.config.cross_sectional_lookback)

        # Online self-reinforcement: realize the prior step for each symbol once the newest bar is known.
        for symbol, df in feats.items():
            if len(df) >= 2:
                realized_ret = float(df["ret"].iloc[-1])
                self.memory.realize_pending(symbol, realized_ret, "observed", self.step_id)

        raw = {}
        for symbol, df in feats.items():
            row = df.iloc[-1]
            regime = self.regime_detector.classify(
                float(row["trend_strength"]),
                float(row["rv"]),
                float(row["compression"]),
                float(spread_bps.get(symbol, 10.0)),
                float(row["entropy"]),
                float(row["hurst"]),
                float(row["stress_flag"]),
            )
            quality = self._recent_engine_quality(df)
            alloc = self.meta.allocate(quality, regime_multiplier=regime.confidence)
            pat = pattern_matches[symbol]
            state_vec = self._encode_state(row, cross, symbol, float(spread_bps.get(symbol, 10.0)), regime.confidence, pat.similarity)
            memory_decision = self.memory.score(state_vec, self.step_id)

            trend_score = float(np.tanh(row["trend_strength"] * 120 + row["slope_long"] / max(row["close"], 1.0) * 180))
            mr_score = float(np.tanh(-row["z_mr"] + row["vwap_dev"] * -8))
            micro_score = float(np.tanh((row["volume_z"] * 0.6) + (row["price_accel"] / max(row["atr"], 1e-9)) - row["momentum_exhaustion"]))
            rotation_score = float(np.tanh(cross.relative_momentum.get(symbol, 0.0) + 0.5 * cross.dominance_score.get(symbol, 0.0) + 0.25 * cross.lead_lag_beta.get(symbol, 0.0)))
            pattern_score = float(np.tanh(pat.directional_edge * 60) * max(0.0, pat.similarity))

            base_ensemble = (
                regime.trend * alloc.trend * trend_score
                + regime.mean_reversion * alloc.mean_reversion * mr_score
                + regime.microstructure * alloc.microstructure * micro_score
                + regime.rotation * alloc.rotation * rotation_score
                + regime.pattern * alloc.pattern * pattern_score
            ) * regime.risk_multiplier
            memory_blend = getattr(self.config, "neighbor_memory_blend", 0.28) * memory_decision.signal
            ensemble = base_ensemble + memory_blend

            if pat.similarity < self.config.similarity_threshold:
                ensemble *= 0.85
            if regime.confidence < self.config.regime_confidence_floor:
                ensemble *= 0.75

            action = self._resolve_action(base_ensemble, memory_decision.signal)
            self.memory.register_pending(symbol, action, state_vec, self.step_id)
            raw[symbol] = SignalSnapshot(
                symbol=symbol,
                regime=regime.label,
                regime_confidence=float(regime.confidence),
                trend_score=trend_score,
                mr_score=mr_score,
                micro_score=micro_score,
                rotation_score=rotation_score,
                pattern_score=pattern_score,
                memory_score=float(memory_decision.signal),
                ensemble_score=float(ensemble),
                target_weight=0.0,
                pattern_similarity=float(pat.similarity),
                stress_flag=float(row["stress_flag"]),
                memory_confidence=float(memory_decision.confidence),
                memory_neighbors=int(memory_decision.neighbor_count),
            )

        gross = sum(abs(s.ensemble_score) for s in raw.values()) or 1.0
        max_w = getattr(self.config, "max_symbol_weight", 0.24)
        for symbol, sig in raw.items():
            sig.target_weight = float(max(-max_w, min(max_w, sig.ensemble_score / gross)))
        self.step_id += 1
        return raw
