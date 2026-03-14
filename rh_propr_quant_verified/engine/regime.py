from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegimeWeights:
    trend: float
    mean_reversion: float
    microstructure: float
    rotation: float
    pattern: float
    risk_multiplier: float
    confidence: float
    label: str


class RegimeDetector:
    """HMM-inspired heuristic state classifier using observable emissions.
    This is not a full HMM fit; it is a stable regime-state proxy intended for live use.
    """

    def classify(
        self,
        trend_strength: float,
        rv: float,
        compression: float,
        spread_bps: float,
        entropy: float,
        hurst: float,
        stress_flag: float,
    ) -> RegimeWeights:
        if spread_bps > 35 or stress_flag > 0:
            return RegimeWeights(0.15, 0.10, 0.45, 0.10, 0.20, 0.45, 0.82, "stress")
        if abs(trend_strength) > 0.012 and hurst > 0.52 and compression < 0.95:
            return RegimeWeights(0.42, 0.08, 0.12, 0.18, 0.20, 1.0, 0.76, "trend")
        if compression < 0.72 and rv < 0.05:
            return RegimeWeights(0.22, 0.06, 0.20, 0.16, 0.36, 0.95, 0.68, "squeeze")
        if entropy > 8.0 or abs(trend_strength) < 0.003:
            return RegimeWeights(0.10, 0.42, 0.18, 0.14, 0.16, 0.78, 0.61, "chop")
        return RegimeWeights(0.20, 0.26, 0.18, 0.18, 0.18, 0.85, 0.55, "rotation")
