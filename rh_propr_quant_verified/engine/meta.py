from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AdaptiveWeights:
    trend: float
    mean_reversion: float
    microstructure: float
    rotation: float
    pattern: float


class MetaSignalAllocator:
    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def allocate(self, recent_scores: Dict[str, float], regime_multiplier: float = 1.0) -> AdaptiveWeights:
        arr = np.array([
            recent_scores.get("trend", 0.0),
            recent_scores.get("mean_reversion", 0.0),
            recent_scores.get("microstructure", 0.0),
            recent_scores.get("rotation", 0.0),
            recent_scores.get("pattern", 0.0),
        ])
        weights = self.softmax(arr * regime_multiplier)
        return AdaptiveWeights(*[float(v) for v in weights])
