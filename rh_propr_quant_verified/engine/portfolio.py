from __future__ import annotations

from typing import Dict


def target_notional_weights(signals: Dict[str, object], max_gross: float, leverage_scalar: float = 1.0) -> Dict[str, float]:
    raw = {s: float(getattr(sig, "target_weight", 0.0)) for s, sig in signals.items()}
    gross = sum(abs(v) for v in raw.values()) or 1.0
    scale = min(1.0, max_gross / gross) * leverage_scalar
    return {s: raw[s] * scale for s in raw}
