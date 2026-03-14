from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskState:
    trading_enabled: bool = True
    reason: str = "ok"
    leverage_scalar: float = 1.0


class RiskManager:
    def __init__(self, config):
        self.config = config

    def validate(self, equity: float, day_pnl: float, spreads_bps: Dict[str, float], stress_fraction: float = 0.0) -> RiskState:
        if equity <= 0:
            return RiskState(False, "equity_depleted", 0.0)
        if day_pnl < 0 and abs(day_pnl) / max(equity, 1.0) >= self.config.daily_loss_limit:
            return RiskState(False, "daily_loss_limit", 0.0)
        if spreads_bps and min(spreads_bps.values()) > self.config.max_spread_bps:
            return RiskState(False, "all_spreads_too_wide", 0.0)
        leverage_scalar = max(0.35, self.config.max_leverage_scalar * (1.0 - 0.7 * stress_fraction))
        return RiskState(True, "ok", leverage_scalar)
