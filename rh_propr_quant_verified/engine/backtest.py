from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .portfolio import target_notional_weights


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    summary: Dict[str, float]
    diagnostics: pd.DataFrame


class SimpleWalkForwardBacktester:
    def __init__(self, strategy, risk_manager, config):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config
        self.rng = np.random.default_rng(config.research.bootstrap_seed)

    def _monte_carlo_cvar(self, returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        sims = []
        arr = returns.to_numpy(dtype=float)
        for _ in range(self.config.research.monte_carlo_runs):
            sample = self.rng.choice(arr, size=len(arr), replace=True)
            sims.append(sample.sum())
        sims = np.sort(np.array(sims))
        tail = sims[: max(1, len(sims) // 20)]
        return float(tail.mean())

    def run(self, panel: Dict[str, pd.DataFrame], fee_bps: float = 15.0) -> BacktestResult:
        min_len = min(len(df) for df in panel.values())
        symbols = list(panel.keys())
        equity = self.config.starting_cash
        day_pnl = 0.0
        positions = {s: 0.0 for s in symbols}
        trades: List[dict] = []
        curve: List[dict] = []
        diagnostics: List[dict] = []

        aligned = {s: df.tail(min_len).reset_index(drop=True) for s, df in panel.items()}

        for i in range(max(140, self.config.research.walkforward_train // 2), min_len - 1):
            hist = {s: df.iloc[: i + 1].copy() for s, df in aligned.items()}
            spreads = {s: 8.0 + 4.0 * ((i + idx) % 7 == 0) for idx, s in enumerate(symbols)}
            stress_fraction = float(np.mean([(hist[s]["close"].pct_change().tail(8).std() or 0.0) > 0.018 for s in symbols]))
            risk_state = self.risk_manager.validate(equity, day_pnl, spreads, stress_fraction=stress_fraction)
            if not risk_state.trading_enabled:
                curve.append({"step": i, "equity": equity})
                diagnostics.append({"step": i, "regime": "paused", "leverage": 0.0, "stress_fraction": stress_fraction})
                continue

            signals = self.strategy.generate(hist, spreads)
            weights = target_notional_weights(signals, self.config.risk.max_gross_exposure, leverage_scalar=risk_state.leverage_scalar)
            next_ret = {s: float(aligned[s]["close"].iloc[i + 1] / aligned[s]["close"].iloc[i] - 1.0) for s in symbols}
            port_ret = sum(weights[s] * next_ret[s] for s in symbols)
            turnover = sum(abs(weights[s] - positions[s]) for s in symbols)
            avg_spread_cost = np.mean(list(spreads.values())) / 10000.0
            cost = turnover * ((fee_bps / 10000.0) + avg_spread_cost + (self.config.risk.slippage_bps / 10000.0))
            pnl = equity * (port_ret - cost)
            equity += pnl
            day_pnl += pnl

            dominant_regime = max(signals.values(), key=lambda x: abs(x.ensemble_score)).regime
            avg_conf = float(np.mean([sig.regime_confidence for sig in signals.values()]))
            diagnostics.append({
                "step": i,
                "regime": dominant_regime,
                "leverage": risk_state.leverage_scalar,
                "stress_fraction": stress_fraction,
                "avg_confidence": avg_conf,
            })

            for s in symbols:
                if abs(weights[s] - positions[s]) > self.config.strategy.rebalance_threshold:
                    trades.append({
                        "step": i,
                        "symbol": s,
                        "old_weight": positions[s],
                        "new_weight": weights[s],
                        "regime": signals[s].regime,
                        "signal": signals[s].ensemble_score,
                    })
            positions = weights
            curve.append({"step": i, "equity": equity})

        curve_df = pd.DataFrame(curve)
        trades_df = pd.DataFrame(trades)
        diag_df = pd.DataFrame(diagnostics)
        returns = curve_df["equity"].pct_change().fillna(0.0) if not curve_df.empty else pd.Series(dtype=float)
        drawdown = (curve_df["equity"] / curve_df["equity"].cummax() - 1.0).fillna(0.0) if not curve_df.empty else pd.Series(dtype=float)
        sharpe = float((returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)) if len(returns) else 0.0
        summary = {
            "ending_equity": float(equity),
            "return_pct": float((equity / self.config.starting_cash - 1.0) * 100.0),
            "trades": float(len(trades_df)),
            "avg_step_return": float(returns.mean() * 100.0) if len(returns) else 0.0,
            "sharpe": sharpe,
            "max_drawdown_pct": float(drawdown.min() * 100.0) if len(drawdown) else 0.0,
            "mc_cvar_pct": float(self._monte_carlo_cvar(returns) * 100.0),
        }
        return BacktestResult(curve_df, trades_df, summary, diag_df)
