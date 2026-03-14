from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .portfolio import target_notional_weights


@dataclass
class PaperTrade:
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    fee: float
    target_weight: float
    resulting_units: float


@dataclass
class PaperPortfolioState:
    cash: float
    equity: float
    net_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    positions: Dict[str, float] = field(default_factory=dict)
    avg_cost: Dict[str, float] = field(default_factory=dict)
    target_weights: Dict[str, float] = field(default_factory=dict)


class PaperTradingEngine:
    def __init__(self, strategy, risk_manager, config, fee_bps: float = 15.0):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config
        self.fee_bps = fee_bps
        self.reset()

    def reset(self):
        self.cash = float(self.config.starting_cash)
        self.starting_cash = float(self.config.starting_cash)
        self.realized_pnl = 0.0
        self.positions: Dict[str, float] = {}
        self.avg_cost: Dict[str, float] = {}
        self.target_weights: Dict[str, float] = {}
        self.trade_log: List[PaperTrade] = []
        self.step_index = 0
        self.running = False
        self.day_pnl = 0.0

    def load_panel(self, panel: Dict[str, pd.DataFrame]):
        min_len = min(len(df) for df in panel.values())
        self.panel = {s: df.tail(min_len).reset_index(drop=True).copy() for s, df in panel.items()}
        self.symbols = list(self.panel.keys())
        for s in self.symbols:
            self.positions.setdefault(s, 0.0)
            self.avg_cost.setdefault(s, 0.0)
            self.target_weights.setdefault(s, 0.0)
        self.step_index = max(140, self.config.research.walkforward_train // 2)

    def mark_prices(self) -> Dict[str, float]:
        idx = min(max(self.step_index, 0), min(len(df) for df in self.panel.values()) - 1)
        return {s: float(df.iloc[idx]["close"]) for s, df in self.panel.items()}

    def equity(self) -> float:
        prices = self.mark_prices()
        pos_val = sum(self.positions.get(s, 0.0) * prices[s] for s in prices)
        return float(self.cash + pos_val)

    def _unrealized_pnl(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for s, units in self.positions.items():
            if abs(units) < 1e-12:
                continue
            total += (prices[s] - self.avg_cost.get(s, prices[s])) * units
        return float(total)

    def snapshot(self) -> PaperPortfolioState:
        prices = self.mark_prices()
        eq = self.equity()
        unreal = self._unrealized_pnl(prices)
        return PaperPortfolioState(
            cash=float(self.cash),
            equity=float(eq),
            net_pnl=float(eq - self.starting_cash),
            realized_pnl=float(self.realized_pnl),
            unrealized_pnl=float(unreal),
            positions={k: float(v) for k, v in self.positions.items()},
            avg_cost={k: float(v) for k, v in self.avg_cost.items()},
            target_weights={k: float(v) for k, v in self.target_weights.items()},
        )

    def _execute_target(self, symbol: str, target_weight: float, price: float, timestamp: str) -> list[PaperTrade]:
        eq = self.equity()
        target_notional = eq * target_weight
        current_units = self.positions.get(symbol, 0.0)
        current_notional = current_units * price
        delta_notional = target_notional - current_notional
        if abs(delta_notional) < max(eq * self.config.strategy.rebalance_threshold, 10.0):
            return []

        side = "BUY" if delta_notional > 0 else "SELL"
        fee = abs(delta_notional) * (self.fee_bps / 10000.0)
        fill_price = price * (1.0 + (self.config.risk.slippage_bps / 10000.0) * (1 if side == "BUY" else -1))
        qty = abs(delta_notional) / max(fill_price, 1e-9)
        signed_qty = qty if side == "BUY" else -qty
        new_units = current_units + signed_qty

        # cash and realized pnl handling
        if side == "BUY":
            self.cash -= qty * fill_price + fee
            if abs(new_units) > 1e-12:
                old_cost = self.avg_cost.get(symbol, fill_price)
                old_units = max(current_units, 0.0)
                total_cost = old_cost * old_units + qty * fill_price
                total_units = old_units + qty
                self.avg_cost[symbol] = total_cost / max(total_units, 1e-12)
        else:
            self.cash += qty * fill_price - fee
            avg_cost = self.avg_cost.get(symbol, fill_price)
            sell_units = min(abs(current_units), qty)
            self.realized_pnl += (fill_price - avg_cost) * sell_units
            if abs(new_units) <= 1e-12:
                self.avg_cost[symbol] = 0.0

        self.positions[symbol] = new_units
        self.target_weights[symbol] = target_weight

        trade = PaperTrade(
            timestamp=str(timestamp),
            symbol=symbol,
            side=side,
            quantity=float(qty),
            price=float(fill_price),
            notional=float(abs(delta_notional)),
            fee=float(fee),
            target_weight=float(target_weight),
            resulting_units=float(new_units),
        )
        self.trade_log.append(trade)
        return [trade]

    def step(self) -> dict:
        if not getattr(self, "panel", None):
            raise RuntimeError("PaperTradingEngine has no market panel loaded")
        min_len = min(len(df) for df in self.panel.values())
        if self.step_index >= min_len - 1:
            self.running = False
            return {"done": True, "reason": "end_of_data", "trades": [], "signals": {}, "state": self.snapshot()}

        hist = {s: df.iloc[: self.step_index + 1].copy() for s, df in self.panel.items()}
        spreads = {s: 8.0 + 4.0 * ((self.step_index + idx) % 7 == 0) for idx, s in enumerate(self.symbols)}
        stress_fraction = float(np.mean([(hist[s]["close"].pct_change().tail(8).std() or 0.0) > 0.018 for s in self.symbols]))
        risk_state = self.risk_manager.validate(self.equity(), self.day_pnl, spreads, stress_fraction=stress_fraction)
        timestamp = str(next(iter(hist.values())).iloc[-1]["timestamp"])
        if not risk_state.trading_enabled:
            self.step_index += 1
            return {
                "done": False,
                "reason": risk_state.reason,
                "trades": [],
                "signals": {},
                "state": self.snapshot(),
                "timestamp": timestamp,
            }

        signals = self.strategy.generate(hist, spreads)
        weights = target_notional_weights(signals, self.config.risk.max_gross_exposure, leverage_scalar=risk_state.leverage_scalar)
        prices = {s: float(self.panel[s].iloc[self.step_index]["close"]) for s in self.symbols}
        trades: list[PaperTrade] = []
        for s in self.symbols:
            trades.extend(self._execute_target(s, weights[s], prices[s], timestamp))

        prior_equity = self.equity()
        self.step_index += 1
        new_equity = self.equity()
        self.day_pnl += new_equity - prior_equity
        return {
            "done": False,
            "reason": "ok",
            "trades": trades,
            "signals": signals,
            "state": self.snapshot(),
            "timestamp": timestamp,
        }
