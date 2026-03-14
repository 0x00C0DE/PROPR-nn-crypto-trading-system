from dataclasses import dataclass, field
from typing import List


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.004
    max_symbol_weight: float = 0.24
    max_gross_exposure: float = 1.00
    daily_loss_limit: float = 0.03
    max_spread_bps: float = 25.0
    min_cash_buffer: float = 0.05
    max_leverage_scalar: float = 1.15
    slippage_bps: float = 8.0


@dataclass
class StrategyConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    fast_ema: int = 16
    slow_ema: int = 64
    mr_window: int = 20
    vol_window: int = 32
    breakout_window: int = 48
    pattern_window: int = 40
    meta_lookback: int = 72
    cross_sectional_lookback: int = 24
    rebalance_threshold: float = 0.025
    similarity_threshold: float = 0.65
    regime_confidence_floor: float = 0.50
    neighbor_k: int = 15
    neighbor_max_cases: int = 1200
    neighbor_prototype_threshold: float = 0.82
    neighbor_exploration: float = 0.18
    neighbor_recency_halflife: int = 240
    neighbor_memory_blend: float = 0.28
    neighbor_memory_path: str = "data/propr_nn_memory.sqlite3"
    neighbor_session_warm_start: bool = True


@dataclass
class ResearchConfig:
    walkforward_train: int = 220
    walkforward_test: int = 80
    monte_carlo_runs: int = 200
    bootstrap_seed: int = 7


@dataclass
class AppConfig:
    mode: str = "paper"
    starting_cash: float = 25000.0
    timeframe: str = "15m"
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
