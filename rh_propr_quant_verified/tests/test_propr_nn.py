from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from engine.config import StrategyConfig
from engine.propr_nn import AdaptiveNeighborPolicy, MemoryCase
from engine.strategy import PROPRStrategy


def _sample_df(n: int = 220, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0.08, 0.9, size=n))
    close = pd.Series(base).clip(lower=10)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.9, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.9, size=n)
    volume = rng.integers(100, 1000, size=n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min").astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_neighbor_policy_learns_directional_bias() -> None:
    engine = AdaptiveNeighborPolicy(feature_names=["a", "b", "c"], k=3, max_cases=50, session_warm_start=False)
    state = np.array([1.0, 0.5, -0.2])
    for i in range(12):
        engine.register_pending("BTC", "long", state, i)
        engine.realize_pending("BTC", next_return=0.02, regime="trend_up", step_id=i + 1)
    decision = engine.score(state, step_id=50)
    assert decision.best_action == "long"
    assert decision.confidence > 0


def test_persistence_and_warm_start(tmp_path: Path) -> None:
    db_path = tmp_path / "propr_nn.sqlite3"
    state = np.array([0.2, -0.1, 0.8])
    engine = AdaptiveNeighborPolicy(feature_names=["a", "b", "c"], k=3, max_cases=50, memory_path=str(db_path))
    for i in range(6):
        engine.register_pending("BTC", "short", state, i)
        engine.realize_pending("BTC", next_return=-0.015, regime="trend_down", step_id=i + 1)
    engine.save_memory()

    reloaded = AdaptiveNeighborPolicy(feature_names=["a", "b", "c"], k=3, max_cases=50, memory_path=str(db_path))
    decision = reloaded.score(state, step_id=100)
    assert len(reloaded.memory) > 0
    assert decision.best_action == "short"


def test_decay_and_compression_reduce_stale_duplicate_cases() -> None:
    engine = AdaptiveNeighborPolicy(feature_names=["a", "b"], k=5, max_cases=100, prototype_threshold=0.95, session_warm_start=False)
    for i in range(12):
        state = np.array([1.0, 1.0 + 1e-4 * i])
        engine.memory.append(MemoryCase(vector=state.copy(), regime="trend", last_step=0, total_count=0.1, prototype=False))
        engine.memory[-1].update("long", 0.01, step_id=0, weight=0.1)
    engine._decay_memory(step_id=500)
    assert len(engine.memory) < 12

    for i in range(10):
        state = np.array([2.0, 2.0 + 1e-4 * i])
        engine.memory.append(MemoryCase(vector=state.copy(), regime="trend", last_step=100 + i, total_count=2.0, prototype=False))
        engine.memory[-1].update("long", 0.02, step_id=100 + i, weight=2.0)
    before = len(engine.memory)
    engine._compress_prototypes(step_id=200)
    assert len(engine.memory) < before


def test_strategy_exposes_memory_signal_fields(tmp_path: Path) -> None:
    cfg = StrategyConfig(neighbor_memory_path=str(tmp_path / "strat.sqlite3"))
    strat = PROPRStrategy(cfg)
    panel = {
        "BTC-USD": _sample_df(seed=1),
        "ETH-USD": _sample_df(seed=2),
        "SOL-USD": _sample_df(seed=4),
    }
    out1 = strat.generate(panel)
    out2 = strat.generate(panel)
    assert set(out1) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    snap = out2["BTC-USD"]
    assert isinstance(snap.memory_score, float)
    assert isinstance(snap.memory_confidence, float)
    assert isinstance(snap.memory_neighbors, int)
