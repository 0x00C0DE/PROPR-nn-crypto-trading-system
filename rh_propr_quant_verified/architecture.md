# RH PROPR Quant Architecture

## What this version adds
This upgrade turns the prior RAME prototype into a more modern **PROPR** stack:

- **Predictive layer**
  - realized volatility and volatility slope
  - rolling trend slope
  - VWAP deviation
  - cross-asset relative momentum and lead/lag beta
  - pattern analogue matching from normalized rolling windows
- **Reactive layer**
  - volatility-adaptive leverage scaling
  - spread-aware trading pause
  - stress-state throttle
  - momentum exhaustion detection
  - volume burst and liquidity-gap response
- **Observational layer**
  - Bollinger compression / squeeze proxy
  - entropy proxy
  - Hurst-style persistence proxy
  - fractal-dimension proxy
  - CUSUM change detection
  - breakout state and structural break flags
- **Pattern recognition layer**
  - rolling price embeddings
  - cosine similarity against historical analogues
  - analogue directional edge estimate
- **Meta allocation layer**
  - softmax reallocation of engine weights using recent engine quality
- **Research layer**
  - cost-aware walk-forward backtest
  - Monte Carlo bootstrap tail-risk estimate

## Module map
- `engine/advanced_features.py` — feature engineering for PROPR
- `engine/patterns.py` — embedding and analogue matching
- `engine/meta.py` — adaptive signal allocator
- `engine/regime.py` — HMM-inspired regime proxy
- `engine/strategy.py` — PROPR ensemble strategy
- `engine/backtest.py` — walk-forward + cost + Monte Carlo diagnostics
- `engine/risk.py` — risk-state validation and leverage throttling
- `gui/main_window.py` — hedge-fund style dashboard

## Important honesty note
This is **not** a claim of live profitability, uniqueness, or production readiness.
It is a serious research architecture that is **closer to what modern crypto quant research stacks look like** than a single-rule strategy.

## Next production steps
- replace heuristic regime proxy with a trained HMM / switching state-space model
- persist feature snapshots and fills to SQLite/Postgres
- add real Robinhood endpoint reconciliation
- integrate async market-data and order-execution buses
- add hyperparameter search and full walk-forward parameter locking
- add slippage calibration from actual fills
