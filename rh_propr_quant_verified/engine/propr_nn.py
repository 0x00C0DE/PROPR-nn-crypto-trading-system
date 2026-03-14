from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List
import json
import sqlite3

import numpy as np


ACTIONS = ("long", "short", "flat", "reduce")
ACTION_TO_SIGNAL = {
    "long": 1.0,
    "short": -1.0,
    "flat": 0.0,
    "reduce": 0.35,
}


@dataclass
class RewardStats:
    mean: float = 0.0
    sq_mean: float = 0.0
    count: float = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        weight = max(1e-9, float(weight))
        total = self.count + weight
        self.mean = (self.mean * self.count + value * weight) / total
        self.sq_mean = (self.sq_mean * self.count + (value**2) * weight) / total
        self.count = total

    def apply_decay(self, factor: float) -> None:
        factor = min(1.0, max(0.0, float(factor)))
        self.count *= factor

    @property
    def variance(self) -> float:
        return max(0.0, self.sq_mean - self.mean**2)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "sq_mean": self.sq_mean, "count": self.count}

    @classmethod
    def from_dict(cls, data: dict) -> "RewardStats":
        return cls(mean=float(data.get("mean", 0.0)), sq_mean=float(data.get("sq_mean", 0.0)), count=float(data.get("count", 0.0)))


@dataclass
class MemoryCase:
    vector: np.ndarray
    action_stats: Dict[str, RewardStats] = field(default_factory=lambda: {a: RewardStats() for a in ACTIONS})
    total_count: float = 0.0
    regime: str = "unknown"
    last_step: int = 0
    reliability: float = 0.0
    prototype: bool = False

    def update(self, action: str, reward: float, step_id: int, weight: float = 1.0) -> None:
        self.action_stats[action].update(reward, weight=weight)
        self.total_count += weight
        self.last_step = step_id
        pos = self.action_stats["long"].mean + self.action_stats["short"].mean
        self.reliability = float(np.tanh(abs(pos) * np.sqrt(max(self.total_count, 1.0))))

    def apply_decay(self, factor: float) -> None:
        factor = min(1.0, max(0.0, float(factor)))
        for stats in self.action_stats.values():
            stats.apply_decay(factor)
        self.total_count *= factor
        self.reliability *= np.sqrt(factor)

    def to_record(self) -> dict:
        return {
            "vector": self.vector.tolist(),
            "action_stats": {k: v.to_dict() for k, v in self.action_stats.items()},
            "total_count": self.total_count,
            "regime": self.regime,
            "last_step": self.last_step,
            "reliability": self.reliability,
            "prototype": self.prototype,
        }

    @classmethod
    def from_record(cls, record: dict) -> "MemoryCase":
        stats = {a: RewardStats.from_dict(record.get("action_stats", {}).get(a, {})) for a in ACTIONS}
        return cls(
            vector=np.array(record.get("vector", []), dtype=float),
            action_stats=stats,
            total_count=float(record.get("total_count", 0.0)),
            regime=str(record.get("regime", "unknown")),
            last_step=int(record.get("last_step", 0)),
            reliability=float(record.get("reliability", 0.0)),
            prototype=bool(record.get("prototype", False)),
        )


@dataclass
class NeighborDecision:
    action_scores: Dict[str, float]
    best_action: str
    signal: float
    confidence: float
    neighbor_count: int
    dispersion: float


class AdaptiveNeighborPolicy:
    """Contextual bandit + nearest-neighbor memory with persistence and maintenance.

    This is intentionally non-neural. It behaves like a trainable policy by:
    - storing contextual market states
    - retrieving similar prior states
    - reinforcing local rewards online
    - learning feature weights from memory effectiveness
    - persisting state across restarts (session warm-start)
    - decaying stale memory
    - compressing repeated patterns into prototypes
    """

    def __init__(
        self,
        feature_names: Iterable[str],
        k: int = 15,
        max_cases: int = 1200,
        prototype_threshold: float = 0.82,
        exploration: float = 0.18,
        recency_halflife: int = 240,
        memory_path: str | None = None,
        session_warm_start: bool = True,
        autosave: bool = True,
    ) -> None:
        self.feature_names = list(feature_names)
        self.k = int(k)
        self.max_cases = int(max_cases)
        self.prototype_threshold = float(prototype_threshold)
        self.exploration = float(exploration)
        self.recency_halflife = int(recency_halflife)
        self.memory_path = memory_path
        self.session_warm_start = bool(session_warm_start)
        self.autosave = bool(autosave)
        self.memory: List[MemoryCase] = []
        self.pending: Dict[str, tuple[np.ndarray, str, int]] = {}
        self.feature_weights = np.ones(len(self.feature_names), dtype=float)
        self._maintenance_counter = 0
        self._ensure_store()
        if self.session_warm_start:
            self.load_memory()

    def _ensure_store(self) -> None:
        if not self.memory_path:
            return
        path = Path(self.memory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS neighbor_memory (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    feature_weights TEXT NOT NULL,
                    saved_step INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def reset(self) -> None:
        self.memory.clear()
        self.pending.clear()
        self.feature_weights = np.ones(len(self.feature_names), dtype=float)
        self._maintenance_counter = 0

    def load_memory(self) -> None:
        if not self.memory_path:
            return
        path = Path(self.memory_path)
        if not path.exists():
            return
        with sqlite3.connect(path) as conn:
            row = conn.execute("SELECT payload, feature_weights FROM neighbor_memory WHERE id = 1").fetchone()
        if not row:
            return
        payload, feature_weights = row
        try:
            records = json.loads(payload)
            self.memory = [MemoryCase.from_record(r) for r in records]
            self.feature_weights = np.array(json.loads(feature_weights), dtype=float)
            if self.feature_weights.shape[0] != len(self.feature_names):
                self.feature_weights = np.ones(len(self.feature_names), dtype=float)
        except Exception:
            self.memory = []
            self.feature_weights = np.ones(len(self.feature_names), dtype=float)

    def save_memory(self) -> None:
        if not self.memory_path:
            return
        path = Path(self.memory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([case.to_record() for case in self.memory])
        weights = json.dumps(self.feature_weights.tolist())
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                INSERT INTO neighbor_memory (id, payload, feature_weights, saved_step)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload = excluded.payload,
                    feature_weights = excluded.feature_weights,
                    saved_step = excluded.saved_step,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (payload, weights, int(max((c.last_step for c in self.memory), default=0))),
            )
            conn.commit()

    def _similarity(self, current: np.ndarray, historical: np.ndarray) -> float:
        weights = self.feature_weights / np.maximum(self.feature_weights.sum(), 1e-9)
        delta = (current - historical) * np.sqrt(weights)
        dist = float(np.sqrt(np.dot(delta, delta)))
        return float(1.0 / (1.0 + dist))

    def _recency_weight(self, step_gap: int) -> float:
        return float(0.5 ** (max(step_gap, 0) / max(self.recency_halflife, 1)))

    def _nearest(self, state: np.ndarray, step_id: int) -> List[tuple[float, MemoryCase]]:
        ranked: List[tuple[float, MemoryCase]] = []
        for case in self.memory:
            sim = self._similarity(state, case.vector)
            sim *= self._recency_weight(step_id - case.last_step)
            sim *= 0.65 + 0.35 * case.reliability
            ranked.append((sim, case))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[: self.k]

    def score(self, state: np.ndarray, step_id: int) -> NeighborDecision:
        neighbors = self._nearest(state, step_id)
        scores = {a: 0.0 for a in ACTIONS}
        dispersion_vals: List[float] = []

        if not neighbors:
            return NeighborDecision(scores, "flat", 0.0, 0.0, 0, 1.0)

        for action in ACTIONS:
            weighted_rewards = []
            total_w = 0.0
            for sim, case in neighbors:
                stats = case.action_stats[action]
                if stats.count <= 0:
                    continue
                exploit = stats.mean
                explore = self.exploration * np.sqrt(np.log(case.total_count + 2.0) / (stats.count + 1.0))
                reward = exploit + explore
                w = sim * (0.25 + 0.75 * min(1.0, stats.count / 5.0))
                weighted_rewards.append((w, reward))
                total_w += w
            if total_w > 0:
                scores[action] = float(sum(w * r for w, r in weighted_rewards) / total_w)
                dispersion_vals.append(float(np.std([r for _, r in weighted_rewards])) if len(weighted_rewards) > 1 else 0.0)

        order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_action = order[0][0]
        margin = order[0][1] - order[1][1] if len(order) > 1 else order[0][1]
        confidence = float(np.tanh(max(0.0, margin) * 8.0) * min(1.0, len(neighbors) / max(3.0, self.k / 2)))
        dispersion = float(np.mean(dispersion_vals)) if dispersion_vals else 1.0
        signal = ACTION_TO_SIGNAL.get(best_action, 0.0) * confidence
        return NeighborDecision(scores, best_action, float(signal), confidence, len(neighbors), dispersion)

    def _insert_case(self, state: np.ndarray, action: str, reward: float, regime: str, step_id: int) -> None:
        if self.memory:
            nearest = self._nearest(state, step_id)
            if nearest and nearest[0][0] >= self.prototype_threshold:
                nearest[0][1].vector = 0.85 * nearest[0][1].vector + 0.15 * state
                nearest[0][1].prototype = True
                nearest[0][1].regime = regime
                nearest[0][1].update(action, reward, step_id, weight=1.0)
                return
        case = MemoryCase(vector=state.copy(), regime=regime, last_step=step_id)
        case.update(action, reward, step_id, weight=1.0)
        self.memory.append(case)
        if len(self.memory) > self.max_cases:
            self.memory.sort(key=lambda c: (c.reliability, c.total_count, c.last_step))
            self.memory = self.memory[-self.max_cases :]

    def _propagate_reward(self, state: np.ndarray, action: str, reward: float, step_id: int) -> None:
        for sim, case in self._nearest(state, step_id):
            if sim < 0.35:
                continue
            case.update(action, reward, step_id, weight=sim * 0.6)
            if reward < 0 and action in ("long", "short"):
                opposite = "short" if action == "long" else "long"
                case.update(opposite, -reward * 0.25, step_id, weight=sim * 0.2)

    def _refresh_feature_weights(self) -> None:
        if len(self.memory) < 20:
            return
        vectors = np.vstack([c.vector for c in self.memory])
        rewards = np.array([
            max(c.action_stats["long"].mean, c.action_stats["short"].mean, c.action_stats["flat"].mean)
            for c in self.memory
        ], dtype=float)
        rewards_std = rewards.std()
        if rewards_std <= 1e-9:
            return
        new_weights = np.ones_like(self.feature_weights)
        for i in range(vectors.shape[1]):
            col = vectors[:, i]
            if np.std(col) <= 1e-9:
                continue
            corr = np.corrcoef(col, rewards)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            new_weights[i] = 1.0 + abs(float(corr)) * 4.0
        self.feature_weights = 0.85 * self.feature_weights + 0.15 * new_weights

    def _decay_memory(self, step_id: int) -> None:
        survivors: List[MemoryCase] = []
        for case in self.memory:
            gap = max(0, step_id - case.last_step)
            factor = self._recency_weight(gap)
            case.apply_decay(factor)
            if case.total_count >= 0.05:
                survivors.append(case)
        self.memory = survivors

    def _compress_prototypes(self, step_id: int) -> None:
        if len(self.memory) < 2:
            return
        self.memory.sort(key=lambda c: (c.prototype, c.reliability, c.total_count), reverse=True)
        compressed: List[MemoryCase] = []
        for case in self.memory:
            merged = False
            for existing in compressed:
                sim = self._similarity(case.vector, existing.vector)
                if sim >= self.prototype_threshold and case.regime == existing.regime:
                    existing.vector = 0.6 * existing.vector + 0.4 * case.vector
                    for action in ACTIONS:
                        stats = case.action_stats[action]
                        if stats.count > 0:
                            existing.action_stats[action].update(stats.mean, weight=stats.count)
                    existing.total_count += case.total_count
                    existing.last_step = max(existing.last_step, case.last_step, step_id)
                    existing.prototype = True
                    existing.reliability = float(max(existing.reliability, case.reliability))
                    merged = True
                    break
            if not merged:
                compressed.append(case)
        self.memory = compressed[-self.max_cases :]

    def _maintenance(self, step_id: int) -> None:
        self._maintenance_counter += 1
        if self._maintenance_counter % 10 == 0:
            self._decay_memory(step_id)
        if self._maintenance_counter % 15 == 0 or len(self.memory) > self.max_cases * 0.9:
            self._compress_prototypes(step_id)
        if len(self.memory) > self.max_cases:
            self.memory.sort(key=lambda c: (c.reliability, c.total_count, c.last_step))
            self.memory = self.memory[-self.max_cases :]
        if self.autosave and self._maintenance_counter % 5 == 0:
            self.save_memory()

    def register_pending(self, symbol: str, action: str, state: np.ndarray, step_id: int) -> None:
        self.pending[symbol] = (state.copy(), action, step_id)

    def realize_pending(self, symbol: str, next_return: float, regime: str, step_id: int) -> None:
        pending = self.pending.pop(symbol, None)
        if pending is None:
            return
        state, action, _pending_step = pending
        reward_map = {
            "long": float(next_return),
            "short": float(-next_return),
            "flat": float(-abs(next_return) * 0.05),
            "reduce": float(max(0.0, next_return) * 0.3 - abs(next_return) * 0.05),
        }
        reward = reward_map.get(action, 0.0)
        self._insert_case(state, action, reward, regime, step_id)
        self._propagate_reward(state, action, reward, step_id)
        self._refresh_feature_weights()
        self._maintenance(step_id)
