"""Self-learning knowledge engine for US-market patterns.

Learns online (one observation at a time) whether today's feature set —
momentum, realized volatility, news sentiment, insider net buying — predicts
tomorrow's direction. Weights are persisted to SQLite so knowledge survives
restarts and grows with every feed cycle.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import SGDClassifier

FEATURE_NAMES = ["ret_5d", "ret_20d", "vol_20d", "sentiment", "insider_net_norm"]


@dataclass
class KnowledgeSnapshot:
    """Serializable summary of what the model currently believes."""
    updated_at: str
    n_observations: int
    accuracy_ema: float
    coefficients: dict[str, float]
    intercept: float
    classes: list[int] = field(default_factory=lambda: [0, 1])


class KnowledgeEngine:
    """Incremental logistic-regression learner over market feature vectors.

    Uses ``SGDClassifier(loss='log_loss')`` + ``partial_fit`` so each new
    daily observation updates the model in O(features) time — the "self
    learning" loop feeding on prices, sentiment, and Form 4 flows.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-3,
                                   random_state=7, shuffle=False)
        self.n_obs = 0
        self.acc_ema = 0.5
        self._warmed = False
        self._init_db()
        self.load()

    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS model_knowledge (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS observations (
                ts TEXT, ticker TEXT, features TEXT, label INTEGER,
                prob_up REAL, correct INTEGER)""")
            con.commit()

    # ------------------------------------------------------------------ #
    def observe(self, ticker: str, features: Iterable[float], next_day_up: bool) -> float:
        """Feed one labeled observation; returns P(up) predicted *before* update."""
        x = np.asarray([list(features)], dtype=float)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {x.shape[1]}")
        y = np.array([1 if next_day_up else 0])

        prob = 0.5
        if self._warmed:
            dist = self.model.predict_proba(x)[0]
            prob = float(dist[1])
            pred = int(dist.argmax())
            hit = int(pred == y[0])
            self.acc_ema = 0.98 * self.acc_ema + 0.02 * hit
        else:  # seed both classes first so partial_fit knows the label space
            self.model.partial_fit(np.vstack([x, -x]), np.array([y[0], 1 - y[0]]), classes=[0, 1])
            self._warmed = True

        self.model.partial_fit(x, y)
        self.n_obs += 1
        with sqlite3.connect(self.db_path) as con:
            con.execute("INSERT INTO observations VALUES (?,?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), ticker,
                         json.dumps(x[0].tolist()), int(y[0]), prob,
                         None if not self._warmed else int(prob >= 0.5 == bool(y[0]))))
            con.commit()
        return prob

    def predict(self, features: Iterable[float]) -> float:
        """P(next day up) for a feature vector (0.5 before any learning)."""
        if not self._warmed:
            return 0.5
        x = np.asarray([list(features)], dtype=float)
        return float(self.model.predict_proba(x)[0][1])

    # ------------------------------------------------------------------ #
    def snapshot(self) -> KnowledgeSnapshot:
        coef = self.model.coef_[0].tolist() if self._warmed else [0.0] * len(FEATURE_NAMES)
        inter = float(self.model.intercept_[0]) if self._warmed else 0.0
        return KnowledgeSnapshot(
            updated_at=datetime.now(timezone.utc).isoformat(),
            n_observations=self.n_obs,
            accuracy_ema=round(self.acc_ema, 4),
            coefficients=dict(zip(FEATURE_NAMES, coef)),
            intercept=inter,
        )

    def save(self) -> None:
        snap = self.snapshot()
        with sqlite3.connect(self.db_path) as con:
            con.execute("INSERT OR REPLACE INTO model_knowledge VALUES ('snapshot', ?)",
                        (json.dumps(snap.__dict__),))
            con.commit()

    def load(self) -> bool:
        """Restore weights from SQLite. Returns True if knowledge was found."""
        with sqlite3.connect(self.db_path) as con:
            row = con.execute("SELECT value FROM model_knowledge WHERE key='snapshot'").fetchone()
        if not row:
            return False
        snap = json.loads(row[0])
        coef = np.asarray([[snap["coefficients"][f] for f in FEATURE_NAMES]], dtype=float)
        self.model.coef_ = coef
        self.model.intercept_ = np.asarray([snap["intercept"]], dtype=float)
        self.model.classes_ = np.asarray([0, 1])
        self.n_obs = int(snap.get("n_observations", 0))
        self.acc_ema = float(snap.get("accuracy_ema", 0.5))
        self._warmed = self.n_obs > 0
        return True
