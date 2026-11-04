# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""Per-port learned-baseline detector: Isolation Forest + autoencoder ensemble.

* One Isolation Forest per port, fit on that port's z-scored training rows.
  Ports with too little history use a pooled "port-class" forest instead.
* One autoencoder on sliding windows of z-scored rows.  Because every port is
  standardised against itself, healthy windows from all ports look alike and
  can train a single shared model.
* Thresholds are per port: a percentile of the port's own training scores.
* A port is flagged only when either normalised score stays above 1.0 for
  ``sustain`` consecutive polls, which keeps transient bursts from alerting,
  and clears only after ``clear`` consecutive polls back below threshold.

``FixedThresholdDetector`` is the conventional comparison: credit-stall
fraction averaged over a short, medium and long window must all exceed fixed
limits - no per-port baseline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .autoencoder import make_autoencoder
from .features import FeatureRow, PortScaler, windows
from .iforest import make_isolation_forest

PortKey = Tuple[str, str]


def percentile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("inf")
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class PortScore:
    timestamp: float
    iforest: float          # normalised: score / port threshold
    autoencoder: float      # normalised: error / port threshold
    raw_stall: float
    flagged: bool = False

    @property
    def ensemble(self) -> float:
        return max(self.iforest, self.autoencoder)


@dataclass
class Alert:
    switch: str
    port: str
    start: float
    end: Optional[float]
    peak: float
    detector: str
    reason: str = ""

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class DetectorConfig:
    train_rows: int = 600
    ae_window: int = 4
    percentile: float = 99.5
    margin: float = 1.15
    sustain: int = 5
    clear: int = 10          # polls below threshold before an alert clears (hysteresis)
    min_history: int = 200
    iforest_backend: str = "auto"
    ae_backend: str = "auto"
    n_estimators: int = 100
    ae_epochs: int = 20
    seed: int = 0


@dataclass
class _PortModel:
    scaler: PortScaler
    forest: object
    if_threshold: float = 1.0
    ae_threshold: float = 1.0
    pooled: bool = False


class SlowDrainDetector:
    def __init__(self, cfg: DetectorConfig = DetectorConfig()) -> None:
        self.cfg = cfg
        self.ports: Dict[PortKey, _PortModel] = {}
        self.ae = None

    # ------------------------------------------------------------------
    def fit(self, features: Dict[PortKey, List[FeatureRow]]) -> "SlowDrainDetector":
        c = self.cfg
        train = {k: [r.values for r in rows[:c.train_rows]] for k, rows in features.items()}
        rich = {k: v for k, v in train.items() if len(v) >= c.min_history}
        if not rich:
            raise ValueError(f"no port has the {c.min_history} training rows needed for a baseline")
        pooled_rows: List[List[float]] = []
        scaled_train: Dict[PortKey, List[List[float]]] = {}
        for k, rows in train.items():
            if k in rich:
                scaler = PortScaler.fit(rows)
            else:  # too little history: scale with pooled statistics
                scaler = PortScaler.fit([r for v in rich.values() for r in v])
            scaled_train[k] = [scaler.transform(r) for r in rows]
            self.ports[k] = _PortModel(scaler, None, pooled=k not in rich)
            if k in rich:
                pooled_rows += scaled_train[k]

        pooled_forest = make_isolation_forest(c.iforest_backend, n_estimators=c.n_estimators,
                                              seed=c.seed).fit(pooled_rows)
        for k, pm in self.ports.items():
            pm.forest = (make_isolation_forest(c.iforest_backend, n_estimators=c.n_estimators,
                                               seed=c.seed).fit(scaled_train[k])
                         if not pm.pooled else pooled_forest)

        ae_train = [w for k in rich for w in windows(scaled_train[k], c.ae_window)]
        self.ae = make_autoencoder(c.ae_backend, epochs=c.ae_epochs, seed=c.seed).fit(ae_train)

        for k, pm in self.ports.items():
            rows = scaled_train[k] or pooled_rows
            pm.if_threshold = percentile(pm.forest.score(rows), c.percentile) * c.margin
            ws = windows(rows, c.ae_window) or ae_train
            pm.ae_threshold = max(percentile(self.ae.score(ws), c.percentile) * c.margin, 1e-9)
        return self

    # ------------------------------------------------------------------
    def score_port(self, key: PortKey, rows: List[FeatureRow]) -> List[PortScore]:
        pm = self.ports.get(key)
        if pm is None:
            raise KeyError(f"port {key} was not seen during training")
        c = self.cfg
        z = [pm.scaler.transform(r.values) for r in rows]
        if_s = pm.forest.score(z)
        ae_raw = self.ae.score(windows(z, c.ae_window))
        ae_s = [0.0] * (c.ae_window - 1) + ae_raw
        out = [PortScore(r.timestamp, i / pm.if_threshold, a / pm.ae_threshold, r.raw_stall)
               for r, i, a in zip(rows, if_s, ae_s)]
        above = below = 0
        active = False
        for s in out:
            if s.ensemble > 1.0:
                above, below = above + 1, 0
            else:
                above, below = 0, below + 1
            if not active and above >= c.sustain:
                active = True
            elif active and below >= c.clear:
                active = False
            s.flagged = active
        return out

    def detect(self, features: Dict[PortKey, List[FeatureRow]]) -> Tuple[Dict[PortKey, List[PortScore]], List[Alert]]:
        scores: Dict[PortKey, List[PortScore]] = {}
        alerts: List[Alert] = []
        for key, rows in features.items():
            s = self.score_port(key, rows)
            scores[key] = s
            alerts += alerts_from_flags(key, [(x.timestamp, x.flagged, x.ensemble) for x in s],
                                        "learned-baseline", self._reason)
        return scores, alerts

    @staticmethod
    def _reason(peak: float) -> str:
        return f"ensemble score {peak:.1f}x the port's own learned threshold"


def alerts_from_flags(key: PortKey, series: Sequence[Tuple[float, bool, float]], detector: str,
                      reason=lambda p: "") -> List[Alert]:
    alerts: List[Alert] = []
    cur: Optional[Alert] = None
    for ts, flagged, val in series:
        if flagged and cur is None:
            cur = Alert(key[0], key[1], ts, None, val, detector)
        elif flagged and cur is not None:
            cur.peak = max(cur.peak, val)
        elif not flagged and cur is not None:
            cur.end = ts
            cur.reason = reason(cur.peak)
            alerts.append(cur)
            cur = None
    if cur is not None:
        cur.reason = reason(cur.peak)
        alerts.append(cur)
    return alerts


@dataclass
class FixedThresholdDetector:
    """Fixed multi-window credit-stall threshold, identical for every port."""
    windows: Tuple[int, int, int] = (3, 10, 30)
    limits: Tuple[float, float, float] = (0.20, 0.20, 0.20)

    def detect(self, features: Dict[PortKey, List[FeatureRow]]) -> List[Alert]:
        alerts: List[Alert] = []
        for key, rows in features.items():
            qs = [deque(maxlen=w) for w in self.windows]
            series = []
            for r in rows:
                for q in qs:
                    q.append(r.raw_stall)
                means = [sum(q) / len(q) for q in qs]
                full = all(len(q) == q.maxlen for q in qs)
                series.append((r.timestamp, full and all(m > lim for m, lim in zip(means, self.limits)), means[0]))
            alerts += alerts_from_flags(key, series, "fixed-threshold",
                                        lambda p: f"credit-stall fraction {p:.0%} above fixed limit")
        return alerts
