# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""Feature engineering (paper Sec. IV-A, Table I).

Three transformations:

1. first difference of cumulative counters -> rates
2. trailing rolling mean / std (default 30 polls)
3. per-port robust z-score against the port's *own* training history, so a
   value is judged "abnormal for this port", not against a global constant
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .telemetry import Record

FEATURES = ("credit_stall_rate", "crc_err_rate", "itw_rate", "c3_discard_rate",
            "txq_latency_mean", "txq_latency_std", "throughput_mean", "throughput_std")


def _delta(cur: float, prev: float) -> float:
    d = cur - prev
    return cur if d < 0 else d      # negative => counter cleared or wrapped


@dataclass
class FeatureRow:
    timestamp: float
    raw_stall: float            # fraction of the interval with zero Tx credit (0..1)
    values: List[float]


def port_features(records: Sequence[Record], window: int = 30, robust: bool = True) -> List[FeatureRow]:
    """Turn one port's ordered records into feature rows (first ``window`` polls are warm-up).

    With ``robust=True`` (default) the rolling statistics are median and
    1.4826*MAD instead of mean and std, so a single-poll burst does not inflate
    the dispersion features for the next ``window`` polls.
    """
    loc, disp = (_median, _mad) if robust else (_mean, _std)
    rows: List[FeatureRow] = []
    lat_q: deque = deque(maxlen=window)
    thr_q: deque = deque(maxlen=window)
    for prev, cur in zip(records, records[1:]):
        dt = cur.timestamp - prev.timestamp
        if dt <= 0:
            continue
        stall = min(1.0, _delta(cur.tx_credit_zero_us, prev.tx_credit_zero_us) / (dt * 1e6))
        line = max(cur.speed_gbps, 1e-9) * 1e8
        thr = _delta(cur.tx_bytes, prev.tx_bytes) / dt / line
        lat_q.append(cur.txq_latency_us)
        thr_q.append(thr)
        if len(lat_q) < window:
            continue
        vals = [
            stall,
            _delta(cur.crc_err, prev.crc_err) / dt,
            _delta(cur.itw, prev.itw) / dt,
            _delta(cur.c3_discard, prev.c3_discard) / dt,
            loc(list(lat_q)), disp(list(lat_q)), loc(list(thr_q)), disp(list(thr_q)),
        ]
        rows.append(FeatureRow(cur.timestamp, stall, vals))
    return rows


def _mean(xs) -> float:
    return sum(xs) / len(xs)


def _std(xs) -> float:
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _mad(xs: List[float]) -> float:
    m = _median(xs)
    return 1.4826 * _median([abs(x - m) for x in xs])


@dataclass
class PortScaler:
    """Robust per-port scaler: (x - median) / (1.4826 * MAD), with a floor."""
    center: List[float]
    scale: List[float]

    @classmethod
    def fit(cls, rows: Sequence[List[float]], floor: float = 1e-6) -> "PortScaler":
        cols = list(zip(*rows))
        center, scale = [], []
        for c in cols:
            med = _median(list(c))
            mad = _median([abs(x - med) for x in c]) * 1.4826
            sd = _std(c)
            # MAD is 0 for sparse counters (mostly zeros); fall back to std, then floor
            scale.append(max(mad if mad > 0 else sd, floor, 1e-3 * abs(med)))
            center.append(med)
        return cls(center, scale)

    def transform(self, row: Sequence[float], clip: float = 50.0) -> List[float]:
        return [max(-clip, min(clip, (x - c) / s)) for x, c, s in zip(row, self.center, self.scale)]


def windows(rows: Sequence[List[float]], length: int) -> List[List[float]]:
    """Flattened sliding windows, aligned so window i ends at row i (first length-1 rows dropped)."""
    return [[v for r in rows[i - length + 1:i + 1] for v in r] for i in range(length - 1, len(rows))]


def split_train(rows: List[FeatureRow], train_fraction: float = 0.0, train_rows: int = 0) -> Tuple[List[FeatureRow], List[FeatureRow]]:
    n = train_rows or int(len(rows) * train_fraction)
    return rows[:n], rows[n:]


def to_matrix(rows: Sequence[FeatureRow]) -> List[List[float]]:
    return [r.values for r in rows]


def feature_table(by_port: Dict, window: int = 30, robust: bool = True) -> Dict:
    return {k: port_features(v, window, robust) for k, v in by_port.items()}
