# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""Telemetry records, CSV I/O and a synthetic fabric generator.

A record is one poll of one port.  Counters are cumulative (as switches
report them); feature engineering turns them into rates.

Columns (vendor-neutral names):

==================  =========================================================
timestamp           seconds (float)
switch, port        identifiers
speed_gbps          negotiated link speed
tx_credit_zero_us   cumulative microseconds the port could not transmit
                    because its transmit buffer-to-buffer credit was zero
crc_err             cumulative CRC errors
itw                 cumulative invalid transmission words
c3_discard          cumulative class-3 frame discards
rx_c3_timeout       cumulative receive-side class-3 timeouts
txq_latency_us      instantaneous transmit-queue latency (gauge)
tx_bytes, rx_bytes  cumulative bytes
==================  =========================================================
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass, fields
from typing import Dict, Iterable, List, Optional, Tuple

COUNTERS = ("tx_credit_zero_us", "crc_err", "itw", "c3_discard", "rx_c3_timeout", "tx_bytes", "rx_bytes")


@dataclass
class Record:
    timestamp: float
    switch: str
    port: str
    speed_gbps: float
    tx_credit_zero_us: float
    crc_err: float
    itw: float
    c3_discard: float
    rx_c3_timeout: float
    txq_latency_us: float
    tx_bytes: float
    rx_bytes: float

    @property
    def key(self) -> Tuple[str, str]:
        return (self.switch, self.port)


FIELDNAMES = [f.name for f in fields(Record)]


def write_csv(path: str, records: Iterable[Record]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def read_csv(path: str) -> List[Record]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(Record(
                float(row["timestamp"]), row["switch"], row["port"],
                *(float(row[n]) for n in FIELDNAMES[3:])))
    return out


def group_by_port(records: Iterable[Record]) -> Dict[Tuple[str, str], List[Record]]:
    g: Dict[Tuple[str, str], List[Record]] = {}
    for r in records:
        g.setdefault(r.key, []).append(r)
    for v in g.values():
        v.sort(key=lambda r: r.timestamp)
    return g


# ---------------------------------------------------------------------------
# Synthetic fabric
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    ports: int = 12
    samples: int = 2000
    interval_s: float = 30.0
    cycle_s: float = 3 * 3600.0           # workload cycle; training history should span >= 1 cycle
    speed_gbps: float = 32.0
    degrade_port: Optional[int] = 3       # index of the port whose HBA degrades
    degrade_start: int = 1000             # sample where degradation begins
    degrade_ramp: int = 700               # samples to reach full degradation
    final_line_rate: float = 0.60         # degraded HBA acknowledges ~60% of line rate
    burst_prob: float = 0.004             # transient credit-stall bursts on healthy ports
    seed: int = 7


def simulate(cfg: SimConfig = SimConfig()) -> List[Record]:
    """Generate a fabric where one HBA degrades gradually instead of failing.

    Healthy ports follow a diurnal load curve with noise and rare one-sample
    bursts.  The degrading port's credit-stall time, latency and CRC errors
    creep upward while its achieved throughput sags toward ``final_line_rate``.
    """
    rng = random.Random(cfg.seed)
    line_bps = cfg.speed_gbps * 1e8          # ~bytes/s at full FC line rate
    out: List[Record] = []
    for p in range(cfg.ports):
        acc = {c: 0.0 for c in COUNTERS}
        base_load = rng.uniform(0.25, 0.55)
        phase = rng.uniform(0, 2 * math.pi)
        base_stall = rng.uniform(0.002, 0.01)
        base_lat = rng.uniform(40, 90)
        for t in range(cfg.samples):
            ts = t * cfg.interval_s
            diurnal = 1 + 0.35 * math.sin(2 * math.pi * ts / cfg.cycle_s + phase)
            load = max(0.02, base_load * diurnal + rng.gauss(0, 0.03))
            stall = max(0.0, base_stall * (1 + load) + rng.gauss(0, 0.002))
            lat = max(5.0, base_lat * (1 + load) + rng.gauss(0, 6))
            crc = 1 if rng.random() < 0.002 else 0
            itw = rng.choice((0, 0, 0, 1))
            c3 = 0
            rxto = 0
            if rng.random() < cfg.burst_prob:          # harmless transient burst
                stall += rng.uniform(0.08, 0.2)
                lat *= 2.5
            if p == cfg.degrade_port and t >= cfg.degrade_start:
                d = min(1.0, (t - cfg.degrade_start) / cfg.degrade_ramp)   # 0 -> 1
                ack = 1 - (1 - cfg.final_line_rate) * d
                load *= ack
                stall += 0.30 * d ** 1.6 + rng.gauss(0, 0.004 * (1 + d))
                lat *= 1 + 2.5 * d ** 1.3
                crc += 1 if rng.random() < 0.02 + 0.2 * d else 0
                itw += rng.randint(0, int(1 + 12 * d))
                if stall > 0.18:
                    c3 += rng.randint(0, int(40 * (stall - 0.18) * 10))
                    rxto += rng.randint(0, 2)
            stall = min(stall, 0.99)
            acc["tx_credit_zero_us"] += stall * cfg.interval_s * 1e6
            acc["crc_err"] += crc
            acc["itw"] += itw
            acc["c3_discard"] += c3
            acc["rx_c3_timeout"] += rxto
            acc["tx_bytes"] += load * line_bps * cfg.interval_s
            acc["rx_bytes"] += load * 0.6 * line_bps * cfg.interval_s
            out.append(Record(ts, "sw1", f"{p}", cfg.speed_gbps, acc["tx_credit_zero_us"], acc["crc_err"],
                              acc["itw"], acc["c3_discard"], acc["rx_c3_timeout"], lat,
                              acc["tx_bytes"], acc["rx_bytes"]))
    return out
