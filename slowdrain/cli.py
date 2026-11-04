# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""Command line::

    slowdrain simulate --out fabric.csv
    slowdrain detect fabric.csv --train-rows 600 --report report.json
    slowdrain detect fabric.csv --on-alert "my-quarantine-tool {switch} {port}"   # printed only
    slowdrain detect fabric.csv --on-alert "..." --execute                       # actually run it
    slowdrain collect --host 10.0.0.5 --command "show port counters kv" --switch sw1 --out live.csv
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from typing import List, Optional

from . import __version__
from .collectors import KeyValueParser, SSHCollector, poll_loop
from .detector import DetectorConfig, FixedThresholdDetector, SlowDrainDetector
from .features import feature_table
from .telemetry import SimConfig, group_by_port, read_csv, simulate, write_csv

_BARS = " .:-=+*#%@"


def sparkline(values: List[float], width: int = 72, cap: float = 4.0) -> str:
    if not values:
        return ""
    step = max(1, len(values) // width)
    buckets = [max(values[i:i + step]) for i in range(0, len(values), step)]
    return "".join(_BARS[min(len(_BARS) - 1, int(min(v, cap) / cap * (len(_BARS) - 1)))] for v in buckets)


def cmd_simulate(a) -> int:
    cfg = SimConfig(ports=a.ports, samples=a.samples, degrade_port=None if a.no_fault else a.degrade_port,
                    degrade_start=a.degrade_start, seed=a.seed)
    recs = simulate(cfg)
    write_csv(a.out, recs)
    print(f"wrote {len(recs)} records for {cfg.ports} ports to {a.out}"
          + ("" if a.no_fault else f" (port {cfg.degrade_port} degrades from poll {cfg.degrade_start})"))
    return 0


def cmd_detect(a) -> int:
    t0 = time.time()
    feats = feature_table(group_by_port(read_csv(a.csv)), window=a.window)
    cfg = DetectorConfig(train_rows=a.train_rows, sustain=a.sustain, percentile=a.percentile,
                         iforest_backend=a.iforest_backend, ae_backend=a.ae_backend, ae_epochs=a.ae_epochs)
    det = SlowDrainDetector(cfg).fit(feats)
    backend = f"{type(next(iter(det.ports.values())).forest).__name__} + {type(det.ae).__name__}"
    scores, alerts = det.detect(feats)
    fixed = FixedThresholdDetector().detect(feats)
    # alerts inside the training period are meaningless (the model saw that data)
    train_end = {k: rows[min(cfg.train_rows, len(rows)) - 1].timestamp for k, rows in feats.items() if rows}
    alerts = [x for x in alerts if x.start > train_end[(x.switch, x.port)]]

    print(f"slowdrain {__version__}: {len(feats)} ports, models: {backend}, trained in {time.time() - t0:.1f}s")
    print(f"learned-baseline alerts: {len(alerts)}   fixed-threshold alerts: {len(fixed)}\n")
    report = {"ports": {}, "alerts": [x.to_dict() for x in alerts], "fixed_threshold_alerts": [x.to_dict() for x in fixed]}
    for key in sorted(scores, key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else k[1])):
        mine = [x for x in alerts if (x.switch, x.port) == key]
        theirs = [x for x in fixed if (x.switch, x.port) == key]
        first = mine[0].start if mine else None
        first_fixed = theirs[0].start if theirs else None
        lead = (first_fixed - first) if (first is not None and first_fixed is not None) else None
        report["ports"][f"{key[0]}/{key[1]}"] = {"first_alert": first, "first_fixed_alert": first_fixed,
                                                  "lead_time_s": lead}
        if mine or theirs or a.verbose:
            print(f"{key[0]}/{key[1]:>3}  {sparkline([s.ensemble for s in scores[key]])}")
            line = f"         learned alert @ {first}s" if first is not None else "         no learned alert"
            line += f" | fixed alert @ {first_fixed}s" if first_fixed is not None else " | no fixed alert"
            if lead is not None:
                line += f" | lead time {lead / 3600:.1f} h ({lead:.0f}s)"
            print(line)
    for x in alerts:
        if a.on_alert:
            cmd = a.on_alert.format(switch=x.switch, port=x.port, score=f"{x.peak:.2f}", start=x.start)
            if a.execute:
                print(f"executing: {cmd}")
                subprocess.run(shlex.split(cmd), check=False)
            else:
                print(f"would run (add --execute): {cmd}")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nreport written to {a.report}")
    return 3 if alerts else 0


def cmd_collect(a) -> int:
    coll = SSHCollector(a.host, a.command, KeyValueParser(a.switch), user=a.user)
    buf = []
    poll_loop(coll, a.interval, a.count, buf.extend)
    write_csv(a.out, buf)
    print(f"wrote {len(buf)} records to {a.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="slowdrain", description="Predictive slow-drain detection for FC fabrics")
    ap.add_argument("--version", action="version", version=f"slowdrain {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate synthetic fabric telemetry")
    s.add_argument("--out", default="fabric.csv")
    s.add_argument("--ports", type=int, default=12)
    s.add_argument("--samples", type=int, default=2000)
    s.add_argument("--degrade-port", type=int, default=3)
    s.add_argument("--degrade-start", type=int, default=1000)
    s.add_argument("--no-fault", action="store_true", help="all ports healthy")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_simulate)

    d = sub.add_parser("detect", help="train per-port baselines and flag degrading ports")
    d.add_argument("csv")
    d.add_argument("--train-rows", type=int, default=600, help="leading feature rows per port used as healthy history")
    d.add_argument("--window", type=int, default=30, help="rolling window (polls)")
    d.add_argument("--sustain", type=int, default=5, help="consecutive polls above threshold before alerting")
    d.add_argument("--percentile", type=float, default=99.5)
    d.add_argument("--iforest-backend", choices=["auto", "sklearn", "builtin"], default="auto")
    d.add_argument("--ae-backend", choices=["auto", "torch", "builtin"], default="auto")
    d.add_argument("--ae-epochs", type=int, default=20)
    d.add_argument("--report", help="write JSON report")
    d.add_argument("--on-alert", help="command template run per alert: {switch} {port} {score} {start}")
    d.add_argument("--execute", action="store_true", help="actually run --on-alert commands")
    d.add_argument("-v", "--verbose", action="store_true", help="show every port")
    d.set_defaults(func=cmd_detect)

    c = sub.add_parser("collect", help="poll a switch over SSH into CSV (key=value output)")
    c.add_argument("--host", required=True)
    c.add_argument("--user", default="admin")
    c.add_argument("--command", required=True, help="CLI command that prints one port per line as key=value")
    c.add_argument("--switch", required=True)
    c.add_argument("--interval", type=float, default=30)
    c.add_argument("--count", type=int, default=10)
    c.add_argument("--out", default="live.csv")
    c.set_defaults(func=cmd_collect)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
