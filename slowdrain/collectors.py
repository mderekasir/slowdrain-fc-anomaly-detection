# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""Telemetry collectors (paper Sec. III).

Switch CLIs differ by vendor and release, so the SSH collector is generic: it
runs a command you choose and hands the text to a parser.  The bundled
``KeyValueParser`` reads one port per line in ``key=value`` form; write your
own parser for fixed-width CLI tables.

``SNMPCollector`` polls standard IF-MIB counters (bytes and errors) with the
optional ``pysnmp`` package.  Credit-stall time is not in IF-MIB; take it from
the switch's own MIB or CLI and merge the records.
"""
from __future__ import annotations

import re
import subprocess
import time
from typing import Callable, Dict, Iterable, List, Optional

from .telemetry import FIELDNAMES, Record

_KV = re.compile(r"(\w+)=([^\s]+)")


class KeyValueParser:
    """``port=3 speed_gbps=32 tx_credit_zero_us=1200 crc_err=0 ... tx_bytes=9.1e9``"""

    def __init__(self, switch: str) -> None:
        self.switch = switch

    def __call__(self, text: str, timestamp: float) -> List[Record]:
        out = []
        for line in text.splitlines():
            kv = dict(_KV.findall(line))
            if "port" not in kv:
                continue
            vals = {n: float(kv.get(n, 0.0)) for n in FIELDNAMES[3:]}
            vals["speed_gbps"] = float(kv.get("speed_gbps", 32))
            out.append(Record(timestamp, self.switch, kv["port"], **vals))
        return out


class SSHCollector:
    """Polls a switch over the OpenSSH client (key-based authentication)."""

    def __init__(self, host: str, command: str, parser: Callable[[str, float], List[Record]],
                 user: str = "admin", port: int = 22, timeout: float = 60) -> None:
        self.host, self.command, self.parser = host, command, parser
        self.user, self.port, self.timeout = user, port, timeout

    def poll(self) -> List[Record]:
        argv = ["ssh", "-o", "BatchMode=yes", "-p", str(self.port), f"{self.user}@{self.host}", self.command]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.host}: {proc.stderr.strip()}")
        return self.parser(proc.stdout, time.time())


class SNMPCollector:  # pragma: no cover - requires pysnmp and a live agent
    IF_MIB = {"tx_bytes": "1.3.6.1.2.1.31.1.1.1.10",   # ifHCOutOctets
              "rx_bytes": "1.3.6.1.2.1.31.1.1.1.6",    # ifHCInOctets
              "crc_err": "1.3.6.1.2.1.2.2.1.14",       # ifInErrors
              "c3_discard": "1.3.6.1.2.1.2.2.1.19"}    # ifOutDiscards

    def __init__(self, host: str, switch: str, if_indexes: Iterable[int], community: str = "public",
                 speed_gbps: float = 32) -> None:
        self.host, self.switch, self.ifx = host, switch, list(if_indexes)
        self.community, self.speed = community, speed_gbps

    def poll(self) -> List[Record]:
        from pysnmp.hlapi import (CommunityData, ContextData, ObjectIdentity, ObjectType, SnmpEngine,
                                  UdpTransportTarget, getCmd)
        ts = time.time()
        out = []
        for i in self.ifx:
            vals: Dict[str, float] = {n: 0.0 for n in FIELDNAMES[3:]}
            vals["speed_gbps"] = self.speed
            oids = [ObjectType(ObjectIdentity(f"{oid}.{i}")) for oid in self.IF_MIB.values()]
            err, status, _idx, binds = next(getCmd(SnmpEngine(), CommunityData(self.community),
                                                   UdpTransportTarget((self.host, 161)), ContextData(), *oids))
            if err or status:
                raise RuntimeError(f"SNMP error from {self.host}: {err or status.prettyPrint()}")
            for name, (_oid, val) in zip(self.IF_MIB, binds):
                vals[name] = float(val)
            out.append(Record(ts, self.switch, str(i), **vals))
        return out


def poll_loop(collector, interval_s: float, count: Optional[int], sink: Callable[[List[Record]], None]) -> None:
    n = 0
    while count is None or n < count:
        start = time.monotonic()
        sink(collector.poll())
        n += 1
        time.sleep(max(0.0, interval_s - (time.monotonic() - start)))
