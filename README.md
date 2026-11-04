# slowdrain: catch a slow-drain device before it congests the fabric

Fixed-threshold congestion monitoring catches gross credit stalls, but it has no idea what is
normal for *this* port. An HBA that slowly degrades to around 60% of line rate stays under fixed
limits until the congestion is already spreading. `slowdrain` learns a baseline for each port from
the port's own history and flags drift away from it, hours earlier.

```
telemetry (SSH / SNMP / CSV)
   -> features: counter rates, rolling statistics, per-port z-scores
   -> Isolation Forest (per port)  ---+
   -> autoencoder (sliding window) ---+-> per-port thresholds -> sustained-window rule -> alert -> remediation hook
```

## Quick start

```bash
pip install .                    # pure-Python models, no dependencies
pip install ".[ml]"              # optional: scikit-learn IsolationForest + PyTorch autoencoder

slowdrain simulate --out fabric.csv             # 12 ports, port 3's HBA degrades from poll 1000
slowdrain detect fabric.csv --report report.json
python -m unittest discover -s tests
```

Example output (pure-Python backends, default settings):

```
learned-baseline alerts: 1   fixed-threshold alerts: 1

sw1/  3  .................=:.........::..-+...:..:#-*+#@@@@@@@@@@@@@@@@@@@@@@@@@@@
         learned alert @ 34530.0s | fixed alert @ 46080.0s | lead time 3.2 h (11550s)
```

Across simulator seeds 1, 2, 4, 5 and 7, the degrading port was flagged **2.7 to 4.2 hours** before
the fixed multi-window threshold fired, with no alerts on healthy ports. With seed 3, the port that
later degrades also got one short false alert before degradation began. Expect some alerts of this
kind on real data, and tune `--percentile`, `--sustain`, and `--train-rows` for your fabric.

## How it works

| Stage | Module | Details |
|---|---|---|
| Collection | `collectors.py` | A generic SSH collector (you supply the command and a parser; `key=value` is built in), SNMP using standard IF-MIB counters (needs `pysnmp`), and CSV |
| Features | `features.py` | First differences of cumulative counters (handles counter clears), trailing 30-poll rolling statistics, and robust z-scores against each port's own training history |
| Isolation Forest | `iforest.py` | scikit-learn if installed, otherwise a pure-Python version of Liu, Ting and Zhou (2008). One forest per port; ports with little history use a pooled port-class forest |
| Autoencoder | `autoencoder.py` | PyTorch if installed, otherwise a pure-Python MLP trained with Adam. Scores sliding windows by mean squared reconstruction error |
| Ensemble | `detector.py` | Per-port percentile thresholds. A port is flagged when either model stays above threshold for `--sustain` polls, and the alert clears after `clear` polls back below |
| Comparison | `detector.py` | `FixedThresholdDetector`: a credit-stall fraction over short, medium and long windows, the same for every port |
| Remediation | `cli.py` | `--on-alert "tool {switch} {port}"` only prints the command unless you add `--execute` |

### Input format

CSV with one row per port per poll. Counters are cumulative, using vendor-neutral names:

```
timestamp,switch,port,speed_gbps,tx_credit_zero_us,crc_err,itw,c3_discard,rx_c3_timeout,txq_latency_us,tx_bytes,rx_bytes
```

`tx_credit_zero_us` is the cumulative time the port could not transmit because its transmit
buffer-to-buffer credit was zero. Map your switch's equivalent counter onto it.

### Practical notes

- **Training history must be healthy and cover at least one full workload cycle**, for example a
  day or a week. Otherwise normal peaks will look like anomalies.
- Rolling statistics use the median and MAD by default (`robust=True`), so a single-poll burst
  doesn't inflate the dispersion features for 30 polls. Pass `robust=False` to get the mean and
  standard deviation.
- The pure-Python backends make everything run anywhere, but they are slower. A 12-port,
  2,000-poll run takes about 15 seconds. Install `.[ml]` for large fabrics.

## Author

**Derek Asir Muthurajan Caleb**: author, owner, and publisher.

## License

Apache License 2.0. See [LICENSE](LICENSE).
