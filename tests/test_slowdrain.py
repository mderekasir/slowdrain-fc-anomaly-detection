# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
import os
import random
import tempfile
import unittest

from slowdrain import (DetectorConfig, FixedThresholdDetector, SimConfig, SlowDrainDetector, feature_table,
                       group_by_port, read_csv, simulate, write_csv)
from slowdrain.autoencoder import PurePythonAutoencoder
from slowdrain.collectors import KeyValueParser
from slowdrain.detector import percentile
from slowdrain.features import PortScaler, port_features, windows
from slowdrain.iforest import PurePythonIsolationForest

FAST = dict(iforest_backend="builtin", ae_backend="builtin", n_estimators=40, ae_epochs=6)


class IsolationForestTests(unittest.TestCase):
    def test_outlier_scores_higher(self):
        rng = random.Random(0)
        X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(300)]
        f = PurePythonIsolationForest(n_estimators=50, seed=1).fit(X)
        normal, outlier = f.score([[0, 0], [8, 8]])
        self.assertGreater(outlier, normal)
        self.assertGreater(outlier, 0.6)


class AutoencoderTests(unittest.TestCase):
    def test_reconstruction_error_separates(self):
        rng = random.Random(0)
        X = []
        for _ in range(300):
            a = rng.gauss(0, 1)
            X.append([a, a, -a, 0.5 * a])          # 1-D structure the model can learn
        ae = PurePythonAutoencoder(hidden=2, epochs=15, seed=0).fit(X)
        good, bad = ae.score([[1, 1, -1, 0.5], [1, -1, 1, -3]])
        self.assertGreater(bad, good * 5)


class FeatureTests(unittest.TestCase):
    def test_rates_and_reset(self):
        recs = simulate(SimConfig(ports=1, samples=60, degrade_port=None))
        rows = port_features(recs, window=10)
        self.assertEqual(len(rows), 59 - 9)
        self.assertTrue(all(0 <= r.raw_stall <= 1 for r in rows))
        recs[30].tx_credit_zero_us = 0                 # simulated counter clear
        self.assertTrue(all(r.raw_stall >= 0 for r in port_features(recs, window=10)))

    def test_scaler_and_windows(self):
        s = PortScaler.fit([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.assertEqual(s.transform([2.0, 0.0]), [0.0, 0.0])
        self.assertEqual(windows([[1], [2], [3]], 2), [[1, 2], [2, 3]])

    def test_percentile(self):
        self.assertAlmostEqual(percentile(list(range(101)), 50), 50)


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feats = feature_table(group_by_port(simulate(SimConfig(ports=4, samples=1600, degrade_port=2,
                                                                    degrade_start=800, degrade_ramp=600))))
        cls.det = SlowDrainDetector(DetectorConfig(train_rows=500, **FAST)).fit(cls.feats)
        cls.scores, cls.alerts = cls.det.detect(cls.feats)

    def test_degrading_port_flagged(self):
        self.assertIn("2", {a.port for a in self.alerts})

    def test_learned_alert_precedes_fixed_threshold(self):
        mine = min(a.start for a in self.alerts if a.port == "2")
        fixed = [a.start for a in FixedThresholdDetector().detect(self.feats) if a.port == "2"]
        self.assertTrue(fixed, "fixed threshold should eventually fire on the degraded port")
        self.assertLess(mine, min(fixed))

    def test_healthy_ports_quiet_after_training(self):
        train_end = {k: v[499].timestamp for k, v in self.feats.items()}
        noisy = [a for a in self.alerts if a.port != "2" and a.start > train_end[(a.switch, a.port)]]
        self.assertEqual(noisy, [])

    def test_short_history_uses_pooled_model(self):
        feats = dict(self.feats)
        feats[("sw1", "new")] = self.feats[("sw1", "0")][:50]
        det = SlowDrainDetector(DetectorConfig(train_rows=500, **FAST)).fit(feats)
        self.assertTrue(det.ports[("sw1", "new")].pooled)
        det.score_port(("sw1", "new"), feats[("sw1", "new")])


class IOTests(unittest.TestCase):
    def test_csv_roundtrip(self):
        recs = simulate(SimConfig(ports=2, samples=5))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            write_csv(p, recs)
            self.assertEqual(read_csv(p), recs)

    def test_kv_parser(self):
        recs = KeyValueParser("sw9")("port=4 speed_gbps=64 tx_credit_zero_us=10 crc_err=2 tx_bytes=1e9\nnoise", 5.0)
        self.assertEqual((recs[0].port, recs[0].speed_gbps, recs[0].crc_err, recs[0].tx_bytes), ("4", 64.0, 2.0, 1e9))


if __name__ == "__main__":
    unittest.main()
