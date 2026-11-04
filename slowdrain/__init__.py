# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""slowdrain - predictive slow-drain / IOPS-latency anomaly detection for Fibre Channel fabrics.

Companion code for "Catching the Slow Drain Early: Predictive IOPS/Latency
Anomaly Detection for Fibre Channel Fabrics with PyTorch and Scikit-learn"
(D. A. M. Caleb, 2026).

Uses scikit-learn's IsolationForest and a PyTorch autoencoder when they are
installed, and pure-Python implementations of both otherwise.
"""
__version__ = "0.1.0"
__author__ = "Derek Asir Muthurajan Caleb"
__copyright__ = "Copyright (c) 2026 Derek Asir Muthurajan Caleb"

from .detector import Alert, DetectorConfig, FixedThresholdDetector, SlowDrainDetector
from .features import FEATURES, feature_table, port_features
from .telemetry import Record, SimConfig, group_by_port, read_csv, simulate, write_csv

__all__ = ["Alert", "DetectorConfig", "FixedThresholdDetector", "SlowDrainDetector", "FEATURES",
           "feature_table", "port_features", "Record", "SimConfig", "group_by_port", "read_csv",
           "simulate", "write_csv", "__version__"]
