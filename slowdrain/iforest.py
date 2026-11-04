# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""Isolation Forest (Liu, Ting & Zhou, ICDM 2008).

``make_isolation_forest(backend="auto")`` returns a scikit-learn model when
scikit-learn is installed, otherwise a pure-Python implementation of the same
algorithm.  Both expose ``fit(X)`` and ``score(X)`` where a *higher* score means
*more anomalous* (the paper's convention; scikit-learn's sign is flipped here).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

EULER = 0.5772156649


def _c(n: int) -> float:
    """Average path length of an unsuccessful BST search over n points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER) - 2.0 * (n - 1) / n


class _Node:
    __slots__ = ("feat", "split", "left", "right", "size")

    def __init__(self, feat=-1, split=0.0, left=None, right=None, size=0):
        self.feat, self.split, self.left, self.right, self.size = feat, split, left, right, size


class PurePythonIsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, seed: Optional[int] = 0) -> None:
        self.n_estimators, self.max_samples = n_estimators, max_samples
        self.rng = random.Random(seed)
        self.trees: List[_Node] = []
        self.psi = 0

    def _build(self, X: List[Sequence[float]], depth: int, limit: int) -> _Node:
        n = len(X)
        if depth >= limit or n <= 1:
            return _Node(size=n)
        dims = len(X[0])
        # pick a feature that actually varies in this partition
        for _ in range(dims * 2):
            f = self.rng.randrange(dims)
            lo = min(x[f] for x in X)
            hi = max(x[f] for x in X)
            if hi > lo:
                break
        else:
            return _Node(size=n)
        s = self.rng.uniform(lo, hi)
        left = [x for x in X if x[f] < s]
        right = [x for x in X if x[f] >= s]
        return _Node(f, s, self._build(left, depth + 1, limit), self._build(right, depth + 1, limit))

    def fit(self, X: Sequence[Sequence[float]]) -> "PurePythonIsolationForest":
        X = list(X)
        if not X:
            raise ValueError("cannot fit on empty data")
        self.psi = min(self.max_samples, len(X))
        limit = math.ceil(math.log2(max(self.psi, 2)))
        self.trees = [self._build(self.rng.sample(X, self.psi), 0, limit) for _ in range(self.n_estimators)]
        return self

    def _path(self, x: Sequence[float], node: _Node, depth: int = 0) -> float:
        while node.feat >= 0:
            node = node.left if x[node.feat] < node.split else node.right
            depth += 1
        return depth + _c(node.size)

    def score(self, X: Sequence[Sequence[float]]) -> List[float]:
        c = _c(self.psi)
        return [2.0 ** (-(sum(self._path(x, t) for t in self.trees) / len(self.trees)) / c) for x in X]


class SklearnIsolationForest:  # pragma: no cover - exercised only when scikit-learn is installed
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, seed: Optional[int] = 0) -> None:
        from sklearn.ensemble import IsolationForest
        self.m = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X) -> List[float]:
        # sklearn: score_samples is higher for normal points; negate to match the paper
        return [float(-s) for s in self.m.score_samples(X)]


def make_isolation_forest(backend: str = "auto", **kw):
    if backend in ("auto", "sklearn"):
        try:
            return SklearnIsolationForest(**kw)
        except ImportError:
            if backend == "sklearn":
                raise
    return PurePythonIsolationForest(**kw)
