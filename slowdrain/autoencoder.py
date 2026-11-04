# Author / Owner / Publisher: Derek Asir Muthurajan Caleb
# Copyright (c) 2026 Derek Asir Muthurajan Caleb
# SPDX-License-Identifier: Apache-2.0
"""Sliding-window autoencoder scored by reconstruction error (paper Sec. IV-C).

``make_autoencoder(backend="auto")`` returns a PyTorch model when torch is
installed, otherwise a small pure-Python MLP autoencoder (one tanh bottleneck,
Adam optimiser).  Both expose ``fit(X)`` and ``score(X)`` -> mean squared
reconstruction error per row (higher = more anomalous).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence


class PurePythonAutoencoder:
    def __init__(self, hidden: int = 8, epochs: int = 25, lr: float = 0.01,
                 max_train: int = 1500, seed: Optional[int] = 0) -> None:
        self.hidden, self.epochs, self.lr, self.max_train = hidden, epochs, lr, max_train
        self.rng = random.Random(seed)
        self.d = 0

    def _init(self, d: int) -> None:
        self.d = d
        h = self.hidden
        a, b = math.sqrt(6 / (d + h)), math.sqrt(6 / (h + d))
        self.W1 = [[self.rng.uniform(-a, a) for _ in range(d)] for _ in range(h)]
        self.b1 = [0.0] * h
        self.W2 = [[self.rng.uniform(-b, b) for _ in range(h)] for _ in range(d)]
        self.b2 = [0.0] * d
        self._m = {k: self._zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._v = {k: self._zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._t = 0

    @staticmethod
    def _zeros_like(p):
        return [[0.0] * len(p[0]) for _ in p] if isinstance(p[0], list) else [0.0] * len(p)

    def _forward(self, x: Sequence[float]):
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        y = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return h, y

    def _adam(self, name: str, grad, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        p, m, v = getattr(self, name), self._m[name], self._v[name]
        c1, c2 = 1 - beta1 ** self._t, 1 - beta2 ** self._t
        if isinstance(p[0], list):
            for i in range(len(p)):
                pi, mi, vi, gi = p[i], m[i], v[i], grad[i]
                for j in range(len(pi)):
                    g = gi[j]
                    mi[j] = beta1 * mi[j] + (1 - beta1) * g
                    vi[j] = beta2 * vi[j] + (1 - beta2) * g * g
                    pi[j] -= self.lr * (mi[j] / c1) / (math.sqrt(vi[j] / c2) + eps)
        else:
            for j in range(len(p)):
                g = grad[j]
                m[j] = beta1 * m[j] + (1 - beta1) * g
                v[j] = beta2 * v[j] + (1 - beta2) * g * g
                p[j] -= self.lr * (m[j] / c1) / (math.sqrt(v[j] / c2) + eps)

    def fit(self, X: Sequence[Sequence[float]]) -> "PurePythonAutoencoder":
        X = [list(x) for x in X]
        if not X:
            raise ValueError("cannot fit on empty data")
        if len(X) > self.max_train:
            X = self.rng.sample(X, self.max_train)
        self._init(len(X[0]))
        d, hdim = self.d, self.hidden
        for _ in range(self.epochs):
            self.rng.shuffle(X)
            for x in X:                                  # per-sample Adam steps
                h, y = self._forward(x)
                dy = [2.0 * (yi - xi) / d for yi, xi in zip(y, x)]
                gW2 = [[dyi * hj for hj in h] for dyi in dy]
                gb2 = dy
                dh = [sum(self.W2[i][j] * dy[i] for i in range(d)) * (1 - h[j] ** 2) for j in range(hdim)]
                gW1 = [[dhj * xi for xi in x] for dhj in dh]
                gb1 = dh
                self._t += 1
                self._adam("W2", gW2); self._adam("b2", gb2)
                self._adam("W1", gW1); self._adam("b1", gb1)
        return self

    def score(self, X: Sequence[Sequence[float]]) -> List[float]:
        out = []
        for x in X:
            _, y = self._forward(x)
            out.append(sum((yi - xi) ** 2 for yi, xi in zip(y, x)) / len(x))
        return out


class TorchAutoencoder:  # pragma: no cover - exercised only when torch is installed
    def __init__(self, hidden: int = 8, epochs: int = 60, lr: float = 1e-3, batch: int = 64,
                 seed: Optional[int] = 0, **_ignored) -> None:
        import torch
        self.torch = torch
        self.hidden, self.epochs, self.lr, self.batch = hidden, epochs, lr, batch
        if seed is not None:
            torch.manual_seed(seed)
        self.model = None

    def fit(self, X):
        torch, nn = self.torch, self.torch.nn
        data = torch.tensor(X, dtype=torch.float32)
        d = data.shape[1]
        mid = max(self.hidden * 2, d // 2)
        self.model = nn.Sequential(nn.Linear(d, mid), nn.ReLU(), nn.Linear(mid, self.hidden), nn.ReLU(),
                                   nn.Linear(self.hidden, mid), nn.ReLU(), nn.Linear(mid, d))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(data))
            for i in range(0, len(data), self.batch):
                xb = data[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = loss_fn(self.model(xb), xb)
                loss.backward()
                opt.step()
        return self

    def score(self, X) -> List[float]:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(X, dtype=torch.float32)
            err = ((self.model(data) - data) ** 2).mean(dim=1)
        return [float(e) for e in err]


def make_autoencoder(backend: str = "auto", **kw):
    if backend in ("auto", "torch"):
        try:
            return TorchAutoencoder(**kw)
        except ImportError:
            if backend == "torch":
                raise
    return PurePythonAutoencoder(**kw)
