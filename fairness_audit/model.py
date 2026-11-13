"""Logistic regression trained by gradient descent, with optional sample weights and a fairness penalty."""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionGD:
    """Minimises

        sum_i sw_i * logloss_i / sum(sw)  +  l2 * ||w||^2  +  fair_lambda * (m1 - m0)^2

    where m_k is the mean predicted probability of group k. With equal_opportunity=True the mean is
    taken over truly positive examples only (equal-opportunity penalty); otherwise over everyone
    (demographic-parity penalty). The intercept is not penalised. Same objective and step size as
    the browser demo, so the two produce the same coefficients.
    """

    def __init__(self, l2: float = 0.005, lr: float = 0.6, n_iter: int = 600,
                 fair_lambda: float = 0.0, equal_opportunity: bool = False):
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.fair_lambda = fair_lambda
        self.equal_opportunity = equal_opportunity
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y, g, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        g = np.asarray(g)
        n, d = X.shape
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sumw = sw.sum()
        mask = (y == 1) if self.equal_opportunity else np.ones(n, dtype=bool)
        s0, s1 = mask & (g == 0), mask & (g == 1)
        w, b = np.zeros(d), 0.0

        for _ in range(self.n_iter):
            p = sigmoid(X @ w + b)
            e = (p - y) * sw / sumw
            gw = X.T @ e + 2 * self.l2 * w
            gb = e.sum()
            if self.fair_lambda > 0 and s0.any() and s1.any():
                q = p * (1 - p)
                m0, m1 = p[s0].mean(), p[s1].mean()
                dm0_w, dm1_w = (q[s0] @ X[s0]) / s0.sum(), (q[s1] @ X[s1]) / s1.sum()
                dm0_b, dm1_b = q[s0].mean(), q[s1].mean()
                k = 2 * self.fair_lambda * (m1 - m0)
                gw = gw + k * (dm1_w - dm0_w)
                gb = gb + k * (dm1_b - dm0_b)
            w = w - self.lr * gw
            b = b - self.lr * gb

        self.coef_, self.intercept_ = w, b
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
