"""The three mitigation strategies: pre-processing (reweighing), in-processing (the fairness penalty
lives in model.py), and post-processing (group-specific thresholds)."""
from __future__ import annotations

import numpy as np

from .metrics import fairness_report


def reweighing_weights(y, g):
    """Kamiran and Calmon (2012) reweighing: w(g, y) = P(g) * P(y) / P(g, y).

    Returns one weight per training example. After weighting, the outcome rate is the same in
    every group, so the learner cannot pick up the group-outcome link from the training data.
    """
    y, g = np.asarray(y), np.asarray(g)
    N = len(y)
    w = np.zeros(N)
    for gk in (0, 1):
        for yk in (0, 1):
            m = (g == gk) & (y == yk)
            n_gy = m.sum()
            if n_gy:
                w[m] = (g == gk).sum() * (y == yk).sum() / (N * n_gy)
    return w


def find_group_thresholds(scores, y, g, base_accuracy, budget, target="dp", step=0.02):
    """Choose one threshold per group (Hardt, Price and Srebro style, by grid search).

    Searches every pair (threshold_A, threshold_B) on a grid, keeps pairs whose accuracy is within
    `budget` of base_accuracy, and returns the pair with the smallest gap on the target metric
    (target "dp" = demographic parity, "eo" = equal opportunity). Ties go to higher accuracy.
    Thresholds should be chosen on training data and then applied to test data.
    """
    scores, y, g = np.asarray(scores), np.asarray(y), np.asarray(g)
    grid = np.round(np.arange(0, 1 + step / 2, step), 10)
    best_key, best = None, None
    for a in grid:
        for b in grid:
            pred = (scores >= np.where(g == 1, b, a)).astype(int)
            r = fairness_report(y, pred, g)
            gap = r["eo"] if target == "eo" else r["dp"]
            ok = r["acc"] >= base_accuracy - budget
            key = (not ok, round(gap, 9), -r["acc"])
            if best_key is None or key < best_key:
                best_key, best = key, (float(a), float(b))
    return best
