import numpy as np
import pytest
from fairlearn.metrics import (demographic_parity_difference, demographic_parity_ratio,
                               equalized_odds_difference)
from sklearn.metrics import precision_score, recall_score

from fairness_audit import fairness_report


@pytest.mark.parametrize("seed", range(5))
def test_matches_fairlearn_and_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = 400
    g = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    pred = (rng.random(n) < 0.3 + 0.4 * y).astype(int)
    r = fairness_report(y, pred, g)
    assert np.isclose(r["acc"], (pred == y).mean())
    assert np.isclose(r["dp"], demographic_parity_difference(y, pred, sensitive_features=g))
    assert np.isclose(r["eodds"], equalized_odds_difference(y, pred, sensitive_features=g))
    assert np.isclose(r["di"], demographic_parity_ratio(y, pred, sensitive_features=g))
    tpr = [recall_score(y[g == k], pred[g == k], zero_division=0) for k in (0, 1)]
    ppv = [precision_score(y[g == k], pred[g == k], zero_division=0) for k in (0, 1)]
    assert np.isclose(r["eo"], abs(tpr[0] - tpr[1]))
    assert np.isclose(r["pp"], abs(ppv[0] - ppv[1]))
