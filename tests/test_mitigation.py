import numpy as np

from fairness_audit import find_group_thresholds, reweighing_weights


def test_reweighing_removes_group_outcome_link():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, 2000)
    y = (rng.random(2000) < np.where(g == 1, 0.2, 0.5)).astype(int)
    w = reweighing_weights(y, g)
    rates = [np.average(y[g == k], weights=w[g == k]) for k in (0, 1)]
    assert abs(rates[0] - rates[1]) < 1e-9
    assert np.isclose(np.average(y, weights=w), y.mean())


def test_group_thresholds_respect_the_accuracy_budget():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 2, 1500)
    y = rng.integers(0, 2, 1500)
    s = np.clip(0.3 + 0.4 * y + 0.1 * rng.standard_normal(1500) - 0.1 * g, 0, 1)
    base_acc = ((s >= 0.5).astype(int) == y).mean()
    ta, tb = find_group_thresholds(s, y, g, base_acc, budget=0.02, target="dp")
    pred = (s >= np.where(g == 1, tb, ta)).astype(int)
    assert (pred == y).mean() >= base_acc - 0.02 - 1e-12
    assert 0 <= ta <= 1 and 0 <= tb <= 1
