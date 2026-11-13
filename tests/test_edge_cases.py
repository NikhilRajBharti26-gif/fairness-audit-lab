import numpy as np

from fairness_audit import generate_synthetic, reweighing_weights, split_and_standardize
from fairness_audit.data import SCENARIOS


def test_reweighing_survives_an_empty_group_outcome_cell():
    y = np.array([1, 0, 0, 1, 0, 0])
    g = np.array([0, 0, 0, 0, 1, 1])
    w = reweighing_weights(y, g)
    assert np.all(np.isfinite(w)) and np.all(w > 0)


def test_reweighing_weights_are_finite_for_every_valid_scenario_and_bias():
    for scenario in SCENARIOS:
        for bias in (0.0, 0.5, 1.0):
            X, y, g = generate_synthetic(scenario, seed=42, bias=bias)
            _, _, ytr, _, gtr, _ = split_and_standardize(X, y, g)
            w = reweighing_weights(ytr, gtr)
            assert np.all(np.isfinite(w)) and np.all(w > 0)
            assert min(((gtr == a) & (ytr == c)).sum() for a in (0, 1) for c in (0, 1)) > 0
