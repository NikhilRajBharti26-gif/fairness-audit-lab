import numpy as np

from fairness_audit import Mulberry32, generate_synthetic


def test_mulberry32_matches_javascript_values():
    # First three outputs of the page's generator for seed 42 (checked in Node.js)
    rng = Mulberry32(42)
    got = [rng.random() for _ in range(3)]
    expected = [0.60110375192016363, 0.44829055899754167, 0.85246579349040985]
    assert np.allclose(got, expected, atol=1e-15)


def test_generation_is_deterministic_and_seed_dependent():
    X1, y1, g1 = generate_synthetic("hiring", seed=42, n=500)
    X2, y2, g2 = generate_synthetic("hiring", seed=42, n=500)
    X3, _, _ = generate_synthetic("hiring", seed=43, n=500)
    assert X1.shape == (500, 3)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2) and np.array_equal(g1, g2)
    assert not np.array_equal(X1, X3)


def test_group_share_and_historical_disadvantage():
    _, y, g = generate_synthetic("hiring", seed=42, bias=0.6, n=3000)
    assert abs(g.mean() - 0.40) < 0.03
    assert y[g == 1].mean() < y[g == 0].mean()
