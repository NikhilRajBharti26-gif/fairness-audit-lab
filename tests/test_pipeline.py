from fairness_audit import run_synthetic
from fairness_audit.pipeline import ROW_NAMES


def test_pipeline_reproduces_the_web_demo_numbers():
    """Seed 42 with the default settings: the page's hiring run reports these values."""
    res = run_synthetic("hiring", seed=42, bias=0.6, threshold=0.5, fair_lambda=3.0,
                        target="dp", budget=0.02)
    t = res["table"]
    assert list(t.index) == ROW_NAMES
    expected_acc = [0.894, 0.891, 0.884, 0.889]
    for name, acc in zip(ROW_NAMES, expected_acc):
        assert abs(t.loc[name, "acc"] - acc) < 0.0015, name
    assert abs(t.loc["Baseline", "dp"] - 0.1113) < 0.002
    ta, tb = res["thresholds"]
    assert abs(ta - 0.58) < 0.021 and abs(tb - 0.30) < 0.021
