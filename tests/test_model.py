import numpy as np
from sklearn.linear_model import LogisticRegression

from fairness_audit import LogisticRegressionGD, generate_synthetic, split_and_standardize


def _data():
    X, y, g = generate_synthetic("hiring", seed=42, bias=0.6)
    return split_and_standardize(X, y, g)


def test_matches_scikit_learn_coefficients():
    Xtr, Xte, ytr, yte, gtr, gte = _data()
    ours = LogisticRegressionGD().fit(Xtr, ytr, gtr)
    ref = LogisticRegression(C=1 / (0.01 * len(ytr)), max_iter=5000, tol=1e-10).fit(Xtr, ytr)
    assert np.allclose(ours.coef_, ref.coef_[0], atol=1e-3)
    assert abs(ours.intercept_ - ref.intercept_[0]) < 1e-3


def test_fairness_penalty_narrows_the_group_gap_in_scores():
    Xtr, Xte, ytr, yte, gtr, gte = _data()
    base = LogisticRegressionGD().fit(Xtr, ytr, gtr)
    fair = LogisticRegressionGD(fair_lambda=5.0).fit(Xtr, ytr, gtr)

    def gap(m):
        p = m.predict_proba(Xte)
        return abs(p[gte == 1].mean() - p[gte == 0].mean())

    assert gap(fair) < gap(base)
