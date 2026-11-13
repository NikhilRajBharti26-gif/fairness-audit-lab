"""End-to-end experiment on synthetic data: the four rows shown in the web demo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import generate_synthetic, split_and_standardize
from .metrics import fairness_report
from .mitigation import find_group_thresholds, reweighing_weights
from .model import LogisticRegressionGD

ROW_NAMES = ["Baseline", "Preprocessing: reweighing", "In-processing: fairness penalty",
             "Post-processing: group thresholds"]


def run_synthetic(scenario="hiring", seed=42, bias=0.6, threshold=0.5, fair_lambda=3.0,
                  target="dp", budget=0.02, n_total=3000, n_train=2000):
    """Train the baseline and the three mitigated models, and evaluate all four on the test set.

    Returns a dict with the metrics table (one row per model), the chosen group thresholds,
    and the baseline test scores (for plotting trade-off curves).
    """
    X, y, g = generate_synthetic(scenario, seed=seed, bias=bias, n=n_total)
    Xtr, Xte, ytr, yte, gtr, gte = split_and_standardize(X, y, g, n_train=n_train)
    eo = target == "eo"

    base = LogisticRegressionGD().fit(Xtr, ytr, gtr)
    s_tr, s_te = base.predict_proba(Xtr), base.predict_proba(Xte)

    rw = LogisticRegressionGD().fit(Xtr, ytr, gtr, sample_weight=reweighing_weights(ytr, gtr))
    ip = LogisticRegressionGD(fair_lambda=fair_lambda, equal_opportunity=eo).fit(Xtr, ytr, gtr)

    base_acc_train = fairness_report(ytr, (s_tr >= threshold).astype(int), gtr)["acc"]
    ta, tb = find_group_thresholds(s_tr, ytr, gtr, base_acc_train, budget, target)
    post_pred = (s_te >= np.where(gte == 1, tb, ta)).astype(int)

    preds = {
        ROW_NAMES[0]: (s_te >= threshold).astype(int),
        ROW_NAMES[1]: rw.predict(Xte, threshold),
        ROW_NAMES[2]: ip.predict(Xte, threshold),
        ROW_NAMES[3]: post_pred,
    }
    reports = {name: fairness_report(yte, p, gte) for name, p in preds.items()}
    table = pd.DataFrame(reports).T[["acc", "dp", "eo", "eodds", "pp", "di"]]
    qualified = [float(yte[gte == k].mean()) if (gte == k).any() else 0.0 for k in (0, 1)]
    return {"table": table, "thresholds": (ta, tb), "scores_test": s_te,
            "y_test": yte, "g_test": gte, "base_accuracy_train": base_acc_train,
            "reports": reports, "qualification": qualified,
            "n_train": int(len(ytr)), "n_test": int(len(yte))}
