"""Real-data benchmark on the UCI Adult income dataset (sex is the sensitive attribute).

    python scripts/run_adult.py --data path/to/adult-all.csv

The data file is the headerless UCI adult.data format. A copy is at:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from fairlearn.postprocessing import ThresholdOptimizer  # noqa: E402
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient  # noqa: E402

from fairness_audit import (LogisticRegressionGD, fairness_report, find_group_thresholds,  # noqa: E402
                            load_adult, reweighing_weights, split_adult)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the Adult CSV")
    ap.add_argument("--out", default="results/adult_results.csv")
    a = ap.parse_args()

    X, y, sex = load_adult(a.data)
    Xtr, Xte, ytr, yte, s_tr, s_te = split_adult(X, y, sex)
    n = len(ytr)
    print(f"people={len(y)}  train={n}  test={len(yte)}  features={Xtr.shape[1]}")
    print(f"base rate female={y[sex == 0].mean():.3f}  male={y[sex == 1].mean():.3f}\n")

    rows = {}
    base = LogisticRegressionGD().fit(Xtr, ytr, s_tr)
    sc_tr, sc_te = base.predict_proba(Xtr), base.predict_proba(Xte)
    base_acc = fairness_report(ytr, (sc_tr >= 0.5).astype(int), s_tr)["acc"]
    rows["Baseline"] = fairness_report(yte, (sc_te >= 0.5).astype(int), s_te)

    rw = LogisticRegressionGD().fit(Xtr, ytr, s_tr, sample_weight=reweighing_weights(ytr, s_tr))
    rows["Reweighing (ours)"] = fairness_report(yte, rw.predict(Xte), s_te)

    for tag, eo in [("DP", False), ("EO", True)]:
        m = LogisticRegressionGD(fair_lambda=3.0, equal_opportunity=eo).fit(Xtr, ytr, s_tr)
        rows[f"Penalty, {tag} (ours)"] = fairness_report(yte, m.predict(Xte), s_te)

    for tag, target in [("DP", "dp"), ("EO", "eo")]:
        ta, tb = find_group_thresholds(sc_tr, ytr, s_tr, base_acc, 0.02, target)
        pred = (sc_te >= np.where(s_te == 1, tb, ta)).astype(int)
        rows[f"Group thresholds, {tag} (ours)"] = fairness_report(yte, pred, s_te)

    # fairlearn reference implementations (scikit-learn's logistic regression with the same penalty)
    C = 1 / (0.01 * n)
    for tag, con in [("DP", DemographicParity()), ("EO", EqualizedOdds())]:
        eg = ExponentiatedGradient(LogisticRegression(C=C, max_iter=5000), constraints=con)
        eg.fit(Xtr, ytr, sensitive_features=s_tr)
        rows[f"fairlearn ExponentiatedGradient, {tag}"] = fairness_report(
            yte, np.asarray(eg.predict(Xte, random_state=0)), s_te)

    ref = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
    for tag, con in [("DP", "demographic_parity"), ("EO", "equalized_odds")]:
        to = ThresholdOptimizer(estimator=ref, constraints=con, prefit=True,
                                predict_method="predict_proba", objective="accuracy_score")
        to.fit(Xtr, ytr, sensitive_features=s_tr)
        rows[f"fairlearn ThresholdOptimizer, {tag}"] = fairness_report(
            yte, np.asarray(to.predict(Xte, sensitive_features=s_te, random_state=0)), s_te)

    table = pd.DataFrame(rows).T[["acc", "dp", "eo", "eodds", "pp", "di"]]
    print(table.round(3).to_string())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
