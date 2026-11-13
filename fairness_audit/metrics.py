"""Group fairness metrics computed from predictions, the same definitions as the web demo."""
from __future__ import annotations

import numpy as np


def _safe_div(a, b):
    return a / b if b else 0.0


def group_stats(y, pred, g):
    """Per-group selection rate, true positive rate, false positive rate and precision."""
    y, pred, g = np.asarray(y), np.asarray(pred), np.asarray(g)
    out = []
    for k in (0, 1):
        m = g == k
        yk, pk = y[m] == 1, pred[m] == 1
        tp = np.sum(pk & yk)
        fp = np.sum(pk & ~yk)
        fn = np.sum(~pk & yk)
        tn = np.sum(~pk & ~yk)
        out.append({"sr": _safe_div(tp + fp, m.sum()), "tpr": _safe_div(tp, tp + fn),
                    "fpr": _safe_div(fp, fp + tn), "ppv": _safe_div(tp, tp + fp)})
    return out


def fairness_report(y, pred, g):
    """Accuracy and the four fairness gaps, plus disparate impact.

    dp    demographic parity difference   |SR_A - SR_B|
    eo    equal opportunity difference    |TPR_A - TPR_B|
    eodds equalized odds difference       max(|TPR gap|, |FPR gap|)
    pp    predictive parity difference    |PPV_A - PPV_B|
    di    disparate impact ratio          min(SR) / max(SR)   (fails the four-fifths rule below 0.8)
    """
    y, pred, g = np.asarray(y), np.asarray(pred), np.asarray(g)
    a, b = group_stats(y, pred, g)
    eo = abs(a["tpr"] - b["tpr"])
    fg = abs(a["fpr"] - b["fpr"])
    hi = max(a["sr"], b["sr"])
    return {
        "acc": float(np.mean(pred == y)),
        "dp": abs(a["sr"] - b["sr"]),
        "eo": eo,
        "eodds": max(eo, fg),
        "pp": abs(a["ppv"] - b["ppv"]),
        "di": min(a["sr"], b["sr"]) / (hi if hi else 1.0),
        "sr_A": a["sr"],
        "sr_B": b["sr"],
    }


def trade_off_curve(scores, y, g, thresholds=None):
    """Metrics for one model at every decision threshold (used for the trade-off chart)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    return [(float(t), fairness_report(y, (np.asarray(scores) >= t).astype(int), g)) for t in thresholds]
