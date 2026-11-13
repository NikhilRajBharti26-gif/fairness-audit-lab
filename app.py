import math
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, HTTPException

from fairness_audit import SCENARIOS, run_synthetic, trade_off_curve

ROOT = Path(__file__).resolve().parent
ADULT_CSV = ROOT / "results" / "adult_results.csv"
METRICS = ["acc", "dp", "eo", "eodds", "pp", "di"]

ADULT_DISPLAY = [
    ("Baseline", "Baseline", "ours"),
    ("Reweighing", "Reweighing (ours)", "ours"),
    ("In-processing, DP constraint", "fairlearn ExponentiatedGradient, DP", "fairlearn"),
    ("In-processing, EO constraint", "fairlearn ExponentiatedGradient, EO", "fairlearn"),
    ("Post-processing, DP", "fairlearn ThresholdOptimizer, DP", "fairlearn"),
    ("Post-processing, EO", "fairlearn ThresholdOptimizer, EO", "fairlearn"),
]

DEFAULTS = {"scenario": "hiring", "seed": 42, "bias": 0.6, "threshold": 0.5,
            "lam": 3.0, "target": "dp", "budget": 0.02}
RANGES = {"bias": (0.0, 1.0), "threshold": (0.05, 0.95), "lam": (0.0, 8.0), "budget": (0.0, 0.1)}
MAX_SEED = 2 ** 32 - 1


def parse_settings(payload):
    if not isinstance(payload, dict):
        raise BadRequest("Request body must be a JSON object.")
    unknown = sorted(set(payload) - set(DEFAULTS))
    if unknown:
        raise BadRequest(f"Unknown field(s): {', '.join(unknown)}.")
    s = {**DEFAULTS, **payload}
    if s["scenario"] not in SCENARIOS:
        raise BadRequest(f"scenario must be one of: {', '.join(SCENARIOS)}.")
    if s["target"] not in ("dp", "eo"):
        raise BadRequest("target must be 'dp' or 'eo'.")
    seed = s["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise BadRequest(f"seed must be an integer between 0 and {MAX_SEED}.")
    for key, (lo, hi) in RANGES.items():
        v = s[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise BadRequest(f"{key} must be a number between {lo} and {hi}.")
        s[key] = float(v)
    return s


def clean(report):
    return {k: float(v) for k, v in report.items()}


def audit(s):
    res = run_synthetic(s["scenario"], seed=s["seed"], bias=s["bias"], threshold=s["threshold"],
                        fair_lambda=s["lam"], target=s["target"], budget=s["budget"])
    sc = SCENARIOS[s["scenario"]]
    curve = trade_off_curve(res["scores_test"], res["y_test"], res["g_test"])
    ta, tb = res["thresholds"]
    return {
        "scenario": {"key": s["scenario"], "name": sc["name"], "p_group_b": sc["p_group_b"],
                     "positive": sc["positive"], "features": sc["features"]},
        "settings": s,
        "seed": s["seed"],
        "n_train": res["n_train"],
        "n_test": res["n_test"],
        "qualification": {"A": res["qualification"][0], "B": res["qualification"][1]},
        "models": [{"name": name, **clean(rep)} for name, rep in res["reports"].items()],
        "thresholds": {"A": float(ta), "B": float(tb)},
        "curve": [{"t": t, **clean(rep)} for t, rep in curve],
    }


def adult_payload(show_all):
    df = pd.read_csv(ADULT_CSV, index_col=0)
    if show_all:
        rows = [{"label": name, "csv_row": name, **{k: float(r[k]) for k in METRICS}}
                for name, r in df.iterrows()]
    else:
        rows = [{"label": label, "csv_row": key, "source": src, **{k: float(df.loc[key, k]) for k in METRICS}}
                for label, key, src in ADULT_DISPLAY]
    return {"precomputed": True, "file": "results/adult_results.csv", "dataset": "UCI Adult",
            "sensitive_attribute": "sex", "rows": rows}


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/scenarios")
    def scenarios():
        return jsonify({"defaults": DEFAULTS, "ranges": RANGES,
                        "scenarios": [{"key": k, **v} for k, v in SCENARIOS.items()]})

    @app.post("/api/audit")
    def audit_route():
        payload = request.get_json(silent=True)
        if payload is None:
            raise BadRequest("Request body must be valid JSON.")
        return jsonify(audit(parse_settings(payload)))

    @app.get("/api/adult")
    def adult():
        if not ADULT_CSV.exists():
            return jsonify({"error": "results/adult_results.csv not found. Run scripts/run_adult.py first."}), 404
        return jsonify(adult_payload(request.args.get("view") == "all"))

    @app.errorhandler(HTTPException)
    def http_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": e.description}), e.code
        return e

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
