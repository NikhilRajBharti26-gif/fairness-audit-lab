import json
import math

import numpy as np
import pandas as pd
import pytest

import app as app_module
from app import ADULT_DISPLAY, create_app
from fairness_audit import SCENARIOS, generate_synthetic, run_synthetic, trade_off_curve
from fairness_audit.pipeline import ROW_NAMES

KEYS = ["acc", "dp", "eo", "eodds", "pp", "di"]


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def audit(client, **body):
    r = client.post("/api/audit", json=body)
    return r.status_code, r.get_json()


def test_index_is_rendered_from_template(client):
    r = client.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Fairness audit lab" in html
    assert "/api/audit" in html
    assert "function fit" not in html and "Math.imul" not in html


def test_health_and_scenarios(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}
    d = client.get("/api/scenarios").get_json()
    assert [s["key"] for s in d["scenarios"]] == list(SCENARIOS)
    assert all(s["features"] and s["positive"] and s["name"] for s in d["scenarios"])
    assert d["defaults"]["seed"] == 42


def test_default_audit_reproduces_readme_numbers(client):
    status, d = audit(client)
    assert status == 200
    assert [m["name"] for m in d["models"]] == ROW_NAMES
    expected = {
        "Baseline": [0.894, 0.111, 0.130, 0.130, 0.010, 0.598],
        "Preprocessing: reweighing": [0.891, 0.086, 0.095, 0.095, 0.051, 0.675],
        "In-processing: fairness penalty": [0.884, 0.065, 0.094, 0.094, 0.118, 0.727],
        "Post-processing: group thresholds": [0.889, 0.048, 0.216, 0.216, 0.190, 0.824],
    }
    for m in d["models"]:
        assert [round(m[k], 3) for k in KEYS] == expected[m["name"]], m["name"]
    assert abs(d["thresholds"]["A"] - 0.58) < 1e-9 and abs(d["thresholds"]["B"] - 0.30) < 1e-9
    assert (d["n_train"], d["n_test"], d["seed"]) == (2000, 1000, 42)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("target", ["dp", "eo"])
def test_api_equals_python_package(client, scenario, target):
    body = dict(scenario=scenario, seed=7, bias=0.45, threshold=0.4, lam=4.5, target=target, budget=0.03)
    status, d = audit(client, **body)
    assert status == 200
    res = run_synthetic(scenario, seed=7, bias=0.45, threshold=0.4, fair_lambda=4.5, target=target, budget=0.03)
    for m in d["models"]:
        row = res["table"].loc[m["name"]]
        for k in KEYS:
            assert m[k] == pytest.approx(row[k], abs=1e-12)
    assert (d["thresholds"]["A"], d["thresholds"]["B"]) == res["thresholds"]
    assert d["scenario"]["features"] == SCENARIOS[scenario]["features"]


def test_group_rates_and_qualification_come_from_the_data(client):
    _, d = audit(client, scenario="loan", seed=11, bias=0.5)
    X, y, g = generate_synthetic("loan", seed=11, bias=0.5)
    yte, gte = y[2000:], g[2000:]
    assert d["qualification"]["A"] == pytest.approx(yte[gte == 0].mean())
    assert d["qualification"]["B"] == pytest.approx(yte[gte == 1].mean())
    base = d["models"][0]
    assert base["dp"] == pytest.approx(abs(base["sr_A"] - base["sr_B"]))
    for m in d["models"]:
        assert 0 <= m["sr_A"] <= 1 and 0 <= m["sr_B"] <= 1


def test_trade_off_curve_is_the_backend_curve(client):
    _, d = audit(client, threshold=0.3)
    res = run_synthetic("hiring", threshold=0.3)
    ref = trade_off_curve(res["scores_test"], res["y_test"], res["g_test"])
    assert len(d["curve"]) == 19
    assert [round(p["t"], 2) for p in d["curve"]] == [round(t, 2) for t, _ in ref]
    for p, (_, rep) in zip(d["curve"], ref):
        for k in KEYS:
            assert p[k] == pytest.approx(rep[k], abs=1e-12)


def test_new_seed_gives_a_new_sample_and_same_seed_repeats(client):
    _, a = audit(client, seed=42)
    _, a2 = audit(client, seed=42)
    _, b = audit(client, seed=43)
    assert a == a2
    assert a["qualification"] != b["qualification"]
    assert a["models"][0]["acc"] != b["models"][0]["acc"]
    assert b["seed"] == 43


def test_every_control_changes_the_result(client):
    _, base = audit(client)
    _, bias0 = audit(client, bias=0.0)
    _, bias1 = audit(client, bias=1.0)
    assert abs(bias0["qualification"]["A"] - bias0["qualification"]["B"]) < abs(bias1["qualification"]["A"] - bias1["qualification"]["B"])
    _, thr = audit(client, threshold=0.2)
    assert thr["models"][0]["sr_A"] > base["models"][0]["sr_A"]
    _, lam0 = audit(client, lam=0)
    _, lam8 = audit(client, lam=8)
    assert lam0["models"][2]["dp"] > lam8["models"][2]["dp"]
    assert lam0["models"][0] == lam8["models"][0]
    _, eo = audit(client, target="eo")
    assert eo["models"][2]["eo"] != base["models"][2]["eo"]
    assert eo["models"][3]["eo"] <= base["models"][3]["eo"]
    _, b0 = audit(client, budget=0.0)
    _, b10 = audit(client, budget=0.1)
    assert b10["thresholds"] != b0["thresholds"]
    _, risk = audit(client, scenario="risk")
    assert risk["scenario"]["name"] == "Risk scoring" and risk["models"][0]["acc"] != base["models"][0]["acc"]


def test_extreme_valid_settings_stay_finite(client):
    for body in [dict(bias=1.0, threshold=0.95, lam=8, budget=0.1, target="eo"),
                 dict(bias=0.0, threshold=0.05, lam=0, budget=0.0, scenario="admission")]:
        status, d = audit(client, **body)
        assert status == 200
        values = [m[k] for m in d["models"] for k in KEYS + ["sr_A", "sr_B"]]
        assert all(math.isfinite(v) for v in values)


@pytest.mark.parametrize("body", [
    {"scenario": "nope"}, {"target": "xx"}, {"threshold": 1.5}, {"threshold": 0.01}, {"bias": -5},
    {"bias": 1.5}, {"budget": -1}, {"budget": 0.5}, {"lam": 99}, {"lam": -1}, {"seed": -1},
    {"seed": 1.5}, {"seed": True}, {"seed": 2 ** 32}, {"bias": "0.5"}, {"bias": None}, {"extra": 1}, [1, 2],
])
def test_invalid_input_is_rejected_with_400(client, body):
    r = client.post("/api/audit", json=body)
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_malformed_requests(client):
    assert client.post("/api/audit", data="{oops", content_type="application/json").status_code == 400
    assert client.post("/api/audit", data="{}", content_type="text/plain").status_code == 400
    r = client.get("/api/audit")
    assert r.status_code == 405 and "error" in r.get_json()
    assert client.get("/api/missing").status_code == 404


def test_adult_endpoint_matches_the_csv(client):
    d = client.get("/api/adult").get_json()
    df = pd.read_csv(app_module.ADULT_CSV, index_col=0)
    assert d["precomputed"] is True and d["file"] == "results/adult_results.csv"
    assert [r["label"] for r in d["rows"]] == [x[0] for x in ADULT_DISPLAY]
    for r in d["rows"]:
        for k in KEYS:
            assert r[k] == pytest.approx(df.loc[r["csv_row"], k], abs=1e-12)
    full = client.get("/api/adult?view=all").get_json()
    assert [r["label"] for r in full["rows"]] == list(df.index)


def test_adult_csv_rows_reproduce_the_page_table(client):
    rows = {r["label"]: r for r in client.get("/api/adult").get_json()["rows"]}
    assert round(rows["Baseline"]["acc"] * 100, 1) == 84.5
    assert round(rows["Baseline"]["dp"], 3) == 0.174
    assert round(rows["Reweighing"]["di"], 3) == 0.553
    assert round(rows["In-processing, DP constraint"]["dp"], 3) == 0.020
    assert round(rows["Post-processing, EO"]["eo"], 3) == 0.023


def test_adult_missing_file_is_a_clear_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "ADULT_CSV", tmp_path / "nope.csv")
    r = client.get("/api/adult")
    assert r.status_code == 404 and "run_adult.py" in r.get_json()["error"]
