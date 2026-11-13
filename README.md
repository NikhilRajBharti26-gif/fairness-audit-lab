# Fairness Audit Lab

A fairness audit dashboard backed by a Python machine-learning package. The page in the
browser sends every control value to a Flask JSON API, and the numbers, charts and tables it
shows are the ones the `fairness_audit` package computes. The repository also holds a
real-data benchmark on the UCI Adult income dataset and a test suite.

## Project layout

```
app.py             Flask app: serves the page, validates requests, returns JSON
templates/
  index.html       the dashboard (rendered by Flask, all calculations come from the API)
legacy/
  Fairness_audit_lab_standalone.html   the original self-contained page with its own JavaScript ML
fairness_audit/
  __init__.py    public API of the package
  data.py        synthetic generator (matches the page), Adult loader and split
  model.py       logistic regression by gradient descent, with sample weights and the fairness penalty
  metrics.py     accuracy, demographic parity, equal opportunity, equalized odds, predictive parity, disparate impact
  mitigation.py  reweighing (pre-processing), group-threshold search (post-processing)
  pipeline.py    the end-to-end experiment behind the four rows of the page
scripts/
  run_synthetic.py   command-line version of the web demo
  run_adult.py       real-data benchmark (ours plus fairlearn references)
tests/               pytest suite: fairlearn and scikit-learn checks, the page's numbers, Flask API tests
results/             output tables
```

## Setup

```bash
pip install -r requirements.txt
```

Developed and tested with Python 3.12. The pinned minimums in `requirements.txt` are the ones the
code needs; the full test suite was run with numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
fairlearn 0.15.0, Flask 3.1.3 and pytest 9.2.0.

## Run the web app

```bash
python app.py
```

Open http://127.0.0.1:5000. Use another port with `PORT=8000 python app.py`.

The controls on the left (scenario, historical disadvantage, decision threshold, fairness penalty,
fairness target, accuracy budget) and the "Draw a new sample" button each trigger a request to the
backend. "Draw a new sample" increases the seed by one, which generates a different set of people.
The seed starts at 42 and is shown under the controls.

Two kinds of results appear on the page:

- **Live synthetic audit** (everything above the Adult section): computed by the Python package on every
  change, for the scenario and seed you choose.
- **Precomputed Adult benchmark** (the real-data table): read from `results/adult_results.csv`, the saved
  output of `scripts/run_adult.py`. It does not change with the controls. To refresh it, re-run
  `python scripts/run_adult.py --data adult-all.csv` and reload the page.

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | the dashboard |
| `GET /api/health` | `{"status": "ok"}` |
| `GET /api/scenarios` | scenarios, default control values and allowed ranges |
| `POST /api/audit` | run the four approaches and return metrics, group rates, thresholds and the trade-off curve |
| `GET /api/adult` | the six Adult benchmark rows shown on the page (`?view=all` returns all ten rows of the CSV) |

`POST /api/audit` takes a JSON object. Every field is optional and falls back to the default:

| Field | Default | Allowed |
|---|---|---|
| `scenario` | `hiring` | `hiring`, `loan`, `admission`, `risk` |
| `seed` | `42` | integer, 0 to 4294967295 |
| `bias` | `0.6` | 0 to 1 |
| `threshold` | `0.5` | 0.05 to 0.95 |
| `lam` | `3.0` | 0 to 8 |
| `target` | `dp` | `dp`, `eo` |
| `budget` | `0.02` | 0 to 0.1 |

```bash
curl -s -X POST http://127.0.0.1:5000/api/audit \
  -H "Content-Type: application/json" \
  -d '{"scenario": "loan", "seed": 43, "bias": 0.4, "target": "eo", "lam": 5}'
```

The response holds `scenario`, `settings`, `seed`, `n_train`, `n_test`, `qualification` (share of each
group that is truly positive in the test data), `models` (the four rows, each with `acc`, `dp`, `eo`,
`eodds`, `pp`, `di`, `sr_A`, `sr_B`), `thresholds` (the group thresholds chosen on training data) and
`curve` (19 baseline results from threshold 0.05 to 0.95). Invalid input returns HTTP 400 with
`{"error": "..."}`.

## Run

Reproduce the web demo (the hiring scenario, seed 42, default sliders):

```bash
python scripts/run_synthetic.py
```

Try other settings:

```bash
python scripts/run_synthetic.py --scenario loan --seed 43 --bias 0.4 --target eo --lam 5
```

Real-data benchmark. Download the file first:

```bash
curl -L -o adult-all.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv
python scripts/run_adult.py --data adult-all.csv
```

The results are printed and saved to `results/adult_results.csv`.

Run the tests from the project root:

```bash
python -m pytest -q
```

## How the pieces map to the web page

| Page element | Python |
|---|---|
| Synthetic data and seed | `data.generate_synthetic` (same PRNG and draw order, so the same people) |
| Train 2,000 / test 1,000 split | `data.split_and_standardize` |
| Logistic regression with fairness penalty | `model.LogisticRegressionGD` |
| Metrics table | `metrics.fairness_report` |
| Trade-off curve | `metrics.trade_off_curve` |
| Reweighing | `mitigation.reweighing_weights` |
| Group thresholds with accuracy budget | `mitigation.find_group_thresholds` |
| The four table rows | `pipeline.run_synthetic` |
| Every control and the new-sample button | `POST /api/audit` in `app.py` |
| Adult table | `GET /api/adult` (reads `results/adult_results.csv`) |

## Method notes

- **Fairness penalty.** The training objective is the weighted logistic loss, plus a small L2
  term, plus λ·(mean prediction for group B − mean prediction for group A)². With
  `equal_opportunity=True` the means use only truly positive examples. This is a soft penalty:
  it trades fairness against accuracy but does not guarantee a constraint. fairlearn's
  ExponentiatedGradient enforces the constraint more directly, and the Adult benchmark compares them.
- **Reweighing.** Weights w(g, y) = P(g)·P(y) / P(g, y) (Kamiran and Calmon, 2012). Tested: after
  weighting, the outcome rate is equal across groups.
- **Group thresholds.** A grid search over pairs of thresholds, using training data only, followed
  by evaluation on test data (Hardt, Price and Srebro, 2016 style). The accuracy budget limits how
  much accuracy the thresholds may give up.

## Verification

- Metrics match fairlearn and scikit-learn exactly (`tests/test_metrics.py`).
- Coefficients match scikit-learn's logistic regression (`tests/test_model.py`).
- The generator matches the JavaScript values (`tests/test_data.py`).
- The pipeline reproduces the page's seed-42 numbers (`tests/test_pipeline.py`).
- The API returns exactly what the package computes, for all scenarios and both targets, rejects invalid
  input, and serves the Adult rows that match the CSV (`tests/test_api.py`).
- The same checks pass over a real HTTP socket (`tests/test_live_http.py`).
- Reweighing stays finite when a group/outcome cell is empty (`tests/test_edge_cases.py`).

## Limitations

- The synthetic data has three features and a known rule, so it is easier than real data.
- Results come from one train/test split and one seed. Repeat over many seeds and report the
  spread before drawing firm conclusions.
- Adult is 1994 census data predicting income over $50K. It is a benchmark, not a lending model.
- The sensitive attribute is binary (sex).

## Expected output

`python scripts/run_synthetic.py` (hiring, seed 42, default settings):

```
                                     acc     dp     eo  eodds     pp     di
Baseline                           0.894  0.111  0.130  0.130  0.010  0.598
Preprocessing: reweighing          0.891  0.086  0.095  0.095  0.051  0.675
In-processing: fairness penalty    0.884  0.065  0.094  0.094  0.118  0.727
Post-processing: group thresholds  0.889  0.048  0.216  0.216  0.190  0.824
```

`python scripts/run_adult.py --data adult-all.csv` (UCI Adult, 70/30 split, sex sensitive):

```
                                       acc     dp     eo  eodds     pp     di
Baseline                             0.845  0.174  0.076  0.076  0.001  0.302
Reweighing (ours)                    0.837  0.094  0.117  0.117  0.178  0.553
Penalty, DP (ours)                   0.829  0.043  0.230  0.230  0.238  0.749
Penalty, EO (ours)                   0.842  0.144  0.008  0.053  0.078  0.393
Group thresholds, DP (ours)          0.823  0.003  0.331  0.331  0.309  0.984
Group thresholds, EO (ours)          0.841  0.146  0.004  0.053  0.085  0.389
fairlearn ExponentiatedGradient, DP  0.824  0.020  0.265  0.265  0.289  0.878
fairlearn ExponentiatedGradient, EO  0.834  0.108  0.016  0.020  0.203  0.485
fairlearn ThresholdOptimizer, DP     0.822  0.002  0.328  0.328  0.318  0.989
fairlearn ThresholdOptimizer, EO     0.827  0.091  0.023  0.023  0.307  0.543
```

These come from a single split and a single seed, so treat small differences as noise.
