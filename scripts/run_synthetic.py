"""Run the synthetic-data experiment from the command line.

    python scripts/run_synthetic.py                       # same defaults as the web demo
    python scripts/run_synthetic.py --scenario loan --seed 43 --target eo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fairness_audit import SCENARIOS, run_synthetic  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Fairness audit on synthetic data.")
    ap.add_argument("--scenario", default="hiring", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bias", type=float, default=0.6, help="historical disadvantage (0 to 1)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=3.0, help="fairness penalty strength")
    ap.add_argument("--target", choices=["dp", "eo"], default="dp")
    ap.add_argument("--budget", type=float, default=0.02, help="accuracy budget for group thresholds")
    a = ap.parse_args()

    res = run_synthetic(a.scenario, seed=a.seed, bias=a.bias, threshold=a.threshold,
                        fair_lambda=a.lam, target=a.target, budget=a.budget)
    print(f"Scenario: {SCENARIOS[a.scenario]['name']}   seed: {a.seed}   bias: {a.bias}   target: {a.target}")
    print(res["table"].round(3).to_string())
    ta, tb = res["thresholds"]
    print(f"\nGroup thresholds chosen on training data: A = {ta:.2f}, B = {tb:.2f}")


if __name__ == "__main__":
    main()
