"""Fairness audit and bias mitigation toolkit (Python backend)."""
from .data import (SCENARIOS, Mulberry32, generate_synthetic, split_and_standardize,
                   load_adult, split_adult)
from .model import LogisticRegressionGD
from .metrics import fairness_report, group_stats, trade_off_curve
from .mitigation import reweighing_weights, find_group_thresholds
from .pipeline import run_synthetic

__all__ = [
    "SCENARIOS", "Mulberry32", "generate_synthetic", "split_and_standardize",
    "load_adult", "split_adult", "LogisticRegressionGD", "fairness_report",
    "group_stats", "trade_off_curve", "reweighing_weights", "find_group_thresholds",
    "run_synthetic",
]
