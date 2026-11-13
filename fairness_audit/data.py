"""Data: the synthetic scenario generator used by the web demo, and the UCI Adult loader.

The generator reproduces the browser code exactly: the same 32-bit Mulberry32 random
generator and the same Box-Muller normal draws, consumed in the same order. A given
seed therefore produces the same people as the page does.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MASK32 = 0xFFFFFFFF

SCENARIOS = {
    "hiring": {"name": "Hiring", "p_group_b": 0.40, "cut": 0.6, "positive": "qualified",
               "features": "skill score, years of experience and a referral network"},
    "loan": {"name": "Loan approval", "p_group_b": 0.35, "cut": 0.2, "positive": "able to repay",
             "features": "credit history, income stability and a postcode index"},
    "admission": {"name": "University admission", "p_group_b": 0.45, "cut": 0.8, "positive": "likely to succeed",
                  "features": "entrance score, extracurriculars and school quality"},
    "risk": {"name": "Risk scoring", "p_group_b": 0.30, "cut": -0.2, "positive": "low risk",
             "features": "claims history, vehicle age and a neighborhood index"},
}

ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income"]


class Mulberry32:
    """The 32-bit Mulberry32 generator, bit-for-bit the same as the JavaScript in the page."""

    def __init__(self, seed: int):
        self.state = seed & MASK32

    def random(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & MASK32
        a = self.state
        t = ((a ^ (a >> 15)) * (1 | a)) & MASK32                      # Math.imul(a ^ a>>>15, 1 | a)
        t2 = ((t ^ (t >> 7)) * (61 | t)) & MASK32                     # Math.imul(t ^ t>>>7, 61 | t)
        t = ((t + t2) & MASK32) ^ t
        return ((t ^ (t >> 14)) & MASK32) / 4294967296.0


def generate_synthetic(scenario: str = "hiring", seed: int = 42, bias: float = 0.6, n: int = 3000):
    """Create n synthetic people.

    Each person has a protected group g (0 = group A, 1 = group B). Group B's underlying merit is
    shifted down by `bias` standard deviations (historical disadvantage), so base rates differ.
    The model sees three features: an observed merit score (noisy), a second independent
    feature, and a proxy that partly reveals group membership. The group itself is never a feature.

    Returns X (n x 3 array), y (0/1 outcome) and g (0/1 group).
    """
    sc = SCENARIOS[scenario]
    rng = Mulberry32(seed)

    def gauss() -> float:
        u = v = 0.0
        while u == 0.0:
            u = rng.random()
        while v == 0.0:
            v = rng.random()
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)

    X = np.empty((n, 3))
    y = np.empty(n, dtype=int)
    g = np.empty(n, dtype=int)
    for i in range(n):
        gi = 1 if rng.random() < sc["p_group_b"] else 0
        x1 = gauss() - bias * gi                  # true merit, shifted for group B
        x2 = gauss()
        q = x1 + 0.5 * x2 + 0.5 * gauss()         # latent qualification
        X[i] = (x1 + 0.3 * gauss(), x2, gi + 0.7 * gauss())   # observed features (noisy merit, proxy)
        y[i] = 1 if q > sc["cut"] else 0
        g[i] = gi
    return X, y, g


def split_and_standardize(X, y, g, n_train: int = 2000):
    """First n_train people for training, the rest for testing. Scaling uses training statistics only."""
    Xtr, Xte = X[:n_train], X[n_train:]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    return ((Xtr - mu) / sd, (Xte - mu) / sd,
            y[:n_train], y[n_train:], g[:n_train], g[n_train:])


def load_adult(path: str):
    """Load the UCI Adult income file (headerless CSV). Returns X (one-hot features, sex removed),
    y (1 = income over 50K) and sex (1 = male, 0 = female). Rows with '?' are dropped."""
    df = pd.read_csv(path, names=ADULT_COLUMNS, skipinitialspace=True)
    df = df[(df == "?").sum(axis=1) == 0].reset_index(drop=True)
    y = (df["income"].str.strip() == ">50K").astype(int).to_numpy()
    sex = (df["sex"].str.strip() == "Male").astype(int).to_numpy()
    X = pd.get_dummies(df.drop(columns=["income", "sex"]), dtype=float).to_numpy()
    return X, y, sex


def split_adult(X, y, sex, test_size: float = 0.3, seed: int = 0):
    """Stratified 70/30 split, then standardize with training statistics."""
    Xtr, Xte, ytr, yte, s_tr, s_te = train_test_split(
        X, y, sex, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte, s_tr, s_te
