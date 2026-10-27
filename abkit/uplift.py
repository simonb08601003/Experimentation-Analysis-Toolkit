"""T- and X-learner uplift models, scored with Qini curves."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

LGBM = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


class TLearner:
    """One model per arm; uplift is the difference in predicted probability."""

    def __init__(self, seed: int = 0):
        self.m0 = LGBMClassifier(random_state=seed, **LGBM)
        self.m1 = LGBMClassifier(random_state=seed + 1, **LGBM)

    def fit(self, X, y, t):
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    """Künzel et al. (2019)."""

    def __init__(self, seed: int = 0):
        self.t = TLearner(seed)
        self.g0 = LGBMRegressor(random_state=seed + 2, **LGBM)
        self.g1 = LGBMRegressor(random_state=seed + 3, **LGBM)

    def fit(self, X, y, t):
        self.t.fit(X, y, t)
        mu0, mu1 = self.t.m0.predict_proba(X)[:, 1], self.t.m1.predict_proba(X)[:, 1]
        self.g1.fit(X[t == 1], y[t == 1] - mu0[t == 1])
        self.g0.fit(X[t == 0], mu1[t == 0] - y[t == 0])
        self.prop = float(t.mean())     # randomised, so the propensity is a constant
        return self

    def predict(self, X):
        return self.prop * self.g0.predict(X) + (1 - self.prop) * self.g1.predict(X)


def qini_curve(uplift, y, t, n_bins: int = 100):
    """Incremental outcomes from targeting the top k% by predicted uplift.

    Returns (fraction targeted, qini, random-targeting line).
    """
    order = np.argsort(-uplift, kind="stable")
    y, t = y[order], t[order]
    cut = np.unique(np.linspace(0, len(y), n_bins + 1).astype(int))
    cum = lambda a: np.concatenate([[0], np.cumsum(a)])[cut]  # noqa: E731
    y_t, y_c, n_t, n_c = cum(y * t), cum(y * (1 - t)), cum(t), cum(1 - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    frac = cut / len(y)
    return frac, q, frac * q[-1]


def qini_coefficient(uplift, y, t, n_bins: int = 100) -> float:
    frac, q, rnd = qini_curve(uplift, y, t, n_bins)
    return float(np.trapezoid(q - rnd, frac) / len(y))


def share_of_uplift_captured(uplift, y, t, top_frac: float) -> float:
    """Share of all incremental outcomes you get by treating only the top `top_frac`."""
    frac, q, _ = qini_curve(uplift, y, t, n_bins=1000)
    return float(np.interp(top_frac, frac, q) / q[-1])
