"""Peeking vs. mSPRT, the always-valid alternative."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _running_stats(y_c, y_t, n):
    """Difference in means and its variance after the first n users of each arm."""
    def mean_var(y):
        m = np.cumsum(y)[n - 1] / n
        return m, np.maximum(np.cumsum(y**2)[n - 1] / n - m**2, 1e-12)
    m_c, v_c = mean_var(y_c)
    m_t, v_t = mean_var(y_t)
    return m_t - m_c, (v_c + v_t) / n


def naive_peeking_p(y_c, y_t, checkpoints) -> np.ndarray:
    """A plain z-test re-run at every checkpoint."""
    d, v = _running_stats(y_c, y_t, checkpoints)
    return 2 * stats.norm.sf(np.abs(d / np.sqrt(v)))


def msprt_always_valid_p(y_c, y_t, checkpoints, tau2: float) -> np.ndarray:
    """Normal-mixture SPRT p-values (Johari, Pekelis & Walsh 2017).

    You can stop the first time p < alpha and type I error stays below alpha.
    """
    d, v = _running_stats(y_c, y_t, checkpoints)
    log_lam = 0.5 * np.log(v / (v + tau2)) + tau2 * d**2 / (2 * v * (v + tau2))
    return np.minimum.accumulate(np.minimum(1.0, np.exp(-log_lam)))


def simulate_peeking(p_base: float, lift_rel: float, n_per_arm: int, n_looks: int, n_sims: int,
                     alpha: float = 0.05, tau2: float | None = None, seed: int = 0) -> dict:
    """Rejection rates for one final look, naive peeking, and mSPRT."""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).astype(int)
    tau2 = tau2 or (0.1 * p_base) ** 2      # prior effect size around 10% of baseline
    p_t = p_base * (1 + lift_rel)
    final = peek = seq = 0
    for _ in range(n_sims):
        y_c = (rng.random(n_per_arm) < p_base).astype(float)
        y_t = (rng.random(n_per_arm) < p_t).astype(float)
        p = naive_peeking_p(y_c, y_t, checkpoints)
        final += p[-1] < alpha
        peek += (p < alpha).any()
        seq += (msprt_always_valid_p(y_c, y_t, checkpoints, tau2) < alpha).any()
    return {"p_base": p_base, "lift_rel": lift_rel, "n_per_arm": n_per_arm, "n_looks": n_looks, "n_sims": n_sims,
            "reject_fixed_horizon": final / n_sims,
            "reject_naive_peeking": peek / n_sims,
            "reject_msprt": seq / n_sims}
