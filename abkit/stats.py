"""Basic A/B test statistics."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    control_mean: float
    treatment_mean: float
    abs_lift: float
    rel_lift: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float

    def as_dict(self) -> dict:
        return asdict(self)


def _result(metric, m_c, m_t, se, alpha):
    diff = m_t - m_c
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(diff / se))
    return TestResult(metric, m_c, m_t, diff, diff / m_c if m_c else np.nan, se, diff - z * se, diff + z * se, p)


def srm_test(counts: dict[str, int], expected_ratio: dict[str, float] | None = None) -> dict:
    """Chi-square test for sample ratio mismatch. Run this before looking at any result."""
    arms = list(counts)
    obs = np.array([counts[a] for a in arms], float)
    exp = np.array([expected_ratio[a] for a in arms], float) if expected_ratio else np.ones(len(arms))
    chi2, p = stats.chisquare(obs, obs.sum() * exp / exp.sum())
    return {"arms": arms, "observed": obs.astype(int).tolist(), "chi2": float(chi2), "p_value": float(p)}


def diff_in_means(y_c, y_t, metric: str = "metric", alpha: float = 0.05) -> TestResult:
    """Welch z-test. Fine for both 0/1 and continuous metrics at A/B sample sizes."""
    y_c, y_t = np.asarray(y_c, float), np.asarray(y_t, float)
    se = np.sqrt(y_c.var(ddof=1) / len(y_c) + y_t.var(ddof=1) / len(y_t))
    return _result(metric, y_c.mean(), y_t.mean(), se, alpha)


def cuped_adjust(y, x) -> tuple[np.ndarray, float]:
    """CUPED (Deng et al. 2013). x has to be measured before treatment."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def cuped_test(y_c, y_t, x_c, x_t, metric: str = "metric", alpha: float = 0.05) -> tuple[TestResult, dict]:
    # theta comes from both arms pooled, so each arm gets the same adjustment
    y = np.concatenate([y_c, y_t])
    y_adj, theta = cuped_adjust(y, np.concatenate([x_c, x_t]))
    n_c = len(y_c)
    res = diff_in_means(y_adj[:n_c], y_adj[n_c:], metric=f"{metric} (CUPED)", alpha=alpha)
    # keep the raw means so the lift still reads in the metric's own units
    res.control_mean, res.treatment_mean = float(np.mean(y_c)), float(np.mean(y_t))
    res.rel_lift = res.abs_lift / res.control_mean if res.control_mean else np.nan
    return res, {"theta": theta, "variance_reduction": float(1 - y_adj.var(ddof=1) / y.var(ddof=1))}


def _ratio_and_var(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    mx, my = num.mean(), den.mean()
    cxy = np.cov(num, den, ddof=1)[0, 1]
    var = (num.var(ddof=1) / my**2 - 2 * mx * cxy / my**3 + mx**2 * den.var(ddof=1) / my**4) / len(num)
    return mx / my, var


def ratio_metric_delta(num_c, den_c, num_t, den_t, metric: str = "ratio", alpha: float = 0.05) -> TestResult:
    """Delta method for sum(num) / sum(den), e.g. revenue per visit when users are randomised."""
    r_c, v_c = _ratio_and_var(num_c, den_c)
    r_t, v_t = _ratio_and_var(num_t, den_t)
    return _result(metric, r_c, r_t, np.sqrt(v_c + v_t), alpha)


def holm(p_values) -> list[float]:
    p = np.asarray(p_values, float)
    m = len(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def sample_size_two_proportions(p_base: float, mde_rel: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per arm needed to detect a relative lift of mde_rel."""
    p1, p2 = p_base, p_base * (1 + mde_rel)
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def sample_size_means(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (z_a + z_b) ** 2 * sd**2 / mde_abs**2))
