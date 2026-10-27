import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abkit import cuped_adjust, diff_in_means, holm, ratio_metric_delta, sample_size_two_proportions, srm_test
from abkit.sequential import simulate_peeking
from abkit.uplift import qini_coefficient


def test_srm_flags_imbalance():
    assert srm_test({"a": 5000, "b": 5000})["p_value"] > 0.9
    assert srm_test({"a": 5000, "b": 5400})["p_value"] < 1e-3


def test_diff_in_means_ci_covers_truth():
    rng = np.random.default_rng(0)
    covered = 0
    for _ in range(300):
        r = diff_in_means(rng.normal(0, 1, 500), rng.normal(0.2, 1, 500))
        covered += r.ci_low <= 0.2 <= r.ci_high
    assert 0.92 < covered / 300 < 0.98


def test_cuped_reduces_variance_with_correlated_covariate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=10_000)
    y = 2 * x + rng.normal(size=10_000)
    y_adj, theta = cuped_adjust(y, x)
    assert abs(theta - 2) < 0.05
    assert y_adj.var() < 0.25 * y.var()


def test_delta_method_matches_bootstrap():
    rng = np.random.default_rng(2)
    den_c, den_t = rng.poisson(3, 4000) + 1, rng.poisson(3, 4000) + 1
    num_c, num_t = rng.gamma(2, den_c), rng.gamma(2, den_t)
    r = ratio_metric_delta(num_c, den_c, num_t, den_t)
    boots = []
    for _ in range(400):
        i, j = rng.integers(0, 4000, 4000), rng.integers(0, 4000, 4000)
        boots.append(num_t[j].sum() / den_t[j].sum() - num_c[i].sum() / den_c[i].sum())
    assert abs(np.std(boots) / r.se - 1) < 0.15


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    assert all(0 <= a <= 1 for a in adj)
    assert abs(adj[0] - 0.04) < 1e-12


def test_sample_size_textbook_value():
    # 10% -> 12% (20% relative lift), alpha .05, power .8: about 3,841 per arm.
    assert abs(sample_size_two_proportions(0.10, 0.20) - 3841) < 20


def test_msprt_controls_type_one_error():
    r = simulate_peeking(p_base=0.1, lift_rel=0.0, n_per_arm=5000, n_looks=10, n_sims=400, seed=3)
    assert r["reject_msprt"] <= 0.06
    assert r["reject_naive_peeking"] > r["reject_fixed_horizon"]


def test_qini_perfect_ranking_beats_random():
    rng = np.random.default_rng(4)
    n = 20_000
    tau = rng.uniform(0, 0.3, n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.1 + tau * t).astype(float)
    assert qini_coefficient(tau, y, t) > 0
    assert qini_coefficient(tau, y, t) > qini_coefficient(rng.random(n), y, t)
