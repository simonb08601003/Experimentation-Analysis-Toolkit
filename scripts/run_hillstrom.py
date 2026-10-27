"""Analysis of the Hillstrom e-mail experiment (64k customers, 3 arms: Mens, Womens, no e-mail)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abkit import cuped_test, diff_in_means, holm, ratio_metric_delta, sample_size_two_proportions, srm_test  # noqa: E402
from abkit.uplift import TLearner, XLearner, qini_coefficient, qini_curve, share_of_uplift_captured  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
CONTROL = "No E-Mail"
ARMS = ["Mens E-Mail", "Womens E-Mail"]
METRICS = ["visit", "conversion", "spend"]
PRE = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]


def features(df):
    return pd.get_dummies(df[PRE], columns=["zip_code", "channel"], dtype=float)


def adjusted_tests(df, covariate):
    """CUPED-style test of every arm x metric, with covariate(metric) as the adjustment column."""
    c = df[df.segment == CONTROL]
    rows = []
    for arm in ARMS:
        t = df[df.segment == arm]
        for m in METRICS:
            raw = diff_in_means(c[m], t[m])
            x = covariate(m)
            res, info = cuped_test(c[m].values, t[m].values, c[x].values, t[x].values, metric=m)
            rows.append({"arm": arm, "metric": m, "se_raw": raw.se, "se_adj": res.se,
                         "variance_reduction": info["variance_reduction"],
                         "sample_size_saving": 1 - (res.se / raw.se) ** 2,
                         "abs_lift": res.abs_lift, "ci_low": res.ci_low, "ci_high": res.ci_high,
                         "p_value": res.p_value})
    return rows


def plot_qini(curves, path, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (f, q, rnd) in curves.items():
        ax.plot(f, q, label=name)
    ax.plot(f, rnd, "k--", label="Random targeting")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "hillstrom.csv.gz")

    con = duckdb.connect()
    con.register("users", df)
    for name in ["arm_metrics", "segment_lift", "history_deciles"]:
        out = "scorecard" if name == "arm_metrics" else name
        con.execute((ROOT / "sql" / f"{name}.sql").read_text()).df().to_csv(REPORTS / f"{out}.csv", index=False)

    srm = srm_test(df.segment.value_counts().to_dict())

    # 2 arms x 3 metrics, Holm across all six
    c = df[df.segment == CONTROL]
    tests = [{**diff_in_means(c[m], df.loc[df.segment == arm, m], metric=m).as_dict(), "arm": arm}
             for arm in ARMS for m in METRICS]
    for r, p in zip(tests, holm([r["p_value"] for r in tests])):
        r["p_holm"] = p

    # CUPED uses last year's spend; CUPAC uses a cross-fitted prediction from all pre-period features
    cuped = adjusted_tests(df, lambda m: "history")
    X = features(df)
    for m in METRICS:
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, min_child_samples=200, verbose=-1)
        df[f"{m}_pred"] = cross_val_predict(model, X, df[m], cv=5)
    cupac = adjusted_tests(df, lambda m: f"{m}_pred")

    ratio = [{**ratio_metric_delta(c.spend, c.visit, t.spend, t.visit, metric="revenue_per_visit").as_dict(), "arm": arm}
             for arm in ARMS for t in [df[df.segment == arm]]]

    base_conv = c.conversion.mean()
    power = {f"mde_{int(k * 100)}pct": sample_size_two_proportions(base_conv, k) for k in (0.05, 0.10, 0.25)}

    # Uplift: Mens e-mail vs control on visits, 50/50 split
    sub = df[df.segment.isin(["Mens E-Mail", CONTROL])]
    X, y = features(sub).values, sub.visit.values
    tr = (sub.segment == "Mens E-Mail").astype(int).values
    rng = np.random.default_rng(42)
    test = rng.random(len(sub)) < 0.5
    uplift, curves, preds = {}, {}, {}
    for name, model in {"T-learner": TLearner(seed=1), "X-learner": XLearner(seed=1)}.items():
        u = preds[name] = model.fit(X[~test], y[~test], tr[~test]).predict(X[test])
        uplift[name] = {"qini_coefficient": qini_coefficient(u, y[test], tr[test]),
                        "share_captured_top30": share_of_uplift_captured(u, y[test], tr[test], 0.3)}
        curves[name] = qini_curve(u, y[test], tr[test])
    u, yt, tt = preds["T-learner"], y[test], tr[test]
    boots = [qini_coefficient(u[i], yt[i], tt[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(300))]
    uplift["T-learner"]["qini_ci95"] = np.percentile(boots, [2.5, 97.5]).tolist()

    results = {"n_users": len(df), "srm": srm, "tests": tests, "cuped": cuped, "cupac": cupac,
               "ratio_metrics": ratio, "baseline_conversion": base_conv, "sample_size_per_arm": power,
               "uplift": uplift}
    (REPORTS / "hillstrom_results.json").write_text(json.dumps(results, indent=2, default=float))

    plot_qini(curves, FIG / "qini_curve.png", "Fraction of customers targeted (ranked by predicted uplift)",
              "Incremental visits (test set)", "Qini curve: Mens e-mail vs no e-mail")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(cuped))
    ax.bar(x - 0.27, 1.0, 0.27, label="Unadjusted")
    ax.bar(x, [r["se_adj"] / r["se_raw"] for r in cuped], 0.27, label="CUPED (pre-period spend)")
    ax.bar(x + 0.27, [r["se_adj"] / r["se_raw"] for r in cupac], 0.27, label="CUPAC (ML on pre-period features)")
    ax.set_ylim(0.95, 1.005)
    ax.set_xticks(x, [f"{r['arm'].split()[0]}\n{r['metric']}" for r in cuped])
    ax.set(ylabel="Standard error (relative to unadjusted)", title="Variance reduction vs unadjusted test")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "cuped_se.png", dpi=150)

    print(json.dumps({"srm_p": srm["p_value"],
                      "tests": [(r["arm"], r["metric"], round(r["rel_lift"], 3), r["p_holm"]) for r in tests],
                      "uplift": uplift, "power": power}, indent=1, default=float))


if __name__ == "__main__":
    main()
