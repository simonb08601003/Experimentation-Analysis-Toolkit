"""Uplift models on the MegaFon dataset (600k users, randomised treatment), scored on a 30% hold-out."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from abkit import diff_in_means, srm_test  # noqa: E402
from abkit.uplift import TLearner, XLearner, qini_coefficient, qini_curve, share_of_uplift_captured  # noqa: E402
from run_hillstrom import plot_qini  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
REPORTS = ROOT / "reports"


def main():
    df = pd.read_csv(DATA / "megafon.csv.gz")
    X, y = df.filter(like="X_").values, df.conversion.values
    t = (df.treatment_group == "treatment").astype(int).values

    rng = np.random.default_rng(0)
    test = rng.random(len(df)) < 0.3
    yt, tt = y[test], t[test]
    out = {"n_users": len(df),
           "srm": srm_test({"control": int((t == 0).sum()), "treatment": int((t == 1).sum())}),
           "ate": diff_in_means(y[t == 0], y[t == 1], metric="conversion").as_dict(),
           "models": {}}
    curves = {}
    for name, model in {"T-learner": TLearner(seed=1), "X-learner": XLearner(seed=1)}.items():
        u = model.fit(X[~test], y[~test], t[~test]).predict(X[test])
        boots = [qini_coefficient(u[i], yt[i], tt[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(200))]
        # does the top 30% actually respond more? check on held-out users
        top = u >= np.quantile(u, 0.7)
        effect = lambda m: diff_in_means(yt[m & (tt == 0)], yt[m & (tt == 1)]).abs_lift  # noqa: E731
        out["models"][name] = {
            "qini_coefficient": qini_coefficient(u, yt, tt),
            "qini_ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "share_of_total_uplift_top30": share_of_uplift_captured(u, yt, tt, 0.3),
            "observed_effect_top30": effect(top),
            "observed_effect_bottom70": effect(~top),
        }
        curves[name] = qini_curve(u, yt, tt)
    (REPORTS / "megafon_uplift_results.json").write_text(json.dumps(out, indent=2, default=float))
    plot_qini(curves, REPORTS / "figures" / "megafon_qini.png", "Fraction of users targeted (ranked by predicted uplift)",
              "Incremental conversions (hold-out)", "Qini curve: MegaFon uplift dataset")
    print(json.dumps(out["models"], indent=1, default=float))


if __name__ == "__main__":
    main()
