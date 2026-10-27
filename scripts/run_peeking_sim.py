"""How much does peeking inflate false positives? A/A plus two real lifts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abkit.sequential import simulate_peeking  # noqa: E402

if __name__ == "__main__":
    # Hillstrom control visit rate, 20k users per arm, 20 looks
    common = dict(p_base=0.106, n_per_arm=20_000, n_looks=20, n_sims=2000)
    rows = [simulate_peeking(lift_rel=lift, seed=seed, **common) for lift, seed in [(0.0, 1), (0.05, 2), (0.10, 3)]]
    (ROOT / "reports" / "peeking_simulation.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)
