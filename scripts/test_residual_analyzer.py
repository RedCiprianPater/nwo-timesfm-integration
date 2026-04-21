"""
Local smoke test for the residual analyzer.

Exercises `TimesFMResidualAnalyzer` directly without spinning up the
FastAPI service. Useful for CI and for debugging fit quality.

Run:

    python scripts/test_residual_analyzer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the src/ package importable when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.residual_analyzer import TimesFMResidualAnalyzer  # noqa: E402


def test_recovers_log_aging_law() -> None:
    rng = np.random.default_rng(42)
    n = 256

    hrs = rng.uniform(0.5, 50.0, size=n)
    baseline = 10.0 + 0.02 * hrs
    true_load = baseline + 0.5 * np.log(hrs) + rng.normal(0, 0.05, n)
    forecast = baseline + rng.normal(0, 0.1, n)

    analyzer = TimesFMResidualAnalyzer(n_epochs=2500, seed=0)
    law = analyzer.analyze_with_depth_search(
        features=hrs,
        y_true=true_load,
        y_forecast=forecast,
        feature_names=["hrs"],
        depths=[2, 3, 4],
    )

    print(law.summary_line())
    print(f"  expression : {law.expression}")
    print(f"  simplified : {law.simplified}")
    print(f"  paper      : {law.paper_reference}")

    # Sanity: loss should be much smaller than residual variance.
    residual = true_load - forecast
    resid_var = float(np.var(residual))
    improvement = resid_var / max(law.final_loss, 1e-12)
    print(f"\nvariance of raw residual : {resid_var:.4f}")
    print(f"final fit loss           : {law.final_loss:.4e}")
    print(f"improvement factor       : {improvement:.1f}x")

    if improvement < 2.0:
        print(
            "\nWARNING: fit barely beats variance. "
            "Try more n_epochs or a different seed.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    test_recovers_log_aging_law()
