"""
End-to-end example: recover a symbolic law from TimesFM residuals.

Constructs a "true" load signal of the form

    load(hrs) = baseline(hrs) + 0.5 * log(hrs) + noise

and a stand-in TimesFM forecast that only captures `baseline`. Hits the
`/api/timesfm/residual-analysis` endpoint and prints the recovered law.

Run the service first:

    python src/server.py

Then, in another shell:

    python examples/residual_law_discovery.py

Reference: Odrzywołek, arXiv:2603.21852 (2026).
"""
from __future__ import annotations

import json
import os
import sys
from urllib import request as urlreq

import numpy as np


DEFAULT_ENDPOINT = os.environ.get(
    "NWO_TIMESFM_ENDPOINT",
    "http://localhost:8000/api/timesfm/residual-analysis",
)


def synthesise_data(n: int = 256, seed: int = 42) -> dict:
    """Build a residual-analysis request with a known log-aging bias."""
    rng = np.random.default_rng(seed)

    operating_hours = rng.uniform(0.5, 50.0, size=n)
    baseline = 10.0 + 0.02 * operating_hours
    aging_term = 0.5 * np.log(operating_hours)

    true_load = baseline + aging_term + rng.normal(0, 0.05, n)
    timesfm_forecast = baseline + rng.normal(0, 0.1, n)

    return {
        "features":      [[h] for h in operating_hours.tolist()],
        "y_true":        true_load.tolist(),
        "y_forecast":    timesfm_forecast.tolist(),
        "feature_names": ["hrs"],
        "n_epochs":      2500,
        "seed":          0,
    }


def call_endpoint(payload: dict, url: str = DEFAULT_ENDPOINT) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlreq.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    payload = synthesise_data()
    print(f"POST {DEFAULT_ENDPOINT}")
    print(f"  samples           : {len(payload['y_true'])}")
    print(f"  ground-truth law  : residual = 0.5 * log(hrs) + small noise")
    print(f"  epochs            : {payload['n_epochs']}")
    print()

    try:
        result = call_endpoint(payload)
    except Exception as e:  # noqa: BLE001
        print(f"request failed: {e}", file=sys.stderr)
        print("is the service running?  python src/server.py", file=sys.stderr)
        return 1

    print("=== recovered law ===")
    print(f"summary        : {result['summary']}")
    print(f"simplified     : {result['simplified']}")
    print(f"raw eml form   : {result['expression']}")
    print(f"final loss     : {result['final_loss']:.4e}")
    print(f"tree size      : {result['tree_size']}")
    print(f"depth used     : {result['depth_used']}")
    print(f"paper ref      : {result['paper_reference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
