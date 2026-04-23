# TimesFM Integration for NWO Robotics API

Time Series Foundation Model (TimesFM) powered forecasting for robot maintenance, task prediction, and anomaly detection — plus a symbolic residual analyzer that extracts closed-form elementary laws from forecast errors using the EML operator from [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).

---

## Status

**Live at** [https://nwo-timesfm.onrender.com](https://nwo-timesfm.onrender.com)

The `/api/timesfm/residual-analysis` endpoint (EML symbolic regression) is the production-grade path and is actively used by downstream services. The direct TimesFM forecasting endpoints currently return mock predictions while the model-load path is being rebuilt — see [Known issues](#known-issues) for details.

---

## Overview

This integration adds two capabilities to the NWO Robotics stack:

1. **Time-series forecasting** via Google's TimesFM 2.5 model (200M parameters).
2. **Symbolic residual analysis** that takes the gap between predictions and reality and finds a closed-form equation describing it — via the EML operator `eml(x, y) = exp(x) − ln(y)`, from which every elementary function can be built (Odrzywołek 2026).

---

## Features

- **Robot Maintenance Prediction** — forecast battery failure, motor degradation
- **Task Duration Forecasting** — estimate task completion times
- **Sensor Anomaly Detection** — predict failures before they happen
- **Swarm Load Forecasting** — scale robot fleets intelligently
- **Residual Symbolic Analysis** — recover closed-form elementary expressions from TimesFM forecast residuals using the EML operator

---

## API Endpoints

| Endpoint | Description | Status |
|---|---|---|
| `POST /api/timesfm/maintenance` | Predict robot maintenance needs | Mock fallback |
| `POST /api/timesfm/task-duration` | Forecast task completion times | Mock fallback |
| `POST /api/timesfm/anomaly` | Detect sensor anomalies | Mock fallback |
| `POST /api/timesfm/swarm-load` | Predict swarm resource needs | Mock fallback |
| **`POST /api/timesfm/residual-analysis`** | **Recover closed-form laws from forecast residuals (EML)** | **Production** |
| `GET /api/timesfm/health` | Check service health | Production |

---

## Quick Start

### Using the live service

No install needed — the service is deployed. You can hit it directly:

```bash
curl https://nwo-timesfm.onrender.com/api/timesfm/health
```

### Running locally

```bash
git clone https://github.com/RedCiprianPater/nwo-timesfm-integration.git
cd nwo-timesfm-integration

pip install -r requirements.txt

# Download TimesFM model (optional — endpoints use mocks if unavailable)
python scripts/download_model.py

# Start the service
python src/server.py
```

### Production deployment (Docker + Gunicorn)

The Render deployment uses:

```bash
gunicorn --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT src.server:app
```

Single worker is intentional — TimesFM model memory footprint is too large for multiple workers on Render's free/starter tiers. Threads handle concurrent requests.

See `Dockerfile` in the repo root for the full build.

---

## Usage Example — Maintenance Prediction

```python
import requests

response = requests.post(
    "https://nwo-timesfm.onrender.com/api/timesfm/maintenance",
    json={
        "robot_id": "unitree_go2_001",
        "metric": "battery_drain",
        "history": [98, 95, 92, 88, 85, 82, 78, 75, 71, 68],
        "horizon": 24
    }
)

print(response.json())
```

**Note:** currently returns mock predictions until the TimesFM 2.5 load path is fixed. See [Known issues](#known-issues).

---

## Usage Example — Residual Symbolic Analysis

This is the production endpoint. It takes observed values, forecasted values, and the features those values depend on, then finds a closed-form expression for the residual.

```python
import requests

response = requests.post(
    "https://nwo-timesfm.onrender.com/api/timesfm/residual-analysis",
    json={
        "features":      [[h] for h in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]],
        "y_true":        [10.0, 10.1, 10.3, 10.8, 11.2, 11.7, 12.3],
        "y_forecast":    [9.9, 10.0, 10.2, 10.7, 11.0, 11.4, 12.0],
        "feature_names": ["hrs"],
        "n_epochs":      2000
    }
)

print(response.json()["summary"])
# e.g. "residual ≈ 0.5*log(hrs)  (loss=3.2e-04, size=7, depth=3)"
```

See `docs/RESIDUAL_ANALYSIS.md` for the full request/response schema and tuning guidance.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   NWO Robotics  │────▶│  TimesFM API     │────▶│  TimesFM Model  │
│   API Gateway   │     │  (this repo)     │     │  (200M params)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ├──▶ Robot Telemetry DB
                               │
                               └──▶ EML Residual Analyzer
                                    (arXiv:2603.21852)
                                        │
                                        ▼
                                    Closed-form elementary law
                                    (log, exp, polynomial, trig…)
```

---

## Used by

**[NWO Agent Graph](https://github.com/RedCiprianPater/nwo-agent-graph)** — [live on Hugging Face](https://huggingface.co/spaces/CPater/nwo-agent-graph)

Robot telemetry (battery level, joint angles, reward signals) flows through a 32-sample circular buffer in the Space. When a buffer fills, the Space sends the series to this service's `/api/timesfm/residual-analysis` endpoint. The returned symbolic expression (e.g. `exp(0.02·t) − ln(capacity)`) is cached in memory and attached to future telemetry nodes as a first-class `symbolic_law` attribute on the graph node. When the downstream LLM expands a telemetry node, it sees the law as context and can reason about the underlying dynamics rather than just the surface text.

**Integration code:**

- [`eml_client.py`](https://github.com/RedCiprianPater/nwo-agent-graph/blob/main/eml_client.py) — HTTP client for this service
- [`telemetry_buffer.py`](https://github.com/RedCiprianPater/nwo-agent-graph/blob/main/telemetry_buffer.py) — Per-(robot, metric) circular buffers + fit cache
- [`agent_loop.py`](https://github.com/RedCiprianPater/nwo-agent-graph/blob/main/agent_loop.py) — See `_fit_symbolic_law` for the full end-to-end path

---

## Known issues

### TimesFM 2.5 model load failure

On startup, the TimesFM 2.5 model load fails with:

```
TimesFmBase.__init__() got an unexpected keyword argument 'context_len'
```

This is a version-mismatch issue between the `timesfm` pip package and the checkpoint format. The service handles this gracefully: the TimesFM-dependent endpoints (`/maintenance`, `/task-duration`, `/anomaly`, `/swarm-load`) fall back to mock predictions, and the service continues to serve the `/residual-analysis` endpoint, which doesn't require TimesFM.

**Impact:** none for downstream consumers that use residual analysis directly (like NWO Agent Graph). Full impact for anyone calling the direct forecasting endpoints — they return structured mock data, not real predictions.

**Fix path:** pin `timesfm` to a known-working version or re-plumb the checkpoint loading. Tracked as an open issue.

### Gunicorn logs `flask_limiter` in-memory storage warning

Harmless — in-memory rate limiting is intentional for single-worker deploys. Would need to swap to Redis for multi-instance setups.

---

## Documentation

- API Reference — `docs/API.md`
- Deployment Guide — `docs/DEPLOYMENT.md`
- Architecture — `docs/ARCHITECTURE.md`
- Residual Symbolic Analysis — `docs/RESIDUAL_ANALYSIS.md`

---

## Citation

The residual symbolic analyzer implements the EML operator from:

> Odrzywołek, A. *All elementary functions from a single binary operator.* [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (2026).

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywołek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}
```

---

## Related repositories

- **[nwo-eml-regression](https://github.com/RedCiprianPater/nwo-eml-regression)** — Standalone Python package for EML symbolic regression (what this service wraps)
- **[nwo-agent-graph](https://github.com/RedCiprianPater/nwo-agent-graph)** — Production consumer; knowledge graph where robot telemetry nodes carry their symbolic laws
- **[mcp-server-robotics](https://github.com/RedCiprianPater/mcp-server-robotics)** — Claude / ChatGPT MCP for robot control

---

## License

MIT License — see `LICENSE`

---

## Support

- Email: ciprian.pater@publicae.org
- Issues: [GitHub Issues](https://github.com/RedCiprianPater/nwo-timesfm-integration/issues)
