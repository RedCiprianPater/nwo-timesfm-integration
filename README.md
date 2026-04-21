# TimesFM Integration for NWO Robotics API

Time Series Foundation Model (TimesFM) powered forecasting for robot maintenance, task prediction, and anomaly detection.

## Overview

This integration adds time-series forecasting capabilities to NWO Robotics using Google's TimesFM 2.5 model (200M parameters). It also includes a **symbolic residual analyzer** that extracts closed-form elementary laws from forecast errors, based on the EML operator from [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).

## Features

* **Robot Maintenance Prediction** - Predict battery failure, motor degradation
* **Task Duration Forecasting** - Estimate task completion times
* **Sensor Anomaly Detection** - Predict failures before they happen
* **Swarm Load Forecasting** - Scale robot fleets intelligently
* **Residual Symbolic Analysis** - Recover closed-form elementary expressions from TimesFM forecast residuals using the EML operator ([arXiv:2603.21852](https://arxiv.org/abs/2603.21852))

## API Endpoints

| Endpoint | Description |
| --- | --- |
| `POST /api/timesfm/maintenance` | Predict robot maintenance needs |
| `POST /api/timesfm/task-duration` | Forecast task completion times |
| `POST /api/timesfm/anomaly` | Detect sensor anomalies |
| `POST /api/timesfm/swarm-load` | Predict swarm resource needs |
| `POST /api/timesfm/residual-analysis` | Recover closed-form laws from forecast residuals (EML, arXiv:2603.21852) |
| `GET /api/timesfm/health` | Check service health |

## Quick Start

### Installation

```
# Clone the repo
git clone https://github.com/RedCiprianPater/nwo-timesfm-integration.git
cd nwo-timesfm-integration

# Install dependencies
pip install -r requirements.txt

# Download TimesFM model
python scripts/download_model.py

# Start the service
python src/server.py
```

### Usage Example — Maintenance Prediction

```python
import requests

# Predict robot maintenance
response = requests.post("http://localhost:8000/api/timesfm/maintenance", json={
    "robot_id": "unitree_go2_001",
    "metric": "battery_drain",
    "history": [98, 95, 92, 88, 85, 82, 78, 75, 71, 68],
    "horizon": 24
})

print(response.json())
```

### Usage Example — Residual Symbolic Analysis

```python
import requests

# Extract a closed-form law from TimesFM forecast residuals
response = requests.post("http://localhost:8000/api/timesfm/residual-analysis", json={
    "features":      [[h] for h in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]],
    "y_true":        [10.0, 10.1, 10.3, 10.8, 11.2, 11.7, 12.3],
    "y_forecast":    [9.9, 10.0, 10.2, 10.7, 11.0, 11.4, 12.0],
    "feature_names": ["hrs"],
    "n_epochs":      2000
})

print(response.json()["summary"])
# e.g. "residual ≈ 0.5*log(hrs)  (loss=3.2e-04, size=7, depth=3)"
```

See [docs/RESIDUAL_ANALYSIS.md](docs/RESIDUAL_ANALYSIS.md) for the full request/response schema and tuning guidance.

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

## Documentation

* [API Reference](docs/API.md)
* [Deployment Guide](docs/DEPLOYMENT.md)
* [Architecture](docs/ARCHITECTURE.md)
* [Residual Symbolic Analysis](docs/RESIDUAL_ANALYSIS.md)

## Citation

The residual symbolic analyzer implements the EML operator from:

> Odrzywołek, A. *All elementary functions from a single binary operator.*
> arXiv preprint [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (2026).

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywołek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Support

* Email: [ciprian.pater@publicae.org](mailto:ciprian.pater@publicae.org)
* Issues: [GitHub Issues](https://github.com/RedCiprianPater/nwo-timesfm-integration/issues)
