# TimesFM Integration for NWO Robotics API

Time Series Foundation Model (TimesFM) powered forecasting for robot maintenance, task prediction, and anomaly detection.

## Overview

This integration adds time-series forecasting capabilities to NWO Robotics using Google's TimesFM 2.5 model (200M parameters).

## Features

- **Robot Maintenance Prediction** - Predict battery failure, motor degradation
- **Task Duration Forecasting** - Estimate task completion times
- **Sensor Anomaly Detection** - Predict failures before they happen
- **Swarm Load Forecasting** - Scale robot fleets intelligently

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/timesfm/maintenance` | Predict robot maintenance needs |
| `POST /api/timesfm/task-duration` | Forecast task completion times |
| `POST /api/timesfm/anomaly` | Detect sensor anomalies |
| `POST /api/timesfm/swarm-load` | Predict swarm resource needs |
| `GET /api/timesfm/health` | Check service health |

## Quick Start

### Installation

```bash
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

### Usage Example

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

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   NWO Robotics  │────▶│  TimesFM API     │────▶│  TimesFM Model  │
│   API Gateway   │     │  (this repo)     │     │  (200M params)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Robot Telemetry │
                        │  Database        │
                        └──────────────────┘
```

## Documentation

- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture](docs/ARCHITECTURE.md)

## License

MIT License - see [LICENSE](LICENSE)

## Support

- Email: ciprian.pater@publicae.org
- Issues: [GitHub Issues](https://github.com/RedCiprianPater/nwo-timesfm-integration/issues)
