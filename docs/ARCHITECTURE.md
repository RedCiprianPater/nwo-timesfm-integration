# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        NWO Robotics Platform                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Web App    │    │  Mobile App  │    │  CLI Tools   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              NWO API Gateway (nwo.capital)               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐  │   │
│  │  │/robotics│ │/iot     │ │/swarm   │ │  /timesfm    │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └──────┬───────┘  │   │
│  └───────────────────────────────────────────────┼─────────┘   │
│                                                  │              │
│                                                  ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TimesFM Service (this repo)                 │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │              Flask API Server                        │ │   │
│  │  │  ┌────────────┐ ┌────────────┐ ┌──────────────────┐ │ │   │
│  │  │  │/maintenance│ │/task-dur   │ │/anomaly          │ │ │   │
│  │  │  └────────────┘ └────────────┘ └──────────────────┘ │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  │                          │                               │   │
│  │                          ▼                               │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │           TimesFM Predictor                          │ │   │
│  │  │  ┌───────────────────────────────────────────────┐ │ │   │
│  │  │  │  TimesFM 2.5 Model (200M parameters)          │ │ │   │
│  │  │  │  - HuggingFace Transformers                   │ │ │   │
│  │  │  │  - PyTorch/JAX Backend                        │ │ │   │
│  │  │  │  - CPU/GPU/TPU Support                        │ │ │   │
│  │  │  └───────────────────────────────────────────────┘ │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Data Sources                                │   │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────────────┐    │   │
│  │  │Robot Telemetry DB         │ │Task History      │    │   │
│  │  │  - Battery │ │  - Duration│ │  - Sensor Logs   │    │   │
│  │  │  - Motor   │ │  - Success │ │  - Anomalies     │    │   │
│  │  │  - Temp    │ │  - Failures│ │  - Alerts        │    │   │
│  │  └────────────┘ └────────────┘ └──────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Maintenance Prediction

```
Robot Telemetry → History Extraction → TimesFM Forecast → Maintenance Decision
     ↓                    ↓                    ↓                    ↓
Battery: [98,95..]    Context: 512      Horizon: 24h      Alert: True/False
Motor: [45,47..]      Freq: hourly      Quantiles:        Action: Schedule
Temp: [65,68..]                          p10/p50/p90       Time: 2026-04-09
```

### 2. Task Duration Forecasting

```
Task Request → Historical Durations → Context Features → TimesFM → Duration Estimate
     ↓                  ↓                    ↓              ↓            ↓
Pick&Place      [45,52,48..]         Weight: 2.5kg      Forecast     54.3s ± 8s
Robot: G1       Context: 8           Distance: 1.2m     p50: 54.3    CI: [48,61]
```

### 3. Anomaly Detection

```
Sensor Stream → Window Extraction → TimesFM Prediction → Threshold Check → Alert
      ↓                ↓                    ↓                  ↓            ↓
IMU: [0.1,0.12..]  Last 50 readings   Next 100 values    > 2σ mean?   Severity: High
Context: 50        Freq: 1min         Quantiles          Prob: 0.89   Action: Inspect
```

## Model Specifications

| Specification | Value |
|--------------|-------|
| Model | TimesFM 2.5 |
| Parameters | 200M |
| Context Length | 512 tokens |
| Horizon | Up to 128 steps |
| Input | Historical time series |
| Output | Point forecast + quantiles (p10, p50, p90) |
| Backend | PyTorch / JAX |
| Hardware | CPU / GPU / TPU |

## API Gateway Integration

The TimesFM service integrates with the existing NWO API Gateway:

```php
// api-timesfm.php
class TimesFMClient {
    private $baseUrl = "http://localhost:8000/api/timesfm";
    
    public function predictMaintenance($robotId, $metric, $history) {
        $response = $this->post("/maintenance", [
            'robot_id' => $robotId,
            'metric' => $metric,
            'history' => $history
        ]);
        return json_decode($response, true);
    }
}
```

## Scaling Considerations

### Horizontal Scaling
- Multiple TimesFM service instances behind load balancer
- Model loaded in each instance (stateless)
- Redis for rate limiting coordination

### GPU Acceleration
- NVIDIA GPU for batch inference
- CUDA 11.8+ required
- 4GB+ VRAM recommended

### Caching
- Forecast results cached for identical inputs
- Redis cache with 5-minute TTL
- Reduces model inference calls by ~60%
