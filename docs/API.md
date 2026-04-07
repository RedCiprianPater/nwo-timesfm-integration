# API Reference for TimesFM NWO Integration

## Base URL

```
Production: https://nwo.capital/api/timesfm
Local: http://localhost:8000/api/timesfm
```

## Authentication

All endpoints require an API key passed in the header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### 1. Predict Maintenance

Forecast robot maintenance needs based on historical metrics.

**Endpoint:** `POST /maintenance`

**Request Body:**
```json
{
  "robot_id": "string",
  "metric": "battery_drain|motor_temp|vibration|joint_stress",
  "history": [number],
  "horizon": 24,
  "context": [string]
}
```

**Example:**
```bash
curl -X POST https://nwo.capital/api/timesfm/maintenance \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "unitree_go2_001",
    "metric": "battery_drain",
    "history": [98, 95, 92, 88, 85, 82, 78, 75, 71, 68],
    "horizon": 24,
    "context": ["joint_stress", "motor_temp"]
  }'
```

**Response:**
```json
{
  "robot_id": "unitree_go2_001",
  "metric": "battery_drain",
  "forecast": [65, 62, 59, 56, 53, 50, 47, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 0, 0],
  "confidence_intervals": {
    "p10": [68, 65, 62, 59, 56, 53, 50, 47, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 0],
    "p90": [62, 59, 56, 53, 50, 47, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 0, 0, 0]
  },
  "maintenance_required": true,
  "estimated_failure_time": "2026-04-09T14:30:00Z",
  "confidence": 0.87
}
```

### 2. Predict Task Duration

Forecast how long a task will take based on historical data.

**Endpoint:** `POST /task-duration`

**Request Body:**
```json
{
  "task_type": "string",
  "robot_model": "string",
  "history_durations": [number],
  "context_features": {
    "object_weight": number,
    "distance": number,
    "complexity": number
  },
  "horizon": 1
}
```

**Example:**
```bash
curl -X POST https://nwo.capital/api/timesfm/task-duration \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "pick_and_place",
    "robot_model": "unitree_g1",
    "history_durations": [45, 52, 48, 61, 55, 49, 58, 53],
    "context_features": {
      "object_weight": 2.5,
      "distance": 1.2,
      "complexity": 0.7
    }
  }'
```

**Response:**
```json
{
  "task_type": "pick_and_place",
  "robot_model": "unitree_g1",
  "predicted_duration": 54.3,
  "confidence_intervals": {
    "p10": 48.2,
    "p50": 54.3,
    "p90": 61.8
  },
  "unit": "seconds",
  "confidence": 0.82
}
```

### 3. Detect Anomalies

Predict sensor anomalies before they cause failures.

**Endpoint:** `POST /anomaly`

**Request Body:**
```json
{
  "sensor_type": "string",
  "robot_id": "string",
  "readings": [number],
  "horizon": 100,
  "threshold_quantile": 0.95
}
```

**Example:**
```bash
curl -X POST https://nwo.capital/api/timesfm/anomaly \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_type": "imu_accelerometer",
    "robot_id": "husky_001",
    "readings": [0.1, 0.12, 0.11, 0.13, 0.15, 0.18, 0.22, 0.28, 0.35, 0.45],
    "horizon": 50,
    "threshold_quantile": 0.95
  }'
```

**Response:**
```json
{
  "sensor_type": "imu_accelerometer",
  "robot_id": "husky_001",
  "anomaly_detected": true,
  "anomaly_probability": 0.89,
  "predicted_anomaly_time": "2026-04-08T03:45:00Z",
  "forecast": [0.52, 0.58, 0.65, 0.73, 0.82, 0.91, 1.02, 1.15, 1.28, 1.42],
  "recommended_action": "inspect_sensor",
  "severity": "high"
}
```

### 4. Predict Swarm Load

Forecast resource needs for robot swarms.

**Endpoint:** `POST /swarm-load`

**Request Body:**
```json
{
  "swarm_id": "string",
  "active_robots": number,
  "task_queue_history": [number],
  "horizon": 48
}
```

**Example:**
```bash
curl -X POST https://nwo.capital/api/timesfm/swarm-load \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "swarm_id": "warehouse_fleet_01",
    "active_robots": 12,
    "task_queue_history": [45, 52, 48, 61, 55, 49, 58, 53, 60, 67],
    "horizon": 48
  }'
```

**Response:**
```json
{
  "swarm_id": "warehouse_fleet_01",
  "active_robots": 12,
  "predicted_queue_depth": [72, 78, 85, 91, 96, 102, 108, 115, 122, 128],
  "recommended_robots": 15,
  "scale_up_recommended": true,
  "optimal_scale_time": "2026-04-08T06:00:00Z",
  "confidence": 0.79
}
```

### 5. Health Check

Check if the TimesFM service is running.

**Endpoint:** `GET /health`

**Example:**
```bash
curl https://nwo.capital/api/timesfm/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2.5-200m",
  "uptime": 86400,
  "requests_served": 15234
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid request",
  "message": "history array must contain at least 10 data points",
  "code": "INVALID_INPUT"
}
```

### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing API key",
  "code": "AUTH_FAILED"
}
```

### 429 Rate Limited
```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 100 requests per minute",
  "retry_after": 60
}
```

### 500 Server Error
```json
{
  "error": "Internal server error",
  "message": "Model inference failed",
  "code": "INFERENCE_ERROR"
}
```

## Rate Limits

| Plan | Requests/Min | Requests/Day |
|------|--------------|--------------|
| Free | 10 | 100 |
| Pro | 100 | 10,000 |
| Enterprise | Unlimited | Unlimited |

## SDK Examples

### Python
```python
import requests

class NWOTimesFMClient:
    def __init__(self, api_key, base_url="https://nwo.capital/api/timesfm"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def predict_maintenance(self, robot_id, metric, history, horizon=24):
        response = requests.post(
            f"{self.base_url}/maintenance",
            headers=self.headers,
            json={
                "robot_id": robot_id,
                "metric": metric,
                "history": history,
                "horizon": horizon
            }
        )
        return response.json()

# Usage
client = NWOTimesFMClient("your-api-key")
result = client.predict_maintenance(
    robot_id="unitree_go2_001",
    metric="battery_drain",
    history=[98, 95, 92, 88, 85, 82, 78, 75, 71, 68]
)
print(f"Maintenance needed: {result['maintenance_required']}")
```

### JavaScript
```javascript
class NWOTimesFMClient {
  constructor(apiKey, baseUrl = 'https://nwo.capital/api/timesfm') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async predictMaintenance(robotId, metric, history, horizon = 24) {
    const response = await fetch(`${this.baseUrl}/maintenance`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        robot_id: robotId,
        metric: metric,
        history: history,
        horizon: horizon
      })
    });
    return response.json();
  }
}

// Usage
const client = new NWOTimesFMClient('your-api-key');
const result = await client.predictMaintenance(
  'unitree_go2_001',
  'battery_drain',
  [98, 95, 92, 88, 85, 82, 78, 75, 71, 68]
);
console.log('Maintenance needed:', result.maintenance_required);
```
