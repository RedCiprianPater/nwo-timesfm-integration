# Deployment Guide

## Prerequisites

- Python 3.9+
- 4GB+ RAM (8GB recommended)
- CUDA-capable GPU (optional, for faster inference)

## Local Deployment

### 1. Clone and Setup

```bash
git clone https://github.com/RedCiprianPater/nwo-timesfm-integration.git
cd nwo-timesfm-integration
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Model

```bash
python scripts/download_model.py
```

### 3. Run Server

```bash
# Development
python src/server.py

# Production with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 src.server:app
```

## Docker Deployment

### Build Image

```bash
docker build -t nwo-timesfm .
```

### Run Container

```bash
docker run -p 8000:8000 -e API_KEY=your-key nwo-timesfm
```

## Cloud Deployment

### Render.com

1. Create a new Web Service
2. Connect your GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn src.server:app`
5. Add environment variables

### AWS Lambda

Use the provided `lambda_handler` in `src/lambda.py` with API Gateway.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `DEBUG` | Debug mode | false |
| `MODEL_PATH` | Path to TimesFM model | ./models |
| `RATE_LIMIT` | Requests per minute | 100 |

## Nginx Configuration

```nginx
server {
    listen 80;
    server_name api.nwo.capital;
    
    location /api/timesfm {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
