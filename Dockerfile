FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Download model (optional - can be mounted as volume)
RUN python scripts/download_model.py || echo "Model download failed, will use mock"

# Expose port
EXPOSE 8000

# Run server
CMD gunicorn --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT src.server:app
