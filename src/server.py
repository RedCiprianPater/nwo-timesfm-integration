#!/usr/bin/env python3
"""
TimesFM Model Server for NWO Robotics API
Serves time-series forecasting endpoints using Google's TimesFM 2.5

Also exposes the EML residual-analysis endpoint from Odrzywołek
(arXiv:2603.21852) via src/routes/eml_residual.py.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from functools import wraps

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np

# TimesFM imports
try:
    from timesfm import TimesFm
    from huggingface_hub import hf_hub_download
    TIMEFM_AVAILABLE = True
except ImportError:
    TIMEFM_AVAILABLE = False
    print("Warning: TimesFM not installed. Using mock predictions.")

app = Flask(__name__)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

# Global state
model = None
model_version = "2.5-200m"
start_time = time.time()
request_count = 0

# Mock data for when TimesFM is not available
def mock_forecast(history: List[float], horizon: int) -> Dict[str, Any]:
    """Generate mock forecasts when model is not available"""
    last_value = history[-1] if history else 0
    trend = (history[-1] - history[0]) / len(history) if len(history) > 1 else 0
    
    forecast = []
    for i in range(horizon):
        value = last_value + trend * (i + 1) + np.random.normal(0, abs(trend) * 0.1)
        forecast.append(round(value, 2))
    
    return {
        "forecast": forecast,
        "p10": [v * 0.9 for v in forecast],
        "p50": forecast,
        "p90": [v * 1.1 for v in forecast]
    }

class TimesFMPredictor:
    """Wrapper for TimesFM model"""
    
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.load_model()
    
    def load_model(self):
        """Load TimesFM 2.5 model from HuggingFace"""
        if not TIMEFM_AVAILABLE:
            print("TimesFM not available, using mock predictions")
            return
        
        try:
            print("Loading TimesFM 2.5 model...")
            
            # Initialize model
            self.model = TimesFm(
                context_len=512,
                horizon_len=128,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend="cpu",  # Change to "gpu" if CUDA available
            )
            
            # Load checkpoint from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id="google/timesfm-2.5-200m",
                filename="timesfm-2.5-200m.pth"
            )
            
            self.model.load_from_checkpoint(checkpoint_path)
            print(f"Model loaded successfully from {checkpoint_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock predictions")
            self.model = None
    
    def predict(
        self, 
        history: List[float], 
        horizon: int,
        freq: str = "H"
    ) -> Dict[str, Any]:
        """Generate forecast"""
        
        if self.model is None:
            return mock_forecast(history, horizon)
        
        try:
            # Prepare input
            context = np.array(history, dtype=np.float32)
            
            # Run prediction
            forecast, quantiles = self.model.forecast(
                inputs=[context],
                freq=[freq],
                horizon_len=horizon,
                return_quantiles=True
            )
            
            # Extract results
            forecast_list = forecast[0].tolist()
            
            # Get quantiles (p10, p50, p90)
            p10 = quantiles[0][0].tolist() if quantiles is not None else [v * 0.9 for v in forecast_list]
            p50 = quantiles[0][1].tolist() if quantiles is not None else forecast_list
            p90 = quantiles[0][2].tolist() if quantiles is not None else [v * 1.1 for v in forecast_list]
            
            return {
                "forecast": [round(v, 2) for v in forecast_list],
                "p10": [round(v, 2) for v in p10],
                "p50": [round(v, 2) for v in p50],
                "p90": [round(v, 2) for v in p90]
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return mock_forecast(history, horizon)

# Initialize predictor
predictor = TimesFMPredictor()

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        # In production, validate against database
        if not api_key:
            return jsonify({
                "error": "Unauthorized",
                "message": "API key required",
                "code": "AUTH_FAILED"
            }), 401
        
        # TODO: Validate API key against database
        # For now, accept any non-empty key
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# EML Residual Analysis Blueprint
# ----------------------------------------------------------------------------
# Registers POST /api/timesfm/residual-analysis which recovers closed-form
# elementary laws from forecast residuals using Odrzywołek's EML operator
# (arXiv:2603.21852). See src/routes/eml_residual.py.
# ============================================================================
try:
    from src.routes.eml_residual import bp as eml_residual_bp
    app.register_blueprint(eml_residual_bp)
    print("EML residual-analysis endpoint registered at /api/timesfm/residual-analysis")
except Exception as e:
    # Never let the EML blueprint failure take down the TimesFM service.
    print(f"Warning: EML residual blueprint failed to load: {e}")

# Health check endpoint
@app.route('/api/timesfm/health', methods=['GET'])
def health_check():
    """Check service health"""
    global request_count
    
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_version": model_version,
        "using_mock": predictor.model is None,
        "uptime": int(time.time() - start_time),
        "requests_served": request_count
    })

# Maintenance prediction endpoint
@app.route('/api/timesfm/maintenance', methods=['POST'])
@limiter.limit("100 per minute")
@require_api_key
def predict_maintenance():
    """Predict robot maintenance needs"""
    global request_count
    request_count += 1
    
    data = request.get_json()
    
    # Validate input
    if not data or 'history' not in data:
        return jsonify({
            "error": "Invalid request",
            "message": "history array required",
            "code": "INVALID_INPUT"
        }), 400
    
    history = data.get('history', [])
    if len(history) < 10:
        return jsonify({
            "error": "Invalid request",
            "message": "history must contain at least 10 data points",
            "code": "INVALID_INPUT"
        }), 400
    
    robot_id = data.get('robot_id', 'unknown')
    metric = data.get('metric', 'generic')
    horizon = min(data.get('horizon', 24), 128)  # Max 128
    
    # Generate forecast
    forecast_data = predictor.predict(history, horizon)
    
    # Determine if maintenance needed
    forecast_values = forecast_data['forecast']
    threshold = min(history) * 0.2  # 20% of min value
    maintenance_required = any(v < threshold for v in forecast_values)
    
    # Estimate failure time
    failure_time = None
    for i, v in enumerate(forecast_values):
        if v < threshold:
            failure_time = datetime.utcnow() + timedelta(hours=i)
            break
    
    # Calculate confidence based on variance
    variance = np.var(forecast_values)
    confidence = max(0.5, 1.0 - (variance / (np.mean(forecast_values) ** 2)))
    
    return jsonify({
        "robot_id": robot_id,
        "metric": metric,
        "forecast": forecast_values,
        "confidence_intervals": {
            "p10": forecast_data['p10'],
            "p50": forecast_data['p50'],
            "p90": forecast_data['p90']
        },
        "maintenance_required": maintenance_required,
        "estimated_failure_time": failure_time.isoformat() + "Z" if failure_time else None,
        "confidence": round(confidence, 2),
        "threshold": round(threshold, 2)
    })

# Task duration prediction endpoint
@app.route('/api/timesfm/task-duration', methods=['POST'])
@limiter.limit("100 per minute")
@require_api_key
def predict_task_duration():
    """Predict task completion time"""
    global request_count
    request_count += 1
    
    data = request.get_json()
    
    if not data or 'history_durations' not in data:
        return jsonify({
            "error": "Invalid request",
            "message": "history_durations array required",
            "code": "INVALID_INPUT"
        }), 400
    
    history = data.get('history_durations', [])
    if len(history) < 5:
        return jsonify({
            "error": "Invalid request",
            "message": "history_durations must contain at least 5 data points",
            "code": "INVALID_INPUT"
        }), 400
    
    task_type = data.get('task_type', 'generic')
    robot_model = data.get('robot_model', 'unknown')
    context = data.get('context_features', {})
    
    # Generate forecast (single horizon for task duration)
    forecast_data = predictor.predict(history, horizon=1)
    predicted_duration = forecast_data['forecast'][0]
    
    # Adjust based on context features
    if 'object_weight' in context:
        predicted_duration *= (1 + context['object_weight'] * 0.1)
    if 'distance' in context:
        predicted_duration *= (1 + context['distance'] * 0.05)
    if 'complexity' in context:
        predicted_duration *= (1 + context['complexity'] * 0.2)
    
    return jsonify({
        "task_type": task_type,
        "robot_model": robot_model,
        "predicted_duration": round(predicted_duration, 1),
        "confidence_intervals": {
            "p10": round(forecast_data['p10'][0], 1),
            "p50": round(forecast_data['p50'][0], 1),
            "p90": round(forecast_data['p90'][0], 1)
        },
        "unit": "seconds",
        "confidence": 0.82,
        "context_applied": context
    })

# Anomaly detection endpoint
@app.route('/api/timesfm/anomaly', methods=['POST'])
@limiter.limit("100 per minute")
@require_api_key
def detect_anomaly():
    """Detect sensor anomalies"""
    global request_count
    request_count += 1
    
    data = request.get_json()
    
    if not data or 'readings' not in data:
        return jsonify({
            "error": "Invalid request",
            "message": "readings array required",
            "code": "INVALID_INPUT"
        }), 400
    
    readings = data.get('readings', [])
    if len(readings) < 10:
        return jsonify({
            "error": "Invalid request",
            "message": "readings must contain at least 10 data points",
            "code": "INVALID_INPUT"
        }), 400
    
    sensor_type = data.get('sensor_type', 'generic')
    robot_id = data.get('robot_id', 'unknown')
    horizon = min(data.get('horizon', 50), 128)
    threshold_quantile = data.get('threshold_quantile', 0.95)
    
    # Generate forecast
    forecast_data = predictor.predict(readings, horizon)
    forecast_values = forecast_data['forecast']
    
    # Calculate threshold based on historical data
    historical_mean = np.mean(readings)
    historical_std = np.std(readings)
    threshold = historical_mean + (historical_std * 2)  # 2 sigma
    
    # Detect anomalies
    a
