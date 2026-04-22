"""
Flask blueprint: POST /api/timesfm/residual-analysis

Recovers closed-form elementary laws from TimesFM forecast residuals.

Based on:
    Andrzej Odrzywołek,
    "All elementary functions from a single binary operator",
    arXiv:2603.21852 (2026).

Registered from src/server.py:

    from src.routes.eml_residual import bp as eml_residual_bp
    app.register_blueprint(eml_residual_bp)
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request
from pydantic import BaseModel, Field, ValidationError

from ..residual_analyzer import TimesFMResidualAnalyzer

logger = logging.getLogger(__name__)

bp = Blueprint("eml_residual", __name__, url_prefix="/api/timesfm")


# ---------------------------------------------------------------------------
# Request schema (Pydantic v2)
# ---------------------------------------------------------------------------

class ResidualAnalysisRequest(BaseModel):
    """Input for POST /api/timesfm/residual-analysis."""

    features: list
    y_true: list
    y_forecast: list
    feature_names: list | None = None
    depth: int | None = Field(default=None, ge=1, le=6)
    n_epochs: int = Field(default=2000, ge=100, le=20000)
    seed: int = 0


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

@bp.route("/residual-analysis", methods=["POST"])
def residual_analysis():
    """Fit a closed-form elementary law to TimesFM forecast residuals.

    The endpoint is stateless — each call trains a fresh EML tree on the
    submitted (features, y_true, y_forecast) tuple and returns the best
    closed-form expression it can recover for `y_true − y_forecast`.

    Returns JSON:
        expression       — raw eml(...) form extracted from the trained tree
        simplified       — human-readable SymPy-simplified closed form
        final_loss       — MSE on residuals after training
        tree_size        — node count in the extracted tree
        depth_used       — depth chosen by depth search (or supplied)
        n_samples        — residual points used for the fit
        feature_names    — column labels passed through into the expression
        paper_reference  — always "Odrzywołek, arXiv:2603.21852 (2026)"
        summary          — one-line human-readable summary

    Status codes:
        200   fit completed (check `final_loss` to judge quality)
        400   input validation failure
        500   unexpected fit failure (try more `n_epochs` or a different `seed`)
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "request body must be JSON"}), 400

    try:
        req = ResidualAnalysisRequest(**payload)
    except ValidationError as e:
        return jsonify({"error": "validation failed", "details": e.errors()}), 400

    analyzer = TimesFMResidualAnalyzer(
        depth=req.depth or 3,
        n_epochs=req.n_epochs,
        seed=req.seed,
    )

    try:
        if req.depth is None:
            law = analyzer.analyze_with_depth_search(
                features=req.features,
                y_true=req.y_true,
                y_forecast=req.y_forecast,
                feature_names=req.feature_names,
            )
        else:
            law = analyzer.analyze(
                features=req.features,
                y_true=req.y_true,
                y_forecast=req.y_forecast,
                feature_names=req.feature_names,
            )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception("EML residual fit failed")
        return jsonify({
            "error": f"residual analysis failed: {e}",
            "hint": "try increasing n_epochs or changing seed",
        }), 500

    return jsonify({
        "expression":      law.expression,
        "simplified":      law.simplified,
        "final_loss":      law.final_loss,
        "tree_size":       law.tree_size,
        "depth_used":      law.depth_used,
        "n_samples":       law.n_samples,
        "feature_names":   law.feature_names,
        "paper_reference": law.paper_reference,
        "summary":         law.summary_line(),
    }), 200


__all__ = ["bp"]
