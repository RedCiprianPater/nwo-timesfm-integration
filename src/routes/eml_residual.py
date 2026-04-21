"""
FastAPI route: POST /api/timesfm/residual-analysis

Adds a symbolic-regression endpoint that extracts closed-form elementary
laws from TimesFM forecast residuals. Based on Odrzywołek's EML operator
(arXiv:2603.21852).

Registered into `src/server.py` via the patch shown in INTEGRATION.md.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from ..residual_analyzer import TimesFMResidualAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/timesfm", tags=["residual-analysis"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ResidualAnalysisRequest(BaseModel):
    """Input for POST /api/timesfm/residual-analysis."""

    features: list[list[float]] = Field(
        ...,
        description="Feature matrix, shape (n_samples, n_features). "
                    "Each row is the context for one forecasted point.",
    )
    y_true: list[float] = Field(
        ...,
        description="Ground-truth values aligned to features rows.",
    )
    y_forecast: list[float] = Field(
        ...,
        description="TimesFM forecasts aligned to features rows.",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Optional column names, used when rendering the expression.",
    )
    depth: int | None = Field(
        default=None,
        ge=1,
        le=6,
        description="Tree depth (1–6). Omit to use automatic depth search.",
    )
    n_epochs: int = Field(default=2000, ge=100, le=20000)
    seed: int = Field(default=0)


class ResidualAnalysisResponse(BaseModel):
    expression: str
    simplified: str
    final_loss: float
    tree_size: int
    depth_used: int
    n_samples: int
    feature_names: list[str]
    paper_reference: str
    summary: str


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

@router.post("/residual-analysis", response_model=ResidualAnalysisResponse)
async def residual_analysis(req: ResidualAnalysisRequest) -> ResidualAnalysisResponse:
    """Fit a closed-form elementary law to TimesFM forecast residuals.

    The endpoint is stateless — each call trains a fresh EML tree on the
    submitted (features, y_true, y_forecast) tuple and returns the best
    closed-form expression it can recover for `y_true − y_forecast`.

    Typical use cases:

    - **Maintenance prediction**: residual has a clean aging curve
      (e.g. `log(hrs)` or `sqrt(hrs)`) → add as explicit feature in the
      next TimesFM finetune.
    - **Anomaly detection**: residual has a structured periodic component
      → systematic bias, not a real anomaly.
    - **Swarm load forecasting**: closed-form law is small enough to
      deploy on-robot, sidestepping the need for on-device TimesFM.

    Raises
    ------
    400
        Input shapes don't match, or fewer than 8 samples.
    500
        Fit failed (usually numerical — try higher `n_epochs` or a
        different `seed`).
    """
    try:
        analyzer = TimesFMResidualAnalyzer(
            depth=req.depth or 3,
            n_epochs=req.n_epochs,
            seed=req.seed,
        )

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
        # Input-validation style errors — surface as 400.
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:  # noqa: BLE001
        logger.exception("EML residual fit failed")
        raise HTTPException(
            status_code=500,
            detail=f"residual analysis failed: {e}",
        )

    return ResidualAnalysisResponse(
        expression=law.expression,
        simplified=law.simplified,
        final_loss=law.final_loss,
        tree_size=law.tree_size,
        depth_used=law.depth_used,
        n_samples=law.n_samples,
        feature_names=law.feature_names,
        paper_reference=law.paper_reference,
        summary=law.summary_line(),
    )


__all__ = ["router"]
