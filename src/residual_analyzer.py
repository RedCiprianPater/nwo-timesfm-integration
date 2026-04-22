"""
TimesFM forecast residual → closed-form elementary law.

Fits a differentiable EML symbolic-regression tree to
`residual = y_true - y_forecast` against a feature matrix, then returns
any closed-form elementary expression the tree recovers.

The math comes from

    Andrzej Odrzywołek,
    "All elementary functions from a single binary operator",
    arXiv:2603.21852 (2026).

The operator `eml(x, y) = exp(x) - ln(y)` together with the constant 1
generates the entire scientific-calculator basis, so any elementary law
hiding in TimesFM's systematic error is representable as a shallow EML
tree. This module is the service-layer wrapper that calls the underlying
regressor (from the `nwo-eml-regression` package) and shapes inputs and
outputs for the NWO API.

Typical call site is `src/routes/eml_residual.py`, which exposes it as
`POST /api/timesfm/residual-analysis`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    from nwo_eml import EMLRegressor
    from nwo_eml.simplify import simplify_tree
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "nwo-eml-regression is required. Install with:\n"
        "    pip install 'nwo-eml-regression[sympy]'"
    ) from e


# ---------------------------------------------------------------------------

@dataclass
class ResidualLaw:
    """A recovered symbolic relationship for the forecast residual."""

    expression: str
    simplified: str
    final_loss: float
    tree_size: int
    depth_used: int
    n_samples: int
    feature_names: list
    paper_reference: str = "Odrzywołek, arXiv:2603.21852 (2026)"

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"residual ≈ {self.simplified}  "
            f"(loss={self.final_loss:.3e}, size={self.tree_size}, "
            f"depth={self.depth_used})"
        )


# ---------------------------------------------------------------------------

class TimesFMResidualAnalyzer:
    """Fit an EML symbolic-regression tree to TimesFM forecast residuals."""

    def __init__(self, *, depth=3, n_epochs=2000, normalize=True, seed=0):
        self.depth = depth
        self.n_epochs = n_epochs
        self.normalize = normalize
        self.seed = seed

    # -- core API ----------------------------------------------------------

    def analyze(self, features, y_true, y_forecast, *, feature_names=None):
        """Fit a closed-form law to `y_true − y_forecast`."""
        features = np.asarray(features, dtype=float)
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_forecast = np.asarray(y_forecast, dtype=float).ravel()

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n = features.shape[0]
        if not (len(y_true) == len(y_forecast) == n):
            raise ValueError(
                "features, y_true, and y_forecast must all have length "
                f"{n}, got {len(y_true)} and {len(y_forecast)}"
            )
        if n < 8:
            raise ValueError(
                "need at least 8 samples for a meaningful residual fit; "
                f"got {n}"
            )

        residual = y_true - y_forecast

        X = features
        if self.normalize:
            std = features.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            X = (features - features.mean(axis=0)) / std

        names = feature_names or [f"x{i}" for i in range(X.shape[1])]

        reg = EMLRegressor(
            depth=self.depth,
            n_epochs=self.n_epochs,
            seed=self.seed,
        ).fit(X, residual, feature_names=names)

        return ResidualLaw(
            expression=reg.result_.expression,
            simplified=simplify_tree(reg.result_.tree),
            final_loss=reg.result_.final_loss,
            tree_size=reg.result_.tree.size(),
            depth_used=self.depth,
            n_samples=n,
            feature_names=names,
        )

    # -- depth search for parsimony ---------------------------------------

    def analyze_with_depth_search(
        self, features, y_true, y_forecast, *,
        depths=None, feature_names=None, min_improvement_factor=0.5,
    ):
        """Pick the shallowest tree that fits well."""
        depths = depths or [2, 3, 4]
        best = None
        for d in depths:
            self.depth = d
            law = self.analyze(
                features, y_true, y_forecast,
                feature_names=feature_names,
            )
            if best is None:
                best = law
                continue
            if law.final_loss < best.final_loss * min_improvement_factor:
                best = law
            else:
                break
        assert best is not None
        return best


__all__ = ["TimesFMResidualAnalyzer", "ResidualLaw"]
