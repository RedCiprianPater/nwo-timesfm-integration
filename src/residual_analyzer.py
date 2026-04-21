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
        "    pip install 'nwo-eml-regression[sympy]'\n"
        "or, from the NWO fork:\n"
        "    pip install 'git+https://github.com/RedCiprianPater/nwo-eml-regression'"
    ) from e


# ---------------------------------------------------------------------------

@dataclass
class ResidualLaw:
    """A recovered symbolic relationship for the forecast residual."""

    expression: str           # raw eml(...) form extracted from the trained tree
    simplified: str           # human-readable closed form (best-effort SymPy pass)
    final_loss: float         # MSE on the residuals after training
    tree_size: int            # number of nodes in the extracted tree
    depth_used: int           # actual tree depth used for this fit
    n_samples: int            # number of residual points the law was fit on
    feature_names: list[str]  # columns used, in order
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
    """Fit an EML symbolic-regression tree to TimesFM forecast residuals.

    Parameters
    ----------
    depth
        Default tree depth when not using `analyze_with_depth_search`.
        The paper reports exact recovery at depths ≤ 4 for elementary
        generators; NWO's typical residuals fit at depth 3.
    n_epochs
        Adam optimiser steps. 2000 is usually enough for clean signals;
        raise to 5000 for noisy residuals.
    normalize
        Z-score features before fitting. Strongly recommended — the
        guarded `exp`/`ln` operator behaves best when inputs are in
        roughly `[-3, 3]`.
    seed
        RNG seed for reproducibility across API calls.
    """

    def __init__(
        self,
        *,
        depth: int = 3,
        n_epochs: int = 2000,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        self.depth = depth
        self.n_epochs = n_epochs
        self.normalize = normalize
        self.seed = seed

    # -- core API ----------------------------------------------------------

    def analyze(
        self,
        features: np.ndarray | list,
        y_true: np.ndarray | list,
        y_forecast: np.ndarray | list,
        *,
        feature_names: list[str] | None = None,
    ) -> ResidualLaw:
        """Fit a closed-form law to `y_true − y_forecast`.

        Parameters
        ----------
        features
            Matrix of shape `(n_samples, n_features)`. Each row is the
            context for one forecasted point — operating hours, payload
            mass, ambient temperature, battery level, whatever you track.
        y_true
            Ground-truth values, aligned row-wise with `features`.
        y_forecast
            TimesFM forecasts, aligned row-wise with `features`.
        feature_names
            Optional, used when rendering the extracted expression.
        """
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
            std = np.where(std < 1e-8, 1.0, std)  # guard constant columns
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
        self,
        features: np.ndarray | list,
        y_true: np.ndarray | list,
        y_forecast: np.ndarray | list,
        *,
        depths: list[int] | None = None,
        feature_names: list[str] | None = None,
        min_improvement_factor: float = 0.5,
    ) -> ResidualLaw:
        """Pick the shallowest tree that fits well.

        Tries the given depths in order; stops climbing once the loss
        fails to drop by at least `min_improvement_factor` (i.e. the
        new fit must be at most half the previous loss to be worth the
        extra depth).
        """
        depths = depths or [2, 3, 4]
        best: ResidualLaw | None = None

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
