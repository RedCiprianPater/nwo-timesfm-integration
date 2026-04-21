# Residual Symbolic Analysis

`POST /api/timesfm/residual-analysis`

Fits a differentiable symbolic-regression tree to the residuals of a
TimesFM forecast and returns a **closed-form elementary expression** for
the systematic error. Built on the EML operator from

> Odrzywołek, A. *All elementary functions from a single binary operator.*
> arXiv:[2603.21852](https://arxiv.org/abs/2603.21852) (2026).

Because `eml(x, y) = exp(x) − ln(y)` together with the constant `1`
generates every elementary function, any well-behaved systematic bias in
a TimesFM forecast is representable as a shallow EML tree. Training that
tree with Adam gives you the symbolic formula back.

## Why this matters

TimesFM produces strong point forecasts but no interpretable law for the
residuals. The residual endpoint is the inverse problem: given that
residual, find the simplest closed-form function of context features
that explains it.

- **Maintenance prediction** — if residuals correlate with operating
  hours via a recognisable aging law (e.g. `0.5·log(hrs)`), the symbolic
  form is a feature-engineering signal for the next finetune.
- **Anomaly detection** — a clean closed-form residual means the
  forecast is *structurally biased*, not that the input is anomalous.
  Correct the bias, drop the false alarms.
- **Swarm load forecasting** — recovered laws are small enough to deploy
  on edge controllers, sidestepping the need for on-robot TimesFM.

## Request schema

```jsonc
POST /api/timesfm/residual-analysis
Content-Type: application/json

{
  "features":       [[0.5, 23.1], [1.0, 23.4], ...],    // (n_samples, n_features)
  "y_true":         [12.3, 12.7, ...],                  // (n_samples,)
  "y_forecast":     [12.2, 12.5, ...],                  // (n_samples,) — TimesFM outputs
  "feature_names":  ["hrs", "temp"],                    // optional, for readable expression
  "depth":          3,                                  // optional; omit for depth search
  "n_epochs":       2000,                               // optional; default 2000
  "seed":           0                                   // optional; default 0
}
```

Minimum `n_samples` is 8. Maximum `depth` is 6.

## Response schema

```jsonc
{
  "expression":      "eml(1, eml(eml(1, hrs), 1))",
  "simplified":      "0.501*log(hrs) - 0.0012",
  "final_loss":      0.00032,
  "tree_size":       7,
  "depth_used":      3,
  "n_samples":       256,
  "feature_names":   ["hrs", "temp"],
  "paper_reference": "Odrzywołek, arXiv:2603.21852 (2026)",
  "summary":         "residual ≈ 0.501*log(hrs) - 0.0012  (loss=3.2e-04, size=7, depth=3)"
}
```

### Interpretation

- `expression` is the raw `eml(...)` tree as extracted from the trained
  model. Always produced.
- `simplified` is a best-effort SymPy pass. When the tree matches one of
  the paper's canonical constructions (e.g. `ln(x) = eml(1, eml(eml(1,
  x), 1))`), the simplifier returns the standard elementary form.
  Otherwise you get a SymPy-simplified nested expression.
- `final_loss` is MSE on the residuals. Below ~1e-3 on normalised data
  is a good fit; above ~1e-1 means there probably isn't a clean
  closed-form law (noise-dominated residual).

## Example: recover an aging law

Synthetic curl against a running service:

```bash
curl -X POST http://localhost:8000/api/timesfm/residual-analysis \
  -H 'Content-Type: application/json' \
  -d @examples/residual_request.json
```

See [`examples/residual_law_discovery.py`](../examples/residual_law_discovery.py)
for an end-to-end synthetic run that constructs a known `log(hrs)` bias
and verifies the endpoint recovers it.

## When the fit fails

If `final_loss` is high (> 1e-2) and `simplified` looks like random
nested `exp`/`log`:

1. **Raise `n_epochs` to 5000–10000.** The loss surface is non-convex.
2. **Try several `seed` values** (1, 2, 3). Restarting often fixes it.
3. **Omit `depth` to let depth search pick** — sometimes depth 4 is
   needed for products of aging terms.
4. **Check your features.** If none of them actually correlate with the
   residual, no symbolic law exists to find. This is a feature, not a
   bug: a high loss here is evidence the residual is genuine noise or
   requires features you haven't supplied.

## Dependencies

This endpoint requires the `nwo-eml-regression` package:

```
pip install "nwo-eml-regression[sympy]"
```

It is listed in `requirements.txt` and installed automatically by the
Dockerfile.

## Paper citation

The mathematical machinery used here is due to Andrzej Odrzywołek.
Please cite the original paper in any published work:

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywołek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}
```
