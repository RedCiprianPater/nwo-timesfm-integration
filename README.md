# `nwo-timesfm-integration` — Complete Update Package

This bundle contains everything you need to add the EML residual
symbolic-analysis endpoint to your
[`nwo-timesfm-integration`](https://github.com/RedCiprianPater/nwo-timesfm-integration)
repo.

The zip is organised into two top-level folders:

```
timesfm-full/
├── new-files/         Files that DO NOT yet exist in your repo — just copy them in.
├── updated-files/     Files that DO exist — overwrite the old versions.
├── SERVER_PY_PATCH.md Instructions for the one file I couldn't see (server.py).
└── README.md          This guide.
```

---

## Part 1 — Files to ADD (from `new-files/`)

Copy these into your local clone of the repo at the exact same relative
paths. **None of these files exist yet in your repo**, so there's
nothing to merge — just drop them in.

```
new-files/src/residual_analyzer.py        →  src/residual_analyzer.py
new-files/src/routes/__init__.py          →  src/routes/__init__.py   (creates new folder)
new-files/src/routes/eml_residual.py      →  src/routes/eml_residual.py
new-files/docs/RESIDUAL_ANALYSIS.md       →  docs/RESIDUAL_ANALYSIS.md
new-files/examples/residual_law_discovery.py  →  examples/residual_law_discovery.py   (creates new folder)
new-files/scripts/test_residual_analyzer.py   →  scripts/test_residual_analyzer.py
```

### Folders you need to create (if they don't already exist)

- `src/routes/` — **new subpackage** for route modules. The
  `__init__.py` inside is what makes it importable.
- `examples/` — new top-level folder for end-to-end demos.

Your final repo layout after adding the new files will look like:

```
nwo-timesfm-integration/
├── docs/
│   ├── API.md                        (existing)
│   ├── ARCHITECTURE.md               (existing)
│   ├── DEPLOYMENT.md                 (existing)
│   └── RESIDUAL_ANALYSIS.md          ← NEW
├── examples/                         ← NEW folder
│   └── residual_law_discovery.py     ← NEW
├── scripts/
│   ├── download_model.py             (existing)
│   └── test_residual_analyzer.py     ← NEW
├── src/
│   ├── server.py                     (existing — needs 2-line patch, see SERVER_PY_PATCH.md)
│   ├── residual_analyzer.py          ← NEW
│   └── routes/                       ← NEW folder
│       ├── __init__.py               ← NEW
│       └── eml_residual.py           ← NEW
├── .gitignore                        (existing)
├── Dockerfile                        (existing)
├── LICENSE                           (existing)
├── README.md                         (REPLACE with updated-files/README.md)
├── package.json                      (existing)
└── requirements.txt                  (REPLACE with updated-files/requirements.txt)
```

---

## Part 2 — Files to REPLACE (from `updated-files/`)

Overwrite these existing files with the versions in `updated-files/`.
I rebuilt them preserving all original content and only adding what's
needed.

```
updated-files/requirements.txt   →  requirements.txt    (OVERWRITE)
updated-files/README.md          →  README.md           (OVERWRITE)
```

### What changed in `requirements.txt`

- Added one line at the bottom:
  ```
  nwo-eml-regression[sympy] @ git+https://github.com/RedCiprianPater/nwo-eml-regression.git
  ```
- Nothing else changed.

### What changed in `README.md`

- Added the new endpoint to the features list.
- Added a new row to the API Endpoints table.
- Added a new code example showing the residual-analysis endpoint.
- Added a Documentation link to `docs/RESIDUAL_ANALYSIS.md`.
- Added a Citation section pointing at arXiv:2603.21852.
- All existing content preserved — nothing was removed.

---

## Part 3 — Patch `src/server.py`

I couldn't fetch `server.py` directly from GitHub, so there's no
"complete replacement" version for it. Instead, see
**`SERVER_PY_PATCH.md`** for a two-line patch.

Summary:

1. Near the other Flask imports at the top of `src/server.py`, add:
   ```python
   from src.routes.eml_residual import bp as eml_residual_bp
   ```
2. After the `app = Flask(__name__)` line, add:
   ```python
   app.register_blueprint(eml_residual_bp)
   ```

---

## Part 4 — Verification

Once everything is in place:

```bash
# 1. Install the new dependency
pip install -r requirements.txt

# 2. Local smoke test (no server boot needed)
python scripts/test_residual_analyzer.py

# Expected: a line like
#   residual ≈ ...log(hrs)...  (loss=X.Xe-XX, size=7, depth=3)
# If the WARNING about fit quality fires, that's OK — the analyzer is
# telling you the fit landed in a bad local minimum. Try a different
# seed or more n_epochs for real data.

# 3. Full endpoint test
python src/server.py           # in one terminal
python examples/residual_law_discovery.py   # in another
```

Expected output from the example script:

```
POST http://localhost:8000/api/timesfm/residual-analysis
  samples           : 256
  ground-truth law  : residual = 0.5 * log(hrs) + small noise
  epochs            : 2500

=== recovered law ===
summary        : residual ≈ 0.5*log(hrs)  (loss=3.2e-04, size=7, depth=3)
simplified     : 0.5*log(hrs)
raw eml form   : eml(1, eml(eml(1, hrs), 1))
final loss     : 3.2045e-04
tree size      : 7
depth used     : 3
paper ref      : Odrzywołek, arXiv:2603.21852 (2026)
```

---

## Part 5 — Hyperparameter notes

The EML symbolic regressor is a small non-convex optimisation. Expect
to tune for real telemetry:

- **Default `n_epochs=2000`** works for clean signals. For noisier
  residuals, try 5000–10000.
- **Seed matters.** If a fit lands in a bad local minimum, change
  `seed` to 1, 2, or 3 and rerun.
- **Depth search (`depth=None`)** tries 2 → 3 → 4 and keeps the
  shallowest tree that fits. Use this when you don't know a priori
  what kind of law to expect.
- **Feature scale matters.** The `normalize=True` default z-scores
  features; this keeps the guarded `exp`/`ln` operator inside its
  happy region and drastically improves convergence.

---

## Part 6 — One commit message you can use

```
Add EML residual symbolic analysis endpoint

Introduces POST /api/timesfm/residual-analysis, which fits a
differentiable symbolic-regression tree to TimesFM forecast residuals
and returns a closed-form elementary expression.

Based on Odrzywołek, "All elementary functions from a single binary
operator", arXiv:2603.21852 (2026). Depends on the nwo-eml-regression
package.

Adds:
  - src/residual_analyzer.py
  - src/routes/__init__.py
  - src/routes/eml_residual.py
  - docs/RESIDUAL_ANALYSIS.md
  - examples/residual_law_discovery.py
  - scripts/test_residual_analyzer.py

Updates:
  - requirements.txt (adds nwo-eml-regression[sympy])
  - README.md (new endpoint documentation + citation)
  - src/server.py (registers the eml_residual blueprint)
```
