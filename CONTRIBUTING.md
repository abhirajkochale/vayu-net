# Contributing to VAYU-NET

Thank you for your interest in contributing to **VAYU-NET** (SIH 2026 Problem Statement 26070). This document outlines our engineering standards, branch conventions, and scientific integrity invariants.

---

## 1. Core Engineering Principles

1. **Non-Destructive Development:**  
   Never delete or overwrite locked research checkpoints (`data/interim/ml/checkpoints/`), manifests, or historical experiment results.
2. **Strict Anti-Leakage Rules:**  
   Operational inference sequences must terminate strictly at observation time $t_0$. Never introduce future ground-truth positions ($t_{+12\text{h}}$, $+24\text{h}$, $+48\text{h}$) into the model input pipeline.
3. **Decoupled Heavy Data:**  
   Never commit raw datasets (`data/raw/`), satellite archives (`data/interim/gridsat/`), or model weights (`*.pt`) to Git. Use `artifacts/manifest.json` and `scripts/setup/download_runtime_artifacts.py` instead.
4. **Causal Attribution Disclaimer:**  
   Always treat Grad-CAM attributions as **interpretation aids**, not causal meteorological claims.

---

## 2. Git Workflow & Branching Strategy

- **Main Branch (`main`):** Production-ready, deployable code. All tests must pass before merging.
- **Feature Branches (`feature/<name>`):** New dashboard views, API routers, or visualization components.
- **Bugfix Branches (`fix/<issue>`):** Bug fixes and operational patches.
- **Documentation Branches (`docs/<topic>`):** Architecture guides, reports, and README updates.

### Commit Conventions
Follow Conventional Commits:
- `feat: add interactive track uncertainty cone layer`
- `fix: resolve CORS header mismatch in Render production deployment`
- `docs: update Supabase Storage bucket policy specifications`
- `chore: update runtime artifact manifest hashes`

---

## 3. Pull Request Checklist

Before submitting a pull request, run the local verification suite:

```bash
# 1. Verify runtime artifact presence and integrity
python scripts/setup/download_runtime_artifacts.py --check

# 2. Run Python audit test suite (Uncertainty, Verification, Analogs, Grad-CAM)
pytest -v scripts/audit/

# 3. Check frontend TypeScript and production build
npm run build --prefix apps/frontend

# 4. Check for unstaged large files or secrets
git status
```
