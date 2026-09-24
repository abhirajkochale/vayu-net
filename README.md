<div align="center">

# 🌪️ VAYU-NET
### North Indian Ocean Tropical Cyclone Intelligence & Multi-Horizon Forecasting System

**Smart India Hackathon 2026 • Problem Statement SIH 26070**  
**Ministry of Earth Sciences (MoES) • India Meteorological Department (IMD)**

[![Backend Test Suite](https://img.shields.io/badge/Backend%20Tests-33%2F33%20Passed-emerald)](file:///c:/GitHub/vayu-net/scripts/audit/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-Proprietary%20MoES%2FIMD-amber)]()

</div>

---

## 📌 Problem Overview (SIH26070)

> *"To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data."*

The North Indian Ocean (Bay of Bengal and Arabian Sea) experiences extreme tropical cyclones with rapid intensification and erratic recurvature tracks. VAYU-NET is an operational deep learning system integrating:
1. **Single-Frame Dedicated Center Localization** (Phase 3C ResNet-18)
2. **Temporal Multi-Horizon Track Forecasting** (+12h, +24h, +48h hybrid kinematic + satellite residual model)
3. **Empirical Uncertainty Cones** (P80/P90 radii derived strictly from validation residuals)
4. **Multimodal IMD Intensity & Wind Estimation** (Joint 6-step Satellite GRU + ERA5 CNN-GRU)
5. **Historical Analog-Storm Retrieval** (7-D standardized nearest-neighbor retrieval)
6. **Grad-CAM Saliency Explainability** (Convolutional interpretability over raw satellite observations)

---

## 🏗️ System Architecture

```
                          ┌─────────────────────────────────────┐
                          │   Vercel Global Edge Network        │
                          │   React 18 + Vite + Tailwind SPA    │
                          └──────────────────┬──────────────────┘
                                             │ HTTPS
                                             ▼
                          ┌─────────────────────────────────────┐
                          │   Render Web Service (FastAPI)       │
                          │   Python 3.11+ / PyTorch Engine     │
                          └──────┬────────────────────────┬─────┘
                                 │                        │
               ┌─────────────────┴────────┐     ┌─────────┴────────────────┐
               ▼                          ▼     ▼                          ▼
     ┌───────────────────┐      ┌─────────────┐ ┌──────────────────┐ ┌─────────────┐
     │  Phase 3C Center  │      │  Phase 5B   │ │ Phase 6 Intensity│ │  Grad-CAM   │
     │  ResNet-18 CNN    │      │ Hybrid Track│ │ & Wind Multi-Task│ │ Saliency Tap│
     └───────────────────┘      └─────────────┘ └──────────────────┘ └─────────────┘
                                 ▲                        ▲
                                 │                        │
                          ┌──────┴────────────────────────┴─────┐
                          │   Supabase Cloud Platform           │
                          │   PostgreSQL + Storage Runtime      │
                          └─────────────────────────────────────┘
```

---

## ⚡ Quick Start for Developers

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/abhirajkochale/vayu-net.git
cd vayu-net

# Create virtual environment
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r apps/backend/requirements.txt
```

### 2. Verify Runtime Artifacts
VAYU-NET does **not** require downloading the 7.7+ GB raw research datasets to run the demo and operational inference:
```bash
# Verify presence and SHA-256 checksums of model weights & metadata
python scripts/setup/download_runtime_artifacts.py --check
```

### 3. Launch the FastAPI Backend
```bash
# Start backend server on http://localhost:8000
python -m uvicorn apps.backend.main:app --reload --port 8000
```
Visit the interactive OpenAPI docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**.

### 4. Launch the React Frontend
```bash
cd apps/frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 🧪 Running Audit Test Suites

All scientific invariants, anti-leakage constraints, and model architectures are backed by automated tests:
```bash
# Run complete audit test suite (Uncertainty, Verification, Analogs, Grad-CAM)
pytest -v scripts/audit/
```

---

## 📁 Repository Structure

```
vayu-net/
 ├── apps/
 │    ├── backend/              # FastAPI application, Pydantic schemas, endpoints
 │    │    ├── main.py
 │    │    ├── requirements.txt
 │    │    └── .env.example
 │    └── frontend/             # React 18, Vite, TypeScript, Tailwind dashboard
 │         ├── src/
 │         │    ├── config/api.ts
 │         │    ├── App.tsx
 │         │    └── main.tsx
 │         └── package.json
 ├── artifacts/
 │    ├── manifest.json         # SHA-256 checksums of production runtime weights
 │    └── README.md
 ├── configs/                   # System and runtime configurations
 ├── data/
 │    ├── manifests/            # Locked sample indices and cyclone catalogs
 │    ├── processed/            # Cleaned IMD best-track tables (v2)
 │    └── interim/ml/
 │         ├── explainability/  # Pre-rendered demo Grad-CAM overlays
 │         └── *.json           # Normalization, uncertainty, result summaries
 ├── docs/                      # Architectural specifications & audits
 ├── ml/                        # Core PyTorch models and forecast modules
 │    ├── explainability/       # Grad-CAM hooks and overlay rendering
 │    ├── forecast/             # Uncertainty, verification, analog retrieval
 │    └── models/               # Phase 3C, 4A, 4B, 5B, 6 architectures
 ├── scripts/
 │    ├── audit/                # Automated pytest test suites
 │    ├── production/           # Production generation pipelines
 │    └── setup/                # Artifact download and integrity verification
 ├── render.yaml                # Render Blueprint deployment specification
 ├── .gitignore                 # Strict separation of Git source from bulk data
 └── README.md
```

---

## 🛡️ Scientific Integrity & Anti-Leakage Invariants

1. **Zero Future Information:** The operational sequence strictly terminates at $t_0$. Future observations ($t_{+12\text{h}}$, $+24\text{h}$, $+48\text{h}$) are never accessed during inference.
2. **Observed vs Forecast Separation:** Verification endpoints clearly distinguish **OBSERVED ACTUAL** ground truth from **AI FORECAST** predictions to prevent misleading evaluation.
3. **Non-Causal Interpretability:** Grad-CAM attributions are explicitly documented as **interpretation aids**, never as causal meteorological claims.
4. **Data Lock:** All splits follow the event-level disjoint rule (`TRAIN`: 1998–2018, `VALIDATION`: 2019–2020, `TEST`: 2021–2024).

---

## 👥 Contributors & Documentation Links

- **[Development Setup Guide](docs/DEVELOPMENT_SETUP.md)**
- **[Runtime Artifacts Guide](docs/RUNTIME_ARTIFACTS.md)**
- **[REST API Documentation](docs/API.md)**
- **[Render Deployment Guide](docs/DEPLOY_RENDER.md)**
- **[Vercel Deployment Guide](docs/DEPLOY_VERCEL.md)**
- **[Supabase Storage Layout](docs/SUPABASE_STORAGE_LAYOUT.md)**
- **[Contributing Guidelines](CONTRIBUTING.md)**
