# VAYU-NET — Render Deployment Guide (FastAPI Backend)

**Module:** DevOps & Cloud Deployment  
**Service:** Python 3.11+ / FastAPI / PyTorch  
**Deployment Target:** Render Web Service (`render.yaml`)  
**Deployment Status:** `MANUAL STEP — NOT DEPLOYED`

> [!NOTE]
> **Cloud Deployment Status:** Cloud deployment has not yet been performed. Manual deployment is pending.
> The local FastAPI backend and PyTorch model inference pipeline are **VERIFIED LOCALLY**. This guide provides the operational runbook for manual deployment.

---

## 1. Architecture & Service Overview

The VAYU-NET backend runs as a containerized web service on Render, providing REST endpoints for multi-horizon track forecasting, empirical uncertainty cones, historical analog storm retrieval, and Grad-CAM explainability assets.

| Setting | Value |
| :--- | :--- |
| **Service Type** | Web Service (`web`) |
| **Environment** | Python 3.11+ |
| **Root Directory** | `.` (Repository root) |
| **Build Command** | `pip install --upgrade pip && pip install -r apps/backend/requirements.txt && python scripts/setup/download_runtime_artifacts.py --check` |
| **Start Command** | `uvicorn apps.backend.main:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/health` |
| **Default RAM Plan** | Standard (2 GB RAM recommended for PyTorch CPU inference) |

---

## 2. Environment Variables Configuration

Set these variables in the Render Dashboard (**Environment** tab):

| Variable | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.8` | Enforces modern Python runtime |
| `ENVIRONMENT` | `production` | Flags production mode |
| `HOST` | `0.0.0.0` | Binds to all network interfaces |
| `CORS_ORIGINS` | `http://localhost:5173,https://vayu-net.vercel.app` | Allows frontend cross-origin requests |
| `SUPABASE_URL` | `https://<ref>.supabase.co` | Supabase API URL |
| `SUPABASE_SERVICE_ROLE_KEY` | *(Secret)* | Server-side Supabase credentials |
| `MODEL_STORAGE_BASE_URL` | `https://<ref>.supabase.co/storage/v1/object/public/vayu-net-runtime` | CDN URL for model download script |

---

## 3. Step-by-Step Render Deployment

1. **Connect GitHub Repository:**
   - In Render Dashboard, click **New +** $\to$ **Blueprint**.
   - Select `abhirajkochale/vayu-net`.
   - Render automatically detects `render.yaml`.
2. **Apply Service Configuration:**
   - Confirm service name `vayu-net-backend`.
   - Review build and start commands.
3. **Set Secrets:**
   - Provide `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in Render environment settings.
4. **Deploy:**
   - Trigger manual deploy or push to `main`.
   - Monitor build logs: `pip install` executes, runtime artifacts are checked, and Uvicorn launches on `0.0.0.0:$PORT`.
5. **Verify Health:**
   ```bash
   curl -s https://vayu-net-backend.onrender.com/health | jq .
   ```
   Expected response:
   ```json
   {
     "status": "healthy",
     "service": "vayu-net-backend",
     "version": "1.0.0",
     "environment": "production",
     "runtime_artifacts": {
       "best_center_localization_cnn.pt": true,
       "best_temporal_track_gru.pt": true,
       "best_phase5b_variant_b.pt": true,
       "best_phase6_intensity_wind.pt": true
     }
   }
   ```

---

## 4. Troubleshooting & Operational Runbook

- **Cold Starts (Free Tier):** If deployed on Render free instances, the first request after 15 minutes of inactivity may take 30–50 seconds to spin up. Standard paid instances eliminate cold starts.
- **Port Binding Errors:** Ensure `PORT` is never hardcoded. The application reads `os.getenv("PORT", 8000)`.
- **Missing Models:** If `/health` shows `runtime_artifacts` false, verify that `MODEL_STORAGE_BASE_URL` is accessible or that `artifacts/manifest.json` points to valid storage locations.
