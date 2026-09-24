# VAYU-NET — Developer Environment Setup Guide

**Module:** Team Onboarding & Local Development  
**System:** VAYU-NET Cyclone Intelligence System  
**Baseline:** SIH 2026 Problem Statement 26070

---

## 1. System Requirements & Prerequisites

Before starting, ensure your system has the following tools installed:

- **Git:** $\ge 2.30$
- **Python:** $\ge 3.10$ (Python 3.11 recommended)
- **Node.js:** $\ge 18.0$ (Node 20 LTS recommended) & `npm` $\ge 9.0$
- **OS:** Linux, macOS, or Windows 10/11 (with PowerShell)

---

## 2. Step-by-Step Local Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/abhirajkochale/vayu-net.git
cd vayu-net
```

### Step 2: Set Up Python Backend Environment
```bash
# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install production and development dependencies
pip install --upgrade pip
pip install -r apps/backend/requirements.txt
pip install pytest
```

### Step 3: Verify Runtime Model Weights
You do **not** need the full 7.7+ GB satellite research dataset to run VAYU-NET! Check that required model weights and metadata are available:
```bash
python scripts/setup/download_runtime_artifacts.py --check
```
If any model checkpoints are missing, download them from the team artifact storage:
```bash
python scripts/setup/download_runtime_artifacts.py
```

### Step 4: Configure Backend Environment Variables
```bash
# Copy template to active .env file
cp apps/backend/.env.example apps/backend/.env
```
Default values work out-of-the-box for local development.

### Step 5: Start the FastAPI Backend
```bash
# Start server on http://localhost:8000
python -m uvicorn apps.backend.main:app --reload --port 8000
```
Verify the server is running by opening:
- API Health Check: **[http://localhost:8000/health](http://localhost:8000/health)**
- Interactive Swagger UI: **[http://localhost:8000/docs](http://localhost:8000/docs)**

### Step 6: Set Up and Start React Frontend
In a new terminal window:
```bash
cd apps/frontend

# Copy environment template
cp .env.example .env

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 3. Running Automated Tests

Run the full audit test suite to verify model loading, uncertainty cones, analog retrieval, and Grad-CAM hooks:

```bash
# Run all audit test suites
pytest -v scripts/audit/
```
Expected output: **33 / 33 passed**.

---

## 4. Building for Production

Test production compilation before submitting changes:

```bash
# Frontend production build
npm run build --prefix apps/frontend

# Backend syntax & smoke test
python -c "from apps.backend.main import app; print('Backend import successful')"
```
