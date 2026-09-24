# VAYU-NET — Vercel Deployment Guide (React + Vite Frontend)

**Module:** DevOps & Frontend Cloud Deployment  
**Framework:** React 18 / TypeScript / Vite / Tailwind CSS  
**Target:** Vercel Global Edge Network

---

## 1. Project Overview & Build Settings

The VAYU-NET frontend is an interactive single-page application (SPA) providing cyclone trajectory visualization, empirical uncertainty cone mapping, analog storm inspection, and Grad-CAM saliency explainability overlays.

### Vercel Project Settings

| Setting | Value |
| :--- | :--- |
| **Framework Preset** | Vite |
| **Root Directory** | `apps/frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Node.js Version** | `18.x` or `20.x` |

---

## 2. Environment Variables Configuration

Set these variables in the Vercel Project Dashboard (**Settings $\to$ Environment Variables**):

| Variable Name | Example Production Value | Purpose |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `https://vayu-net-backend.onrender.com` | Base URL of the deployed Render FastAPI backend |
| `VITE_SUPABASE_URL` | `https://<project-ref>.supabase.co` | Supabase API endpoint (optional) |
| `VITE_SUPABASE_ANON_KEY` | *(Public Anon Key)* | Supabase anonymous public token (NEVER use Service Role key!) |

> [!WARNING]
> **Zero Hardcoded URLs:**  
> All frontend network communication flows through `src/config/api.ts` using `import.meta.env.VITE_API_BASE_URL`. Do not hardcode localhost or backend IPs in UI components.

---

## 3. SPA Routing & Client-Side Rewrites

To prevent HTTP 404 errors when reloading browser routes on a single-page app, `apps/frontend/vercel.json` provides edge rewrites to `index.html`:

```json
{
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

---

## 4. Deployment Instructions

1. **Import Git Repository:**
   - Log into Vercel and click **Add New... $\to$ Project**.
   - Select `abhirajkochale/vayu-net`.
2. **Configure Root Directory:**
   - Click **Edit** next to Root Directory and set it to `apps/frontend`.
3. **Configure Environment Variables:**
   - Add `VITE_API_BASE_URL = https://vayu-net-backend.onrender.com`.
4. **Deploy:**
   - Click **Deploy**. Vercel will run `npm install`, compile TypeScript, build with Vite, and distribute the bundle to Vercel's global CDN.
5. **Post-Deployment Verification:**
   - Open `https://vayu-net.vercel.app`.
   - Verify that the backend status pill displays `healthy`.
   - Select reference cyclone **AMPHAN** and confirm that forecasts, analogs, and Grad-CAM overlays render correctly.
