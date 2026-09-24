import { useEffect, useState } from 'react';
import { vayuApi, resolveAssetUrl } from './config/api';

interface Cyclone {
  cyclone_id: string;
  storm_name: string;
  year: number;
  split: string;
  peak_category: string;
  max_wind_kt: number;
}

export default function App() {
  const [cyclones, setCyclones] = useState<Cyclone[]>([]);
  const [selectedCyclone, setSelectedCyclone] = useState<string>('AMPHAN');
  const [forecastMode, setForecastMode] = useState<'MODEL_INFERENCE' | 'PRECOMPUTED_DEMO'>('MODEL_INFERENCE');
  const [healthStatus, setHealthStatus] = useState<string>('checking...');
  const [forecastData, setForecastData] = useState<any>(null);
  const [verificationData, setVerificationData] = useState<any>(null);
  const [analogsData, setAnalogsData] = useState<any>(null);
  const [explainabilityData, setExplainabilityData] = useState<any>(null);
  const [predictData, setPredictData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check backend health
    vayuApi.getHealth()
      .then((h) => setHealthStatus(h.status))
      .catch(() => setHealthStatus('offline'));

    // Fetch shortlist cyclones
    vayuApi.listCyclones(true)
      .then((data) => {
        setCyclones(data);
        if (data.length > 0) {
          const defaultStorm = data.find((c) => c.storm_name.includes('AMPHAN')) || data[0];
          setSelectedCyclone(defaultStorm.storm_name);
        }
      })
      .catch((err) => setError(`Failed to load cyclone catalog: ${err.message}`));
  }, []);

  useEffect(() => {
    if (!selectedCyclone) return;
    setLoading(true);
    setError(null);

    // Call predict endpoint to fetch consolidated forecast + real intensity prediction
    vayuApi.predict({ cyclone_id: selectedCyclone, percentile: 'p80', mode: forecastMode })
      .then((pred) => {
        setPredictData(pred);
        setForecastData(pred.forecast);
        setVerificationData(pred.verification);
        setAnalogsData(pred.analogs);
        setExplainabilityData(pred.explainability);
        setLoading(false);
      })
      .catch((err) => {
        // Clear forecast on error to avoid showing stale / mismatched data
        setForecastData(null);
        setVerificationData(null);
        setError(err.message);
        setLoading(false);
      });
  }, [selectedCyclone, forecastMode]);

  const predSource = forecastData?.prediction_source || predictData?.prediction_source;
  const isLiveInference = predSource === 'MODEL_INFERENCE';

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 bg-[#0d1424] px-6 py-4 flex items-center justify-between sticky top-0 z-50 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-black text-slate-900 text-lg shadow-cyan-500/20 shadow-lg">
            V
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              VAYU-NET
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800">
                SIH 26070
              </span>
            </h1>
            <p className="text-xs text-slate-400">North Indian Ocean Tropical Cyclone Intelligence System</p>
          </div>
        </div>

        {/* System Status & Provenance Header */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-xs">
            <span className="text-slate-400">Backend:</span>
            <span className={`inline-flex items-center px-2 py-0.5 rounded-md font-mono text-xs ${
              healthStatus === 'healthy' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${healthStatus === 'healthy' ? 'bg-emerald-400' : 'bg-rose-400'}`}></span>
              {healthStatus}
            </span>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* Cyclone Selector & Mode Bar */}
        <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 shadow-sm">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center space-x-3">
              <label htmlFor="cyclone-select" className="text-sm font-medium text-slate-300">Cyclone:</label>
              <select
                id="cyclone-select"
                value={selectedCyclone}
                onChange={(e) => setSelectedCyclone(e.target.value)}
                className="bg-[#1e293b] border border-slate-700 text-white rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-cyan-500 font-medium"
              >
                {cyclones.map((c) => (
                  <option key={c.cyclone_id} value={c.storm_name}>
                    {c.storm_name} ({c.year}) — {c.peak_category} ({c.max_wind_kt} kt) [{c.split}]
                  </option>
                ))}
              </select>
            </div>

            {/* Explicit Mode Selector */}
            <div className="flex items-center space-x-2 bg-[#1e293b] p-1 rounded-lg border border-slate-700">
              <button
                type="button"
                onClick={() => setForecastMode('MODEL_INFERENCE')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                  forecastMode === 'MODEL_INFERENCE'
                    ? 'bg-cyan-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                AI MODEL INFERENCE
              </button>
              <button
                type="button"
                onClick={() => setForecastMode('PRECOMPUTED_DEMO')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                  forecastMode === 'PRECOMPUTED_DEMO'
                    ? 'bg-amber-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                PRECOMPUTED DEMO
              </button>
            </div>
          </div>

          <div className="text-xs text-slate-400 flex flex-wrap items-center gap-4">
            {loading && <span className="text-cyan-400 animate-pulse font-medium">Computing inference...</span>}
            <span>Observed Issue: <strong className="text-slate-200">{forecastData?.issue_timestamp_utc || 't0'}</strong></span>
            <span>AI Detected Center: <strong className="text-cyan-400">{forecastData?.ai_detected_center ? `${forecastData.ai_detected_center.latitude}°N, ${forecastData.ai_detected_center.longitude}°E` : '...'}</strong></span>
            <span>Observed IMD Reference: <strong className="text-emerald-400">{forecastData?.observed_reference_center ? `${forecastData.observed_reference_center.latitude}°N, ${forecastData.observed_reference_center.longitude}°E` : (forecastData?.current_center ? `${forecastData.current_center[0]}°N, ${forecastData.current_center[1]}°E` : '...')}</strong></span>
          </div>
        </div>

        {/* Structured Error Banner */}
        {error && (
          <div className="bg-rose-950/70 border border-rose-800 rounded-xl p-4 text-rose-200 text-sm space-y-1">
            <div className="font-semibold flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-rose-400"></span>
              Prediction Unavailable
            </div>
            <p className="text-xs text-rose-300/90">{error}</p>
            {forecastMode === 'PRECOMPUTED_DEMO' && (
              <p className="text-xs text-amber-300 mt-2 font-medium">
                Tip: The selected cyclone may belong to the VALIDATION split. Switch mode to <strong>AI MODEL INFERENCE</strong> to execute the runtime model pipeline.
              </p>
            )}
          </div>
        )}

        {/* F03 Center Detection & Operational Anchor Provenance Panel */}
        <div className="bg-[#0b1329] border border-cyan-900/60 rounded-xl p-4 shadow-sm space-y-2">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-300">
                F03 Center Detection & Operational Anchor Provenance
              </span>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              Checkpoint: {forecastData?.checkpoint_identity || 'best_phase5b_variant_a.pt'}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
            {/* Box 1: AI Detected Center */}
            <div className="bg-[#131e36] border border-cyan-800/60 rounded-lg p-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase font-bold tracking-wide text-cyan-400">
                  AI Detected Center
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  {forecastData?.center_detection_source || 'MODEL_INFERENCE'}
                </span>
              </div>
              <p className="text-base font-mono font-bold text-cyan-100">
                {forecastData?.ai_detected_center
                  ? `${forecastData.ai_detected_center.latitude}°N, ${forecastData.ai_detected_center.longitude}°E`
                  : 'Pending...'}
              </p>
              <p className="text-[10px] text-slate-400">
                Phase 3C Localization CNN (best_center_localization_cnn.pt)
              </p>
            </div>

            {/* Box 2: Observed IMD Reference */}
            <div className="bg-[#112423] border border-emerald-800/60 rounded-lg p-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase font-bold tracking-wide text-emerald-400">
                  Observed IMD Reference
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                  IMD $t_0$
                </span>
              </div>
              <p className="text-base font-mono font-bold text-emerald-100">
                {forecastData?.observed_reference_center
                  ? `${forecastData.observed_reference_center.latitude}°N, ${forecastData.observed_reference_center.longitude}°E`
                  : forecastData?.current_center
                  ? `${forecastData.current_center[0]}°N, ${forecastData.current_center[1]}°E`
                  : 'Pending...'}
              </p>
              <p className="text-[10px] text-slate-400">
                Official IMD Synoptic Best-Track Analysis at $t_0$
              </p>
            </div>

            {/* Box 3: Forecast Initialized From */}
            <div className="bg-[#241c14] border border-amber-800/60 rounded-lg p-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase font-bold tracking-wide text-amber-400">
                  Forecast Initialized From
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800">
                  HYBRID ANCHOR
                </span>
              </div>
              <p className="text-sm font-semibold text-amber-100 pt-0.5">
                Observed IMD t0 reference
              </p>
              <p className="text-[10px] text-slate-400">
                Phase 5B Variant A Forecaster-in-the-Loop Architecture
              </p>
            </div>
          </div>
        </div>

        {/* Phase 6 Intensity / Wind Disclaimer Banner */}
        <div className="bg-[#121c2f] border border-blue-900/40 rounded-xl p-3 flex items-start gap-3 text-xs text-blue-200/90">
          <div className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold flex-shrink-0 mt-0.5">
            i
          </div>
          <div className="space-y-0.5">
            <span className="font-semibold text-white">Scientific & Operational Notice:</span>
            <p className="text-slate-300 leading-relaxed">
              Track forecasts are generated using the Phase 5B hybrid residual model (observed center anchor).
              Intensity and wind speed estimates are produced by the Phase 6 multi-task neural network (validation macro F1: 0.2006, accuracy: 25.10%, wind MAE: 25.44 kt).
              These neural predictions are experimental guidance and are <strong>not</strong> a certified operational replacement for official IMD Dvorak / ADT bulletins.
            </p>
          </div>
        </div>

        {/* Forecast & Verification Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Card 1: Multi-Horizon Forecast */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 flex flex-col space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Track Forecast
                </h2>
                <div className="flex items-center gap-2 mt-1">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase ${
                    isLiveInference
                      ? 'bg-cyan-950 text-cyan-400 border border-cyan-800'
                      : 'bg-amber-950 text-amber-400 border border-amber-800'
                  }`}>
                    <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${isLiveInference ? 'bg-cyan-400 animate-pulse' : 'bg-amber-400'}`}></span>
                    {forecastData?.prediction_source || forecastMode}
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">{forecastData?.model_version}</span>
                </div>
              </div>
              <span className="text-xs font-mono text-cyan-400">P80 Cones</span>
            </div>

            <div className="space-y-3 flex-1">
              {forecastData?.forecast_horizons ? (
                forecastData.forecast_horizons.map((h: any) => (
                  <div key={h.horizon} className="bg-[#1e293b]/70 border border-slate-700/60 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <span className="text-sm font-bold text-cyan-400">{h.horizon}</span>
                      <p className="text-xs text-slate-400">{h.target_timestamp_utc?.slice(0, 16).replace('T', ' ')} UTC</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-mono font-medium text-slate-200">{h.latitude}°N, {h.longitude}°E</p>
                      <p className="text-xs text-amber-400">± {h.empirical_uncertainty?.radius_km} km cone</p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 italic py-6 text-center">No active forecast data.</div>
              )}
            </div>

            {/* Intensity Mini-Panel if available */}
            {predictData?.intensity_prediction && !predictData.intensity_prediction.error && (
              <div className="bg-[#1a253a] border border-cyan-900/40 rounded-lg p-3 space-y-1 text-xs">
                <div className="flex justify-between items-center text-slate-300">
                  <span>AI Predicted Category:</span>
                  <strong className="text-cyan-400 font-bold">{predictData.intensity_prediction.predicted_category}</strong>
                </div>
                <div className="flex justify-between items-center text-slate-300">
                  <span>AI Predicted Sustained Wind:</span>
                  <strong className="text-slate-100">{predictData.intensity_prediction.predicted_wind_kt} kt (± {predictData.intensity_prediction.wind_uncertainty_p80_kt} kt)</strong>
                </div>
              </div>
            )}

            <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-2 flex justify-between">
              <span>Forecast Anchor: <strong className="text-amber-300">Observed IMD t0 reference</strong></span>
              <span>Data: GridSat-B1 + ERA5</span>
            </div>
          </div>

          {/* Card 2: Verification vs IMD Held-Out Truth */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 flex flex-col space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Deterministic Verification
                </h2>
                <span className="text-[11px] text-emerald-400 font-medium">Evaluated Strictly Post-Prediction</span>
              </div>
              <span className="text-xs text-slate-400 font-mono">Held-Out Truth</span>
            </div>

            <div className="space-y-3 flex-1">
              {verificationData?.horizons ? (
                Object.entries(verificationData.horizons).map(([hKey, hVal]: [string, any]) => (
                  <div key={hKey} className="bg-[#1e293b]/70 border border-slate-700/60 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <span className="text-sm font-bold text-slate-200">+{hKey}</span>
                      <p className="text-xs text-slate-400">Truth: {hVal.actual.latitude}°N, {hVal.actual.longitude}°E</p>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-mono font-bold text-emerald-400">{hVal.error.dpe_km} km DPE</span>
                      <p className="text-xs text-slate-400">Δ ({hVal.error.delta_lat_deg}°, {hVal.error.delta_lon_deg}°)</p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 italic py-6 text-center">No verification records available.</div>
              )}
            </div>

            <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-2 flex justify-between">
              <span>Mean Track DPE: <strong className="text-emerald-400">{verificationData?.summary?.mean_dpe_km ?? 'N/A'} km</strong></span>
              <span className="text-emerald-400 font-medium">Zero Ground-Truth Leakage</span>
            </div>
          </div>

          {/* Card 3: Top-2 Historical Analogs */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 flex flex-col space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Historical Analog Storms
                </h2>
                <span className="text-[11px] text-cyan-400 font-medium">TRAIN Split Only (No Leakage)</span>
              </div>
              <span className="text-xs text-cyan-400 font-mono">7-D Space</span>
            </div>

            <div className="space-y-3 flex-1">
              {analogsData?.analogs ? (
                analogsData.analogs.map((an: any) => (
                  <div key={an.sample_id} className="bg-[#1e293b]/70 border border-slate-700/60 rounded-lg p-3 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-white">#{an.rank} {an.storm_name} ({an.year})</span>
                      <span className="text-xs font-mono text-cyan-400">d = {an.standardized_distance}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
                        {an.candidate_split || 'TRAIN'}
                      </span>
                      <p className="text-xs text-slate-300">
                        Category: <strong>{an.matched_snapshot.category}</strong> ({an.matched_snapshot.wind_kt} kt) • {an.matched_snapshot.pressure_hpa} hPa
                      </p>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Timestamp: {an.matched_snapshot.timestamp_utc?.slice(0, 16).replace('T', ' ')} UTC • [{an.matched_snapshot.latitude}°N, {an.matched_snapshot.longitude}°E]
                    </p>
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 italic py-6 text-center">No analog records available.</div>
              )}
            </div>

            <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-2">
              Provenance: 696 candidate snapshots strictly from 81 historical TRAIN storms (1998–2018).
            </div>
          </div>
        </div>

        {/* Explainability & Saliency Viewport */}
        {explainabilityData && (
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 space-y-4 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Grad-CAM Vision Interpretability ({explainabilityData.storm_name})
                </h2>
                <p className="text-xs text-slate-400">
                  Target Layer: <code className="text-cyan-400">Phase 3C spatial_encoder.layer2</code> • Predicted Category: <strong className="text-white">{explainabilityData.predicted_class}</strong> ({(explainabilityData.confidence * 100).toFixed(1)}% conf)
                </p>
              </div>
              <div className="text-xs text-slate-400 text-right">
                <span>Centroid: ({explainabilityData.activation_centroid?.[0]?.toFixed(1)}, {explainabilityData.activation_centroid?.[1]?.toFixed(1)})</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-1.5 text-center">
                <span className="text-xs font-medium text-slate-400">1. Original IR Observation ($t_0$)</span>
                <div className="border border-slate-800 rounded-lg overflow-hidden bg-black/40">
                  <img
                    src={resolveAssetUrl(explainabilityData.original_url || explainabilityData.original_path)}
                    alt="Original satellite observation"
                    className="w-full h-auto object-cover"
                  />
                </div>
              </div>

              <div className="space-y-1.5 text-center">
                <span className="text-xs font-medium text-slate-400">2. Normalized Grad-CAM Saliency</span>
                <div className="border border-slate-800 rounded-lg overflow-hidden bg-black/40">
                  <img
                    src={resolveAssetUrl(explainabilityData.heatmap_url || explainabilityData.heatmap_path)}
                    alt="Grad-CAM normalized heatmap"
                    className="w-full h-auto object-cover"
                  />
                </div>
              </div>

              <div className="space-y-1.5 text-center">
                <span className="text-xs font-medium text-cyan-400">3. Spatial Interpretation Overlay</span>
                <div className="border border-cyan-900/60 rounded-lg overflow-hidden bg-black/40 shadow-cyan-950/40 shadow-lg">
                  <img
                    src={resolveAssetUrl(explainabilityData.overlay_url || explainabilityData.overlay_path)}
                    alt="Grad-CAM blended overlay"
                    className="w-full h-auto object-cover"
                  />
                </div>
              </div>
            </div>

            <div className="bg-amber-950/30 border border-amber-900/40 rounded-lg p-3 text-xs text-amber-300/90 leading-relaxed">
              <strong>Mandatory Scientific Interpretation Notice:</strong> Saliency maps visualize intermediate feature activations within the Phase 3C localization encoder (<code className="text-cyan-400">{explainabilityData.target_layer}</code>). This is an interpretability diagnostic to inspect deep spatial features, <strong>not</strong> a causal physical or meteorological explanation of cyclone atmospheric dynamics.
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-[#070b14] px-6 py-4 text-center text-xs text-slate-400">
        <p>VAYU-NET AI/ML Cyclone Intelligence System • Smart India Hackathon (SIH 2026 PS 26070)</p>
        <p className="mt-1 text-[11px] text-slate-400">Ministry of Earth Sciences (MoES) • India Meteorological Department (IMD)</p>
      </footer>
    </div>
  );
}
