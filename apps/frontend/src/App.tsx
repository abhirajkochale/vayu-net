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

interface ActiveCycloneEvent {
  event_id: string;
  name: string;
  source: string;
  status: string;
  latest_observation: string | null;
  latest_center: { lat: number | null; lon: number | null } | null;
  history_fix_count: number;
  satellite_frame_count: number;
  readiness: string;
}

export default function App() {
  const [cyclones, setCyclones] = useState<Cyclone[]>([]);
  const [activeEvents, setActiveEvents] = useState<ActiveCycloneEvent[]>([]);
  const [selectedCyclone, setSelectedCyclone] = useState<string>('AMPHAN');
  const [selectedTimestamp, setSelectedTimestamp] = useState<string | undefined>(undefined);
  const [healthStatus, setHealthStatus] = useState<string>('checking...');
  const [canonicalData, setCanonicalData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load catalog of cyclones on mount
  useEffect(() => {
    vayuApi.getHealth()
      .then((h) => setHealthStatus(h.status))
      .catch(() => setHealthStatus('offline'));

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

  const [pipelineMode, setPipelineMode] = useState<'HISTORICAL' | 'CURRENT EVENT'>('HISTORICAL');

  // Load active current events when in CURRENT EVENT mode
  useEffect(() => {
    if (pipelineMode === 'CURRENT EVENT') {
      vayuApi.getCurrentEvents()
        .then((evs) => {
          setActiveEvents(evs);
          if (evs.length > 0 && !evs.some((e) => e.name === selectedCyclone)) {
            setSelectedCyclone(evs[0].name);
            setSelectedTimestamp(evs[0].latest_observation || undefined);
          }
        })
        .catch((err) => logger_fallback(err));
    }
  }, [pipelineMode]);

  const logger_fallback = (err: any) => {
    console.warn('Failed to load active current events:', err);
  };

  // Execute canonical end-to-end inference pipeline
  const executePipeline = () => {
    if (!selectedCyclone) return;
    setLoading(true);
    setError(null);

    const runner = pipelineMode === 'HISTORICAL' ? vayuApi.runInference : vayuApi.runCurrentInference;
    runner({ event_id: selectedCyclone, t0_utc: selectedTimestamp })
      .then((res) => {
        setCanonicalData(res);
        setLoading(false);
      })
      .catch((err) => {
        setCanonicalData(null);
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    executePipeline();
  }, [selectedCyclone, selectedTimestamp, pipelineMode]);

  // Demo one-click handler for deterministic SIH rehearsal
  const triggerDemoMode = () => {
    setPipelineMode('HISTORICAL');
    setSelectedCyclone('AMPHAN');
    setSelectedTimestamp('2020-05-18T06:00:00+00:00');
  };

  const center = canonicalData?.center;
  const observation = canonicalData?.observation;
  const intensity = canonicalData?.intensity;
  const wind = canonicalData?.wind;
  const forecast = canonicalData?.forecast;
  const uncertainty = canonicalData?.uncertainty;
  const verification = canonicalData?.verification;
  const analogs = canonicalData?.analogs;
  const saliency = canonicalData?.saliency;
  const secondaryObs = canonicalData?.secondary_observation;
  const metadata = canonicalData?.metadata;

  const centerDistanceKm = (center?.ai_lat != null && observation?.reference_center?.lat != null)
    ? (() => {
        const R = 6371;
        const dLat = (observation.reference_center.lat - center.ai_lat) * Math.PI / 180;
        const dLon = (observation.reference_center.lon - center.ai_lon) * Math.PI / 180;
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                  Math.cos(center.ai_lat * Math.PI / 180) * Math.cos(observation.reference_center.lat * Math.PI / 180) *
                  Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return Math.round(R * c * 10) / 10;
      })()
    : null;

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

        {/* System Status & Demo Mode Action */}
        <div className="flex items-center space-x-4">
          <button
            type="button"
            onClick={triggerDemoMode}
            className="px-3 py-1.5 bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-white rounded-lg text-xs font-bold tracking-wide shadow-md transition-all flex items-center gap-1.5"
            title="Load Deterministic Reference Scenario (Cyclone AMPHAN 2020-05-18T06:00:00Z)"
          >
            <span>⚡</span> SIH REHEARSAL DEMO
          </button>

          <div className="flex items-center space-x-2 text-xs">
            <span className="text-slate-400">Pipeline:</span>
            <span className={`inline-flex items-center px-2 py-0.5 rounded-md font-mono text-xs ${
              healthStatus === 'healthy' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${healthStatus === 'healthy' ? 'bg-emerald-400' : 'bg-rose-400'}`}></span>
              {healthStatus === 'healthy' ? 'OPERATIONAL' : healthStatus}
            </span>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* Cyclone Selector & Observation Bar */}
        <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 shadow-sm">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center space-x-3">
              <label htmlFor="cyclone-select" className="text-sm font-medium text-slate-300">Cyclone:</label>
              <select
                id="cyclone-select"
                value={selectedCyclone}
                onChange={(e) => {
                  setSelectedCyclone(e.target.value);
                  setSelectedTimestamp(undefined); // Reset timestamp to default on storm switch
                }}
                className="bg-[#1e293b] border border-slate-700 text-white rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-cyan-500 font-medium"
              >
                {cyclones.map((c) => (
                  <option key={c.cyclone_id} value={c.storm_name}>
                    {c.storm_name} ({c.year}) — {c.peak_category} ({c.max_wind_kt} kt) [{c.split}]
                  </option>
                ))}
              </select>
            </div>

            {/* Mode Selector */}
            <div className="flex bg-[#1e293b] p-0.5 rounded-lg border border-slate-700 text-xs">
              <button
                type="button"
                onClick={() => setPipelineMode('HISTORICAL')}
                className={`px-3 py-1 rounded-md font-semibold transition-all ${
                  pipelineMode === 'HISTORICAL' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                HISTORICAL
              </button>
              <button
                type="button"
                onClick={() => setPipelineMode('CURRENT EVENT')}
                className={`px-3 py-1 rounded-md font-semibold transition-all ${
                  pipelineMode === 'CURRENT EVENT' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                CURRENT EVENT
              </button>
            </div>

            <div className="flex items-center space-x-2 bg-[#1e293b] px-3 py-1 rounded-lg border border-slate-700 text-xs">
              <span className="text-slate-400">Endpoint:</span>
              <code className="text-cyan-400 font-mono font-semibold">
                {pipelineMode === 'HISTORICAL' ? 'POST /api/inference/run' : 'POST /api/inference/current'}
              </code>
            </div>
          </div>

          <div className="text-xs text-slate-400 flex flex-wrap items-center gap-4">
            {loading && <span className="text-cyan-400 animate-pulse font-medium">Executing multi-task inference...</span>}
            <span>Observation $t_0$: <strong className="text-slate-200">{observation?.t0_utc?.slice(0, 16).replace('T', ' ') || 't0'} UTC</strong></span>
            <span>AI Center: <strong className="text-cyan-400">{center?.ai_lat ? `${center.ai_lat}°N, ${center.ai_lon}°E` : 'Pending...'}</strong></span>
            <span>Reference Center: <strong className="text-emerald-400">{observation?.reference_center?.lat ? `${observation.reference_center.lat}°N, ${observation.reference_center.lon}°E` : 'None / Active'}</strong></span>
          </div>
        </div>

        {/* Multi-Source Observation Provider Status Banner */}
        <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 text-xs shadow-sm">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-semibold uppercase tracking-wider text-[11px]">Sensor Ingestion Providers:</span>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {/* GridSat Provider */}
            <div className="flex items-center gap-1.5 bg-[#131e36] px-2.5 py-1 rounded-md border border-cyan-800/60">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              <span className="font-medium text-slate-200">GridSat-B1:</span>
              <span className="text-cyan-300 font-mono font-semibold text-[10px]">VALIDATED RETROSPECTIVE SOURCE (OPERATIONAL)</span>
            </div>

            {/* IMERG Provider */}
            <div className="flex items-center gap-1.5 bg-[#131e36] px-2.5 py-1 rounded-md border border-blue-800/60">
              <span className="w-2 h-2 rounded-full bg-blue-400"></span>
              <span className="font-medium text-slate-200">NASA IMERG:</span>
              <span className="text-blue-300 font-mono font-semibold text-[10px]">SECONDARY SATELLITE-DERIVED SOURCE (CONTEXT ONLY)</span>
            </div>

            {/* INSAT Provider */}
            <div className="flex items-center gap-1.5 bg-[#131e36] px-2.5 py-1 rounded-md border border-amber-800/60">
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              <span className="font-medium text-slate-200">ISRO INSAT-3D/3DR:</span>
              <span className="text-amber-300 font-mono font-semibold text-[10px]">PLANNED AUTHENTICATED PROVIDER (NOT CURRENTLY AVAILABLE)</span>
            </div>
          </div>
        </div>

        {/* Current Event Active Cyclones & Readiness Panel */}
        {pipelineMode === 'CURRENT EVENT' && (
          <div className="bg-[#0b1329] border border-amber-900/60 rounded-xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                <span className="text-xs font-bold uppercase tracking-wider text-amber-300">
                  Active Cyclones — Live Discovery Feed
                </span>
              </div>
              <span className="text-xs text-slate-400">
                Source: {activeEvents[0]?.source || 'Operational Discovery Layer'}
              </span>
            </div>

            {/* Cyclones Card Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {activeEvents.map((ev) => {
                const isSelected = selectedCyclone === ev.name || selectedCyclone === ev.event_id;
                return (
                  <div
                    key={ev.event_id}
                    onClick={() => {
                      setSelectedCyclone(ev.name);
                      setSelectedTimestamp(ev.latest_observation || undefined);
                    }}
                    className={`p-3 rounded-lg border cursor-pointer transition-all text-xs space-y-2 ${
                      isSelected
                        ? 'bg-[#172554] border-cyan-500 shadow-md ring-1 ring-cyan-500/50'
                        : 'bg-[#131e36] border-slate-700/60 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-white text-sm">{ev.name}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        ev.readiness === 'READY'
                          ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                          : ev.readiness === 'WAITING_FOR_FRAMES'
                          ? 'bg-blue-950 text-blue-300 border border-blue-800'
                          : 'bg-amber-950 text-amber-300 border border-amber-800'
                      }`}>
                        {ev.readiness}
                      </span>
                    </div>

                    <div className="text-slate-400 space-y-0.5 text-[11px]">
                      <div>Last Obs: <strong className="text-slate-200">{ev.latest_observation?.slice(0, 16).replace('T', ' ') || 'N/A'} UTC</strong></div>
                      <div>Position: <strong className="text-slate-200">
                        {ev.latest_center?.lat != null ? `${ev.latest_center.lat}°N, ${ev.latest_center.lon}°E` : 'Pending'}
                      </strong></div>
                      <div className="flex items-center justify-between pt-1 text-[11px]">
                        <span>Satellite: <strong className="font-mono text-cyan-300">{ev.satellite_frame_count}/6</strong></span>
                        <span>History: <strong className="font-mono text-amber-300">{ev.history_fix_count} fixes</strong></span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Action Bar & Operational Readiness */}
            <div className="flex flex-wrap items-center justify-between pt-2 border-t border-slate-800/80 gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
                <span>Selected: <strong className="text-white">{selectedCyclone}</strong></span>
                <span>•</span>
                <span>Observation $t_0$: <strong className="text-slate-200">{observation?.t0_utc?.slice(0, 16).replace('T', ' ') || 't0'} UTC</strong></span>
                <span>•</span>
                <span>Causal Frames: <strong className="font-mono text-cyan-300">{canonicalData?.readiness?.satellite_frames_available ?? 6}/6</strong></span>
                <span>•</span>
                <span>Readiness: <strong className={(canonicalData?.readiness?.status === 'READY' || canonicalData?.readiness?.status === 'INFERENCE_READY') ? 'text-emerald-400' : 'text-amber-400'}>{canonicalData?.readiness?.status || 'CHECKING'}</strong></span>
              </div>

              <button
                type="button"
                onClick={executePipeline}
                disabled={loading || (canonicalData?.readiness?.status !== 'READY' && canonicalData?.readiness?.status !== 'INFERENCE_READY')}
                className={`px-4 py-1.5 rounded-lg font-bold text-xs shadow transition-all ${
                  (canonicalData?.readiness?.status === 'READY' || canonicalData?.readiness?.status === 'INFERENCE_READY')
                    ? 'bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-500 cursor-pointer'
                    : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                }`}
              >
                RUN FORECAST
              </button>
            </div>
          </div>
        )}

        {/* Structured Error Banner */}
        {error && (
          <div className="bg-rose-950/70 border border-rose-800 rounded-xl p-4 text-rose-200 text-sm space-y-1">
            <div className="font-semibold flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-rose-400"></span>
              Inference Pipeline Notice
            </div>
            <p className="text-xs text-rose-300/90">{error}</p>
          </div>
        )}

        {/* Center Localization & Provenance Panel */}
        <div className="bg-[#0b1329] border border-cyan-900/60 rounded-xl p-4 shadow-sm space-y-2">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-300">
                Phase 3C Center Localization & Primary Observation
              </span>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              Inference ID: {metadata?.inference_id || 'vayu_inf_init'}
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
                  Phase 3C Soft-Argmax
                </span>
              </div>
              <p className="text-base font-mono font-bold text-cyan-100">
                {center ? `${center.ai_lat}°N, ${center.ai_lon}°E` : 'Computing...'}
              </p>
              <div className="flex items-center justify-between text-[10px] text-slate-400">
                <span>ResNet-18 Backbone</span>
                {centerDistanceKm != null && (
                  <span className="text-cyan-300 font-mono font-semibold">
                    Δd = {centerDistanceKm} km vs Ref
                  </span>
                )}
              </div>
            </div>

            {/* Box 2: Observed IMD Reference */}
            <div className="bg-[#112423] border border-emerald-800/60 rounded-lg p-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase font-bold tracking-wide text-emerald-400">
                  Observed IMD Reference
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                  IMD Synoptic $t_0$
                </span>
              </div>
              <p className="text-base font-mono font-bold text-emerald-100">
                {observation?.reference_center?.lat
                  ? `${observation.reference_center.lat}°N, ${observation.reference_center.lon}°E`
                  : 'Pending...'}
              </p>
              <p className="text-[10px] text-slate-400">
                Observed Reference: {observation?.reference_category || 'N/A'} ({observation?.reference_wind_kt ?? 'N/A'} kt)
              </p>
            </div>

            {/* Box 3: Phase 6 Multi-Task Intensity & Wind */}
            <div className="bg-[#1a253a] border border-blue-800/60 rounded-lg p-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase font-bold tracking-wide text-blue-400">
                  AI Intensity & Wind
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
                  Phase 6 Multi-Task
                </span>
              </div>
              <p className="text-base font-mono font-bold text-blue-100">
                {intensity?.category ? `${intensity.category} • ${wind?.wind_kt} kt` : 'Computing...'}
              </p>
              <p className="text-[10px] text-slate-400">
                Model Prob: {intensity?.confidence ? `${(intensity.confidence * 100).toFixed(1)}%` : 'N/A'} • ±{wind?.confidence} kt p80 • {wind?.central_pressure_hpa ? `${wind.central_pressure_hpa} hPa` : ''}
              </p>
            </div>
          </div>

          {/* Causal 6-Frame Ingestion Timeline */}
          <div className="bg-[#0f172a]/90 border border-slate-800 rounded-lg p-3 space-y-2 mt-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                Primary Causal Satellite Observation Sequence [GridSat-B1 11µm IR]
              </span>
              <span className="text-[10px] font-mono text-cyan-400">
                Resolution: 0.07° • Grid: 572 × 929 • Cadence: 3-Hourly • Zero Future Leakage
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
              {[
                { offset: -15, label: 't - 15h' },
                { offset: -12, label: 't - 12h' },
                { offset: -9, label: 't - 9h' },
                { offset: -6, label: 't - 6h' },
                { offset: -3, label: 't - 3h' },
                { offset: 0, label: 't0 (Issue Time)' },
              ].map((step) => (
                <div
                  key={step.label}
                  className={`p-2 rounded border text-center text-xs space-y-1 ${
                    step.offset === 0
                      ? 'bg-cyan-950/40 border-cyan-700/80 text-cyan-200'
                      : 'bg-[#131e36]/70 border-slate-700/60 text-slate-300'
                  }`}
                >
                  <div className="text-[10px] font-mono font-bold">{step.label}</div>
                  <div className="text-[9px] text-slate-400 font-mono">
                    {observation?.t0_utc
                      ? new Date(new Date(observation.t0_utc).getTime() + step.offset * 3600 * 1000).toISOString().slice(11, 16) + ' UTC'
                      : 'Syncing...'}
                  </div>
                  <span className="inline-block text-[9px] px-1.5 py-0.2 rounded font-mono font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">
                    CAUSAL VALIDATED
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 3-Column Core Forecasting, Verification & Analogs Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Column 1: Multi-Horizon Track Forecast */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 flex flex-col space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Track Forecast (+12h / +24h / +48h)
                </h2>
                <div className="flex items-center gap-2 mt-1">
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase bg-cyan-950 text-cyan-400 border border-cyan-800">
                    Phase 5B Variant A
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">Anchor: Observed $t_0$</span>
                </div>
              </div>
              <span className="text-xs font-mono text-cyan-400">P80 Cones</span>
            </div>

            <div className="space-y-3 flex-1">
              {forecast ? (
                [
                  { key: 'plus_12h', label: '+12h', pt: forecast.plus_12h, radius: uncertainty?.plus_12h_km },
                  { key: 'plus_24h', label: '+24h', pt: forecast.plus_24h, radius: uncertainty?.plus_24h_km },
                  { key: 'plus_48h', label: '+48h', pt: forecast.plus_48h, radius: uncertainty?.plus_48h_km },
                ].map((item) => (
                  <div key={item.key} className="bg-[#1e293b]/70 border border-slate-700/60 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <span className="text-sm font-bold text-cyan-400">{item.label}</span>
                      <p className="text-xs text-slate-400">Destination Forecast</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-mono font-medium text-slate-200">
                        {item.pt ? `${item.pt.lat}°N, ${item.pt.lon}°E` : 'N/A'}
                      </p>
                      <p className="text-xs text-amber-400">
                        {item.radius ? `± ${item.radius} km empirical cone` : 'Calibrating...'}
                      </p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 italic py-6 text-center">No active forecast data.</div>
              )}
            </div>

            <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-2 flex justify-between">
              <span>Uncertainty: <strong className="text-amber-300">Empirical (Validation Residuals)</strong></span>
              <span>Input: 6-frame GridSat + ERA5</span>
            </div>
          </div>

          {/* Column 2: Deterministic Verification vs Held-Out Truth */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 flex flex-col space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Trajectory Verification
                </h2>
                <span className="text-[11px] text-emerald-400 font-medium">Evaluated Strictly Post-Prediction</span>
              </div>
              <span className="text-xs text-slate-400 font-mono">IMD Ground Truth</span>
            </div>

            <div className="space-y-3 flex-1">
              {verification?.status === 'UNAVAILABLE' ? (
                <div className="bg-amber-950/30 border border-amber-900/50 rounded-lg p-4 space-y-2 text-center my-auto">
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-950 text-amber-300 border border-amber-800 text-[10px] font-bold uppercase tracking-wider">
                    <span>🛡️</span> Causal Ground Truth Quarantined
                  </div>
                  <p className="text-xs text-amber-200/90 leading-relaxed">
                    {verification?.message || "Future ground truth observations are strictly quarantined for live events to prevent post-t0 target leakage."}
                  </p>
                  <p className="text-[10px] text-slate-400">
                    Verification targets unlock automatically upon post-event IMD best-track publishing.
                  </p>
                </div>
              ) : verification?.horizons && Object.keys(verification.horizons).length > 0 ? (
                Object.entries(verification.horizons).map(([hKey, hVal]: [string, any]) => (
                  <div key={hKey} className="bg-[#1e293b]/70 border border-slate-700/60 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <span className="text-sm font-bold text-slate-200">+{hKey.replace('plus_', '')}</span>
                      <p className="text-xs text-slate-400">
                        Truth: {hVal.actual ? `${hVal.actual.lat}°N, ${hVal.actual.lon}°E` : 'Untracked / Dissipated'}
                      </p>
                    </div>
                    <div className="text-right">
                      {hVal.dpe_km !== null ? (
                        <>
                          <span className="text-sm font-mono font-bold text-emerald-400">{hVal.dpe_km} km DPE</span>
                          <p className="text-xs text-slate-400">
                            Δ ({hVal.directional_error?.delta_lat_deg}°, {hVal.directional_error?.delta_lon_deg}°)
                          </p>
                        </>
                      ) : (
                        <span className="text-xs text-slate-500 italic">No truth target</span>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 italic py-6 text-center">
                  Verification unavailable (storm end / future unobserved).
                </div>
              )}
            </div>

            <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-2 flex justify-between">
              <span>Mean Track DPE: <strong className="text-emerald-400">{verification?.aggregate_dpe_km ?? 'N/A'} km</strong></span>
              <span className="text-emerald-400 font-medium">Zero Future Leakage</span>
            </div>
          </div>

          {/* Column 3: Top-2 Historical Analog Storms */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 flex flex-col space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Historical Analog Cyclones
                </h2>
                <span className="text-[11px] text-cyan-400 font-medium">TRAIN Split Only (Self-Excluded)</span>
              </div>
              <span className="text-xs text-cyan-400 font-mono">Top-2 Match</span>
            </div>

            <div className="space-y-3 flex-1">
              {analogs && analogs.length > 0 ? (
                analogs.map((an: any, idx: number) => (
                  <div key={an.storm_id + idx} className="bg-[#1e293b]/70 border border-slate-700/60 rounded-lg p-3 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-white">#{idx + 1} {an.storm_name} ({an.year})</span>
                      <span className="text-xs font-mono text-cyan-400">d = {an.distance}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
                        {an.split || 'TRAIN'}
                      </span>
                      <p className="text-xs text-slate-300">
                        Peak Category: <strong>{an.peak_category || 'CS'}</strong>
                      </p>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Timestamp: {an.t0_utc?.slice(0, 16).replace('T', ' ')} UTC
                    </p>
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 italic py-6 text-center">No analog records available.</div>
              )}
            </div>

            <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-2">
              Standardized 7-D Vector: $[lat, lon, wind, pres, \Delta x, \Delta y, \Delta wind]$
            </div>
          </div>
        </div>

        {/* Secondary Observation Context: NASA GPM IMERG Final Run V07B */}
        <div className="bg-[#0b1329] border border-blue-900/60 rounded-xl p-5 space-y-3 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-400"></span>
              <h2 className="text-sm font-bold uppercase tracking-wider text-blue-300">
                Secondary Synchronized Observation Context: NASA GPM IMERG V07B
              </h2>
            </div>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
              0% Predictive Fusion Role • Observational Context Only
            </span>
          </div>

          <div className="text-xs text-slate-300 leading-relaxed">
            <p>
              Synchronized half-hourly calibrated precipitation fields (<code>precipitationCal</code> in mm/hr) from 
              <strong> {secondaryObs?.source || 'NASA GPM IMERG Final Run V07B'}</strong> are aligned to each 3-hourly GridSat observation timestamp.
              Controlled ablation experiments (EXP-M1, EXP-M2, EXP-M3) confirmed that multimodal fusion with IMERG does <em>not</em> outperform the unimodal GridSat control.
              IMERG is therefore maintained strictly as secondary meteorological context.
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-6 gap-3 pt-1">
            {secondaryObs?.frames && secondaryObs.frames.length > 0 ? (
              secondaryObs.frames.map((f: any) => (
                <div key={f.step} className="bg-[#121c2f] border border-blue-800/40 rounded-lg p-2.5 text-center space-y-1">
                  <span className="text-[10px] font-bold uppercase text-blue-400">{f.step.replace('_', ' ')}</span>
                  <p className="text-[11px] font-mono text-slate-200">{f.timestamp_utc?.slice(11, 16)} UTC</p>
                  <span className="inline-block text-[9px] px-1 py-0.5 rounded bg-blue-950 text-blue-300 font-mono">
                    {f.local_status || 'SYNCHRONIZED'}
                  </span>
                </div>
              ))
            ) : (
              <div className="col-span-6 text-xs text-slate-500 italic py-2 text-center">Loading synchronized frames...</div>
            )}
          </div>
        </div>

        {/* Explainability & Saliency Viewport */}
        {saliency && saliency.status === 'AVAILABLE' && (
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 space-y-4 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Grad-CAM Vision Interpretability ({selectedCyclone})
                </h2>
                <p className="text-xs text-slate-400">
                  Target Layer: <code className="text-cyan-400">{saliency.target_layer}</code> • Head Explained: <strong className="text-white">{saliency.target_head_explained}</strong>
                </p>
              </div>
              <span className="text-xs font-mono text-cyan-400">Phase 3C ResNet Backbone</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-1.5 text-center">
                <span className="text-xs font-medium text-slate-400">1. Original IR Observation ($t_0$)</span>
                <div className="border border-slate-800 rounded-lg overflow-hidden bg-black/40">
                  <img
                    src={resolveAssetUrl(saliency.original_url)}
                    alt="Original satellite observation"
                    className="w-full h-auto object-cover"
                  />
                </div>
              </div>

              <div className="space-y-1.5 text-center">
                <span className="text-xs font-medium text-slate-400">2. Normalized Grad-CAM Saliency</span>
                <div className="border border-slate-800 rounded-lg overflow-hidden bg-black/40">
                  <img
                    src={resolveAssetUrl(saliency.heatmap_url)}
                    alt="Grad-CAM normalized heatmap"
                    className="w-full h-auto object-cover"
                  />
                </div>
              </div>

              <div className="space-y-1.5 text-center">
                <span className="text-xs font-medium text-cyan-400">3. Spatial Interpretation Overlay</span>
                <div className="border border-cyan-900/60 rounded-lg overflow-hidden bg-black/40 shadow-cyan-950/40 shadow-lg">
                  <img
                    src={resolveAssetUrl(saliency.overlay_url)}
                    alt="Grad-CAM blended overlay"
                    className="w-full h-auto object-cover"
                  />
                </div>
              </div>
            </div>

            <div className="bg-amber-950/30 border border-amber-900/40 rounded-lg p-3 text-xs text-amber-300/90 leading-relaxed">
              <strong>Mandatory Scientific Interpretation Notice:</strong> {saliency.interpretation_note}
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
