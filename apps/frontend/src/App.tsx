import { useEffect, useState, useCallback } from 'react';
import { vayuApi } from './config/api';
import { Header } from './components/Header';
import { CycloneSummary } from './components/CycloneSummary';
import { CycloneMap } from './components/CycloneMap';
import { CurrentConditions } from './components/CurrentConditions';
import { ForecastTimeline } from './components/ForecastTimeline';
import { ForecastAccuracy } from './components/ForecastAccuracy';
import { SimilarStorms } from './components/SimilarStorms';
import { AiInterpretability } from './components/AiInterpretability';
import { SatelliteSequence } from './components/SatelliteSequence';
import { RainfallContext } from './components/RainfallContext';
import { CurrentEventFeed, ActiveCycloneEvent } from './components/CurrentEventFeed';
import { DataSources } from './components/DataSources';
import { TechnicalDetailsModal } from './components/TechnicalDetailsModal';

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
  const [activeEvents, setActiveEvents] = useState<ActiveCycloneEvent[]>([]);
  const [selectedCyclone, setSelectedCyclone] = useState<string>('AMPHAN');
  const [selectedTimestamp, setSelectedTimestamp] = useState<string | undefined>(undefined);
  const [pipelineMode, setPipelineMode] = useState<'HISTORICAL' | 'CURRENT / REPLAY'>('HISTORICAL');
  const [healthStatus, setHealthStatus] = useState<string>('checking...');
  const [canonicalData, setCanonicalData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [technicalModalOpen, setTechnicalModalOpen] = useState<boolean>(false);

  // Initial catalog & health load
  useEffect(() => {
    vayuApi
      .getHealth()
      .then((h) => setHealthStatus(h.status))
      .catch(() => setHealthStatus('offline'));

    vayuApi
      .listCyclones(true)
      .then((data) => {
        setCyclones(data);
        if (data.length > 0) {
          const defaultStorm = data.find((c) => c.storm_name.includes('AMPHAN')) || data[0];
          setSelectedCyclone(defaultStorm.storm_name);
        }
      })
      .catch((err) => setError(`Failed to load cyclone catalog: ${err.message}`));
  }, []);

  // Load active current events when in CURRENT / REPLAY mode
  useEffect(() => {
    if (pipelineMode === 'CURRENT / REPLAY') {
      vayuApi
        .getCurrentEvents()
        .then((evs) => {
          setActiveEvents(evs);
          const readyEvents = evs.filter(
            (e) =>
              e.readiness === 'READY' ||
              e.readiness === 'INFERENCE_READY' ||
              e.readiness === 'WAITING_FOR_FRAMES' ||
              (e.satellite_frame_count > 0 && e.history_fix_count > 0)
          );
          if (readyEvents.length > 0 && !readyEvents.some((e) => e.name === selectedCyclone)) {
            setSelectedCyclone(readyEvents[0].name);
            setSelectedTimestamp(readyEvents[0].latest_observation || undefined);
          }
        })
        .catch((err) => console.warn('Failed to load active current events:', err));
    }
  }, [pipelineMode, selectedCyclone]);

  // Execute canonical end-to-end inference pipeline
  const executePipeline = useCallback(() => {
    if (!selectedCyclone) return;
    setLoading(true);
    setError(null);

    const runner =
      pipelineMode === 'HISTORICAL' ? vayuApi.runInference : vayuApi.runCurrentInference;

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
  }, [selectedCyclone, selectedTimestamp, pipelineMode]);

  useEffect(() => {
    executePipeline();
  }, [executePipeline]);

  // Deterministic SIH Rehearsal Handler
  const triggerDemoMode = () => {
    setPipelineMode('HISTORICAL');
    setSelectedCyclone('AMPHAN');
    setSelectedTimestamp('2020-05-18T06:00:00+00:00');
  };

  // Canonical data accessors
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

  const currentCenterPoint = observation?.reference_center?.lat != null
    ? { lat: observation.reference_center.lat, lon: observation.reference_center.lon }
    : null;

  const aiCenterPoint = center?.ai_lat != null
    ? { lat: center.ai_lat, lon: center.ai_lon }
    : null;

  const forecastPoints = forecast
    ? {
        plus_12h: forecast.plus_12h ? { lat: forecast.plus_12h.lat, lon: forecast.plus_12h.lon } : null,
        plus_24h: forecast.plus_24h ? { lat: forecast.plus_24h.lat, lon: forecast.plus_24h.lon } : null,
        plus_48h: forecast.plus_48h ? { lat: forecast.plus_48h.lat, lon: forecast.plus_48h.lon } : null,
      }
    : null;

  return (
    <div className="min-h-screen bg-[#F5F7FA] text-slate-900 flex flex-col font-sans antialiased">
      {/* 1. Header */}
      <Header
        pipelineMode={pipelineMode}
        setPipelineMode={setPipelineMode}
        healthStatus={healthStatus}
        triggerDemoMode={triggerDemoMode}
        openTechnicalDetails={() => setTechnicalModalOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-6 space-y-6">
        {/* Notice Banner */}
        {error && (
          <div className="bg-rose-50 border border-rose-200 rounded-lg p-3.5 text-xs text-rose-800 flex items-center justify-between gap-3">
            <div>
              <span className="font-semibold">Notice:</span> {error}
            </div>
            <button
              type="button"
              onClick={executePipeline}
              className="px-2.5 py-1 rounded bg-rose-100 hover:bg-rose-200 text-rose-900 font-medium transition-colors shrink-0"
            >
              Retry
            </button>
          </div>
        )}

        {/* Available Cyclone Events Feed (in Current / Replay mode) */}
        {pipelineMode === 'CURRENT / REPLAY' && (
          <CurrentEventFeed
            events={activeEvents}
            selectedCyclone={selectedCyclone}
            onSelectCyclone={(name, timestamp) => {
              setSelectedCyclone(name);
              setSelectedTimestamp(timestamp);
            }}
            onRunForecast={executePipeline}
            isLoading={loading}
          />
        )}

        {/* 2. Primary Cyclone Summary Block */}
        <CycloneSummary
          cyclones={cyclones}
          selectedCyclone={selectedCyclone}
          onSelectCyclone={(name) => {
            setSelectedCyclone(name);
            setSelectedTimestamp(undefined);
          }}
          observationTime={observation?.t0_utc}
          intensityCategory={intensity?.category || observation?.reference_category}
          windKt={wind?.wind_kt ?? observation?.reference_wind_kt}
          centralPressureHpa={wind?.central_pressure_hpa}
          currentCenter={currentCenterPoint}
          aiCenter={aiCenterPoint}
          pipelineMode={pipelineMode}
          isLoading={loading}
        />

        {/* 3. Dominant Geographic Map */}
        <CycloneMap
          currentCenter={currentCenterPoint}
          aiCenter={aiCenterPoint}
          forecast={forecastPoints}
          uncertainty={uncertainty}
          verification={verification}
          cycloneName={selectedCyclone}
          isHistoricalMode={pipelineMode === 'HISTORICAL'}
          t0Utc={canonicalData?.observation?.t0_utc ?? null}
        />

        {/* 4. Information Grid (2 Columns) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {/* Left Column: Current Conditions, Forecast Timeline, Accuracy */}
          <div className="space-y-6">
            <CurrentConditions
              center={currentCenterPoint || aiCenterPoint}
              intensityCategory={intensity?.category || observation?.reference_category}
              windKt={wind?.wind_kt ?? observation?.reference_wind_kt}
              centralPressureHpa={wind?.central_pressure_hpa}
              satelliteSource="GridSat-B1 11µm IR"
              intensityConfidence={intensity?.confidence}
            />

            <ForecastTimeline
              t0Utc={observation?.t0_utc}
              forecast={forecastPoints}
              uncertainty={uncertainty}
            />

            <ForecastAccuracy
              isHistoricalMode={pipelineMode === 'HISTORICAL'}
              verification={verification}
            />
          </div>

          {/* Right Column: Similar Cyclones, AI Focus, Satellite Sequence, Rainfall Context */}
          <div className="space-y-6">
            <SimilarStorms analogs={analogs} />

            <AiInterpretability
              saliency={saliency}
              cycloneName={selectedCyclone}
            />

            <SatelliteSequence
              frames={observation?.gridsat_frames}
              sourceName="GridSat-B1 11µm IR"
            />

            <RainfallContext secondaryObs={secondaryObs} />
          </div>
        </div>

        {/* 5. Small Footer-Level Data Sources */}
        <DataSources />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white px-6 py-4 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>VAYU-NET Tropical Cyclone Intelligence · SIH 26070</span>
          <span>India Meteorological Department · Ministry of Earth Sciences</span>
        </div>
      </footer>

      {/* Technical Details Inspection Modal */}
      <TechnicalDetailsModal
        isOpen={technicalModalOpen}
        onClose={() => setTechnicalModalOpen(false)}
        metadata={metadata}
        saliencyDetails={
          saliency
            ? {
                target_layer: saliency.target_layer,
                target_head_explained: saliency.target_head_explained,
              }
            : null
        }
      />
    </div>
  );
}
