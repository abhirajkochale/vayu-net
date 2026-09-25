import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import geoData from './geoData.json';

// ── Wind field types ──────────────────────────────────────────────────────────
interface WindSample {
  lon: number;
  lat: number;
  speedKt: number;   // wind speed in knots
  dirDeg: number;    // meteorological wind direction (FROM, degrees)
}

interface WindFieldState {
  samples: WindSample[];
  fetchedAt: string; // ISO UTC string
  loading: boolean;
  error: string | null;
}

interface Point {
  lat: number;
  lon: number;
}

interface UncertaintyData {
  plus_12h_km?: number;
  plus_24h_km?: number;
  plus_48h_km?: number;
  cone_geometries?: {
    plus_12h?: [number, number][];
    plus_24h?: [number, number][];
    plus_48h?: [number, number][];
  };
}

interface VerificationData {
  status?: string;
  horizons?: {
    plus_12h?: { actual?: Point; dpe_km?: number };
    plus_24h?: { actual?: Point; dpe_km?: number };
    plus_48h?: { actual?: Point; dpe_km?: number };
  };
}

interface CycloneMapProps {
  currentCenter?: Point | null;
  aiCenter?: Point | null;
  forecast?: {
    plus_12h?: Point | null;
    plus_24h?: Point | null;
    plus_48h?: Point | null;
  } | null;
  uncertainty?: UncertaintyData | null;
  verification?: VerificationData | null;
  cycloneName: string;
  isHistoricalMode: boolean;
  /** Canonical t0 UTC timestamp used for inference — used to fetch the matching satellite frame */
  t0Utc?: string | null;
}

// Satellite observation layer state
interface SatelliteLayerState {
  imageB64: string | null;
  timestampUtc: string | null;
  loading: boolean;
  error: string | null;  // null = not yet fetched, 'UNAVAILABLE' = 404
}

export const CycloneMap: React.FC<CycloneMapProps> = ({
  currentCenter,
  aiCenter,
  forecast,
  uncertainty,
  verification,
  cycloneName,
  isHistoricalMode,
  t0Utc,
}) => {
  // NIO basin base bounds
  const baseBounds = {
    minLon: 56,
    maxLon: 98,
    minLat: 3,
    maxLat: 28,
  };

  const mapWidth = 920;
  const mapHeight = 560;

  const [zoom, setZoom] = useState<number>(1.4);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [showUncertainty, setShowUncertainty] = useState<boolean>(true);
  const [showActualTrack, setShowActualTrack] = useState<boolean>(true);
  const [showWind, setShowWind] = useState<boolean>(false);
  // Satellite ON by default — Historical: always on; Current/Replay: on when available
  const [showSatellite, setShowSatellite] = useState<boolean>(true);
  const [satOpacity, setSatOpacity] = useState<number>(0.72);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    title: string;
    lines: string[];
  } | null>(null);

  // ── Satellite observation layer state ─────────────────────────────────────
  const [satLayer, setSatLayer] = useState<SatelliteLayerState>({
    imageB64: null,
    timestampUtc: null,
    loading: false,
    error: null,
  });
  const satFetchKeyRef = useRef<string>('');

  // ── Wind field state ───────────────────────────────────────────────────────
  const [windField, setWindField] = useState<WindFieldState>({
    samples: [],
    fetchedAt: '',
    loading: false,
    error: null,
  });
  const windFetchedRef = useRef(false);

  // Grid of lat/lon points covering the NIO region — 8×9 = 72 points
  // step ~3° lat (4–25°N), ~5° lon (57–97°E) for coherent spatial coverage
  const WIND_GRID = useMemo(() => {
    const pts: { lat: number; lon: number }[] = [];
    const lats = [4, 7, 10, 13, 16, 19, 22, 25];
    const lons = [57, 62, 67, 72, 77, 82, 87, 92, 97];
    for (const lat of lats)
      for (const lon of lons)
        pts.push({ lat, lon });
    return pts;
  }, []);

  // ── Satellite fetch ──────────────────────────────────────────────────────
  const fetchSatelliteImage = useCallback(async (ts: string) => {
    const key = ts;
    if (satFetchKeyRef.current === key && satLayer.imageB64) return; // cached
    setSatLayer({ imageB64: null, timestampUtc: null, loading: true, error: null });
    satFetchKeyRef.current = key;
    try {
      const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';
      const url = `${API_BASE}/api/satellite/image?t0_utc=${encodeURIComponent(ts)}`;
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (res.status === 404) {
        setSatLayer({ imageB64: null, timestampUtc: ts, loading: false, error: 'UNAVAILABLE' });
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSatLayer({
        imageB64: data.image_b64 ?? null,
        timestampUtc: data.t0_utc ?? ts,
        loading: false,
        error: null,
      });
    } catch (err: any) {
      setSatLayer({ imageB64: null, timestampUtc: ts, loading: false, error: err.message ?? 'fetch error' });
    }
  }, [satLayer.imageB64]);

  // Fetch when satellite layer is toggled ON or t0Utc changes while ON
  useEffect(() => {
    if (showSatellite && t0Utc) {
      fetchSatelliteImage(t0Utc);
    }
  }, [showSatellite, t0Utc]);

  // Reset satellite cache when cyclone changes
  useEffect(() => {
    setSatLayer({ imageB64: null, timestampUtc: null, loading: false, error: null });
    satFetchKeyRef.current = '';
  }, [cycloneName]);

  const fetchWindField = useCallback(async () => {
    if (windField.loading) return;
    setWindField(prev => ({ ...prev, loading: true, error: null }));
    try {
      // Open-Meteo GFS – hourly 10m wind speed & direction, no API key needed
      const latStr = WIND_GRID.map(p => p.lat).join(',');
      const lonStr = WIND_GRID.map(p => p.lon).join(',');
      const url =
        `https://api.open-meteo.com/v1/forecast?` +
        `latitude=${latStr}&longitude=${lonStr}` +
        `&hourly=wind_speed_10m,wind_direction_10m` +
        `&wind_speed_unit=kn` +
        `&forecast_days=1&timezone=UTC&timeformat=unixtime`;

      const res = await fetch(url);
      if (!res.ok) throw new Error(`Open-Meteo returned ${res.status}`);
      const json = await res.json();

      // json is an array when multiple lat/lon are passed
      const arr = Array.isArray(json) ? json : [json];
      const nowMs = Date.now();
      const samples: WindSample[] = [];
      const fetchedAt = new Date().toISOString();

      arr.forEach((item: any, idx: number) => {
        const times: number[] = item?.hourly?.time ?? [];
        const speeds: number[] = item?.hourly?.wind_speed_10m ?? [];
        const dirs: number[] = item?.hourly?.wind_direction_10m ?? [];

        // Pick the hour closest to now
        let closest = 0;
        let minDiff = Infinity;
        times.forEach((t, i) => {
          const diff = Math.abs(t * 1000 - nowMs);
          if (diff < minDiff) { minDiff = diff; closest = i; }
        });

        const grid = WIND_GRID[idx];
        if (grid && speeds[closest] != null && dirs[closest] != null) {
          samples.push({
            lat: grid.lat,
            lon: grid.lon,
            speedKt: Math.round(speeds[closest] * 10) / 10,
            dirDeg: dirs[closest],
          });
        }
      });

      setWindField({ samples, fetchedAt, loading: false, error: null });
    } catch (err: any) {
      setWindField(prev => ({
        ...prev,
        loading: false,
        error: err?.message ?? 'Failed to load wind data',
      }));
    }
  }, [WIND_GRID]);

  // Fetch wind data the first time the layer is toggled on
  useEffect(() => {
    if (showWind && !windFetchedRef.current) {
      windFetchedRef.current = true;
      fetchWindField();
    }
  }, [showWind]);

  // Restrained slate palette — monochromatic, no neon or rainbow.
  // Three tones only: lighter for calm, darker for strong.
  const windColour = (kt: number): string => {
    if (kt < 15) return '#94a3b8';  // calm   – slate-400
    if (kt < 30) return '#64748b';  // moderate – slate-500
    return '#475569';               // strong  – slate-600
  };

  // Wind arrow: thin needle with a minimal chevron tip.
  // Small and subtle — track/centers must stay visually dominant.
  const WindArrow = ({
    cx, cy, speedKt, dirDeg,
  }: { cx: number; cy: number; speedKt: number; dirDeg: number }) => {
    const toRad = (d: number) => (d * Math.PI) / 180;
    // Arrow points in the TO direction
    const arrowDir = (dirDeg + 180) % 360;
    // Length: 5px base + 0.25px/kt, capped at 18px — small and proportional
    const len = Math.min(5 + speedKt * 0.25, 18);
    const strokeW = speedKt < 15 ? 0.9 : speedKt < 30 ? 1.1 : 1.4;
    const angle = toRad(arrowDir - 90); // account for SVG y-axis flip
    const colour = windColour(speedKt);

    // Shaft: centred on the grid point
    const x1 = cx - Math.cos(angle) * len * 0.42;
    const y1 = cy - Math.sin(angle) * len * 0.42;
    const x2 = cx + Math.cos(angle) * len * 0.58;
    const y2 = cy + Math.sin(angle) * len * 0.58;

    // Minimal open-chevron tip (no filled triangle)
    const tipLen = Math.max(2.5, len * 0.22);
    const wingSpread = tipLen * 0.45;
    const perpAngle = angle + Math.PI / 2;
    const wingBase = {
      x: x2 - Math.cos(angle) * tipLen,
      y: y2 - Math.sin(angle) * tipLen,
    };
    const wL = {
      x: wingBase.x + Math.cos(perpAngle) * wingSpread,
      y: wingBase.y + Math.sin(perpAngle) * wingSpread,
    };
    const wR = {
      x: wingBase.x - Math.cos(perpAngle) * wingSpread,
      y: wingBase.y - Math.sin(perpAngle) * wingSpread,
    };

    return (
      <g opacity={0.62}>
        <line
          x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={colour} strokeWidth={strokeW} strokeLinecap="round"
        />
        {/* Open chevron — two lines, no fill */}
        <line x1={x2} y1={y2} x2={wL.x} y2={wL.y} stroke={colour} strokeWidth={strokeW} strokeLinecap="round" />
        <line x1={x2} y1={y2} x2={wR.x} y2={wR.y} stroke={colour} strokeWidth={strokeW} strokeLinecap="round" />
      </g>
    );
  };

  // Geographic projection: [lon, lat] -> [x, y]
  const project = (lon: number, lat: number): [number, number] => {
    const x = ((lon - baseBounds.minLon) / (baseBounds.maxLon - baseBounds.minLon)) * mapWidth;
    const y = ((baseBounds.maxLat - lat) / (baseBounds.maxLat - baseBounds.minLat)) * mapHeight;
    return [x, y];
  };

  const toPointsString = (coords: [number, number][]): string => {
    return coords.map(([lon, lat]) => project(lon, lat).join(',')).join(' ');
  };

  // Pre-calculate SVG paths for land polygons and coastlines
  const landPaths = useMemo(() => {
    return geoData.landPolygons.map((ring) => {
      const pts = ring.map(([lon, lat]) => project(lon, lat));
      return pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ') + ' Z';
    });
  }, []);

  const coastPaths = useMemo(() => {
    return geoData.coastlines.map((segment) => {
      const pts = segment.map(([lon, lat]) => project(lon, lat));
      return pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
    });
  }, []);

  // Track points
  const p0 = currentCenter || aiCenter;
  const p12 = forecast?.plus_12h;
  const p24 = forecast?.plus_24h;
  const p48 = forecast?.plus_48h;

  const xy0 = p0 ? project(p0.lon, p0.lat) : null;
  const xyAi = aiCenter ? project(aiCenter.lon, aiCenter.lat) : null;
  const xy12 = p12 ? project(p12.lon, p12.lat) : null;
  const xy24 = p24 ? project(p24.lon, p24.lat) : null;
  const xy48 = p48 ? project(p48.lon, p48.lat) : null;

  // Auto-center map on storm track initially and when cyclone changes
  const centerOnCyclone = () => {
    if (!xy0) {
      setZoom(1);
      setPanOffset({ x: 0, y: 0 });
      return;
    }
    // Calculate centroid of trajectory
    const trackPoints = [xy0, xy12, xy24, xy48].filter(Boolean) as [number, number][];
    const avgX = trackPoints.reduce((acc, p) => acc + p[0], 0) / trackPoints.length;
    const avgY = trackPoints.reduce((acc, p) => acc + p[1], 0) / trackPoints.length;

    const targetZoom = 1.45;
    const offsetX = (mapWidth / 2 - avgX) * targetZoom;
    const offsetY = (mapHeight / 2 - avgY) * targetZoom;

    setZoom(targetZoom);
    setPanOffset({ x: offsetX, y: offsetY });
  };

  useEffect(() => {
    centerOnCyclone();
  }, [cycloneName, xy0 ? `${xy0[0]},${xy0[1]}` : null]);

  // Full basin reset
  const viewFullBasin = () => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  // Forecast path string
  const forecastPath = useMemo(() => {
    const pts = [xy0, xy12, xy24, xy48].filter(Boolean) as [number, number][];
    if (pts.length < 2) return '';
    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  }, [xy0, xy12, xy24, xy48]);

  // Ground truth actual path string
  const actualPath = useMemo(() => {
    if (!isHistoricalMode || !verification?.horizons || verification.status === 'UNAVAILABLE') return '';
    const pts: [number, number][] = [];
    if (xy0) pts.push(xy0);
    const h12 = verification.horizons.plus_12h?.actual;
    const h24 = verification.horizons.plus_24h?.actual;
    const h48 = verification.horizons.plus_48h?.actual;
    if (h12) pts.push(project(h12.lon, h12.lat));
    if (h24) pts.push(project(h24.lon, h24.lat));
    if (h48) pts.push(project(h48.lon, h48.lat));
    if (pts.length < 2) return '';
    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  }, [isHistoricalMode, verification, xy0]);

  const coneGeoms = uncertainty?.cone_geometries;

  return (
    <div className="bg-white border border-slate-200 rounded-lg overflow-hidden flex flex-col">
      {/* Map Header & Toolbar */}
      <div className="px-4 py-2.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-800">Forecast Track & Geographic Situation</span>
          <span className="text-xs text-slate-400">· North Indian Ocean</span>
        </div>

        {/* Map Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Layer toggles */}
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-400 mr-0.5">Layers:</span>
            <button
              type="button"
              onClick={() => setShowUncertainty(!showUncertainty)}
              className={`text-xs px-2.5 py-1 rounded border transition-colors ${
                showUncertainty
                  ? 'bg-blue-50 text-blue-700 border-blue-200 font-medium'
                  : 'bg-white text-slate-500 border-slate-300 hover:bg-slate-50'
              }`}
            >
              Uncertainty
            </button>

            {isHistoricalMode && (
              <button
                type="button"
                onClick={() => setShowActualTrack(!showActualTrack)}
                className={`text-xs px-2.5 py-1 rounded border transition-colors ${
                  showActualTrack
                    ? 'bg-slate-100 text-slate-800 border-slate-300 font-medium'
                    : 'bg-white text-slate-500 border-slate-300 hover:bg-slate-50'
                }`}
              >
                Actual Track
              </button>
            )}

            <button
              type="button"
              onClick={() => setShowSatellite(!showSatellite)}
              className={`text-xs px-2.5 py-1 rounded border transition-colors flex items-center gap-1 ${
                showSatellite
                  ? 'bg-slate-100 text-slate-800 border-slate-400 font-medium'
                  : 'bg-white text-slate-500 border-slate-300 hover:bg-slate-50'
              }`}
              title="Toggle GridSat-B1 11µm IR satellite observation layer"
            >
              Satellite
              {showSatellite && satLayer.loading && (
                <span className="inline-block w-2.5 h-2.5 border border-slate-500 border-t-transparent rounded-full animate-spin" />
              )}
            </button>

            <button
              type="button"
              onClick={() => setShowWind(!showWind)}
              className={`text-xs px-2.5 py-1 rounded border transition-colors flex items-center gap-1 ${
                showWind
                  ? 'bg-slate-100 text-slate-800 border-slate-400 font-medium'
                  : 'bg-white text-slate-500 border-slate-300 hover:bg-slate-50'
              }`}
              title="Toggle GFS 10m wind field (Open-Meteo)"
            >
              Wind
              {showWind && windField.loading && (
                <span className="inline-block w-2.5 h-2.5 border border-slate-500 border-t-transparent rounded-full animate-spin" />
              )}
            </button>
          </div>

          <div className="h-4 w-px bg-slate-200 mx-0.5" />

          <button
            type="button"
            onClick={centerOnCyclone}
            className="text-xs px-2.5 py-1 rounded bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors"
          >
            Center
          </button>

          <button
            type="button"
            onClick={viewFullBasin}
            className="text-xs px-2.5 py-1 rounded bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors"
          >
            Full basin
          </button>

          <div className="flex items-center bg-white rounded border border-slate-300">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
              className="px-2 py-1 text-slate-700 hover:bg-slate-50 text-xs font-bold border-r border-slate-200"
              title="Zoom In"
            >
              +
            </button>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(z - 0.25, 0.75))}
              className="px-2 py-1 text-slate-700 hover:bg-slate-50 text-xs font-bold"
              title="Zoom Out"
            >
              −
            </button>
          </div>
        </div>
      </div>

      {/* SVG Viewport */}
      <div className="relative w-full aspect-[16/10] sm:aspect-[16/9] bg-[#eef3f8] overflow-hidden select-none">
        <svg
          viewBox={`0 0 ${mapWidth} ${mapHeight}`}
          className="w-full h-full transition-transform duration-150"
          style={{
            transform: `scale(${zoom}) translate(${panOffset.x / zoom}px, ${panOffset.y / zoom}px)`,
            transformOrigin: 'center center',
          }}
          onMouseLeave={() => setTooltip(null)}
        >
          <defs>
            {/* Arrow marker for track heading */}
            <marker
              id="geoTrackArrow"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563EB" />
            </marker>
          </defs>

          {/* Ocean Base Fill */}
          <rect width={mapWidth} height={mapHeight} fill="#eef3f8" />

          {/* ── GridSat-B1 11µm IR Satellite Observation Layer ────────────────
               Rendered first (bottom of stack) — all overlays sit above.     */}
          {showSatellite && satLayer.imageB64 && (() => {
            // The NPZ covers the full NIO domain [-5°N,35°N, 40°E,105°E].
            // Project domain corners into SVG pixel space.
            const [x0, y0] = project(40, 35);   // top-left  (lon=40, lat=35)
            const [x1, y1] = project(105, -5);  // bot-right (lon=105, lat=-5)
            return (
              <image
                href={`data:image/png;base64,${satLayer.imageB64}`}
                x={x0}
                y={y0}
                width={x1 - x0}
                height={y1 - y0}
                opacity={satOpacity}
                preserveAspectRatio="none"
                style={{ imageRendering: 'pixelated' }}
              />
            );
          })()}

          {/* Satellite unavailable notice inside SVG */}
          {showSatellite && !satLayer.loading && satLayer.error === 'UNAVAILABLE' && (
            <text
              x={mapWidth / 2} y={mapHeight / 2 - 10}
              textAnchor="middle"
              fill="#64748b"
              fontSize="11"
              fontFamily="sans-serif"
            >
              No GridSat-B1 observation archived for this inference time.
            </text>
          )}

          {/* Geographic Graticule Grid Lines */}
          <g stroke="#e2e8f0" strokeWidth="0.8" strokeDasharray="3 3">
            {[5, 10, 15, 20, 25].map((lat) => {
              const y = project(baseBounds.minLon, lat)[1];
              return (
                <g key={`lat-${lat}`}>
                  <line x1={0} y1={y} x2={mapWidth} y2={y} />
                  <text x={8} y={y - 3} fill="#64748b" fontSize="9" fontFamily="sans-serif">
                    {lat}°N
                  </text>
                </g>
              );
            })}
            {[60, 70, 80, 90].map((lon) => {
              const x = project(lon, baseBounds.minLat)[0];
              return (
                <g key={`lon-${lon}`}>
                  <line x1={x} y1={0} x2={x} y2={mapHeight} />
                  <text x={x + 4} y={mapHeight - 8} fill="#64748b" fontSize="9" fontFamily="sans-serif">
                    {lon}°E
                  </text>
                </g>
              );
            })}
          </g>

          {/* Watermark Basin Names */}
          <g fill="#94a3b8" fontSize="11" fontWeight="500" letterSpacing="0.18em" fontFamily="sans-serif">
            <text x={project(66, 17)[0]} y={project(66, 17)[1]} textAnchor="middle">
              ARABIAN SEA
            </text>
            <text x={project(88, 15)[0]} y={project(88, 15)[1]} textAnchor="middle">
              BAY OF BENGAL
            </text>
            <text x={project(78, 5)[0]} y={project(78, 5)[1]} textAnchor="middle">
              INDIAN OCEAN
            </text>
          </g>

          {/* Landmass Polygons (Light Gray) */}
          <g fill="#e2e8f0" stroke="none">
            {landPaths.map((d, i) => (
              <path key={`land-${i}`} d={d} />
            ))}
          </g>

          {/* Subtle Coastlines */}
          <g fill="none" stroke="#94a3b8" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
            {coastPaths.map((d, i) => (
              <path key={`coast-${i}`} d={d} />
            ))}
          </g>

          {/* ── Wind Field Layer: GFS 10m — rendered BELOW uncertainty & track */}
          {showWind && !windField.loading && windField.samples.length > 0 && (
            <g>
              {windField.samples.map((s, i) => {
                const [cx, cy] = project(s.lon, s.lat);
                return (
                  <WindArrow
                    key={i}
                    cx={cx}
                    cy={cy}
                    speedKt={s.speedKt}
                    dirDeg={s.dirDeg}
                  />
                );
              })}
            </g>
          )}

          {/* Uncertainty Cones (Very subtle translucent blue, no glow) */}
          {showUncertainty && (
            <g>
              {coneGeoms?.plus_48h && (
                <polygon
                  points={toPointsString(coneGeoms.plus_48h)}
                  fill="rgba(37, 99, 235, 0.08)"
                  stroke="rgba(37, 99, 235, 0.35)"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
              )}
              {coneGeoms?.plus_24h && (
                <polygon
                  points={toPointsString(coneGeoms.plus_24h)}
                  fill="rgba(37, 99, 235, 0.12)"
                  stroke="rgba(37, 99, 235, 0.45)"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
              )}
              {coneGeoms?.plus_12h && (
                <polygon
                  points={toPointsString(coneGeoms.plus_12h)}
                  fill="rgba(37, 99, 235, 0.16)"
                  stroke="rgba(37, 99, 235, 0.6)"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                />
              )}
            </g>
          )}

          {/* Historical Actual Ground Truth Track (Muted gray dashed line) */}
          {showActualTrack && actualPath && (
            <g>
              <path
                d={actualPath}
                fill="none"
                stroke="#64748b"
                strokeWidth="1.6"
                strokeDasharray="4 3"
              />
              {/* Actual points */}
              {verification?.horizons?.plus_12h?.actual && (
                <circle
                  cx={project(verification.horizons.plus_12h.actual.lon, verification.horizons.plus_12h.actual.lat)[0]}
                  cy={project(verification.horizons.plus_12h.actual.lon, verification.horizons.plus_12h.actual.lat)[1]}
                  r="3.5"
                  fill="#64748b"
                  stroke="#ffffff"
                  strokeWidth="1"
                  className="cursor-pointer"
                  onMouseEnter={() => {
                    const act = verification.horizons?.plus_12h?.actual;
                    if (!act) return;
                    const [px, py] = project(act.lon, act.lat);
                    setTooltip({
                      x: px,
                      y: py,
                      title: 'Actual Position +12h',
                      lines: [
                        `IMD Best-track: ${act.lat.toFixed(2)}°N, ${act.lon.toFixed(2)}°E`,
                        `Forecast error: ${verification.horizons?.plus_12h?.dpe_km?.toFixed(1) ?? '—'} km`,
                      ],
                    });
                  }}
                />
              )}
              {verification?.horizons?.plus_24h?.actual && (
                <circle
                  cx={project(verification.horizons.plus_24h.actual.lon, verification.horizons.plus_24h.actual.lat)[0]}
                  cy={project(verification.horizons.plus_24h.actual.lon, verification.horizons.plus_24h.actual.lat)[1]}
                  r="3.5"
                  fill="#64748b"
                  stroke="#ffffff"
                  strokeWidth="1"
                  className="cursor-pointer"
                  onMouseEnter={() => {
                    const act = verification.horizons?.plus_24h?.actual;
                    if (!act) return;
                    const [px, py] = project(act.lon, act.lat);
                    setTooltip({
                      x: px,
                      y: py,
                      title: 'Actual Position +24h',
                      lines: [
                        `IMD Best-track: ${act.lat.toFixed(2)}°N, ${act.lon.toFixed(2)}°E`,
                        `Forecast error: ${verification.horizons?.plus_24h?.dpe_km?.toFixed(1) ?? '—'} km`,
                      ],
                    });
                  }}
                />
              )}
              {verification?.horizons?.plus_48h?.actual && (
                <circle
                  cx={project(verification.horizons.plus_48h.actual.lon, verification.horizons.plus_48h.actual.lat)[0]}
                  cy={project(verification.horizons.plus_48h.actual.lon, verification.horizons.plus_48h.actual.lat)[1]}
                  r="3.5"
                  fill="#64748b"
                  stroke="#ffffff"
                  strokeWidth="1"
                  className="cursor-pointer"
                  onMouseEnter={() => {
                    const act = verification.horizons?.plus_48h?.actual;
                    if (!act) return;
                    const [px, py] = project(act.lon, act.lat);
                    setTooltip({
                      x: px,
                      y: py,
                      title: 'Actual Position +48h',
                      lines: [
                        `IMD Best-track: ${act.lat.toFixed(2)}°N, ${act.lon.toFixed(2)}°E`,
                        `Forecast error: ${verification.horizons?.plus_48h?.dpe_km?.toFixed(1) ?? '—'} km`,
                      ],
                    });
                  }}
                />
              )}
            </g>
          )}

          {/* Predicted Forecast Track Path (Single clear blue line) */}
          {forecastPath && (
            <g>
              <path
                d={forecastPath}
                fill="none"
                stroke="#2563EB"
                strokeWidth="2.2"
                strokeLinecap="round"
                markerEnd="url(#geoTrackArrow)"
              />
            </g>
          )}

          {/* AI Center Point */}
          {xyAi && (
            <g
              className="cursor-pointer"
              onMouseEnter={() =>
                aiCenter &&
                setTooltip({
                  x: xyAi[0],
                  y: xyAi[1],
                  title: 'AI Detected Center (t0)',
                  lines: [`${aiCenter.lat.toFixed(2)}°N, ${aiCenter.lon.toFixed(2)}°E`],
                })
              }
            >
              <circle cx={xyAi[0]} cy={xyAi[1]} r="4.5" fill="none" stroke="#2563EB" strokeWidth="1.5" />
            </g>
          )}

          {/* Current / Observed Center Point (Clear solid marker) */}
          {xy0 && (
            <g
              className="cursor-pointer"
              onMouseEnter={() =>
                p0 &&
                setTooltip({
                  x: xy0[0],
                  y: xy0[1],
                  title: `Current Center (${cycloneName})`,
                  lines: [`${p0.lat.toFixed(2)}°N, ${p0.lon.toFixed(2)}°E`],
                })
              }
            >
              <circle cx={xy0[0]} cy={xy0[1]} r="5.5" fill="#0f172a" stroke="#ffffff" strokeWidth="1.5" />
              <text x={xy0[0] + 8} y={xy0[1] - 6} fill="#0f172a" fontSize="10" fontWeight="600" fontFamily="sans-serif">
                t0
              </text>
            </g>
          )}

          {/* Forecast Points: +12h, +24h, +48h (Solid blue points with clean labels) */}
          {xy12 && p12 && (
            <g
              className="cursor-pointer"
              onMouseEnter={() =>
                setTooltip({
                  x: xy12[0],
                  y: xy12[1],
                  title: 'Forecast +12 hours',
                  lines: [
                    `Predicted: ${p12.lat.toFixed(2)}°N, ${p12.lon.toFixed(2)}°E`,
                    `Uncertainty: ±${uncertainty?.plus_12h_km ?? '—'} km`,
                  ],
                })
              }
            >
              <circle cx={xy12[0]} cy={xy12[1]} r="4" fill="#2563EB" stroke="#ffffff" strokeWidth="1.5" />
              <text x={xy12[0] + 7} y={xy12[1] + 3} fill="#1e293b" fontSize="10" fontWeight="500" fontFamily="sans-serif">
                +12h
              </text>
            </g>
          )}

          {xy24 && p24 && (
            <g
              className="cursor-pointer"
              onMouseEnter={() =>
                setTooltip({
                  x: xy24[0],
                  y: xy24[1],
                  title: 'Forecast +24 hours',
                  lines: [
                    `Predicted: ${p24.lat.toFixed(2)}°N, ${p24.lon.toFixed(2)}°E`,
                    `Uncertainty: ±${uncertainty?.plus_24h_km ?? '—'} km`,
                  ],
                })
              }
            >
              <circle cx={xy24[0]} cy={xy24[1]} r="4" fill="#2563EB" stroke="#ffffff" strokeWidth="1.5" />
              <text x={xy24[0] + 7} y={xy24[1] + 3} fill="#1e293b" fontSize="10" fontWeight="500" fontFamily="sans-serif">
                +24h
              </text>
            </g>
          )}

          {xy48 && p48 && (
            <g
              className="cursor-pointer"
              onMouseEnter={() =>
                setTooltip({
                  x: xy48[0],
                  y: xy48[1],
                  title: 'Forecast +48 hours',
                  lines: [
                    `Predicted: ${p48.lat.toFixed(2)}°N, ${p48.lon.toFixed(2)}°E`,
                    `Uncertainty: ±${uncertainty?.plus_48h_km ?? '—'} km`,
                  ],
                })
              }
            >
              <circle cx={xy48[0]} cy={xy48[1]} r="4" fill="#2563EB" stroke="#ffffff" strokeWidth="1.5" />
              <text x={xy48[0] + 7} y={xy48[1] + 3} fill="#1e293b" fontSize="10" fontWeight="500" fontFamily="sans-serif">
                +48h
              </text>
            </g>
          )}
        </svg>

        {/* Clean, Non-Futuristic Tooltip */}
        {tooltip && (
          <div
            className="absolute z-20 pointer-events-none bg-white border border-slate-300 rounded shadow-md px-2.5 py-1.5 text-xs"
            style={{
              left: `${Math.min(tooltip.x + 12, mapWidth - 170)}px`,
              top: `${Math.max(tooltip.y - 45, 10)}px`,
            }}
          >
            <div className="font-semibold text-slate-900">{tooltip.title}</div>
            {tooltip.lines.map((l, i) => (
              <div key={i} className="text-slate-600 text-[11px]">
                {l}
              </div>
            ))}
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-3 left-3 bg-white/95 border border-slate-200 rounded px-3 py-1.5 text-xs text-slate-600 flex flex-wrap items-center gap-x-4 gap-y-1 shadow-sm max-w-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-900"></span>
            <span>Observed</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-600"></span>
            <span>Forecast track</span>
          </div>
          {showUncertainty && (
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-2 rounded border border-blue-400 bg-blue-100"></span>
              <span>Uncertainty</span>
            </div>
          )}
          {isHistoricalMode && showActualTrack && (
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-0.5 border-t border-dashed border-slate-500"></span>
              <span>Actual track</span>
            </div>
          )}
          {showSatellite && satLayer.imageB64 && (
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-2.5 rounded-sm bg-slate-400 border border-slate-300" />
              <span>GridSat-B1 IR</span>
            </div>
          )}
          {showWind && windField.samples.length > 0 && (
            <div className="flex items-center gap-1.5">
              <svg width="16" height="10" viewBox="0 0 16 10">
                <line x1="2" y1="5" x2="11" y2="5" stroke="#64748b" strokeWidth="1.2" />
                <line x1="11" y1="5" x2="8" y2="3" stroke="#64748b" strokeWidth="1.2" strokeLinecap="round" />
                <line x1="11" y1="5" x2="8" y2="7" stroke="#64748b" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
              <span>GFS 10m wind</span>
            </div>
          )}
        </div>

        {/* Satellite attribution + opacity slider */}
        {showSatellite && satLayer.imageB64 && !satLayer.loading && (
          <div className="absolute bottom-3 right-3 bg-white/95 border border-slate-200 rounded px-2.5 py-1.5 text-[10px] text-slate-500 shadow-sm space-y-1" style={{ minWidth: 220 }}>
            <div className="text-[10px] text-slate-600">
              <span className="font-medium text-slate-700">GridSat-B1</span> · 11 µm IR ·{' '}
              {satLayer.timestampUtc
                ? new Date(satLayer.timestampUtc).toLocaleString('en-GB', {
                    year: 'numeric', month: 'short', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', timeZone: 'UTC',
                  }) + ' UTC'
                : '—'
              }
            </div>
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-slate-500 whitespace-nowrap">Opacity</label>
              <input
                type="range"
                min={0.15}
                max={1}
                step={0.05}
                value={satOpacity}
                onChange={(e) => setSatOpacity(parseFloat(e.target.value))}
                className="flex-1 h-1.5 accent-slate-500 cursor-pointer"
              />
              <span className="text-[10px] text-slate-500 w-7 text-right">{Math.round(satOpacity * 100)}%</span>
            </div>
            <div className="text-[9px] text-slate-400">NOAA NCEI IRWIN CDR</div>
          </div>
        )}

        {/* Wind data attribution + timestamp — always visible when wind is active */}
        {showWind && windField.fetchedAt && !windField.loading && (
          <div className={`absolute bg-white/90 border border-slate-200 rounded px-2 py-1 text-[10px] text-slate-500 shadow-sm ${
            showSatellite && satLayer.imageB64
              ? 'bottom-16 right-3'  // shift up so it doesn't overlap satellite attribution
              : 'bottom-3 right-3'
          }`}>
            <span className="font-medium text-slate-700">Wind</span> · GFS 10m via Open-Meteo ·{' '}
            <span>
              {new Date(windField.fetchedAt).toLocaleTimeString('en-GB', {
                hour: '2-digit',
                minute: '2-digit',
                timeZone: 'UTC',
              })}{' '}
              UTC
            </span>
            {windField.error && (
              <span className="text-red-500 ml-1">({windField.error})</span>
            )}
            <button
              onClick={fetchWindField}
              className="ml-2 text-blue-600 underline hover:no-underline"
              title="Refresh wind data"
            >
              Refresh
            </button>
          </div>
        )}

        {/* Wind speed scale */}
        {showWind && windField.samples.length > 0 && !windField.loading && (
          <div className="absolute top-3 right-3 bg-white/95 border border-slate-200 rounded px-2.5 py-1.5 text-[10px] shadow-sm">
            <div className="text-slate-500 font-medium mb-1">Wind speed (kt)</div>
            {[
              { label: '< 15',  color: '#94a3b8' },
              { label: '15–30', color: '#64748b' },
              { label: '> 30',  color: '#475569' },
            ].map(({ label, color }) => (
              <div key={label} className="flex items-center gap-1.5 leading-snug">
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: color }} />
                <span className="text-slate-600">{label}</span>
              </div>
            ))}
          </div>
        )}

        {/* Wind error state */}
        {showWind && windField.error && !windField.loading && (
          <div className="absolute top-10 left-1/2 -translate-x-1/2 bg-white border border-red-200 rounded px-3 py-1.5 text-xs text-red-600 shadow">
            Wind data unavailable: {windField.error}
          </div>
        )}
      </div>
    </div>
  );
};
