import React from 'react';

interface Cyclone {
  cyclone_id: string;
  storm_name: string;
  year: number;
  split: string;
  peak_category: string;
  max_wind_kt: number;
}

interface CycloneSummaryProps {
  cyclones: Cyclone[];
  selectedCyclone: string;
  onSelectCyclone: (name: string) => void;
  observationTime?: string;
  intensityCategory?: string;
  windKt?: number | null;
  centralPressureHpa?: number | null;
  currentCenter?: { lat: number; lon: number } | null;
  aiCenter?: { lat: number; lon: number } | null;
  pipelineMode: 'HISTORICAL' | 'CURRENT / REPLAY';
  isLoading: boolean;
}

export const CycloneSummary: React.FC<CycloneSummaryProps> = ({
  cyclones,
  selectedCyclone,
  onSelectCyclone,
  observationTime,
  intensityCategory,
  windKt,
  centralPressureHpa,
  currentCenter,
  aiCenter,
  pipelineMode,
  isLoading,
}) => {
  // Format observation date strictly from canonical observationTime
  const formattedDate = observationTime
    ? (() => {
        const d = new Date(observationTime);
        const day = d.getUTCDate();
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const month = months[d.getUTCMonth()];
        const year = d.getUTCFullYear();
        const time = d.toISOString().slice(11, 16);
        return `${day} ${month} ${year} · ${time} UTC`;
      })()
    : 'Observation pending';

  const activeCenter = currentCenter || aiCenter;

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        {/* Cyclone Name, Category, Observation Time */}
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              {selectedCyclone.replace(/^NIO_\d{4}_/, '')}
            </h1>

            {/* Storm Switcher (in Historical Mode) */}
            {pipelineMode === 'HISTORICAL' && cyclones.length > 0 && (
              <select
                value={selectedCyclone}
                onChange={(e) => onSelectCyclone(e.target.value)}
                className="bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-md px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 font-medium cursor-pointer"
              >
                {cyclones.map((c) => (
                  <option key={c.cyclone_id} value={c.storm_name}>
                    {c.storm_name} ({c.year}) · {c.peak_category}
                  </option>
                ))}
              </select>
            )}

            {isLoading && (
              <span className="text-xs text-blue-600 font-normal">Updating...</span>
            )}
          </div>

          <div className="text-sm font-medium text-slate-700">
            {intensityCategory || 'Intensity Analysis'}
          </div>

          <div className="text-xs text-slate-500">
            Observation time: <span className="font-medium text-slate-700">{formattedDate}</span>
          </div>
        </div>

        {/* Essential Metrics: Position, Intensity, Wind, Pressure */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 pt-4 md:pt-0 border-t md:border-t-0 border-slate-100">
          {/* Current Position */}
          <div>
            <div className="text-xs text-slate-500 mb-0.5">Current Position</div>
            <div className="text-base font-semibold text-slate-900 tabular-nums">
              {activeCenter ? `${activeCenter.lat.toFixed(1)}°N · ${activeCenter.lon.toFixed(1)}°E` : '—'}
            </div>
            <div className="text-[11px] text-slate-400">
              {currentCenter ? 'Observed center' : 'AI detected'}
            </div>
          </div>

          {/* Intensity */}
          <div>
            <div className="text-xs text-slate-500 mb-0.5">Intensity</div>
            <div className="text-base font-semibold text-slate-900 truncate">
              {intensityCategory || 'Analyzing'}
            </div>
            <div className="text-[11px] text-slate-400">Classification</div>
          </div>

          {/* Wind */}
          <div>
            <div className="text-xs text-slate-500 mb-0.5">Wind</div>
            <div className="text-base font-semibold text-slate-900 tabular-nums">
              {windKt != null ? `${Math.round(windKt * 10) / 10} kt` : '—'}
            </div>
            <div className="text-[11px] text-slate-400">
              {windKt != null ? `${Math.round(windKt * 1.852)} km/h` : 'Sustained'}
            </div>
          </div>

          {/* Pressure */}
          <div>
            <div className="text-xs text-slate-500 mb-0.5">Pressure</div>
            <div className="text-base font-semibold text-slate-900 tabular-nums">
              {centralPressureHpa ? `${Math.round(centralPressureHpa)} hPa` : '—'}
            </div>
            <div className="text-[11px] text-slate-400">Central minimum</div>
          </div>
        </div>
      </div>
    </div>
  );
};
