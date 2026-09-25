import React from 'react';

interface CurrentConditionsProps {
  center?: { lat: number; lon: number } | null;
  intensityCategory?: string | null;
  windKt?: number | null;
  centralPressureHpa?: number | null;
  satelliteSource?: string;
  intensityConfidence?: number | null;
}

export const CurrentConditions: React.FC<CurrentConditionsProps> = ({
  center,
  intensityCategory,
  windKt,
  centralPressureHpa,
  satelliteSource = 'GridSat-B1 11µm IR',
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5">
      <div className="text-xs font-semibold text-slate-900 mb-3">
        Current Conditions
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        {/* Center */}
        <div>
          <div className="text-xs text-slate-500 mb-0.5">Center</div>
          <div className="text-sm font-semibold text-slate-900 tabular-nums">
            {center ? `${center.lat.toFixed(1)}°N ${center.lon.toFixed(1)}°E` : '—'}
          </div>
          <div className="text-[11px] text-slate-400">Observed fix</div>
        </div>

        {/* Intensity */}
        <div>
          <div className="text-xs text-slate-500 mb-0.5">Intensity</div>
          <div className="text-sm font-semibold text-slate-900 truncate">
            {intensityCategory || 'Analyzing'}
          </div>
          <div className="text-[11px] text-slate-400">Classification</div>
        </div>

        {/* Wind */}
        <div>
          <div className="text-xs text-slate-500 mb-0.5">Wind</div>
          <div className="text-sm font-semibold text-slate-900 tabular-nums">
            {windKt != null ? `${windKt} kt` : '—'}
          </div>
          <div className="text-[11px] text-slate-400">
            {windKt != null ? `${Math.round(windKt * 1.852)} km/h` : 'Sustained'}
          </div>
        </div>

        {/* Pressure */}
        <div>
          <div className="text-xs text-slate-500 mb-0.5">Pressure</div>
          <div className="text-sm font-semibold text-slate-900 tabular-nums">
            {centralPressureHpa ? `${centralPressureHpa} hPa` : '—'}
          </div>
          <div className="text-[11px] text-slate-400">Central minimum</div>
        </div>

        {/* Satellite Source */}
        <div className="col-span-2 sm:col-span-1">
          <div className="text-xs text-slate-500 mb-0.5">Satellite Source</div>
          <div className="text-sm font-semibold text-slate-900 truncate">
            {satelliteSource}
          </div>
          <div className="text-[11px] text-slate-400">Primary IR channel</div>
        </div>
      </div>
    </div>
  );
};
