import React from 'react';

interface Point {
  lat: number;
  lon: number;
}

interface ForecastTimelineProps {
  t0Utc?: string;
  forecast?: {
    plus_12h?: Point | null;
    plus_24h?: Point | null;
    plus_48h?: Point | null;
  } | null;
  uncertainty?: {
    plus_12h_km?: number;
    plus_24h_km?: number;
    plus_48h_km?: number;
  } | null;
}

export const ForecastTimeline: React.FC<ForecastTimelineProps> = ({
  forecast,
  uncertainty,
}) => {
  const steps = [
    {
      horizon: '+12 hours',
      point: forecast?.plus_12h,
      radiusKm: uncertainty?.plus_12h_km,
    },
    {
      horizon: '+24 hours',
      point: forecast?.plus_24h,
      radiusKm: uncertainty?.plus_24h_km,
    },
    {
      horizon: '+48 hours',
      point: forecast?.plus_48h,
      radiusKm: uncertainty?.plus_48h_km,
    },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
        <h2 className="text-xs font-semibold text-slate-900">Track Forecast</h2>
        <span className="text-xs text-slate-500">Predicted trajectory</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {steps.map((step) => (
          <div key={step.horizon} className="bg-slate-50/80 rounded border border-slate-200 p-3.5 space-y-1.5">
            <div className="text-xs font-bold text-blue-600">
              {step.horizon}
            </div>

            <div>
              <div className="text-[11px] text-slate-500">Predicted position</div>
              <div className="text-sm font-semibold text-slate-900 tabular-nums">
                {step.point ? `${step.point.lat.toFixed(2)}°N · ${step.point.lon.toFixed(2)}°E` : 'Pending'}
              </div>
            </div>

            <div className="text-xs text-slate-600 pt-1 border-t border-slate-200/60">
              Uncertainty <span className="font-medium text-slate-800">±{step.radiusKm ? Math.round(step.radiusKm) : '—'} km</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
