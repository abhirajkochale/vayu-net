import React from 'react';

interface VerificationHorizon {
  status?: string;
  predicted?: { lat: number; lon: number };
  actual?: { lat: number; lon: number };
  dpe_km?: number | null;
}

interface VerificationData {
  status?: string;
  message?: string;
  horizons?: {
    plus_12h?: VerificationHorizon;
    plus_24h?: VerificationHorizon;
    plus_48h?: VerificationHorizon;
  };
  aggregate_dpe_km?: number | null;
}

interface ForecastAccuracyProps {
  isHistoricalMode: boolean;
  verification?: VerificationData | null;
}

export const ForecastAccuracy: React.FC<ForecastAccuracyProps> = ({
  isHistoricalMode,
  verification,
}) => {
  const isAvailable =
    isHistoricalMode &&
    verification &&
    verification.status !== 'UNAVAILABLE' &&
    verification.horizons &&
    Object.keys(verification.horizons).length > 0;

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
        <h2 className="text-xs font-semibold text-slate-900">Forecast Accuracy</h2>
        {isAvailable && verification.aggregate_dpe_km != null && (
          <span className="text-xs text-slate-500">
            Average error: {verification.aggregate_dpe_km.toFixed(1)} km
          </span>
        )}
      </div>

      {isHistoricalMode ? (
        isAvailable ? (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: '+12h', dpe: verification.horizons?.plus_12h?.dpe_km },
                { label: '+24h', dpe: verification.horizons?.plus_24h?.dpe_km },
                { label: '+48h', dpe: verification.horizons?.plus_48h?.dpe_km },
              ].map(({ label, dpe }) => (
                <div key={label} className="bg-slate-50 rounded border border-slate-200 p-2.5">
                  <div className="text-xs text-slate-500 mb-0.5">{label}</div>
                  <div className="text-sm font-semibold text-slate-900 tabular-nums">
                    {dpe != null ? `${dpe.toFixed(1)} km` : '—'}
                  </div>
                </div>
              ))}
            </div>

            <p className="text-xs text-slate-500 pt-0.5">
              Historical verification against IMD best-track data.
            </p>
          </div>
        ) : (
          <p className="text-xs text-slate-500">
            Historical best-track verification unavailable for this observation step.
          </p>
        )
      ) : (
        <div className="text-xs text-slate-600 space-y-1">
          <p className="font-medium text-slate-800">
            Forecast verification unavailable for an active event.
          </p>
          <p className="text-slate-500">
            Future observations are not used during prediction.
          </p>
        </div>
      )}
    </div>
  );
};
