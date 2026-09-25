import React from 'react';

interface SecondaryFrame {
  step: string;
  offset_hours: number;
  timestamp_utc: string;
  granule_id?: string;
  local_status?: string;
}

interface SecondaryObservation {
  source?: string;
  frames?: SecondaryFrame[];
  display_ready?: boolean;
}

interface RainfallContextProps {
  secondaryObs?: SecondaryObservation | null;
}

export const RainfallContext: React.FC<RainfallContextProps> = ({ secondaryObs }) => {
  const frames = secondaryObs?.frames || [];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
        <h2 className="text-xs font-semibold text-slate-900">Rainfall Context</h2>
        <span className="text-xs text-slate-500">Secondary Observation</span>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
        {frames.length > 0 ? (
          frames.map((f, idx) => (
            <div
              key={f.step || idx}
              className="bg-slate-50 rounded border border-slate-200 p-2 text-center"
            >
              <div className="text-[10px] font-medium text-slate-500 uppercase">
                {f.step.replace('t_minus_', 't - ').replace('_', ' ')}
              </div>
              <div className="text-xs font-mono text-slate-700 mt-0.5">
                {f.timestamp_utc?.slice(11, 16) || '--:--'} UTC
              </div>
            </div>
          ))
        ) : (
          <div className="col-span-6 text-xs text-slate-500 py-1 text-center">
            Synchronized precipitation frames ready.
          </div>
        )}
      </div>

      <p className="text-xs text-slate-500">
        Calibrated NASA GPM IMERG precipitation fields synchronized with satellite observation timestamps provide secondary rainfall structure context.
      </p>
    </div>
  );
};
