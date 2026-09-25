import React from 'react';

interface GridSatFrame {
  step: string;
  offset_hours: number;
  timestamp_utc: string;
  source?: string;
  status?: string;
}

interface SatelliteSequenceProps {
  frames?: GridSatFrame[];
  sourceName?: string;
}

export const SatelliteSequence: React.FC<SatelliteSequenceProps> = ({
  frames,
  sourceName = 'GridSat-B1 11µm IR',
}) => {
  const stepsConfig = [
    { offset: -15, label: '15h ago' },
    { offset: -12, label: '12h ago' },
    { offset: -9, label: '9h ago' },
    { offset: -6, label: '6h ago' },
    { offset: -3, label: '3h ago' },
    { offset: 0, label: 'Now' },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
        <h2 className="text-xs font-semibold text-slate-900">Satellite Observation Sequence</h2>
        <span className="text-xs text-slate-500">{sourceName}</span>
      </div>

      {/* Clean horizontal progression: 15h ago → 12h ago → 9h ago → 6h ago → 3h ago → Now */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-1.5 pt-1">
        {stepsConfig.map((step, idx) => {
          const isNow = step.offset === 0;
          // Match with canonical frame from inference response if available
          const matchedFrame = frames?.find((f) => f.offset_hours === step.offset);
          const timeStr = matchedFrame?.timestamp_utc
            ? matchedFrame.timestamp_utc.slice(11, 16) + ' UTC'
            : '—';

          return (
            <React.Fragment key={step.label}>
              <div
                className={`flex-1 w-full sm:w-auto p-2 rounded text-center border ${
                  isNow
                    ? 'bg-blue-50/70 border-blue-300 text-blue-900'
                    : 'bg-slate-50 border-slate-200 text-slate-700'
                }`}
              >
                <div className="text-xs font-semibold">{step.label}</div>
                <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                  {timeStr}
                </div>
                <div className="mt-1 flex items-center justify-center gap-1">
                  <span className={`w-1.5 h-1.5 rounded-full ${isNow ? 'bg-[#2563EB]' : 'bg-[#15803D]'}`} />
                  <span className="text-[10px] text-slate-400">Captured</span>
                </div>
              </div>

              {idx < stepsConfig.length - 1 && (
                <span className="hidden sm:inline text-slate-300 text-xs px-0.5 select-none">
                  →
                </span>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
