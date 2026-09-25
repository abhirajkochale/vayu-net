import React from 'react';

interface AnalogCyclone {
  storm_id: string;
  storm_name: string;
  year: number;
  distance: number;
  t0_utc?: string;
  peak_category?: string;
  split?: string;
}

interface SimilarStormsProps {
  analogs?: AnalogCyclone[] | null;
}

export const SimilarStorms: React.FC<SimilarStormsProps> = ({ analogs }) => {
  const topAnalogs = analogs?.slice(0, 2) || [];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
        <h2 className="text-xs font-semibold text-slate-900">Similar Historical Cyclones</h2>
        <span className="text-xs text-slate-500">Analog matches</span>
      </div>

      {topAnalogs.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {topAnalogs.map((item, idx) => (
            <div
              key={item.storm_id + idx}
              className="bg-slate-50 rounded border border-slate-200 p-3 space-y-1"
            >
              <div className="text-xs font-bold text-slate-900">
                {item.storm_name.replace(/^NIO_\d{4}_/, '')} · {item.year}
              </div>
              <div className="text-xs text-slate-600">
                Peak intensity: <span className="font-medium text-slate-800">{item.peak_category || 'CS'}</span>
              </div>
              <div className="text-xs text-slate-600">
                Similarity: <span className="font-medium text-slate-800">{item.distance.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-slate-500">
          No historical analogs available.
        </p>
      )}
    </div>
  );
};
