import React from 'react';

export const DataSources: React.FC = () => {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 py-3 px-1 text-xs text-slate-500 border-t border-slate-200">
      <div className="font-medium text-slate-700">Data sources</div>
      <div className="flex flex-wrap items-center gap-4">
        <span>GridSat-B1 · <span className="text-slate-700 font-medium">Primary</span></span>
        <span>NASA GPM IMERG · <span className="text-slate-700 font-medium">Context</span></span>
        <span>INSAT-3D/3DR · <span className="text-slate-700 font-medium">Planned</span></span>
      </div>
    </div>
  );
};
