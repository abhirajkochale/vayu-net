import React from 'react';

interface HeaderProps {
  pipelineMode: 'HISTORICAL' | 'CURRENT / REPLAY';
  setPipelineMode: (mode: 'HISTORICAL' | 'CURRENT / REPLAY') => void;
  healthStatus?: string;
  triggerDemoMode: () => void;
  openTechnicalDetails: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  pipelineMode,
  setPipelineMode,
  healthStatus,
  triggerDemoMode,
  openTechnicalDetails,
}) => {
  const isHealthy = healthStatus === 'healthy' || healthStatus === 'OPERATIONAL' || !healthStatus;
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Left: Branding & Tag */}
        <div className="flex items-center gap-3">
          <div className="flex items-baseline gap-2">
            <span className="text-base font-bold text-slate-900 tracking-tight">VAYU-NET</span>
            <span className="text-xs text-slate-500 font-normal hidden sm:inline">
              Tropical Cyclone Intelligence
            </span>
          </div>
          <span className="text-[11px] font-medium text-slate-500 bg-slate-100 border border-slate-200 px-1.5 py-0.5 rounded">
            SIH 26070
          </span>
        </div>

        {/* Center: Historical | Current / Replay */}
        <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200">
          <button
            type="button"
            onClick={() => setPipelineMode('HISTORICAL')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              pipelineMode === 'HISTORICAL'
                ? 'bg-white text-slate-900 shadow-sm border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Historical
          </button>
          <button
            type="button"
            onClick={() => setPipelineMode('CURRENT / REPLAY')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              pipelineMode === 'CURRENT / REPLAY'
                ? 'bg-white text-slate-900 shadow-sm border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Current / Replay
          </button>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={triggerDemoMode}
            className="text-xs font-medium bg-[#D97706] hover:bg-amber-700 text-white px-2.5 py-1.5 rounded-md transition-colors"
            title="Load Deterministic Reference Scenario (Cyclone AMPHAN)"
          >
            SIH Rehearsal Demo
          </button>

          <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-600">
            <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-[#15803D]' : 'bg-rose-500'}`} />
            <span>System Ready</span>
          </div>

          <button
            type="button"
            onClick={openTechnicalDetails}
            className="text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-300 px-2.5 py-1.5 rounded-md transition-colors"
          >
            Model Specs
          </button>
        </div>
      </div>
    </header>
  );
};
