import React, { useState, useMemo } from 'react';

export interface ActiveCycloneEvent {
  event_id: string;
  name: string;
  source: string;
  status: string;
  latest_observation: string | null;
  latest_center: { lat: number | null; lon: number | null } | null;
  history_fix_count: number;
  satellite_frame_count: number;
  readiness: string;
}

interface CurrentEventFeedProps {
  events: ActiveCycloneEvent[];
  selectedCyclone: string;
  onSelectCyclone: (name: string, timestamp?: string) => void;
  onRunForecast: () => void;
  isLoading: boolean;
}

export const CurrentEventFeed: React.FC<CurrentEventFeedProps> = ({
  events,
  selectedCyclone,
  onSelectCyclone,
  onRunForecast,
  isLoading,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  // Show only usable events (filter out catalog stubs)
  const usableEvents = useMemo(() => {
    return events.filter(
      (ev) =>
        ev.readiness === 'READY' ||
        ev.readiness === 'INFERENCE_READY' ||
        ev.readiness === 'WAITING_FOR_FRAMES' ||
        (ev.satellite_frame_count > 0 && ev.history_fix_count > 0)
    );
  }, [events]);

  const filteredEvents = useMemo(() => {
    if (!searchQuery.trim()) return usableEvents;
    const q = searchQuery.toLowerCase();
    return usableEvents.filter(
      (ev) =>
        ev.name.toLowerCase().includes(q) ||
        ev.event_id.toLowerCase().includes(q)
    );
  }, [usableEvents, searchQuery]);

  const formatEventDate = (isoStr: string | null) => {
    if (!isoStr) return 'Date unavailable';
    const d = new Date(isoStr);
    const day = d.getUTCDate();
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[d.getUTCMonth()];
    const year = d.getUTCFullYear();
    const time = d.toISOString().slice(11, 16);
    return `${day} ${month} ${year} · ${time} UTC`;
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Available Cyclone Events</h2>
          <div className="text-xs text-slate-500 mt-0.5">
            Observation source: <strong className="text-slate-700 font-medium">Replay / Historical archive</strong>
          </div>
        </div>

        {/* Search */}
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Search event name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 w-48 text-slate-800"
          />
        </div>
      </div>

      {/* Usable Events Grid */}
      <div className="max-h-60 overflow-y-auto pr-1 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {filteredEvents.map((ev) => {
          const isSelected = selectedCyclone === ev.name || selectedCyclone === ev.event_id;
          const isReady = ev.readiness === 'READY' || ev.readiness === 'INFERENCE_READY';

          return (
            <div
              key={ev.event_id}
              onClick={() => onSelectCyclone(ev.name, ev.latest_observation || undefined)}
              className={`p-3 rounded border cursor-pointer transition-colors space-y-2 ${
                isSelected
                  ? 'bg-blue-50/50 border-blue-500 ring-1 ring-blue-500/20'
                  : 'bg-slate-50 border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-slate-900">
                  {ev.name.replace(/^NIO_\d{4}_/, '')}
                </span>
                <span className="text-[10px] font-semibold text-[#15803D] bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded">
                  {isReady ? 'READY' : ev.readiness}
                </span>
              </div>

              <div className="text-xs text-slate-500">
                {formatEventDate(ev.latest_observation)}
              </div>

              <div className="text-xs text-slate-600">
                {ev.satellite_frame_count}/6 observations · {ev.history_fix_count} track fixes
              </div>
            </div>
          );
        })}
      </div>

      {/* Action Strip */}
      <div className="pt-2 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs text-slate-600">
          Selected: <strong className="text-slate-900">{selectedCyclone}</strong>
        </div>

        <button
          type="button"
          onClick={onRunForecast}
          disabled={isLoading}
          className="bg-[#2563EB] hover:bg-blue-700 disabled:bg-slate-300 text-white text-xs font-semibold px-4 py-2 rounded transition-colors cursor-pointer"
        >
          {isLoading ? 'Running...' : 'Run forecast'}
        </button>
      </div>
    </div>
  );
};
