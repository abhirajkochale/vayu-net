import React from 'react';

interface TechnicalDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  metadata?: {
    inference_id?: string;
    execution_timestamp_utc?: string;
    model_versions?: {
      center?: string;
      track?: string;
      intensity_wind?: string;
    };
    scientific_disclaimer?: string;
  } | null;
  saliencyDetails?: {
    target_layer?: string;
    target_head_explained?: string;
  } | null;
}

export const TechnicalDetailsModal: React.FC<TechnicalDetailsModalProps> = ({
  isOpen,
  onClose,
  metadata,
  saliencyDetails,
}) => {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="bg-white border border-slate-300 rounded-lg max-w-2xl w-full p-6 space-y-5 my-8 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <div>
            <h2 className="text-base font-bold text-slate-900">Model Specifications & Architecture</h2>
            <p className="text-xs text-slate-500">Technical documentation and verification details</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-700 text-sm font-semibold p-1"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="space-y-4 max-h-[65vh] overflow-y-auto pr-1 text-xs text-slate-600">
          {/* Active Model Versions */}
          <div className="bg-slate-50 rounded border border-slate-200 p-4 space-y-2.5">
            <div className="font-semibold text-slate-800">Pipeline Model Components</div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="bg-white p-2.5 rounded border border-slate-200">
                <div className="text-[11px] text-slate-500">Center Localization</div>
                <div className="font-semibold text-slate-900 mt-0.5">Phase 3C</div>
                <div className="text-[11px] text-slate-500">ResNet-18 Soft-Argmax</div>
              </div>
              <div className="bg-white p-2.5 rounded border border-slate-200">
                <div className="text-[11px] text-slate-500">Track Forecasting</div>
                <div className="font-semibold text-slate-900 mt-0.5">Phase 5B Variant A</div>
                <div className="text-[11px] text-slate-500">Hybrid Residual GRU</div>
              </div>
              <div className="bg-white p-2.5 rounded border border-slate-200">
                <div className="text-[11px] text-slate-500">Intensity & Wind</div>
                <div className="font-semibold text-slate-900 mt-0.5">Phase 6</div>
                <div className="text-[11px] text-slate-500">Multi-Task Conv-GRU</div>
              </div>
            </div>
            {metadata?.inference_id && (
              <div className="text-[11px] font-mono text-slate-500 pt-1 flex justify-between">
                <span>Inference ID: {metadata.inference_id}</span>
                <span>UTC: {metadata.execution_timestamp_utc?.slice(0, 19).replace('T', ' ')}</span>
              </div>
            )}
          </div>

          {/* Architectural Specifications */}
          <div className="bg-slate-50 rounded border border-slate-200 p-4 space-y-2">
            <div className="font-semibold text-slate-800">Architectural Specifications</div>
            <ul className="list-disc list-inside space-y-1 text-slate-600 leading-relaxed">
              <li>
                <strong>Spatial Feature Extractor:</strong> ResNet-18 backbone trained on full-basin GridSat-B1 11µm infrared fields (572 × 929 grid, 0.07° resolution).
              </li>
              <li>
                <strong>Center Localization:</strong> Continuous 2D Soft-Argmax expectation coordinate head eliminating global pooling spatial collapse.
              </li>
              <li>
                <strong>Sequence Recurrence:</strong> 6-frame causal sequence (t - 15h to t0 at 3-hourly intervals) fed into a recurrent Gated Recurrent Unit (GRU).
              </li>
              <li>
                <strong>P80 Empirical Uncertainty:</strong> 80th-percentile error cones derived directly from historical validation residuals.
              </li>
              {saliencyDetails && (
                <li>
                  <strong>Interpretability Layer:</strong> Grad-CAM gradients evaluated at <code>{saliencyDetails.target_layer}</code> explaining the <code>{saliencyDetails.target_head_explained}</code> prediction head.
                </li>
              )}
            </ul>
          </div>

          {/* Dataset & Controlled Ablations */}
          <div className="bg-slate-50 rounded border border-slate-200 p-4 space-y-1.5">
            <div className="font-semibold text-slate-800">Dataset & Multimodal Ablations</div>
            <p className="leading-relaxed text-slate-600">
              Trained on 1998–2024 North Indian Ocean cyclones from the IMD digital best-track repository. Controlled ablation studies (EXP-M1, EXP-M2, EXP-M3) evaluated NASA IMERG multimodal fusion; unimodal satellite infrared demonstrated superior generalizability and was selected as the operational standard.
            </p>
          </div>

          {/* Operational Advisory */}
          <div className="p-3 bg-amber-50 border border-amber-200 rounded text-[11px] text-amber-900 leading-relaxed">
            <strong>Operational Advisory:</strong> VAYU-NET is an artificial intelligence decision-support tool. Model predictions should always be reviewed alongside official bulletins from the India Meteorological Department (IMD) and RSMC New Delhi.
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end pt-3 border-t border-slate-200">
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
