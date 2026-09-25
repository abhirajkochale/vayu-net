import React, { useState } from 'react';
import { resolveAssetUrl } from '../config/api';

interface SaliencyData {
  status?: string;
  original_url?: string;
  heatmap_url?: string;
  overlay_url?: string;
  interpretation_note?: string;
}

interface AiInterpretabilityProps {
  saliency?: SaliencyData | null;
  cycloneName: string;
}

export const AiInterpretability: React.FC<AiInterpretabilityProps> = ({
  saliency,
}) => {
  const [expandedImage, setExpandedImage] = useState<{ url: string; title: string } | null>(null);

  if (!saliency || saliency.status !== 'AVAILABLE') {
    return null;
  }

  const items = [
    {
      title: 'Original',
      url: saliency.original_url ? resolveAssetUrl(saliency.original_url) : '',
    },
    {
      title: 'AI Focus',
      url: saliency.heatmap_url ? resolveAssetUrl(saliency.heatmap_url) : '',
    },
    {
      title: 'Overlay',
      url: saliency.overlay_url ? resolveAssetUrl(saliency.overlay_url) : '',
    },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
      <div className="border-b border-slate-100 pb-2.5">
        <h2 className="text-xs font-semibold text-slate-900">Where the model is looking</h2>
        <p className="text-xs text-slate-500">
          Shows the image regions contributing to the model output.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {items.map((item, idx) => (
          <div key={idx} className="space-y-1.5">
            <div className="text-[11px] font-medium text-slate-600">
              {item.title}
            </div>
            <div
              onClick={() => item.url && setExpandedImage({ url: item.url, title: item.title })}
              className="aspect-square w-full rounded border border-slate-200 overflow-hidden bg-slate-100 cursor-pointer hover:border-slate-400 transition-colors"
            >
              {item.url ? (
                <img
                  src={item.url}
                  alt={item.title}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-xs text-slate-400">
                  Image loading...
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox Modal */}
      {expandedImage && (
        <div
          className="fixed inset-0 z-50 bg-black/50 backdrop-blur-none flex items-center justify-center p-4"
          onClick={() => setExpandedImage(null)}
        >
          <div
            className="bg-white border border-slate-300 rounded-lg max-w-xl w-full p-4 space-y-3 shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-200 pb-2">
              <span className="text-xs font-semibold text-slate-900">{expandedImage.title}</span>
              <button
                type="button"
                onClick={() => setExpandedImage(null)}
                className="text-slate-500 hover:text-slate-800 text-xs px-2 py-0.5 rounded"
              >
                Close
              </button>
            </div>
            <div className="bg-slate-100 rounded overflow-hidden flex items-center justify-center">
              <img
                src={expandedImage.url}
                alt={expandedImage.title}
                className="max-h-[60vh] w-auto object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
