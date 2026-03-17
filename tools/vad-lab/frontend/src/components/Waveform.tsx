import { useRef, useEffect, useState } from "react";

interface WaveformProps {
  samples: number[];
  width?: number;
  height?: number;
  className?: string;
}

const ZOOM_LEVELS = [1, 2, 4, 8, 16, 32];

export function Waveform({ samples, width = 800, height = 150, className }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoomIndex, setZoomIndex] = useState(0);
  const zoom = ZOOM_LEVELS[zoomIndex];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || samples.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    // Draw waveform
    const mid = height / 2;
    const samplesPerPixel = Math.max(1, Math.floor(samples.length / width));

    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let x = 0; x < width; x++) {
      const start = x * samplesPerPixel;
      const end = Math.min(start + samplesPerPixel, samples.length);

      let min = 0;
      let max = 0;
      for (let i = start; i < end; i++) {
        const val = samples[i] / 32768;
        if (val < min) min = val;
        if (val > max) max = val;
      }

      // Apply vertical zoom and clamp to canvas bounds
      const yMin = Math.max(0, mid + min * mid * zoom);
      const yMax = Math.min(height, mid + max * mid * zoom);

      ctx.moveTo(x, yMin);
      ctx.lineTo(x, yMax);
    }

    ctx.stroke();

    // Clipping indicator: draw red lines at top/bottom when signal is clipped
    if (zoom > 1) {
      ctx.strokeStyle = "rgba(239, 68, 68, 0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, 0.5);
      ctx.lineTo(width, 0.5);
      ctx.moveTo(0, height - 0.5);
      ctx.lineTo(width, height - 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Center line
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();
  }, [samples, width, height, zoom]);

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <button
          className="px-2 py-0.5 text-xs border rounded hover:bg-muted disabled:opacity-30"
          disabled={zoomIndex <= 0}
          onClick={() => setZoomIndex((i) => Math.max(0, i - 1))}
        >
          −
        </button>
        <span className="text-xs text-muted-foreground w-10 text-center">{zoom}x</span>
        <button
          className="px-2 py-0.5 text-xs border rounded hover:bg-muted disabled:opacity-30"
          disabled={zoomIndex >= ZOOM_LEVELS.length - 1}
          onClick={() => setZoomIndex((i) => Math.min(ZOOM_LEVELS.length - 1, i + 1))}
        >
          +
        </button>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className={className}
      />
    </div>
  );
}
