import { useRef, useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";

interface WaveformProps {
  samples: number[];
  totalDurationMs: number;
  width?: number;
  height?: number;
  className?: string;
  hoverTimeMs?: number | null;
  onHoverTimeChange?: (timeMs: number | null) => void;
}

const ZOOM_LEVELS = [1, 2, 4, 8, 16, 32];

export function Waveform({
  samples,
  totalDurationMs,
  width = 800,
  height = 150,
  className,
  hoverTimeMs,
  onHoverTimeChange,
}: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomIndex, setZoomIndex] = useState(0);
  const zoom = ZOOM_LEVELS[zoomIndex];

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!onHoverTimeChange || totalDurationMs <= 0) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const timeMs = (x / width) * totalDurationMs;
      onHoverTimeChange(Math.max(0, Math.min(totalDurationMs, timeMs)));
    },
    [onHoverTimeChange, totalDurationMs, width]
  );

  const handleMouseLeave = useCallback(() => {
    onHoverTimeChange?.(null);
  }, [onHoverTimeChange]);

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

    // Draw crosshair
    if (hoverTimeMs != null && totalDurationMs > 0) {
      const x = (hoverTimeMs / totalDurationMs) * width;

      // Vertical line
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      // Time label
      const timeStr = (hoverTimeMs / 1000).toFixed(3) + "s";
      ctx.font = "11px monospace";
      ctx.fillStyle = "#f97316";
      const textWidth = ctx.measureText(timeStr).width;
      const labelX = x + 4 > width - textWidth - 4 ? x - textWidth - 4 : x + 4;
      ctx.fillText(timeStr, labelX, 12);
    }
  }, [samples, width, height, zoom, hoverTimeMs, totalDurationMs]);

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <Button
          variant="outline"
          size="xs"
          disabled={zoomIndex <= 0}
          onClick={() => setZoomIndex((i) => Math.max(0, i - 1))}
        >
          −
        </Button>
        <span className="text-xs text-muted-foreground w-10 text-center">{zoom}x</span>
        <Button
          variant="outline"
          size="xs"
          disabled={zoomIndex >= ZOOM_LEVELS.length - 1}
          onClick={() => setZoomIndex((i) => Math.min(ZOOM_LEVELS.length - 1, i + 1))}
        >
          +
        </Button>
      </div>
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ width, height, cursor: "crosshair" }}
      >
        <canvas
          ref={canvasRef}
          style={{ width, height }}
          className={className}
        />
      </div>
    </div>
  );
}
