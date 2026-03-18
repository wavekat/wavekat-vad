import { useRef, useEffect, useCallback } from "react";

interface VadTimelineProps {
  label: string;
  results: Array<{ timestamp_ms: number; probability: number }>;
  totalDurationMs: number;
  width?: number;
  height?: number;
  color?: string;
  className?: string;
  hoverTimeMs?: number | null;
  onHoverTimeChange?: (timeMs: number | null) => void;
}

export function VadTimeline({
  label,
  results,
  totalDurationMs,
  width = 800,
  height = 40,
  color = "#22c55e",
  className,
  hoverTimeMs,
  onHoverTimeChange,
}: VadTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    if (results.length === 0 || totalDurationMs <= 0) return;

    // Draw speech probability as filled bars
    for (const result of results) {
      const x = (result.timestamp_ms / totalDurationMs) * width;
      const barWidth = Math.max(1, width / results.length);
      const barHeight = result.probability * height;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.3 + result.probability * 0.7;
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);
    }

    ctx.globalAlpha = 1;

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

      // Find probability at hover time
      const closestResult = results.reduce<{ timestamp_ms: number; probability: number } | null>(
        (closest, r) => {
          if (!closest) return r;
          return Math.abs(r.timestamp_ms - hoverTimeMs) < Math.abs(closest.timestamp_ms - hoverTimeMs)
            ? r
            : closest;
        },
        null
      );

      if (closestResult && Math.abs(closestResult.timestamp_ms - hoverTimeMs) < 100) {
        const probStr = (closestResult.probability * 100).toFixed(0) + "%";
        ctx.font = "10px monospace";
        ctx.fillStyle = "#f97316";
        const textWidth = ctx.measureText(probStr).width;
        const labelX = x + 4 > width - textWidth - 4 ? x - textWidth - 4 : x + 4;
        ctx.fillText(probStr, labelX, height / 2 + 4);
      }
    }
  }, [results, totalDurationMs, width, height, color, hoverTimeMs]);

  return (
    <div className={className}>
      <div className="text-xs text-muted-foreground mb-1 font-mono">{label}</div>
      <div
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ width, height, cursor: "crosshair" }}
      >
        <canvas
          ref={canvasRef}
          style={{ width, height }}
        />
      </div>
    </div>
  );
}
