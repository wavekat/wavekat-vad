import { useRef, useEffect } from "react";

interface VadTimelineProps {
  label: string;
  results: Array<{ timestamp_ms: number; probability: number }>;
  totalDurationMs: number;
  width?: number;
  height?: number;
  color?: string;
  className?: string;
}

export function VadTimeline({
  label,
  results,
  totalDurationMs,
  width = 800,
  height = 40,
  color = "#22c55e",
  className,
}: VadTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
  }, [results, totalDurationMs, width, height, color]);

  return (
    <div className={className}>
      <div className="text-xs text-muted-foreground mb-1 font-mono">{label}</div>
      <canvas
        ref={canvasRef}
        style={{ width, height }}
      />
    </div>
  );
}
