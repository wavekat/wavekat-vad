import { useRef, useEffect, useCallback } from "react";
import { type Viewport, pixelToTime, timeToPixel } from "@/lib/viewport";

interface VadTimelineProps {
  label: string;
  results: Array<{ timestamp_ms: number; probability: number }>;
  totalDurationMs: number;
  viewport: Viewport;
  width?: number;
  height?: number;
  color?: string;
  className?: string;
  hoverTimeMs?: number | null;
  onHoverTimeChange?: (timeMs: number | null) => void;
  /** When true, "now" is anchored to right edge */
  recording?: boolean;
  /** Current playhead position in milliseconds (for playback) */
  playheadMs?: number | null;
}

export function VadTimeline({
  label,
  results,
  totalDurationMs,
  viewport,
  width = 800,
  height = 40,
  color = "#22c55e",
  className,
  hoverTimeMs,
  onHoverTimeChange,
  recording = false,
  playheadMs,
}: VadTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // During recording, anchor "now" to the right edge
  const effectiveViewport = recording
    ? {
        viewStartMs: totalDurationMs - viewport.viewDurationMs,
        viewDurationMs: viewport.viewDurationMs,
      }
    : viewport;

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!onHoverTimeChange) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const timeMs = pixelToTime(x, width, effectiveViewport);
      onHoverTimeChange(Math.max(0, Math.min(totalDurationMs, timeMs)));
    },
    [onHoverTimeChange, totalDurationMs, width, effectiveViewport]
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

    // Draw light grey background for recorded area
    if (totalDurationMs > 0) {
      const recordedStartX = Math.max(
        0,
        ((0 - effectiveViewport.viewStartMs) / effectiveViewport.viewDurationMs) * width
      );
      const recordedEndX = Math.min(
        width,
        ((totalDurationMs - effectiveViewport.viewStartMs) / effectiveViewport.viewDurationMs) * width
      );
      if (recordedEndX > recordedStartX) {
        ctx.fillStyle = "#f3f4f6"; // light grey
        ctx.fillRect(recordedStartX, 0, recordedEndX - recordedStartX, height);
      }
    }

    if (results.length === 0) return;

    // Filter to visible results and render them
    const viewEndMs = effectiveViewport.viewStartMs + effectiveViewport.viewDurationMs;

    // Draw speech probability as filled bars
    for (const result of results) {
      // Skip results outside viewport (effectiveViewport.viewStartMs can be negative)
      if (result.timestamp_ms < effectiveViewport.viewStartMs - 50) continue;
      if (result.timestamp_ms > viewEndMs + 50) continue;

      const x = timeToPixel(result.timestamp_ms, width, effectiveViewport);
      // Calculate bar width based on typical 20ms frame interval
      const barWidth = Math.max(1, (20 / effectiveViewport.viewDurationMs) * width);
      const barHeight = result.probability * height;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.3 + result.probability * 0.7;
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);
    }

    ctx.globalAlpha = 1;

    // Draw crosshair
    if (hoverTimeMs != null) {
      const x = timeToPixel(hoverTimeMs, width, effectiveViewport);

      if (x >= 0 && x <= width) {
        // Vertical line
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Find probability at hover time
        const closestResult = results.reduce<{
          timestamp_ms: number;
          probability: number;
        } | null>((closest, r) => {
          if (!closest) return r;
          return Math.abs(r.timestamp_ms - hoverTimeMs) <
            Math.abs(closest.timestamp_ms - hoverTimeMs)
            ? r
            : closest;
        }, null);

        if (
          closestResult &&
          Math.abs(closestResult.timestamp_ms - hoverTimeMs) < 100
        ) {
          const probStr = (closestResult.probability * 100).toFixed(0) + "%";
          ctx.font = "10px monospace";
          ctx.fillStyle = "#f97316";
          const textWidth = ctx.measureText(probStr).width;
          const labelX =
            x + 4 > width - textWidth - 4 ? x - textWidth - 4 : x + 4;
          ctx.fillText(probStr, labelX, height / 2 + 4);
        }
      }
    }

    // Draw playhead
    if (playheadMs != null && totalDurationMs > 0) {
      const x = timeToPixel(playheadMs, width, effectiveViewport);

      if (x >= 0 && x <= width) {
        // Playhead line (bright blue)
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Show probability at playhead position
        const closestResult = results.reduce<{
          timestamp_ms: number;
          probability: number;
        } | null>((closest, r) => {
          if (!closest) return r;
          return Math.abs(r.timestamp_ms - playheadMs) <
            Math.abs(closest.timestamp_ms - playheadMs)
            ? r
            : closest;
        }, null);

        if (
          closestResult &&
          Math.abs(closestResult.timestamp_ms - playheadMs) < 100
        ) {
          const probStr = (closestResult.probability * 100).toFixed(0) + "%";
          ctx.font = "10px monospace";
          ctx.fillStyle = "#3b82f6";
          const textWidth = ctx.measureText(probStr).width;
          const labelX =
            x + 4 > width - textWidth - 4 ? x - textWidth - 4 : x + 4;
          ctx.fillText(probStr, labelX, height / 2 + 4);
        }
      }
    }
  }, [results, width, height, color, hoverTimeMs, playheadMs, totalDurationMs, effectiveViewport]);

  return (
    <div className={className}>
      <div className="text-xs text-muted-foreground mb-1 font-mono">{label}</div>
      <div
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ width, height, cursor: "crosshair" }}
      >
        <canvas ref={canvasRef} style={{ width, height }} />
      </div>
    </div>
  );
}
