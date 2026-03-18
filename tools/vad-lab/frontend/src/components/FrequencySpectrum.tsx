import { useRef, useEffect, useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  type Viewport,
  pixelToTime,
  timeToPixel,
  panViewport,
  zoomViewport,
} from "@/lib/viewport";

interface SpectrumFrame {
  timestamp_ms: number;
  magnitudes: number[];
}

interface FrequencySpectrumProps {
  /** Array of spectrum frames over time */
  spectrumData: SpectrumFrame[];
  /** Sample rate in Hz, used to calculate frequency labels */
  sampleRate: number;
  /** Total duration of the recording in ms */
  totalDurationMs: number;
  /** Current viewport (time window) */
  viewport: Viewport;
  /** Callback when viewport changes */
  onViewportChange: (viewport: Viewport) => void;
  /** Width of the canvas in pixels */
  width?: number;
  /** Height of the canvas in pixels */
  height?: number;
  /** Additional CSS classes */
  className?: string;
  /** Current hover time for crosshair sync */
  hoverTimeMs?: number | null;
  /** Callback when hover time changes */
  onHoverTimeChange?: (timeMs: number | null) => void;
  /** Whether we're in recording mode (anchors view to right edge) */
  recording?: boolean;
  /** Current number of frequency bins */
  bins?: number;
  /** Callback when bins setting changes */
  onBinsChange?: (bins: number) => void;
}

type FreqScale = "linear" | "log";

/** Convert linear frequency to logarithmic position (0-1) */
function freqToLogPosition(freq: number, minFreq: number, maxFreq: number): number {
  if (freq <= minFreq) return 0;
  if (freq >= maxFreq) return 1;
  return Math.log(freq / minFreq) / Math.log(maxFreq / minFreq);
}

/** Map dB value (-80 to 0) to a color */
function dbToColor(db: number): string {
  // Normalize to 0-1
  const normalized = (db + 80) / 80;
  const clamped = Math.max(0, Math.min(1, normalized));

  // Color gradient: dark blue -> blue -> cyan -> green -> yellow -> red
  if (clamped < 0.2) {
    // Dark blue to blue
    const t = clamped / 0.2;
    return `rgb(0, 0, ${Math.round(50 + t * 155)})`;
  } else if (clamped < 0.4) {
    // Blue to cyan
    const t = (clamped - 0.2) / 0.2;
    return `rgb(0, ${Math.round(t * 255)}, ${Math.round(205 + t * 50)})`;
  } else if (clamped < 0.6) {
    // Cyan to green
    const t = (clamped - 0.4) / 0.2;
    return `rgb(0, 255, ${Math.round(255 * (1 - t))})`;
  } else if (clamped < 0.8) {
    // Green to yellow
    const t = (clamped - 0.6) / 0.2;
    return `rgb(${Math.round(t * 255)}, 255, 0)`;
  } else {
    // Yellow to red
    const t = (clamped - 0.8) / 0.2;
    return `rgb(255, ${Math.round(255 * (1 - t))}, 0)`;
  }
}

const BINS_OPTIONS = [32, 64, 128, 256, 512];

export function FrequencySpectrum({
  spectrumData,
  sampleRate,
  totalDurationMs,
  viewport,
  onViewportChange,
  width = 800,
  height = 150,
  className,
  hoverTimeMs,
  onHoverTimeChange,
  recording = false,
  bins = 64,
  onBinsChange,
}: FrequencySpectrumProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [freqScale, setFreqScale] = useState<FreqScale>("log");
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; viewStartMs: number } | null>(null);

  // Frequency range
  const minFreq = 20; // 20 Hz
  const maxFreq = sampleRate / 2; // Nyquist frequency
  const numBins = spectrumData.length > 0 ? spectrumData[0].magnitudes.length : 512;
  const freqPerBin = sampleRate / (numBins * 2);

  // During recording, anchor "now" to the right edge
  const effectiveViewport = recording
    ? {
        viewStartMs: totalDurationMs - viewport.viewDurationMs,
        viewDurationMs: viewport.viewDurationMs,
      }
    : viewport;

  // Refs for throttled rendering
  const rafIdRef = useRef<number | null>(null);
  const lastRenderRef = useRef<{
    dataLength: number;
    viewStartMs: number;
    viewDurationMs: number;
    freqScale: FreqScale;
    hoverTimeMs: number | null;
  } | null>(null);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;

    // Resize canvas if needed
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    } else {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    ctx.clearRect(0, 0, width, height);

    // Dark background for better color contrast
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, width, height);

    if (spectrumData.length === 0) {
      ctx.fillStyle = "#9ca3af";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No spectrum data", width / 2, height / 2);
      return;
    }

    // Find visible spectrum frames
    const visibleFrames = spectrumData.filter(
      (frame) =>
        frame.timestamp_ms >= effectiveViewport.viewStartMs - 50 &&
        frame.timestamp_ms <= effectiveViewport.viewStartMs + effectiveViewport.viewDurationMs + 50
    );

    // Frame duration (assume 20ms per frame)
    const frameDurationMs = 20;
    const frameWidth = Math.max(1, (frameDurationMs / effectiveViewport.viewDurationMs) * width);

    // Draw each visible frame as a vertical column
    for (const frame of visibleFrames) {
      const x = timeToPixel(frame.timestamp_ms, width, effectiveViewport);

      if (x < -frameWidth || x > width + frameWidth) continue;

      // Draw each frequency bin as a horizontal stripe
      for (let binIdx = 0; binIdx < frame.magnitudes.length; binIdx++) {
        const freq = binIdx * freqPerBin;
        if (freq < minFreq || freq > maxFreq) continue;

        let yNorm: number;
        if (freqScale === "log") {
          yNorm = freqToLogPosition(freq, minFreq, maxFreq);
        } else {
          yNorm = freq / maxFreq;
        }

        // Y is inverted: low freq at bottom, high at top
        const y = height * (1 - yNorm);
        const binHeight = Math.max(1, height / frame.magnitudes.length);

        const db = frame.magnitudes[binIdx];
        ctx.fillStyle = dbToColor(db);
        ctx.fillRect(x, y - binHeight / 2, frameWidth + 0.5, binHeight + 0.5);
      }
    }

    // Draw frequency grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 0.5;
    ctx.font = "9px monospace";
    ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
    ctx.textAlign = "left";

    const gridFreqs = [100, 200, 500, 1000, 2000, 5000, 10000, 20000];
    for (const freq of gridFreqs) {
      if (freq > maxFreq) continue;

      let yNorm: number;
      if (freqScale === "log") {
        yNorm = freqToLogPosition(freq, minFreq, maxFreq);
      } else {
        yNorm = freq / maxFreq;
      }

      const y = height * (1 - yNorm);
      if (y > 0 && y < height - 10) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
        ctx.fillText(label, 4, y - 2);
      }
    }

    // Draw hover crosshair
    if (hoverTimeMs != null && totalDurationMs > 0) {
      const x = timeToPixel(hoverTimeMs, width, effectiveViewport);

      if (x >= 0 && x <= width) {
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
    }
  }, [
    spectrumData,
    width,
    height,
    effectiveViewport,
    freqScale,
    hoverTimeMs,
    totalDurationMs,
    sampleRate,
    minFreq,
    maxFreq,
    freqPerBin,
  ]);

  // Throttled render with requestAnimationFrame
  useEffect(() => {
    const currentState = {
      dataLength: spectrumData.length,
      viewStartMs: effectiveViewport.viewStartMs,
      viewDurationMs: effectiveViewport.viewDurationMs,
      freqScale,
      hoverTimeMs: hoverTimeMs ?? null,
    };

    const lastRender = lastRenderRef.current;
    if (
      lastRender &&
      lastRender.dataLength === currentState.dataLength &&
      lastRender.viewStartMs === currentState.viewStartMs &&
      lastRender.viewDurationMs === currentState.viewDurationMs &&
      lastRender.freqScale === currentState.freqScale &&
      lastRender.hoverTimeMs === currentState.hoverTimeMs
    ) {
      return;
    }

    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
    }

    rafIdRef.current = requestAnimationFrame(() => {
      render();
      lastRenderRef.current = currentState;
      rafIdRef.current = null;
    });

    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, [
    spectrumData.length,
    effectiveViewport.viewStartMs,
    effectiveViewport.viewDurationMs,
    freqScale,
    hoverTimeMs,
    render,
  ]);

  // Force re-render when dimensions change
  useEffect(() => {
    lastRenderRef.current = null;
  }, [width, height, sampleRate]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (isDragging && dragStartRef.current) {
        const deltaX = e.clientX - dragStartRef.current.x;
        const newViewport = panViewport(
          { ...viewport, viewStartMs: dragStartRef.current.viewStartMs },
          deltaX,
          width,
          totalDurationMs
        );
        onViewportChange(newViewport);
        return;
      }

      if (!onHoverTimeChange) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const timeMs = pixelToTime(x, width, effectiveViewport);
      onHoverTimeChange(Math.max(0, Math.min(totalDurationMs, timeMs)));
    },
    [
      isDragging,
      viewport,
      width,
      totalDurationMs,
      onViewportChange,
      onHoverTimeChange,
      effectiveViewport,
    ]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (recording) return;
      setIsDragging(true);
      dragStartRef.current = { x: e.clientX, viewStartMs: viewport.viewStartMs };
    },
    [recording, viewport.viewStartMs]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    dragStartRef.current = null;
  }, []);

  const handleMouseLeave = useCallback(() => {
    onHoverTimeChange?.(null);
    if (isDragging) {
      setIsDragging(false);
      dragStartRef.current = null;
    }
  }, [onHoverTimeChange, isDragging]);

  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      if (recording) return;
      if (!e.ctrlKey && !e.metaKey) return;

      e.preventDefault();
      e.stopPropagation();

      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const centerTimeMs = pixelToTime(x, width, effectiveViewport);

      const zoomIn = e.deltaY < 0;
      const newViewport = zoomViewport(viewport, centerTimeMs, zoomIn, totalDurationMs);
      onViewportChange(newViewport);
    },
    [recording, width, viewport, effectiveViewport, totalDurationMs, onViewportChange]
  );

  const cursor = recording ? "default" : isDragging ? "grabbing" : "grab";

  return (
    <div>
      <div className="flex items-center gap-4 mb-1">
        <div className="flex items-center gap-1">
          <span className="text-xs text-muted-foreground">Freq scale:</span>
          <div className="inline-flex rounded-md border border-input">
            <Button
              variant="ghost"
              size="xs"
              className={`rounded-none rounded-l-md border-0 ${
                freqScale === "linear" ? "bg-accent text-accent-foreground" : ""
              }`}
              onClick={() => setFreqScale("linear")}
            >
              Linear
            </Button>
            <Button
              variant="ghost"
              size="xs"
              className={`rounded-none rounded-r-md border-0 border-l border-input ${
                freqScale === "log" ? "bg-accent text-accent-foreground" : ""
              }`}
              onClick={() => setFreqScale("log")}
            >
              Log
            </Button>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-xs text-muted-foreground">Bins:</span>
          <Select
            value={String(bins)}
            onValueChange={(v) => { if (v) onBinsChange?.(parseInt(v)); }}
            disabled={recording}
          >
            <SelectTrigger className="h-6 w-16 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {BINS_OPTIONS.map((b) => (
                <SelectItem key={b} value={String(b)}>
                  {b}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <span className="text-xs text-muted-foreground">
          {(maxFreq / 1000).toFixed(1)}kHz max
        </span>
      </div>
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        style={{ width, height, cursor }}
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
