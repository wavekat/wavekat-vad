import { useRef, useEffect } from "react";

interface WaveformProps {
  samples: number[];
  width?: number;
  height?: number;
  className?: string;
}

export function Waveform({ samples, width = 800, height = 150, className }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

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

      const yMin = mid + min * mid;
      const yMax = mid + max * mid;

      ctx.moveTo(x, yMin);
      ctx.lineTo(x, yMax);
    }

    ctx.stroke();

    // Center line
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();
  }, [samples, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className={className}
    />
  );
}
