import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Waveform } from "@/components/Waveform";
import { VadTimeline } from "@/components/VadTimeline";
import { FrequencySpectrum } from "@/components/FrequencySpectrum";
import { ConfigPanel } from "@/components/ConfigPanel";
import { LogPanel } from "@/components/LogPanel";
import { ZoomControls } from "@/components/ZoomControls";
import {
  type Viewport,
  createDefaultViewport,
  calculateViewDuration,
} from "@/lib/viewport";
import { useAudioPlayback } from "@/lib/useAudioPlayback";
import {
  VadLabSocket,
  type AudioDevice,
  type VadConfig,
  type ParamInfo,
  type ServerMessage,
  type ConnectionState,
  type LogEntry,
} from "@/lib/websocket";

const COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];
const MAX_LOG_ENTRIES = 500;
const MAX_RECORDING_DURATION_SECS = 120; // 2 minutes

interface SpectrumFrame {
  timestamp_ms: number;
  magnitudes: number[];
}

function App() {
  const socketRef = useRef<VadLabSocket | null>(null);
  const waveformContainerRef = useRef<HTMLDivElement>(null);
  const recordingRef = useRef(false);
  const [containerWidth, setContainerWidth] = useState(800);
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [backends, setBackends] = useState<Record<string, ParamInfo[]>>({});
  const [preprocessingParams, setPreprocessingParams] = useState<ParamInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>("");
  const [recording, setRecording] = useState(false);
  const [configs, setConfigs] = useState<VadConfig[]>([]);
  const [samples, setSamples] = useState<number[]>([]);
  const [spectrumData, setSpectrumData] = useState<SpectrumFrame[]>([]);
  const [vadResults, setVadResults] = useState<
    Record<string, Array<{ timestamp_ms: number; probability: number }>>
  >({});
  const [totalDurationMs, setTotalDurationMs] = useState(0);
  const [sampleRate, setSampleRate] = useState<number | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logsOpen, setLogsOpen] = useState(true);
  const [hoverTimeMs, setHoverTimeMs] = useState<number | null>(null);
  const [viewport, setViewport] = useState<Viewport>(createDefaultViewport);
  const [spectrumBins, setSpectrumBins] = useState(256);

  // Preprocessed data per config
  const [preprocessedSamples, setPreprocessedSamples] = useState<Record<string, number[]>>({});
  const [preprocessedSpectrumData, setPreprocessedSpectrumData] = useState<
    Record<string, SpectrumFrame[]>
  >({});
  // Which configs should display their preprocessed waveform/spectrum
  const [showPreprocessed, setShowPreprocessed] = useState<Record<string, boolean>>({});

  // Playback source: "original" or a config ID for preprocessed audio
  const [playbackSource, setPlaybackSource] = useState<string>("original");

  const connected = connectionState === "connected";

  // Resolve playback samples based on selected source
  const playbackSamples =
    playbackSource === "original" ? samples : (preprocessedSamples[playbackSource] ?? []);
  const playbackSampleRate =
    playbackSource === "original"
      ? sampleRate
      : // Preprocessed audio is at the same sample rate; if sample count differs
        // slightly due to internal buffering, derive effective rate from duration
        totalDurationMs > 0 && (preprocessedSamples[playbackSource]?.length ?? 0) > 0
        ? Math.round((preprocessedSamples[playbackSource].length / totalDurationMs) * 1000)
        : sampleRate;

  // Audio playback
  const playback = useAudioPlayback({
    samples: playbackSamples,
    sampleRate: playbackSampleRate,
  });

  const addLog = useCallback((entry: LogEntry) => {
    setLogs((prev) => {
      const next = [...prev, entry];
      return next.length > MAX_LOG_ENTRIES ? next.slice(-MAX_LOG_ENTRIES) : next;
    });
  }, []);

  // Responsive width measurement
  useEffect(() => {
    const container = waveformContainerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width && width > 0) {
        setContainerWidth(width);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const fetchInitialData = useCallback((socket: VadLabSocket) => {
    socket.send({ type: "list_devices" });
    socket.send({ type: "list_backends" });
  }, []);

  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case "devices":
        setDevices(msg.devices);
        setSelectedDevice((prev) => {
          const stillExists = msg.devices.some((d) => String(d.index) === prev);
          if (stillExists) return prev;
          return msg.devices.length > 0 ? String(msg.devices[0].index) : "";
        });
        break;

      case "backends":
        setBackends(msg.backends);
        setPreprocessingParams(msg.preprocessing_params);
        break;

      case "recording_started":
        setSampleRate(msg.sample_rate);
        break;

      case "audio": {
        setSamples((prev) => [...prev, ...msg.samples]);
        const newEndMs = msg.timestamp_ms + 20;
        setTotalDurationMs(newEndMs);

        // Update viewport during recording so it's set correctly when recording stops
        if (recordingRef.current) {
          setViewport((prev) => ({
            ...prev,
            viewStartMs: Math.max(0, newEndMs - prev.viewDurationMs),
          }));
        }
        break;
      }

      case "spectrum":
        setSpectrumData((prev) => [
          ...prev,
          { timestamp_ms: msg.timestamp_ms, magnitudes: msg.magnitudes },
        ]);
        break;

      case "vad":
        setVadResults((prev) => ({
          ...prev,
          [msg.config_id]: [
            ...(prev[msg.config_id] ?? []),
            { timestamp_ms: msg.timestamp_ms, probability: msg.probability },
          ],
        }));
        break;

      case "preprocessed_audio":
        setPreprocessedSamples((prev) => ({
          ...prev,
          [msg.config_id]: [...(prev[msg.config_id] ?? []), ...msg.samples],
        }));
        break;

      case "preprocessed_spectrum":
        setPreprocessedSpectrumData((prev) => ({
          ...prev,
          [msg.config_id]: [
            ...(prev[msg.config_id] ?? []),
            { timestamp_ms: msg.timestamp_ms, magnitudes: msg.magnitudes },
          ],
        }));
        break;

      case "done":
        recordingRef.current = false;
        setRecording(false);
        break;

      case "error":
        break;
    }
  }, []);

  useEffect(() => {
    const socket = new VadLabSocket();
    socketRef.current = socket;

    const unsubMsg = socket.onMessage(handleMessage);
    const unsubLog = socket.onLog(addLog);

    const unsubConn = socket.onConnection((state) => {
      setConnectionState(state);
      if (state === "connected") {
        fetchInitialData(socket);
      }
      if (state === "reconnecting") {
        recordingRef.current = false;
        setRecording(false);
      }
    });

    socket.connect().catch(() => {
      // reconnect is handled internally
    });

    return () => {
      unsubMsg();
      unsubConn();
      unsubLog();
      socket.disconnect();
    };
  }, [handleMessage, addLog, fetchInitialData]);

  const startRecording = () => {
    const socket = socketRef.current;
    if (!socket || !connected) return;

    setSamples([]);
    setSpectrumData([]);
    setVadResults({});
    setPreprocessedSamples({});
    setPreprocessedSpectrumData({});
    setPlaybackSource("original");
    setTotalDurationMs(0);
    setSampleRate(null);
    // Calculate viewport duration based on container width for consistent scroll speed
    setViewport({
      viewStartMs: 0,
      viewDurationMs: calculateViewDuration(containerWidth),
    });

    socket.send({ type: "set_configs", configs });
    socket.send({
      type: "start_recording",
      device_index: parseInt(selectedDevice),
      max_duration_secs: MAX_RECORDING_DURATION_SECS,
    });
    recordingRef.current = true;
    setRecording(true);
  };

  const stopRecording = () => {
    const socket = socketRef.current;
    if (!socket) return;

    socket.send({ type: "stop_recording" });
    recordingRef.current = false;
    setRecording(false);
  };

  const handleSpectrumBinsChange = useCallback((bins: number) => {
    setSpectrumBins(bins);
    socketRef.current?.send({ type: "set_spectrum_bins", bins });
  }, []);

  const connectionColor: Record<ConnectionState, string> = {
    connected: "text-green-500",
    connecting: "text-yellow-500",
    reconnecting: "text-yellow-500",
    disconnected: "text-red-500",
  };

  return (
    <div className="min-h-screen bg-background p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <h1 className="text-xl font-bold">vad-lab</h1>
          <span className="text-xs text-muted-foreground">by WaveKat</span>
        </div>
        <span className={`text-xs ${connectionColor[connectionState]}`}>
          {connectionState}
        </span>
      </div>

      <Separator />

      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <Select value={selectedDevice} onValueChange={(v) => { if (v) setSelectedDevice(v); }}>
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Select microphone">
              {devices.find((d) => String(d.index) === selectedDevice)?.name}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {devices.map((d) => (
              <SelectItem key={d.index} value={String(d.index)}>
                {d.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          variant="outline"
          disabled={!connected}
          onClick={() => socketRef.current?.send({ type: "list_devices" })}
        >
          Refresh
        </Button>

        {!recording ? (
          <Button onClick={startRecording} disabled={!connected || configs.length === 0}>
            Record
          </Button>
        ) : (
          <Button variant="destructive" onClick={stopRecording}>
            Stop
          </Button>
        )}

        {/* Playback controls */}
        {!recording && samples.length > 0 && (
          <>
            <div className="w-px h-6 bg-border" />
            <Select value={playbackSource} onValueChange={(v) => { if (v) { playback.stop(); setPlaybackSource(v); } }}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="original">Original</SelectItem>
                {configs
                  .filter((c) => (preprocessedSamples[c.id]?.length ?? 0) > 0)
                  .map((c) => (
                    <SelectItem key={c.id} value={c.id}>
                      Preprocessed: {c.label}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
            {playback.canPlay && (
              <>
                {!playback.state.isPlaying ? (
                  <Button variant="outline" onClick={playback.play}>
                    ▶ Play
                  </Button>
                ) : (
                  <Button variant="outline" onClick={playback.pause}>
                    ⏸ Pause
                  </Button>
                )}
                <Button variant="ghost" onClick={playback.stop} disabled={playback.state.positionMs === 0}>
                  ⏹ Stop
                </Button>
              </>
            )}
          </>
        )}
      </div>

      <Separator />

      {/* Waveform and VAD Timelines */}
      <div ref={waveformContainerRef} className="space-y-4">
        <div className="flex items-center gap-2">
          <h3
            className={`text-sm font-medium cursor-pointer select-none inline-flex items-center gap-1.5 ${
              playbackSource === "original" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
            onClick={() => { playback.stop(); setPlaybackSource("original"); }}
          >
            {playbackSource === "original" && <span className="text-xs">&#9835;</span>}
            Waveform
          </h3>
          {!recording && playbackSource === "original" && playback.canPlay && (
            <div className="flex items-center gap-1">
              {!playback.state.isPlaying ? (
                <Button variant="ghost" size="xs" onClick={playback.play}>&#9654;</Button>
              ) : (
                <Button variant="ghost" size="xs" onClick={playback.pause}>&#9646;&#9646;</Button>
              )}
              <Button variant="ghost" size="xs" onClick={playback.stop} disabled={playback.state.positionMs === 0}>&#9632;</Button>
            </div>
          )}
        </div>
        <Waveform
          samples={samples}
          totalDurationMs={totalDurationMs}
          sampleRate={sampleRate}
          viewport={viewport}
          onViewportChange={setViewport}
          width={containerWidth}
          height={120}
          className={`border rounded ${playbackSource === "original" ? "ring-1 ring-foreground/30" : ""}`}
          hoverTimeMs={hoverTimeMs}
          onHoverTimeChange={setHoverTimeMs}
          interactionEnabled={!recording}
          recording={recording}
          playheadMs={!recording && playback.canPlay ? playback.state.positionMs : null}
          onSeek={playback.seek}
          rightControls={
            <ZoomControls
              viewport={viewport}
              totalDurationMs={totalDurationMs}
              onViewportChange={setViewport}
              disabled={recording}
            />
          }
        />

        {/* Spectrogram */}
        <h3 className="text-sm font-medium pt-2">Spectrogram</h3>
        <FrequencySpectrum
          spectrumData={spectrumData}
          sampleRate={sampleRate ?? 48000}
          totalDurationMs={totalDurationMs}
          viewport={viewport}
          onViewportChange={setViewport}
          width={containerWidth}
          height={120}
          className="border rounded"
          hoverTimeMs={hoverTimeMs}
          onHoverTimeChange={setHoverTimeMs}
          recording={recording}
          bins={spectrumBins}
          onBinsChange={handleSpectrumBinsChange}
          playheadMs={!recording && playback.canPlay ? playback.state.positionMs : null}
        />

        {/* VAD Timelines */}
        {configs.length > 0 && (
          <h3 className="text-sm font-medium pt-2">VAD Results</h3>
        )}
        {configs.map((config, i) => (
          <VadTimeline
            key={config.id}
            label={config.label}
            config={config}
            results={vadResults[config.id] ?? []}
            totalDurationMs={totalDurationMs}
            viewport={viewport}
            width={containerWidth}
            height={32}
            color={COLORS[i % COLORS.length]}
            hoverTimeMs={hoverTimeMs}
            onHoverTimeChange={setHoverTimeMs}
            recording={recording}
            playheadMs={!recording && playback.canPlay ? playback.state.positionMs : null}
          />
        ))}

        {/* Preprocessed Waveforms/Spectrograms/VAD - only for configs with showPreprocessed enabled */}
        {configs.filter((c) => showPreprocessed[c.id]).map((config) => {
          const configIndex = configs.findIndex((c) => c.id === config.id);
          const color = COLORS[configIndex % COLORS.length];
          const configSamples = preprocessedSamples[config.id] ?? [];
          const configSpectrum = preprocessedSpectrumData[config.id] ?? [];

          return (
            <div key={`preprocessed-${config.id}`} className="space-y-2 pt-4 border-t">
              <div className="flex items-center gap-2 pt-2">
                <div
                  className={`flex items-center gap-2 cursor-pointer select-none ${
                    playbackSource === config.id ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                  onClick={() => { playback.stop(); setPlaybackSource(config.id); }}
                >
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <h3 className="text-sm font-medium">
                    {playbackSource === config.id && <span className="text-xs mr-1">&#9835;</span>}
                    Preprocessed: {config.label}
                  </h3>
                </div>
                {!recording && playbackSource === config.id && playback.canPlay && (
                  <div className="flex items-center gap-1">
                    {!playback.state.isPlaying ? (
                      <Button variant="ghost" size="xs" onClick={playback.play}>&#9654;</Button>
                    ) : (
                      <Button variant="ghost" size="xs" onClick={playback.pause}>&#9646;&#9646;</Button>
                    )}
                    <Button variant="ghost" size="xs" onClick={playback.stop} disabled={playback.state.positionMs === 0}>&#9632;</Button>
                  </div>
                )}
              </div>
              <Waveform
                samples={configSamples}
                totalDurationMs={totalDurationMs}
                // Don't pass sampleRate for preprocessed audio - let it calculate
                // samplesPerMs from actual samples.length / totalDurationMs.
                // Preprocessing (especially denoise) uses internal buffering that
                // can cause the output sample count to differ from the input.
                viewport={viewport}
                onViewportChange={setViewport}
                width={containerWidth}
                height={80}
                className={`border rounded ${playbackSource === config.id ? "ring-1 ring-foreground/30" : ""}`}
                hoverTimeMs={hoverTimeMs}
                onHoverTimeChange={setHoverTimeMs}
                interactionEnabled={!recording}
                recording={recording}
                playheadMs={!recording && playback.canPlay ? playback.state.positionMs : null}
                onSeek={playback.seek}
              />
              <FrequencySpectrum
                spectrumData={configSpectrum}
                sampleRate={sampleRate ?? 48000}
                totalDurationMs={totalDurationMs}
                viewport={viewport}
                onViewportChange={setViewport}
                width={containerWidth}
                height={80}
                className="border rounded"
                hoverTimeMs={hoverTimeMs}
                onHoverTimeChange={setHoverTimeMs}
                recording={recording}
                bins={spectrumBins}
                onBinsChange={handleSpectrumBinsChange}
                playheadMs={!recording && playback.canPlay ? playback.state.positionMs : null}
              />
              <VadTimeline
                label={config.label}
                config={config}
                results={vadResults[config.id] ?? []}
                totalDurationMs={totalDurationMs}
                viewport={viewport}
                width={containerWidth}
                height={32}
                color={color}
                hoverTimeMs={hoverTimeMs}
                onHoverTimeChange={setHoverTimeMs}
                recording={recording}
                playheadMs={!recording && playback.canPlay ? playback.state.positionMs : null}
              />
            </div>
          );
        })}
      </div>

      <Separator />

      {/* Config Panel */}
      <ConfigPanel
        configs={configs}
        backends={backends}
        preprocessingParams={preprocessingParams}
        onConfigsChange={setConfigs}
        showPreprocessed={showPreprocessed}
        onShowPreprocessedChange={(configId, show) =>
          setShowPreprocessed((prev) => ({ ...prev, [configId]: show }))
        }
      />

      <Separator />

      {/* Logs */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium">Logs</h3>
          <div className="flex gap-2">
            <Button variant="ghost" size="xs" onClick={() => setLogs([])}>
              Clear
            </Button>
            <Button variant="ghost" size="xs" onClick={() => setLogsOpen((v) => !v)}>
              {logsOpen ? "Hide" : "Show"}
            </Button>
          </div>
        </div>
        {logsOpen && <LogPanel logs={logs} maxHeight={240} />}
      </div>
    </div>
  );
}

export default App;
