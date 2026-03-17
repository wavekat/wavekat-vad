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
import { ConfigPanel } from "@/components/ConfigPanel";
import {
  VadLabSocket,
  type AudioDevice,
  type VadConfig,
  type ParamInfo,
  type ServerMessage,
} from "@/lib/websocket";

const COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

function App() {
  const socketRef = useRef<VadLabSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [backends, setBackends] = useState<Record<string, ParamInfo[]>>({});
  const [selectedDevice, setSelectedDevice] = useState<string>("");
  const [recording, setRecording] = useState(false);
  const [configs, setConfigs] = useState<VadConfig[]>([]);
  const [samples, setSamples] = useState<number[]>([]);
  const [vadResults, setVadResults] = useState<
    Record<string, Array<{ timestamp_ms: number; probability: number }>>
  >({});
  const [totalDurationMs, setTotalDurationMs] = useState(0);

  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case "devices":
        setDevices(msg.devices);
        if (msg.devices.length > 0) {
          setSelectedDevice(String(msg.devices[0].index));
        }
        break;

      case "backends":
        setBackends(msg.backends);
        break;

      case "audio":
        setSamples((prev) => [...prev, ...msg.samples]);
        setTotalDurationMs(msg.timestamp_ms + 20);
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

      case "done":
        setRecording(false);
        break;

      case "error":
        console.error("Server error:", msg.message);
        break;
    }
  }, []);

  useEffect(() => {
    const socket = new VadLabSocket();
    socketRef.current = socket;

    socket
      .connect()
      .then(() => {
        setConnected(true);
        socket.send({ type: "list_devices" });
        socket.send({ type: "list_backends" });
      })
      .catch((err) => {
        console.error("Connection failed:", err);
      });

    const unsub = socket.onMessage(handleMessage);

    return () => {
      unsub();
      socket.disconnect();
    };
  }, [handleMessage]);

  const startRecording = () => {
    const socket = socketRef.current;
    if (!socket || !connected) return;

    setSamples([]);
    setVadResults({});
    setTotalDurationMs(0);

    socket.send({ type: "set_configs", configs });
    socket.send({
      type: "start_recording",
      device_index: parseInt(selectedDevice),
      sample_rate: 16000,
    });
    setRecording(true);
  };

  const stopRecording = () => {
    const socket = socketRef.current;
    if (!socket) return;

    socket.send({ type: "stop_recording" });
    setRecording(false);
  };

  return (
    <div className="min-h-screen bg-background p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">vad-lab</h1>
        <span className={`text-xs ${connected ? "text-green-500" : "text-red-500"}`}>
          {connected ? "connected" : "disconnected"}
        </span>
      </div>

      <Separator />

      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <Select value={selectedDevice} onValueChange={(v) => { if (v) setSelectedDevice(v); }}>
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Select microphone" />
          </SelectTrigger>
          <SelectContent>
            {devices.map((d) => (
              <SelectItem key={d.index} value={String(d.index)}>
                {d.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {!recording ? (
          <Button onClick={startRecording} disabled={!connected || configs.length === 0}>
            Record
          </Button>
        ) : (
          <Button variant="destructive" onClick={stopRecording}>
            Stop
          </Button>
        )}
      </div>

      <Separator />

      {/* Waveform */}
      <div>
        <h3 className="text-sm font-medium mb-2">Waveform</h3>
        <Waveform samples={samples} width={800} height={120} className="border rounded" />
      </div>

      {/* VAD Timelines */}
      {configs.map((config, i) => (
        <VadTimeline
          key={config.id}
          label={config.label}
          results={vadResults[config.id] ?? []}
          totalDurationMs={totalDurationMs}
          width={800}
          height={32}
          color={COLORS[i % COLORS.length]}
        />
      ))}

      <Separator />

      {/* Config Panel */}
      <ConfigPanel configs={configs} backends={backends} onConfigsChange={setConfigs} />
    </div>
  );
}

export default App;
