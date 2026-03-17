export interface AudioDevice {
  index: number;
  name: string;
}

export interface VadConfig {
  id: string;
  label: string;
  backend: string;
  params: Record<string, unknown>;
}

export interface ParamInfo {
  name: string;
  description: string;
  param_type: { type: "Select"; options: string[] } | { type: "Float"; options: { min: number; max: number } } | { type: "Int"; options: { min: number; max: number } };
  default: unknown;
}

// Server -> Client messages
export type ServerMessage =
  | { type: "devices"; devices: AudioDevice[] }
  | { type: "backends"; backends: Record<string, ParamInfo[]> }
  | { type: "audio"; timestamp_ms: number; samples: number[] }
  | { type: "vad"; config_id: string; timestamp_ms: number; probability: number }
  | { type: "done" }
  | { type: "error"; message: string };

// Client -> Server messages
export type ClientMessage =
  | { type: "list_devices" }
  | { type: "list_backends" }
  | { type: "start_recording"; device_index: number; sample_rate: number }
  | { type: "stop_recording" }
  | { type: "load_file"; path: string }
  | { type: "set_configs"; configs: VadConfig[] };

export type MessageHandler = (msg: ServerMessage) => void;

export class VadLabSocket {
  private ws: WebSocket | null = null;
  private handlers: Set<MessageHandler> = new Set();
  private url: string;

  constructor(url?: string) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    this.url = url ?? `${protocol}//${window.location.host}/ws`;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => resolve();
      this.ws.onerror = () => reject(new Error("WebSocket connection failed"));

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as ServerMessage;
          for (const handler of this.handlers) {
            handler(msg);
          }
        } catch (e) {
          console.error("Failed to parse server message:", e);
        }
      };

      this.ws.onclose = () => {
        console.log("WebSocket disconnected");
      };
    });
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }

  send(msg: ClientMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  onMessage(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
