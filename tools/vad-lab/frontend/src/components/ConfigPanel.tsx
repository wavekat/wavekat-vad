import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { VadConfig, ParamInfo } from "@/lib/websocket";

interface ConfigPanelProps {
  configs: VadConfig[];
  backends: Record<string, ParamInfo[]>;
  onConfigsChange: (configs: VadConfig[]) => void;
}

export function ConfigPanel({ configs, backends, onConfigsChange }: ConfigPanelProps) {
  const [nextId, setNextId] = useState(1);

  const addConfig = () => {
    const backendNames = Object.keys(backends);
    if (backendNames.length === 0) return;

    const backend = backendNames[0];
    const params: Record<string, unknown> = {};
    for (const p of backends[backend]) {
      params[p.name] = p.default;
    }

    const id = `config-${nextId}`;
    setNextId((n) => n + 1);

    onConfigsChange([
      ...configs,
      {
        id,
        label: `${backend}-${nextId}`,
        backend,
        params,
      },
    ]);
  };

  const removeConfig = (id: string) => {
    onConfigsChange(configs.filter((c) => c.id !== id));
  };

  const updateConfig = (id: string, updates: Partial<VadConfig>) => {
    onConfigsChange(
      configs.map((c) => {
        if (c.id !== id) return c;
        const updated = { ...c, ...updates };

        // If backend changed, reset params to defaults
        if (updates.backend && updates.backend !== c.backend) {
          const newParams: Record<string, unknown> = {};
          for (const p of backends[updates.backend] ?? []) {
            newParams[p.name] = p.default;
          }
          updated.params = newParams;
        }

        return updated;
      })
    );
  };

  const updateParam = (configId: string, paramName: string, value: unknown) => {
    onConfigsChange(
      configs.map((c) => {
        if (c.id !== configId) return c;
        return { ...c, params: { ...c.params, [paramName]: value } };
      })
    );
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">VAD Configurations</h3>
        <Button size="sm" variant="outline" onClick={addConfig}>
          + Add Config
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {configs.map((config) => (
          <Card key={config.id} className="relative">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">
                  <input
                    className="bg-transparent border-none outline-none w-full text-sm font-semibold"
                    value={config.label}
                    onChange={(e) => updateConfig(config.id, { label: e.target.value })}
                  />
                </CardTitle>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 w-6 p-0 text-muted-foreground"
                  onClick={() => removeConfig(config.id)}
                >
                  x
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1">
                <Label className="text-xs">Backend</Label>
                <Select
                  value={config.backend}
                  onValueChange={(v) => { if (v) updateConfig(config.id, { backend: v }); }}
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.keys(backends).map((b) => (
                      <SelectItem key={b} value={b}>
                        {b}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {(backends[config.backend] ?? []).map((param) => (
                <div key={param.name} className="space-y-1">
                  <Label className="text-xs">{param.description}</Label>
                  {param.param_type.type === "Select" && (
                    <Select
                      value={String(config.params[param.name] ?? param.default)}
                      onValueChange={(v) => { if (v) updateParam(config.id, param.name, v); }}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {param.param_type.options.map((opt) => (
                          <SelectItem key={opt} value={opt}>
                            {opt}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
