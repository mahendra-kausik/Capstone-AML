"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getAlerts, getDrift, updateAlert } from "@/lib/api";
import type { AlertItem } from "@/lib/types";

const LEVELS = ["critical", "high", "medium", "low"] as const;

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [driftAlerts, setDriftAlerts] = useState<any[]>([]);
  const [level, setLevel] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getAlerts({ limit: 100 }), getDrift()])
      .then(([a, d]) => {
        setAlerts(a.items);
        setDriftAlerts(d.alerts || []);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load alerts"))
      .finally(() => setLoading(false));
  }, []);

  const filtered = level ? alerts.filter((a) => a.severity === level) : alerts;

  async function acknowledge(id: string) {
    await updateAlert(id, "acknowledged");
    setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, status: "acknowledged" } : a)));
  }

  if (loading) return <Skeleton className="h-96 w-full rounded-xl" />;
  if (error) {
    return (
      <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-8 text-center text-destructive">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        <Button variant={!level ? "default" : "outline"} size="sm" onClick={() => setLevel("")}>
          All ({alerts.length})
        </Button>
        {LEVELS.map((l) => (
          <Button
            key={l}
            variant={level === l ? "default" : "outline"}
            size="sm"
            onClick={() => setLevel(l)}
            className="capitalize"
          >
            {l}
          </Button>
        ))}
      </div>

      {driftAlerts.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-risk-high" /> Model Drift Alerts
          </h2>
          {driftAlerts.map((a, i) => (
            <div key={i} className="glass-card flex items-center justify-between p-4 border-risk-high/20">
              <div>
                <p className="font-medium">{a.model} · {a.window}</p>
                <p className="text-xs text-muted-foreground">Kendall τ instability detected</p>
              </div>
              <Badge variant="high">τ {a.tau?.toFixed(3)}</Badge>
            </div>
          ))}
        </div>
      )}

      <div className="space-y-2">
        <h2 className="text-sm font-semibold">Alert Queue</h2>
        {filtered.length === 0 ? (
          <p className="py-12 text-center text-muted-foreground">No alerts match this filter</p>
        ) : (
          filtered.map((alert) => (
            <div
              key={alert.id}
              className="glass-card flex flex-wrap items-center justify-between gap-4 p-4 transition hover:border-primary/30"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`h-2 w-2 rounded-full ${alert.status === "open" ? "animate-pulse" : ""} ${
                    alert.severity === "critical" ? "bg-risk-critical" : alert.severity === "high" ? "bg-risk-high" : "bg-risk-medium"
                  }`}
                />
                <div>
                  <p className="text-sm font-medium">{alert.title}</p>
                  <p className="text-xs text-muted-foreground">{alert.message}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{new Date(alert.created_at).toLocaleString()}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Badge variant={alert.severity as "critical" | "high" | "medium" | "low"}>{alert.severity}</Badge>
                <Badge variant="secondary">{alert.status}</Badge>
                {alert.status === "open" && (
                  <Button variant="outline" size="sm" onClick={() => acknowledge(alert.id)}>
                    Acknowledge
                  </Button>
                )}
                {alert.prediction_id && (
                  <Button variant="outline" size="sm" asChild>
                    <Link href={`/transactions/${alert.prediction_id}`}>Investigate</Link>
                  </Button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
