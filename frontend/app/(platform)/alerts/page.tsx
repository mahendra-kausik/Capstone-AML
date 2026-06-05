"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getDrift, getHistory } from "@/lib/api";
import { formatPercent, riskSeverity } from "@/lib/utils";
import type { HistoryItem } from "@/lib/types";

const LEVELS = ["critical", "high", "medium", "low"] as const;

export default function AlertsPage() {
  const [txAlerts, setTxAlerts] = useState<HistoryItem[]>([]);
  const [driftAlerts, setDriftAlerts] = useState<any[]>([]);
  const [level, setLevel] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getHistory({ limit: 200 }), getDrift()])
      .then(([h, d]) => {
        setTxAlerts(h.items.filter((i) => i.risk_score >= 0.4 || i.prediction === "illicit"));
        setDriftAlerts(d.alerts || []);
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = level
    ? txAlerts.filter((i) => riskSeverity(i.risk_score) === level)
    : txAlerts;

  if (loading) return <Skeleton className="h-96 w-full rounded-xl" />;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        <Button variant={!level ? "default" : "outline"} size="sm" onClick={() => setLevel("")}>
          All
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
        <h2 className="text-sm font-semibold">Transaction Risk Alerts</h2>
        {filtered.length === 0 ? (
          <p className="py-12 text-center text-muted-foreground">No alerts match this filter</p>
        ) : (
          filtered.map((item) => {
            const sev = riskSeverity(item.risk_score);
            return (
              <div
                key={item.prediction_id}
                className="glass-card flex flex-wrap items-center justify-between gap-4 p-4 transition hover:border-primary/30"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`h-2 w-2 rounded-full animate-pulse ${
                      sev === "critical" ? "bg-risk-critical" : sev === "high" ? "bg-risk-high" : "bg-risk-medium"
                    }`}
                  />
                  <div>
                    <p className="font-mono text-sm">{item.tx_id}</p>
                    <p className="text-xs text-muted-foreground">{new Date(item.created_at).toLocaleString()}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-lg font-bold">{formatPercent(item.risk_score)}</span>
                  <Badge variant={sev}>{sev}</Badge>
                  <Button variant="outline" size="sm" asChild>
                    <Link href={`/transactions/${item.prediction_id}`}>Investigate</Link>
                  </Button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
