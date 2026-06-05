"use client";

import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { DriftPayload } from "@/lib/types";

interface RecentAlertsProps {
  drift: DriftPayload;
}

export function RecentAlerts({ drift }: RecentAlertsProps) {
  const alerts = drift.alerts ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Alerts</CardTitle>
        <CardDescription>Drift and stability warnings</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {alerts.length === 0 ? (
          <p className="text-sm text-muted-foreground py-4 text-center">No active drift alerts</p>
        ) : (
          alerts.slice(0, 5).map((a, i) => (
            <div
              key={i}
              className="flex items-start gap-3 rounded-lg border border-risk-high/20 bg-risk-high/5 px-3 py-2.5"
            >
              <AlertTriangle className="h-4 w-4 text-risk-high shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">
                  {a.model} · {a.window}
                </p>
                <p className="text-xs text-muted-foreground">Kendall τ = {a.tau?.toFixed(3)}</p>
              </div>
              <Badge variant="high">{a.severity || "warning"}</Badge>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
