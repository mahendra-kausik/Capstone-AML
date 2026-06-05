"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AuthGuard } from "@/components/auth-guard";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Loading } from "@/components/ui/loading";
import { PageHeader } from "@/components/ui/page-header";
import { getDrift, getDriftEvents } from "@/lib/api";
import type { DriftPayload } from "@/lib/types";

export default function DriftPage() {
  return (
    <AuthGuard>
      <DriftContent />
    </AuthGuard>
  );
}

function DriftContent() {
  const [data, setData] = useState<DriftPayload>({});
  const [events, setEvents] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getDrift(), getDriftEvents()])
      .then(([d, e]) => {
        setData(d);
        setEvents(e);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;

  const tauStatic =
    data.kendall_tau?.static?.map((r) => ({
      window: r.comparison,
      static: r.tau,
      evolve: null as number | null,
    })) ?? [];
  const tauEvolve =
    data.kendall_tau?.evolve?.map((r) => ({
      window: r.comparison,
      static: null as number | null,
      evolve: r.tau,
    })) ?? [];
  const windows = [...new Set([...tauStatic, ...tauEvolve].map((x) => x.window))];
  const combined = windows.map((w) => ({
    window: w,
    static: tauStatic.find((x) => x.window === w)?.static ?? undefined,
    evolve: tauEvolve.find((x) => x.window === w)?.evolve ?? undefined,
  }));

  const t43 = data.t43_drift?.static;
  const f1Timeline = [
    ...(t43?.pre_snapshots?.map((f1, i) => ({
      step: 37 + i,
      f1,
      phase: "Pre-T43",
    })) ?? []),
    ...(t43?.post_snapshots?.map((f1, i) => ({
      step: (t43?.shutdown_step ?? 43) + i,
      f1,
      phase: "Post-T43",
    })) ?? []),
  ];

  const alerts = data.alerts ?? [];

  return (
    <div>
      <PageHeader
        title="Drift Monitoring"
        description="SHAP stability (Kendall τ) and T43-style performance collapse from research artifacts."
      />

      {error && <div className="mb-4"><Alert variant="error">{error}</Alert></div>}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card title="Drift alerts" value={alerts.length} subtitle="τ &lt; 0.70 threshold" />
        <Card
          title="W3→W4 τ (Static)"
          value={
            data.kendall_tau?.static?.find((x) => x.comparison.includes("W3"))?.tau?.toFixed(3) ?? "—"
          }
          subtitle="research SHAP windows"
        />
        <Card
          title="F1 drop (T43)"
          value={t43?.f1_drop != null ? `${(t43.f1_drop * 100).toFixed(1)}%` : "—"}
          subtitle="static model"
        />
        <Card
          title="Post-T43 mean F1"
          value={t43?.post_t43_mean_f1?.toFixed(3) ?? "—"}
          subtitle="vs pre-T43 baseline"
        />
      </div>

      {alerts.length > 0 && (
        <div className="mt-8 rounded-xl border bg-white p-4 shadow-sm">
          <h2 className="font-semibold">Active drift alerts</h2>
          <div className="mt-3 space-y-2">
            {alerts.map((a, i) => (
              <div
                key={i}
                className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm"
              >
                <span>
                  <strong>{a.model}</strong> · {a.window}
                </span>
                <span>τ = {a.tau?.toFixed(3)}</span>
                <Badge label={a.severity || "warning"} variant="high" />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-8 h-80 rounded-xl border bg-white p-4 shadow-sm">
        <h2 className="font-semibold">Kendall τ — SHAP rank stability</h2>
        <ResponsiveContainer width="100%" height="90%">
          <BarChart data={combined}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="window" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={0.7} stroke="#64748b" strokeDasharray="4 4" label="τ=0.70" />
            <Bar dataKey="static" name="Static GCN" fill="#1a56a0" />
            <Bar dataKey="evolve" name="EvolveGCN-H" fill="#0d9488" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {f1Timeline.length > 0 && (
        <div className="mt-8 h-72 rounded-xl border bg-white p-4 shadow-sm">
          <h2 className="font-semibold">T43 F1 trajectory (Static GCN)</h2>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={f1Timeline}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <ReferenceLine x={t43?.shutdown_step ?? 43} stroke="#dc2626" strokeDasharray="4 4" label="T43" />
              <Line type="monotone" dataKey="f1" stroke="#dc2626" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="mt-8 rounded-xl border bg-white shadow-sm">
        <h2 className="border-b p-4 font-semibold">Drift event log</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="p-3">Time</th>
                <th className="p-3">Model</th>
                <th className="p-3">Window</th>
                <th className="p-3">τ</th>
                <th className="p-3">Alert</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id} className="border-b">
                  <td className="p-3 text-xs">{new Date(e.detected_at).toLocaleString()}</td>
                  <td className="p-3">{e.model_name}</td>
                  <td className="p-3">{e.window}</td>
                  <td className="p-3">{e.metric_value?.toFixed(3)}</td>
                  <td className="p-3">
                    {e.is_alert ? (
                      <Badge label="alert" variant="high" />
                    ) : (
                      <span className="text-slate-400">ok</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
