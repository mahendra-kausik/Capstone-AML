"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AuthGuard } from "@/components/auth-guard";
import { Alert } from "@/components/ui/alert";
import { Card } from "@/components/ui/card";
import { Loading } from "@/components/ui/loading";
import { PageHeader } from "@/components/ui/page-header";
import { getBatches, getDrift, getHistory, getMetrics } from "@/lib/api";

export default function DashboardPage() {
  return (
    <AuthGuard>
      <DashboardContent />
    </AuthGuard>
  );
}

function DashboardContent() {
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [history, setHistory] = useState<any[]>([]);
  const [drift, setDrift] = useState<Record<string, any>>({});
  const [batches, setBatches] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getMetrics(),
      getHistory({ limit: 200 }),
      getDrift(),
      getBatches({ limit: 5 }),
    ])
      .then(([m, h, d, b]) => {
        setMetrics(m);
        setHistory(h.items || []);
        setDrift(d);
        setBatches(b.items || []);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load dashboard"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;

  const highRisk = history.filter((x) => x.prediction === "illicit").length;
  const openCases = history.filter((x) => x.case_status === "open" || x.case_status === "investigating").length;
  const alerts = (drift.alerts as any[])?.length ?? 0;
  const riskBuckets = [
    { band: "0–25%", count: history.filter((x) => x.risk_score < 0.25).length },
    { band: "25–50%", count: history.filter((x) => x.risk_score >= 0.25 && x.risk_score < 0.5).length },
    { band: "50–75%", count: history.filter((x) => x.risk_score >= 0.5 && x.risk_score < 0.75).length },
    { band: "75–100%", count: history.filter((x) => x.risk_score >= 0.75).length },
  ];

  return (
    <div>
      <PageHeader
        title="Risk Dashboard"
        description="Operational overview — scored transactions, open cases, and model drift signals."
        action={
          <Link href="/upload" className="rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white">
            New batch upload
          </Link>
        }
      />

      {error && <div className="mt-4"><Alert variant="error">{error}</Alert></div>}

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card title="Scored transactions" value={history.length} subtitle="in platform DB" />
        <Card title="High risk" value={highRisk} subtitle="illicit classification" />
        <Card title="Open cases" value={openCases} subtitle="investigations active" />
        <Card title="Drift alerts" value={alerts} subtitle="τ &lt; 0.70" />
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        <div className="rounded-xl border bg-white p-4 shadow-sm">
          <h2 className="font-semibold">Risk distribution</h2>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskBuckets}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="band" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#1a56a0" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border bg-white p-4 shadow-sm">
          <h2 className="font-semibold">Research benchmarks</h2>
          <dl className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between border-b pb-2">
              <dt>Static GCN AUROC</dt>
              <dd className="font-semibold">{metrics.static_gcn?.test_auroc?.toFixed(3) ?? "—"}</dd>
            </div>
            <div className="flex justify-between border-b pb-2">
              <dt>Static GCN F1</dt>
              <dd>{metrics.static_gcn?.test_f1?.toFixed(3) ?? "—"}</dd>
            </div>
            <div className="flex justify-between border-b pb-2">
              <dt>EvolveGCN-H AUROC</dt>
              <dd className="font-semibold">{metrics.evolvegcn_h?.test_auroc?.toFixed(3) ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt>EvolveGCN-H F1</dt>
              <dd>{metrics.evolvegcn_h?.test_f1?.toFixed(3) ?? "—"}</dd>
            </div>
          </dl>
        </div>
      </div>

      {batches.length > 0 && (
        <div className="mt-8 rounded-xl border bg-white p-4 shadow-sm">
          <h2 className="font-semibold">Recent batch jobs</h2>
          <table className="mt-3 w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr>
                <th className="pb-2">File</th>
                <th className="pb-2">Rows</th>
                <th className="pb-2">High risk</th>
                <th className="pb-2">Time</th>
              </tr>
            </thead>
            <tbody>
              {batches.map((b) => (
                <tr key={String(b.job_id)} className="border-t">
                  <td className="py-2">{String(b.filename || "—")}</td>
                  <td className="py-2">{String(b.total_rows)}</td>
                  <td className="py-2 font-medium text-red-600">{String(b.high_risk_count)}</td>
                  <td className="py-2 text-xs text-slate-500">
                    {b.created_at ? new Date(String(b.created_at)).toLocaleString() : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
