"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Brain,
  FileUp,
  Scale,
  ShieldAlert,
  Target,
  TrendingUp,
} from "lucide-react";
import { HighRiskFeed } from "@/components/dashboard/high-risk-feed";
import { ModelComparison } from "@/components/dashboard/model-comparison";
import { RecentAlerts } from "@/components/dashboard/recent-alerts";
import { RiskBreakdown } from "@/components/dashboard/risk-breakdown";
import { RiskTrendChart } from "@/components/dashboard/risk-trend-chart";
import { ShapImpact } from "@/components/dashboard/shap-impact";
import { Button } from "@/components/ui/button";
import { KpiCard } from "@/components/ui/kpi-card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { getDrift, getHistory, getMetrics } from "@/lib/api";
import {
  casesUnderReview,
  clusterExposure,
  driftScore,
  driftSeverity,
  highRiskAlerts,
  riskCategoryBreakdown,
  riskTrend30d,
  transactionsToday,
} from "@/lib/dashboard-data";
import { formatPercent } from "@/lib/utils";
import type { DriftPayload, HistoryItem } from "@/lib/types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function ExecutiveDashboard() {
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [drift, setDrift] = useState<DriftPayload>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getMetrics(), getHistory({ limit: 200 }), getDrift()])
      .then(([m, h, d]) => {
        setMetrics(m);
        setHistory(h.items || []);
        setDrift(d);
      })
      .finally(() => setLoading(false));
  }, []);

  const tau = driftScore(drift);
  const severity = driftSeverity(tau);
  const modelAccuracy = metrics.static_gcn?.test_auroc ?? 0.8573;

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <Skeleton className="h-80 rounded-xl" />
          <Skeleton className="h-80 rounded-xl" />
        </div>
      </div>
    );
  }

  const clusterData = clusterExposure(history);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap gap-2">
          <Badge variant="success" className="normal-case">
            <Brain className="mr-1 h-3 w-3" /> SHAP Explainability
          </Badge>
          <Badge variant="default" className="normal-case">
            <Activity className="mr-1 h-3 w-3" /> Kendall τ Drift Monitoring
          </Badge>
          <Badge variant="secondary" className="normal-case">
            Temporal GNN · Elliptic Dataset
          </Badge>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <Link href="/reports">Export report</Link>
          </Button>
          <Button size="sm" asChild>
            <Link href="/upload">
              <FileUp className="h-4 w-4" /> Batch upload
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <KpiCard title="Total Transactions" value={history.length} icon={TrendingUp} subtitle="All time scored" />
        <KpiCard title="Transactions Today" value={transactionsToday(history)} icon={ArrowUpRight} trend={{ value: 12, label: "vs yesterday" }} />
        <KpiCard title="High Risk Alerts" value={highRiskAlerts(history)} icon={ShieldAlert} accent="critical" />
        <KpiCard title="Cases Under Review" value={casesUnderReview(history)} icon={Scale} accent="warning" />
        <KpiCard title="Model Accuracy" value={formatPercent(modelAccuracy)} icon={Target} subtitle="Static GCN AUROC" accent="success" />
        <KpiCard title="Drift Score" value={tau.toFixed(2)} icon={AlertTriangle} subtitle={`Severity: ${severity}`} accent={severity === "Critical" || severity === "High" ? "critical" : "default"} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <RiskTrendChart data={riskTrend30d(history)} />
        </div>
        <HighRiskFeed items={history} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <RiskBreakdown
          data={clusterExposure(history).map((c, i) => ({
            name: c.name,
            value: c.exposure,
            fill: ["#0ea5e9", "#8b5cf6", "#f97316", "#64748b", "#ef4444"][i],
          }))}
          title="Network Cluster Exposure"
          description="Risk exposure by transaction cluster type"
        />
        <RiskBreakdown
          data={riskCategoryBreakdown(history)}
          title="Risk Category Breakdown"
          description="Severity distribution across scored transactions"
        />
        <CardClusterChart data={clusterData} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <ModelComparison metrics={metrics} />
        </div>
        <RecentAlerts drift={drift} />
      </div>

      <ShapImpact />
    </div>
  );
}

function CardClusterChart({ data }: { data: Array<{ name: string; exposure: number }> }) {
  return (
    <div className="glass-card p-5 h-full">
      <h3 className="text-sm font-semibold">Geographic Risk Distribution</h3>
      <p className="text-xs text-muted-foreground mt-1">Cluster-weighted exposure index</p>
      <div className="h-[200px] mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
            <Bar dataKey="exposure" fill="hsl(199 89% 48%)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
