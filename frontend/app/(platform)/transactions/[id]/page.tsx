"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Suspense } from "react";
import dynamic from "next/dynamic";
import { Network, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getHistoryDetail } from "@/lib/api";
import { formatPercent, riskSeverity } from "@/lib/utils";
import type { HistoryDetail } from "@/lib/types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const NetworkGraph = dynamic(
  () => import("@/components/transactions/mini-network").then((m) => m.MiniNetwork),
  { ssr: false, loading: () => <Skeleton className="h-64 w-full rounded-xl" /> }
);

export default function TransactionDetailPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <DetailContent />
    </Suspense>
  );
}

function DetailContent() {
  const params = useParams();
  const id = params.id as string;
  const [data, setData] = useState<HistoryDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistoryDetail(id)
      .then(setData)
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <Skeleton className="h-96 w-full rounded-xl" />;
  if (!data) return <p className="text-muted-foreground">Transaction not found.</p>;

  const sev = riskSeverity(data.risk_score);
  const shapData = (data.shap?.top_features || data.top_features || []).slice(0, 8).map((t) => ({
    name: t.name,
    value: Math.abs(t.shap_value ?? t.contribution ?? 0),
  }));

  const narrative =
    data.prediction === "illicit"
      ? `This transaction was classified as High Risk (${formatPercent(data.risk_score)}) due to abnormal feature patterns and elevated illicit-class probability from the ${data.model} model.`
      : `This transaction presents low AML risk (${formatPercent(data.risk_score)}) with licit-class probability dominant.`;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-mono text-2xl font-bold">{data.tx_id}</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Prediction ID · {data.prediction_id.slice(0, 8)}…
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <Link href={`/network?tx_ids=${data.tx_id}`}>
              <Network className="h-4 w-4" /> Full network
            </Link>
          </Button>
          <Button size="sm" asChild>
            <Link href={`/explainability?prediction_id=${data.prediction_id}`}>
              <Sparkles className="h-4 w-4" /> SHAP analysis
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Risk Score" value={formatPercent(data.risk_score)} />
        <MetricCard label="Prediction" value={data.prediction} badge={data.prediction === "illicit" ? "critical" : "low"} />
        <MetricCard label="Confidence" value={formatPercent(data.confidence)} />
        <MetricCard label="Severity" value={sev} badge={sev} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Decision Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-muted-foreground">{narrative}</p>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>SHAP Feature Attribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={shapData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis dataKey="name" type="category" width={60} tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
                  <Bar dataKey="value" fill="hsl(262 83% 58%)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Related Network</CardTitle>
          </CardHeader>
          <CardContent>
            <NetworkGraph txId={data.tx_id} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Transaction Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-3 sm:grid-cols-2 text-sm">
            <Meta label="Model" value={data.model} />
            <Meta label="Time step" value={String(data.time_step ?? "—")} />
            <Meta label="Prob. licit" value={formatPercent(data.prob_licit ?? 0)} />
            <Meta label="Prob. illicit" value={formatPercent(data.prob_illicit ?? 0)} />
            <Meta label="Case status" value={data.case_status || "No case"} />
            <Meta label="Scored at" value={new Date(data.created_at).toLocaleString()} />
          </dl>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricCard({
  label,
  value,
  badge,
}: {
  label: string;
  value: string;
  badge?: string;
}) {
  return (
    <div className="glass-card p-4">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <div className="mt-2 flex items-center gap-2">
        <p className="text-xl font-bold capitalize">{value}</p>
        {badge && <Badge variant={badge as "critical" | "low"}>{badge}</Badge>}
      </div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-border/50 pb-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}
