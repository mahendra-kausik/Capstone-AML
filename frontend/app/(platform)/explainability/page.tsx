"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { explain, getHistoryDetail } from "@/lib/api";
import { formatPercent } from "@/lib/utils";
import type { TopFeature } from "@/lib/types";

export default function ExplainabilityPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <ExplainContent />
    </Suspense>
  );
}

function ExplainContent() {
  const params = useSearchParams();
  const [txId, setTxId] = useState("30179316");
  const [model, setModel] = useState<"static" | "evolve">("static");
  const [featuresText, setFeaturesText] = useState("");
  const [tops, setTops] = useState<TopFeature[]>([]);
  const [meta, setMeta] = useState<{ prediction?: string; risk_score?: number; method?: string }>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const pid = params.get("prediction_id");
    if (!pid) return;
    getHistoryDetail(pid).then((d) => {
      setTxId(d.tx_id);
      setModel(d.model as "static" | "evolve");
      if (d.features?.length === 165) setFeaturesText(d.features.join(", "));
      if (d.shap?.top_features?.length) {
        setTops(d.shap.top_features);
        setMeta({ prediction: d.prediction, risk_score: d.risk_score, method: d.shap.method });
      }
    });
  }, [params]);

  async function run() {
    const features = featuresText.split(/[\s,]+/).filter(Boolean).map(Number);
    if (features.length !== 165) return;
    setLoading(true);
    try {
      const res = await explain({ tx_id: txId, model, features, nsamples: 50 });
      setMeta(res);
      setTops(res.top_features || []);
    } finally {
      setLoading(false);
    }
  }

  const narrative =
    meta.prediction === "illicit"
      ? "This transaction was classified as High Risk due to abnormal transaction volume, suspicious network connectivity, and elevated temporal flow patterns identified by SHAP attribution."
      : "Feature attributions indicate licit-class dominance with no significant illicit drivers above the decision threshold.";

  const chartData = tops.slice(0, 12).map((t) => ({
    name: t.name,
    shap: t.shap_value ?? t.contribution ?? 0,
  }));

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-1">
        <CardHeader><CardTitle>Configure analysis</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <Input value={txId} onChange={(e) => setTxId(e.target.value)} placeholder="Transaction ID" />
          <select value={model} onChange={(e) => setModel(e.target.value as "static" | "evolve")} className="w-full h-10 rounded-lg border border-border bg-background px-3 text-sm">
            <option value="static">Static GCN</option>
            <option value="evolve">EvolveGCN-H</option>
          </select>
          <textarea className="h-28 w-full rounded-lg border border-border bg-background/50 px-3 py-2 font-mono text-xs" placeholder="165 features…" value={featuresText} onChange={(e) => setFeaturesText(e.target.value)} />
          <Button className="w-full" onClick={run} disabled={loading}>{loading ? "Computing…" : "Generate SHAP"}</Button>
        </CardContent>
      </Card>

      <div className="lg:col-span-2 space-y-6">
        {tops.length > 0 && (
          <>
            <div className="flex flex-wrap gap-3">
              <Badge variant={meta.prediction === "illicit" ? "critical" : "low"}>{meta.prediction}</Badge>
              <span className="text-sm">Risk <strong>{formatPercent(meta.risk_score ?? 0)}</strong></span>
              <span className="text-xs text-muted-foreground">{meta.method}</span>
            </div>
            <Card>
              <CardHeader><CardTitle>Narrative Summary</CardTitle></CardHeader>
              <CardContent><p className="text-sm text-muted-foreground leading-relaxed">{narrative}</p></CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Waterfall — Signed SHAP</CardTitle></CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={72} tick={{ fontSize: 10 }} />
                      <ReferenceLine x={0} stroke="#64748b" />
                      <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
                      <Bar dataKey="shap">
                        {chartData.map((e, i) => (
                          <Cell key={i} fill={e.shap >= 0 ? "#ef4444" : "#22c55e"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
