"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AuthGuard } from "@/components/auth-guard";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Loading } from "@/components/ui/loading";
import { PageHeader } from "@/components/ui/page-header";
import { explain, getHistoryDetail } from "@/lib/api";
import type { TopFeature } from "@/lib/types";

export default function ExplainabilityPage() {
  return (
    <AuthGuard>
      <Suspense fallback={<Loading />}>
        <ExplainContent />
      </Suspense>
    </AuthGuard>
  );
}

function ExplainContent() {
  const params = useSearchParams();
  const predictionId = params.get("prediction_id");

  const [txId, setTxId] = useState("30179316");
  const [model, setModel] = useState<"static" | "evolve">("static");
  const [nsamples, setNsamples] = useState(50);
  const [featuresText, setFeaturesText] = useState("");
  const [tops, setTops] = useState<TopFeature[]>([]);
  const [meta, setMeta] = useState<{
    prediction?: string;
    risk_score?: number;
    method?: string;
  }>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!predictionId) return;
    getHistoryDetail(predictionId)
      .then((d) => {
        setTxId(d.tx_id);
        setModel(d.model as "static" | "evolve");
        if (d.features?.length === 165) {
          setFeaturesText(d.features.join(", "));
        }
        if (d.shap?.top_features?.length) {
          setTops(d.shap.top_features);
          setMeta({
            prediction: d.prediction,
            risk_score: d.risk_score,
            method: d.shap.method,
          });
        }
      })
      .catch(() => {});
  }, [predictionId]);

  async function run() {
    const features = featuresText.split(/[\s,]+/).filter(Boolean).map(Number);
    if (features.length !== 165) {
      setError("Enter exactly 165 numeric features (comma or space separated).");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await explain({ tx_id: txId, model, features, nsamples });
      setMeta(res);
      setTops(res.top_features || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "SHAP failed");
    } finally {
      setLoading(false);
    }
  }

  const chartData = tops.slice(0, 12).map((t) => ({
    name: t.name,
    shap: t.shap_value ?? t.contribution ?? 0,
    value: t.feature_value ?? 0,
  }));

  return (
    <div>
      <PageHeader
        title="SHAP Explainability"
        description="Kernel SHAP attributions for illicit-class probability (research models, no retraining)."
      />

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1 space-y-4 rounded-xl border bg-white p-5 shadow-sm">
          <div>
            <label className="text-sm font-medium">Transaction ID</label>
            <input
              className="mt-1 w-full rounded-lg border px-3 py-2 font-mono text-sm"
              value={txId}
              onChange={(e) => setTxId(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-sm font-medium">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value as "static" | "evolve")}
                className="mt-1 w-full rounded-lg border px-2 py-2 text-sm"
              >
                <option value="static">Static GCN</option>
                <option value="evolve">EvolveGCN-H</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium">SHAP samples</label>
              <input
                type="number"
                min={10}
                max={200}
                value={nsamples}
                onChange={(e) => setNsamples(Number(e.target.value))}
                className="mt-1 w-full rounded-lg border px-2 py-2 text-sm"
              />
            </div>
          </div>
          <div>
            <label className="text-sm font-medium">165 features</label>
            <textarea
              className="mt-1 h-28 w-full rounded-lg border px-3 py-2 font-mono text-xs"
              placeholder="Paste feat_0…feat_164 values from CSV or history"
              value={featuresText}
              onChange={(e) => setFeaturesText(e.target.value)}
            />
          </div>
          <button
            onClick={run}
            disabled={loading}
            className="w-full rounded-lg bg-brand py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading ? "Computing SHAP…" : "Generate explanation"}
          </button>
          {error && <Alert variant="error">{error}</Alert>}
        </div>

        <div className="lg:col-span-2 space-y-4">
          {tops.length > 0 && (
            <>
              <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-white p-4">
                <Badge label={String(meta.prediction)} variant={String(meta.prediction)} />
                <span className="text-sm">
                  Risk: <strong>{((meta.risk_score as number) * 100).toFixed(1)}%</strong>
                </span>
                <span className="text-sm text-slate-500">
                  Method: {String(meta.method || "kernel_shap")}
                </span>
              </div>

              <div className="rounded-xl border bg-white p-4 shadow-sm">
                <h2 className="font-semibold">Signed SHAP contributions</h2>
                <p className="text-xs text-slate-500">Positive → pushes toward illicit</p>
                <div className="mt-4 h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={72} tick={{ fontSize: 11 }} />
                      <Tooltip
                        formatter={(v: number, name: string) => [
                          v.toFixed(4),
                          name === "shap" ? "SHAP" : "Feature value",
                        ]}
                      />
                      <ReferenceLine x={0} stroke="#64748b" />
                      <Bar dataKey="shap" name="shap">
                        {chartData.map((entry, i) => (
                          <Cell
                            key={i}
                            fill={entry.shap >= 0 ? "#dc2626" : "#059669"}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="overflow-x-auto rounded-xl border bg-white shadow-sm">
                <table className="w-full text-left text-sm">
                  <thead className="border-b bg-slate-50 text-xs uppercase text-slate-500">
                    <tr>
                      <th className="p-3">Feature</th>
                      <th className="p-3">SHAP</th>
                      <th className="p-3">Value (scaled)</th>
                      <th className="p-3">|Contribution|</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tops.map((t) => (
                      <tr key={t.name} className="border-b">
                        <td className="p-3 font-mono">{t.name}</td>
                        <td className="p-3">{(t.shap_value ?? 0).toFixed(4)}</td>
                        <td className="p-3">{(t.feature_value ?? 0).toFixed(4)}</td>
                        <td className="p-3">{(t.contribution ?? 0).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
