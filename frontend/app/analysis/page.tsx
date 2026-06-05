"use client";

import Link from "next/link";
import { useState } from "react";
import { AuthGuard } from "@/components/auth-guard";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/ui/page-header";
import { predict } from "@/lib/api";
import type { PredictResult } from "@/lib/types";

export default function AnalysisPage() {
  return (
    <AuthGuard requireWrite>
      <AnalysisContent />
    </AuthGuard>
  );
}

function AnalysisContent() {
  const [txId, setTxId] = useState("demo-tx-1");
  const [model, setModel] = useState<"static" | "evolve">("static");
  const [featuresText, setFeaturesText] = useState("");
  const [out, setOut] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function analyze() {
    const features = featuresText.split(/[\s,]+/).filter(Boolean).map(Number);
    if (features.length !== 165) {
      setError("Enter exactly 165 numeric features (space or comma separated).");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await predict({ tx_id: txId, model, features });
      setOut(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Single Transaction Analysis"
        description="Score one transaction with 165 raw Elliptic features."
      />

      <div className="mt-6 max-w-2xl space-y-4 rounded-xl border bg-white p-6 shadow-sm">
        <input
          className="w-full rounded-lg border px-3 py-2 font-mono text-sm"
          placeholder="tx_id"
          value={txId}
          onChange={(e) => setTxId(e.target.value)}
        />
        <select
          value={model}
          onChange={(e) => setModel(e.target.value as "static" | "evolve")}
          className="rounded-lg border px-3 py-2 text-sm"
        >
          <option value="static">Static GCN</option>
          <option value="evolve">EvolveGCN-H</option>
        </select>
        <textarea
          className="h-32 w-full rounded-lg border px-3 py-2 font-mono text-xs"
          placeholder="165 features from demo_transactions.csv…"
          value={featuresText}
          onChange={(e) => setFeaturesText(e.target.value)}
        />
        <button
          onClick={analyze}
          disabled={loading}
          className="rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? "Scoring…" : "Run prediction"}
        </button>
        {error && <Alert variant="error">{error}</Alert>}
      </div>

      {out && (
        <div className="mt-6 max-w-2xl rounded-xl border bg-white p-6 shadow-sm">
          <div className="flex flex-wrap items-center gap-3">
            <Badge label={out.prediction} variant={out.prediction} />
            <span className="text-lg font-semibold">
              Risk {(out.risk_score * 100).toFixed(1)}%
            </span>
            <span className="text-sm text-slate-500">
              Confidence {(out.confidence * 100).toFixed(1)}%
            </span>
          </div>
          {out.prediction_id && (
            <div className="mt-4 flex gap-3 text-sm">
              <Link
                href={`/explainability?prediction_id=${out.prediction_id}`}
                className="text-brand hover:underline"
              >
                Explain with SHAP →
              </Link>
              <Link href={`/history?highlight=${out.prediction_id}`} className="text-brand hover:underline">
                Open in cases →
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
