"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import { FileUp, Upload } from "lucide-react";
import { useAuth } from "@/contexts/auth";
import { AuthGuard } from "@/components/auth-guard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/contexts/toast";
import { uploadCsv } from "@/lib/api";
import { formatPercent } from "@/lib/utils";
import type { UploadResult } from "@/lib/types";

export default function UploadPage() {
  return (
    <AuthGuard requireWrite>
      <UploadContent />
    </AuthGuard>
  );
}

function UploadContent() {
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<"static" | "evolve">("static");
  const [result, setResult] = useState<UploadResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith(".csv")) setFile(f);
  }, []);

  async function run() {
    if (!file) return;
    setLoading(true);
    try {
      const res = await uploadCsv(file, model);
      setResult(res);
      toast(`Analyzed ${res.count} transactions`, "success");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Upload failed", "error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`rounded-xl border-2 border-dashed p-12 text-center transition ${
          dragOver ? "border-primary bg-primary/5" : "border-border bg-card/50"
        }`}
      >
        <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
        <p className="mt-4 font-medium">Drop CSV file here</p>
        <p className="text-sm text-muted-foreground">tx_id, time_step, feat_0…feat_164</p>
        <input type="file" accept=".csv" className="mt-4 text-sm" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        {file && <p className="mt-2 text-sm text-primary">{file.name}</p>}
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <select
          value={model}
          onChange={(e) => setModel(e.target.value as "static" | "evolve")}
          className="h-10 rounded-lg border border-border bg-background px-3 text-sm"
        >
          <option value="static">Static GCN</option>
          <option value="evolve">EvolveGCN-H</option>
        </select>
        <Button onClick={run} disabled={!file || loading}>
          <FileUp className="h-4 w-4" /> {loading ? "Analyzing…" : "Run batch analysis"}
        </Button>
        <a href="/samples/demo_transactions.csv" download className="text-sm text-primary hover:underline ml-auto">
          Download sample CSV
        </a>
      </div>

      {result && (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="glass-card p-4 text-center">
              <p className="text-2xl font-bold">{result.count}</p>
              <p className="text-xs text-muted-foreground">Transactions</p>
            </div>
            <div className="glass-card p-4 text-center">
              <p className="text-2xl font-bold text-risk-critical">{result.summary.high_risk}</p>
              <p className="text-xs text-muted-foreground">High risk</p>
            </div>
            <div className="glass-card p-4 text-center">
              <p className="text-2xl font-bold">{result.summary.mean_risk_score.toFixed(3)}</p>
              <p className="text-xs text-muted-foreground">Mean risk</p>
            </div>
          </div>
          <div className="overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-secondary/50 text-[10px] uppercase text-muted-foreground">
                <tr>
                  <th className="px-4 py-3 text-left">TX</th>
                  <th className="px-4 py-3 text-left">Risk</th>
                  <th className="px-4 py-3 text-left">Class</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r) => (
                  <tr key={r.prediction_id || r.tx_id} className="border-t border-border/60">
                    <td className="px-4 py-3 font-mono">{r.tx_id}</td>
                    <td className="px-4 py-3">{formatPercent(r.risk_score)}</td>
                    <td className="px-4 py-3">
                      <Badge variant={r.prediction === "illicit" ? "critical" : "low"}>{r.prediction}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Button variant="outline" asChild>
            <Link href="/transactions">View in registry →</Link>
          </Button>
        </>
      )}
    </div>
  );
}
