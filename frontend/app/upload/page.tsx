"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import { FileUp, Upload } from "lucide-react";
import { AuthGuard } from "@/components/auth-guard";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/ui/page-header";
import { uploadCsv } from "@/lib/api";
import type { PredictResult, UploadResult } from "@/lib/types";

export default function UploadPage() {
  return (
    <AuthGuard requireWrite>
      <UploadContent />
    </AuthGuard>
  );
}

function UploadContent() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<"static" | "evolve">("static");
  const [result, setResult] = useState<UploadResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
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
    setError("");
    setResult(null);
    try {
      const res = await uploadCsv(file, model);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Batch Upload"
        description="Upload a CSV with tx_id, time_step, and 165 features (feat_0…feat_164)."
        action={
          <a
            href="/samples/demo_transactions.csv"
            download
            className="text-sm text-brand hover:underline"
          >
            Download sample CSV
          </a>
        }
      />

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={`rounded-xl border-2 border-dashed bg-white p-8 text-center transition ${
              dragOver ? "border-brand bg-brand/5" : "border-slate-200"
            }`}
          >
            <Upload className="mx-auto h-10 w-10 text-slate-400" />
            <p className="mt-3 font-medium">Drag & drop CSV here</p>
            <p className="mt-1 text-sm text-slate-500">or browse from your machine</p>
            <input
              type="file"
              accept=".csv"
              className="mt-4 text-sm"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file && (
              <p className="mt-3 text-sm text-slate-700">
                <FileUp className="mr-1 inline h-4 w-4" />
                {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-4 rounded-xl border bg-white p-4">
            <label className="text-sm font-medium">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as "static" | "evolve")}
              className="rounded-lg border px-3 py-2 text-sm"
            >
              <option value="static">Static GCN</option>
              <option value="evolve">EvolveGCN-H</option>
            </select>
            <button
              onClick={run}
              disabled={!file || loading}
              className="ml-auto rounded-lg bg-brand px-5 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading ? "Analyzing…" : "Run batch analysis"}
            </button>
          </div>

          {error && <Alert variant="error">{error}</Alert>}
        </div>

        {result && (
          <div className="rounded-xl border bg-white p-5 shadow-sm">
            <h2 className="font-semibold">Job summary</h2>
            <dl className="mt-4 space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-slate-500">File</dt>
                <dd className="font-medium">{result.filename}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Transactions</dt>
                <dd className="font-medium">{result.count}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">High risk</dt>
                <dd className="font-bold text-red-600">{result.summary.high_risk}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Mean risk</dt>
                <dd>{result.summary.mean_risk_score.toFixed(4)}</dd>
              </div>
            </dl>
            <Link
              href="/history"
              className="mt-4 block text-center text-sm text-brand hover:underline"
            >
              View in case history →
            </Link>
          </div>
        )}
      </div>

      {result && result.results.length > 0 && (
        <div className="mt-8 overflow-x-auto rounded-xl border bg-white shadow-sm">
          <table className="w-full text-left text-sm">
            <thead className="border-b bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="p-3">Transaction</th>
                <th className="p-3">Risk</th>
                <th className="p-3">Classification</th>
                <th className="p-3">Confidence</th>
                <th className="p-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((row: PredictResult) => (
                <tr key={row.prediction_id || row.tx_id} className="border-b hover:bg-slate-50">
                  <td className="p-3 font-mono">{row.tx_id}</td>
                  <td className="p-3">{(row.risk_score * 100).toFixed(1)}%</td>
                  <td className="p-3">
                    <Badge label={row.prediction} variant={row.prediction} />
                  </td>
                  <td className="p-3">{(row.confidence * 100).toFixed(1)}%</td>
                  <td className="p-3">
                    <Link
                      href={`/history?highlight=${row.prediction_id}`}
                      className="text-brand hover:underline"
                    >
                      Open case
                    </Link>
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
