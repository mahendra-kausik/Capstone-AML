"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { AuthGuard } from "@/components/auth-guard";
import { useAuth } from "@/contexts/auth";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Loading } from "@/components/ui/loading";
import { PageHeader } from "@/components/ui/page-header";
import {
  addCaseNote,
  createCase,
  getCase,
  getCases,
  getHistory,
  getHistoryDetail,
  updateCase,
} from "@/lib/api";
import type { CaseItem, HistoryDetail, HistoryItem } from "@/lib/types";

export default function HistoryPage() {
  return (
    <AuthGuard>
      <Suspense fallback={<Loading />}>
        <HistoryContent />
      </Suspense>
    </AuthGuard>
  );
}

function HistoryContent() {
  const { canWrite } = useAuth();
  const params = useSearchParams();
  const highlight = params.get("highlight");

  const [tab, setTab] = useState<"transactions" | "cases">("transactions");
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<HistoryDetail | null>(null);
  const [activeCase, setActiveCase] = useState<
    CaseItem & { notes: Array<{ id: string; content: string; author_email?: string; created_at: string }> }
  | null>(null);
  const [filters, setFilters] = useState({ risk_min: "", model: "", prediction: "" });
  const [offset, setOffset] = useState(0);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const loadHistory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getHistory({
        limit: 25,
        offset,
        risk_min: filters.risk_min ? Number(filters.risk_min) : undefined,
        model: filters.model || undefined,
        prediction: filters.prediction || undefined,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [offset, filters]);

  const loadCases = useCallback(async () => {
    const res = await getCases({ limit: 50 });
    setCases(res.items);
  }, []);

  useEffect(() => {
    loadHistory();
    loadCases();
  }, [loadHistory, loadCases]);

  useEffect(() => {
    if (highlight) openDetail(highlight);
  }, [highlight]);

  async function openDetail(predictionId: string) {
    try {
      const d = await getHistoryDetail(predictionId);
      setSelected(d);
      if (d.case_id) {
        const c = await getCase(d.case_id);
        setActiveCase(c);
      } else {
        setActiveCase(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detail failed");
    }
  }

  async function openCase(caseId: string) {
    const c = await getCase(caseId);
    setActiveCase(c);
    setTab("cases");
    if (c.prediction_id) openDetail(c.prediction_id);
  }

  async function handleCreateCase() {
    if (!selected || !canWrite) return;
    try {
      const c = await createCase({
        prediction_id: selected.prediction_id,
        priority: selected.risk_score >= 0.75 ? "high" : "medium",
      });
      setActiveCase({ ...c, notes: [] });
      loadCases();
      openDetail(selected.prediction_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Create case failed");
    }
  }

  async function handleAddNote() {
    if (!activeCase || !note.trim()) return;
    await addCaseNote(activeCase.id, note);
    setNote("");
    const c = await getCase(activeCase.id);
    setActiveCase(c);
  }

  return (
    <div>
      <PageHeader
        title="Case Management"
        description="Review scored transactions, open investigations, and track disposition."
      />

      <div className="mt-4 flex gap-2 border-b">
        {(["transactions", "cases"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`border-b-2 px-4 py-2 text-sm font-medium capitalize ${
              tab === t ? "border-brand text-brand" : "border-transparent text-slate-500"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {error && <div className="mt-4"><Alert variant="error">{error}</Alert></div>}

      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          {tab === "transactions" ? (
            <>
              <div className="mb-4 flex flex-wrap gap-2">
                <select
                  className="rounded-lg border px-2 py-1.5 text-sm"
                  value={filters.prediction}
                  onChange={(e) => { setFilters((f) => ({ ...f, prediction: e.target.value })); setOffset(0); }}
                >
                  <option value="">All classes</option>
                  <option value="illicit">Illicit only</option>
                  <option value="licit">Licit only</option>
                </select>
                <select
                  className="rounded-lg border px-2 py-1.5 text-sm"
                  value={filters.model}
                  onChange={(e) => { setFilters((f) => ({ ...f, model: e.target.value })); setOffset(0); }}
                >
                  <option value="">All models</option>
                  <option value="static">Static GCN</option>
                  <option value="evolve">EvolveGCN-H</option>
                </select>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  placeholder="Min risk"
                  className="w-28 rounded-lg border px-2 py-1.5 text-sm"
                  value={filters.risk_min}
                  onChange={(e) => { setFilters((f) => ({ ...f, risk_min: e.target.value })); setOffset(0); }}
                />
              </div>

              {loading ? (
                <Loading />
              ) : (
                <div className="overflow-x-auto rounded-xl border bg-white shadow-sm">
                  <table className="w-full text-left text-sm">
                    <thead className="border-b bg-slate-50 text-xs uppercase text-slate-500">
                      <tr>
                        <th className="p-3">TX</th>
                        <th className="p-3">Risk</th>
                        <th className="p-3">Class</th>
                        <th className="p-3">Model</th>
                        <th className="p-3">Case</th>
                        <th className="p-3">Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((row) => (
                        <tr
                          key={row.prediction_id}
                          onClick={() => openDetail(row.prediction_id)}
                          className={`cursor-pointer border-b hover:bg-slate-50 ${
                            selected?.prediction_id === row.prediction_id ? "bg-brand/5" : ""
                          }`}
                        >
                          <td className="p-3 font-mono">{row.tx_id}</td>
                          <td className="p-3">{(row.risk_score * 100).toFixed(1)}%</td>
                          <td className="p-3"><Badge label={row.prediction} variant={row.prediction} /></td>
                          <td className="p-3">{row.model}</td>
                          <td className="p-3">
                            {row.case_status ? (
                              <Badge label={row.case_status} variant={row.case_status} />
                            ) : (
                              <span className="text-slate-400">—</span>
                            )}
                          </td>
                          <td className="p-3 text-xs text-slate-500">
                            {new Date(row.created_at).toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="flex items-center justify-between border-t p-3 text-sm">
                    <span className="text-slate-500">{total} total</span>
                    <div className="flex gap-2">
                      <button
                        disabled={offset === 0}
                        onClick={() => setOffset(Math.max(0, offset - 25))}
                        className="rounded border px-3 py-1 disabled:opacity-40"
                      >
                        Previous
                      </button>
                      <button
                        disabled={offset + 25 >= total}
                        onClick={() => setOffset(offset + 25)}
                        className="rounded border px-3 py-1 disabled:opacity-40"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="overflow-x-auto rounded-xl border bg-white shadow-sm">
              <table className="w-full text-left text-sm">
                <thead className="border-b bg-slate-50 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="p-3">Case</th>
                    <th className="p-3">TX</th>
                    <th className="p-3">Status</th>
                    <th className="p-3">Priority</th>
                    <th className="p-3">Assignee</th>
                  </tr>
                </thead>
                <tbody>
                  {cases.map((c) => (
                    <tr
                      key={c.id}
                      onClick={() => openCase(c.id)}
                      className="cursor-pointer border-b hover:bg-slate-50"
                    >
                      <td className="p-3 font-medium">{c.title}</td>
                      <td className="p-3 font-mono">{c.tx_id}</td>
                      <td className="p-3"><Badge label={c.status} variant={c.status} /></td>
                      <td className="p-3"><Badge label={c.priority} variant={c.priority} /></td>
                      <td className="p-3 text-slate-600">{c.assignee_email || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="lg:col-span-2">
          {selected ? (
            <div className="sticky top-4 space-y-4 rounded-xl border bg-white p-5 shadow-sm">
              <h2 className="font-semibold">Transaction detail</h2>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-slate-500">TX ID</dt>
                  <dd className="font-mono">{selected.tx_id}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Risk score</dt>
                  <dd className="font-bold">{(selected.risk_score * 100).toFixed(1)}%</dd>
                </div>
                <div className="flex justify-between items-center">
                  <dt className="text-slate-500">Classification</dt>
                  <dd><Badge label={selected.prediction} variant={selected.prediction} /></dd>
                </div>
              </dl>

              <div className="flex flex-wrap gap-2 pt-2">
                {canWrite && !selected.case_id && (
                  <button
                    onClick={handleCreateCase}
                    className="rounded-lg bg-brand px-3 py-1.5 text-sm text-white"
                  >
                    Open case
                  </button>
                )}
                <Link
                  href={`/explainability?prediction_id=${selected.prediction_id}`}
                  className="rounded-lg border px-3 py-1.5 text-sm text-brand hover:bg-brand/5"
                >
                  SHAP explain
                </Link>
                <Link
                  href={`/network?tx_ids=${selected.tx_id}`}
                  className="rounded-lg border px-3 py-1.5 text-sm text-brand hover:bg-brand/5"
                >
                  View network
                </Link>
              </div>

              {activeCase && (
                <div className="border-t pt-4">
                  <h3 className="font-medium">Investigation case</h3>
                  {canWrite && (
                    <select
                      value={activeCase.status}
                      onChange={async (e) => {
                        await updateCase(activeCase.id, { status: e.target.value });
                        const c = await getCase(activeCase.id);
                        setActiveCase(c);
                        loadCases();
                      }}
                      className="mt-2 w-full rounded-lg border px-2 py-1.5 text-sm"
                    >
                      {["open", "investigating", "escalated", "closed", "false_positive"].map((s) => (
                        <option key={s} value={s}>{s.replace("_", " ")}</option>
                      ))}
                    </select>
                  )}
                  <div className="mt-3 max-h-40 space-y-2 overflow-y-auto text-sm">
                    {activeCase.notes?.map((n) => (
                      <div key={n.id} className="rounded bg-slate-50 p-2">
                        <p className="text-xs text-slate-500">{n.author_email} · {new Date(n.created_at).toLocaleString()}</p>
                        <p>{n.content}</p>
                      </div>
                    ))}
                  </div>
                  {canWrite && (
                    <div className="mt-2 flex gap-2">
                      <input
                        className="flex-1 rounded-lg border px-2 py-1.5 text-sm"
                        placeholder="Add investigation note…"
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                      />
                      <button onClick={handleAddNote} className="rounded-lg bg-slate-800 px-3 py-1.5 text-sm text-white">
                        Add
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-xl border bg-white p-8 text-center text-slate-500">
              Select a transaction to view details and manage cases.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
