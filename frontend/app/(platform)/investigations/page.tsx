"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Plus } from "lucide-react";
import { useAuth } from "@/contexts/auth";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/ui/empty-state";
import { Scale } from "lucide-react";
import {
  createCase,
  getCases,
  getHistory,
  updateCase,
} from "@/lib/api";
import { formatPercent, riskSeverity } from "@/lib/utils";
import type { CaseItem, HistoryItem } from "@/lib/types";

export default function InvestigationsPage() {
  const { canWrite } = useAuth();
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  const load = useCallback(async () => {
    const [c, h] = await Promise.all([getCases({ limit: 100 }), getHistory({ limit: 100 })]);
    setCases(c.items);
    setHistory(h.items);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function openCase(predictionId: string) {
    if (!canWrite) return;
    await createCase({ prediction_id: predictionId, priority: "high" });
    load();
  }

  const rows = cases.length
    ? cases
    : history
        .filter((h) => h.risk_score >= 0.5)
        .map((h) => ({
          id: h.case_id || h.prediction_id,
          prediction_id: h.prediction_id,
          title: `TX ${h.tx_id}`,
          status: h.case_status || "open",
          priority: riskSeverity(h.risk_score),
          tx_id: h.tx_id,
          risk_score: h.risk_score,
          prediction: h.prediction,
          model: h.model,
          assignee_email: "—",
          notes_count: 0,
          created_at: h.created_at,
          updated_at: h.created_at,
        })) as CaseItem[];

  const filtered = rows.filter(
    (r) =>
      !filter ||
      r.tx_id?.includes(filter) ||
      r.title.toLowerCase().includes(filter.toLowerCase())
  );

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-96 w-full rounded-xl" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Input
          placeholder="Search cases…"
          className="max-w-xs bg-secondary/50"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <Button size="sm" disabled={!canWrite} asChild>
          <Link href="/upload">
            <Plus className="h-4 w-4" /> Import transactions
          </Link>
        </Button>
      </div>

      {filtered.length === 0 ? (
        <EmptyState
          icon={Scale}
          title="No investigations yet"
          description="Upload transactions or flag high-risk alerts to open cases."
          action={{ label: "Upload batch", onClick: () => (window.location.href = "/upload") }}
        />
      ) : (
        <div className="overflow-hidden rounded-xl border border-border">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border bg-secondary/50 text-[10px] uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Case</th>
                <th className="px-4 py-3">Transaction</th>
                <th className="px-4 py-3">Risk</th>
                <th className="px-4 py-3">Severity</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Analyst</th>
                <th className="px-4 py-3">Created</th>
                <th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => (
                <tr key={row.id} className="border-b border-border/60 hover:bg-secondary/30">
                  <td className="px-4 py-3 font-mono text-xs">{row.id.slice(0, 8)}…</td>
                  <td className="px-4 py-3 font-mono">{row.tx_id}</td>
                  <td className="px-4 py-3 font-semibold">
                    {row.risk_score != null ? formatPercent(row.risk_score) : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={row.priority as "critical" | "high" | "medium" | "low"}>
                      {row.priority}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    {canWrite ? (
                      <select
                        className="rounded-md border border-border bg-background px-2 py-1 text-xs"
                        value={row.status}
                        onChange={async (e) => {
                          if (row.id.length > 20) {
                            await updateCase(row.id, { status: e.target.value });
                            load();
                          }
                        }}
                      >
                        {["open", "investigating", "escalated", "closed", "false_positive"].map((s) => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    ) : (
                      <Badge variant="secondary">{row.status}</Badge>
                    )}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">{row.assignee_email || "—"}</td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    {new Date(row.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-1">
                      <Button variant="ghost" size="sm" asChild>
                        <Link href={`/transactions/${row.prediction_id}`}>Open</Link>
                      </Button>
                      {canWrite && !cases.some((c) => c.prediction_id === row.prediction_id) && (
                        <Button variant="outline" size="sm" onClick={() => openCase(row.prediction_id)}>
                          + Case
                        </Button>
                      )}
                    </div>
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
