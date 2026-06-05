"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Download, FileUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { getHistory } from "@/lib/api";
import { formatPercent, riskSeverity } from "@/lib/utils";
import type { HistoryItem } from "@/lib/types";

export default function TransactionsPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [classFilter, setClassFilter] = useState("");

  useEffect(() => {
    getHistory({ limit: 200, prediction: classFilter || undefined })
      .then((r) => setItems(r.items))
      .finally(() => setLoading(false));
  }, [classFilter]);

  const filtered = items.filter(
    (i) => !search || i.tx_id.includes(search) || i.model.includes(search)
  );

  function exportCsv() {
    const header = "tx_id,risk_score,prediction,model,created_at\n";
    const rows = filtered
      .map((i) => `${i.tx_id},${i.risk_score},${i.prediction},${i.model},${i.created_at}`)
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "transactions_export.csv";
    a.click();
  }

  if (loading) return <Skeleton className="h-96 w-full rounded-xl" />;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-3">
        <Input
          placeholder="Search tx_id…"
          className="max-w-xs bg-secondary/50"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="h-10 rounded-lg border border-border bg-background px-3 text-sm"
          value={classFilter}
          onChange={(e) => setClassFilter(e.target.value)}
        >
          <option value="">All classes</option>
          <option value="illicit">Illicit</option>
          <option value="licit">Licit</option>
        </select>
        <div className="ml-auto flex gap-2">
          <Button variant="outline" size="sm" onClick={exportCsv}>
            <Download className="h-4 w-4" /> Export CSV
          </Button>
          <Button size="sm" asChild>
            <Link href="/upload"><FileUp className="h-4 w-4" /> Upload</Link>
          </Button>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-border">
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-secondary/50 text-[10px] uppercase tracking-wider text-muted-foreground">
            <tr>
              <th className="px-4 py-3 text-left">Transaction ID</th>
              <th className="px-4 py-3 text-left">Risk Score</th>
              <th className="px-4 py-3 text-left">Classification</th>
              <th className="px-4 py-3 text-left">Model</th>
              <th className="px-4 py-3 text-left">Severity</th>
              <th className="px-4 py-3 text-left">Scored At</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row) => (
              <tr key={row.prediction_id} className="border-b border-border/60 hover:bg-secondary/30">
                <td className="px-4 py-3">
                  <Link href={`/transactions/${row.prediction_id}`} className="font-mono text-primary hover:underline">
                    {row.tx_id}
                  </Link>
                </td>
                <td className="px-4 py-3 font-semibold">{formatPercent(row.risk_score)}</td>
                <td className="px-4 py-3">
                  <Badge variant={row.prediction === "illicit" ? "critical" : "low"}>
                    {row.prediction}
                  </Badge>
                </td>
                <td className="px-4 py-3 text-muted-foreground">{row.model}</td>
                <td className="px-4 py-3">
                  <Badge variant={riskSeverity(row.risk_score)}>{riskSeverity(row.risk_score)}</Badge>
                </td>
                <td className="px-4 py-3 text-xs text-muted-foreground">
                  {new Date(row.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
