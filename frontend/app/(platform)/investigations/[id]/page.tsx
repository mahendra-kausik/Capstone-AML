"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Send } from "lucide-react";
import { useAuth } from "@/contexts/auth";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/contexts/toast";
import { addCaseNote, getCase, getUsers, updateCase } from "@/lib/api";
import { formatPercent } from "@/lib/utils";
import type { CaseItem, User } from "@/lib/types";

export default function CaseDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { canWrite } = useAuth();
  const { toast } = useToast();
  const [caseData, setCaseData] = useState<CaseItem | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [c, u] = await Promise.all([
        getCase(id),
        canWrite ? getUsers().catch(() => []) : Promise.resolve([]),
      ]);
      setCaseData(c);
      setUsers(u);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load case");
    } finally {
      setLoading(false);
    }
  }, [id, canWrite]);

  useEffect(() => {
    load();
  }, [load]);

  async function submitNote() {
    if (!note.trim()) return;
    try {
      await addCaseNote(id, note.trim());
      setNote("");
      toast("Note added", "success");
      load();
    } catch (e) {
      toast(e instanceof Error ? e.message : "Failed to add note", "error");
    }
  }

  if (loading) return <Skeleton className="h-96 w-full rounded-xl" />;
  if (error || !caseData) {
    return (
      <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-8 text-center">
        <p className="text-destructive">{error || "Case not found"}</p>
        <Button variant="outline" className="mt-4" asChild>
          <Link href="/investigations">Back to investigations</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" asChild>
        <Link href="/investigations">
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>
      </Button>

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">{caseData.title}</h2>
          <p className="text-sm text-muted-foreground font-mono">Case {caseData.id.slice(0, 8)}…</p>
        </div>
        <div className="flex gap-2">
          <Badge variant={caseData.priority as "critical" | "high" | "medium" | "low"}>
            {caseData.priority}
          </Badge>
          <Badge variant="secondary">{caseData.status}</Badge>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader><CardTitle>Case Details</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Transaction</span>
              <Link href={`/transactions/${caseData.prediction_id}`} className="font-mono text-primary hover:underline">
                {caseData.tx_id}
              </Link>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Risk Score</span>
              <span className="font-semibold">{caseData.risk_score != null ? formatPercent(caseData.risk_score) : "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Prediction</span>
              <span>{caseData.prediction}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Model</span>
              <span>{caseData.model}</span>
            </div>
            {canWrite && (
              <>
                <div className="pt-2">
                  <label className="text-xs text-muted-foreground">Status</label>
                  <select
                    className="mt-1 w-full h-9 rounded-lg border border-border bg-background px-2 text-sm"
                    value={caseData.status}
                    onChange={async (e) => {
                      await updateCase(id, { status: e.target.value });
                      load();
                    }}
                  >
                    {["open", "investigating", "escalated", "closed", "false_positive"].map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                {users.length > 0 && (
                  <div>
                    <label className="text-xs text-muted-foreground">Assign Analyst</label>
                    <select
                      className="mt-1 w-full h-9 rounded-lg border border-border bg-background px-2 text-sm"
                      value={caseData.assignee_id || ""}
                      onChange={async (e) => {
                        if (e.target.value) {
                          await updateCase(id, { assignee_id: e.target.value });
                          load();
                        }
                      }}
                    >
                      <option value="">Unassigned</option>
                      {users.filter((u) => u.role !== "viewer").map((u) => (
                        <option key={u.id} value={u.id}>{u.email}</option>
                      ))}
                    </select>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Activity Timeline</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {(caseData.notes || []).length === 0 ? (
              <p className="text-sm text-muted-foreground">No notes yet.</p>
            ) : (
              caseData.notes!.map((n) => (
                <div key={n.id} className="border-l-2 border-primary/30 pl-4 py-1">
                  <p className="text-sm">{n.content}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {n.author_email} · {new Date(n.created_at).toLocaleString()}
                  </p>
                </div>
              ))
            )}
            {canWrite && (
              <div className="flex gap-2 pt-2">
                <Input
                  placeholder="Add investigation note…"
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && submitNote()}
                />
                <Button size="sm" onClick={submitNote}>
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
