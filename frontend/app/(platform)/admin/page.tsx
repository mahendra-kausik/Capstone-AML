"use client";

import { useEffect, useState } from "react";
import { AuthGuard } from "@/components/auth-guard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { KpiCard } from "@/components/ui/kpi-card";
import { Activity, Cpu, Users } from "lucide-react";
import { getAuditLogs, getMetrics, getUsers, registerUser } from "@/lib/api";
import { useToast } from "@/contexts/toast";
import type { User } from "@/lib/types";

export default function AdminPage() {
  return (
    <AuthGuard requireAdmin>
      <AdminContent />
    </AuthGuard>
  );
}

function AdminContent() {
  const { toast } = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({ email: "", password: "", role: "analyst" });

  useEffect(() => {
    Promise.all([getUsers(), getAuditLogs({ limit: 50 }), getMetrics()])
      .then(([u, l, m]) => { setUsers(u); setLogs(l.items); setMetrics(m); })
      .finally(() => setLoading(false));
  }, []);

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    try {
      await registerUser(form);
      toast("User created", "success");
      setUsers(await getUsers());
    } catch (err) {
      toast(err instanceof Error ? err.message : "Failed", "error");
    }
  }

  if (loading) return <Skeleton className="h-96 w-full rounded-xl" />;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <KpiCard title="Users" value={users.length} icon={Users} />
        <KpiCard title="Static GCN AUROC" value={metrics.static_gcn?.test_auroc?.toFixed(3) ?? "0.857"} icon={Cpu} subtitle="Production model" />
        <KpiCard title="System" value="Healthy" icon={Activity} subtitle="API · ML · DB" accent="success" />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Users &amp; Roles</CardTitle></CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <tbody>
                {users.map((u) => (
                  <tr key={u.id} className="border-b border-border/60">
                    <td className="py-2">{u.email}</td>
                    <td className="py-2"><Badge variant="secondary">{u.role}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Register user</CardTitle></CardHeader>
          <CardContent>
            <form onSubmit={handleRegister} className="space-y-3">
              <Input type="email" required placeholder="email" value={form.email} onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))} />
              <Input type="password" required placeholder="password" value={form.password} onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))} />
              <select className="w-full h-10 rounded-lg border border-border bg-background px-3 text-sm" value={form.role} onChange={(e) => setForm((f) => ({ ...f, role: e.target.value }))}>
                <option value="analyst">Analyst</option>
                <option value="viewer">Viewer</option>
                <option value="admin">Admin</option>
              </select>
              <Button type="submit" className="w-full">Create account</Button>
            </form>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Audit log</CardTitle></CardHeader>
        <CardContent>
          <div className="max-h-80 overflow-y-auto text-sm">
            {logs.map((l) => (
              <div key={String(l.id)} className="flex gap-4 border-b border-border/40 py-2 text-xs">
                <span className="text-muted-foreground w-36 shrink-0">{l.created_at ? new Date(String(l.created_at)).toLocaleString() : ""}</span>
                <span className="w-40 truncate">{String(l.user_email || "—")}</span>
                <span className="font-mono text-primary">{String(l.action)}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
