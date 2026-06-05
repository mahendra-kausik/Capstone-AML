"use client";

import { useEffect, useState } from "react";
import { AuthGuard } from "@/components/auth-guard";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Loading } from "@/components/ui/loading";
import { PageHeader } from "@/components/ui/page-header";
import { getAuditLogs, getUsers, registerUser } from "@/lib/api";
import type { User } from "@/lib/types";

export default function AdminPage() {
  return (
    <AuthGuard requireAdmin>
      <AdminContent />
    </AuthGuard>
  );
}

function AdminContent() {
  const [users, setUsers] = useState<User[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({ email: "", password: "", role: "analyst" });

  useEffect(() => {
    Promise.all([getUsers(), getAuditLogs({ limit: 50 })])
      .then(([u, l]) => {
        setUsers(u);
        setLogs(l.items);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Load failed"))
      .finally(() => setLoading(false));
  }, []);

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setSuccess("");
    try {
      await registerUser(form);
      setSuccess(`User ${form.email} created.`);
      setForm({ email: "", password: "", role: "analyst" });
      const u = await getUsers();
      setUsers(u);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Registration failed");
    }
  }

  if (loading) return <Loading />;

  return (
    <div>
      <PageHeader
        title="Administration"
        description="User management, role-based access, and compliance audit trail."
      />

      {error && <div className="mb-4"><Alert variant="error">{error}</Alert></div>}
      {success && <div className="mb-4"><Alert variant="success">{success}</Alert></div>}

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="rounded-xl border bg-white p-5 shadow-sm">
          <h2 className="font-semibold">Users</h2>
          <table className="mt-4 w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr>
                <th className="pb-2">Email</th>
                <th className="pb-2">Role</th>
                <th className="pb-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-t">
                  <td className="py-2">{u.email}</td>
                  <td className="py-2"><Badge label={u.role} variant={u.role} /></td>
                  <td className="py-2 text-slate-500">active</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <form onSubmit={handleRegister} className="rounded-xl border bg-white p-5 shadow-sm">
          <h2 className="font-semibold">Register user</h2>
          <div className="mt-4 space-y-3">
            <input
              type="email"
              required
              placeholder="email@example.com"
              className="w-full rounded-lg border px-3 py-2 text-sm"
              value={form.email}
              onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))}
            />
            <input
              type="password"
              required
              minLength={8}
              placeholder="Password"
              className="w-full rounded-lg border px-3 py-2 text-sm"
              value={form.password}
              onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
            />
            <select
              className="w-full rounded-lg border px-3 py-2 text-sm"
              value={form.role}
              onChange={(e) => setForm((f) => ({ ...f, role: e.target.value }))}
            >
              <option value="analyst">Analyst</option>
              <option value="viewer">Viewer (read-only)</option>
              <option value="admin">Administrator</option>
            </select>
            <button type="submit" className="w-full rounded-lg bg-brand py-2 text-sm font-medium text-white">
              Create account
            </button>
          </div>
        </form>
      </div>

      <div className="mt-8 rounded-xl border bg-white shadow-sm">
        <h2 className="border-b p-4 font-semibold">Audit log</h2>
        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="p-3">Time</th>
                <th className="p-3">User</th>
                <th className="p-3">Action</th>
                <th className="p-3">Resource</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((l) => (
                <tr key={String(l.id)} className="border-b">
                  <td className="p-3 text-xs whitespace-nowrap">
                    {l.created_at ? new Date(String(l.created_at)).toLocaleString() : "—"}
                  </td>
                  <td className="p-3">{String(l.user_email || "—")}</td>
                  <td className="p-3 font-mono text-xs">{String(l.action)}</td>
                  <td className="p-3 text-slate-500">{String(l.resource)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
