"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Shield } from "lucide-react";
import { useAuth } from "@/contexts/auth";
import { Alert } from "@/components/ui/alert";
import { Loading } from "@/components/ui/loading";

const DEMO_ACCOUNTS = [
  { email: "analyst@example.com", role: "Analyst (full access)" },
  { email: "viewer@example.com", role: "Viewer (read-only)" },
  { email: "admin@example.com", role: "Administrator" },
];

export default function HomePage() {
  const router = useRouter();
  const { user, loading, login } = useAuth();
  const [email, setEmail] = useState("analyst@example.com");
  const [password, setPassword] = useState("demoaml2024");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (loading) return <Loading label="Checking session…" />;
  if (user) {
    router.replace("/dashboard");
    return <Loading />;
  }

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await login(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-lg">
      <div className="flex items-center gap-3">
        <Shield className="h-10 w-10 text-brand" />
        <div>
          <h1 className="text-2xl font-bold">AML Intelligence Platform</h1>
          <p className="text-slate-600">Elliptic Bitcoin · Static GCN · EvolveGCN-H</p>
        </div>
      </div>

      <form
        onSubmit={handleLogin}
        className="mt-8 space-y-4 rounded-xl border bg-white p-6 shadow-sm"
      >
        <div>
          <label className="text-sm font-medium text-slate-700">Email</label>
          <input
            type="email"
            required
            className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div>
          <label className="text-sm font-medium text-slate-700">Password</label>
          <input
            type="password"
            required
            className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        {error && <Alert variant="error">{error}</Alert>}
        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-lg bg-brand py-2.5 font-medium text-white hover:bg-brand-dark disabled:opacity-60"
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <div className="mt-6 rounded-xl border bg-white p-4 text-sm text-slate-600 shadow-sm">
        <p className="font-medium text-slate-800">Demo accounts (password: demoaml2024)</p>
        <ul className="mt-2 space-y-1">
          {DEMO_ACCOUNTS.map((a) => (
            <li key={a.email}>
              <button
                type="button"
                className="text-brand hover:underline"
                onClick={() => setEmail(a.email)}
              >
                {a.email}
              </button>
              <span className="text-slate-400"> — {a.role}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
