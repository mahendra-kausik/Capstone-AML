"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Shield } from "lucide-react";
import { useAuth } from "@/contexts/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

const DEMO = [
  { email: "analyst@example.com", role: "Analyst" },
  { email: "admin@example.com", role: "Admin" },
  { email: "viewer@example.com", role: "Viewer" },
];

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState("analyst@example.com");
  const [password, setPassword] = useState("demoaml2024");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await login(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen">
      <div className="hidden w-1/2 flex-col justify-between bg-secondary/30 p-12 lg:flex gradient-mesh">
        <div className="flex items-center gap-2">
          <Shield className="h-8 w-8 text-primary" />
          <span className="text-xl font-bold">AML Intelligence</span>
        </div>
        <div>
          <h1 className="text-3xl font-bold leading-tight">
            Enterprise AML
            <br />
            <span className="text-gradient">Intelligence Platform</span>
          </h1>
          <p className="mt-4 max-w-md text-muted-foreground">
            Graph neural networks, SHAP explainability, and drift monitoring for cryptocurrency compliance.
          </p>
          <div className="mt-8 flex gap-3">
            <Badge variant="success" className="normal-case">AUROC 0.857</Badge>
            <Badge variant="default" className="normal-case">SHAP Powered</Badge>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">© AML Intelligence · Elliptic Research</p>
      </div>

      <div className="flex flex-1 items-center justify-center p-8">
        <div className="w-full max-w-md">
          <h2 className="text-2xl font-bold">Sign in</h2>
          <p className="mt-1 text-sm text-muted-foreground">Access the compliance dashboard</p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Email</label>
              <Input
                type="email"
                required
                className="mt-1.5"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Password</label>
              <Input
                type="password"
                required
                className="mt-1.5"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            {error && (
              <p className="rounded-lg border border-risk-critical/30 bg-risk-critical/10 px-3 py-2 text-sm text-risk-critical">
                {error}
              </p>
            )}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Signing in…" : "Sign in to platform"}
            </Button>
          </form>

          <div className="mt-8 rounded-xl border border-border bg-secondary/30 p-4">
            <p className="text-xs font-semibold text-muted-foreground">Demo accounts · password: demoaml2024</p>
            <ul className="mt-3 space-y-2">
              {DEMO.map((d) => (
                <li key={d.email}>
                  <button
                    type="button"
                    className="text-sm text-primary hover:underline"
                    onClick={() => setEmail(d.email)}
                  >
                    {d.email}
                  </button>
                  <span className="text-muted-foreground text-xs"> — {d.role}</span>
                </li>
              ))}
            </ul>
          </div>

          <Link href="/" className="mt-6 block text-center text-sm text-muted-foreground hover:text-primary">
            ← Back to homepage
          </Link>
        </div>
      </div>
    </div>
  );
}
