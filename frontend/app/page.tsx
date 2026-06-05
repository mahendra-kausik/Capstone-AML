"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import {
  Activity,
  ArrowRight,
  Brain,
  Network,
  Shield,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import { useAuth } from "@/contexts/auth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const STATS = [
  { label: "Static GCN AUROC", value: "0.857" },
  { label: "EvolveGCN AUROC", value: "0.767" },
  { label: "Features Engineered", value: "165" },
  { label: "Drift Windows Monitored", value: "4" },
];

const CAPABILITIES = [
  {
    icon: Network,
    title: "Graph Neural Networks",
    desc: "Static GCN and EvolveGCN-H detect illicit flows across transaction networks.",
  },
  {
    icon: Brain,
    title: "SHAP Explainability",
    desc: "Kernel SHAP attributions provide audit-ready reasoning for every alert.",
  },
  {
    icon: Activity,
    title: "Concept Drift Monitoring",
    desc: "Kendall τ stability analysis catches model degradation before compliance risk.",
  },
  {
    icon: Sparkles,
    title: "Investigation Workflows",
    desc: "Enterprise case management with escalation, notes, and network tracing.",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const { user, loading } = useAuth();

  useEffect(() => {
    if (!loading && user) router.replace("/dashboard");
  }, [user, loading, router]);

  return (
    <div className="min-h-screen gradient-mesh">
      <header className="border-b border-border/50 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <div className="flex items-center gap-2">
            <Shield className="h-7 w-7 text-primary" />
            <span className="text-lg font-bold">AML Intelligence</span>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="ghost" asChild>
              <Link href="/login">Sign in</Link>
            </Button>
            <Button asChild>
              <Link href="/login">View Platform</Link>
            </Button>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-6 py-24 text-center">
        <Badge variant="default" className="mb-6 normal-case">
          Venture-grade AML Intelligence
        </Badge>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
          AI-Powered Anti-Money
          <br />
          <span className="text-gradient">Laundering Intelligence</span>
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
          Detect illicit cryptocurrency transactions using graph neural networks,
          explainable AI, and concept drift monitoring.
        </p>
        <div className="mt-10 flex flex-wrap justify-center gap-4">
          <Button size="lg" asChild>
            <Link href="mailto:demo@aml-intelligence.com">
              Book Demo <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
          <Button size="lg" variant="outline" asChild>
            <Link href="/login">View Platform</Link>
          </Button>
        </div>
      </section>

      <section className="border-y border-border/50 bg-card/30 py-12">
        <div className="mx-auto grid max-w-6xl grid-cols-2 gap-8 px-6 md:grid-cols-4">
          {STATS.map((s) => (
            <div key={s.label} className="text-center">
              <p className="text-3xl font-bold text-primary">{s.value}</p>
              <p className="mt-1 text-xs text-muted-foreground uppercase tracking-wider">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-20">
        <h2 className="text-center text-2xl font-bold">Platform Capabilities</h2>
        <p className="mx-auto mt-2 max-w-xl text-center text-muted-foreground">
          Built on published Elliptic research — production-ready inference, no retraining required.
        </p>
        <div className="mt-12 grid gap-6 sm:grid-cols-2">
          {CAPABILITIES.map((c) => (
            <div key={c.title} className="glass-card p-6 transition hover:border-primary/30">
              <c.icon className="h-8 w-8 text-primary" />
              <h3 className="mt-4 font-semibold">{c.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{c.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="border-t border-border/50 bg-secondary/20 py-20">
        <div className="mx-auto max-w-6xl px-6">
          <h2 className="text-2xl font-bold">Architecture</h2>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {[
              { step: "01", title: "Ingest", desc: "CSV batch upload · 165-dim Elliptic features" },
              { step: "02", title: "Score", desc: "Static GCN / EvolveGCN-H inference via FastAPI" },
              { step: "03", title: "Investigate", desc: "SHAP · network graph · case management · drift" },
            ].map((a) => (
              <div key={a.step} className="rounded-xl border border-border p-5">
                <span className="text-xs font-mono text-primary">{a.step}</span>
                <h3 className="mt-2 font-semibold">{a.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{a.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-20 text-center">
        <TrendingUp className="mx-auto h-10 w-10 text-primary" />
        <h2 className="mt-4 text-2xl font-bold">Ready for compliance teams</h2>
        <p className="mt-2 text-muted-foreground">Trusted model benchmarks · Full audit trail · Role-based access</p>
        <Button className="mt-8" size="lg" asChild>
          <Link href="/login">Access Platform</Link>
        </Button>
      </section>

      <footer className="border-t border-border py-8 text-center text-xs text-muted-foreground">
        AML Intelligence Platform · Temporal GNN Research · Elliptic Bitcoin Dataset
      </footer>
    </div>
  );
}
