"use client";

import { usePathname } from "next/navigation";
import { AuthGuard } from "@/components/auth-guard";
import { Sidebar } from "@/components/layout/sidebar";
import { TopBar } from "@/components/layout/top-bar";

const PAGE_META: Record<string, { title: string; description: string }> = {
  "/dashboard": {
    title: "Executive Dashboard",
    description: "Real-time AML risk intelligence and compliance overview",
  },
  "/investigations": {
    title: "Investigation Center",
    description: "Case management and analyst workflows",
  },
  "/transactions": {
    title: "Transactions",
    description: "Scored cryptocurrency transaction registry",
  },
  "/alerts": {
    title: "Alerts Center",
    description: "Risk alerts and escalation queue",
  },
  "/explainability": {
    title: "Explainability",
    description: "SHAP-based model reasoning and feature attribution",
  },
  "/drift": {
    title: "Drift Monitor",
    description: "Concept drift and model stability analytics",
  },
  "/reports": {
    title: "Reports",
    description: "Compliance and executive reporting",
  },
  "/admin": {
    title: "Administration",
    description: "Users, audit logs, and system health",
  },
  "/network": {
    title: "Network Analysis",
    description: "Transaction graph and relationship tracing",
  },
  "/upload": {
    title: "Batch Upload",
    description: "Import transaction data for AML scoring",
  },
};

function resolveMeta(pathname: string) {
  if (pathname.startsWith("/transactions/")) {
    return { title: "Transaction Detail", description: "Risk assessment and investigation context" };
  }
  for (const [path, meta] of Object.entries(PAGE_META)) {
    if (pathname === path || pathname.startsWith(path + "/")) return meta;
  }
  return { title: "AML Intelligence", description: "" };
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const meta = resolveMeta(pathname);

  return (
    <AuthGuard>
      <div className="min-h-screen bg-background">
        <Sidebar />
        <div className="pl-64">
          <TopBar title={meta.title} description={meta.description} />
          <main className="p-6 animate-fade-in">{children}</main>
        </div>
      </div>
    </AuthGuard>
  );
}
