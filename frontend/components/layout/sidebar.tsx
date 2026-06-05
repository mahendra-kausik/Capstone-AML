"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  FileBarChart,
  FileText,
  LayoutDashboard,
  Network,
  Scale,
  Settings,
  Shield,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/auth";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/investigations", label: "Investigations", icon: Scale },
  { href: "/transactions", label: "Transactions", icon: Network },
  { href: "/alerts", label: "Alerts", icon: AlertTriangle, badge: true },
  { href: "/explainability", label: "Explainability", icon: Sparkles },
  { href: "/drift", label: "Drift Monitor", icon: Activity },
  { href: "/reports", label: "Reports", icon: FileText },
];

export function Sidebar() {
  const pathname = usePathname();
  const { user, isAdmin } = useAuth();

  return (
    <aside className="fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border bg-sidebar">
      <div className="flex h-16 items-center gap-2.5 border-b border-border px-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15">
          <Shield className="h-5 w-5 text-primary" />
        </div>
        <div>
          <p className="text-sm font-bold tracking-tight">AML Intelligence</p>
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Enterprise</p>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto p-3">
        <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Platform
        </p>
        {navItems.map((item) => {
          const active = pathname === item.href || pathname.startsWith(item.href + "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
                active
                  ? "bg-primary/10 text-primary shadow-glow"
                  : "text-sidebar-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <Icon className={cn("h-4 w-4 shrink-0", active && "text-primary")} />
              <span className="flex-1">{item.label}</span>
              {item.badge && (
                <span className="h-1.5 w-1.5 rounded-full bg-risk-critical animate-pulse" />
              )}
            </Link>
          );
        })}

        {isAdmin && (
          <>
            <Separator className="my-3" />
            <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              Admin
            </p>
            <Link
              href="/admin"
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
                pathname.startsWith("/admin")
                  ? "bg-primary/10 text-primary"
                  : "text-sidebar-foreground hover:bg-accent"
              )}
            >
              <Settings className="h-4 w-4" />
              Administration
            </Link>
          </>
        )}
      </nav>

      <div className="border-t border-border p-4">
        <div className="rounded-lg bg-secondary/50 p-3">
          <div className="flex items-center gap-2">
            <FileBarChart className="h-4 w-4 text-primary" />
            <p className="text-xs font-semibold">Model Performance</p>
          </div>
          <div className="mt-2 space-y-1 text-[11px] text-muted-foreground">
            <div className="flex justify-between">
              <span>Static GCN</span>
              <span className="font-mono text-foreground">AUROC 0.857</span>
            </div>
            <div className="flex justify-between">
              <span>EvolveGCN-H</span>
              <span className="font-mono text-foreground">AUROC 0.767</span>
            </div>
          </div>
        </div>
        {user && (
          <div className="mt-3 flex items-center justify-between">
            <p className="truncate text-xs text-muted-foreground">{user.email}</p>
            <Badge variant="secondary" className="text-[9px]">
              {user.role}
            </Badge>
          </div>
        )}
      </div>
    </aside>
  );
}
