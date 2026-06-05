"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LogOut, Shield } from "lucide-react";
import { useAuth, roleLabel } from "@/contexts/auth";
import { Badge } from "@/components/ui/badge";

const links = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/upload", label: "Upload", write: true },
  { href: "/analysis", label: "Analysis", write: true },
  { href: "/explainability", label: "Explainability" },
  { href: "/network", label: "Network" },
  { href: "/history", label: "Cases" },
  { href: "/drift", label: "Drift" },
  { href: "/admin", label: "Admin", admin: true },
];

export function Nav() {
  const path = usePathname();
  const { user, loading, logout, isAdmin, canWrite } = useAuth();
  const isLogin = path === "/";

  if (isLogin) return null;

  return (
    <header className="border-b border-slate-200 bg-white shadow-sm">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3">
        <Link href={user ? "/dashboard" : "/"} className="flex items-center gap-2">
          <Shield className="h-6 w-6 text-brand" />
          <span className="text-lg font-bold text-brand">AML Intelligence</span>
        </Link>

        {!loading && user && (
          <>
            <nav className="hidden flex-wrap gap-1 md:flex">
              {links
                .filter((l) => (!l.admin || isAdmin) && (!l.write || canWrite))
                .map((l) => (
                  <Link
                    key={l.href}
                    href={l.href}
                    className={
                      path === l.href || path.startsWith(l.href + "/")
                        ? "rounded-lg bg-brand/10 px-3 py-1.5 text-sm font-semibold text-brand"
                        : "rounded-lg px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50 hover:text-brand"
                    }
                  >
                    {l.label}
                  </Link>
                ))}
            </nav>

            <div className="flex items-center gap-3">
              <div className="hidden text-right sm:block">
                <p className="text-xs font-medium text-slate-900">{user.email}</p>
                <Badge label={roleLabel(user.role)} variant={user.role} />
              </div>
              <button
                onClick={logout}
                className="flex items-center gap-1 rounded-lg border px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50"
                title="Sign out"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Sign out</span>
              </button>
            </div>
          </>
        )}
      </div>
    </header>
  );
}
