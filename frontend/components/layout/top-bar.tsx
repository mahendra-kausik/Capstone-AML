"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Bell, LogOut, Search } from "lucide-react";
import { useAuth } from "@/contexts/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { getAlerts } from "@/lib/api";

interface TopBarProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
}

export function TopBar({ title, description, actions }: TopBarProps) {
  const { user, logout } = useAuth();
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [unreadAlerts, setUnreadAlerts] = useState(0);

  useEffect(() => {
    getAlerts({ status: "open", limit: 1 })
      .then((a) => setUnreadAlerts(a.unread_count))
      .catch(() => {});
  }, []);

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    const q = search.trim();
    if (!q) return;
    router.push(`/transactions?search=${encodeURIComponent(q)}`);
  }

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="flex h-16 items-center justify-between gap-4 px-6">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-semibold tracking-tight">{title}</h1>
          {description && (
            <p className="truncate text-xs text-muted-foreground">{description}</p>
          )}
        </div>

        <div className="flex items-center gap-3">
          <form onSubmit={handleSearch} className="relative hidden md:block">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search transactions…"
              className="w-64 pl-9 h-9 bg-secondary/50"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </form>

          <Button variant="ghost" size="icon" className="relative" asChild>
            <Link href="/alerts">
              <Bell className="h-4 w-4" />
              {unreadAlerts > 0 && (
                <span className="absolute right-1.5 top-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-risk-critical text-[9px] font-bold text-white">
                  {unreadAlerts > 9 ? "9+" : unreadAlerts}
                </span>
              )}
            </Link>
          </Button>

          {actions}

          {user && (
            <div className="flex items-center gap-2 border-l border-border pl-3">
              <Badge variant="secondary" className="hidden sm:inline-flex capitalize">
                {user.role}
              </Badge>
              <Button variant="ghost" size="icon" onClick={logout} title="Sign out">
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
