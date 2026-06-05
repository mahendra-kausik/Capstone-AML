"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/auth";
import { Loading } from "@/components/ui/loading";

export function AuthGuard({
  children,
  requireWrite = false,
  requireAdmin = false,
}: {
  children: React.ReactNode;
  requireWrite?: boolean;
  requireAdmin?: boolean;
}) {
  const { user, loading, canWrite, isAdmin } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) router.replace("/");
    if (!loading && requireAdmin && !isAdmin) router.replace("/dashboard");
    if (!loading && requireWrite && !canWrite) router.replace("/dashboard");
  }, [loading, user, requireWrite, requireAdmin, canWrite, isAdmin, router]);

  if (loading) return <Loading label="Loading session…" />;
  if (!user) return null;
  if (requireAdmin && !isAdmin) return null;
  if (requireWrite && !canWrite) return null;

  return <>{children}</>;
}
