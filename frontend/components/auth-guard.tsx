"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/auth";
import { Skeleton } from "@/components/ui/skeleton";

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
    if (!loading && !user) router.replace("/login");
    if (!loading && requireAdmin && !isAdmin) router.replace("/dashboard");
    if (!loading && requireWrite && !canWrite) router.replace("/dashboard");
  }, [loading, user, requireWrite, requireAdmin, canWrite, isAdmin, router]);

  if (loading) return <div className="p-6"><Skeleton className="h-96 w-full rounded-xl" /></div>;
  if (!user) return null;
  if (requireAdmin && !isAdmin) return null;
  if (requireWrite && !canWrite) return null;

  return <>{children}</>;
}
