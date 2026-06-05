import { ReactNode } from "react";
import clsx from "clsx";

export function Card({
  title,
  value,
  subtitle,
  className,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        "rounded-xl border border-slate-200 bg-white p-5 shadow-sm",
        className
      )}
    >
      <p className="text-sm font-medium text-slate-500">{title}</p>
      <p className="mt-2 text-3xl font-semibold text-slate-900">{value}</p>
      {subtitle && <p className="mt-1 text-xs text-slate-400">{subtitle}</p>}
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">{children}</div>
  );
}
