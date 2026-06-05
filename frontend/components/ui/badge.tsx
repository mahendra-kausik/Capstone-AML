import clsx from "clsx";

const styles: Record<string, string> = {
  licit: "bg-emerald-50 text-emerald-800 border-emerald-200",
  illicit: "bg-red-50 text-red-800 border-red-200",
  open: "bg-amber-50 text-amber-800 border-amber-200",
  investigating: "bg-blue-50 text-blue-800 border-blue-200",
  escalated: "bg-orange-50 text-orange-800 border-orange-200",
  closed: "bg-slate-100 text-slate-600 border-slate-200",
  false_positive: "bg-slate-100 text-slate-500 border-slate-200",
  critical: "bg-red-100 text-red-900 border-red-300",
  high: "bg-orange-50 text-orange-800 border-orange-200",
  medium: "bg-amber-50 text-amber-800 border-amber-200",
  low: "bg-slate-50 text-slate-600 border-slate-200",
  admin: "bg-violet-50 text-violet-800 border-violet-200",
  analyst: "bg-brand/10 text-brand border-brand/20",
  viewer: "bg-slate-50 text-slate-600 border-slate-200",
};

export function Badge({ label, variant }: { label: string; variant?: string }) {
  const key = (variant || label).toLowerCase().replace(" ", "_");
  return (
    <span
      className={clsx(
        "inline-flex rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize",
        styles[key] || "bg-slate-50 text-slate-700 border-slate-200"
      )}
    >
      {label}
    </span>
  );
}
