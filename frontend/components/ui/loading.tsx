export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-slate-500">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-brand border-t-transparent" />
      <p className="mt-4 text-sm">{label}</p>
    </div>
  );
}
