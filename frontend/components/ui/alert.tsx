import clsx from "clsx";

export function Alert({
  children,
  variant = "error",
}: {
  children: React.ReactNode;
  variant?: "error" | "success" | "info";
}) {
  const styles = {
    error: "border-red-200 bg-red-50 text-red-800",
    success: "border-emerald-200 bg-emerald-50 text-emerald-800",
    info: "border-blue-200 bg-blue-50 text-blue-800",
  };
  return (
    <div className={clsx("rounded-lg border px-4 py-3 text-sm", styles[variant])}>
      {children}
    </div>
  );
}
