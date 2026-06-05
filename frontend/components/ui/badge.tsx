import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide transition-colors",
  {
    variants: {
      variant: {
        default: "border-primary/30 bg-primary/10 text-primary",
        secondary: "border-border bg-secondary text-secondary-foreground",
        destructive: "border-risk-critical/30 bg-risk-critical/10 text-risk-critical",
        outline: "border-border text-muted-foreground",
        critical: "border-risk-critical/40 bg-risk-critical/15 text-risk-critical",
        high: "border-risk-high/40 bg-risk-high/15 text-risk-high",
        medium: "border-risk-medium/40 bg-risk-medium/15 text-risk-medium",
        low: "border-risk-low/40 bg-risk-low/15 text-risk-low",
        success: "border-risk-low/40 bg-risk-low/15 text-risk-low",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
