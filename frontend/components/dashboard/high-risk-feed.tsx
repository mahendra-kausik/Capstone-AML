"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatPercent, riskSeverity } from "@/lib/utils";
import type { HistoryItem } from "@/lib/types";

interface HighRiskFeedProps {
  items: HistoryItem[];
}

export function HighRiskFeed({ items }: HighRiskFeedProps) {
  const feed = items
    .filter((i) => i.risk_score >= 0.5 || i.prediction === "illicit")
    .slice(0, 6);

  return (
    <Card className="h-full">
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle>High-Risk Feed</CardTitle>
          <CardDescription>Latest flagged transactions</CardDescription>
        </div>
        <Link href="/alerts" className="text-xs text-primary hover:underline flex items-center gap-1">
          View all <ArrowRight className="h-3 w-3" />
        </Link>
      </CardHeader>
      <CardContent className="space-y-2">
        {feed.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">No high-risk transactions yet</p>
        ) : (
          feed.map((item) => {
            const sev = riskSeverity(item.risk_score);
            return (
              <Link
                key={item.prediction_id}
                href={`/transactions/${item.prediction_id}`}
                className="flex items-center justify-between rounded-lg border border-border/60 bg-secondary/30 px-3 py-2.5 transition hover:border-primary/30 hover:bg-secondary/50"
              >
                <div>
                  <p className="font-mono text-xs">{item.tx_id}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold">{formatPercent(item.risk_score)}</span>
                  <Badge variant={sev}>{sev}</Badge>
                </div>
              </Link>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}
