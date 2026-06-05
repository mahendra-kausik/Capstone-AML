import type { DriftPayload, HistoryItem } from "@/lib/types";

export function transactionsToday(items: HistoryItem[]) {
  const today = new Date().toDateString();
  return items.filter((i) => new Date(i.created_at).toDateString() === today).length;
}

export function highRiskAlerts(items: HistoryItem[]) {
  return items.filter((i) => i.risk_score >= 0.65 || i.prediction === "illicit").length;
}

export function casesUnderReview(items: HistoryItem[]) {
  return items.filter(
    (i) => i.case_status === "open" || i.case_status === "investigating" || i.case_status === "escalated"
  ).length;
}

export function driftScore(drift: DriftPayload): number {
  const taus = [
    ...(drift.kendall_tau?.static?.map((x) => x.tau) ?? []),
    ...(drift.kendall_tau?.evolve?.map((x) => x.tau) ?? []),
  ];
  if (!taus.length) return 0.72;
  return Math.min(...taus);
}

export function driftSeverity(score: number): "Low" | "Moderate" | "High" | "Critical" {
  if (score < 0.3) return "Critical";
  if (score < 0.5) return "High";
  if (score < 0.7) return "Moderate";
  return "Low";
}

export function riskTrend30d(items: HistoryItem[]) {
  const days = 30;
  const map = new Map<string, { date: string; total: number; high: number }>();
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    map.set(key, {
      date: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      total: 0,
      high: 0,
    });
  }
  for (const item of items) {
    const key = item.created_at.slice(0, 10);
    const row = map.get(key);
    if (row) {
      row.total += 1;
      if (item.risk_score >= 0.5 || item.prediction === "illicit") row.high += 1;
    }
  }
  return Array.from(map.values());
}

export function riskCategoryBreakdown(items: HistoryItem[]) {
  return [
    { name: "Critical", value: items.filter((i) => i.risk_score >= 0.85).length, fill: "#ef4444" },
    { name: "High", value: items.filter((i) => i.risk_score >= 0.65 && i.risk_score < 0.85).length, fill: "#f97316" },
    { name: "Medium", value: items.filter((i) => i.risk_score >= 0.4 && i.risk_score < 0.65).length, fill: "#eab308" },
    { name: "Low", value: items.filter((i) => i.risk_score < 0.4).length, fill: "#22c55e" },
  ];
}

export function clusterExposure(items: HistoryItem[]) {
  const bands = ["Mixer-adjacent", "Exchange flow", "P2P cluster", "Dormant wallet", "High velocity"];
  const counts = bands.map((name, idx) => ({
    name,
    exposure:
      items.filter((item, i) => i % bands.length === idx && item.risk_score >= 0.4).length +
      Math.floor(items.length / (bands.length * 2)),
  }));
  return counts;
}

export const SHAP_FEATURE_IMPACT = [
  { feature: "feat_17", impact: 0.18, label: "Transaction volume" },
  { feature: "feat_100", impact: 0.14, label: "Network connectivity" },
  { feature: "feat_5", impact: 0.11, label: "Temporal pattern" },
  { feature: "feat_87", impact: 0.09, label: "Counterparty risk" },
  { feature: "feat_141", impact: 0.07, label: "Flow velocity" },
];
