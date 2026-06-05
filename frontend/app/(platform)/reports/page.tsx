"use client";

import { useState } from "react";
import { FileText, Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/contexts/toast";
import { downloadReportCsv, generateReport } from "@/lib/api";

const REPORTS = [
  {
    id: "compliance",
    title: "Compliance Report",
    desc: "SAR-ready summary of flagged transactions, model metrics, and drift status.",
    format: "CSV",
  },
  {
    id: "investigation",
    title: "AML Investigation Report",
    desc: "Case details, SHAP attributions, and network context for flagged alerts.",
    format: "JSON",
  },
  {
    id: "executive",
    title: "Executive Summary",
    desc: "KPI dashboard snapshot for leadership and board reporting.",
    format: "JSON",
  },
  {
    id: "risk_summary",
    title: "Risk Summary Export",
    desc: "Full high-risk transaction export for audit and regulatory review.",
    format: "CSV",
  },
];

export default function ReportsPage() {
  const { toast } = useToast();
  const [loading, setLoading] = useState<string | null>(null);

  async function handleGenerate(id: string, title: string, format: string) {
    setLoading(id);
    try {
      if (format === "CSV") {
        await downloadReportCsv(id);
        toast(`${title} downloaded`, "success");
      } else {
        const report = await generateReport(id, title);
        const blob = new Blob([JSON.stringify(report.summary, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${id}_report.json`;
        a.click();
        URL.revokeObjectURL(url);
        toast(`${title} generated`, "success");
      }
    } catch (e) {
      toast(e instanceof Error ? e.message : "Report generation failed", "error");
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      {REPORTS.map((r) => (
        <Card key={r.id} className="transition hover:border-primary/30">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-primary/10 p-2">
                <FileText className="h-5 w-5 text-primary" />
              </div>
              <div>
                <CardTitle>{r.title}</CardTitle>
                <CardDescription>{r.desc}</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Button
              variant="outline"
              size="sm"
              disabled={loading === r.id}
              onClick={() => handleGenerate(r.id, r.title, r.format)}
            >
              {loading === r.id ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Export {r.format}
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
