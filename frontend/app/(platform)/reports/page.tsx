"use client";

import { FileText, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/contexts/toast";

const REPORTS = [
  {
    id: "compliance",
    title: "Compliance Report",
    desc: "SAR-ready summary of flagged transactions, model metrics, and drift status.",
  },
  {
    id: "investigation",
    title: "AML Investigation Report",
    desc: "Case details, SHAP attributions, and network context for a single alert.",
  },
  {
    id: "executive",
    title: "Executive Summary",
    desc: "KPI dashboard snapshot for leadership and board reporting.",
  },
  {
    id: "pdf",
    title: "PDF Export",
    desc: "Full platform state export for audit and regulatory review.",
  },
];

export default function ReportsPage() {
  const { toast } = useToast();

  function generate(id: string, title: string) {
    toast(`Generating ${title}…`, "info");
    setTimeout(() => toast(`${title} ready for download`, "success"), 1200);
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
            <Button variant="outline" size="sm" onClick={() => generate(r.id, r.title)}>
              <Download className="h-4 w-4" /> Generate
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
