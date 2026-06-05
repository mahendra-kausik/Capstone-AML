"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SHAP_FEATURE_IMPACT } from "@/lib/dashboard-data";

export function ShapImpact() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>SHAP Feature Impact</CardTitle>
        <CardDescription>Top drivers of illicit classification (research aggregate)</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={SHAP_FEATURE_IMPACT} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis type="number" domain={[0, 0.25]} tick={{ fontSize: 10 }} />
              <YAxis dataKey="label" type="category" width={110} tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  background: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: 8,
                }}
              />
              <Bar dataKey="impact" fill="hsl(262 83% 58%)" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
