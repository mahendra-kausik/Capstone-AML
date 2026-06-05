import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/layout/nav";
import { AuthProvider } from "@/contexts/auth";

export const metadata: Metadata = {
  title: "AML Intelligence Platform",
  description: "Temporal GNN AML — Static GCN & EvolveGCN-H",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        <AuthProvider>
          <Nav />
          <main className="mx-auto max-w-7xl px-4 py-8">{children}</main>
        </AuthProvider>
      </body>
    </html>
  );
}
