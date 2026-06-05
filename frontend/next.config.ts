import type { NextConfig } from "next";

const API_TARGET = process.env.API_PROXY_TARGET || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  // Avoids SegmentViewNode manifest corruption in Next.js 15.5 dev (webpack HMR bug).
  experimental: {
    devtoolSegmentExplorer: false,
  },
  async redirects() {
    return [
      { source: "/history", destination: "/investigations", permanent: true },
      { source: "/analysis", destination: "/transactions", permanent: true },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${API_TARGET}/api/v1/:path*`,
      },
      {
        source: "/health",
        destination: `${API_TARGET}/health`,
      },
    ];
  },
};

export default nextConfig;
