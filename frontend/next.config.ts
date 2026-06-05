import type { NextConfig } from "next";

const API_TARGET = process.env.API_PROXY_TARGET || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  output: "standalone",
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
