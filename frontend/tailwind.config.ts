import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: { DEFAULT: "#1a56a0", dark: "#0f3460" },
        risk: { high: "#dc2626", low: "#16a34a" },
      },
    },
  },
  plugins: [],
};
export default config;
