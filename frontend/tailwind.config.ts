import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        neon: {
          pink: "#ff6ec7",
          blue: "#00c6ff",
          purple: "#7f5af0"
        }
      }
    }
  },
  plugins: []
};

export default config;
