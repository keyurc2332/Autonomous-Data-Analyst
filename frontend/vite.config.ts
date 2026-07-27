import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    watch: { usePolling: true },   // required for bind mounts on Windows
    proxy: {
      // Same-origin in dev, so no CORS and no base-URL juggling.
      "/api": { target: "http://api:8000", changeOrigin: true },
    },
  },
});
