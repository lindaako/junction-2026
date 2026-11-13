import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api/* is proxied to the FastAPI backend, so there are no CORS headaches.
// In production set VITE_API_URL (e.g. your Modal / Cloudflare tunnel URL).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
});
