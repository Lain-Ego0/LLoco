import { defineConfig } from "vite";
export default defineConfig({
  base: "/viewer/",
  build: { outDir: "../static/viewer", emptyOutDir: true },
});
