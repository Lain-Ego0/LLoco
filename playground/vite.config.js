import { defineConfig } from "vite";

// The playground is published below /playground/ on the personal website.
// Relative output keeps it portable if the parent site's URL changes.
export default defineConfig({ base: "./" });
