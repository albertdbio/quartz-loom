import { solidStart } from "@solidjs/start/config"
import { nitroV2Plugin } from "@solidjs/vite-plugin-nitro-2"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [
    solidStart(),
    // Nitro v2, node-server preset -> emits .output/server/index.mjs (run with
    // `node .output/server/index.mjs`). Same verified path mochi/core uses; the
    // Nitro 3 migration is deferred until SolidStart v2 reaches beta.
    nitroV2Plugin({
      preset: "node-server",
      compatibilityDate: "2026-07-06",
    }),
  ],
  server: {
    // Keep the dev server on 3000 (Vite defaults to 5173) so the Expo app's
    // default EXPO_PUBLIC_API_URL and the docs agree. `host: true` binds all
    // interfaces so a phone on the same LAN can reach the API during
    // development (the production node-server build already listens on [::]).
    host: true,
    port: 3000,
  },
})
