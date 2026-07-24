// @refresh reload
import { createHandler, StartServer } from "@solidjs/start/server"

export default createHandler(() => (
  <StartServer
    document={({ assets, children, scripts }) => (
      <html lang="en">
        <head>
          <meta charset="utf-8" />
          {/* viewport-fit=cover lets the full-screen wand run edge-to-edge under
              the notch/home indicator; its floating controls inset themselves
              with env(safe-area-inset-*). user-scalable=no keeps a two-finger
              pinch from breaking the fixed camera stage. */}
          <meta
            name="viewport"
            content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no"
          />
          <meta name="theme-color" content="#07070d" />
          <meta name="apple-mobile-web-app-capable" content="yes" />
          <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
          <title>studio</title>
          {assets}
        </head>
        <body>
          <div id="app">{children}</div>
          {scripts}
        </body>
      </html>
    )}
  />
))
