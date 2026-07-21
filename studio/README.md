# Studio

Studio is a real-time video editing app powered by Decart's Lucy 2.5 model. Point a camera at a scene, describe an edit, and watch the transformed video stream live.

It was cloned from opentxt's SolidStart + Effect skeleton and adapted for Decart's real-time video-to-video API.

## Run locally

Studio requires Node.js 22.5 or newer and pnpm.

```bash
pnpm install
```

Create `studio/.env` with a Decart API key:

```dotenv
DECART_API_KEY=<your Decart API key>
```

Start the development server:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

## Architecture

The permanent Decart API key stays on the server:

1. The browser calls `POST /api/decart/token`.
2. The server-side Effect service exchanges `DECART_API_KEY` for a short-lived Decart token and returns it as `{ "apiKey": "..." }`.
3. The browser creates an `@decartai/sdk` client and sends the camera stream over WebRTC with `client.realtime.connect(cameraStream, { onRemoteStream })`.
4. `onRemoteStream` receives Lucy 2.5's live edited stream for playback in the browser.
5. Prompt changes are applied without reconnecting through `realtimeClient.set({ prompt })`.

## Security and cost

The token endpoint is IP rate-limited to reduce accidental or abusive session creation. Add real authentication, authorization, quotas, and a shared rate-limit store before scaling or running multiple instances.

Each Decart real-time session is paid usage at **$0.04 per second**. Keep the permanent API key server-side and monitor session creation and duration.
