import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(
  new URL("../static/streaming_demo.js", import.meta.url),
  "utf8",
);
const htmlSource = readFileSync(
  new URL("../static/streaming_demo.html", import.meta.url),
  "utf8",
);

function bootClient({ bitmapFactory } = {}) {
  let lastSocket = null;
  const animationFrames = [];
  const status = { value: "" };
  function eventTarget(properties = {}) {
    const listeners = new Map();
    return {
      ...properties,
      addEventListener(type, listener) {
        listeners.set(type, listener);
      },
      dispatch(type, event = {}) {
        listeners.get(type)?.(event);
      },
    };
  }
  const form = eventTarget();
  const prompt = eventTarget({ value: "test prompt" });
  const seed = { value: "7" };
  const enhancePrompt = eventTarget();
  const drawing = { count: 0, frames: [] };
  const canvas = {
    width: 8,
    height: 8,
    getContext() {
      return {
        drawImage(image) {
          drawing.count += 1;
          drawing.frames.push(image.tag);
        },
      };
    },
  };
  const body = { dataset: {} };
  const document = {
    body,
    querySelector(selector) {
      return {
        "#status": status,
        "#prompt-form": form,
        "#prompt": prompt,
        "#seed": seed,
        "#enhance-prompt": enhancePrompt,
        "#frame": canvas,
      }[selector];
    },
  };

  class FakeWebSocket {
    static OPEN = 1;

    constructor() {
      this.readyState = FakeWebSocket.OPEN;
      this.listeners = new Map();
      this.sent = [];
      this.closed = [];
      lastSocket = this;
    }

    addEventListener(type, listener) {
      this.listeners.set(type, listener);
    }

    send(value) {
      this.sent.push(JSON.parse(value));
    }

    close(code, reason) {
      this.closed.push({ code, reason });
    }

    dispatch(type, event) {
      this.listeners.get(type)?.(event);
    }
  }

  const context = vm.createContext({
    Blob,
    WebSocket: FakeWebSocket,
    console,
    createImageBitmap: bitmapFactory ?? (async (blob) => ({
      close() {},
      tag: await blob.text(),
    })),
    document,
    location: { host: "127.0.0.1:8765", protocol: "http:" },
    performance: { now: () => 0 },
    requestAnimationFrame(callback) {
      animationFrames.push(callback);
      return animationFrames.length;
    },
    setTimeout(callback) {
      callback();
      return 1;
    },
  });
  vm.runInContext(source, context, { filename: "streaming_demo.js" });
  assert.ok(lastSocket);
  return {
    animationFrames,
    body,
    drawing,
    enhancePrompt,
    form,
    prompt,
    seed,
    socket: lastSocket,
    status,
  };
}

function text(socket, message) {
  socket.dispatch("message", { data: JSON.stringify(message) });
}

async function flushMicrotasks() {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((resolve) => setImmediate(resolve));
}

function payload(value) {
  return new TextEncoder().encode(value).buffer;
}

async function advancePaint(client) {
  assert.ok(client.animationFrames.length > 0, "expected a queued animation frame");
  client.animationFrames.shift()();
  await flushMicrotasks();
}

function acceptJob(client, jobId = "job-a") {
  text(client.socket, {
    type: "start_accepted",
    job_id: jobId,
    latent_frames: 21,
    expected_rgb_frames: 81,
  });
}

test("live presentation targets 24 fps while completed replay remains 16 fps", () => {
  assert.match(source, /const streamFrameIntervalMs = 1000 \/ 24;/);
  assert.match(source, /const replayFrameIntervalMs = 1000 \/ 16;/);
  assert.match(source, /frame\.frameMediaType,\s*replayFrameIntervalMs,/);
  assert.match(source, /chunk\.frame_media_type,\s*streamFrameIntervalMs,/);
});

function fixedChunkFrameCount(chunkIndex) {
  return chunkIndex === 0 ? 1 : 4;
}

function fixedChunkFirstFrameIndex(chunkIndex) {
  return chunkIndex === 0 ? 0 : 1 + (chunkIndex - 1) * 4;
}

function sendFixedChunk(
  client,
  {
    chunkIndex,
    frameLabels,
    jobId = "job-a",
    payloadByteLength = null,
  },
) {
  const frameCount = fixedChunkFrameCount(chunkIndex);
  assert.equal(frameLabels.length, frameCount);
  text(client.socket, {
    type: "chunk",
    kind: "chunk_ready",
    job_id: jobId,
    chunk_index: chunkIndex,
    first_frame_index: fixedChunkFirstFrameIndex(chunkIndex),
    frame_count: frameCount,
    frame_media_type: "image/png",
    renderable: true,
  });
  for (const label of frameLabels) {
    const frame = payload(label);
    if (payloadByteLength !== null) {
      Object.defineProperty(frame, "byteLength", { value: payloadByteLength });
    }
    client.socket.dispatch("message", { data: frame });
  }
  text(client.socket, {
    type: "chunk_committed",
    job_id: jobId,
    chunk_index: chunkIndex,
    delivery_id: `delivery-${jobId}-${chunkIndex}`,
  });
}

async function presentFixedChunk(client, chunkIndex) {
  for (let index = 0; index < fixedChunkFrameCount(chunkIndex); index += 1) {
    await advancePaint(client);
    await advancePaint(client);
  }
}

async function deliverFixedClip(
  client,
  {
    frameLabel = (frameIndex) => `F${frameIndex}`,
    jobId = "job-a",
    payloadByteLength = null,
  } = {},
) {
  let frameIndex = 0;
  for (let chunkIndex = 0; chunkIndex < 21; chunkIndex += 1) {
    const labels = [];
    for (let index = 0; index < fixedChunkFrameCount(chunkIndex); index += 1) {
      labels.push(frameLabel(frameIndex));
      frameIndex += 1;
    }
    sendFixedChunk(client, {
      chunkIndex,
      frameLabels: labels,
      jobId,
      payloadByteLength,
    });
    await flushMicrotasks();
    await presentFixedChunk(client, chunkIndex);
  }
  assert.equal(frameIndex, 81);
}

test("enhancement rewrites the one editable prompt without starting generation", () => {
  const client = bootClient();
  client.prompt.value = "a bouncing ball";
  client.seed.value = "20260719";
  client.enhancePrompt.dispatch("click");

  assert.deepEqual(client.socket.sent, [{
    type: "resolve_prompt",
    request_id: "prompt-resolution-1",
    prompt: "a bouncing ball",
  }]);

  text(client.socket, {
    type: "prompt_resolved",
    request_id: "prompt-resolution-1",
    effective_prompt: "one ball visibly completes three bounce cycles",
    raw_prompt_sha256: "a".repeat(64),
    effective_prompt_sha256: "b".repeat(64),
    prompt_transform_id: "gemini-3.1-flash-lite-temporal-v1",
    prompt_changed: true,
    prompt_resolution_ms: 1289.3,
  });

  assert.equal(client.prompt.value, "one ball visibly completes three bounce cycles");
  assert.equal(client.socket.sent.length, 1);
  assert.equal(client.body.dataset.streamState, "prompt-enhanced");

  client.form.dispatch("submit", { preventDefault() {} });

  assert.deepEqual(client.socket.sent[1], {
    type: "start",
    prompt: "one ball visibly completes three bounce cycles",
    prompt_resolution_id: "prompt-resolution-1",
    seed: 20260719,
  });
});

test("generate sends the current prompt directly when enhancement was not used", () => {
  const client = bootClient();
  client.prompt.value = "my own production-ready prompt";
  client.seed.value = "11";
  client.form.dispatch("submit", { preventDefault() {} });

  assert.deepEqual(client.socket.sent, [{
    type: "start",
    prompt: "my own production-ready prompt",
    seed: 11,
  }]);
});

test("editing an enhanced prompt invalidates provenance before generation", () => {
  const client = bootClient();
  client.prompt.value = "a bouncing ball";
  client.enhancePrompt.dispatch("click");
  text(client.socket, {
    type: "prompt_resolved",
    request_id: "prompt-resolution-1",
    effective_prompt: "one ball visibly completes three bounce cycles",
    raw_prompt_sha256: "a".repeat(64),
    effective_prompt_sha256: "b".repeat(64),
    prompt_transform_id: "gemini-3.1-flash-lite-temporal-v1",
    prompt_changed: true,
    prompt_resolution_ms: 1289.3,
  });

  client.prompt.value += " from a low camera angle";
  client.prompt.dispatch("input");
  client.form.dispatch("submit", { preventDefault() {} });

  assert.deepEqual(client.socket.sent[1], {
    type: "start",
    prompt: "one ball visibly completes three bounce cycles from a low camera angle",
    seed: 7,
  });
});

test("editing after generate cannot invalidate an in-flight start acceptance", () => {
  const client = bootClient();
  client.prompt.value = "a bouncing ball";
  client.enhancePrompt.dispatch("click");
  text(client.socket, {
    type: "prompt_resolved",
    request_id: "prompt-resolution-1",
    effective_prompt: "one ball visibly completes three bounce cycles",
    raw_prompt_sha256: "a".repeat(64),
    effective_prompt_sha256: "b".repeat(64),
    prompt_transform_id: "gemini-3.1-flash-lite-temporal-v1",
    prompt_changed: true,
    prompt_resolution_ms: 1289.3,
  });
  client.form.dispatch("submit", { preventDefault() {} });

  client.prompt.value = "the next prompt I want to try";
  client.prompt.dispatch("input");
  text(client.socket, {
    type: "start_accepted",
    job_id: "job-a",
    latent_frames: 21,
    expected_rgb_frames: 81,
    prompt_resolution_id: "prompt-resolution-1",
    raw_prompt_sha256: "a".repeat(64),
    effective_prompt_sha256: "b".repeat(64),
    prompt_transform_id: "gemini-3.1-flash-lite-temporal-v1",
  });

  assert.equal(client.body.dataset.streamState, "streaming");
  assert.deepEqual(client.socket.closed, []);
});

test("double generate consumes enhanced provenance only once", () => {
  const client = bootClient();
  client.prompt.value = "a bouncing ball";
  client.enhancePrompt.dispatch("click");
  text(client.socket, {
    type: "prompt_resolved",
    request_id: "prompt-resolution-1",
    effective_prompt: "one ball visibly completes three bounce cycles",
    raw_prompt_sha256: "a".repeat(64),
    effective_prompt_sha256: "b".repeat(64),
    prompt_transform_id: "gemini-3.1-flash-lite-temporal-v1",
    prompt_changed: true,
    prompt_resolution_ms: 1289.3,
  });

  client.form.dispatch("submit", { preventDefault() {} });
  client.form.dispatch("submit", { preventDefault() {} });

  assert.equal(client.socket.sent.length, 2);
  assert.equal(client.socket.sent[1].type, "start");
  assert.equal(client.body.dataset.streamState, "starting");
});

test("a second generate after acceptance cannot replace the active job", () => {
  const client = bootClient();
  client.prompt.value = "my direct prompt";
  client.form.dispatch("submit", { preventDefault() {} });
  acceptJob(client);

  client.form.dispatch("submit", { preventDefault() {} });

  assert.equal(client.socket.sent.length, 1);
  assert.equal(client.body.dataset.streamState, "streaming");
});

test("editing during enhancement fences a late response from the textarea", () => {
  const client = bootClient();
  client.prompt.value = "a bouncing ball";
  client.enhancePrompt.dispatch("click");
  client.prompt.value = "my manually revised prompt";
  client.prompt.dispatch("input");

  text(client.socket, {
    type: "prompt_resolved",
    request_id: "prompt-resolution-1",
    effective_prompt: "stale enhanced prompt",
    raw_prompt_sha256: "a".repeat(64),
    effective_prompt_sha256: "b".repeat(64),
    prompt_transform_id: "gemini-3.1-flash-lite-temporal-v1",
    prompt_changed: true,
    prompt_resolution_ms: 1289.3,
  });

  assert.equal(client.prompt.value, "my manually revised prompt");
  assert.equal(client.socket.sent.length, 1);
  assert.equal(client.body.dataset.streamState, "ready");
});

test("generating during enhancement fences the late enhancement response", () => {
  const client = bootClient();
  client.prompt.value = "my prompt exactly as written";
  client.enhancePrompt.dispatch("click");
  client.form.dispatch("submit", { preventDefault() {} });

  text(client.socket, {
    type: "prompt_resolved",
    request_id: "prompt-resolution-1",
    effective_prompt: "stale enhanced prompt",
    raw_prompt_sha256: "a".repeat(64),
    effective_prompt_sha256: "b".repeat(64),
    prompt_transform_id: "gemini-3.1-flash-lite-temporal-v1",
    prompt_changed: true,
    prompt_resolution_ms: 1289.3,
  });

  assert.equal(client.prompt.value, "my prompt exactly as written");
  assert.deepEqual(client.socket.sent[1], {
    type: "start",
    prompt: "my prompt exactly as written",
    seed: 7,
  });
  assert.equal(client.socket.sent.length, 2);
  assert.equal(client.body.dataset.streamState, "starting");
});

test("a repeated enhance click does not duplicate provider work", () => {
  const client = bootClient();
  client.prompt.value = "a bouncing ball";

  client.enhancePrompt.dispatch("click");
  client.enhancePrompt.dispatch("click");

  assert.equal(client.socket.sent.length, 1);
  assert.equal(client.socket.sent[0].type, "resolve_prompt");
  assert.equal(client.body.dataset.streamState, "resolving-prompt");
});

test("invalid seeds are rejected before a start command", () => {
  for (const value of ["", "-1", "1.5", "4294967296", "not-a-number"]) {
    const client = bootClient();
    client.seed.value = value;
    client.form.dispatch("submit", { preventDefault() {} });
    assert.deepEqual(client.socket.sent, [], `seed ${JSON.stringify(value)}`);
    assert.equal(client.body.dataset.streamState, "input-error");
  }
});

test("the page offers one editable prompt with optional Gemini enhancement", () => {
  assert.match(htmlSource, /optionally enhance it with Gemini 3\.1 Flash-Lite/);
  assert.match(htmlSource, /repeatedly falls straight down under gravity/);
  assert.match(htmlSource, /completing three distinct bounce cycles/);
  assert.match(htmlSource, /id="seed"/);
  assert.match(htmlSource, /id="enhance-prompt"/);
  assert.match(htmlSource, /id="status" aria-live="polite"/);
  assert.doesNotMatch(htmlSource, /id="exact-prompt"/);
  assert.doesNotMatch(htmlSource, /id="effective-prompt"/);
});

test("a stale commit after replacement is ignored", () => {
  const client = bootClient();
  acceptJob(client, "job-b");
  text(client.socket, {
    type: "chunk_committed",
    job_id: "job-a",
    chunk_index: 0,
    delivery_id: "delivery-a",
  });

  assert.deepEqual(client.socket.closed, []);
  assert.equal(client.body.dataset.streamState, "streaming");
});

test("pre-completion disconnect fences an in-flight committed paint", async () => {
  const client = bootClient();
  acceptJob(client);
  sendFixedChunk(client, {
    chunkIndex: 0,
    frameLabels: ["A"],
  });
  await flushMicrotasks();

  client.socket.dispatch("close", {});
  assert.equal(client.body.dataset.streamState, "disconnected");
  await advancePaint(client);
  assert.deepEqual(client.drawing.frames, []);
  assert.equal(client.animationFrames.length, 0);
  assert.deepEqual(client.socket.sent, []);
});

test("socket close cannot overwrite a specific terminal protocol error", () => {
  const client = bootClient();

  client.socket.dispatch("message", { data: "{" });
  const protocolStatus = client.status.value;

  assert.equal(client.body.dataset.streamState, "protocol-error");
  assert.match(protocolStatus, /server sent invalid JSON/);

  client.socket.dispatch("close", {});

  assert.equal(client.body.dataset.streamState, "protocol-error");
  assert.equal(client.status.value, protocolStatus);
});

test("job cancellation fences an in-flight committed paint", async () => {
  const client = bootClient();
  acceptJob(client);
  sendFixedChunk(client, {
    chunkIndex: 0,
    frameLabels: ["A"],
  });
  await flushMicrotasks();
  text(client.socket, {
    type: "stream_event",
    kind: "job_cancelled",
    job_id: "job-a",
  });

  await advancePaint(client);
  assert.equal(client.body.dataset.streamState, "cancelled");
  assert.deepEqual(client.drawing.frames, []);
  assert.equal(client.animationFrames.length, 0);
  assert.deepEqual(client.socket.sent, []);
});

test("invalid expected frame topology fails closed", () => {
  const client = bootClient();
  text(client.socket, {
    type: "start_accepted",
    job_id: "job-a",
    latent_frames: 21,
    expected_rgb_frames: 0,
  });

  assert.equal(client.body.dataset.streamState, "protocol-error");
  assert.deepEqual(client.socket.closed, [{
    code: 1002,
    reason: "stream protocol error",
  }]);
});

test("a non-fixed accepted topology fails closed", () => {
  for (const message of [
    {
      type: "start_accepted",
      job_id: "job-a",
      latent_frames: 20,
      expected_rgb_frames: 81,
    },
    {
      type: "start_accepted",
      job_id: "job-a",
      latent_frames: 21,
      expected_rgb_frames: 80,
    },
  ]) {
    const client = bootClient();
    text(client.socket, message);
    assert.equal(client.body.dataset.streamState, "protocol-error");
  }
});

test("job completion before all fixed chunks fails closed", () => {
  const client = bootClient();
  text(client.socket, {
    type: "start_accepted",
    job_id: "job-a",
    latent_frames: 21,
    expected_rgb_frames: 81,
  });
  text(client.socket, {
    type: "stream_event",
    kind: "job_completed",
    job_id: "job-a",
  });
  assert.equal(client.body.dataset.streamState, "protocol-error");
});

test("a malformed fixed chunk fails before paint", () => {
  const client = bootClient();
  text(client.socket, {
    type: "start_accepted",
    job_id: "job-a",
    latent_frames: 21,
    expected_rgb_frames: 81,
  });
  text(client.socket, {
    type: "chunk",
    kind: "chunk_ready",
    job_id: "job-a",
    chunk_index: 1,
    first_frame_index: 1,
    frame_count: 1,
    frame_media_type: "image/png",
    renderable: true,
  });
  assert.equal(client.body.dataset.streamState, "protocol-error");
  assert.equal(client.drawing.count, 0);
});

test("replacement clears an uncommitted predecessor binary group", async () => {
  const client = bootClient();
  acceptJob(client);
  text(client.socket, {
    type: "chunk",
    kind: "chunk_ready",
    job_id: "job-a",
    chunk_index: 0,
    first_frame_index: 0,
    frame_count: 1,
    frame_media_type: "image/png",
    renderable: true,
  });
  client.socket.dispatch("message", { data: new ArrayBuffer(4) });

  acceptJob(client, "job-b");
  sendFixedChunk(client, {
    chunkIndex: 0,
    frameLabels: ["B"],
    jobId: "job-b",
  });
  await flushMicrotasks();

  assert.deepEqual(client.socket.closed, []);
  assert.equal(client.animationFrames.length, 1);
  client.animationFrames.shift()();
  await flushMicrotasks();
  client.animationFrames.shift()();
  await flushMicrotasks();
  assert.equal(client.socket.sent.length, 1);
  assert.equal(client.socket.sent[0].type, "presented");
  assert.equal(client.socket.sent[0].job_id, "job-b");
});

test("a current-job commit without metadata is still fatal", () => {
  const client = bootClient();
  acceptJob(client);
  text(client.socket, {
    type: "chunk_committed",
    job_id: "job-a",
    chunk_index: 0,
    delivery_id: "delivery-a",
  });

  assert.equal(client.socket.closed.length, 1);
  assert.equal(client.socket.closed[0].code, 1002);
  assert.equal(client.body.dataset.streamState, "protocol-error");
});

test("an oversized binary chunk fails before browser decode", () => {
  const client = bootClient();
  acceptJob(client);
  text(client.socket, {
    type: "chunk",
    kind: "chunk_ready",
    job_id: "job-a",
    chunk_index: 0,
    first_frame_index: 0,
    frame_count: 1,
    frame_media_type: "image/png",
    renderable: true,
  });
  const oversized = new ArrayBuffer(1);
  Object.defineProperty(oversized, "byteLength", {
    value: 16 * 1024 * 1024 + 1,
  });
  client.socket.dispatch("message", { data: oversized });

  assert.equal(client.body.dataset.streamState, "protocol-error");
  assert.equal(client.drawing.count, 0);
  assert.deepEqual(client.socket.closed, [{
    code: 1002,
    reason: "stream protocol error",
  }]);
});

test("presentation ACK occurs only after draw and a presentation opportunity", async () => {
  const client = bootClient();
  acceptJob(client);
  sendFixedChunk(client, {
    chunkIndex: 0,
    frameLabels: ["A"],
  });
  await flushMicrotasks();

  assert.equal(client.socket.sent.length, 0);
  assert.equal(client.animationFrames.length, 1);
  client.animationFrames.shift()();
  await flushMicrotasks();
  assert.equal(client.drawing.count, 1);
  assert.equal(client.socket.sent.length, 0);
  assert.equal(client.animationFrames.length, 1);

  client.animationFrames.shift()();
  await flushMicrotasks();
  assert.equal(client.socket.sent.length, 1);
  assert.equal(client.socket.sent[0].type, "presented");
  assert.equal(client.body.dataset.renderedFrames, "1");
  assert.equal(client.body.dataset.ackCount, "1");
});

test("committed chunks paint serially and ACK in order", async () => {
  const client = bootClient();
  acceptJob(client);
  sendFixedChunk(client, {
    chunkIndex: 0,
    frameLabels: ["A"],
  });
  sendFixedChunk(client, {
    chunkIndex: 1,
    frameLabels: ["B0", "B1", "B2", "B3"],
  });
  await flushMicrotasks();

  assert.equal(client.animationFrames.length, 1);
  await advancePaint(client);
  assert.deepEqual(client.drawing.frames, ["A"]);
  assert.deepEqual(client.socket.sent, []);
  await advancePaint(client);
  assert.deepEqual(
    client.socket.sent.map((message) => message.delivery_id),
    ["delivery-job-a-0"],
  );
  assert.equal(client.animationFrames.length, 1);
  await advancePaint(client);
  assert.deepEqual(client.drawing.frames, ["A", "B0"]);
  assert.deepEqual(
    client.socket.sent.map((message) => message.delivery_id),
    ["delivery-job-a-0"],
  );
  await advancePaint(client);
  for (const expected of ["B1", "B2", "B3"]) {
    await advancePaint(client);
    assert.equal(client.drawing.frames.at(-1), expected);
    await advancePaint(client);
  }
  assert.deepEqual(
    client.socket.sent.map((message) => message.delivery_id),
    ["delivery-job-a-0", "delivery-job-a-1"],
  );
});

test("a completed clip visibly loops without duplicate ACKs", async () => {
  const client = bootClient();
  acceptJob(client);
  await deliverFixedClip(client, {
    frameLabel(frameIndex) {
      if (frameIndex === 0) return "A";
      if (frameIndex === 1) return "B";
      return `F${frameIndex}`;
    },
  });
  text(client.socket, {
    type: "stream_event",
    kind: "job_completed",
    job_id: "job-a",
  });
  await flushMicrotasks();

  assert.equal(client.drawing.frames.length, 81);
  assert.deepEqual(client.drawing.frames.slice(0, 2), ["A", "B"]);
  assert.equal(client.socket.sent.at(-1).type, "presented");
  assert.equal(client.animationFrames.length, 1);

  await advancePaint(client);
  assert.equal(client.drawing.frames.at(-1), "A");
  assert.equal(client.body.dataset.replayState, "replaying");
  assert.equal(
    client.socket.sent.filter((message) => message.type === "presented").length,
    21,
  );

  client.socket.dispatch("close", {});
  assert.equal(client.body.dataset.streamState, "complete");
  assert.equal(client.body.dataset.replayState, "replaying");
  await advancePaint(client);
  await advancePaint(client);
  assert.deepEqual(client.drawing.frames.slice(-2), ["A", "B"]);

  acceptJob(client, "job-b");
  assert.equal(client.body.dataset.replayState, "idle");
  await advancePaint(client);
  assert.deepEqual(client.drawing.frames.slice(-2), ["A", "B"]);
  assert.equal(client.animationFrames.length, 0);
});

test("a replay decode failure preserves successful completion evidence", async () => {
  let bitmapCalls = 0;
  const client = bootClient({
    async bitmapFactory(blob) {
      bitmapCalls += 1;
      if (bitmapCalls > 81) {
        throw new Error("decode boom");
      }
      return { close() {}, tag: await blob.text() };
    },
  });
  acceptJob(client);
  await deliverFixedClip(client);
  text(client.socket, {
    type: "stream_event",
    kind: "job_completed",
    job_id: "job-a",
  });
  await flushMicrotasks();

  assert.equal(client.body.dataset.streamState, "complete");
  assert.equal(client.body.dataset.replayState, "unavailable");
  assert.match(client.status.value, /Replay unavailable/);
  assert.equal(client.body.dataset.renderedFrames, "81");
  assert.equal(client.body.dataset.ackCount, "21");
  assert.deepEqual(
    client.socket.sent.map((message) => message.type),
    Array(21).fill("presented"),
  );
});

test("the replay byte budget degrades replay without erasing completion", async () => {
  const client = bootClient();
  acceptJob(client);
  await deliverFixedClip(client, { payloadByteLength: 1024 * 1024 });
  text(client.socket, {
    type: "stream_event",
    kind: "job_completed",
    job_id: "job-a",
  });
  await flushMicrotasks();

  assert.equal(client.body.dataset.streamState, "complete");
  assert.equal(client.body.dataset.replayState, "unavailable");
  assert.match(client.status.value, /64 MiB replay budget/);
  assert.equal(client.body.dataset.renderedFrames, "81");
  assert.equal(client.body.dataset.renderedChunks, "21");
  assert.equal(client.body.dataset.ackCount, "21");
  assert.equal(client.animationFrames.length, 0);
  assert.deepEqual(
    client.socket.sent.map((message) => message.type),
    Array(21).fill("presented"),
  );
});

test("a replaced committed frame cannot paint over the new job", async () => {
  const client = bootClient();
  acceptJob(client);
  sendFixedChunk(client, {
    chunkIndex: 0,
    frameLabels: ["A"],
  });
  await flushMicrotasks();

  acceptJob(client, "job-b");
  sendFixedChunk(client, {
    chunkIndex: 0,
    frameLabels: ["B"],
    jobId: "job-b",
  });
  await flushMicrotasks();

  await advancePaint(client);
  assert.deepEqual(client.drawing.frames, []);
  await advancePaint(client);
  assert.deepEqual(client.drawing.frames, ["B"]);
  await advancePaint(client);
  assert.deepEqual(
    client.socket.sent.filter((message) => message.type === "presented"),
    [{
      type: "presented",
      job_id: "job-b",
      chunk_index: 0,
      delivery_id: "delivery-job-b-0",
      client_presented_ns: 0,
    }],
  );
});
