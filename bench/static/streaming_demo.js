"use strict";

const protocol = "realtime-video.websocket.v1";
const streamFrameIntervalMs = 1000 / 24;
const replayFrameIntervalMs = 1000 / 16;
const maxChunkBytes = 16 * 1024 * 1024;
const maxReplayBytes = 64 * 1024 * 1024;
const fixedLatentFrames = 21;
const fixedExpectedFrames = 81;
const status = document.querySelector("#status");
const form = document.querySelector("#prompt-form");
const promptInput = document.querySelector("#prompt");
const seedInput = document.querySelector("#seed");
const enhancePromptButton = document.querySelector("#enhance-prompt");
const canvas = document.querySelector("#frame");
const context = canvas.getContext("2d");
const wsScheme = location.protocol === "https:" ? "wss" : "ws";
const socket = new WebSocket(`${wsScheme}://${location.host}/ws`, protocol);
socket.binaryType = "arraybuffer";

let currentJob = null;
let pendingChunk = null;
let presentationChain = Promise.resolve();
let expectedFrames = fixedExpectedFrames;
let receivedChunks = 0;
let receivedFrames = 0;
let renderedFrames = 0;
let renderedChunks = 0;
let ackCount = 0;
let serverCompleted = false;
let promptResolutionCounter = 0;
let pendingPromptResolution = null;
let availablePromptProvenance = null;
let awaitingStartPromptProvenance = null;
let startPending = false;
let generationBusy = false;
let terminalState = null;
let streamPacing = { nextFrameDeadlineMs: null };
let replayFrames = [];
let replayBytes = 0;
let replayEpoch = 0;
let replayJob = null;
let replayUnavailableReason = null;
document.body.dataset.replayState = "idle";

function publishCounters() {
  document.body.dataset.renderedFrames = String(renderedFrames);
  document.body.dataset.renderedChunks = String(renderedChunks);
  document.body.dataset.ackCount = String(ackCount);
  document.body.dataset.expectedFrames = String(expectedFrames);
  document.body.dataset.serverCompleted = String(serverCompleted);
}

function setStatus(message, state) {
  status.value = message;
  document.body.dataset.streamState = state;
  publishCounters();
}

function streamingStatus(prefix = "Streaming") {
  return `${prefix}: ${renderedFrames}/${expectedFrames} frames, `
    + `${renderedChunks} chunks, ${ackCount} ACKs.`;
}

function announceBusyGeneration() {
  if (startPending) {
    setStatus("A generation request is already awaiting acceptance.", "starting");
  } else {
    setStatus("A generation is already in progress.", "streaming");
  }
}

function resetJob(message) {
  const provenanceFields = [
    "prompt_resolution_id",
    "raw_prompt_sha256",
    "effective_prompt_sha256",
    "prompt_transform_id",
  ];
  if (
    typeof message.job_id !== "string"
    || message.job_id.length === 0
    || message.latent_frames !== fixedLatentFrames
    || message.expected_rgb_frames !== fixedExpectedFrames
    || (
      awaitingStartPromptProvenance === null
      && provenanceFields.some((field) => (
        Object.prototype.hasOwnProperty.call(message, field)
      ))
    )
    || (
      awaitingStartPromptProvenance !== null
      && (
        message.prompt_resolution_id
          !== awaitingStartPromptProvenance.requestId
        || message.raw_prompt_sha256
          !== awaitingStartPromptProvenance.rawPromptSha256
        || message.effective_prompt_sha256
          !== awaitingStartPromptProvenance.effectivePromptSha256
        || message.prompt_transform_id
          !== awaitingStartPromptProvenance.promptTransformId
      )
    )
  ) {
    protocolFailure("start acceptance declared an invalid job topology");
    return;
  }
  startPending = false;
  awaitingStartPromptProvenance = null;
  invalidateReplay();
  currentJob = message.job_id;
  pendingChunk = null;
  expectedFrames = message.expected_rgb_frames;
  receivedChunks = 0;
  receivedFrames = 0;
  renderedFrames = 0;
  renderedChunks = 0;
  ackCount = 0;
  serverCompleted = false;
  terminalState = null;
  streamPacing = { nextFrameDeadlineMs: null };
  presentationChain = Promise.resolve();
  setStatus(streamingStatus(), "streaming");
}

function invalidateReplay() {
  replayEpoch += 1;
  replayFrames = [];
  replayBytes = 0;
  replayJob = null;
  replayUnavailableReason = null;
  document.body.dataset.replayState = "idle";
}

function disableReplay(reason) {
  replayEpoch += 1;
  replayFrames = [];
  replayBytes = 0;
  replayJob = null;
  replayUnavailableReason = reason;
  document.body.dataset.replayState = "unavailable";
}

function maybeAnnounceComplete() {
  if (terminalState !== null || !serverCompleted) {
    return;
  }
  if (renderedFrames === expectedFrames) {
    generationBusy = false;
    const replayDetail = replayUnavailableReason === null
      ? ""
      : ` Replay unavailable: ${replayUnavailableReason}.`;
    setStatus(
      `Complete: ${renderedFrames} frames painted in ${renderedChunks} chunks; `
        + `${ackCount} presentation ACKs sent.${replayDetail}`,
      "complete",
    );
  } else {
    setStatus(streamingStatus("Server complete; still presenting"), "presenting");
  }
}

function protocolFailure(message) {
  invalidateReplay();
  pendingPromptResolution = null;
  availablePromptProvenance = null;
  awaitingStartPromptProvenance = null;
  startPending = false;
  generationBusy = false;
  terminalState = "protocol-error";
  currentJob = null;
  pendingChunk = null;
  setStatus(`Protocol error: ${message}`, "protocol-error");
  if (socket.readyState === WebSocket.OPEN) {
    try {
      socket.close(1002, "stream protocol error");
    } catch (_error) {
      // State is already terminal; a close race must not escape the handler.
    }
  }
}

function presentationFailure(error, jobId) {
  if (jobId !== currentJob || terminalState !== null) {
    return;
  }
  const detail = error instanceof Error ? error.message : String(error);
  invalidateReplay();
  terminalState = "presentation-error";
  generationBusy = false;
  pendingChunk = null;
  setStatus(`Presentation failed: ${detail}`, "presentation-error");
  if (socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(JSON.stringify({
        type: "cancel",
        job_id: jobId,
      }));
    } catch (_error) {
      // The WebSocket may close between readyState and send().
    }
  }
  currentJob = null;
}

function connectionFailure(message, state) {
  if (
    document.body.dataset.streamState === "complete"
    || terminalState !== null
  ) {
    return;
  }
  invalidateReplay();
  pendingPromptResolution = null;
  availablePromptProvenance = null;
  awaitingStartPromptProvenance = null;
  startPending = false;
  generationBusy = false;
  terminalState = state;
  currentJob = null;
  pendingChunk = null;
  setStatus(message, state);
}

function nextPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForFrameSlot(pacing, frameIntervalMs, paintAllowed) {
  if (!paintAllowed()) {
    return false;
  }
  const now = performance.now();
  if (pacing.nextFrameDeadlineMs === null) {
    pacing.nextFrameDeadlineMs = now;
  }
  const remaining = pacing.nextFrameDeadlineMs - now;
  if (remaining > 0) {
    await delay(remaining);
  }
  if (!paintAllowed()) {
    return false;
  }
  pacing.nextFrameDeadlineMs = Math.max(
    pacing.nextFrameDeadlineMs + frameIntervalMs,
    performance.now(),
  );
  return true;
}

async function paintFrame(payload, frameMediaType, frameIntervalMs, pacing, paintAllowed) {
  if (!await waitForFrameSlot(pacing, frameIntervalMs, paintAllowed)) {
    return false;
  }
  const image = await createImageBitmap(new Blob([payload], { type: frameMediaType }));
  try {
    const painted = await new Promise((resolve, reject) => requestAnimationFrame(() => {
      if (!paintAllowed()) {
        resolve(false);
        return;
      }
      try {
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(true);
      } catch (error) {
        reject(error);
      }
    }));
    if (!painted) {
      return false;
    }
    // The following animation frame is the browser presentation opportunity;
    // frame counts and ACKs advance only after it has occurred.
    await nextPaint();
    return paintAllowed();
  } finally {
    image.close();
  }
}

async function replayClip(jobId, epoch) {
  const pacing = { nextFrameDeadlineMs: null };
  const stillCurrent = () => (
    epoch === replayEpoch
    && currentJob === jobId
    && terminalState === null
    && serverCompleted
  );
  while (stillCurrent()) {
    for (const frame of replayFrames) {
      if (!stillCurrent()) {
        return;
      }
      const painted = await paintFrame(
        frame.payload,
        frame.frameMediaType,
        replayFrameIntervalMs,
        pacing,
        stillCurrent,
      );
      if (!painted) {
        return;
      }
    }
  }
}

function replayFailure(error, jobId, epoch) {
  if (
    epoch !== replayEpoch
    || jobId !== currentJob
    || replayJob !== jobId
    || terminalState !== null
  ) {
    return;
  }
  const detail = error instanceof Error ? error.message : String(error);
  disableReplay(`browser decode failed (${detail})`);
  maybeAnnounceComplete();
}

function maybeStartReplay() {
  if (
    terminalState !== null
    || !serverCompleted
    || currentJob === null
    || renderedFrames !== expectedFrames
    || replayFrames.length !== expectedFrames
    || replayJob === currentJob
  ) {
    return;
  }
  const jobId = currentJob;
  const epoch = replayEpoch;
  replayJob = jobId;
  document.body.dataset.replayState = "replaying";
  void replayClip(jobId, epoch).catch((error) => {
    replayFailure(error, jobId, epoch);
  });
}

function retainReplayFrame(payload, frameMediaType) {
  if (replayUnavailableReason !== null) {
    return;
  }
  const byteCount = payload.byteLength;
  if (
    !Number.isSafeInteger(byteCount)
    || byteCount < 0
    || replayBytes + byteCount > maxReplayBytes
  ) {
    disableReplay("clip exceeds the 64 MiB replay budget");
    return;
  }
  replayFrames.push({ payload, frameMediaType });
  replayBytes += byteCount;
}

async function presentChunk(chunk) {
  if (currentJob !== chunk.job_id || terminalState !== null) {
    return;
  }
  if (!chunk.renderable) {
    throw new Error(`backend declared ${chunk.frame_media_type} non-renderable`);
  }
  const stillCurrent = () => (
    currentJob === chunk.job_id && terminalState === null
  );
  for (const payload of chunk.payloads) {
    if (!stillCurrent()) {
      return;
    }
    const painted = await paintFrame(
      payload,
      chunk.frame_media_type,
      streamFrameIntervalMs,
      streamPacing,
      stillCurrent,
    );
    if (!painted || !stillCurrent()) {
      return;
    }
    retainReplayFrame(payload, chunk.frame_media_type);
    renderedFrames += 1;
    if (renderedFrames > expectedFrames) {
      throw new Error(`received more than ${expectedFrames} frames`);
    }
    setStatus(streamingStatus(), "streaming");
    maybeAnnounceComplete();
  }

  renderedChunks += 1;
  if (currentJob !== chunk.job_id || socket.readyState !== WebSocket.OPEN) {
    throw new Error("socket closed before the presentation ACK");
  }
  socket.send(JSON.stringify({
    type: "presented",
    job_id: chunk.job_id,
    chunk_index: chunk.chunk_index,
    delivery_id: chunk.delivery_id,
    client_presented_ns: Math.round(performance.now() * 1_000_000),
  }));
  ackCount += 1;
  setStatus(streamingStatus(), "streaming");
  maybeAnnounceComplete();
  maybeStartReplay();
}

function queuePresentation(chunk) {
  presentationChain = presentationChain.then(() => presentChunk(chunk));
  presentationChain = presentationChain.catch((error) => {
    presentationFailure(error, chunk.job_id);
  });
}

function handleTextMessage(raw) {
  let message;
  try {
    message = JSON.parse(raw);
  } catch (_error) {
    protocolFailure("server sent invalid JSON");
    return;
  }

  if (message.type === "prompt_resolved") {
    if (
      pendingPromptResolution === null
      || message.request_id !== pendingPromptResolution.requestId
    ) {
      return;
    }
    if (promptInput.value !== pendingPromptResolution.rawPrompt) {
      pendingPromptResolution = null;
      availablePromptProvenance = null;
      if (!generationBusy) {
        setStatus("Prompt edited. Generate it directly or enhance again.", "ready");
      }
      return;
    }
    const sha256Pattern = /^[0-9a-f]{64}$/;
    if (
      typeof message.effective_prompt !== "string"
      || message.effective_prompt.trim() === ""
      || message.effective_prompt.length > 4096
      || !sha256Pattern.test(message.raw_prompt_sha256)
      || !sha256Pattern.test(message.effective_prompt_sha256)
      || typeof message.prompt_transform_id !== "string"
      || message.prompt_transform_id.length === 0
      || message.prompt_transform_id.length > 128
      || typeof message.prompt_changed !== "boolean"
      || message.prompt_changed
        !== (message.effective_prompt !== pendingPromptResolution.rawPrompt)
      || typeof message.prompt_resolution_ms !== "number"
      || !Number.isFinite(message.prompt_resolution_ms)
      || message.prompt_resolution_ms < 0
    ) {
      protocolFailure("server sent invalid prompt resolution provenance");
      return;
    }
    pendingPromptResolution = null;
    availablePromptProvenance = {
      requestId: message.request_id,
      rawPromptSha256: message.raw_prompt_sha256,
      effectivePromptSha256: message.effective_prompt_sha256,
      promptTransformId: message.prompt_transform_id,
      effectivePrompt: message.effective_prompt,
    };
    promptInput.value = message.effective_prompt;
    const promptStatus = message.prompt_changed
      ? `Prompt enhanced in ${Math.round(message.prompt_resolution_ms)} ms. `
        + "Review or edit it, then generate."
      : "Prompt is already detailed. Review it or generate as written.";
    setStatus(promptStatus, "prompt-enhanced");
  } else if (message.type === "prompt_resolution_failed") {
    if (
      pendingPromptResolution === null
      || message.request_id !== pendingPromptResolution.requestId
    ) {
      return;
    }
    if (message.error_code !== "prompt_resolution_failed") {
      protocolFailure("server sent invalid prompt resolution failure");
      return;
    }
    pendingPromptResolution = null;
    availablePromptProvenance = null;
    setStatus(
      "Prompt enhancement failed. Retry or generate the current prompt as written.",
      "prompt-resolution-error",
    );
  } else if (message.type === "start_accepted") {
    resetJob(message);
  } else if (message.type === "chunk") {
    if (pendingChunk !== null) {
      protocolFailure("chunk metadata arrived before the prior chunk committed");
      return;
    }
    if (message.job_id !== currentJob) {
      return;
    }
    if (Object.prototype.hasOwnProperty.call(message, "delivery_id")) {
      protocolFailure("delivery token was disclosed before binary commit");
      return;
    }
    const expectedFrameCount = receivedChunks === 0 ? 1 : 4;
    if (
      receivedChunks >= fixedLatentFrames
      || message.chunk_index !== receivedChunks
      || message.first_frame_index !== receivedFrames
      || message.frame_count !== expectedFrameCount
    ) {
      protocolFailure("chunk violated the fixed frame topology");
      return;
    }
    pendingChunk = { ...message, payloadBytes: 0, payloads: [] };
  } else if (message.type === "chunk_committed") {
    if (pendingChunk === null) {
      if (message.job_id !== currentJob) {
        return;
      }
      protocolFailure("chunk commit arrived without metadata");
      return;
    }
    if (pendingChunk.job_id !== currentJob) {
      pendingChunk = null;
      return;
    }
    if (
      pendingChunk.job_id !== message.job_id
      || pendingChunk.chunk_index !== message.chunk_index
      || pendingChunk.payloads.length !== pendingChunk.frame_count
    ) {
      protocolFailure("chunk commit did not match its complete binary group");
      return;
    }
    const completeChunk = {
      ...pendingChunk,
      delivery_id: message.delivery_id,
    };
    receivedChunks += 1;
    receivedFrames += completeChunk.frame_count;
    pendingChunk = null;
    queuePresentation(completeChunk);
  } else if (message.type === "stream_event" && message.kind === "job_completed") {
    if (message.job_id === currentJob && terminalState === null) {
      if (
        pendingChunk !== null
        || receivedChunks !== fixedLatentFrames
        || receivedFrames !== fixedExpectedFrames
      ) {
        protocolFailure("job completed before the fixed frame topology");
        return;
      }
      serverCompleted = true;
      maybeAnnounceComplete();
      maybeStartReplay();
    }
  } else if (message.type === "stream_event" && message.kind === "job_failed") {
    if (message.job_id === currentJob) {
      invalidateReplay();
      terminalState = "failed";
      generationBusy = false;
      pendingChunk = null;
      currentJob = null;
      setStatus(`Generation failed: ${message.error_code}.`, "failed");
    }
  } else if (message.type === "stream_event" && message.kind === "job_cancelled") {
    if (message.job_id === currentJob) {
      invalidateReplay();
      terminalState = "cancelled";
      generationBusy = false;
      pendingChunk = null;
      currentJob = null;
      setStatus("Generation cancelled.", "cancelled");
    }
  } else if (message.type === "command_error") {
    pendingPromptResolution = null;
    availablePromptProvenance = null;
    awaitingStartPromptProvenance = null;
    startPending = false;
    generationBusy = false;
    setStatus(`Command rejected: ${message.code}`, "command-error");
  }
}

socket.addEventListener("open", () => {
  setStatus("Connected. Submit a prompt to stream frames.", "ready");
});

socket.addEventListener("message", (event) => {
  if (typeof event.data === "string") {
    handleTextMessage(event.data);
    return;
  }

  if (pendingChunk === null) {
    protocolFailure("binary payload arrived without chunk metadata");
    return;
  }
  if (pendingChunk.payloads.length >= pendingChunk.frame_count) {
    protocolFailure("chunk carried too many binary payloads");
    return;
  }
  const byteLength = event.data?.byteLength;
  if (
    !Number.isSafeInteger(byteLength)
    || byteLength <= 0
    || pendingChunk.payloadBytes + byteLength > maxChunkBytes
  ) {
    protocolFailure("chunk payload exceeded the browser byte limit");
    return;
  }
  pendingChunk.payloadBytes += byteLength;
  pendingChunk.payloads.push(event.data);
});

socket.addEventListener("error", () => {
  connectionFailure(
    "WebSocket error. Reload the page for a fresh session.",
    "socket-error",
  );
});

socket.addEventListener("close", () => {
  connectionFailure(
    "Disconnected. Reload the page for a fresh session.",
    "disconnected",
  );
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (socket.readyState !== WebSocket.OPEN) {
    setStatus("WebSocket is not connected.", "disconnected");
    return;
  }
  if (generationBusy) {
    announceBusyGeneration();
    return;
  }
  const seed = Number(seedInput.value);
  if (
    seedInput.value.trim() === ""
    || !Number.isInteger(seed)
    || seed < 0
    || seed > 0xffffffff
  ) {
    setStatus("Seed must be an integer from 0 through 4294967295.", "input-error");
    return;
  }
  const prompt = promptInput.value;
  if (typeof prompt !== "string" || prompt.trim() === "") {
    setStatus("Prompt must not be empty.", "input-error");
    return;
  }
  pendingPromptResolution = null;
  const provenance = availablePromptProvenance;
  availablePromptProvenance = null;
  const start = {
    type: "start",
    prompt,
    seed,
  };
  if (provenance !== null && provenance.effectivePrompt === prompt) {
    start.prompt_resolution_id = provenance.requestId;
    awaitingStartPromptProvenance = provenance;
  } else {
    awaitingStartPromptProvenance = null;
  }
  socket.send(JSON.stringify(start));
  startPending = true;
  generationBusy = true;
  setStatus("Start requested; waiting for server acceptance…", "starting");
});

enhancePromptButton.addEventListener("click", () => {
  if (socket.readyState !== WebSocket.OPEN) {
    setStatus("WebSocket is not connected.", "disconnected");
    return;
  }
  if (generationBusy) {
    announceBusyGeneration();
    return;
  }
  if (pendingPromptResolution !== null) {
    setStatus("Prompt enhancement is already in progress.", "resolving-prompt");
    return;
  }
  const rawPrompt = promptInput.value;
  if (typeof rawPrompt !== "string" || rawPrompt.trim() === "") {
    setStatus("Prompt must not be empty.", "input-error");
    return;
  }
  promptResolutionCounter += 1;
  const requestId = `prompt-resolution-${promptResolutionCounter}`;
  pendingPromptResolution = { requestId, rawPrompt };
  availablePromptProvenance = null;
  socket.send(JSON.stringify({
    type: "resolve_prompt",
    request_id: requestId,
    prompt: rawPrompt,
  }));
  setStatus("Enhancing prompt for visible temporal motion…", "resolving-prompt");
});

promptInput.addEventListener("input", () => {
  const discardedEnhancement = (
    pendingPromptResolution !== null || availablePromptProvenance !== null
  );
  pendingPromptResolution = null;
  availablePromptProvenance = null;
  if (discardedEnhancement && !generationBusy) {
    setStatus("Prompt edited. Generate it directly or enhance again.", "ready");
  }
});

publishCounters();
