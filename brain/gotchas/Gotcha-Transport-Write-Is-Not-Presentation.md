---
type: gotcha
status: active
date: 2026-07-20
description: "A socket write may become peer-visible before its await returns, a job-fenced false send is not an I/O failure, and binary receipt is not browser presentation; use explicit registration, commit, and presentation phases."
anchors: ["bench/streaming_transport.py#_ClientSession", "bench/streaming_websocket.py#_WebSocketSession", "bench/static/streaming_demo.js"]
related: ["[[State]]", "[[Streaming-Service-Boundary]]", "[[Browser-Streaming-Transport]]", "[[Gotcha-Async-Timeouts-Need-Task-Isolation]]", "[[session-7-browser-streaming-transport]]"]
---

# Transport write is not presentation

Three different moments must not be collapsed into one metric or state flag:

1. local code queued/wrote bytes;
2. the peer received a complete delivery and may acknowledge it;
3. a browser decoded, drew, and reached a presentation opportunity.

`StreamWriter.write()` can make bytes peer-readable before `drain()` returns.
Likewise, an awaited WebSocket binary send can expose bytes before the sending
coroutine resumes. Registration must therefore happen at the first point where
the peer can legitimately act, and rollback must handle later drain/send
failure.

For a multi-message binary chunk, registration alone is not enough because the
client must not guess or reuse an ACK coordinate from header metadata. Send the
header and all binary frames, register an unpredictable delivery ID, then send
a separate commit containing that ID. The browser may ACK only after seeing the
commit and presenting the full group.

Fencing creates another distinction: a send helper returning `False` because
`expected_job_id` is stale means "correctly suppressed," not "socket failed."
Disconnecting on that result kills the replacement job. Only actual I/O,
timeout, or serialization exceptions justify disconnecting the connection.

Finally, replacement can retire an in-flight binary group. Do not emit its
commit token after retirement. A client should ignore a commit already known to
belong to an older job, while still treating a current-job commit without its
metadata/binary group as a fatal protocol violation.

The regressions live in both transport suites and in the executable Node client
test. They are the mechanical memory for these phase distinctions.
