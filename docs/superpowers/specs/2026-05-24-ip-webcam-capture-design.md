# IP Webcam Capture Source — Design

**Date:** 2026-05-24
**Status:** Approved (pending spec review)
**Relates to:** `CLAUDE.md` → "Long-term direction: streaming server for mobile client" (the `StreamCapture` drop-in seam)

## Goal

Let the pipeline ingest frames from an **Android IP Webcam** app over the local
network instead of (or alongside) the local USB/built-in webcam. This is the
first concrete step toward the documented end-state — a server that processes a
mobile client's video feed — but kept deliberately simple for prototyping: the
phone runs the off-the-shelf *IP Webcam* Android app, which exposes an MJPEG
stream on the LAN, and this machine pulls and processes it.

The new source must be a **drop-in replacement** for `WebcamCapture`
(`src/capture.py`). `CLAUDE.md` already names this seam: *"a future `StreamCapture`
(from a socket / WebRTC track) should be drop-in compatible — preserve `start()`
/ `read_latest(timeout)` / `stop()` semantics on any new capture source."*

## Decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Transport | **Manual MJPEG reader** over the `/video` endpoint via `requests` (stream=True), keep-latest-frame in a background thread. Chosen over `cv2.VideoCapture(url)` (which buffers internally and drifts behind real-time) and `/shot.jpg` polling (per-request overhead caps frame rate). |
| Source selection | **CLI flag** `--source {webcam,ip}` + `--url`, on both entry points, defaulting to the local webcam so current behavior is unchanged. |
| Module location | New package `src/camera/`. |
| Standalone verify | A `python -m camera` preview entry point to confirm the phone connection **before** wiring into the pipeline. |

## Architecture

New package `src/camera/`:

```
src/camera/
  __init__.py      # exports IPWebcamCapture, iter_jpeg_frames
  ip_webcam.py     # IPWebcamCapture class + iter_jpeg_frames (pure parser)
  __main__.py      # standalone preview: open stream, cv2.imshow, q/Esc to quit
```

### IP Webcam endpoints (off-the-shelf Android app)

- `http://<phone-ip>:8080/video` — **MJPEG stream** (`multipart/x-mixed-replace`). This is what we read.
- `http://<phone-ip>:8080/shot.jpg` — single JPEG (not used).

Resolution and quality are configured **in the phone app**, not by this code —
so unlike `WebcamCapture`, `IPWebcamCapture` takes no `width`/`height`.

### `iter_jpeg_frames` — the pure, testable core

```python
def iter_jpeg_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Extract every complete JPEG from `buffer`.

    Scans for JPEG SOI (0xFFD8) … EOI (0xFFD9) marker pairs and returns
    (complete_jpeg_byte_blobs, leftover_tail). The tail is the bytes after the
    last complete EOI (a partial next frame), to be prepended to the next chunk.
    """
```

Scanning SOI/EOI markers — rather than parsing the multipart boundary string —
is robust to per-part header variation across IP Webcam app versions. This
function is pure (bytes in, bytes + frames out) and unit-testable with synthetic
data, no phone required. Mirrors how `calibration.py` factors out
`calibrate_from_samples` as the testable core.

### `IPWebcamCapture` — the drop-in capture class

Interface mirrors `WebcamCapture` method-for-method:

```python
class IPWebcamCapture:
    def __init__(self, url: str, *, connect_timeout: float = 5.0,
                 read_timeout: float = 5.0): ...
    def start(self) -> None: ...                       # opens stream, raises RuntimeError on failure, spawns thread
    def read_latest(self, timeout: float = 1.0) -> Optional[np.ndarray]: ...
    def read_latest_with_ts(self, timeout: float = 1.0) -> Optional[tuple[np.ndarray, float]]: ...
    def stop(self) -> None: ...
```

- `url` may be given as `http://<ip>:8080/video` or just `http://<ip>:8080` /
  `<ip>:8080` — `start()` normalizes: prepend `http://` if no scheme, append
  `/video` if the path is empty.
- `start()` performs the initial `requests.get(url, stream=True,
  timeout=(connect_timeout, read_timeout))`. On connection failure it raises
  `RuntimeError(f"Could not open IP webcam stream at {url}: {exc}")` —
  fail-fast, matching `WebcamCapture` raising on a bad device. On success it
  hands the open response to a daemon `_loop` thread.
- `_loop` reads the stream in chunks (e.g. 4 KB), maintains a rolling byte
  buffer, runs it through `iter_jpeg_frames`, and for **each complete JPEG**
  decodes via `cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)`
  → BGR `ndarray` (identical format to `VideoCapture.read()`). Only the **latest**
  decoded frame is kept, under a lock, with `time.monotonic()` as its timestamp.
- **Reconnect on drop.** A read timeout, connection reset, or empty read does
  not kill the thread: it closes the response, sleeps a short backoff
  (~0.5–1 s), and reopens the stream. The loop exits only when `stop()` clears
  the running flag.
- `read_latest_with_ts` has **byte-identical semantics** to `WebcamCapture`: the
  timestamp advances only when a genuinely new frame is decoded, so the
  duplicate-frame dedup gate in `app.py` / `test_2D_3D.py`
  (`if frame_ts != last_processed_ts`) keeps working unchanged.
- `stop()` clears the running flag, joins the thread (timeout), closes the
  response/session.

## Data flow

```
Android IP Webcam app (phone, LAN)
  └─ MJPEG /video  ──HTTP──▶  IPWebcamCapture._loop (bg thread)
        chunks → rolling buffer → iter_jpeg_frames → cv2.imdecode → keep latest (lock)
                                                                         │
   main loop ── read_latest_with_ts() ──────────────────────────────────┘
        → (unchanged) Pose2D → measure → score → … exactly as with WebcamCapture
```

## Integration

Both entry points gain `argparse`:

- **`src/app.py`** — `run()` accepts the parsed source/url; constructs
  `IPWebcamCapture(url)` when `--source ip` else `WebcamCapture(device=0,
  width=1280, height=720)`. Everything downstream is untouched (interface match).
- **`src/test_2D_3D.py`** — same `--source` / `--url` flags; replaces the
  hardcoded `WebcamCapture(device=0, ...)` construction.
- `--source ip` **without** `--url` exits with a clear error.
- Default `--source webcam` ⇒ no behavior change for existing usage.

CLI examples:

```bash
uv run python main.py --source ip --url http://192.168.1.42:8080/video
uv run python src/test_2D_3D.py --source ip --url 192.168.1.42:8080
uv run python -m camera --url http://192.168.1.42:8080   # standalone preview
```

(`run()` parses `sys.argv` itself via `argparse`, so the existing
`main.py` — which just calls `run()` — needs no change.)

## Error handling

| Failure | Behavior |
| --- | --- |
| Initial connect fails | `start()` raises `RuntimeError` with the URL (fail-fast). |
| Mid-stream drop / read timeout | `_loop` reconnects with short backoff; does not crash. |
| Corrupt JPEG (`imdecode` → None) | Skip that blob; keep the previous latest frame. |
| No frame yet within `timeout` | `read_latest*` returns `None` (same as `WebcamCapture`). |

## Testing

`tests/pipeline/test_ip_webcam.py` — **pure-logic** tests for `iter_jpeg_frames`
(no network, no phone, fast):

- single complete JPEG → one frame, empty remainder
- two JPEGs in one buffer → two frames
- JPEG split across a chunk boundary → buffered, emitted once completed
- trailing partial frame → returned as remainder
- garbage/headers between EOI and next SOI → ignored, frames still extracted
- buffer with no complete frame → no frames, whole buffer (from first SOI) as remainder

URL normalization (`<ip>:8080` → `http://<ip>:8080/video`) is also pure and gets
a small test. No `*_smoke` test — a live stream is environment-dependent; the
`__main__` preview is the manual smoke check.

## Dependencies

Add `requests>=2.31` to `pyproject.toml` `[project].dependencies` (currently
only present transitively).

## Non-goals (YAGNI)

- No WebRTC / WebSocket / server endpoint yet — that's the later production
  shape; this is the LAN-prototype step.
- No audio, no auth, no TLS (IP Webcam on a trusted LAN).
- No `/shot.jpg` fallback path.
- No automatic phone discovery — the URL is passed explicitly.
