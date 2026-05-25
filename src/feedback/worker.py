import threading
from typing import Optional


class LLMWorker:
    """Background thread that processes the most recent payload at a time, dropping older ones.

    Accepts any payload type (RepAnalysis, HoldAnalysis, LiveSnapshot, etc.) via submit().
    Extra keyword arguments passed to submit() are forwarded to llm.generate().
    """

    def __init__(self, llm):
        self._llm = llm
        self._lock = threading.Lock()
        self._pending: Optional[tuple] = None
        self._latest_text: Optional[str] = None
        self._cv = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, payload, **kwargs):
        with self._cv:
            self._pending = (payload, kwargs)  # newer overwrites older
            self._cv.notify()

    def latest(self) -> Optional[str]:
        with self._lock:
            return self._latest_text

    def _loop(self):
        while self._running:
            with self._cv:
                while self._running and self._pending is None:
                    self._cv.wait(timeout=0.1)
                if not self._running:
                    return
                pending = self._pending
                self._pending = None
            if pending is None:
                continue  # spurious wake; loop again
            payload, kwargs = pending
            try:
                text = self._llm.generate(payload, **kwargs)
            except Exception as e:
                text = f"[LLM error: {e}]"
            with self._lock:
                self._latest_text = text

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
