import time
from feedback.worker import LLMWorker
from analysis.types import RepAnalysis


class FakeLLM:
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0

    def generate(self, rep, max_tokens: int = 120, frame_bgr=None) -> str:
        time.sleep(self.delay)
        self.calls += 1
        return f"feedback for rep {rep.rep_index}"


def make_rep(idx: int) -> RepAnalysis:
    return RepAnalysis(
        rep_index=idx,
        score=80,
        components={"depth": 30, "valgus": 20, "torso": 15, "symmetry": 10, "tempo": 5},
        violations=[],
        descent_ms=1000,
        ascent_ms=1000,
    )


def test_worker_returns_feedback_async():
    llm = FakeLLM(delay=0.05)
    w = LLMWorker(llm)
    w.start()
    try:
        w.submit(make_rep(0))
        deadline = time.monotonic() + 2.0
        text = None
        while time.monotonic() < deadline:
            text = w.latest()
            if text:
                break
            time.sleep(0.01)
        assert text == "feedback for rep 0"
    finally:
        w.stop()


def test_worker_drops_stale_submissions():
    llm = FakeLLM(delay=0.2)
    w = LLMWorker(llm)
    w.start()
    try:
        for i in range(5):
            w.submit(make_rep(i))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.latest() == "feedback for rep 4":
                break
            time.sleep(0.05)
        assert llm.calls <= 2
        assert w.latest() == "feedback for rep 4"
    finally:
        w.stop()
