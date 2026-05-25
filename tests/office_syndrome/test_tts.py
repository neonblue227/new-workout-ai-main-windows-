import io
import wave


def test_pcm_to_wav_header():
    from feedback.tts import _pcm_to_wav

    pcm = b"\x00\x01" * 240  # 240 16-bit samples
    wav = _pcm_to_wav(pcm, sample_rate=24000)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 240


def test_synthesize_falls_back_to_say_without_client(monkeypatch):
    from feedback.tts import GeminiTTS

    tts = GeminiTTS(api_key=None)  # no key -> no client
    assert tts._client is None
    monkeypatch.setattr(tts, "_synthesize_say", lambda text: b"AIFFDATA")
    audio, suffix = tts.synthesize("สวัสดี")
    assert audio == b"AIFFDATA"
    assert suffix == ".aiff"


def test_synthesize_falls_back_when_gemini_raises(monkeypatch):
    from feedback.tts import GeminiTTS

    tts = GeminiTTS(api_key=None)
    tts._client = object()  # pretend a client exists

    def boom(text):
        raise RuntimeError("network down")

    monkeypatch.setattr(tts, "_synthesize_gemini", boom)
    monkeypatch.setattr(tts, "_synthesize_say", lambda text: b"AIFFDATA")
    audio, suffix = tts.synthesize("สวัสดี")
    assert audio == b"AIFFDATA"
    assert suffix == ".aiff"


def test_synthesize_uses_gemini_when_available(monkeypatch):
    from feedback.tts import GeminiTTS

    tts = GeminiTTS(api_key=None)
    tts._client = object()
    monkeypatch.setattr(tts, "_synthesize_gemini", lambda text: b"WAVDATA")
    audio, suffix = tts.synthesize("hi")
    assert audio == b"WAVDATA"
    assert suffix == ".wav"


def test_synthesize_returns_empty_bytes_when_no_say_backend(monkeypatch):
    from feedback.tts import GeminiTTS

    tts = GeminiTTS(api_key=None)
    monkeypatch.setattr("feedback.tts.shutil.which", lambda name: None)

    audio, suffix = tts.synthesize("สวัสดี")

    assert audio == b""
    assert suffix == ".aiff"


class _FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return (b"AUDIO", ".wav")


def test_precache_synthesizes_each_phrase():
    from feedback.tts import TTSWorker

    fake = _FakeTTS()
    w = TTSWorker(fake)
    w.precache({"count_3": "สาม", "done": "เยี่ยม"})
    assert sorted(fake.calls) == ["สาม", "เยี่ยม"]
    assert "count_3" in w._cue_cache and "done" in w._cue_cache


def test_play_cue_enqueues_cached_only():
    from feedback.tts import TTSWorker

    w = TTSWorker(_FakeTTS())
    w.precache({"count_3": "สาม"})
    w.play_cue("count_3")
    assert len(w._cue_queue) == 1
    w.play_cue("missing")  # unknown cue -> no-op
    assert len(w._cue_queue) == 1


def test_submit_feedback_is_drop_stale():
    from feedback.tts import TTSWorker

    w = TTSWorker(_FakeTTS())
    w.submit_feedback("a")
    w.submit_feedback("b")
    assert w._pending_feedback_text == "b"
