"""Thai text-to-speech for the neck-stretch demo.

GeminiTTS synthesizes via Google AI Studio (gemini-2.5-flash-preview-tts) and
falls back to the offline macOS `say -v Kanya` voice on any error, so the demo
never goes mute. TTSWorker plays audio on a background thread (afplay), with a
priority cue channel (pre-cached fixed phrases) and a drop-stale feedback
channel for live LLM coaching.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
_GEMINI_VOICE = "Kore"  # multilingual prebuilt voice; handles Thai
_SAMPLE_RATE = 24000  # Gemini TTS returns 24kHz 16-bit mono PCM
_SAY_VOICE = "Kanya"  # macOS th_TH voice


def _pcm_to_wav(pcm: bytes, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Wrap raw signed-16-bit mono PCM in a WAV container so afplay can play it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class GeminiTTS:
    """Synthesize Thai speech. Returns (audio_bytes, file_suffix)."""

    def __init__(self, api_key: Optional[str] = None, voice: str = _GEMINI_VOICE):
        self.voice = voice
        self._api_key = api_key or os.getenv("google_ai_studio_api_key")
        self._client = None
        if self._api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:  # pragma: no cover - env dependent
                print(f"[tts] Gemini client init failed; using macOS say: {e}")
                self._client = None

    def synthesize(self, text: str) -> tuple[bytes, str]:
        if self._client is not None:
            try:
                return self._synthesize_gemini(text), ".wav"
            except Exception as e:
                print(f"[tts] Gemini synth failed ({e}); falling back to say")
        return self._synthesize_say(text), ".aiff"

    def _synthesize_gemini(self, text: str) -> bytes:
        assert (
            self._client is not None
        )  # only called from synthesize() when client is set
        from google.genai import types

        resp = self._client.models.generate_content(
            model=_GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            ),
        )
        pcm = resp.candidates[0].content.parts[0].inline_data.data
        assert isinstance(pcm, bytes), (
            f"Gemini TTS: expected bytes PCM, got {type(pcm)}"
        )
        return _pcm_to_wav(pcm)

    def _synthesize_say(self, text: str) -> bytes:
        if shutil.which("say") is None:
            return b""

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            path = f.name
        try:
            subprocess.run(
                ["say", "-v", _SAY_VOICE, "-o", path, text],
                check=True,
                capture_output=True,
            )
            return Path(path).read_bytes()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class TTSWorker:
    """Background audio player.

    - Cue channel: FIFO queue of pre-synthesized (bytes, suffix) clips. Never
      dropped; checked before feedback so cues win at scheduling boundaries.
    - Feedback channel: a single pending text (drop-stale, like LLMWorker).
      Synthesized in the worker thread and played only when no cue is queued.
    One clip plays at a time (afplay is blocking in the worker), so audio never
    overlaps.
    """

    def __init__(self, tts: GeminiTTS):
        self._tts = tts
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._cue_cache: dict[str, tuple[bytes, str]] = {}
        self._cue_queue: list[tuple[bytes, str]] = []
        self._pending_feedback_text: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def precache(self, phrases: dict[str, str]) -> None:
        """Synthesize fixed cue phrases once (blocking). Call at startup."""
        assert not self._running, "precache() must be called before start()"
        for key, text in phrases.items():
            self._cue_cache[key] = self._tts.synthesize(text)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def play_cue(self, key: str) -> None:
        with self._cv:
            clip = self._cue_cache.get(key)
            if clip is not None:
                self._cue_queue.append(clip)
                self._cv.notify()

    def submit_feedback(self, text: str) -> None:
        if not text:
            return
        with self._cv:
            self._pending_feedback_text = text  # newer overwrites older
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            kind = None
            clip = None
            text = None
            with self._cv:
                while (
                    self._running
                    and not self._cue_queue
                    and (self._pending_feedback_text is None)
                ):
                    self._cv.wait(timeout=0.1)
                if not self._running:
                    return
                if self._cue_queue:
                    kind, clip = "cue", self._cue_queue.pop(0)
                elif self._pending_feedback_text is not None:
                    kind, text = "feedback", self._pending_feedback_text
                    self._pending_feedback_text = None
            if kind == "cue":
                assert clip is not None
                self._play(clip[0], clip[1])
            elif kind == "feedback":
                assert text is not None
                try:
                    audio, suffix = self._tts.synthesize(text)
                except Exception as e:
                    print(f"[tts] feedback synth failed: {e}")
                    continue
                # A cue may have arrived during synthesis — it wins; drop this.
                with self._lock:
                    cue_waiting = bool(self._cue_queue)
                if not cue_waiting:
                    self._play(audio, suffix)

    def _play(self, audio: bytes, suffix: str) -> None:
        if not audio:
            return

        player = shutil.which("afplay")
        if player is None:
            return

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio)
            path = f.name
        try:
            subprocess.run([player, path], check=False)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
