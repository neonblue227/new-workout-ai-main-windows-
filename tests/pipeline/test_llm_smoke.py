import builtins
from pathlib import Path
import pytest

QWEN_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "qwen3_4b"


def test_llm_falls_back_when_transformers_is_unavailable(monkeypatch):
    from analysis.types import HoldAnalysis, HoldState, LiveSnapshot, RepAnalysis

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers" or name.startswith("transformers."):
            raise ModuleNotFoundError("no module named 'transformers'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    from feedback.llm import ThaiCoachLLM

    llm = ThaiCoachLLM()
    assert llm._backend == "fallback"

    rep = RepAnalysis(
        rep_index=0,
        score=72,
        components={"depth": 20, "valgus": 15, "torso": 18, "symmetry": 10, "tempo": 9},
        violations=[],
        descent_ms=1200,
        ascent_ms=900,
    )
    hold = HoldAnalysis(
        exercise_name="neck_stretch_left",
        score=80,
        components={"duration": 40, "precision": 25, "stability": 15},
        violations=[],
        in_target_ms=20000,
        drift_count=1,
    )
    snap = LiveSnapshot(
        exercise_name="neck_stretch_left",
        state=HoldState.HOLDING,
        progress_ratio=0.5,
        current_violations=[],
    )

    rep_text = llm.generate(rep)
    hold_text = llm.generate(hold, exercise="neck_stretch_left")
    live_text = llm.generate(snap, exercise="neck_stretch_left")

    assert isinstance(rep_text, str)
    assert isinstance(hold_text, str)
    assert isinstance(live_text, str)
    assert rep_text
    assert hold_text
    assert live_text


@pytest.mark.skipif(not QWEN_DIR.exists(), reason="Qwen model not downloaded")
def test_llm_generates_thai_text():
    from feedback.llm import ThaiCoachLLM
    from analysis.types import RepAnalysis

    llm = ThaiCoachLLM()
    rep = RepAnalysis(
        rep_index=0,
        score=78,
        components={"depth": 30, "valgus": 18, "torso": 20, "symmetry": 12, "tempo": 8},
        violations=[],
        descent_ms=1200,
        ascent_ms=900,
    )
    text = llm.generate(rep, max_tokens=120)
    assert isinstance(text, str)
    assert len(text) > 5
    assert any("฀" <= c <= "๿" for c in text)


# Smoke tests for hold/live payloads. Requires the Qwen weights.


@pytest.mark.skipif(not QWEN_DIR.exists(), reason="Qwen weights not downloaded")
def test_generate_accepts_hold_analysis():
    from analysis.types import HoldAnalysis
    from exercises.neck_stretch import NeckStretchLeft
    from feedback.llm import ThaiCoachLLM

    llm = ThaiCoachLLM()
    ex = NeckStretchLeft()
    a = HoldAnalysis(
        exercise_name=ex.name,
        score=88,
        components={"duration": 50, "precision": 25, "stability": 13},
        violations=[],
        in_target_ms=20_000,
        drift_count=1,
    )
    text = llm.generate(a, max_tokens=32, exercise=ex)
    assert text
