from analysis.types import HoldAnalysis, HoldState, LiveSnapshot, RuleViolation
from exercises.neck_stretch import NeckStretchLeft
from feedback.prompt_th import build_hold_summary_prompt, build_live_prompt


def test_build_live_prompt_uses_exercise_template():
    ex = NeckStretchLeft()
    snap = LiveSnapshot(
        exercise_name=ex.name,
        state=HoldState.HOLDING,
        progress_ratio=0.5,
        current_violations=[RuleViolation("head_lateral_tilt", 0.3, "เอียงเพิ่ม")],
    )
    text = build_live_prompt(snap, ex)
    assert ex.display_th in text
    assert "50%" in text
    assert "เอียงเพิ่ม" in text


def test_build_summary_prompt_uses_exercise_template():
    ex = NeckStretchLeft()
    analysis = HoldAnalysis(
        exercise_name=ex.name,
        score=82,
        components={"duration": 50, "precision": 22, "stability": 10},
        violations=[],
        in_target_ms=20_000,
        drift_count=1,
    )
    text = build_hold_summary_prompt(analysis, ex)
    assert ex.display_th in text
    assert "82" in text
    assert "50/50" in text


def test_build_live_prompt_handles_no_violations():
    ex = NeckStretchLeft()
    snap = LiveSnapshot(ex.name, HoldState.HOLDING, 0.1, [])
    text = build_live_prompt(snap, ex)
    assert text  # renders without KeyError or empty-list crash
