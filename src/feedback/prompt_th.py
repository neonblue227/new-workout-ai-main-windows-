from analysis.types import HoldAnalysis, LiveSnapshot, RepAnalysis, RuleViolation
from exercises.base import Exercise

SYSTEM_TH = (
    "คุณเป็นโค้ชฟิตเนสที่ให้คำแนะนำเกี่ยวกับท่าสควอตอย่างกระชับและสุภาพ "
    "ตอบเป็นภาษาไทย 2-3 ประโยคเท่านั้น ระบุสิ่งที่ทำถูกและสิ่งที่ควรแก้ไข "
    "ห้ามใช้ Markdown ห้ามใช้หัวข้อหรือบุลเล็ต ห้ามใช้ภาษาอังกฤษ "
    "ตอบเป็นข้อความธรรมดาที่อ่านออกเสียงได้โดยตรง"
)


def build_user_prompt(rep: RepAnalysis) -> str:
    lines = [
        f"ผลสควอตรอบที่ {rep.rep_index + 1}: คะแนนรวม {rep.score}/100",
        f"  ความลึก: {rep.components['depth']}/30",
        f"  หัวเข่า: {rep.components['valgus']}/25",
        f"  ลำตัว: {rep.components['torso']}/20",
        f"  สมมาตร: {rep.components['symmetry']}/15",
        f"  จังหวะ: {rep.components['tempo']}/10",
        f"เวลาลง: {rep.descent_ms} ms | เวลาขึ้น: {rep.ascent_ms} ms",
    ]
    if rep.violations:
        lines.append("ข้อสังเกต:")
        for v in rep.violations:
            lines.append(f"  - {v.detail_th} (ระดับ {v.severity:.2f})")
    else:
        lines.append("ไม่พบข้อสังเกตที่ต้องแก้ไข")
    lines.append("ช่วยสรุปสั้นๆ ว่าทำดีตรงไหน ควรปรับตรงไหน")
    return "\n".join(lines)


SYSTEM_TH_HOLD = (
    "คุณเป็นโค้ชยืดเหยียดที่ให้คำแนะนำกระชับและสุภาพ "
    "ตอบเป็นภาษาไทย 1-2 ประโยคเท่านั้น ไม่ใช้ Markdown ไม่ใช้บุลเล็ต ไม่ใช้ภาษาอังกฤษ"
)


def _format_violations(vs: list[RuleViolation]) -> str:
    if not vs:
        return "(ไม่มี)"
    return "; ".join(f"{v.detail_th} (ระดับ {v.severity:.2f})" for v in vs)


def build_live_prompt(snapshot: LiveSnapshot, exercise: Exercise) -> str:
    return exercise.prompt.live.format(
        exercise_th=exercise.display_th,
        state=snapshot.state.value,
        progress_pct=int(round(snapshot.progress_ratio * 100)),
        violations=_format_violations(snapshot.current_violations),
    )


def build_hold_summary_prompt(analysis: HoldAnalysis, exercise: Exercise) -> str:
    return exercise.prompt.summary.format(
        exercise_th=exercise.display_th,
        score=analysis.score,
        duration=analysis.components.get("duration", 0),
        precision=analysis.components.get("precision", 0),
        stability=analysis.components.get("stability", 0),
        violations=_format_violations(analysis.violations),
    )
