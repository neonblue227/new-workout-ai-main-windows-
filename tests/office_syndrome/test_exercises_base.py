from exercises.base import JointTarget, TargetPose, PromptTemplate


def test_joint_target_immutable():
    j = JointTarget("head_lateral_tilt", -35.0, 10.0, "เอียงเพิ่ม")
    assert j.name == "head_lateral_tilt"
    assert j.target_deg == -35.0
    assert j.tolerance_deg == 10.0


def test_target_pose_defaults():
    t = TargetPose(joints=(JointTarget("k", 0.0, 5.0, "x"),))
    assert t.hold_seconds == 20.0
    assert t.side is None


def test_prompt_template_holds_both_strings():
    p = PromptTemplate(live="live {progress}", summary="sum {score}")
    assert p.live.format(progress=0.5) == "live 0.5"
    assert p.summary.format(score=88) == "sum 88"
