from exercises import EXERCISES


def test_registry_contains_neck_stretch_left():
    assert "neck_stretch_left" in EXERCISES


def test_every_exercise_has_required_attrs():
    for key, ex in EXERCISES.items():
        assert ex.name == key, f"key/name mismatch for {key}"
        assert ex.display_th, f"{key} missing display_th"
        assert ex.target.joints, f"{key} has no target joints"
        assert ex.prompt.live, f"{key} missing live prompt"
        assert ex.prompt.summary, f"{key} missing summary prompt"
