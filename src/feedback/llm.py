import re
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.types import HoldAnalysis, HoldState, LiveSnapshot, RepAnalysis
from feedback.prompt_th import (
    SYSTEM_TH,
    SYSTEM_TH_HOLD,
    build_hold_summary_prompt,
    build_live_prompt,
    build_user_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "qwen3_5_4b_mxfp4"

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


class ThaiCoachLLM:
    """Wraps Qwen3.5-4B (vision-language) via mlx-vlm, with a Thai fallback."""

    def __init__(self, model_dir: Path | str | None = None):
        self._backend = "fallback"
        self._model = None
        self._processor = None
        self._config = None

        model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            self._model, self._processor = load(str(model_path))
            self._config = load_config(str(model_path))
            self._backend = "mlx_vlm"
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"[llm] mlx_vlm unavailable ({exc}); using fallback Thai feedback")

    def generate(
        self,
        payload,  # RepAnalysis | HoldAnalysis | LiveSnapshot
        max_tokens: int = 160,
        frame_bgr: Optional[np.ndarray] = None,
        exercise=None,  # required for HoldAnalysis / LiveSnapshot
    ) -> str:
        if self._backend == "mlx_vlm":
            from mlx_vlm import generate as mlx_generate
            from mlx_vlm.prompt_utils import apply_chat_template

            if isinstance(payload, RepAnalysis):
                system = SYSTEM_TH
                user = build_user_prompt(payload)
            elif isinstance(payload, HoldAnalysis):
                if exercise is None:
                    raise ValueError("exercise= required for HoldAnalysis")
                system = SYSTEM_TH_HOLD
                user = build_hold_summary_prompt(payload, exercise)
            elif isinstance(payload, LiveSnapshot):
                if exercise is None:
                    raise ValueError("exercise= required for LiveSnapshot")
                system = SYSTEM_TH_HOLD
                user = build_live_prompt(payload, exercise)
            else:
                raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = apply_chat_template(
                self._processor,
                self._config,
                messages,
                num_images=0,
                enable_thinking=False,
            )
            result = mlx_generate(
                self._model,
                self._processor,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
            text = getattr(result, "text", str(result))
            return _THINK_BLOCK.sub("", text).strip()

        if isinstance(payload, RepAnalysis):
            return self._fallback_rep(payload)
        if isinstance(payload, HoldAnalysis):
            if exercise is None:
                raise ValueError("exercise= required for HoldAnalysis")
            return self._fallback_hold(payload, exercise)
        if isinstance(payload, LiveSnapshot):
            if exercise is None:
                raise ValueError("exercise= required for LiveSnapshot")
            return self._fallback_live(payload, exercise)
        raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

    def warmup(self):
        """First call is slow due to compilation. Run once at app start."""
        if self._backend == "fallback":
            return

        dummy = RepAnalysis(
            rep_index=-1,
            score=50,
            components={
                "depth": 10,
                "valgus": 10,
                "torso": 10,
                "symmetry": 10,
                "tempo": 10,
            },
            violations=[],
            descent_ms=0,
            ascent_ms=0,
        )
        _ = self.generate(dummy, max_tokens=16)

    def _fallback_rep(self, payload: RepAnalysis) -> str:
        detail = payload.violations[0].detail_th if payload.violations else ""
        if payload.score >= 85:
            base = "ดีมากค่ะ คุณกำลังทำได้ดี"
        elif payload.score >= 65:
            base = "กำลังดีค่ะ ช่วยคงท่าตรงและลึกพอเหมาะ"
        else:
            base = "ลองช้าลงและปรับให้คออยู่ในตำแหน่งที่ถูกต้อง"
        if detail:
            return f"{base} ({detail})"
        return base

    def _fallback_hold(self, payload: HoldAnalysis, exercise) -> str:
        name = getattr(exercise, "name", str(exercise))
        detail = payload.violations[0].detail_th if payload.violations else ""
        if payload.score >= 80:
            base = f"การฝึก{name} ทำได้ดีในตอนนี้"
        elif payload.score >= 60:
            base = f"ช่วยปรับการคงท่าใน{name} ให้มั่นคงขึ้น"
        else:
            base = f"ใน{name} ลองลดการ drift และคงท่าที่ปลอดภัย"
        if detail:
            return f"{base} ({detail})"
        return base

    def _fallback_live(self, payload: LiveSnapshot, exercise) -> str:
        name = getattr(exercise, "name", str(exercise))
        if payload.state == HoldState.HOLDING:
            base = f"กำลังดีค่ะ คงท่าใน{name} ได้สม่ำเสมอ"
        elif payload.state in {HoldState.DRIFTED, HoldState.ENTERING}:
            base = f"ช่วยกลับไปคงท่าใน{name} ให้ตรงและปลอดภัย"
        else:
            base = f"เตรียมพร้อมและเริ่มฝึก{name} ด้วยความช้า ๆ"
        if payload.current_violations:
            return f"{base} ({payload.current_violations[0].detail_th})"
        return base
