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
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "qwen3_4b"

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


class ThaiCoachLLM:
    """Wraps Qwen3-4B via HuggingFace Transformers + bitsandbytes INT4 on CUDA.

    Falls back to static Thai templates when the model directory is absent or
    transformers/bitsandbytes are unavailable (e.g. CPU-only CI environment).
    """

    def __init__(self, model_dir: Path | str | None = None):
        self._backend = "fallback"
        self._model = None
        self._tokenizer = None

        model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not model_path.exists():
                raise FileNotFoundError(
                    f"[llm] Model directory not found: {model_path}. "
                    "Run: uv run python scripts/download_models.py"
                )

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )

            # Prefer CUDA; fall back to CPU (float32) so code still runs on
            # machines without a GPU (e.g. CI runners).
            if torch.cuda.is_available():
                from transformers import BitsAndBytesConfig

                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=quant_cfg,
                    device_map="cuda",
                    trust_remote_code=True,
                )
                self._backend = "transformers_cuda"
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )
                self._backend = "transformers_cpu"

        except Exception as exc:
            print(f"[llm] transformers unavailable ({exc}); using fallback Thai feedback")

    def generate(
        self,
        payload,  # RepAnalysis | HoldAnalysis | LiveSnapshot
        max_tokens: int = 160,
        frame_bgr: Optional[np.ndarray] = None,
        exercise=None,  # required for HoldAnalysis / LiveSnapshot
    ) -> str:
        if self._backend.startswith("transformers"):
            import torch

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

            # apply_chat_template handles system/user formatting for Qwen3
            text_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # suppress <think> blocks at the model level
            )
            inputs = self._tokenizer(text_input, return_tensors="pt")

            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (skip the prompt)
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return _THINK_BLOCK.sub("", text).strip()

        # ── fallback (static templates) ────────────────────────────────────
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
        """First call compiles CUDA kernels. Run once at app start."""
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

    # ── static fallback templates ──────────────────────────────────────────

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
