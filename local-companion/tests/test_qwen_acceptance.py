from __future__ import annotations

from contextlib import nullcontext
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import tda_companion.qwen_acceptance as acceptance
from tda_companion.qwen_acceptance import (
    QwenAcceptanceError,
    _qwen_inference_failure_code,
    run_qwen_gpu_acceptance,
)


class _Monitor:
    def start(self) -> None:
        return None

    def stop(self) -> dict:
        return {
            "available": True,
            "index": 0,
            "name": "NVIDIA GeForce RTX 4070 Laptop GPU",
            "driver": "999.1",
            "memory_total_bytes": 8 * 1024**3,
            "baseline_memory_used_bytes": 512 * 1024**2,
            "peak_memory_used_bytes": 7 * 1024**3,
            "peak_utilization_percent": 96,
        }


def _cuda() -> dict:
    return {
        "available": True,
        "device_count": 1,
        "bf16_supported": True,
        "torch_cuda": "12.6",
        "driver_version": "570.144",
        "execution_ready": True,
        "execution_error": None,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4070 Laptop GPU",
                "compute_capability": "8.9",
                "total_memory_bytes": 8 * 1024**3,
            }
        ],
    }


def _prepare_model(_models_root: Path, _profile) -> Path:
    return Path("qwen-model")


def _prepare_aligner(_models_root: Path) -> Path:
    return Path("qwen-aligner")


def _asr(_model_root: Path, _audio: Path, plan, *, prompt: str) -> dict:
    assert plan.device == "cuda"
    assert plan.dtype == "bfloat16"
    assert "Vocabulário" in prompt
    return {
        "text": "segredo da mesa",
        "language": "Portuguese",
        "compute_type": "bfloat16",
        "model_load_seconds": 0.2,
        "inference_seconds": 0.5,
    }


def _align(_aligner_root: Path, _audio: Path, text: str, language: str, plan) -> dict:
    assert text == "segredo da mesa"
    assert language == "Portuguese"
    assert plan.device == "cuda"
    return {
        "words": [
            {"text": "segredo", "start": 0.1, "end": 0.4},
            {"text": "da", "start": 0.4, "end": 0.5},
            {"text": "mesa", "start": 0.5, "end": 0.8},
        ],
        "compute_type": "bfloat16",
        "model_load_seconds": 0.1,
        "inference_seconds": 0.2,
    }


def _duration(_audio: Path) -> float:
    return 2.0


def _fake_qwen_modules(monkeypatch, processor):
    torch = ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    torch.inference_mode = lambda: nullcontext()

    class FakeModel:
        hf_device_map = {"": "cuda:0"}
        device = "cuda:0"
        dtype = "bfloat16"
        config = SimpleNamespace(timestamp_token_id=1)

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return processor

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = FakeAutoProcessor
    transformers.AutoModelForMultimodalLM = FakeAutoModel
    transformers.AutoModelForTokenClassification = FakeAutoModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_qwen_asr_gate_passes_decoded_waveform_not_filesystem_path(monkeypatch, tmp_path: Path):
    waveform = object()
    seen: dict[str, object] = {}

    class Processor:
        def apply_transcription_request(self, *, audio, language, prompt):
            seen["audio"] = audio
            seen["language"] = language
            seen["prompt"] = prompt
            raise QwenAcceptanceError("SENTINEL_WAVEFORM_REACHED")

    _fake_qwen_modules(monkeypatch, Processor())
    monkeypatch.setattr(acceptance, "_decode_audio_array", lambda _path: waveform)

    plan = acceptance.QwenPlan(
        profile_id="qwen-quality",
        device="cuda",
        dtype="bfloat16",
        compute_capability="8.9",
    )
    with pytest.raises(QwenAcceptanceError, match="SENTINEL_WAVEFORM_REACHED"):
        acceptance.run_qwen_asr_sample(
            tmp_path / "model",
            tmp_path / "acceptance-window.wav",
            plan,
            prompt="Dandelion",
        )

    assert seen == {
        "audio": waveform,
        "language": "Portuguese",
        "prompt": "Dandelion",
    }


def test_qwen_aligner_gate_passes_decoded_waveform_not_filesystem_path(monkeypatch, tmp_path: Path):
    waveform = object()
    seen: dict[str, object] = {}

    class Processor:
        def prepare_forced_aligner_inputs(self, *, audio, transcript, language):
            seen["audio"] = audio
            seen["transcript"] = transcript
            seen["language"] = language
            raise QwenAcceptanceError("SENTINEL_ALIGN_WAVEFORM_REACHED")

    _fake_qwen_modules(monkeypatch, Processor())
    monkeypatch.setattr(acceptance, "_decode_audio_array", lambda _path: waveform)

    plan = acceptance.QwenPlan(
        profile_id="qwen-fast",
        device="cuda",
        dtype="bfloat16",
        compute_capability="8.9",
    )
    with pytest.raises(QwenAcceptanceError, match="SENTINEL_ALIGN_WAVEFORM_REACHED"):
        acceptance.run_qwen_alignment_sample(
            tmp_path / "aligner",
            tmp_path / "acceptance-window.wav",
            "segredo da mesa",
            "Portuguese",
            plan,
        )

    assert seen == {
        "audio": waveform,
        "transcript": "segredo da mesa",
        "language": "Portuguese",
    }


def test_qwen_receipt_proves_gpu_and_alignment_without_leaking_transcript(tmp_path: Path):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"fake-audio")

    receipt = run_qwen_gpu_acceptance(
        audio,
        tmp_path / "Models",
        profile_id="qwen-fast",
        glossary="Dandelion, Pipipi",
        required_gpu_name="RTX 4070",
        cuda_status=_cuda(),
        prepare_model=_prepare_model,
        prepare_aligner=_prepare_aligner,
        asr_runner=_asr,
        aligner_runner=_align,
        monitor_factory=_Monitor,
        duration_reader=_duration,
    )

    assert receipt["pass"] is True
    assert receipt["inference"]["device"] == "cuda"
    assert receipt["inference"]["compute_type"] == "bfloat16"
    assert receipt["alignment"]["word_count"] == 3
    assert receipt["gpu"]["required_name_match"] is True
    assert receipt["alignment_gpu"]["required_name_match"] is True
    assert receipt["model_revision"]
    assert receipt["alignment_revision"]
    serialized = json.dumps(receipt, ensure_ascii=False)
    assert "segredo da mesa" not in serialized
    assert "fake-audio" not in serialized


def test_qwen_writes_transcript_only_when_explicit(tmp_path: Path):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"fake-audio")
    transcript = tmp_path / "qwen-acceptance-transcript.json"

    receipt = run_qwen_gpu_acceptance(
        audio,
        tmp_path / "Models",
        profile_id="qwen-quality",
        glossary="Dandelion",
        transcript_out=transcript,
        cuda_status=_cuda(),
        prepare_model=_prepare_model,
        prepare_aligner=_prepare_aligner,
        asr_runner=_asr,
        aligner_runner=_align,
        monitor_factory=_Monitor,
        duration_reader=_duration,
    )

    assert receipt["inference"]["transcript_written"] is True
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    assert payload["schema"] == "tda_qwen_acceptance_transcript_v1"
    assert payload["text"] == "segredo da mesa"
    assert payload["words"][0]["text"] == "segredo"


def test_qwen_requires_cuda_capability_and_expected_gpu(tmp_path: Path):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"fake-audio")

    with pytest.raises(QwenAcceptanceError, match="QWEN_CUDA_UNAVAILABLE"):
        run_qwen_gpu_acceptance(
            audio,
            tmp_path / "Models",
            profile_id="qwen-fast",
            cuda_status={"available": False, "device_count": 0, "devices": []},
            prepare_model=_prepare_model,
            prepare_aligner=_prepare_aligner,
            asr_runner=_asr,
            aligner_runner=_align,
            monitor_factory=_Monitor,
            duration_reader=_duration,
        )

    turing = _cuda()
    turing["bf16_supported"] = False
    turing["devices"] = [{**turing["devices"][0], "compute_capability": "7.5"}]
    plan = acceptance.resolve_qwen_plan("qwen-fast", cuda_status=turing)
    assert plan.compute_capability == "7.5"
    assert plan.dtype == "float16"

    unsupported = _cuda()
    unsupported["devices"] = [{**unsupported["devices"][0], "compute_capability": "7.0"}]
    with pytest.raises(QwenAcceptanceError, match="QWEN_CUDA_CAPABILITY_UNSUPPORTED"):
        acceptance.resolve_qwen_plan("qwen-fast", cuda_status=unsupported)

    with pytest.raises(QwenAcceptanceError, match="QWEN_ACCEPTANCE_GPU_NAME_MISMATCH"):
        run_qwen_gpu_acceptance(
            audio,
            tmp_path / "Models",
            profile_id="qwen-fast",
            glossary="Dandelion",
            required_gpu_name="RTX 3090",
            cuda_status=_cuda(),
            prepare_model=_prepare_model,
            prepare_aligner=_prepare_aligner,
            asr_runner=_asr,
            aligner_runner=_align,
            monitor_factory=_Monitor,
            duration_reader=_duration,
        )


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            RuntimeError("CUDA driver version is insufficient for CUDA runtime version"),
            "QWEN_CUDA_DRIVER_INCOMPATIBLE",
        ),
        (RuntimeError("CUDA out of memory"), "QWEN_ASR_GPU_MEMORY_EXHAUSTED"),
        (RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"), "QWEN_ASR_CUDA_FAILED"),
        (ImportError("librosa is required to load audio"), "QWEN_ASR_AUDIO_BACKEND_MISSING"),
        (RuntimeError("torchcodec audio backend is unavailable"), "QWEN_ASR_AUDIO_BACKEND_MISSING"),
        (AttributeError("processor has no attribute"), "QWEN_ASR_RUNTIME_API_FAILED"),
        (ValueError("invalid audio shape"), "QWEN_ASR_INPUT_FAILED"),
        (RuntimeError("unknown backend failure"), "QWEN_ASR_INFERENCE_FAILED"),
    ],
)
def test_qwen_inference_failure_classifier_keeps_safe_cause_class(
    error: BaseException,
    code: str,
):
    assert _qwen_inference_failure_code(error) == code


def test_qwen_rejects_cuda_status_that_never_proved_execution(tmp_path: Path):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"fake-audio")
    unproven = _cuda()
    unproven.pop("execution_ready", None)
    unproven.pop("execution_error", None)

    with pytest.raises(QwenAcceptanceError, match="QWEN_CUDA_EXECUTION_FAILED"):
        run_qwen_gpu_acceptance(
            audio,
            tmp_path / "Models",
            profile_id="qwen-fast",
            cuda_status=unproven,
            prepare_model=_prepare_model,
            prepare_aligner=_prepare_aligner,
            asr_runner=_asr,
            aligner_runner=_align,
            monitor_factory=_Monitor,
            duration_reader=_duration,
        )


def test_qwen_rejects_discovered_gpu_when_cuda_execution_is_not_compatible(tmp_path: Path):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"fake-audio")
    incompatible = _cuda()
    incompatible["execution_ready"] = False
    incompatible["execution_error"] = "QWEN_CUDA_DRIVER_INCOMPATIBLE"

    with pytest.raises(QwenAcceptanceError, match="QWEN_CUDA_DRIVER_INCOMPATIBLE"):
        run_qwen_gpu_acceptance(
            audio,
            tmp_path / "Models",
            profile_id="qwen-fast",
            cuda_status=incompatible,
            prepare_model=_prepare_model,
            prepare_aligner=_prepare_aligner,
            asr_runner=_asr,
            aligner_runner=_align,
            monitor_factory=_Monitor,
            duration_reader=_duration,
        )


def test_qwen_acceptance_rejects_samples_above_alignment_window(tmp_path: Path):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"fake-audio")

    with pytest.raises(QwenAcceptanceError, match="QWEN_ACCEPTANCE_AUDIO_TOO_LONG"):
        run_qwen_gpu_acceptance(
            audio,
            tmp_path / "Models",
            profile_id="qwen-fast",
            cuda_status=_cuda(),
            prepare_model=_prepare_model,
            prepare_aligner=_prepare_aligner,
            asr_runner=_asr,
            aligner_runner=_align,
            monitor_factory=_Monitor,
            duration_reader=lambda _path: 241.0,
        )
