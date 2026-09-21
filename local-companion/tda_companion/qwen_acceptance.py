from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .asr_acceptance import NvmlPeakMonitor
from .asr_models import (
    AsrProfile,
    ModelRegistryError,
    QWEN_FORCED_ALIGNER_MODEL_ID,
    QWEN_FORCED_ALIGNER_REVISION,
    get_profile,
    inspect_model_install,
    model_path,
    reset_model_install,
    write_install_marker,
)

ACCEPTANCE_SCHEMA = "tda_qwen_gpu_acceptance_v1"
TRANSCRIPT_SCHEMA = "tda_qwen_acceptance_transcript_v1"
ALIGNER_DIRECTORY = "qwen3-forced-aligner-0.6b-hf"
MAX_ACCEPTANCE_AUDIO_BYTES = 2 * 1024**3
MAX_ACCEPTANCE_AUDIO_SECONDS = 240.0
QWEN_ACCEPTANCE_MAX_NEW_TOKENS = 512
_COPY_CHUNK = 1024 * 1024

ALIGNER_PROFILE = AsrProfile(
    id="qwen-forced-aligner",
    engine="qwen3",
    label="Qwen Forced Aligner",
    description="Qwen3 Forced Aligner usado somente após a transcrição ASR.",
    model_id=QWEN_FORCED_ALIGNER_MODEL_ID,
    revision=QWEN_FORCED_ALIGNER_REVISION,
    directory=ALIGNER_DIRECTORY,
    language="Portuguese",
    alignment=QWEN_FORCED_ALIGNER_MODEL_ID,
    alignment_revision=QWEN_FORCED_ALIGNER_REVISION,
    required_files=(
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
)


class QwenAcceptanceError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class QwenPlan:
    profile_id: str
    device: str
    dtype: str
    compute_capability: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_versions() -> dict[str, str]:
    values = {
        "torch": _distribution_version("torch"),
        "transformers": _distribution_version("transformers"),
        "accelerate": _distribution_version("accelerate"),
        "huggingface_hub": _distribution_version("huggingface-hub"),
        "safetensors": _distribution_version("safetensors"),
        "av": _distribution_version("av"),
    }
    try:
        import torch

        values["torch_cuda"] = str(torch.version.cuda or "none")
    except Exception:
        values["torch_cuda"] = "unavailable"
    return values


def _nvml_driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            value = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        text = str(value or "").strip()
        return text or None
    except Exception:
        return None


def _cuda_execution_failure_code(exc: BaseException) -> str:
    value = f"{type(exc).__name__}: {exc}".casefold()
    if any(
        marker in value
        for marker in (
            "driver version is insufficient",
            "cuda driver version is insufficient",
            "forward compatibility was attempted",
            "unsupported display driver",
        )
    ):
        return "QWEN_CUDA_DRIVER_INCOMPATIBLE"
    return "QWEN_CUDA_EXECUTION_FAILED"


def probe_qwen_cuda() -> dict[str, Any]:
    """Probe both CUDA discovery and an actual kernel execution.

    torch.cuda.is_available() alone is not a sufficient compatibility check: a
    packaged runtime can discover the GPU while the installed NVIDIA driver is
    too old for the CUDA family bundled with Torch. The physical gate therefore
    performs a tiny allocation/kernel/synchronize before any model is loaded.
    """
    driver_version = _nvml_driver_version()
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        devices: list[dict[str, Any]] = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            major, minor = torch.cuda.get_device_capability(index)
            devices.append(
                {
                    "index": index,
                    "name": str(props.name),
                    "compute_capability": f"{major}.{minor}",
                    "total_memory_bytes": int(props.total_memory),
                }
            )

        execution_ready: bool | None = None
        execution_error: str | None = None
        if available and count > 0:
            try:
                probe = torch.ones((32,), device="cuda:0", dtype=torch.float32)
                observed = float((probe * 2.0).sum().item())
                torch.cuda.synchronize()
                if observed != 64.0:
                    raise RuntimeError("CUDA_EXECUTION_RESULT_INVALID")
                execution_ready = True
            except Exception as exc:
                execution_ready = False
                execution_error = _cuda_execution_failure_code(exc)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return {
            "available": available,
            "device_count": count,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if available else False,
            "torch_cuda": str(torch.version.cuda or "none"),
            "driver_version": driver_version,
            "execution_ready": execution_ready,
            "execution_error": execution_error,
            "devices": devices,
        }
    except Exception:
        return {
            "available": False,
            "device_count": 0,
            "bf16_supported": False,
            "torch_cuda": "unavailable",
            "driver_version": driver_version,
            "execution_ready": False,
            "execution_error": "QWEN_CUDA_EXECUTION_FAILED",
            "devices": [],
        }

def _capability_tuple(value: str) -> tuple[int, int]:
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except (AttributeError, TypeError, ValueError) as exc:
        raise QwenAcceptanceError("QWEN_CUDA_CAPABILITY_INVALID") from exc


def resolve_qwen_plan(profile_id: str, *, cuda_status: dict[str, Any] | None = None) -> QwenPlan:
    profile = get_profile(profile_id)
    if profile.engine != "qwen3":
        raise QwenAcceptanceError("QWEN_PROFILE_REQUIRED")
    status = cuda_status if cuda_status is not None else probe_qwen_cuda()
    if not status.get("available") or int(status.get("device_count") or 0) < 1:
        raise QwenAcceptanceError("QWEN_CUDA_UNAVAILABLE")
    if status.get("execution_ready") is not True:
        code = str(status.get("execution_error") or "QWEN_CUDA_EXECUTION_FAILED")
        if code not in {"QWEN_CUDA_DRIVER_INCOMPATIBLE", "QWEN_CUDA_EXECUTION_FAILED"}:
            code = "QWEN_CUDA_EXECUTION_FAILED"
        raise QwenAcceptanceError(code)
    devices = status.get("devices") or []
    first = devices[0] if isinstance(devices, list) and devices else {}
    capability = str(first.get("compute_capability") or "")
    if _capability_tuple(capability) < (7, 5):
        raise QwenAcceptanceError("QWEN_CUDA_CAPABILITY_UNSUPPORTED")
    dtype = "bfloat16" if bool(status.get("bf16_supported")) else "float16"
    return QwenPlan(profile_id=profile.id, device="cuda", dtype=dtype, compute_capability=capability)


def _download_snapshot(
    profile: AsrProfile,
    target: Path,
    *,
    downloader: Callable[..., Any] | None = None,
) -> None:
    if not profile.revision:
        raise QwenAcceptanceError("QWEN_MODEL_REVISION_REQUIRED")
    if downloader is None:
        # Keep individual Hugging Face network reads finite. The full snapshot may
        # legitimately take much longer and remains resumable on the next attempt.
        os.environ["HF_HUB_ETAG_TIMEOUT"] = "15"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise QwenAcceptanceError("QWEN_RUNTIME_NOT_INSTALLED") from exc
        downloader = snapshot_download
    try:
        downloader(repo_id=profile.model_id, revision=profile.revision, local_dir=str(target))
    except Exception as exc:
        raise QwenAcceptanceError("QWEN_MODEL_DOWNLOAD_FAILED") from exc
    shutil.rmtree(target / ".cache", ignore_errors=True)


def _resumable_model_staging(downloads: Path, directory: str) -> Path:
    candidates: list[tuple[float, Path]] = []
    for candidate in downloads.glob(f"{directory}-*.partial"):
        try:
            # Resumable staging is writable input. Never follow a symlink or
            # Windows junction left in .downloads, otherwise model preparation
            # could write outside the configured Models root.
            is_junction = bool(
                getattr(candidate, "is_junction", lambda: False)()
            )
            if candidate.is_symlink() or is_junction or not candidate.is_dir():
                continue
            modified = candidate.stat().st_mtime
        except OSError:
            continue
        candidates.append((modified, candidate))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = candidates[0][1]
        for _, stale in candidates[1:]:
            shutil.rmtree(stale, ignore_errors=True)
        return chosen
    return downloads / f"{directory}-{uuid4().hex}.partial"


def prepare_qwen_model(
    models_root: Path,
    profile: AsrProfile,
    *,
    downloader: Callable[..., Any] | None = None,
) -> Path:
    if profile.engine != "qwen3" or profile.id not in {"qwen-fast", "qwen-quality"}:
        raise QwenAcceptanceError("QWEN_PROFILE_REQUIRED")
    state = inspect_model_install(models_root, profile)
    target = model_path(models_root, profile)
    if state["status"] == "ready":
        return target
    if target.exists() or target.is_symlink():
        try:
            reset_model_install(models_root, profile)
        except ModelRegistryError as exc:
            raise QwenAcceptanceError("QWEN_MODEL_REPAIR_FAILED") from exc

    downloads = models_root.resolve() / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    staging = _resumable_model_staging(downloads, profile.directory)
    staging.mkdir(parents=False, exist_ok=True)
    try:
        _download_snapshot(profile, staging, downloader=downloader)
        missing = [name for name in profile.required_files if not (staging / name).is_file()]
        if missing:
            raise QwenAcceptanceError("QWEN_MODEL_DOWNLOAD_INCOMPLETE")
        write_install_marker(staging, profile)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return target
    except QwenAcceptanceError as exc:
        if exc.code != "QWEN_MODEL_DOWNLOAD_FAILED":
            shutil.rmtree(staging, ignore_errors=True)
        raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare_qwen_aligner(
    models_root: Path,
    *,
    downloader: Callable[..., Any] | None = None,
) -> Path:
    state = inspect_model_install(models_root, ALIGNER_PROFILE)
    target = model_path(models_root, ALIGNER_PROFILE)
    if state["status"] == "ready":
        return target
    if target.exists() or target.is_symlink():
        try:
            reset_model_install(models_root, ALIGNER_PROFILE)
        except ModelRegistryError as exc:
            raise QwenAcceptanceError("QWEN_ALIGNER_REPAIR_FAILED") from exc

    downloads = models_root.resolve() / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    staging = _resumable_model_staging(downloads, ALIGNER_DIRECTORY)
    staging.mkdir(parents=False, exist_ok=True)
    try:
        _download_snapshot(ALIGNER_PROFILE, staging, downloader=downloader)
        missing = [name for name in ALIGNER_PROFILE.required_files if not (staging / name).is_file()]
        if missing:
            raise QwenAcceptanceError("QWEN_ALIGNER_DOWNLOAD_INCOMPLETE")
        write_install_marker(staging, ALIGNER_PROFILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return target
    except QwenAcceptanceError as exc:
        if exc.code != "QWEN_MODEL_DOWNLOAD_FAILED":
            shutil.rmtree(staging, ignore_errors=True)
        raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _model_is_cuda_only(model: Any) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        values = {str(value).lower() for value in device_map.values()}
        has_cuda = any(value.startswith("cuda") or value.isdigit() for value in values)
        has_offload = any(value.startswith("cpu") or value.startswith("disk") for value in values)
        return has_cuda and not has_offload
    return str(getattr(model, "device", "")).lower().startswith("cuda")


def _bounded_prompt(context: str, glossary: str) -> str:
    context_value = " ".join(context.split())[:2000].strip()
    glossary_value = " ".join(glossary.split())[:2000].strip()
    parts: list[str] = []
    if context_value:
        parts.append(f"Contexto da campanha: {context_value}")
    if glossary_value:
        parts.append(f"Vocabulário e nomes importantes: {glossary_value}")
    return "\n".join(parts)


def _audio_duration_seconds(path: Path) -> float:
    try:
        import av
    except ImportError as exc:
        raise QwenAcceptanceError("QWEN_RUNTIME_NOT_INSTALLED") from exc
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_STREAM_MISSING")
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
                if duration > 0:
                    return duration
            total = 0.0
            for frame in container.decode(stream):
                rate = int(frame.sample_rate or 0)
                if rate > 0:
                    total += float(frame.samples) / rate
            return total
    except QwenAcceptanceError:
        raise
    except Exception as exc:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_INVALID") from exc


def _validate_audio(path: Path, duration_reader: Callable[[Path], float]) -> tuple[Path, float]:
    source = path.resolve()
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_NOT_FOUND") from exc
    if not source.is_file() or size <= 0:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_INVALID")
    if size > MAX_ACCEPTANCE_AUDIO_BYTES:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_TOO_LARGE")
    duration = float(duration_reader(source))
    if duration <= 0:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_DURATION_MISSING")
    if duration > MAX_ACCEPTANCE_AUDIO_SECONDS:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_TOO_LONG")
    return source, duration


def _decode_audio_array(path: Path, *, sample_rate: int = 16_000) -> Any:
    """Decode acceptance audio with the runtime's bundled PyAV.

    Transformers 5.17 resolves string paths through torchcodec/librosa. The TDA
    runtime already owns deterministic audio decoding through PyAV, so hand the
    processor a NumPy waveform instead of a filesystem path.
    """
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise QwenAcceptanceError("QWEN_RUNTIME_NOT_INSTALLED") from exc

    pieces: list[Any] = []
    try:
        with av.open(str(path.resolve())) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_STREAM_MISSING")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    array = converted.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                    if array.size:
                        pieces.append(array.copy())
            for converted in resampler.resample(None):
                array = converted.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                if array.size:
                    pieces.append(array.copy())
    except QwenAcceptanceError:
        raise
    except Exception as exc:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_DECODE_FAILED") from exc

    if not pieces:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_AUDIO_EMPTY")
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _torch_dtype(torch: Any, name: str) -> Any:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise QwenAcceptanceError("QWEN_DTYPE_UNAVAILABLE") from exc


def _qwen_inference_failure_code(exc: BaseException) -> str:
    value = f"{type(exc).__name__}: {exc}".casefold()
    if any(
        marker in value
        for marker in (
            "driver version is insufficient",
            "cuda driver version is insufficient",
            "forward compatibility was attempted",
            "unsupported display driver",
        )
    ):
        return "QWEN_CUDA_DRIVER_INCOMPATIBLE"
    if any(
        marker in value
        for marker in (
            "out of memory",
            "cuda_error_out_of_memory",
            "cublas_status_alloc_failed",
            "failed to allocate",
            "not enough memory",
        )
    ):
        return "QWEN_ASR_GPU_MEMORY_EXHAUSTED"
    if any(
        marker in value
        for marker in (
            "cuda error",
            "device-side assert",
            "cublas",
            "cudnn",
        )
    ):
        return "QWEN_ASR_CUDA_FAILED"
    if any(marker in value for marker in ("librosa", "torchcodec", "audio backend")):
        return "QWEN_ASR_AUDIO_BACKEND_MISSING"
    if isinstance(exc, (AttributeError, TypeError)):
        return "QWEN_ASR_RUNTIME_API_FAILED"
    if isinstance(exc, ValueError):
        return "QWEN_ASR_INPUT_FAILED"
    return "QWEN_ASR_INFERENCE_FAILED"


def run_qwen_asr_sample(
    model_root: Path,
    audio_path: Path,
    plan: QwenPlan,
    *,
    prompt: str,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise QwenAcceptanceError("QWEN_RUNTIME_NOT_INSTALLED") from exc

    dtype = _torch_dtype(torch, plan.dtype)
    model: Any = None
    try:
        load_started = time.monotonic()
        processor = AutoProcessor.from_pretrained(str(model_root), local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_root), dtype=dtype, device_map={"": "cuda:0"}, local_files_only=True
        )
        load_seconds = max(time.monotonic() - load_started, 0.0)
        if not _model_is_cuda_only(model):
            raise QwenAcceptanceError("QWEN_MODEL_NOT_GPU_RESIDENT")

        inference_started = time.monotonic()
        audio = _decode_audio_array(audio_path)
        inputs = processor.apply_transcription_request(
            audio=audio, language="Portuguese", prompt=prompt or None
        )
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=QWEN_ACCEPTANCE_MAX_NEW_TOKENS)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        inference_seconds = max(time.monotonic() - inference_started, 0.0)
        text = str(parsed.get("transcription") or "").strip() if isinstance(parsed, dict) else ""
        language = str(parsed.get("language") or "Portuguese") if isinstance(parsed, dict) else "Portuguese"
        if not text:
            raise QwenAcceptanceError("QWEN_ACCEPTANCE_NO_SPEECH_RECOGNIZED")
        return {
            "text": text,
            "language": language,
            "compute_type": plan.dtype,
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
        }
    except QwenAcceptanceError:
        raise
    except Exception as exc:
        raise QwenAcceptanceError(_qwen_inference_failure_code(exc)) from exc
    finally:
        model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_qwen_alignment_sample(
    aligner_root: Path,
    audio_path: Path,
    text: str,
    language: str,
    plan: QwenPlan,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
    except ImportError as exc:
        raise QwenAcceptanceError("QWEN_RUNTIME_NOT_INSTALLED") from exc

    dtype = _torch_dtype(torch, plan.dtype)
    model: Any = None
    try:
        load_started = time.monotonic()
        processor = AutoProcessor.from_pretrained(str(aligner_root), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            str(aligner_root), dtype=dtype, device_map={"": "cuda:0"}, local_files_only=True
        )
        load_seconds = max(time.monotonic() - load_started, 0.0)
        if not _model_is_cuda_only(model):
            raise QwenAcceptanceError("QWEN_ALIGNER_NOT_GPU_RESIDENT")

        inference_started = time.monotonic()
        audio = _decode_audio_array(audio_path)
        aligner_inputs, word_lists = processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=text, language=language or "Portuguese"
        )
        aligner_inputs = aligner_inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            outputs = model(**aligner_inputs)
        timestamps = processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )[0]
        inference_seconds = max(time.monotonic() - inference_started, 0.0)
        words: list[dict[str, Any]] = []
        for item in timestamps or []:
            if not isinstance(item, dict):
                continue
            word = str(item.get("text") or "").strip()
            if not word:
                continue
            start = float(item.get("start_time") or 0.0)
            end = float(item.get("end_time") or start)
            if end < start:
                raise QwenAcceptanceError("QWEN_ALIGNMENT_TIMESTAMPS_INVALID")
            words.append({"text": word, "start": round(start, 3), "end": round(end, 3)})
        if not words:
            raise QwenAcceptanceError("QWEN_ALIGNMENT_EMPTY")
        return {
            "words": words,
            "compute_type": plan.dtype,
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
        }
    except QwenAcceptanceError:
        raise
    except Exception as exc:
        code = _qwen_inference_failure_code(exc)
        if code in {
            "QWEN_CUDA_DRIVER_INCOMPATIBLE",
            "QWEN_ASR_GPU_MEMORY_EXHAUSTED",
            "QWEN_ASR_CUDA_FAILED",
            "QWEN_ASR_RUNTIME_API_FAILED",
        }:
            raise QwenAcceptanceError(code) from exc
        raise QwenAcceptanceError("QWEN_ALIGNMENT_FAILED") from exc
    finally:
        model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _gpu_name_matches(metrics: dict[str, Any], required: str) -> bool:
    if not required:
        return True
    return required.casefold() in str(metrics.get("name") or "").casefold()


def run_qwen_gpu_acceptance(
    audio_path: Path,
    models_root: Path,
    *,
    profile_id: str,
    glossary: str = "",
    context: str = "",
    required_gpu_name: str | None = None,
    transcript_out: Path | None = None,
    cuda_status: dict[str, Any] | None = None,
    prepare_model: Callable[..., Path] = prepare_qwen_model,
    prepare_aligner: Callable[..., Path] = prepare_qwen_aligner,
    asr_runner: Callable[..., dict[str, Any]] = run_qwen_asr_sample,
    aligner_runner: Callable[..., dict[str, Any]] = run_qwen_alignment_sample,
    monitor_factory: Callable[[], Any] = NvmlPeakMonitor,
    duration_reader: Callable[[Path], float] = _audio_duration_seconds,
) -> dict[str, Any]:
    source, duration_seconds = _validate_audio(audio_path, duration_reader)
    profile = get_profile(profile_id)
    if profile.engine != "qwen3":
        raise QwenAcceptanceError("QWEN_PROFILE_REQUIRED")
    status = cuda_status if cuda_status is not None else probe_qwen_cuda()
    plan = resolve_qwen_plan(profile.id, cuda_status=status)
    prompt = _bounded_prompt(context, glossary)

    total_started = time.monotonic()
    prepare_started = time.monotonic()
    model_root = prepare_model(models_root.resolve(), profile)
    asr_prepare_seconds = max(time.monotonic() - prepare_started, 0.0)

    asr_monitor = monitor_factory()
    asr_monitor.start()
    try:
        asr_result = asr_runner(model_root, source, plan, prompt=prompt)
    finally:
        asr_gpu = asr_monitor.stop()

    text = str(asr_result.get("text") or "").strip()
    language = str(asr_result.get("language") or "Portuguese")
    if not text:
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_NO_SPEECH_RECOGNIZED")

    align_prepare_started = time.monotonic()
    aligner_root = prepare_aligner(models_root.resolve())
    aligner_prepare_seconds = max(time.monotonic() - align_prepare_started, 0.0)

    align_monitor = monitor_factory()
    align_monitor.start()
    try:
        alignment_result = aligner_runner(aligner_root, source, text, language, plan)
    finally:
        alignment_gpu = align_monitor.stop()

    words = alignment_result.get("words")
    if not isinstance(words, list) or not words:
        raise QwenAcceptanceError("QWEN_ALIGNMENT_EMPTY")

    required = (required_gpu_name or "").strip()
    asr_gpu_match = _gpu_name_matches(asr_gpu, required)
    aligner_gpu_match = _gpu_name_matches(alignment_gpu, required)
    if required and (not asr_gpu_match or not aligner_gpu_match):
        raise QwenAcceptanceError("QWEN_ACCEPTANCE_GPU_NAME_MISMATCH")

    if transcript_out is not None:
        _atomic_json(
            transcript_out.resolve(),
            {
                "schema": TRANSCRIPT_SCHEMA,
                "profile_id": profile.id,
                "model": profile.model_id,
                "model_revision": profile.revision,
                "alignment_model": QWEN_FORCED_ALIGNER_MODEL_ID,
                "alignment_revision": QWEN_FORCED_ALIGNER_REVISION,
                "language": language,
                "text": text,
                "words": words,
            },
        )

    asr_inference = max(0.0, float(asr_result.get("inference_seconds") or 0.0))
    alignment_inference = max(0.0, float(alignment_result.get("inference_seconds") or 0.0))
    total_seconds = max(time.monotonic() - total_started, 0.0)
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "pass": True,
        "profile_id": profile.id,
        "model": profile.model_id,
        "model_revision": profile.revision,
        "alignment_model": QWEN_FORCED_ALIGNER_MODEL_ID,
        "alignment_revision": QWEN_FORCED_ALIGNER_REVISION,
        "language": language,
        "audio_sha256": _sha256_file(source),
        "runtime": _runtime_versions(),
        "cuda": status,
        "gpu": {**asr_gpu, "required_name": required or None, "required_name_match": asr_gpu_match},
        "alignment_gpu": {
            **alignment_gpu,
            "required_name": required or None,
            "required_name_match": aligner_gpu_match,
        },
        "inference": {
            "device": plan.device,
            "compute_type": str(asr_result.get("compute_type") or plan.dtype),
            "audio_seconds": round(duration_seconds, 3),
            "prepare_seconds": round(asr_prepare_seconds, 3),
            "model_load_seconds": round(float(asr_result.get("model_load_seconds") or 0.0), 3),
            "transcription_seconds": round(asr_inference, 3),
            "rtf": round(asr_inference / duration_seconds, 6),
            "transcript_sha256": text_sha,
            "transcript_written": transcript_out is not None,
        },
        "alignment": {
            "compute_type": str(alignment_result.get("compute_type") or plan.dtype),
            "prepare_seconds": round(aligner_prepare_seconds, 3),
            "model_load_seconds": round(float(alignment_result.get("model_load_seconds") or 0.0), 3),
            "inference_seconds": round(alignment_inference, 3),
            "rtf": round(alignment_inference / duration_seconds, 6),
            "word_count": len(words),
        },
        "total_seconds": round(total_seconds, 3),
    }
