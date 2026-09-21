from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .asr_models import (
    ModelRegistryError,
    compute_model_metadata_sha256,
    get_profile,
    inspect_model_install,
    model_path,
    verify_and_upgrade_model_install,
)
from .qwen_acceptance import (
    ACCEPTANCE_SCHEMA,
    ALIGNER_PROFILE,
    QWEN_FORCED_ALIGNER_MODEL_ID,
    QWEN_FORCED_ALIGNER_REVISION,
)
from .qwen_runtime import QWEN_RUNTIME_ID, inspect_qwen_runtime, qwen_version_root

LEGACY_GATE_SCHEMA = "tda_qwen_physical_gate_v1"
GATE_SCHEMA = "tda_qwen_physical_gate_v2"
GATE_DIRECTORY = "qwen-physical-gates"
QWEN_PROFILES = ("qwen-fast", "qwen-quality")
MIN_GATE_AUDIO_SECONDS = 60.0
MAX_GATE_BYTES = 128 * 1024
_SHA256 = set("0123456789abcdef")
_FORBIDDEN_KEYS = {"text", "words", "transcript", "segments", "audio"}


class QwenPhysicalGateError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _SHA256 for char in value)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.partial")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _gate_path(state_root: Path, profile_id: str) -> Path:
    if profile_id not in QWEN_PROFILES:
        raise QwenPhysicalGateError("QWEN_GATE_PROFILE_INVALID")
    return state_root.resolve() / GATE_DIRECTORY / f"{profile_id}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
        if stat.st_size <= 0 or stat.st_size > MAX_GATE_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _contains_private_payload(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                return True
            if _contains_private_payload(child):
                return True
    elif isinstance(value, list):
        return any(_contains_private_payload(child) for child in value)
    return False


def _runtime_identity(runtime_root: Path, *, verify_worker: bool) -> dict[str, str]:
    state = inspect_qwen_runtime(runtime_root, verify_worker=verify_worker)
    version = state.get("version")
    worker_value = state.get("worker")
    if (
        state.get("status") != "ready"
        or not isinstance(version, str)
        or not isinstance(worker_value, str)
    ):
        raise QwenPhysicalGateError("QWEN_GATE_RUNTIME_NOT_READY")
    marker = _read_json(qwen_version_root(runtime_root, version) / ".tda-runtime.json")
    if marker is None:
        raise QwenPhysicalGateError("QWEN_GATE_RUNTIME_MARKER_INVALID")
    worker_sha = marker.get("worker_sha256")
    archive_sha = marker.get("archive_sha256")
    if (
        marker.get("runtime_id") != QWEN_RUNTIME_ID
        or marker.get("version") != version
        or not _is_sha256(worker_sha)
        or not _is_sha256(archive_sha)
    ):
        raise QwenPhysicalGateError("QWEN_GATE_RUNTIME_MARKER_INVALID")
    try:
        worker = Path(worker_value)
        stat = worker.stat()
    except OSError as exc:
        raise QwenPhysicalGateError("QWEN_GATE_RUNTIME_NOT_READY") from exc
    worker_metadata_sha256 = _canonical_sha256(
        {
            "name": worker.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    return {
        "runtime_id": QWEN_RUNTIME_ID,
        "version": version,
        "worker_sha256": str(worker_sha),
        "archive_sha256": str(archive_sha),
        "worker_metadata_sha256": worker_metadata_sha256,
    }


def _model_identity(
    models_root: Path,
    profile_id: str,
    *,
    verify_hash: bool,
    verification: dict[str, bool] | None = None,
) -> dict[str, str]:
    profile = get_profile(profile_id)
    if profile.engine != "qwen3" or profile.id not in QWEN_PROFILES:
        raise QwenPhysicalGateError("QWEN_GATE_PROFILE_INVALID")
    state = inspect_model_install(
        models_root,
        profile,
        verify_hash=verify_hash,
        verification=verification,
    )
    if state.get("status") == "corrupt":
        raise QwenPhysicalGateError("QWEN_GATE_BINDING_CHANGED")
    if state.get("status") != "ready" or not _is_sha256(state.get("content_sha256")):
        raise QwenPhysicalGateError("QWEN_GATE_MODEL_NOT_READY")
    try:
        metadata_sha256 = compute_model_metadata_sha256(model_path(models_root, profile))
    except (ModelRegistryError, OSError) as exc:
        raise QwenPhysicalGateError("QWEN_GATE_MODEL_NOT_READY") from exc
    return {
        "profile_id": profile.id,
        "model_id": profile.model_id,
        "revision": str(profile.revision or ""),
        "content_sha256": str(state["content_sha256"]),
        "metadata_sha256": metadata_sha256,
    }


def _aligner_identity(
    models_root: Path,
    *,
    verify_hash: bool,
    verification: dict[str, bool] | None = None,
) -> dict[str, str]:
    state = inspect_model_install(
        models_root,
        ALIGNER_PROFILE,
        verify_hash=verify_hash,
        verification=verification,
    )
    if state.get("status") == "corrupt":
        raise QwenPhysicalGateError("QWEN_GATE_BINDING_CHANGED")
    if state.get("status") != "ready" or not _is_sha256(state.get("content_sha256")):
        raise QwenPhysicalGateError("QWEN_GATE_ALIGNER_NOT_READY")
    try:
        metadata_sha256 = compute_model_metadata_sha256(
            model_path(models_root, ALIGNER_PROFILE)
        )
    except (ModelRegistryError, OSError) as exc:
        raise QwenPhysicalGateError("QWEN_GATE_ALIGNER_NOT_READY") from exc
    return {
        "model_id": QWEN_FORCED_ALIGNER_MODEL_ID,
        "revision": QWEN_FORCED_ALIGNER_REVISION,
        "content_sha256": str(state["content_sha256"]),
        "metadata_sha256": metadata_sha256,
    }


def _capability_tuple(value: object) -> tuple[int, int]:
    try:
        major, minor = str(value).split(".", 1)
        return int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise QwenPhysicalGateError("QWEN_GATE_GPU_INVALID") from exc


def _validate_acceptance(
    receipt: object,
    profile_id: str,
    required_gpu_name: str | None,
) -> dict[str, Any]:
    if not isinstance(receipt, dict) or receipt.get("schema") != ACCEPTANCE_SCHEMA or receipt.get("pass") is not True:
        raise QwenPhysicalGateError("QWEN_GATE_ACCEPTANCE_INVALID")
    if _contains_private_payload(receipt):
        raise QwenPhysicalGateError("QWEN_GATE_PRIVATE_PAYLOAD_REJECTED")
    profile = get_profile(profile_id)
    if (
        receipt.get("profile_id") != profile.id
        or receipt.get("model") != profile.model_id
        or receipt.get("model_revision") != profile.revision
        or receipt.get("alignment_model") != QWEN_FORCED_ALIGNER_MODEL_ID
        or receipt.get("alignment_revision") != QWEN_FORCED_ALIGNER_REVISION
    ):
        raise QwenPhysicalGateError("QWEN_GATE_ACCEPTANCE_IDENTITY_MISMATCH")

    cuda = receipt.get("cuda") if isinstance(receipt.get("cuda"), dict) else {}
    devices = cuda.get("devices") if isinstance(cuda.get("devices"), list) else []
    first = devices[0] if devices and isinstance(devices[0], dict) else {}
    if cuda.get("available") is not True or int(cuda.get("device_count") or 0) < 1:
        raise QwenPhysicalGateError("QWEN_GATE_CUDA_REQUIRED")
    if cuda.get("execution_ready") is not True:
        code = str(cuda.get("execution_error") or "QWEN_CUDA_EXECUTION_FAILED")
        if code not in {"QWEN_CUDA_DRIVER_INCOMPATIBLE", "QWEN_CUDA_EXECUTION_FAILED"}:
            code = "QWEN_CUDA_EXECUTION_FAILED"
        raise QwenPhysicalGateError(code)
    if _capability_tuple(first.get("compute_capability")) < (7, 5):
        raise QwenPhysicalGateError("QWEN_GATE_GPU_UNSUPPORTED")

    required = (required_gpu_name or "").strip()
    gpu = receipt.get("gpu") if isinstance(receipt.get("gpu"), dict) else {}
    alignment_gpu = receipt.get("alignment_gpu") if isinstance(receipt.get("alignment_gpu"), dict) else {}
    for value in (gpu, alignment_gpu):
        if not str(value.get("name") or "").strip():
            raise QwenPhysicalGateError("QWEN_GATE_GPU_INVALID")
        if required:
            if value.get("required_name_match") is not True:
                raise QwenPhysicalGateError("QWEN_GATE_GPU_NAME_MISMATCH")
            if required.casefold() not in str(value.get("name") or "").casefold():
                raise QwenPhysicalGateError("QWEN_GATE_GPU_NAME_MISMATCH")

    inference = receipt.get("inference") if isinstance(receipt.get("inference"), dict) else {}
    alignment = receipt.get("alignment") if isinstance(receipt.get("alignment"), dict) else {}
    try:
        audio_seconds = float(inference.get("audio_seconds") or 0.0)
    except (TypeError, ValueError) as exc:
        raise QwenPhysicalGateError("QWEN_GATE_ACCEPTANCE_METRICS_INVALID") from exc
    if audio_seconds < MIN_GATE_AUDIO_SECONDS:
        raise QwenPhysicalGateError("QWEN_GATE_AUDIO_TOO_SHORT")
    if (
        inference.get("device") != "cuda"
        or float(inference.get("rtf") or 0.0) < 0
        or int(alignment.get("word_count") or 0) < 1
    ):
        raise QwenPhysicalGateError("QWEN_GATE_ACCEPTANCE_METRICS_INVALID")
    return receipt


def _binding_payload(
    profile_id: str,
    runtime: dict[str, str],
    model: dict[str, str],
    aligner: dict[str, str],
    *,
    schema: str = GATE_SCHEMA,
) -> dict[str, object]:
    return {
        "schema": schema,
        "profile_id": profile_id,
        "runtime": runtime,
        "model": model,
        "aligner": aligner,
    }


def _legacy_identity(value: dict[str, str]) -> dict[str, str]:
    return {
        key: child
        for key, child in value.items()
        if key not in {"worker_metadata_sha256", "metadata_sha256"}
    }


def record_qwen_physical_gate(
    state_root: Path,
    runtime_root: Path,
    models_root: Path,
    receipt: object,
    *,
    profile_id: str,
    required_gpu_name: str | None = None,
) -> dict[str, Any]:
    accepted = _validate_acceptance(receipt, profile_id, required_gpu_name)
    # Gate creation remains the expensive byte-verification trust event.
    # Normal dispatch validates the sealed byte identities plus cheap metadata.
    runtime = _runtime_identity(runtime_root, verify_worker=True)
    model = _model_identity(models_root, profile_id, verify_hash=True)
    aligner = _aligner_identity(models_root, verify_hash=True)
    binding = _binding_payload(profile_id, runtime, model, aligner)
    gpu = accepted["gpu"]
    cuda = accepted["cuda"]
    devices = cuda.get("devices") if isinstance(cuda.get("devices"), list) else []
    first = devices[0] if devices and isinstance(devices[0], dict) else {}
    inference = accepted["inference"]
    alignment = accepted["alignment"]
    required = (required_gpu_name or "").strip()
    value: dict[str, Any] = {
        **binding,
        "binding_sha256": _canonical_sha256(binding),
        "acceptance_sha256": _canonical_sha256(accepted),
        "accepted_at": _utc_now(),
        "metadata_sealed_at": _utc_now(),
        "required_gpu_name": required or None,
        "gpu": {
            "name": str(gpu.get("name") or first.get("name") or ""),
            "compute_capability": str(first.get("compute_capability") or ""),
            "total_memory_bytes": int(first.get("total_memory_bytes") or 0),
        },
        "metrics": {
            "compute_type": str(inference.get("compute_type") or ""),
            "audio_seconds": float(inference.get("audio_seconds") or 0.0),
            "transcription_rtf": float(inference.get("rtf") or 0.0),
            "alignment_rtf": float(alignment.get("rtf") or 0.0),
            "word_count": int(alignment.get("word_count") or 0),
        },
        "contains_audio": False,
        "contains_transcript": False,
    }
    _atomic_json(_gate_path(state_root, profile_id), value)
    return inspect_qwen_physical_gate(
        state_root,
        runtime_root,
        models_root,
        profile_id=profile_id,
        verify_model_content=False,
    )


def inspect_qwen_physical_gate(
    state_root: Path,
    runtime_root: Path,
    models_root: Path,
    *,
    profile_id: str,
    verify_model_content: bool = False,
) -> dict[str, Any]:
    """Validate the sealed Qwen gate using cheap metadata by default.

    v2 fingerprints file paths, sizes and mtimes, catching ordinary corruption
    without reading multi-GB model bytes. Explicit diagnostics can still request
    full byte hashing. A valid v1 gate is upgraded once in place; this migration
    is no weaker than the previous v1 dispatch behavior.
    """
    path = _gate_path(state_root, profile_id)
    value = _read_json(path)
    if value is None:
        return {"status": "missing", "ready": False, "profile_id": profile_id}
    schema = value.get("schema")
    if (
        schema not in {GATE_SCHEMA, LEGACY_GATE_SCHEMA}
        or value.get("profile_id") != profile_id
        or value.get("contains_audio") is not False
        or value.get("contains_transcript") is not False
    ):
        return {"status": "invalid", "ready": False, "profile_id": profile_id}

    metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else {}
    try:
        accepted_audio_seconds = float(metrics.get("audio_seconds") or 0.0)
    except (TypeError, ValueError):
        accepted_audio_seconds = 0.0
    if accepted_audio_seconds < MIN_GATE_AUDIO_SECONDS:
        return {
            "status": "invalid",
            "ready": False,
            "profile_id": profile_id,
            "reason": "QWEN_GATE_AUDIO_TOO_SHORT",
        }

    model_verification: dict[str, bool] = {}
    aligner_verification: dict[str, bool] = {}
    try:
        runtime = _runtime_identity(runtime_root, verify_worker=verify_model_content)
        model = _model_identity(
            models_root,
            profile_id,
            verify_hash=verify_model_content,
            verification=model_verification,
        )
        aligner = _aligner_identity(
            models_root,
            verify_hash=verify_model_content,
            verification=aligner_verification,
        )
    except QwenPhysicalGateError as exc:
        return {
            "status": "stale",
            "ready": False,
            "profile_id": profile_id,
            "reason": exc.code,
        }

    if schema == LEGACY_GATE_SCHEMA:
        legacy_runtime = _legacy_identity(runtime)
        legacy_model = _legacy_identity(model)
        legacy_aligner = _legacy_identity(aligner)
        legacy_binding = _binding_payload(
            profile_id,
            legacy_runtime,
            legacy_model,
            legacy_aligner,
            schema=LEGACY_GATE_SCHEMA,
        )
        if (
            value.get("runtime") != legacy_runtime
            or value.get("model") != legacy_model
            or value.get("aligner") != legacy_aligner
            or value.get("binding_sha256") != _canonical_sha256(legacy_binding)
            or not _is_sha256(value.get("acceptance_sha256"))
        ):
            return {
                "status": "stale",
                "ready": False,
                "profile_id": profile_id,
                "reason": "QWEN_GATE_BINDING_CHANGED",
            }
        # Legacy v1 did not seal cheap file metadata. Before promoting it
        # to v2, prove the current runtime/model/aligner bytes still match the
        # accepted identities. This is intentionally a one-time deep verification,
        # never part of normal dispatch.
        try:
            runtime = _runtime_identity(runtime_root, verify_worker=True)
            model_state = verify_and_upgrade_model_install(models_root, profile_id)
            aligner_state = verify_and_upgrade_model_install(models_root, ALIGNER_PROFILE)
            if model_state.get("status") != "ready":
                raise QwenPhysicalGateError("QWEN_GATE_BINDING_CHANGED")
            if aligner_state.get("status") != "ready":
                raise QwenPhysicalGateError("QWEN_GATE_BINDING_CHANGED")
            model = _model_identity(models_root, profile_id, verify_hash=False)
            aligner = _aligner_identity(models_root, verify_hash=False)
        except (ModelRegistryError, OSError, QwenPhysicalGateError):
            return {
                "status": "stale",
                "ready": False,
                "profile_id": profile_id,
                "reason": "QWEN_GATE_BINDING_CHANGED",
            }

        deep_legacy_binding = _binding_payload(
            profile_id,
            _legacy_identity(runtime),
            _legacy_identity(model),
            _legacy_identity(aligner),
            schema=LEGACY_GATE_SCHEMA,
        )
        if (
            value.get("runtime") != _legacy_identity(runtime)
            or value.get("model") != _legacy_identity(model)
            or value.get("aligner") != _legacy_identity(aligner)
            or value.get("binding_sha256") != _canonical_sha256(deep_legacy_binding)
        ):
            return {
                "status": "stale",
                "ready": False,
                "profile_id": profile_id,
                "reason": "QWEN_GATE_BINDING_CHANGED",
            }

        binding = _binding_payload(profile_id, runtime, model, aligner)
        value = {
            **value,
            **binding,
            "binding_sha256": _canonical_sha256(binding),
            "metadata_sealed_at": _utc_now(),
        }
        _atomic_json(path, value)
    else:
        binding = _binding_payload(profile_id, runtime, model, aligner)
        binding_matches = (
            value.get("runtime") == runtime
            and value.get("model") == model
            and value.get("aligner") == aligner
            and value.get("binding_sha256") == _canonical_sha256(binding)
            and _is_sha256(value.get("acceptance_sha256"))
        )
        if not binding_matches:
            stored_runtime = value.get("runtime") if isinstance(value.get("runtime"), dict) else {}
            stored_model = value.get("model") if isinstance(value.get("model"), dict) else {}
            stored_aligner = value.get("aligner") if isinstance(value.get("aligner"), dict) else {}
            metadata_only_drift = (
                _is_sha256(value.get("acceptance_sha256"))
                and _legacy_identity(stored_runtime) == _legacy_identity(runtime)
                and _legacy_identity(stored_model) == _legacy_identity(model)
                and _legacy_identity(stored_aligner) == _legacy_identity(aligner)
            )
            if not metadata_only_drift:
                return {
                    "status": "stale",
                    "ready": False,
                    "profile_id": profile_id,
                    "reason": "QWEN_GATE_BINDING_CHANGED",
                }

            # A metadata-only difference can come from copy/backup/AV activity.
            # Deep-verify only the components whose sealed metadata actually
            # changed. Unchanged component bindings already match the accepted v2
            # receipt and must not trigger unrelated multi-GB reads.
            runtime_drift = stored_runtime != runtime
            model_drift = stored_model != model
            aligner_drift = stored_aligner != aligner
            try:
                if runtime_drift:
                    runtime = _runtime_identity(runtime_root, verify_worker=True)
                if model_drift and not model_verification.get("content_verified"):
                    model_state = verify_and_upgrade_model_install(models_root, profile_id)
                    if model_state.get("status") != "ready":
                        raise QwenPhysicalGateError("QWEN_GATE_BINDING_CHANGED")
                    model = _model_identity(models_root, profile_id, verify_hash=False)
                if aligner_drift and not aligner_verification.get("content_verified"):
                    aligner_state = verify_and_upgrade_model_install(
                        models_root,
                        ALIGNER_PROFILE,
                    )
                    if aligner_state.get("status") != "ready":
                        raise QwenPhysicalGateError("QWEN_GATE_BINDING_CHANGED")
                    aligner = _aligner_identity(models_root, verify_hash=False)
            except (ModelRegistryError, OSError, QwenPhysicalGateError):
                return {
                    "status": "stale",
                    "ready": False,
                    "profile_id": profile_id,
                    "reason": "QWEN_GATE_BINDING_CHANGED",
                }

            if (
                _legacy_identity(stored_runtime) != _legacy_identity(runtime)
                or _legacy_identity(stored_model) != _legacy_identity(model)
                or _legacy_identity(stored_aligner) != _legacy_identity(aligner)
            ):
                return {
                    "status": "stale",
                    "ready": False,
                    "profile_id": profile_id,
                    "reason": "QWEN_GATE_BINDING_CHANGED",
                }

            binding = _binding_payload(profile_id, runtime, model, aligner)
            value = {
                **value,
                **binding,
                "binding_sha256": _canonical_sha256(binding),
                "metadata_sealed_at": _utc_now(),
            }
            _atomic_json(path, value)

    gpu = value.get("gpu") if isinstance(value.get("gpu"), dict) else {}
    required = str(value.get("required_gpu_name") or "").strip()
    if required and required.casefold() not in str(gpu.get("name") or "").casefold():
        return {
            "status": "invalid",
            "ready": False,
            "profile_id": profile_id,
            "reason": "QWEN_GATE_GPU_NAME_MISMATCH",
        }
    return {
        "status": "ready",
        "ready": True,
        "profile_id": profile_id,
        "accepted_at": value.get("accepted_at"),
        "runtime_version": runtime["version"],
        "gpu": gpu,
        "metrics": metrics,
    }


def ready_qwen_profiles(
    state_root: Path,
    runtime_root: Path,
    models_root: Path,
) -> list[str]:
    return [
        profile_id
        for profile_id in QWEN_PROFILES
        if inspect_qwen_physical_gate(
            state_root,
            runtime_root,
            models_root,
            profile_id=profile_id,
        ).get("ready")
        is True
    ]
