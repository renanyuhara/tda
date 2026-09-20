from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

import tda_companion.qwen_acceptance as qwen_acceptance
import tda_companion.qwen_physical_gate as gate_module
import tda_companion.qwen_runtime as qwen_runtime_module
from tda_companion.asr_models import MODEL_MARKER, get_profile, model_path, write_install_marker
from tda_companion.qwen_acceptance import ALIGNER_PROFILE
from tda_companion.qwen_physical_gate import (
    MIN_GATE_AUDIO_SECONDS,
    QwenPhysicalGateError,
    inspect_qwen_physical_gate,
    ready_qwen_profiles,
    record_qwen_physical_gate,
)
from tda_companion.qwen_runtime import install_qwen_runtime_archive
from tda_companion.runtime_compat import MIN_COMPATIBLE_QWEN_RUNTIME_VERSION


def test_qwen_resumable_staging_ignores_redirected_partial(tmp_path: Path):
    downloads = tmp_path / ".downloads"
    downloads.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = downloads / "qwen3-asr-redirect.partial"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this runner")

    chosen = qwen_acceptance._resumable_model_staging(downloads, "qwen3-asr")

    assert chosen != redirected
    assert chosen.parent == downloads
    assert chosen.name.startswith("qwen3-asr-")
    assert chosen.name.endswith(".partial")


def _install_runtime(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "qwen-runtime.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("TDAQwenWorker.exe", b"worker-v1")
        bundle.writestr("_internal/torch.dll", b"torch")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    install_qwen_runtime_archive(
        archive,
        root / "Runtime",
        version=MIN_COMPATIBLE_QWEN_RUNTIME_VERSION,
        expected_sha256=digest,
    )


def _install_model(models_root: Path, profile_id: str) -> Path:
    profile = get_profile(profile_id)
    target = model_path(models_root, profile)
    target.mkdir(parents=True)
    for index, name in enumerate(profile.required_files, start=1):
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{profile_id}:{index}".encode())
    write_install_marker(target, profile)
    return target


def _install_aligner(models_root: Path) -> Path:
    target = model_path(models_root, ALIGNER_PROFILE)
    target.mkdir(parents=True)
    for index, name in enumerate(ALIGNER_PROFILE.required_files, start=1):
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"aligner:{index}".encode())
    write_install_marker(target, ALIGNER_PROFILE)
    return target


def _receipt(profile_id: str = "qwen-fast") -> dict:
    profile = get_profile(profile_id)
    gpu = "NVIDIA GeForce RTX 4070 Laptop GPU"
    return {
        "schema": "tda_qwen_gpu_acceptance_v1",
        "pass": True,
        "profile_id": profile.id,
        "model": profile.model_id,
        "model_revision": profile.revision,
        "alignment_model": ALIGNER_PROFILE.model_id,
        "alignment_revision": ALIGNER_PROFILE.revision,
        "language": "Portuguese",
        "audio_sha256": "a" * 64,
        "runtime": {
            "torch": "2.13.0+cu126",
            "transformers": "5.17.0",
            "torch_cuda": "12.6",
        },
        "cuda": {
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
                    "name": gpu,
                    "compute_capability": "8.9",
                    "total_memory_bytes": 8 * 1024**3,
                }
            ],
        },
        "gpu": {
            "name": gpu,
            "required_name": "RTX 4070",
            "required_name_match": True,
            "peak_used_memory_bytes": 6 * 1024**3,
        },
        "alignment_gpu": {
            "name": gpu,
            "required_name": "RTX 4070",
            "required_name_match": True,
            "peak_used_memory_bytes": 5 * 1024**3,
        },
        "inference": {
            "device": "cuda",
            "compute_type": "bfloat16",
            "audio_seconds": MIN_GATE_AUDIO_SECONDS,
            "rtf": 0.42,
            "transcript_sha256": "b" * 64,
            "transcript_written": False,
        },
        "alignment": {
            "compute_type": "bfloat16",
            "rtf": 0.18,
            "word_count": 87,
        },
        "total_seconds": 108.0,
    }


def _prepared(tmp_path: Path, profile_id: str = "qwen-fast") -> tuple[Path, Path, Path]:
    _install_runtime(tmp_path)
    models = tmp_path / "Models"
    _install_model(models, profile_id)
    _install_aligner(models)
    return tmp_path / "State", tmp_path / "Runtime", models


def test_physical_gate_binds_runtime_model_aligner_and_contains_no_private_text(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    gate = record_qwen_physical_gate(
        state,
        runtime,
        models,
        _receipt(),
        profile_id="qwen-fast",
    )

    assert gate["ready"] is True
    assert gate["runtime_version"] == MIN_COMPATIBLE_QWEN_RUNTIME_VERSION
    assert gate["metrics"]["audio_seconds"] == MIN_GATE_AUDIO_SECONDS
    persisted = (state / "qwen-physical-gates" / "qwen-fast.json").read_text(encoding="utf-8")
    assert "transcript_sha256" not in persisted
    assert "audio_sha256" not in persisted
    assert "contains_transcript\":false" in persisted
    assert ready_qwen_profiles(state, runtime, models) == ["qwen-fast"]


def test_gate_rejects_gpu_discovery_without_executed_cuda(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    bad = _receipt()
    bad["cuda"]["execution_ready"] = False
    bad["cuda"]["execution_error"] = "QWEN_CUDA_DRIVER_INCOMPATIBLE"

    with pytest.raises(QwenPhysicalGateError, match="QWEN_CUDA_DRIVER_INCOMPATIBLE"):
        record_qwen_physical_gate(state, runtime, models, bad, profile_id="qwen-fast")

    assert ready_qwen_profiles(state, runtime, models) == []


def test_gate_rejects_audio_shorter_than_production_window(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    short = _receipt()
    short["inference"]["audio_seconds"] = MIN_GATE_AUDIO_SECONDS - 0.001

    with pytest.raises(QwenPhysicalGateError, match="QWEN_GATE_AUDIO_TOO_SHORT"):
        record_qwen_physical_gate(state, runtime, models, short, profile_id="qwen-fast")

    gate_path = state / "qwen-physical-gates" / "qwen-fast.json"
    assert not gate_path.exists()
    assert ready_qwen_profiles(state, runtime, models) == []


def test_reader_invalidates_legacy_short_gate(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")
    gate_path = state / "qwen-physical-gates" / "qwen-fast.json"
    persisted = json.loads(gate_path.read_text(encoding="utf-8"))
    persisted["metrics"]["audio_seconds"] = 30.0
    gate_path.write_text(json.dumps(persisted, separators=(",", ":")), encoding="utf-8")

    inspected = inspect_qwen_physical_gate(state, runtime, models, profile_id="qwen-fast")
    assert inspected["ready"] is False
    assert inspected["status"] == "invalid"
    assert inspected["reason"] == "QWEN_GATE_AUDIO_TOO_SHORT"
    assert ready_qwen_profiles(state, runtime, models) == []



def test_gate_fast_path_trusts_receipt_and_full_revalidation_detects_tamper(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")

    worker = runtime / "qwen" / MIN_COMPATIBLE_QWEN_RUNTIME_VERSION / "TDAQwenWorker.exe"
    worker.write_bytes(b"tampered")
    # Normal dispatch does not re-hash bytes, but the metadata seal catches
    # ordinary post-gate file changes immediately.
    lightweight_worker = inspect_qwen_physical_gate(
        state, runtime, models, profile_id="qwen-fast"
    )
    assert lightweight_worker["ready"] is False
    assert lightweight_worker["status"] == "stale"
    assert lightweight_worker["reason"] == "QWEN_GATE_BINDING_CHANGED"

    other = tmp_path / "other"
    state2, runtime2, models2 = _prepared(other)
    record_qwen_physical_gate(state2, runtime2, models2, _receipt(), profile_id="qwen-fast")
    model = model_path(models2, "qwen-fast") / "model.safetensors"
    model.write_bytes(b"changed-after-acceptance")
    lightweight_model = inspect_qwen_physical_gate(
        state2, runtime2, models2, profile_id="qwen-fast"
    )
    assert lightweight_model["ready"] is False
    assert lightweight_model["status"] == "stale"
    assert lightweight_model["reason"] == "QWEN_GATE_BINDING_CHANGED"

def test_v2_gate_reseals_metadata_only_runtime_drift_once(monkeypatch, tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")
    gate_path = state / "qwen-physical-gates" / "qwen-fast.json"
    before_gate = json.loads(gate_path.read_text(encoding="utf-8"))

    worker = runtime / "qwen" / MIN_COMPATIBLE_QWEN_RUNTIME_VERSION / "TDAQwenWorker.exe"
    before = worker.stat()
    os.utime(
        worker,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )

    inspected = inspect_qwen_physical_gate(
        state,
        runtime,
        models,
        profile_id="qwen-fast",
    )

    assert inspected["ready"] is True
    resealed = json.loads(gate_path.read_text(encoding="utf-8"))
    assert (
        resealed["runtime"]["worker_metadata_sha256"]
        != before_gate["runtime"]["worker_metadata_sha256"]
    )
    assert resealed["runtime"]["worker_sha256"] == before_gate["runtime"]["worker_sha256"]

    monkeypatch.setattr(
        qwen_runtime_module,
        "_sha256_file",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("resealed Qwen gate must return to metadata-only fast path")
        ),
    )
    second = inspect_qwen_physical_gate(
        state,
        runtime,
        models,
        profile_id="qwen-fast",
    )
    assert second["ready"] is True


def test_v2_gate_model_metadata_drift_does_not_rehash_unchanged_components(
    monkeypatch,
    tmp_path: Path,
):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")

    model = model_path(models, "qwen-fast") / "model.safetensors"
    before = model.stat()
    os.utime(
        model,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )

    monkeypatch.setattr(
        qwen_runtime_module,
        "_sha256_file",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("unchanged runtime must not be deep-hashed for model drift")
        ),
    )
    monkeypatch.setattr(
        gate_module,
        "verify_and_upgrade_model_install",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model already verified by metadata-drift inspection")
        ),
    )

    inspected = inspect_qwen_physical_gate(
        state,
        runtime,
        models,
        profile_id="qwen-fast",
    )

    assert inspected["ready"] is True


def test_deep_verification_detects_same_metadata_worker_tamper(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")

    worker = runtime / "qwen" / MIN_COMPATIBLE_QWEN_RUNTIME_VERSION / "TDAQwenWorker.exe"
    before = worker.stat()
    original = worker.read_bytes()
    replacement = b"x" * len(original)
    assert replacement != original
    worker.write_bytes(replacement)
    os.utime(worker, ns=(before.st_atime_ns, before.st_mtime_ns))

    # The fast path intentionally does not promise cryptographic detection when
    # an attacker preserves every sealed metadata field.
    assert inspect_qwen_physical_gate(
        state, runtime, models, profile_id="qwen-fast"
    )["ready"] is True

    deep = inspect_qwen_physical_gate(
        state,
        runtime,
        models,
        profile_id="qwen-fast",
        verify_model_content=True,
    )
    assert deep["ready"] is False
    assert deep["status"] == "stale"
    assert deep["reason"] == "QWEN_GATE_RUNTIME_NOT_READY"


def test_gate_is_per_profile_and_rejects_receipts_with_private_payload(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")
    assert inspect_qwen_physical_gate(state, runtime, models, profile_id="qwen-quality")["ready"] is False

    bad = _receipt()
    bad["text"] = "conteúdo que não pode entrar no receipt de gate"
    with pytest.raises(QwenPhysicalGateError, match="QWEN_GATE_PRIVATE_PAYLOAD_REJECTED"):
        record_qwen_physical_gate(state, runtime, models, bad, profile_id="qwen-fast")


def test_gate_default_accepts_any_supported_cuda_gpu(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    receipt = _receipt()
    receipt["gpu"]["name"] = "NVIDIA GeForce RTX 3090"
    receipt["alignment_gpu"]["name"] = "NVIDIA GeForce RTX 3090"
    receipt["cuda"]["devices"][0]["name"] = "NVIDIA GeForce RTX 3090"
    receipt["cuda"]["devices"][0]["compute_capability"] = "8.6"

    gate = record_qwen_physical_gate(
        state,
        runtime,
        models,
        receipt,
        profile_id="qwen-fast",
    )
    assert gate["ready"] is True
    assert gate["gpu"]["name"] == "NVIDIA GeForce RTX 3090"



def test_gate_accepts_turing_sm75_and_rejects_older_cuda_capability(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    turing = _receipt()
    turing["gpu"]["name"] = "NVIDIA GeForce RTX 2080 SUPER"
    turing["alignment_gpu"]["name"] = "NVIDIA GeForce RTX 2080 SUPER"
    turing["cuda"]["devices"][0]["name"] = "NVIDIA GeForce RTX 2080 SUPER"
    turing["cuda"]["devices"][0]["compute_capability"] = "7.5"

    gate = record_qwen_physical_gate(
        state,
        runtime,
        models,
        turing,
        profile_id="qwen-fast",
    )
    assert gate["ready"] is True
    assert gate["gpu"]["compute_capability"] == "7.5"

    older = _receipt()
    older["cuda"]["devices"][0]["compute_capability"] = "7.0"
    with pytest.raises(QwenPhysicalGateError, match="QWEN_GATE_GPU_UNSUPPORTED"):
        record_qwen_physical_gate(
            state,
            runtime,
            models,
            older,
            profile_id="qwen-fast",
        )

def test_gate_can_still_require_an_explicit_gpu_name(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    bad = _receipt()
    bad["gpu"]["name"] = "NVIDIA GeForce RTX 3090"
    with pytest.raises(QwenPhysicalGateError, match="QWEN_GATE_GPU_NAME_MISMATCH"):
        record_qwen_physical_gate(
            state,
            runtime,
            models,
            bad,
            profile_id="qwen-fast",
            required_gpu_name="RTX 4070",
        )


def test_legacy_gate_is_deep_verified_once_and_upgrades_model_markers(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")
    gate_path = state / "qwen-physical-gates" / "qwen-fast.json"
    persisted = json.loads(gate_path.read_text(encoding="utf-8"))

    persisted["schema"] = "tda_qwen_physical_gate_v1"
    for section, metadata_key in (
        ("runtime", "worker_metadata_sha256"),
        ("model", "metadata_sha256"),
        ("aligner", "metadata_sha256"),
    ):
        persisted[section].pop(metadata_key)
    legacy_binding = {
        "schema": "tda_qwen_physical_gate_v1",
        "profile_id": "qwen-fast",
        "runtime": persisted["runtime"],
        "model": persisted["model"],
        "aligner": persisted["aligner"],
    }
    persisted["binding_sha256"] = hashlib.sha256(
        json.dumps(
            legacy_binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    gate_path.write_text(json.dumps(persisted, separators=(",", ":")), encoding="utf-8")

    for profile in (get_profile("qwen-fast"), ALIGNER_PROFILE):
        marker_path = model_path(models, profile) / MODEL_MARKER
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.pop("metadata_sha256")
        marker_path.write_text(json.dumps(marker, separators=(",", ":")), encoding="utf-8")

    migrated = inspect_qwen_physical_gate(state, runtime, models, profile_id="qwen-fast")
    assert migrated["ready"] is True
    upgraded = json.loads(gate_path.read_text(encoding="utf-8"))
    assert upgraded["schema"] == "tda_qwen_physical_gate_v2"
    assert "worker_metadata_sha256" in upgraded["runtime"]
    assert "metadata_sha256" in upgraded["model"]
    assert "metadata_sha256" in upgraded["aligner"]
    for profile in (get_profile("qwen-fast"), ALIGNER_PROFILE):
        marker = json.loads(
            (model_path(models, profile) / MODEL_MARKER).read_text(encoding="utf-8")
        )
        assert len(marker["metadata_sha256"]) == 64


def test_legacy_gate_refuses_to_promote_tampered_legacy_model(tmp_path: Path):
    state, runtime, models = _prepared(tmp_path)
    record_qwen_physical_gate(state, runtime, models, _receipt(), profile_id="qwen-fast")
    gate_path = state / "qwen-physical-gates" / "qwen-fast.json"
    persisted = json.loads(gate_path.read_text(encoding="utf-8"))

    persisted["schema"] = "tda_qwen_physical_gate_v1"
    for section, metadata_key in (
        ("runtime", "worker_metadata_sha256"),
        ("model", "metadata_sha256"),
        ("aligner", "metadata_sha256"),
    ):
        persisted[section].pop(metadata_key)
    legacy_binding = {
        "schema": "tda_qwen_physical_gate_v1",
        "profile_id": "qwen-fast",
        "runtime": persisted["runtime"],
        "model": persisted["model"],
        "aligner": persisted["aligner"],
    }
    persisted["binding_sha256"] = hashlib.sha256(
        json.dumps(
            legacy_binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    gate_path.write_text(json.dumps(persisted, separators=(",", ":")), encoding="utf-8")

    marker_path = model_path(models, "qwen-fast") / MODEL_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop("metadata_sha256")
    marker_path.write_text(json.dumps(marker, separators=(",", ":")), encoding="utf-8")
    model_path_file = model_path(models, "qwen-fast") / "model.safetensors"
    model_path_file.write_bytes(model_path_file.read_bytes() + b"-tampered-before-upgrade")

    inspected = inspect_qwen_physical_gate(
        state,
        runtime,
        models,
        profile_id="qwen-fast",
    )

    assert inspected["ready"] is False
    assert inspected["status"] == "stale"
    assert inspected["reason"] == "QWEN_GATE_BINDING_CHANGED"
    after = json.loads(gate_path.read_text(encoding="utf-8"))
    assert after["schema"] == "tda_qwen_physical_gate_v1"
    marker_after = json.loads(marker_path.read_text(encoding="utf-8"))
    assert "metadata_sha256" not in marker_after
