from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .asr_runtime import inspect_whisper_runtime, install_whisper_runtime_archive
from .qwen_runtime import inspect_qwen_runtime, install_qwen_runtime_archive
from .qwen_runtime_bundle import assemble_qwen_runtime_bundle, parse_qwen_runtime_bundle_manifest
from .runtime_release_evidence import CANDIDATE_SCHEMA, verify_candidate_assets

RC_WHISPER_VERSION = "1.1.4"
RC_QWEN_VERSION = "1.0.9"
_ACTIONS_ARTIFACT_MAX_ENTRIES = 128
_ACTIONS_ARTIFACT_MAX_UNCOMPRESSED_BYTES = 8 * 1024**3
_COPY_CHUNK = 1024 * 1024
_WHISPER_ARCHIVE = re.compile(r"^TDAWhisperRuntime-(\d+\.\d+\.\d+)-windows-x64\.zip$")
_QWEN_BUNDLE = re.compile(r"^TDAQwenRuntimeBundle-(\d+\.\d+\.\d+)-windows-x64\.json$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class RcRuntimeArtifactError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_actions_artifact(artifact: Path, expected_sha256: str) -> Path:
    source = artifact.resolve()
    if not source.is_file():
        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_NOT_FOUND")
    if not _SHA256.fullmatch(expected_sha256):
        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_HASH_INVALID")
    try:
        actual = _sha256_file(source)
    except OSError as exc:
        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_READ_FAILED") from exc
    if actual != expected_sha256:
        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_HASH_MISMATCH")
    return source


def _safe_member(info: zipfile.ZipInfo) -> PurePosixPath:
    value = info.filename.replace("\\", "/")
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_PATH_INVALID")
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and stat.S_ISLNK(mode):
        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_SYMLINK")
    return path


def _extract_actions_artifact(artifact: Path, target: Path) -> Path:
    try:
        package = zipfile.ZipFile(artifact, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_INVALID") from exc
    with package:
        infos = [item for item in package.infolist() if not item.is_dir()]
        if not infos or len(infos) > _ACTIONS_ARTIFACT_MAX_ENTRIES:
            raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_ENTRY_LIMIT")
        seen: set[str] = set()
        total = 0
        for info in infos:
            relative = _safe_member(info)
            key = relative.as_posix().casefold()
            if key in seen:
                raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_DUPLICATE")
            seen.add(key)
            if info.file_size < 0:
                raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_SIZE_INVALID")
            total += info.file_size
            if total > _ACTIONS_ARTIFACT_MAX_UNCOMPRESSED_BYTES:
                raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_SIZE_LIMIT")
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with package.open(info, "r") as input_handle, destination.open("xb") as output_handle:
                while True:
                    chunk = input_handle.read(_COPY_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size:
                        raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_SIZE_MISMATCH")
                    output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            if written != info.file_size:
                raise RcRuntimeArtifactError("RC_RUNTIME_ARTIFACT_SIZE_MISMATCH")
    return target


def _one(root: Path, pattern: str, code: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise RcRuntimeArtifactError(code)
    return matches[0]


def _install_whisper(root: Path, runtime_root: Path) -> dict[str, object]:
    archive = _one(root, "TDAWhisperRuntime-*-windows-x64.zip", "RC_WHISPER_ARCHIVE_AMBIGUOUS")
    match = _WHISPER_ARCHIVE.fullmatch(archive.name)
    if not match:
        raise RcRuntimeArtifactError("RC_WHISPER_ARCHIVE_NAME_INVALID")
    version = match.group(1)
    if version != RC_WHISPER_VERSION:
        raise RcRuntimeArtifactError("RC_WHISPER_VERSION_MISMATCH")
    digest_file = archive.with_name(archive.name + ".sha256")
    try:
        digest_line = digest_file.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RcRuntimeArtifactError("RC_WHISPER_DIGEST_MISSING") from exc
    parts = digest_line.split()
    if len(parts) != 2 or parts[1] != archive.name or not _SHA256.fullmatch(parts[0]):
        raise RcRuntimeArtifactError("RC_WHISPER_DIGEST_INVALID")
    digest = parts[0]
    if _sha256_file(archive) != digest:
        raise RcRuntimeArtifactError("RC_WHISPER_DIGEST_MISMATCH")

    state = inspect_whisper_runtime(runtime_root, verify_worker=True)
    if state.get("status") == "ready" and state.get("version") == version:
        return {"runtime": "whisper", "version": version, "status": "ready", "reused": True}
    target = runtime_root / "whisper" / version
    repairing = target.exists() or target.is_symlink()
    try:
        marker = install_whisper_runtime_archive(
            archive,
            runtime_root,
            version=version,
            expected_sha256=digest,
            replace_corrupt=repairing,
        )
    except RuntimeError as exc:
        raise RcRuntimeArtifactError(str(exc) or "RC_WHISPER_INSTALL_FAILED") from exc
    verified = inspect_whisper_runtime(runtime_root, verify_worker=True)
    if verified.get("status") != "ready" or verified.get("version") != version:
        raise RcRuntimeArtifactError("RC_WHISPER_INSTALL_VERIFY_FAILED")
    return {
        "runtime": "whisper",
        "version": version,
        "status": "ready",
        "reused": False,
        "repaired": repairing,
        "worker_sha256": marker["worker_sha256"],
        "archive_sha256": marker["archive_sha256"],
    }


def _install_qwen(root: Path, runtime_root: Path, cache_root: Path) -> dict[str, object]:
    manifest_path = _one(root, "TDAQwenRuntimeBundle-*-windows-x64.json", "RC_QWEN_BUNDLE_AMBIGUOUS")
    match = _QWEN_BUNDLE.fullmatch(manifest_path.name)
    if not match:
        raise RcRuntimeArtifactError("RC_QWEN_BUNDLE_NAME_INVALID")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = parse_qwen_runtime_bundle_manifest(value)
    except Exception as exc:
        raise RcRuntimeArtifactError("RC_QWEN_BUNDLE_INVALID") from exc
    version = match.group(1)
    if version != RC_QWEN_VERSION:
        raise RcRuntimeArtifactError("RC_QWEN_VERSION_MISMATCH")
    if manifest.version != version or manifest.runtime_id != "qwen3-transformers":
        raise RcRuntimeArtifactError("RC_QWEN_BUNDLE_IDENTITY_MISMATCH")

    state = inspect_qwen_runtime(runtime_root, verify_worker=True)
    if state.get("status") == "ready" and state.get("version") == version:
        return {"runtime": "qwen", "version": version, "status": "ready", "reused": True}

    assembly = cache_root.resolve() / "physical-setup" / "qwen" / version
    shutil.rmtree(assembly, ignore_errors=True)
    assembly.mkdir(parents=True, exist_ok=False)
    target = runtime_root / "qwen" / version
    repairing = target.exists() or target.is_symlink()
    try:
        archive = assemble_qwen_runtime_bundle(manifest, manifest_path.parent, assembly)
        marker = install_qwen_runtime_archive(
            archive,
            runtime_root,
            version=version,
            expected_sha256=manifest.archive_sha256,
            replace_corrupt=repairing,
        )
    except RuntimeError as exc:
        raise RcRuntimeArtifactError(str(exc) or "RC_QWEN_INSTALL_FAILED") from exc
    finally:
        shutil.rmtree(assembly, ignore_errors=True)
    verified = inspect_qwen_runtime(runtime_root, verify_worker=True)
    if verified.get("status") != "ready" or verified.get("version") != version:
        raise RcRuntimeArtifactError("RC_QWEN_INSTALL_VERIFY_FAILED")
    return {
        "runtime": "qwen",
        "version": version,
        "status": "ready",
        "reused": False,
        "repaired": repairing,
        "worker_sha256": marker["worker_sha256"],
        "archive_sha256": marker["archive_sha256"],
        "part_count": len(manifest.parts),
    }


def install_rc_runtime_artifact(
    family: str,
    artifact: Path,
    *,
    expected_artifact_sha256: str,
    runtime_root: Path,
    cache_root: Path,
) -> dict[str, object]:
    """Legacy physical setup from an exact GitHub Actions artifact archive."""
    if family not in {"whisper", "qwen"}:
        raise RcRuntimeArtifactError("RC_RUNTIME_FAMILY_INVALID")
    source = _verify_actions_artifact(artifact, expected_artifact_sha256)
    runtime_root.resolve().mkdir(parents=True, exist_ok=True)
    cache_root.resolve().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"tda-{family}-artifact-", dir=cache_root.resolve()) as value:
        extracted = _extract_actions_artifact(source, Path(value))
        if family == "whisper":
            return _install_whisper(extracted, runtime_root.resolve())
        return _install_qwen(extracted, runtime_root.resolve(), cache_root.resolve())


def install_runtime_candidate(
    candidate_manifest: Path,
    assets_root: Path,
    *,
    runtime_root: Path,
    cache_root: Path,
) -> dict[str, object]:
    """Install the exact files attached to a formal runtime RC release."""
    try:
        candidate = json.loads(candidate_manifest.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RcRuntimeArtifactError("RC_RUNTIME_CANDIDATE_INVALID") from exc
    if not isinstance(candidate, dict) or candidate.get("schema") != CANDIDATE_SCHEMA:
        raise RcRuntimeArtifactError("RC_RUNTIME_CANDIDATE_INVALID")
    try:
        verify_candidate_assets(candidate, assets_root.resolve())
    except RuntimeError as exc:
        raise RcRuntimeArtifactError(str(exc) or "RC_RUNTIME_CANDIDATE_ASSETS_INVALID") from exc
    family = candidate.get("family")
    version = candidate.get("version")
    if family == "whisper":
        if version != RC_WHISPER_VERSION:
            raise RcRuntimeArtifactError("RC_WHISPER_VERSION_MISMATCH")
    elif family == "qwen":
        if version != RC_QWEN_VERSION:
            raise RcRuntimeArtifactError("RC_QWEN_VERSION_MISMATCH")
    else:
        raise RcRuntimeArtifactError("RC_RUNTIME_FAMILY_INVALID")
    runtime_root.resolve().mkdir(parents=True, exist_ok=True)
    cache_root.resolve().mkdir(parents=True, exist_ok=True)
    if family == "whisper":
        result = _install_whisper(assets_root.resolve(), runtime_root.resolve())
    else:
        result = _install_qwen(assets_root.resolve(), runtime_root.resolve(), cache_root.resolve())
    archive_sha = result.get("archive_sha256")
    if result.get("reused") is not True and archive_sha != candidate.get("runtime_archive_sha256"):
        raise RcRuntimeArtifactError("RC_RUNTIME_INSTALLED_ARCHIVE_MISMATCH")
    if result.get("reused") is True:
        family_root = runtime_root.resolve() / family / str(version)
        try:
            marker = json.loads((family_root / ".tda-runtime.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RcRuntimeArtifactError("RC_RUNTIME_INSTALLED_MARKER_INVALID") from exc
        if not isinstance(marker, dict) or marker.get("archive_sha256") != candidate.get("runtime_archive_sha256"):
            raise RcRuntimeArtifactError("RC_RUNTIME_INSTALLED_ARCHIVE_MISMATCH")
    return {**result, "candidate_tag": candidate["candidate_tag"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rc-runtime-artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("install-candidate")
    install.add_argument("--candidate-manifest", type=Path, required=True)
    install.add_argument("--assets-root", type=Path, required=True)
    install.add_argument("--runtime-root", type=Path, required=True)
    install.add_argument("--cache-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = install_runtime_candidate(
        args.candidate_manifest,
        args.assets_root,
        runtime_root=args.runtime_root,
        cache_root=args.cache_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
