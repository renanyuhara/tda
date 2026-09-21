from __future__ import annotations

import re

MIN_COMPATIBLE_WHISPER_RUNTIME_VERSION = "1.1.4"
MIN_COMPATIBLE_QWEN_RUNTIME_VERSION = "1.0.9"
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def version_tuple(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise ValueError("RUNTIME_VERSION_INVALID")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def runtime_version_compatible(value: str, minimum: str) -> bool:
    try:
        return version_tuple(value) >= version_tuple(minimum)
    except ValueError:
        return False


def whisper_runtime_version_compatible(value: str) -> bool:
    return runtime_version_compatible(value, MIN_COMPATIBLE_WHISPER_RUNTIME_VERSION)


def qwen_runtime_version_compatible(value: str) -> bool:
    return runtime_version_compatible(value, MIN_COMPATIBLE_QWEN_RUNTIME_VERSION)
