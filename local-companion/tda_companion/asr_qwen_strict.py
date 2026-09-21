from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .asr_checkpoints import build_checkpoint_signature, load_track_checkpoint, save_track_checkpoint
from .asr_models import QWEN_FORCED_ALIGNER_MODEL_ID, get_profile
from .asr_qwen import (
    QWEN_MAX_NEW_TOKENS,
    QWEN_SAMPLE_RATE,
    QWEN_SEGMENT_GAP_SECONDS,
    QWEN_SEGMENT_MAX_SECONDS,
    QWEN_WINDOW_SECONDS,
    AlignerSession,
    AsrSession,
    AudioWindow,
    CancelCallback,
    EnergyReader,
    ProgressCallback,
    QwenRuntimeError,
    QwenWindowTranscript,
    _bounded_prompt,
    _default_aligner_session,
    _default_asr_session,
    _fallback_segment,
    _prepare_aligner,
    _prepare_model,
    _resolve_plan,
    _runtime_fingerprint,
    _safe_track_path,
    _segments_from_words,
    _validated_words,
    _window_energy_db,
)
from .asr_timeline import build_turns, deduplicate_cross_track_segments, flatten_tracks
from .craig import CraigPackage
from .qwen_acceptance import QwenPlan
from .transcript import TranscriptDocument, TranscriptEngine, TranscriptSegment, TranscriptTrack, TranscriptWord, stats_for_tracks

QWEN_WINDOW_OVERLAP_SECONDS = 6.0
QWEN_WINDOW_STRIDE_SECONDS = QWEN_WINDOW_SECONDS - QWEN_WINDOW_OVERLAP_SECONDS
QWEN_ALIGNMENT_RECOVERY_PARTS = (2, 4)
_RECOVERABLE_ALIGNMENT_CODES = {
    "QWEN_ALIGNMENT_EMPTY",
    "QWEN_ALIGNMENT_TIMESTAMPS_INVALID",
}
_RECOVERABLE_ASR_CODES = {"QWEN_ASR_EMPTY_TRANSCRIPT"}


def iter_audio_windows_overlap(
    path: Path,
    *,
    window_seconds: float = QWEN_WINDOW_SECONDS,
    overlap_seconds: float = QWEN_WINDOW_OVERLAP_SECONDS,
    sample_rate: int = QWEN_SAMPLE_RATE,
) -> Iterable[AudioWindow]:
    """Decode bounded overlapping windows without materializing the full track.

    A small overlap protects words crossing each bounded ASR window. Alignment
    later assigns each word to exactly one ownership interval, so overlap does not
    become duplicated transcript content.
    """
    if (
        window_seconds <= 0
        or window_seconds > 240
        or overlap_seconds <= 0
        or overlap_seconds >= window_seconds / 2
        or sample_rate != QWEN_SAMPLE_RATE
    ):
        raise QwenRuntimeError("QWEN_WINDOW_CONFIG_INVALID")
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise QwenRuntimeError("QWEN_RUNTIME_NOT_INSTALLED") from exc

    window_samples = int(round(window_seconds * sample_rate))
    overlap_samples = int(round(overlap_seconds * sample_rate))
    stride_samples = window_samples - overlap_samples
    pieces: list[Any] = []
    buffered = 0
    start_sample = 0
    index = 0

    def compact() -> Any:
        nonlocal pieces, buffered
        if not pieces:
            return np.empty((0,), dtype=np.float32)
        joined = np.concatenate(pieces).astype(np.float32, copy=False)
        pieces = [joined]
        buffered = int(joined.size)
        return joined

    def drain_full() -> list[AudioWindow]:
        nonlocal pieces, buffered, start_sample, index
        output: list[AudioWindow] = []
        while buffered >= window_samples:
            joined = compact()
            chunk = joined[:window_samples].copy()
            index += 1
            start = start_sample / sample_rate
            output.append(
                AudioWindow(index=index, start=start, end=start + chunk.size / sample_rate, audio=chunk)
            )
            remainder = joined[stride_samples:].copy()
            start_sample += stride_samples
            pieces = [remainder] if remainder.size else []
            buffered = int(remainder.size)
        return output

    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise QwenRuntimeError("QWEN_AUDIO_STREAM_MISSING")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    array = converted.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                    if array.size:
                        pieces.append(array.copy())
                        buffered += int(array.size)
                    yield from drain_full()
            for converted in resampler.resample(None):
                array = converted.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                if array.size:
                    pieces.append(array.copy())
                    buffered += int(array.size)
                yield from drain_full()
            if buffered:
                joined = compact()
                # A trailing buffer can contain only the overlap already emitted by
                # the previous full window. Do not transcribe that overlap twice.
                if index == 0 or buffered > overlap_samples:
                    index += 1
                    start = start_sample / sample_rate
                    yield AudioWindow(
                        index=index,
                        start=start,
                        end=start + joined.size / sample_rate,
                        audio=joined.copy(),
                    )
    except QwenRuntimeError:
        raise
    except Exception as exc:
        raise QwenRuntimeError("QWEN_AUDIO_DECODE_FAILED") from exc

    if index == 0:
        raise QwenRuntimeError("QWEN_AUDIO_EMPTY")


def _owned_words(
    words: list[TranscriptWord],
    window: AudioWindow,
    *,
    first: bool,
    last: bool,
    overlap_seconds: float = QWEN_WINDOW_OVERLAP_SECONDS,
) -> list[TranscriptWord]:
    half = overlap_seconds / 2.0
    left = window.start if first else window.start + half
    right = window.end if last else window.end - half
    return [
        word
        for word in words
        if left <= (word.start + word.end) / 2.0 < right or (last and math.isclose((word.start + word.end) / 2.0, right))
    ]


def _copy_audio_slice(audio: Any, start: int, end: int) -> Any:
    value = audio[start:end]
    copier = getattr(value, "copy", None)
    return copier() if callable(copier) else value


def _recovery_overlap_seconds(window: AudioWindow, parts: int) -> float:
    duration = window.end - window.start
    if duration <= 0 or parts < 2:
        raise QwenRuntimeError("QWEN_WINDOW_CONFIG_INVALID")
    core_seconds = duration / parts
    return min(QWEN_WINDOW_OVERLAP_SECONDS, max(0.1, core_seconds * 0.4))


def _recovery_windows(window: AudioWindow, *, parts: int) -> tuple[AudioWindow, ...]:
    """Split one failed window into overlapping subwindows without dropping audio.

    Each child owns an equal core interval. Half of the recovery overlap is added
    on either side of internal boundaries, allowing ownership filtering to keep
    words that cross a split point exactly once.
    """
    duration = window.end - window.start
    total_samples = len(window.audio)
    if duration <= 0 or total_samples <= 0 or parts < 2:
        raise QwenRuntimeError("QWEN_WINDOW_CONFIG_INVALID")

    overlap_seconds = _recovery_overlap_seconds(window, parts)
    samples_per_second = total_samples / duration
    half_overlap_samples = int(round((overlap_seconds / 2.0) * samples_per_second))

    values: list[AudioWindow] = []
    for position in range(parts):
        core_start = int(round(total_samples * position / parts))
        core_end = int(round(total_samples * (position + 1) / parts))
        sample_start = core_start if position == 0 else max(0, core_start - half_overlap_samples)
        sample_end = (
            core_end
            if position == parts - 1
            else min(total_samples, core_end + half_overlap_samples)
        )
        child_start = window.start + duration * (sample_start / total_samples)
        child_end = window.start + duration * (sample_end / total_samples)
        values.append(
            AudioWindow(
                index=position + 1,
                start=child_start,
                end=child_end,
                audio=_copy_audio_slice(window.audio, sample_start, sample_end),
            )
        )
    return tuple(values)


def _is_recoverable_alignment_failure(exc: QwenRuntimeError) -> bool:
    current: BaseException | None = exc
    while isinstance(current, QwenRuntimeError):
        if current.code in _RECOVERABLE_ALIGNMENT_CODES:
            return True
        current = current.__cause__
    return False


def _recovery_alignment_segments(
    track_number: int,
    parent: AudioWindow,
    pending_windows: tuple[tuple[AudioWindow, QwenWindowTranscript], ...],
    aligner: AlignerSession,
    *,
    first: bool,
    last: bool,
    overlap_seconds: float,
) -> tuple[TranscriptSegment, ...]:
    recovered_words: list[TranscriptWord] = []
    final_index = len(pending_windows) - 1
    for position, (window, pending) in enumerate(pending_windows):
        if not pending.text.strip():
            continue
        aligned = aligner.align(window.audio, pending.text, pending.language)
        words = _validated_words(
            aligned,
            window,
            discard_trailing_overflow_from=(
                None if position == final_index else window.end - overlap_seconds / 2.0
            ),
        )
        recovered_words.extend(
            _owned_words(
                words,
                window,
                first=position == 0,
                last=position == final_index,
                overlap_seconds=overlap_seconds,
            )
        )

    if not recovered_words:
        raise QwenRuntimeError("QWEN_ALIGNMENT_EMPTY")

    recovered_words.sort(key=lambda item: (item.start, item.end, item.text))
    previous_end = parent.start
    for word_index, word in enumerate(recovered_words, start=1):
        if word.start + 0.05 < previous_end:
            raise QwenRuntimeError(
                "QWEN_ALIGNMENT_TIMESTAMPS_INVALID",
                details={
                    "timestamp_failure": "recovery_temporal_regression",
                    "word_index": word_index,
                    "relative_start": round(word.start - parent.start, 3),
                    "relative_end": round(word.end - parent.start, 3),
                    "previous_end_relative": round(previous_end - parent.start, 3),
                },
            )
        previous_end = max(previous_end, word.end)

    owned = _owned_words(recovered_words, parent, first=first, last=last)
    if not owned:
        if recovered_words and (not first or not last):
            return ()
        raise QwenRuntimeError("QWEN_ALIGNMENT_EMPTY")

    segments = _segments_from_words(track_number, parent, owned)
    if not segments:
        raise QwenRuntimeError("QWEN_ALIGNMENT_EMPTY")
    return segments


def _recover_alignment_window(
    track_number: int,
    window: AudioWindow,
    *,
    first: bool,
    last: bool,
    prompt: str,
    model_root: Path,
    aligner_root: Path,
    plan: QwenPlan,
    asr_session_factory: Any,
    aligner_session_factory: Any,
    is_cancelled: CancelCallback,
) -> tuple[tuple[TranscriptSegment, ...] | None, AlignerSession | None, int | None]:
    """Retry a content-local alignment failure using progressively smaller audio.

    Recovery never discards the original audio. If both bounded subdivision
    attempts remain unalignable, the caller can preserve the original ASR text
    as an explicitly warned coarse fallback segment.
    """
    for parts in QWEN_ALIGNMENT_RECOVERY_PARTS:
        if is_cancelled():
            raise QwenRuntimeError("ASR_CANCELLED")
        recovery_windows = _recovery_windows(window, parts=parts)
        overlap_seconds = _recovery_overlap_seconds(window, parts)

        asr_session: AsrSession | None = None
        try:
            asr_session = asr_session_factory(model_root, plan)
            pending_values: list[tuple[AudioWindow, QwenWindowTranscript]] = []
            for child in recovery_windows:
                if is_cancelled():
                    raise QwenRuntimeError("ASR_CANCELLED")
                text, language = asr_session.transcribe(child.audio, prompt=prompt)
                pending_values.append(
                    (
                        child,
                        QwenWindowTranscript(
                            index=child.index,
                            start=child.start,
                            end=child.end,
                            text=text.strip(),
                            language=language or "Portuguese",
                        ),
                    )
                )
        except QwenRuntimeError as exc:
            if exc.code in _RECOVERABLE_ASR_CODES:
                continue
            raise
        finally:
            if asr_session is not None:
                asr_session.close()

        recovery_aligner: AlignerSession | None = None
        try:
            recovery_aligner = aligner_session_factory(aligner_root, plan)
            segments = _recovery_alignment_segments(
                track_number,
                window,
                tuple(pending_values),
                recovery_aligner,
                first=first,
                last=last,
                overlap_seconds=overlap_seconds,
            )
        except QwenRuntimeError as exc:
            if recovery_aligner is not None:
                recovery_aligner.close()
            if _is_recoverable_alignment_failure(exc):
                continue
            raise
        return segments, recovery_aligner, parts

    return None, None, None


def _strict_alignment_segments(
    track_number: int,
    window: AudioWindow,
    pending: QwenWindowTranscript,
    aligner: AlignerSession,
    *,
    first: bool,
    last: bool,
) -> tuple[TranscriptSegment, ...]:
    # Silence is a valid ASR outcome. Forced alignment is mandatory for actual
    # transcript text, but an empty/whitespace-only window has nothing to align
    # and must contribute zero segments instead of failing the whole session.
    if not pending.text.strip():
        return ()
    try:
        aligned = aligner.align(window.audio, pending.text, pending.language)
        discard_trailing_overflow_from = (
            None if last else window.end - QWEN_WINDOW_OVERLAP_SECONDS / 2.0
        )
        words = _validated_words(
            aligned,
            window,
            discard_trailing_overflow_from=discard_trailing_overflow_from,
        )
    except QwenRuntimeError as exc:
        raise QwenRuntimeError("QWEN_ALIGNMENT_REQUIRED") from exc
    owned = _owned_words(words, window, first=first, last=last)
    if not owned:
        # It is valid for an overlap-only window to contribute no owned words only
        # when the aligner returned words entirely in the neighbor-owned overlap.
        if words and (not first or not last):
            return ()
        raise QwenRuntimeError("QWEN_ALIGNMENT_REQUIRED")
    segments = _segments_from_words(track_number, window, owned)
    if not segments:
        raise QwenRuntimeError("QWEN_ALIGNMENT_REQUIRED")
    return segments


def transcribe_craig_package_qwen_strict(
    package: CraigPackage,
    package_root: Path,
    models_root: Path,
    *,
    profile_id: str,
    glossary: str = "",
    context: str = "",
    report: ProgressCallback | None = None,
    is_cancelled: CancelCallback | None = None,
    checkpoints: bool = True,
    plan_resolver=_resolve_plan,
    model_prepare=_prepare_model,
    aligner_prepare=_prepare_aligner,
    asr_session_factory: Any = _default_asr_session,
    aligner_session_factory: Any = _default_aligner_session,
    window_reader=iter_audio_windows_overlap,
    energy_reader: EnergyReader = _window_energy_db,
) -> TranscriptDocument:
    profile = get_profile(profile_id)
    if profile.engine != "qwen3":
        raise QwenRuntimeError("QWEN_PROFILE_REQUIRED")
    report = report or (lambda _: None)
    is_cancelled = is_cancelled or (lambda: False)
    if is_cancelled():
        raise QwenRuntimeError("ASR_CANCELLED")
    report(
        {
            "type": "stage",
            "stage": "runtime_validation",
            "profile": profile.id,
        }
    )
    plan: QwenPlan = plan_resolver(profile.id)
    report({"type": "event", "code": "QWEN_RUNTIME_PLAN_READY", "stage": "runtime_validation", "device": plan.device, "dtype": plan.dtype})
    prompt = _bounded_prompt(context, glossary)
    recipe = {
        "window_seconds": QWEN_WINDOW_SECONDS,
        "window_overlap_seconds": QWEN_WINDOW_OVERLAP_SECONDS,
        "sample_rate": QWEN_SAMPLE_RATE,
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "segment_gap_seconds": QWEN_SEGMENT_GAP_SECONDS,
        "segment_max_seconds": QWEN_SEGMENT_MAX_SECONDS,
        "device": plan.device,
        "dtype": plan.dtype,
        "alignment": QWEN_FORCED_ALIGNER_MODEL_ID,
        "alignment_policy": "strict-overlap-v3-recovery",
    }
    report({"type": "event", "code": "QWEN_CHECKPOINT_SIGNATURE_STARTED", "stage": "runtime_validation"})
    signature = build_checkpoint_signature(
        package,
        profile,
        recipe=recipe,
        context=" ".join(context.split())[:2000].strip(),
        glossary=" ".join(glossary.split())[:2000].strip(),
        runtime_fingerprint=_runtime_fingerprint(),
    )
    report({"type": "event", "code": "QWEN_CHECKPOINT_SIGNATURE_READY", "stage": "runtime_validation"})
    # Checkpoint discovery may need to parse large JSON artifacts from previous
    # attempts. Publish a distinct stage before any filesystem work so the
    # Companion UI and persisted job state do not misleadingly remain on
    # runtime_validation while the GPU is intentionally idle.
    report({"type": "stage", "stage": "checkpoint_scan", "profile": profile.id})
    report({"type": "event", "code": "QWEN_CHECKPOINT_SCAN_STARTED", "stage": "checkpoint_scan", "total_tracks": len(package.tracks)})

    cached_tracks: dict[int, TranscriptTrack] = {}
    pending_tracks = []
    total_tracks = len(package.tracks)
    completed_tracks = 0
    for track in package.tracks:
        if is_cancelled():
            raise QwenRuntimeError("ASR_CANCELLED")
        _safe_track_path(package_root, track)
        cached = load_track_checkpoint(package_root, signature, track) if checkpoints else None
        if cached is not None and not any(segment.id.endswith("-fallback") for segment in cached.segments):
            cached_tracks[track.number] = cached
            report(
                {
                    "type": "event",
                    "code": "ASR_CHECKPOINT_REUSED",
                    "stage": "checkpoint_scan",
                    "track": track.number,
                    "total_tracks": total_tracks,
                    "speaker": track.speaker,
                }
            )
            completed_tracks += 1
            report(
                {
                    "type": "progress",
                    "completed": completed_tracks,
                    "total": total_tracks,
                    "unit": "tracks",
                    "stage": "source_validation",
                }
            )
        else:
            pending_tracks.append(track)

    report({"type": "event", "code": "QWEN_CHECKPOINT_SCAN_READY", "stage": "checkpoint_scan", "cached_tracks": len(cached_tracks), "pending_tracks": len(pending_tracks)})
    started = time.monotonic()
    pending_text: dict[int, list[QwenWindowTranscript]] = {}
    if pending_tracks:
        report({"type": "stage", "stage": "model_prepare", "profile": profile.id})
        model_root = model_prepare(models_root.resolve(), profile)
        report({"type": "stage", "stage": "model_load", "profile": profile.id})
        asr_session: AsrSession = asr_session_factory(model_root, plan)
        report({"type": "stage", "stage": "transcription", "profile": profile.id})
        try:
            for track in pending_tracks:
                source = _safe_track_path(package_root, track)
                report(
                    {
                        "type": "event",
                        "code": "TRACK_STARTED",
                        "stage": "transcription",
                        "track": track.number,
                        "total_tracks": total_tracks,
                        "speaker": track.speaker,
                    }
                )
                values: list[QwenWindowTranscript] = []
                for window in window_reader(source):
                    if is_cancelled():
                        raise QwenRuntimeError("ASR_CANCELLED")
                    text, language = asr_session.transcribe(window.audio, prompt=prompt)
                    values.append(
                        QwenWindowTranscript(
                            index=window.index,
                            start=window.start,
                            end=window.end,
                            text=text.strip(),
                            language=language or "Portuguese",
                        )
                    )
                    report(
                        {
                            "type": "event",
                            "code": "QWEN_WINDOW_TRANSCRIBED",
                            "stage": "transcription",
                            "track": track.number,
                            "total_tracks": total_tracks,
                            "speaker": track.speaker,
                            "window": window.index,
                        }
                    )
                if not values:
                    raise QwenRuntimeError("QWEN_AUDIO_EMPTY")
                pending_text[track.number] = values
        finally:
            asr_session.close()

    new_tracks: dict[int, TranscriptTrack] = {}
    energy_by_segment: dict[tuple[int, str], float] = {}
    warnings: list[str] = []
    if pending_tracks:
        report({"type": "stage", "stage": "alignment", "profile": profile.id})
        aligner_root = aligner_prepare(models_root.resolve())
        aligner: AlignerSession | None = aligner_session_factory(aligner_root, plan)
        try:
            for track in pending_tracks:
                source = _safe_track_path(package_root, track)
                report(
                    {
                        "type": "event",
                        "code": "TRACK_ALIGNMENT_STARTED",
                        "stage": "alignment",
                        "track": track.number,
                        "total_tracks": total_tracks,
                        "speaker": track.speaker,
                    }
                )
                expected = pending_text[track.number]
                expected_by_index = {item.index: item for item in expected}
                seen: set[int] = set()
                segments: list[TranscriptSegment] = []
                duration = 0.0
                last_index = expected[-1].index
                # Replay one decoded window at a time. Audio from previous windows
                # is released before the next one, keeping RAM bounded by one window.
                for window in window_reader(source):
                    if is_cancelled():
                        raise QwenRuntimeError("ASR_CANCELLED")
                    pending = expected_by_index.get(window.index)
                    if (
                        pending is None
                        or not math.isclose(pending.start, window.start, abs_tol=0.001)
                        or not math.isclose(pending.end, window.end, abs_tol=0.001)
                    ):
                        raise QwenRuntimeError("QWEN_WINDOW_REPLAY_MISMATCH")
                    seen.add(window.index)
                    if aligner is None:
                        aligner = aligner_session_factory(aligner_root, plan)
                    try:
                        window_segments = _strict_alignment_segments(
                            track.number,
                            window,
                            pending,
                            aligner,
                            first=window.index == expected[0].index,
                            last=window.index == last_index,
                        )
                    except QwenRuntimeError as exc:
                        cause = exc.__cause__
                        reason = (
                            cause.code if isinstance(cause, QwenRuntimeError) else exc.code
                        )
                        failure_details = {
                            "stage": "alignment",
                            "track": track.number,
                            "total_tracks": total_tracks,
                            "speaker": track.speaker,
                            "window": window.index,
                            "window_start": round(window.start, 3),
                            "window_end": round(window.end, 3),
                            "reason": reason,
                            "text_chars": len(pending.text),
                            "language": pending.language,
                            **(
                                cause.details
                                if isinstance(cause, QwenRuntimeError)
                                else {}
                            ),
                        }
                        if not _is_recoverable_alignment_failure(exc):
                            report(
                                {
                                    "type": "event",
                                    "code": "QWEN_ALIGNMENT_WINDOW_FAILED",
                                    **failure_details,
                                }
                            )
                            raise

                        report(
                            {
                                "type": "event",
                                "code": "QWEN_ALIGNMENT_WINDOW_RECOVERY_STARTED",
                                **failure_details,
                            }
                        )
                        aligner.close()
                        aligner = None
                        try:
                            recovered, recovery_aligner, recovery_parts = _recover_alignment_window(
                                track.number,
                                window,
                                first=window.index == expected[0].index,
                                last=window.index == last_index,
                                prompt=prompt,
                                model_root=model_root,
                                aligner_root=aligner_root,
                                plan=plan,
                                asr_session_factory=asr_session_factory,
                                aligner_session_factory=aligner_session_factory,
                                is_cancelled=is_cancelled,
                            )
                        except QwenRuntimeError as recovery_exc:
                            report(
                                {
                                    "type": "event",
                                    "code": "QWEN_ALIGNMENT_WINDOW_FAILED",
                                    **failure_details,
                                    "recovery_reason": recovery_exc.code,
                                }
                            )
                            raise

                        aligner = recovery_aligner
                        if recovered is None:
                            window_segments = (
                                _fallback_segment(track.number, window, pending.text),
                            )
                            warning = (
                                f"QWEN_ALIGNMENT_RAW_FALLBACK:track-{track.number}:"
                                f"window-{window.index}"
                            )
                            warnings.append(warning)
                            report(
                                {
                                    "type": "event",
                                    "code": "QWEN_ALIGNMENT_RAW_FALLBACK",
                                    **failure_details,
                                    "warning": warning,
                                }
                            )
                        else:
                            window_segments = recovered
                            report(
                                {
                                    "type": "event",
                                    "code": "QWEN_ALIGNMENT_WINDOW_RECOVERED",
                                    **failure_details,
                                    "recovery_parts": recovery_parts,
                                }
                            )
                    segments.extend(window_segments)
                    for segment in window_segments:
                        key = (track.number, segment.id)
                        value = energy_reader(window, segment.start, segment.end)
                        energy_by_segment[key] = max(
                            energy_by_segment.get(key, -120.0),
                            value,
                        )
                    duration = max(duration, window.end)
                if seen != set(expected_by_index):
                    raise QwenRuntimeError("QWEN_WINDOW_REPLAY_MISMATCH")
                transcript_track = TranscriptTrack(
                    number=track.number,
                    speaker=track.speaker,
                    source_filename=track.filename,
                    source_sha256=track.sha256,
                    duration_seconds=round(duration, 3),
                    segments=tuple(segments),
                    timeline_offset_seconds=track.timeline_offset_seconds,
                    identity=asdict(track.identity) if track.identity is not None else None,
                )
                transcript_track.validate()
                new_tracks[track.number] = transcript_track
                if checkpoints:
                    try:
                        save_track_checkpoint(package_root, signature, track, transcript_track)
                        report({"type": "event", "code": "ASR_CHECKPOINT_SAVED", "track": track.number})
                    except (OSError, ValueError):
                        report({"type": "event", "code": "ASR_CHECKPOINT_WRITE_SKIPPED", "track": track.number})
                report(
                    {
                        "type": "event",
                        "code": "TRACK_COMPLETED",
                        "stage": "alignment",
                        "track": track.number,
                        "total_tracks": total_tracks,
                        "speaker": track.speaker,
                    }
                )
                completed_tracks += 1
                report(
                    {
                        "type": "progress",
                        "completed": completed_tracks,
                        "total": total_tracks,
                        "unit": "tracks",
                        "stage": "alignment",
                    }
                )
        finally:
            if aligner is not None:
                aligner.close()

    if completed_tracks != total_tracks:
        raise QwenRuntimeError("QWEN_TRACK_PROGRESS_INCOMPLETE")

    transcript_tracks = tuple(cached_tracks.get(track.number) or new_tracks[track.number] for track in package.tracks)

    if cached_tracks:
        report({"type": "stage", "stage": "energy_analysis", "profile": profile.id})
        tracks_by_number = {track.number: track for track in transcript_tracks}
        for source_track in package.tracks:
            if source_track.number not in cached_tracks:
                continue
            report(
                {
                    "type": "event",
                    "code": "TRACK_ENERGY_STARTED",
                    "stage": "energy_analysis",
                    "track": source_track.number,
                    "total_tracks": total_tracks,
                    "speaker": source_track.speaker,
                }
            )
            source = _safe_track_path(package_root, source_track)
            transcript_track = tracks_by_number[source_track.number]
            remaining = list(transcript_track.segments)
            for window in window_reader(source):
                if is_cancelled():
                    raise QwenRuntimeError("ASR_CANCELLED")
                next_remaining: list[TranscriptSegment] = []
                for segment in remaining:
                    if segment.end <= window.start or segment.start >= window.end:
                        next_remaining.append(segment)
                        continue
                    key = (transcript_track.number, segment.id)
                    value = energy_reader(window, segment.start, segment.end)
                    energy_by_segment[key] = max(
                        energy_by_segment.get(key, -120.0),
                        value,
                    )
                remaining = next_remaining

    report({"type": "stage", "stage": "cross_track_dedup", "profile": profile.id})
    flattened = flatten_tracks(transcript_tracks)
    deduplicated, decisions = deduplicate_cross_track_segments(flattened, energy_by_segment=energy_by_segment)
    report({"type": "stage", "stage": "merge_timeline", "profile": profile.id})
    merged = tuple(sorted(deduplicated, key=lambda item: (item.start, item.end, item.track_number, item.segment_id)))
    report({"type": "stage", "stage": "turn_building", "profile": profile.id})
    turns = build_turns(merged)

    elapsed = max(time.monotonic() - started, 0.0)
    document = TranscriptDocument(
        recording_id=package.recording_id,
        source_sha256=package.source_sha256,
        language="pt",
        engine=TranscriptEngine(
            engine="qwen3",
            model=profile.model_id,
            profile=profile.id,
            device=plan.device,
            compute_type=plan.dtype,
            alignment=f"{QWEN_FORCED_ALIGNER_MODEL_ID}+strict-overlap-v3-recovery",
            model_revision=profile.revision,
        ),
        tracks=transcript_tracks,
        turns=turns,
        stats=stats_for_tracks(
            transcript_tracks,
            processing_seconds=elapsed,
            turn_count=len(turns),
            deduplicated_segment_count=len(decisions),
        ),
        warnings=tuple(sorted(set(warnings))),
    )
    report({"type": "stage", "stage": "result_prepare", "profile": profile.id})
    document.validate()
    return document
