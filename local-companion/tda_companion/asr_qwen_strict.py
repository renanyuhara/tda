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
        words = _validated_words(aligned, window)
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
        "alignment_policy": "strict-overlap-v2",
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
    report({"type": "event", "code": "QWEN_CHECKPOINT_SCAN_STARTED", "stage": "runtime_validation", "total_tracks": len(package.tracks)})

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
                    "stage": "source_validation",
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

    report({"type": "event", "code": "QWEN_CHECKPOINT_SCAN_READY", "stage": "runtime_validation", "cached_tracks": len(cached_tracks), "pending_tracks": len(pending_tracks)})
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
    if pending_tracks:
        report({"type": "stage", "stage": "alignment", "profile": profile.id})
        aligner_root = aligner_prepare(models_root.resolve())
        aligner: AlignerSession = aligner_session_factory(aligner_root, plan)
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
                    if pending is None or not math.isclose(pending.start, window.start, abs_tol=0.001) or not math.isclose(pending.end, window.end, abs_tol=0.001):
                        raise QwenRuntimeError("QWEN_WINDOW_REPLAY_MISMATCH")
                    seen.add(window.index)
                    window_segments = _strict_alignment_segments(
                        track.number,
                        window,
                        pending,
                        aligner,
                        first=window.index == expected[0].index,
                        last=window.index == last_index,
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
            alignment=f"{QWEN_FORCED_ALIGNER_MODEL_ID}+strict-overlap-v2",
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
        warnings=(),
    )
    report({"type": "stage", "stage": "result_prepare", "profile": profile.id})
    document.validate()
    return document
