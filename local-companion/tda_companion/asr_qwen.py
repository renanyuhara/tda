from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .asr_checkpoints import build_checkpoint_signature, load_track_checkpoint, save_track_checkpoint
from .asr_models import QWEN_FORCED_ALIGNER_MODEL_ID, AsrProfile, get_profile
from .asr_timeline import build_turns, deduplicate_cross_track_segments, flatten_tracks
from .craig import CraigPackage, CraigTrack
from .qwen_acceptance import (
    QwenAcceptanceError,
    QwenPlan,
    _qwen_inference_failure_code,
    prepare_qwen_aligner,
    prepare_qwen_model,
    resolve_qwen_plan,
)
from .transcript import (
    TranscriptDocument,
    TranscriptEngine,
    TranscriptSegment,
    TranscriptTrack,
    TranscriptWord,
    stats_for_tracks,
)

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]
WindowReader = Callable[[Path], Iterable["AudioWindow"]]
EnergyReader = Callable[["AudioWindow", float, float], float]

QWEN_WINDOW_SECONDS = 60.0
QWEN_SAMPLE_RATE = 16_000
QWEN_MAX_NEW_TOKENS = 512
QWEN_SEGMENT_GAP_SECONDS = 1.0
QWEN_SEGMENT_MAX_SECONDS = 30.0


class QwenRuntimeError(RuntimeError):
    def __init__(self, code: str, *, details: dict[str, Any] | None = None):
        super().__init__(code)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start: float
    end: float
    audio: Any


@dataclass(frozen=True)
class QwenWindowTranscript:
    index: int
    start: float
    end: float
    text: str
    language: str


class AsrSession(Protocol):
    def transcribe(self, audio: Any, *, prompt: str) -> tuple[str, str]: ...
    def close(self) -> None: ...


class AlignerSession(Protocol):
    def align(self, audio: Any, text: str, language: str) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def _safe_track_path(package_root: Path, track: CraigTrack) -> Path:
    root = package_root.resolve()
    candidate = (root / track.path).resolve()
    if candidate == root or root not in candidate.parents or not candidate.is_file():
        raise QwenRuntimeError("CRAIG_TRACK_PATH_INVALID")
    return candidate


def _bounded_prompt(context: str, glossary: str) -> str:
    context_value = " ".join(context.split())[:2000].strip()
    glossary_value = " ".join(glossary.split())[:2000].strip()
    parts = []
    if context_value:
        parts.append(f"Contexto da campanha: {context_value}")
    if glossary_value:
        parts.append(f"Vocabulário e nomes importantes: {glossary_value}")
    return "\n".join(parts)


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_fingerprint() -> str:
    return ";".join(
        (
            "checkpoint=qwen-track-v2",
            f"runtime={os.environ.get('TDA_ASR_RUNTIME_VERSION', 'development')}",
            f"torch={_distribution_version('torch')}",
            f"transformers={_distribution_version('transformers')}",
            f"accelerate={_distribution_version('accelerate')}",
        )
    )


def _model_is_cuda_only(model: Any) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        values = {str(value).lower() for value in device_map.values()}
        has_cuda = any(value.startswith("cuda") or value.isdigit() for value in values)
        has_offload = any(value.startswith("cpu") or value.startswith("disk") for value in values)
        return has_cuda and not has_offload
    return str(getattr(model, "device", "")).lower().startswith("cuda")


def iter_audio_windows(
    path: Path,
    *,
    window_seconds: float = QWEN_WINDOW_SECONDS,
    sample_rate: int = QWEN_SAMPLE_RATE,
) -> Iterable[AudioWindow]:
    if window_seconds <= 0 or window_seconds > 240 or sample_rate != QWEN_SAMPLE_RATE:
        raise QwenRuntimeError("QWEN_WINDOW_CONFIG_INVALID")
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise QwenRuntimeError("QWEN_RUNTIME_NOT_INSTALLED") from exc

    maximum = int(round(window_seconds * sample_rate))
    pieces: list[Any] = []
    buffered = 0
    consumed = 0
    index = 0

    def drain(force: bool = False) -> list[AudioWindow]:
        nonlocal pieces, buffered, consumed, index
        output: list[AudioWindow] = []
        if not pieces:
            return output
        if not force and buffered < maximum:
            return output
        joined = np.concatenate(pieces).astype(np.float32, copy=False)
        pieces = []
        buffered = 0
        cursor = 0
        while joined.size - cursor >= maximum:
            chunk = joined[cursor : cursor + maximum].copy()
            start = consumed / sample_rate
            consumed += chunk.size
            index += 1
            output.append(
                AudioWindow(index=index, start=start, end=consumed / sample_rate, audio=chunk)
            )
            cursor += maximum
        remainder = joined[cursor:]
        if remainder.size:
            pieces = [remainder.copy()]
            buffered = int(remainder.size)
        if force and pieces:
            tail = pieces[0]
            pieces = []
            buffered = 0
            start = consumed / sample_rate
            consumed += int(tail.size)
            index += 1
            output.append(
                AudioWindow(index=index, start=start, end=consumed / sample_rate, audio=tail)
            )
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
                    for value in drain():
                        yield value
            for converted in resampler.resample(None):
                array = converted.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                if array.size:
                    pieces.append(array.copy())
                    buffered += int(array.size)
                for value in drain():
                    yield value
            for value in drain(force=True):
                yield value
    except QwenRuntimeError:
        raise
    except Exception as exc:
        raise QwenRuntimeError("QWEN_AUDIO_DECODE_FAILED") from exc

    if index == 0:
        raise QwenRuntimeError("QWEN_AUDIO_EMPTY")


def _torch_dtype(torch: Any, name: str) -> Any:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise QwenRuntimeError("QWEN_DTYPE_UNAVAILABLE") from exc


class QwenAsrSession:
    def __init__(self, model_root: Path, plan: QwenPlan):
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise QwenRuntimeError("QWEN_RUNTIME_NOT_INSTALLED") from exc
        self._torch = torch
        try:
            dtype = _torch_dtype(torch, plan.dtype)
            self.processor = AutoProcessor.from_pretrained(str(model_root), local_files_only=True)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                str(model_root), dtype=dtype, device_map={"": "cuda:0"}, local_files_only=True
            )
        except Exception as exc:
            code = _qwen_inference_failure_code(exc)
            if code in {
                "QWEN_CUDA_DRIVER_INCOMPATIBLE",
                "QWEN_ASR_GPU_MEMORY_EXHAUSTED",
                "QWEN_ASR_CUDA_FAILED",
                "QWEN_ASR_RUNTIME_API_FAILED",
            }:
                raise QwenRuntimeError(code) from exc
            raise QwenRuntimeError("QWEN_MODEL_LOAD_FAILED") from exc
        if not _model_is_cuda_only(self.model):
            self.close()
            raise QwenRuntimeError("QWEN_MODEL_NOT_GPU_RESIDENT")

    def transcribe(self, audio: Any, *, prompt: str) -> tuple[str, str]:
        try:
            inputs = self.processor.apply_transcription_request(
                audio=audio,
                language="Portuguese",
                prompt=prompt or None,
            )
            inputs = inputs.to(self.model.device, self.model.dtype)
            with self._torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
        except Exception as exc:
            raise QwenRuntimeError(_qwen_inference_failure_code(exc)) from exc
        if not isinstance(parsed, dict):
            raise QwenRuntimeError("QWEN_ASR_OUTPUT_INVALID")
        text = str(parsed.get("transcription") or "").strip()
        if not text:
            raise QwenRuntimeError("QWEN_ASR_EMPTY_TRANSCRIPT")
        language = str(parsed.get("language") or "Portuguese")
        return text, language

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass


class QwenAlignerSession:
    def __init__(self, model_root: Path, plan: QwenPlan):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoProcessor
        except ImportError as exc:
            raise QwenRuntimeError("QWEN_RUNTIME_NOT_INSTALLED") from exc
        self._torch = torch
        try:
            dtype = _torch_dtype(torch, plan.dtype)
            self.processor = AutoProcessor.from_pretrained(str(model_root), local_files_only=True)
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(model_root), dtype=dtype, device_map={"": "cuda:0"}, local_files_only=True
            )
        except Exception as exc:
            code = _qwen_inference_failure_code(exc)
            if code in {
                "QWEN_CUDA_DRIVER_INCOMPATIBLE",
                "QWEN_ASR_GPU_MEMORY_EXHAUSTED",
                "QWEN_ASR_CUDA_FAILED",
                "QWEN_ASR_RUNTIME_API_FAILED",
            }:
                raise QwenRuntimeError(code) from exc
            raise QwenRuntimeError("QWEN_ALIGNER_LOAD_FAILED") from exc
        if not _model_is_cuda_only(self.model):
            self.close()
            raise QwenRuntimeError("QWEN_ALIGNER_NOT_GPU_RESIDENT")

    def align(self, audio: Any, text: str, language: str) -> list[dict[str, Any]]:
        try:
            inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
                audio=audio,
                transcript=text,
                language=language or "Portuguese",
            )
            inputs = inputs.to(self.model.device, self.model.dtype)
            with self._torch.inference_mode():
                outputs = self.model(**inputs)
            value = self.processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=self.model.config.timestamp_token_id,
            )[0]
        except Exception as exc:
            code = _qwen_inference_failure_code(exc)
            if code in {
                "QWEN_CUDA_DRIVER_INCOMPATIBLE",
                "QWEN_ASR_GPU_MEMORY_EXHAUSTED",
                "QWEN_ASR_CUDA_FAILED",
                "QWEN_ASR_RUNTIME_API_FAILED",
            }:
                raise QwenRuntimeError(code) from exc
            raise QwenRuntimeError("QWEN_ALIGNMENT_FAILED") from exc
        if not isinstance(value, list) or not value:
            raise QwenRuntimeError("QWEN_ALIGNMENT_EMPTY")
        return value

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass


def _default_asr_session(model_root: Path, plan: QwenPlan) -> AsrSession:
    return QwenAsrSession(model_root, plan)


def _default_aligner_session(model_root: Path, plan: QwenPlan) -> AlignerSession:
    return QwenAlignerSession(model_root, plan)


def _window_energy_db(window: AudioWindow, segment_start: float, segment_end: float) -> float:
    try:
        import numpy as np
    except ImportError as exc:
        raise QwenRuntimeError("QWEN_RUNTIME_NOT_INSTALLED") from exc
    relative_start = max(segment_start - window.start, 0.0)
    relative_end = min(segment_end - window.start, window.end - window.start)
    first = max(0, int(math.floor(relative_start * QWEN_SAMPLE_RATE)))
    last = min(len(window.audio), int(math.ceil(relative_end * QWEN_SAMPLE_RATE)))
    if last <= first:
        return -120.0
    sample = np.asarray(window.audio[first:last], dtype=np.float32)
    if sample.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(sample, dtype=np.float64))))
    return round(20.0 * math.log10(max(rms, 1e-6)), 3)


def _validated_words(
    items: list[dict[str, Any]],
    window: AudioWindow,
    *,
    discard_trailing_overflow_from: float | None = None,
) -> list[TranscriptWord]:
    """Validate aligner timestamps, optionally discarding an unusable tail in overlap.

    Strict overlapping alignment only owns the center of non-final windows. If
    the aligner predicts a word that starts entirely inside the trailing
    neighbor-owned overlap but extends beyond the physical window, the next
    window is authoritative for that region. Stop before that impossible tail
    instead of failing already-valid owned words.
    """
    words: list[TranscriptWord] = []
    previous_end = window.start
    word_index = 0

    def invalid(kind: str, *, relative_start: float | None = None, relative_end: float | None = None) -> QwenRuntimeError:
        details: dict[str, Any] = {"timestamp_failure": kind, "word_index": word_index}
        if relative_start is not None and math.isfinite(relative_start):
            details["relative_start"] = round(relative_start, 3)
        if relative_end is not None and math.isfinite(relative_end):
            details["relative_end"] = round(relative_end, 3)
        details["previous_end_relative"] = round(previous_end - window.start, 3)
        return QwenRuntimeError("QWEN_ALIGNMENT_TIMESTAMPS_INVALID", details=details)

    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        word_index += 1
        try:
            relative_start = float(item.get("start_time"))
            relative_end = float(item.get("end_time"))
        except (TypeError, ValueError) as exc:
            raise invalid("not_numeric") from exc
        start = window.start + relative_start
        end = window.start + relative_end
        if not math.isfinite(start) or not math.isfinite(end):
            raise invalid("non_finite", relative_start=relative_start, relative_end=relative_end)
        if relative_start < -0.05:
            raise invalid("negative_start", relative_start=relative_start, relative_end=relative_end)
        if end < start:
            raise invalid("end_before_start", relative_start=relative_start, relative_end=relative_end)
        if start + 0.05 < previous_end:
            raise invalid("temporal_regression", relative_start=relative_start, relative_end=relative_end)
        if end > window.end + 0.25:
            if (
                discard_trailing_overflow_from is not None
                and start >= discard_trailing_overflow_from
            ):
                break
            raise invalid("beyond_window", relative_start=relative_start, relative_end=relative_end)
        words.append(TranscriptWord(text=text, start=round(start, 3), end=round(end, 3)))
        previous_end = max(previous_end, end)
    if not words:
        raise QwenRuntimeError("QWEN_ALIGNMENT_EMPTY")
    return words


def _segments_from_words(
    track_number: int,
    window: AudioWindow,
    words: list[TranscriptWord],
) -> tuple[TranscriptSegment, ...]:
    groups: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            duration = word.end - current[0].start
            if gap >= QWEN_SEGMENT_GAP_SECONDS or duration > QWEN_SEGMENT_MAX_SECONDS:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    segments: list[TranscriptSegment] = []
    for group_index, group in enumerate(groups, start=1):
        text = " ".join(item.text for item in group).strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                id=f"{track_number}-q{window.index:04d}-{group_index:03d}",
                start=group[0].start,
                end=group[-1].end,
                text=text,
                words=tuple(group),
            )
        )
    return tuple(segments)


def _fallback_segment(track_number: int, window: AudioWindow, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        id=f"{track_number}-q{window.index:04d}-fallback",
        start=round(window.start, 3),
        end=round(window.end, 3),
        text=text.strip(),
        words=(),
    )


def _prepare_model(models_root: Path, profile: AsrProfile) -> Path:
    try:
        return prepare_qwen_model(models_root, profile)
    except QwenAcceptanceError as exc:
        raise QwenRuntimeError(exc.code) from exc


def _prepare_aligner(models_root: Path) -> Path:
    try:
        return prepare_qwen_aligner(models_root)
    except QwenAcceptanceError as exc:
        raise QwenRuntimeError(exc.code) from exc


def _resolve_plan(profile_id: str) -> QwenPlan:
    try:
        return resolve_qwen_plan(profile_id)
    except QwenAcceptanceError as exc:
        raise QwenRuntimeError(exc.code) from exc


def transcribe_craig_package_qwen(
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
    plan_resolver: Callable[[str], QwenPlan] = _resolve_plan,
    model_prepare: Callable[[Path, AsrProfile], Path] = _prepare_model,
    aligner_prepare: Callable[[Path], Path] = _prepare_aligner,
    asr_session_factory: Callable[[Path, QwenPlan], AsrSession] = _default_asr_session,
    aligner_session_factory: Callable[[Path, QwenPlan], AlignerSession] = _default_aligner_session,
    window_reader: WindowReader = iter_audio_windows,
    energy_reader: EnergyReader = _window_energy_db,
) -> TranscriptDocument:
    profile = get_profile(profile_id)
    if profile.engine != "qwen3":
        raise QwenRuntimeError("QWEN_PROFILE_REQUIRED")
    report = report or (lambda _: None)
    is_cancelled = is_cancelled or (lambda: False)
    plan = plan_resolver(profile.id)
    prompt = _bounded_prompt(context, glossary)
    recipe = {
        "window_seconds": QWEN_WINDOW_SECONDS,
        "sample_rate": QWEN_SAMPLE_RATE,
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "segment_gap_seconds": QWEN_SEGMENT_GAP_SECONDS,
        "segment_max_seconds": QWEN_SEGMENT_MAX_SECONDS,
        "device": plan.device,
        "dtype": plan.dtype,
        "alignment": QWEN_FORCED_ALIGNER_MODEL_ID,
    }
    signature = build_checkpoint_signature(
        package,
        profile,
        recipe=recipe,
        context=" ".join(context.split())[:2000].strip(),
        glossary=" ".join(glossary.split())[:2000].strip(),
        runtime_fingerprint=_runtime_fingerprint(),
    )

    cached_tracks: dict[int, TranscriptTrack] = {}
    pending_tracks: list[CraigTrack] = []
    for track in package.tracks:
        if is_cancelled():
            raise QwenRuntimeError("ASR_CANCELLED")
        _safe_track_path(package_root, track)
        cached = load_track_checkpoint(package_root, signature, track) if checkpoints else None
        if cached is not None:
            cached_tracks[track.number] = cached
            report({"type": "event", "code": "ASR_CHECKPOINT_REUSED", "track": track.number})
        else:
            pending_tracks.append(track)

    started = time.monotonic()
    pending_text: dict[int, list[QwenWindowTranscript]] = {}
    warnings: list[str] = []
    for track_number, cached in cached_tracks.items():
        if any(segment.id.endswith("-fallback") for segment in cached.segments):
            warnings.append(f"QWEN_ALIGNMENT_FALLBACK:track-{track_number}")

    if pending_tracks:
        report({"type": "stage", "stage": "model_prepare", "profile": profile.id})
        model_root = model_prepare(models_root.resolve(), profile)
        report({"type": "stage", "stage": "transcription", "profile": profile.id})
        asr_session = asr_session_factory(model_root, plan)
        try:
            for track in pending_tracks:
                source = _safe_track_path(package_root, track)
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
                            "window": window.index,
                        }
                    )
                if not values:
                    raise QwenRuntimeError("QWEN_AUDIO_EMPTY")
                pending_text[track.number] = values
        finally:
            asr_session.close()

    new_tracks: dict[int, TranscriptTrack] = {}
    if pending_tracks:
        if is_cancelled():
            raise QwenRuntimeError("ASR_CANCELLED")
        report({"type": "stage", "stage": "alignment", "profile": profile.id})
        aligner_root = aligner_prepare(models_root.resolve())
        aligner = aligner_session_factory(aligner_root, plan)
        try:
            for track in pending_tracks:
                source = _safe_track_path(package_root, track)
                windows = {window.index: window for window in window_reader(source)}
                segments: list[TranscriptSegment] = []
                fallback_used = False
                for pending in pending_text[track.number]:
                    if is_cancelled():
                        raise QwenRuntimeError("ASR_CANCELLED")
                    window = windows.get(pending.index)
                    if window is None:
                        raise QwenRuntimeError("QWEN_WINDOW_REPLAY_MISMATCH")
                    try:
                        aligned = aligner.align(window.audio, pending.text, pending.language)
                        words = _validated_words(aligned, window)
                        window_segments = _segments_from_words(track.number, window, words)
                        if not window_segments:
                            raise QwenRuntimeError("QWEN_ALIGNMENT_EMPTY")
                        segments.extend(window_segments)
                    except QwenRuntimeError:
                        fallback_used = True
                        segments.append(_fallback_segment(track.number, window, pending.text))
                        report(
                            {
                                "type": "event",
                                "code": "QWEN_ALIGNMENT_FALLBACK",
                                "stage": "alignment",
                                "track": track.number,
                                "window": pending.index,
                            }
                        )
                if fallback_used:
                    warnings.append(f"QWEN_ALIGNMENT_FALLBACK:track-{track.number}")
                duration = max((window.end for window in windows.values()), default=0.0)
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
        finally:
            aligner.close()

    transcript_tracks = tuple(
        cached_tracks.get(track.number) or new_tracks[track.number] for track in package.tracks
    )

    energy_by_segment: dict[tuple[int, str], float] = {}
    tracks_by_number = {track.number: track for track in transcript_tracks}
    for source_track in package.tracks:
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
                energy_by_segment[(transcript_track.number, segment.id)] = energy_reader(
                    window, segment.start, segment.end
                )
            remaining = next_remaining

    report({"type": "stage", "stage": "cross_track_dedup", "profile": profile.id})
    flattened = flatten_tracks(transcript_tracks)
    deduplicated, decisions = deduplicate_cross_track_segments(
        flattened,
        energy_by_segment=energy_by_segment,
    )
    report({"type": "stage", "stage": "merge_timeline", "profile": profile.id})
    merged = tuple(sorted(deduplicated, key=lambda item: (item.start, item.end, item.track_number, item.segment_id)))
    report({"type": "stage", "stage": "turn_building", "profile": profile.id})
    turns = build_turns(merged)

    elapsed = max(time.monotonic() - started, 0.0)
    alignment_name = QWEN_FORCED_ALIGNER_MODEL_ID
    if warnings:
        alignment_name = f"{QWEN_FORCED_ALIGNER_MODEL_ID}+window-fallback"
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
            alignment=alignment_name,
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
    document.validate()
    report({"type": "stage", "stage": "result_prepare", "profile": profile.id})
    return document
