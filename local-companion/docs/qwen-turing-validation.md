# Qwen Turing / RTX 2080 validation notes

This note records the compatibility investigation performed while validating the
isolated Qwen runtime on a physical NVIDIA GeForce RTX 2080 SUPER. It documents
observed behavior and the reasoning behind the overlap fix; it is not a general
support declaration for all Turing GPUs.

## Scope and tested environment

The physical validation used:

- NVIDIA GeForce RTX 2080 SUPER, 8 GB (Turing, compute capability 7.5)
- NVIDIA driver 595.97
- Qwen runtime 1.0.7 experimental build
- PyTorch 2.13.0+cu126 / CUDA runtime 12.6
- Transformers 5.17.0
- Qwen3 ASR and native Forced Aligner
- Companion target: 0.3.14 RC

The runtime probe reported CUDA execution ready and BF16 supported. Physical
sample acceptance executed both ASR and Forced Aligner in BF16 successfully on
this exact hardware/runtime combination. Do not infer support for every SM 7.5
device from this single validation.

## Original compatibility gate

The Qwen runtime originally rejected devices below compute capability 8.0.
During this investigation the minimum experimental gate was lowered to 7.5 so a
real Turing device could be tested instead of treating architecture alone as
proof of incompatibility.

An early manifest experiment added a `turing_dtype=float16` hint. It was later
removed because the runtime did not consume that field and the physical RTX
2080 SUPER probe reported BF16 support; the effective runtime plan already
selects BF16 when the probe reports it supported and otherwise selects FP16.

## Physical acceptance

A ~61 second physical acceptance sample completed ASR and Forced Aligner on the
RTX 2080 SUPER. Observed ASR real-time factor was approximately 0.032 and
alignment real-time factor approximately 0.013. Peak GPU memory remained below
the 8 GB device capacity.

This established that CUDA execution, model loading, ASR and alignment can all
run on the tested SM 7.5 device. Full-session stability still required a Craig
end-to-end run.

## Full Craig failure

A four-track Craig session transcribed successfully but strict alignment failed
on track 2. Instrumentation made the generic error chain observable:

```text
QWEN_ALIGNMENT_REQUIRED
└── QWEN_ALIGNMENT_TIMESTAMPS_INVALID
    └── timestamp_failure=beyond_window
```

The failure reproduced at:

```text
track:                 2
window:                88
window absolute:       4698.0 -> 4758.0 s
word index:            237
relative start:        59.92 s
relative end:          61.68 s
previous end relative: 59.92 s
```

The strict Qwen pipeline uses 60 second windows with 6 seconds of overlap. For
an intermediate window, ownership of the right overlap changes at half the
overlap: relative second 57. A word beginning at 59.92 seconds therefore starts
entirely in the region owned by the following window.

## Root cause

Timestamp validation ran before overlap ownership filtering. Consequently,
`_validated_words(...)` rejected the 61.68 second end timestamp even though
the corresponding word began in a trailing-overlap region that
`_owned_words(...)` would not publish from this window.

The failure was therefore not evidence by itself of a CUDA or BF16 execution
failure. It exposed an integration edge case between forced-alignment output
and the strict overlap ownership policy.

## Fix

For non-final strict windows, validation is now allowed to stop when a
`beyond_window` span begins at or after the right-side ownership boundary. The
following overlapping window is authoritative for that region.

The fix deliberately does **not** increase the global timestamp tolerance.

The fail-closed behavior remains in place when:

- an overflowing word begins inside the region the current window may own;
- the current window is the final window and no following window can own the
  tail;
- timestamps are non-numeric or non-finite;
- a start is invalidly negative;
- an end precedes its start;
- timestamps regress beyond the existing tolerance.

Diagnostic errors include safe numeric metadata and the failure category but do
not log transcript text.

## Regression coverage

Focused Qwen tests after the fix:

```text
14 passed
```

The regression tests cover both sides of the policy:

1. the observed `59.92 -> 61.68`-style trailing-overlap overflow is discarded;
2. a beyond-window span beginning in the current window's owned region still
   fails closed.

Full local Companion suite after the fix:

```text
641 passed, 3 skipped, 2 warnings in 32.75s
```

The two warnings were dependency deprecations in the test stack, not functional
test failures.

## Validation status

| Gate | Status |
| --- | --- |
| Runtime probe on RTX 2080 SUPER / SM 7.5 | PASS |
| ~61 s physical ASR acceptance | PASS |
| ~61 s physical Forced Aligner acceptance | PASS |
| Focused regression tests | PASS (14) |
| Full local Companion test suite | PASS (641; 3 skipped) |
| Full Craig run with overlap fix | PENDING |
| Final Companion/Web flow without manual commands | PENDING |
| General Turing / SM 7.5 support declaration | NOT ESTABLISHED |

The pending physical and Companion/Web gates must pass before this work should
be presented as complete support for the tested workflow.

## Related work

- Fork issue #2: invalid Qwen forced-alignment timestamps
- Upstream Faysk/tda#416: physical validation of Qwen on Turing / RTX 2080
- Upstream Faysk/tda#423: persist Qwen text before alignment for safe retry

The retry behavior in #423 is separate from the overlap bug: when alignment
fails before a track checkpoint is written, already completed ASR work for
pending tracks may need to be repeated.
