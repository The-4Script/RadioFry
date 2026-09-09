# OP.md — Operational Log

Lightweight run log for RadioFry (SIH26147). One block per run.
Engineering detail lives in `BANK.md`; this file is just what was run and what happened.

---

## Run 001

- **Date:** 2026-09-08
- **Timestamp:** 00:42:46 IST
- **Task:** Verification run over the work completed in BANK.md Entries 001–008
  (Synthetic Dataset V1, evaluation harness, CNN input-adapter fix, adaptive symbol-rate
  feature selection, QAM scale-invariance fix). No production code changed in this run.
- **Result:** PASS (1 known pre-existing failure)

### Tests run

`python -m pytest -q --ignore=tests/test_model_report.py` — 290 collected.

| Module | Result |
|---|---|
| test_synthetic_v1_modulation | 47 passed |
| test_synthetic_v1_io | 10 passed |
| test_synthetic_v1_dataset | 20 passed |
| test_v1_harness | 17 passed |
| test_v1_harness_metrics | 14 passed |
| test_modulation_inference_input | 9 passed |
| test_symbol_rate_diagnostics | 19 passed |
| test_symbol_rate_feature_comparison | 25 passed |
| test_symbol_rate_estimator_regression | 38 passed |
| test_qam_demodulation_diagnostics | 12 passed |
| test_qam_demodulation_scale | 28 passed |
| test_qam_routing_diagnostics | 10 passed |

**Totals: 289 passed, 1 failed.**

### Pass/fail detail

- FAIL — `tests/test_decoding_correlation.py::test_reed_solomon_round_trip`.
  Pre-existing and unrelated: `reedsolo` is not installed in this environment.
  Also absent: `commpy`, `pyldpc`, `h5py` (the last skips `tests/test_model_report.py`).

### Important observations

- Production files confirmed unchanged since Entry 008:
  `modulation_inference.py` 78 lines, `parameter_estimation.py` 140 lines,
  `qam_demod.py` 27 lines.
- The multi-window CNN inference change (Entry 008 recommendation) is **not started**.
- Generated artefacts on disk total ~11 MB and are **not** covered by `.gitignore`
  (it only ignores `reports/*.png` and `data/synthetic/**/*.npy`). Commit decision is
  pending — see `ANTIGRAVITY_COMMIT.txt`.

---

## Run 002

- **Date:** 2026-09-08
- **Timestamp:** 09:23:06 IST
- **Task:** Multi-window CNN inference — controlled experiment (300 captures, paired),
  decision, implementation, and V1 re-evaluation. See BANK.md Entry 010.
- **Result:** PASS — change implemented and justified by measurement.

### Tests run

- `tests/test_modulation_inference_windows.py` (new) + `tests/test_modulation_inference_input.py` — **30 passed**
- Full suite `python -m pytest -q --ignore=tests/test_model_report.py` — **310 passed, 1 failed**

The single failure is the same pre-existing `reedsolo` environment gap.

### Production change

`src/radiofry/models/modulation_inference.py` (78 → 116 lines): 4-window default,
mean-softmax aggregation. `_fixed_iq` untouched, so its 9 existing tests pass unmodified.

### Key results

| Metric | Before | After |
|---|---|---|
| CNN top-1 | 0.400 | 0.733 |
| CNN top-3 | 0.733 | 0.833 |
| Fusion accuracy | 0.400 | 0.633 |
| Median BER (strict) | 0.4944 | 0.0293 |
| Demod reached | 0.867 | 0.800 |
| Rejection rate | 0.133 | 0.200 |

16QAM CNN 0.00 → 0.80, 64QAM 0.00 → 0.60. 16QAM reaches 0.000 BER at 20 dB.

### Important observations

- Window count 4 was chosen because the paired McNemar test gives p = 0.0003 against
  1 window while 8/16/32/64 are all indistinguishable from 4 (p ≥ 0.26). Entry 008's
  preference for 64 windows was sampling noise.
- Vote-share confidence was rejected: it is degenerate at one window (confidence 1.000,
  rejection 0.000) and would have silently disabled fusion's rejection path.
- **Regression reported, not hidden:** 3 captures lost demodulation as confidence fell
  below the 0.4 threshold. All three previously produced chance-level BER under a wrong
  label, so no real capability was lost. Fusion was not adjusted.
- BFSK remains 0% / ~0.51 BER — unrelated open defect.

---

## Run 003

- **Date:** 2026-09-08
- **Timestamp:** 09:34:06 IST
- **Task:** BFSK forensic investigation (BANK.md Entry 011). No production code changed.
- **Result:** PASS — root causes identified with controlled evidence.

### Tests run

Full suite `python -m pytest -q --ignore=tests/test_model_report.py` — **310 passed,
1 failed** (the pre-existing `reedsolo` environment gap). No new tests added; this was a
diagnostic run using the existing V1 captures and generator.

### Important observations

- BFSK failure chain: ingestion, preprocessing and parameter estimation all PASS
  (estimated Rs exactly 25 000 Hz); the CNN FAILS, predicting `8PSK` at 0.96 confidence
  across all four windows, so the FSK demodulator is never invoked.
- The V1 BFSK waveform is measurably correct (phase advance exactly pi/8 per sample,
  tones at ±12 500 Hz) but uses modulation index **h = 1**.
- Deviation probe: the CNN returns `CPFSK` at 1.000 confidence for h = 0.5 and `GFSK` at
  1.000 for h = 0.25. The classification failure is specific to h = 1.
- `demodulate_fsk` itself is correct: 0.0000 BER at full rate. The production path
  decimates to one sample per symbol, where h = 1's ±pi per-symbol advance is ambiguous
  (0.2512 even after searching all offsets and shifts, noiseless).
- Multi-window inference cannot help BFSK — all four windows agree on the wrong label.

---

## Run 004

- **Date:** 2026-09-08
- **Timestamp:** 09:48:07 IST
- **Task:** Make FSK deviation an explicit swept V1 dimension (BANK.md Entry 012).
- **Result:** PASS - classification fixed at h=0.5; production demodulation still blocked.

### Tests run

- `tests/test_synthetic_v1_modulation.py` - 52 passed
- `tests/test_synthetic_v1_dataset.py` - 28 passed
- Full suite - **323 passed, 1 failed** (pre-existing `reedsolo` gap)

### Production change

V1 generator only: `config.py` (140->179), `generator.py` (~350->386), `cli.py`,
`__init__.py`. Default FSK index now h=0.5; h=1.0 retained as a labelled `known_hard`
arm.

### Key results

- Dataset 30 -> 40 captures. **25/25 non-FSK captures byte-identical**; 5/5 old BFSK
  captures preserved as the h=1.0 arm.
- h=0.5: CNN returns **CPFSK at 1.000 confidence**, fusion forwards it - routing fixed.
- h=1.0: CNN still returns 8PSK at 0.956 - reproduced exactly, now labelled known-hard.
- Full-rate FSK demodulation: **0.0000 BER** at 20 dB for both arms.
- Production dispatch BER remains **~0.25** at h=0.5.

### Important observations

- The remaining failure is now **measured, not hypothesised**: 7 of 8 timing offsets
  reach 0.0049 BER, but the smoothness heuristic in `dispatch.py` picks offset 3 - the
  single worst one. Location: `dispatch.py:36-42`.
- V1 median BER moved 0.0293 -> 0.0796, which is a denominator effect from adding ten
  BFSK captures to the set, not a regression. No capture got worse.

---

## Run 005

- **Date:** 2026-09-08
- **Timestamp:** 10:00:42 IST
- **Task:** V1 Task #13 - FSK-aware timing-offset selection in dispatch (BANK.md Entry 013).
- **Result:** PASS - h=0.5 FSK now demodulates correctly; no non-FSK regression.

### Tests run

- `tests/test_fsk_timing_dispatch.py` (new) - 21 passed
- Full suite - **344 passed, 1 failed** (pre-existing `reedsolo` gap)

### Production change

`src/radiofry/decoding/demodulators/dispatch.py` (64 -> 99 lines): FSK labels now select
the timing offset that minimises instantaneous-frequency variance inside each candidate
symbol window; every other modulation keeps the existing heuristic unchanged.

### Key results

- Criterion comparison (30 cases): existing heuristic 0.2677 mean BER, new criterion
  **0.0508**, oracle 0.0494 - essentially optimal.
- h=0.5 through production dispatch: **0.0180** at >=10 dB versus 0.2517 before, offset 0
  selected for every seed and SNR.
- V1 median strict BER **0.0796 -> 0.0138**; BFSK h=0.5 at 20/15/10 dB now 0.0049/0.0049/0.0054.
- **Zero non-FSK captures changed** - verified capture-by-capture.

### Important observations

- h=1.0 is not solved and is not claimed to be. In the full pipeline it never reaches
  this code path anyway, because the CNN labels it 8PSK and dispatch routes it to PSK.
- h=0.5 at 5 dB and 0 dB is still ~0.49, caused by the **symbol-rate estimator** failing
  below ~10 dB on FSK (estimated 7 574 Hz and 23 486 Hz, giving sps 26 and 9 instead of
  8). Not a timing problem; left untouched.

---

## Run 006

- **Date:** 2026-09-08
- **Timestamp:** 10:08:18 IST
- **Task:** Investigate low-SNR FSK symbol-rate estimation (BANK.md Entry 014).
- **Result:** NEGATIVE - no safe small fix. No production code changed.

### Tests run

Full suite - **344 passed, 1 failed** (pre-existing `reedsolo` gap). Unchanged from
Run 005, as expected for a diagnostic-only run.

### Important observations

- At 5 dB and 0 dB **none** of the three existing symbol-rate features finds 25 kHz
  (0/5 each across seeds), so changing which feature wins cannot help.
- The 25 kHz line in `phase_second_difference` is at **0.5x the noise floor at 5 dB**
  (rank 68 of 706 peaks) and 1.7x at 0 dB (rank 78 of 574). There is no peak to recover.
- Separate finding recorded for later: at 10 dB the correct feature is right 4/5 but
  production selects it only 2/5, a selection issue rather than an information one. Not
  acted on - it would change selection for all modulations.

### Decision

**V1 frozen with this known limitation.** FSK h=0.5 works at 20/15/10 dB
(BER 0.0049-0.0054); 5 dB and 0 dB remain broken.

---

## Run 007

- **Date:** 2026-09-08
- **Timestamp:** 10:14:33 IST
- **Task:** V1 freeze - documentation only. Record the five carry-forward problems and
  the final V1 baseline for V2.
- **Result:** PASS - V1 FROZEN.

### Tests run

None. This was a documentation-only operation; no code or tests were touched, so the
last recorded suite result stands (Run 006: 344 passed, 1 pre-existing `reedsolo`
failure).

### Changes

- `BANK.md`: appended a new section **"V2 CARRY-FORWARD PROBLEMS / REQUIRED
  INVESTIGATIONS"** after Entry 014, holding the final frozen V1 baseline and the five
  carry-forward items (low-SNR FSK symbol rate, FSK h=1.0 known-hard, CNN QAM16/QAM64
  separation, fusion confidence calibration, FSK decimate-then-discriminate BER cost).
  Each is cited to the entry that measured it and is explicitly labelled V2 backlog
  rather than unfinished V1 work.
- `OP.md`: this entry.
- `ANTIGRAVITY_COMMIT.txt`: **not modified** - there are no production or test changes to
  record.

### Confirmation

- **Zero production-code changes.** No file under `src/` was touched, and no test file
  was added or modified.
- **V1 is frozen** at: 40 captures, CNN top-1 0.7714, fusion 0.6857, demod reached
  0.8286, median strict BER 0.0138, symbol rate within 1% 0.9143.

---

## Run 008

- **Date:** 2026-09-08
- **Timestamp:** 10:38:02 IST
- **Task:** V2.0 model training on the synthetic V1 distribution (BANK.md Entry 015).
- **Result:** PASS - new checkpoint trained and benchmarked; conclusion B.

### Tests run

- `tests/test_v2_training_pipeline.py` (new) - 14 passed
- Full suite - **358 passed, 1 failed** (pre-existing `reedsolo` gap, reported separately)

### Training

840 captures / 53 760 frames, capture-level 60/20/20 split (split seed 20 260 908,
torch seed 7), iqap 4-channel 128-sample frames built with the inference preprocessing.
Best epoch 18 (validation loss 0.12028), early stopped at 24, 450 s on CPU.

### Results

- Held-out synthetic test: top-1 **0.9382**, top-3 0.9992.
- Independent generalisation seed set: **0.9406** - matches test, so no memorisation.
- Frozen V1, OLD -> NEW: CNN top-1 0.7714 -> **0.9714**, fusion 0.6857 -> **0.9714**,
  rejection 0.1714 -> **0.0000**, demod reached 0.8286 -> **1.0000**.
- Per capture: 5 improved, **0 worse**, 24 unchanged, 6 newly demodulated, 0 lost.

### Important observations

- Full-set median BER rose 0.0138 -> 0.0796, but on the 58 records **both** models scored
  the median is **identical**. The rise is entirely 12 newly-scored hard low-SNR records.
- BFSK 0.50 -> 1.00 including h=1.0; its residual 0.25 BER is the known +/-pi DSP limit.
- QAM64 remains weakest: 0.7025 on the synthetic test set, 450/1600 frames called QAM16.
- The new checkpoint has only the 6 V1 classes; the old one had 11 RML classes. Not a
  drop-in replacement without a decision on the missing five.
- Nothing in fusion/dispatch/DSP was touched to flatter the result.

---

## Run 009

- **Date:** 2026-09-08
- **Timestamp:** 23:27:31 IST
- **Task:** Add PAM4 + GFSK to the V2 synthetic generator, 6 -> 8 classes (BANK.md Entry 016).
- **Result:** PASS - generator support added and validated; no training run launched.

### Tests run

- `tests/test_synthetic_pam4_gfsk.py` (new) - 30 passed
- `tests/test_v2_training_pipeline.py` (updated for 8 classes) - 16 passed
- Full suite - **390 passed, 1 failed** (pre-existing `reedsolo` gap)

### Production change

V1 generator package only: `config.py`, `modulation.py`, `generator.py`, `__init__.py`.
PAM4 as a new `pam` constellation family; GFSK as an FSK with a unit-area Gaussian
frequency pulse (BT 0.3 default).

### Key results

- Frozen V1 aggregate .iq SHA-256 **unchanged** (`d6d3f918687d0700a43e46211c3f04b9`,
  40 captures); six V1 captures additionally re-derived and matched sample by sample.
- PAM4: noiseless capture recovers the **exact** source bits via an independent oracle.
- GFSK: unit pulse area (1.000000), accumulated phase preserved within 1.5%, peak
  deviation never exceeded (6250 Hz), narrower occupied bandwidth than CPFSK.
- 8-class dataset builds correctly: labels match production vocabulary, capture-level
  split intact, frames (N, 4, 128) float32 all finite, V1 seed-collision check passes.

### Important observations

- One of my own tests initially asserted the wrong GFSK invariant (mean absolute
  frequency). Gaussian shaping preserves pulse *area*, not that mean; the test was
  corrected to the real physics rather than the measurement being adjusted.
- CPFSK and GFSK will each get 2x the captures of the other six classes because both are
  swept over h=0.5 and h=1.0. Flagged as a conscious decision needed before training.
- PAM4 remains classifiable but **not demodulatable** - no dispatch route was added.

---

## Run 010

- **Date:** 2026-09-08
- **Timestamp:** 23:47:30 IST
- **Task:** Train and evaluate the 8-class V2.0 modulation CNN (BANK.md Entry 017).
- **Result:** PASS - 8-class model trained; strict improvement, zero V1 regression.

### Tests run

Full suite - **390 passed, 1 failed** (pre-existing `reedsolo` gap). No production code
changed in this run, so the count is unchanged from Run 009.

### Training

1200 captures / 76 800 frames, capture-level split 700/250/250, Entry 015 configuration
unchanged (Adam 1e-3, batch 256, patience 6, split seed 20 260 908, torch seed 7).
Best epoch 25, validation loss 0.09626, early stopped at 31, 584 s on CPU.

### Key results

| Metric | 6-class | 8-class |
|---|---|---|
| Held-out test top-1 | 0.9382 | **0.9583** |
| Independent generalisation | 0.9406 | **0.9545** |
| QAM64 per-class | 0.7025 | **0.8050** |
| Frozen V1 CNN top-1 | 0.9714 | 0.9714 (identical) |

Frozen V1 per-capture: **0 improved, 0 worse, 35 unchanged** - adding PAM4 and GFSK cost
the original six classes nothing.

### Important observations

- New classes learn well: PAM4 0.9950, GFSK 0.9881. CPFSK/GFSK confusion is only
  1.81%/1.16%.
- **QAM64 collapses at low SNR: 0.334 at 0 dB, 0.744 at 5 dB**; 285/1600 test frames go
  to QAM16. Dominant residual error, reported as measured.
- CPFSK dipped 0.9991 -> 0.9816 and QAM16 0.9313 -> 0.9119 because of the added
  neighbours; QAM64 improved by 10 points. Net accuracy rose.
- The frozen V1 set has no PAM4/GFSK captures, so it only regression-tests the original
  six; the new classes rest on the synthetic held-out and generalisation sets.
- Fusion rejection is 0.0000 on V1 for both V2 models - the uncalibrated 0.4 threshold
  now rejects nothing.

---

## Run 011

- **Date:** 2026-09-08
- **Timestamp:** 23:58:00 IST
- **Task:** Analog ground-truth design investigation for AM-DSB / AM-SSB / WBFM
  (BANK.md Entry 018). Design only - nothing implemented.
- **Result:** PASS - design produced; four concrete blockers measured.

### Tests run

None beyond read-only probes; no code changed. Last recorded suite result stands
(Run 010: 390 passed, 1 pre-existing `reedsolo` failure).

### Important observations

- **Four blockers verified, not assumed:**
  1. `expected_family_for("analog")` raises `KeyError` and would crash the harness on the
     first analog capture.
  2. `eb_n0_db` evaluates to **-inf** when `bits_per_symbol = 0`.
  3. `dispatch` synthesises bits for analog (`analog > median(analog)`), so an analog
     capture would publish a meaningless numeric BER.
  4. `dispatch` decimates analog by the estimated symbol rate, a quantity with no
     physical meaning for these classes.
- Labels and fusion need **no** work: all three analog labels already map to
  `analog-like` in `confidence_fusion.py`.
- Proposed: deterministic multi-tone message (no audio assets), one shared analog schema
  with a `scheme` discriminator, BER explicitly unavailable, and message-recovery
  correlation as the positive metric with a documented decimation caveat.
- Recommended first move is the two latent fixes (blockers 1 and 2), which are safe and
  independent of the analog work.

---

## Run 012

- **Date:** 2026-09-09
- **Timestamp:** 00:05:11 IST
- **Task:** Analog prerequisite safety fixes - family lookup, Eb/N0 guard, symbol-rate
  experiment pin (BANK.md Entry 019). No analog generation.
- **Result:** PASS - two fixes applied, one confirmed already safe.

### Tests run

- `tests/test_analog_safety_guards.py` (new) - 25 passed
- Full suite - **415 passed, 1 failed** (pre-existing `reedsolo` gap)

### Production changes

- `evaluation/metrics.py` (140 -> 149): `_FAMILY_BY_V1_FAMILY` gains `pam -> QAM-like`
  and `analog -> analog-like`, matching fusion's vocabulary.
- `synthetic_gen/v1/generator.py` (387 -> 400): new `_eb_n0_db` helper returning `None`
  for bit-less captures instead of `-inf`.
- `evaluation/symbol_rate_experiment.py`: **no change** - already digital-pinned.

### Important observations

- **Entry 018 correction 1:** the family-lookup crash was not analog-only.
  `expected_family_for("pam")` also raised `KeyError`, and PAM4 is already in the
  registry, so the defect was live today rather than hypothetical.
- **Entry 018 correction 2:** `symbol_rate_experiment` never imported the registry
  `MODULATIONS`; it has its own digital list at line 39 and could not have widened.
  Locked with five guard tests instead of a code change.
- One pre-existing test (`test_expected_family_rejects_an_unknown_family`) used
  `"analog"` as its unknown-family example and was updated deliberately.
- Frozen V1 unchanged; no analog classes exist in the registry.

---

## Run 013

- **Date:** 2026-09-09
- **Timestamp:** 00:57:46 IST
- **Task:** Implement AM-DSB analog synthetic generation + independent oracle
  (BANK.md Entry 020). AM-SSB and WBFM not implemented.
- **Result:** PASS - one new file, zero existing production files modified.

### Tests run

- `tests/test_synthetic_am_dsb.py` (new) - **31 passed**
- Full suite - **446 passed, 1 failed** (pre-existing `reedsolo` gap)

### Production change

`src/radiofry/synthetic_gen/v1/analog.py` (240 lines, new). No other production file
touched - `config.py` 204, `modulation.py` 97, `generator.py` 400, `analog_demod.py` 30,
`dispatch.py` 99 all unchanged.

### Key results

- AM-DSB oracle: carrier at the expected frequency (0 Hz and 12 kHz), symmetric sidebands
  matching within 25%, envelope recovery **correlation > 0.99** on a clean capture, and
  graceful degradation to > 0.60 at 10 dB SNR.
- Ground truth carries `capture_kind`, `analog` and `message` blocks; every bit-derived
  field is explicitly null.
- Frozen V1 SHA-256 unchanged; digital registry unchanged; symbol-rate experiment still
  digital-only.
- No analog artifacts written to the repository - tests use `tmp_path`.

### Important observations

- Kept `ANALOG_MODULATIONS` as a **separate registry** from `config.MODULATIONS`. This
  structurally prevents the three `tuple(MODULATIONS)` defaults from widening, resolving
  Entry 018's risk #1 without patching three call sites.
- `AnalogSampleSpec` has no symbol-rate, sps, num_symbols or bits_seed field at all,
  rather than nulling them after the fact.
- The oracle validates the generator only. No claim is made about production analog
  demodulation - Entry 018 facts 3 and 4 (fake bits, symbol-rate decimation) still stand.
- `research_memory/CURRENT.md` is stale: it lists the 8-class training run as pending,
  but Entry 017 completed it.

---

## Run 014

- **Date:** 2026-09-09
- **Timestamp:** 01:09:37 IST
- **Task:** Validate an AM-DSB capture end-to-end through the evaluation harness
  (BANK.md Entry 021). No production analog demodulation.
- **Result:** PASS - analog safety path proven; two required fixes made.

### Tests run

- `tests/test_analog_harness_safety.py` (new) - **15 passed**
- Full suite - **461 passed, 1 failed** (pre-existing `reedsolo` gap)

### Production changes

- `synthetic_gen/v1/generator.py` (400 -> 405): `load_ground_truth` returns `None` bits
  for a null bits block instead of crashing.
- `evaluation/harness.py` (327 -> 358): `has_source_bits` flag; BER forced unavailable
  with reason `analog_no_transmitted_bits`; `expected_bits`/`bit_count_ratio` null;
  new `UNAVAILABLE_METRICS["bit_error_rate_analog"]`.

### Important observations

- **Entry 018's prediction was wrong about the first blocker.** The crash was
  `load_ground_truth` subscripting a null `bits` block at harness line 111 - which fires
  *before* `expected_family_for` on line 123. Traced, not assumed.
- Observed record: `expected_family='analog-like'`, `ber_status='unavailable'`,
  `ber_reason='analog_no_transmitted_bits'`, `ber_strict=None`, `expected_bits=None`,
  `es_n0_db=None`, `truth_symbol_rate_hz=None`. All seven required validations hold.
- **Caveat recorded, not hidden:** the CNN labelled the capture `PAM4` (0.462), which has
  no dispatch route, so demodulation never ran and the fake-bit branch was not exercised
  end-to-end. Two extra tests were added to exercise the guard directly instead of
  claiming coverage that run did not have.
- Parameter estimation still emits a meaningless `est_symbol_rate_hz` (1806.6 Hz) for
  analog; harmless, since there is no truth value to compare against. Not acted on.
- `research_memory/CURRENT.md` remains stale (Entry 017 run listed as pending).

---

## Run 015

- **Date:** 2026-09-09
- **Task:** Implement AM-SSB synthetic generation plus an independent sideband oracle
  (BANK.md Entry 022). Generation only - no production demodulation, no WBFM.
- **Result:** PASS.

### Tests run

- `tests/test_synthetic_am_ssb.py` (new) - **28 passed**
- Both analog generation files together - **59 passed**
- Full suite - **489 passed, 1 failed** (pre-existing `reedsolo` gap)

### Production change

- `synthetic_gen/v1/analog.py` (240 -> 293): `AM-SSB` in the analog registry,
  `sideband` field with USB default, `modulate_am_ssb()` via the analytic signal,
  `modulate_analog()` dispatcher, generalised ground-truth `modulation`/`analog` blocks.

### Measured

- Unwanted-sideband suppression per tone: **140.1 / 171.1 / 175.7 dB** (target >= 30 dB).
- Total energy above vs below carrier: **72.6 dB**.
- Independent product-detector recovery correlation **> 0.99** for USB and LSB.

### Notes

- **Sideband convention: USB is the default**, LSB configurable, recorded in ground truth.
- Three Entry 020 tests were deliberately updated because they asserted "AM-SSB is not
  implemented". Nothing was loosened or deleted.
- The oracle validates generation only. AM-SSB captures still never reach
  `demodulate_ssb` in production (Entry 021 routing problem is unchanged).
- `research_memory/CURRENT.md` still stale; reported, not modified.

---

## Run 016

- **Date:** 2026-09-09
- **Task:** Post-AM-SSB housekeeping and documentation audit (BANK.md Entry 023).
  Documentation only - no DSP/ML change.
- **Result:** PASS, with two documentation corrections recorded.

### Checks performed

- Entries 021/022 audited against the code: all named files exist at stated sizes, guard
  symbols present in `harness.py`, test counts re-measured (28 / 31 / 15) and matching.
- Registry regression: digital 8 classes, analog `['AM-DSB','AM-SSB']`,
  `symbol_rate_experiment` six digital names, WBFM absent.
- Frozen V1 integrity reproduced over 40 `.iq` files.
- Read-only `git status` on branch `main`.

### Corrections found

1. The frozen V1 "SHA-256" quoted in six entries is the **first 32 characters** of
   `d6d3f918687d0700a43e46211c3f04b9ef74be9d8b232eac5d0e4cf4bf2390ab`, not an MD5 and not
   a full digest. History left intact; full digest now in `CURRENT.md`.
2. The "489 passed" full-suite figure requires `--ignore=tests/test_model_report.py`;
   without it, collection aborts on a missing `h5py`. Two environmental gaps, not one.

### Files changed

- `research_memory/CURRENT.md` - rewritten (was stale since Entry 016)
- `research_memory/INDEX.md` - added Entry 017, an Analog section (018-022), Entry 023
- `BANK.md` - Entry 023 appended
- `ANTIGRAVITY_COMMIT.txt` - rewritten; it named the wrong branch and listed
  already-committed files
- `OP.md` - this run

### Tests

- Analog focused: **74 passed**
- Full suite: **489 passed, 1 failed** (pre-existing `reedsolo`), `test_model_report.py`
  ignored (pre-existing `h5py`)

### Remaining issues

- `scratch.py` and `commit_msg.txt` are spent one-off files; flagged, not deleted.
- Analog still does not route to an analog demodulator (Entry 021 blocker, unchanged).

---

## Run 017

- **Date:** 2026-09-09
- **Task:** Implement WBFM synthetic generation plus an independent FM oracle
  (BANK.md Entry 024). Generation only - no classification, fusion or routing change.
- **Result:** PASS.

### Tests run

- `tests/test_synthetic_wbfm.py` (new) - **34 passed**
- Analog focused (4 files) - **108 passed**
- Full suite - **523 passed, 1 failed** (pre-existing `reedsolo`), `test_model_report.py`
  ignored (pre-existing missing `h5py`)

### Production change

- `synthetic_gen/v1/analog.py` (293 -> 367): WBFM in the analog registry,
  `frequency_deviation_hz` with Nyquist validation, `modulate_wbfm()` by phase
  integration, FM fields in the ground-truth analog block.

### Measured

- Peak `abs(f_i - f_c)` = **15000.000 Hz** against a configured 15000.0; envelope
  constant to 4.2e-08.
- Independent discriminator recovery: correlation **1.000000**, NRMSE 8.7e-08.
- Recovered tones within **0.25-1.92 Hz** of the recorded tone list.
- beta = 6.07; measured 99% occupied bandwidth **36926 Hz** vs Carson's approximate
  **34945 Hz** (ratio 1.057). AM-DSB 4950 Hz, AM-SSB 2063 Hz for the same message.
- SNR sweep 40 -> -5 dB: 0.9987, 0.9870, 0.8883, 0.5120, 0.2822, 0.1167, 0.0481 -
  monotone.

### Pipeline observation (evidence only, not patched)

- Classical detector calls WBFM **`analog-like`** - correct, and better than its
  `QAM-like` verdict on AM-DSB in Entry 021.
- CNN calls it **`BPSK`** (0.407); fusion follows; dispatch runs the **2PSK**
  demodulator and returns bits.
- **First analog capture to reach a demodulator end-to-end** - and the Entry 021 guard
  held: `ber_status="unavailable"`, `compared_bits=0`, no fabricated BER.

### Notes

- Five prior tests asserting "WBFM is not implemented" were deliberately updated;
  nothing loosened or deleted.
- Frozen V1 unchanged (`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

---

## Run 018

- **Date:** 2026-09-09
- **Task:** Minimal analog-aware fusion fallback (BANK.md Entry 026), plus recording the
  Entry 025 forensic investigation. No threshold, CNN, dispatch or preprocessing change.
- **Result:** PASS functionally; **limited practical benefit**, reported honestly.

### Tests run

- `tests/test_fusion_analog_fallback.py` (new) - **19 passed**
- Focused regression (9 files: fusion, QAM routing, FSK timing, QAM scale, analog
  safety, AM-DSB, AM-SSB, WBFM) - **193 passed**
- Full suite - **542 passed, 1 failed** (pre-existing `reedsolo`),
  `test_model_report.py` ignored (pre-existing missing `h5py`)

### Production changes

- `fusion/confidence_fusion.py`: `ANALOG_LABELS`, `_recover_analog_alternative()`, new
  optional `ranked_alternatives` kwarg, new defaulted `FusionResult.analog_fallback`.
  Fallback fires only when the CNN's digital top-1 is already rejected AND the classical
  family is `analog-like`.
- `pipeline.py`: passes CNN confidences through as `ranked_alternatives`; no second
  inference pass.

### Measured

- Multi-seed (5 seeds x 3 schemes): fallback fired **1 of 15**; WBFM Unclassified rate
  4/5 -> 3/5. The single firing selected **AM-SSB for a WBFM capture - wrong**.
- AM-DSB/AM-SSB unaffected: the gate never opens because the classical detector says
  `QAM-like` in 10/10 cases, even though an analog label is in the CNN top-k in 10/10.
- Frozen V1: gate open 0/40, fallback fired 0/40, fusion label changed 0/40. Hash
  `d6d3f918687d0700a43e46211c3f04b9` unchanged.

### Notes

- `predict_modulation` defaults to `top_k=3`, so fusion sees only two alternatives - a
  structural limiter on the fallback, left unchanged.
- Dominant remaining blocker has moved upstream to `dsp/cyclostationary.py`: AM-DSB
  misses the `analog-like` branch by 0.007 of `amplitude_cv`.

---

## Run 019

- **Date:** 2026-09-09
- **Task:** Positive family-level analog evidence in `dsp/cyclostationary.py`
  (BANK.md Entry 027). No CNN, fusion-threshold, dispatch, preprocessing or generator
  change.
- **Result:** PASS - measured improvement with zero digital false positives.

### Tests run

- `tests/test_classical_analog_detection.py` (new) - **40 passed**
- Full suite - **582 passed, 1 failed** (pre-existing `reedsolo`),
  `test_model_report.py` ignored (pre-existing missing `h5py`)

### Production change

- `dsp/cyclostationary.py`: `ANALOG_FREQUENCY_CV_MAX = 0.9`; new analog branch placed
  after FSK/PSK and before QAM, requiring `frequency_cv < 0.9` **and**
  `fourth_power_line < 0.2`; evidence-based confidence 0.5-0.9; final `else` now returns
  `"unknown"` instead of `analog-like`.

### Measured

- Feature choice was evidence-driven: 7 features compared over 72 captures; only
  `frequency_cv` separated. Three purpose-built candidates were measured and rejected.
- Threshold from a held-out sweep: lowest digital `frequency_cv` anywhere is **1.032**
  (BFSK sps=4); T=0.9 gives **0/640 false positives** with 13% margin. T=1.05 breaks it.
- Analog detection at >=10 dB: AM-DSB@20k 15/15, AM-SSB 15/15, WBFM 15/15. Below 10 dB
  it fails - noise makes all analog look impulsive.
- Digital controls (800 captures): **0 analog false positives; not one family changed.**
- Frozen V1 (40): analog-like 0, unknown 0, hash `d6d3f918687d0700a43e46211c3f04b9`.

### End-to-end with the Entry 026 fallback (5 held-out seeds, 20 dB)

- **AM-DSB@20k: classical 0/5 -> 5/5, fallback fires 5/5, 4/5 correctly labelled
  `AM-DSB` and reaching `demodulate_am`.** Entry 026 alone achieved 0/5.
- AM-SSB: now analog 5/5, but still 0/5 correct - the CNN confidently says `WBFM`.
- WBFM: unchanged, 0/5 correct.
- Correct analog labels overall: **6 of 20**. `analog-like` appearing is not claimed as
  success.

### Notes

- **AM-DSB at carrier offset 0 is unrecoverable at this layer**: raw `frequency_cv`
  0.0000 -> 7.0457 after DC removal, more impulsive than any digital control. Documented
  and pinned by a test; no workaround was added.
- Dominant analog blocker has moved from the classical detector to **CNN quality**.

---

## Run 020

- **Date:** 2026-09-09
- **Task:** Make analog dispatch symbol-rate independent (BANK.md Entry 028). Dispatch
  only - no CNN, fusion, detector, preprocessing or generator change.
- **Result:** PASS.

### Tests run

- `tests/test_analog_dispatch_bypass.py` (new) - **32 passed**
- Full suite - **614 passed, 1 failed** (pre-existing `reedsolo`),
  `test_model_report.py` ignored (pre-existing missing `h5py`)

### Production change

- `decoding/demodulators/dispatch.py`: `ANALOG_LABELS` constant; new
  `_demodulate_analog()` doing full-rate analog demodulation; one dispatch line placed
  **before** the symbol-rate requirement. Digital branch untouched, dead analog branch
  removed from it.

### Measured

- WBFM message recovery through dispatch: **0.1454 -> 1.0000**.
- AM-DSB recovered tones: 2 of 3 aliased before (2326->1004 Hz, 2836->1519 Hz), all 3
  correct after.
- AM/SSB "before" correlations of 1.0000 were against a **decimated** reference and are
  misleading; recorded as such rather than as evidence the old path worked.
- SSB at full rate: true carrier **+1.0000**, estimated 22147 Hz **+0.0115** - dispatch
  fixed, carrier estimation still broken.
- End to end: symbol rate used in **0 of 12** analog cases (including estimates of
  244 Hz and 14526 Hz). **AM-DSB@20k recovers 0.94 correlation through production.**
- Correct labels still 4/12 - classification was not addressed and no success is claimed.
- Frozen V1 `d6d3f918687d0700a43e46211c3f04b9`; BER guard intact.

### Notes

- Two prior tests asserted the removed behaviour and were deliberately updated; the SSB
  one now compares against the full-rate message, a strictly stronger assertion.
- AM-DSB@0 is labelled and routed correctly but recovers ~0.00 - preprocessing removed
  its carrier. Not a dispatch problem; not fixed here.

---

## Run 021

- **Date:** 2026-09-09
- **Task:** Improve AM-SSB carrier estimation (BANK.md Entry 029).
- **Result:** **NEGATIVE RESULT - no production change made.** Blind SSB carrier
  estimation cannot reach the required accuracy; forcing a change would have improved a
  number without improving capability.

### Reproduced

- True carrier 20000.0 Hz; estimated **21859.5 Hz** (+1859.5). Recovery: true carrier
  **+1.0000**, estimated **+0.0066**.
- Cause identified exactly: `parameter_estimation.py:119` uses the **PSD centroid**,
  which equals the carrier only for a symmetric spectrum. For one-sided SSB it lands at
  carrier + message centroid - predicted 21859.8 vs measured 21859.5 Hz.

### Decisive measurement

- SSB recovery needs the carrier to **~1-2 Hz** (corr 0.99 at 1 Hz, 0.21 at 10 Hz, gone
  by 20 Hz).
- Best signal-only candidate (occupied-band edge): **~500 Hz noiseless**, and
  **-16 kHz at 20 dB / -108 kHz at 10 dB**. Finer FFT resolution makes it worse.
- The carrier sits below the lowest message tone and carries no power; that gap varies
  per capture (536-1053 Hz across seeds) and is not observable.

### Candidates tested and rejected

PSD centroid, occupied-band edges, finer FFT resolution, spectral skew for sideband
detection, sideband symmetry / peak-pair geometry, and assuming a fixed message
low-cutoff (rejected as disguised ground-truth leakage).

### Already-working legitimate path

`parameter_estimation.py:120-121` already prefers `metadata["center_frequency_hz"]`, and
`preprocess` preserves metadata. Verified: **error 0.0 Hz, SSB recovery 1.0000**, no code
change required. The ingestion layer just has to populate it from real capture headers.

### Digital baseline recorded (true carrier 0 Hz, 20 dB, seed 101)

BPSK -19.7, QPSK -167.5, 8PSK +32.7, 16QAM +123.3, 64QAM -44.4, BFSK +106.4,
GFSK +113.5 Hz. Any future estimator change must be measured against these.

### Verification

- Full suite **614 passed, 1 failed** (pre-existing `reedsolo`; `test_model_report.py`
  ignored for missing `h5py`) - unchanged.
- Frozen V1 `d6d3f918687d0700a43e46211c3f04b9`, 40 captures.
- `git status` / `git diff --stat` identical to the pre-investigation state.

---

## Run 022

- **Date:** 2026-09-09
- **Task:** Ingestion-layer centre-frequency metadata (BANK.md Entry 030), following the
  Entry 029 negative result. Ingestion only.
- **Result:** PASS - SSB now recovers end to end when legitimate capture metadata exists.

### Tests run

- `tests/test_ingestion_sigmf_metadata.py` (new) - **32 passed**
- Full suite - **646 passed, 1 failed** (pre-existing `reedsolo`),
  `test_model_report.py` ignored (pre-existing missing `h5py`)
- No existing test required modification.

### Production change

- New `ingestion/sidecar.py`: minimal SigMF `.sigmf-meta` reader for `core:frequency`
  and `core:sample_rate` only. Returns `{}` on missing/malformed input; rejects
  non-numeric, bool, NaN, inf, negative and absurd values; accepts 0.0 (baseband).
- `iq_parser.read_iq` and `wav_parser.read_wav` consult it and set
  `center_frequency_hz` + `center_frequency_source` **only** when a valid value exists.

### Measured

- **The gap was that no parser ever set `center_frequency_hz`** - the estimator already
  consumed it and `preprocess` already preserved it.
- SSB with sidecar: carrier error **0.0 Hz** and recovery **0.9948-1.0000** across USB,
  LSB, two seeds, noiseless and 20 dB (8 cases).
- SSB without sidecar: error 1353-1860 Hz, recovery -0.20..+0.14 - **Entry 029's negative
  result stands** and is pinned by a test.
- Digital carrier estimates identical to the Entry 029 baseline (max delta 0.05 Hz).
- Frozen V1 `d6d3f918687d0700a43e46211c3f04b9`; V1 still uses the centroid path.

### Anti-leakage

The generator writes no `.sigmf-meta`; production ingestion never reads the ground-truth
JSON. Three dedicated tests enforce this, including one proving a `cap.json` containing
`core:frequency` is ignored.

### Notes

- WAV headers carry no RF tuning field; a sidecar is the honest route. SDR# `auxi` chunks
  were judged disproportionate and are not parsed.
- CNN analog misclassification remains the dominant blocker: most SSB captures still
  never reach `demodulate_ssb` in production.

---

## Run 023

- **Date:** 2026-09-09
- **Task:** Analog routing dominance forensic investigation (BANK.md Entry 031).
  Read-only.
- **Result:** Cause identified; a conservative gate validated at 100% / 0% on held-out
  data. **No production change made** - recommendation recorded for a follow-up.

### Experiment

96 captures (4 analog configs + 8 digital controls, 2 SNR, 4 seeds) traced through
ingestion -> preprocessing -> parameter estimation -> classical detector -> CNN top-k ->
fusion -> dispatch -> demodulation, plus a held-out run on unseen seeds 211-233.

### Findings

- End-to-end correct analog labels: **3/32**.
- Loss mechanism: **11/32** true label absent from CNN top-3; **10/32** out-ranked below
  threshold; **8/32** a wrong label >= 0.4 accepted; 3/32 rescued by the fallback.
- **AM-SSB LSB: the CNN never emits `AM-SSB` in any of 8 captures.** No re-ranking can
  fix that.
- Dominance condition, exactly: fusion accepts `ml_label` whenever confidence >= 0.4, and
  the Entry 026 fallback requires the top-1 to be a *rejected digital* label. A wrong
  analog label above threshold (AM-SSB called WBFM at 0.44-0.49) is accepted regardless
  of a 0.73-0.76 confidence `analog-like` verdict.
- **Separability:** instantaneous-frequency statistics do NOT separate the analog types.
  `env_flat` separates WBFM (0.54-0.58) from AM (<= 0.33); `amp_cv` separates AM-DSB
  (0.22-0.30) from AM-SSB (0.45-0.48).
- Candidate gate (`env_flat > 0.40` -> WBFM; `amp_cv >= 0.37` -> AM-SSB; else AM-DSB),
  consulted only behind the classical `analog-like` verdict: **60/60 correct** on
  held-out seeds; **0/480 digital captures could reach it**.
- **Limitation:** the AM-DSB/AM-SSB boundary is depth-dependent - AM-DSB at modulation
  depth >= 0.9 (`amp_cv` 0.379) is misread as AM-SSB. Valid to depth ~0.8; project
  default is 0.5.

### Verification

- Repository untouched: `git status` 26 entries and `git diff --stat -- src/ tests/`
  9 files / 201 insertions, both identical to the Entry 030 state.
- Frozen V1 `d6d3f918687d0700a43e46211c3f04b9`.
