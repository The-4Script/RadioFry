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

---

## Run 024

- **Date:** 2026-09-09
- **Task:** Conservative analog subtype routing (BANK.md Entry 032).
- **Result:** PASS. Analog subtype routing 55/60 (55/55 wherever the gate engaged);
  the gate itself contributes 0/360 digital false positives.

### Production changes

- `dsp/cyclostationary.py`: `_envelope_flatness()`; `envelope_flatness` added to the
  evidence dict. Family decision untouched.
- `fusion/confidence_fusion.py`: `select_analog_subtype()` +
  `ANALOG_ENVELOPE_FLATNESS_MAX=0.40` / `ANALOG_AMPLITUDE_CV_SSB_MIN=0.37`; optional
  `classical_evidence` / `classical_confidence`; new defaulted `analog_route` field.
- `pipeline.py`: passes classical evidence and confidence through.

### Rule

Gate on `classical_family == "analog-like"` (Entry 027), then
`envelope_flatness > 0.40 -> WBFM`, `amplitude_cv >= 0.37 -> AM-SSB`, else `AM-DSB`.
The CNN gets no vote once the gate opens.

### Measured (unseen seeds 307-331, SNR 20/15/10)

- Analog correct **55/60 = 91.7%**; **55/55** wherever the gate engaged.
- AM-SSB LSB 20/15 dB: **10/10 correct** - was 0/8 in Entry 031 with the label absent
  from every CNN top-3.
- The 5 failures are AM-SSB LSB at 10 dB, where the classical detector says `QAM-like`
  and the gate never opens. Detector limitation, not a rule failure.
- Digital controls (360): `analog_route` was `""` in **all 360** - the gate never fired.
- **6/360 digital captures still route to WBFM**, all GFSK at sps=16 where the CNN emits
  WBFM at 0.52-0.90. Verified identical with and without the gate: **pre-existing, not
  caused by this entry**, and out of scope to fix.

### Tests

- `tests/test_analog_subtype_routing.py` (new) - **57 passed, 1 skipped**
- Full suite - **703 passed, 1 failed, 1 skipped**; the failure is the pre-existing
  `reedsolo` gap, `test_model_report.py` ignored for missing `h5py`.
- Three prior tests updated: two asserted an exact 3-key evidence set (now subset
  checks); the pipeline open-set test failed on a stub without `.evidence` and was fixed
  **in production** via `getattr`, leaving the test unmodified.

### Notes

- Confidence for a routed label is the classical **family** confidence, explicitly not a
  calibrated subtype probability. No ML calibration was invented.
- V1 hash `d6d3f918687d0700a43e46211c3f04b9` unchanged.

---

## Run 025

- **Date:** 2026-09-09
- **Task:** Final synthetic end-to-end benchmark (BANK.md Entry 033). Measurement only.
- **Result:** Benchmark complete. **Recommendation: TARGETED FIX** - one safety defect
  blocks a freeze; retraining is not justified.

### Methodology

350 captures (200 digital at sps=8, 150 analog incl. SSB with/without SigMF metadata),
5 SNR levels, fresh seeds 401-431. Production checkpoint is the 11-class RadioML model,
never trained on RadioFry synthetic data - no leakage. No ground truth entered the
pipeline.

### Headline

- Digital top-1 **59.0%**, top-3 **96.0%**; **top-3 at >=10 dB = 100.0%**.
- Analog **54.0%** overall, **88.9% at >=10 dB**; **81/81 correct whenever the analog
  gate engaged**.
- Digital falsely routed as analog at sps=8: **0/200**.
- System: correct+confident **56.9%**, rejected **30.0%**, false confident **13.1%**.

### Critical findings

- **GFSK -> WBFM is systematic and oversampling-driven**: 0/25 at sps 4 and 8, 8/25 at
  sps 16, **19/25 = 76% at sps 32** with CNN confidence to **0.999**. `analog_route` is
  `""` throughout - the Entry 032 gate is not involved; fusion accepts a confident CNN
  over a correct `FSK-like` verdict. **Severe enough to block a freeze.**
- **AM-SSB LSB is a detector problem, not demodulation**: reliable to 15 dB, cliff to
  0/10 at 10 dB where classical returns `QAM-like`. USB has one SNR step more margin.
  Whenever the gate engaged, LSB recovered at 0.985-0.995 with metadata.
- **SSB metadata is decisive for recovery**: ~0.00 blind vs **0.9951 / 0.9847 / 0.9536**
  with metadata. Confirms Entries 029/030.
- **PAM4 never demodulates** (no dispatch route); **GFSK sits at ~0.47 BER even when
  correctly labelled** (order-2 FSK demodulator ignores the Gaussian pulse).
- No analog BER fabricated anywhere: 0/150.

### Parameter estimation

Symbol rate within 1%: **80.5%**. Digital carrier median |err| 147 Hz. Analog carrier
median |err| **2147 Hz blind vs 0.0 Hz with metadata**. SNR estimate biased low
(true 20 dB -> 12.2 dB).

### Verification

- Full suite **703 passed, 1 failed, 1 skipped**; failure is the pre-existing `reedsolo`
  gap, `test_model_report.py` ignored for missing `h5py`. Unchanged.
- V1 hash `d6d3f918687d0700a43e46211c3f04b9` - **MATCH**.
- No production code modified.

### Recommendation

**TARGETED FIX**, four items, none requiring training: (1) fusion must refuse a CNN
analog label when classical asserts a digital family - blocks the freeze; (2) add a PAM4
dispatch route; (3) fix GFSK demodulation; (4) extend analog detection to 10 dB for LSB.
Re-benchmark after (1).

---

## Run 026

- **Date:** 2026-09-09
- **Task:** Fusion safety gate - a classical digital family blocks CNN analog labels
  (BANK.md Entry 034). Closes the Entry 033 freeze blocker.
- **Result:** PASS. GFSK -> WBFM **27/50 -> 0/50**; digital false-analog routing
  **128/600 -> 1/600**; Entry 032 analog routing byte-for-byte preserved.

### Production change

- `fusion/confidence_fusion.py` only: `DIGITAL_FAMILIES` (PSK/FSK/QAM-like; `unknown`
  excluded), `DIGITAL_FAMILY_MIN_CONFIDENCE = 0.5` (measured floor - real digital
  verdicts score 0.557-1.000), `_highest_ranked_digital()`, the guard, and a new
  defaulted `digital_family_block` field. `pipeline.py` needed no change.

### Fallback

Retain the CNN's own highest-ranked **digital** alternative if it clears the existing 0.4
threshold; otherwise **Unclassified**. The family is never turned into a subtype. Chosen
because a wrong analog demodulation is a wrong-domain error, while a rejection costs only
an answer - and because `GFSK` was already in the CNN top-3 in **all 31** measured cases.

### Measured

| Metric | BEFORE | AFTER |
|---|---|---|
| GFSK -> WBFM (sps 16 + 32) | 27/50 | **0/50** |
| digital false-analog routing | 128/600 | **1/600** |
| digital top-1 | 33.5% | 34.3% |
| digital top-3 | 71.7% | 71.7% |
| digital rejection | 21.8% | 41.8% |
| analog overall | 56/100 | **56/100** |

127 fused labels changed, **all previously analog**; 5 newly correct, **0 newly wrong**.
Analog gate engaged 56/100, correct when engaged **56/56**.

Residual: 1/600, CPFSK sps=32 seed 421, where classical returned `unknown` @ 0.200 - a
non-verdict, deliberately not a blocking condition.

### Tests

- `tests/test_fusion_digital_family_guard.py` (new) - **28 passed**
- Full suite - **731 passed, 1 failed, 1 skipped** (pre-existing `reedsolo`;
  `test_model_report.py` ignored for missing `h5py`)
- One Entry 032 test deliberately updated: it had recorded this leak as pre-existing and
  unfixed, and would otherwise assert the bug.

### Notes

- No claim that the CNN improved. Fusion now refuses to let a CNN analog prediction beat
  an independent classical digital verdict.
- Entry 033 items 2-4 remain open: PAM4 dispatch route, GFSK demodulation, LSB detection
  at 10 dB.
- V1 hash `d6d3f918687d0700a43e46211c3f04b9` unchanged.

---

## Run 027

- **Date:** 2026-09-09
- **Task:** PAM4 demodulation and dispatch (BANK.md Entry 035). Closes Entry 033 item 2.
- **Result:** PASS. PAM4 demodulation success **0/25 -> 10/10 at every SNR**, BER
  0.00000 at 20 dB.

### Did it already exist?

**No.** No PAM implementation existed anywhere in `decoding/demodulators/`, and dispatch
had no PAM4 branch - the label fell through to "No demodulator is registered".

### Production changes

- `decoding/demodulators/pam_demod.py` - **new**, `demodulate_pam()`.
- `decoding/demodulators/dispatch.py` - one import, one `elif` branch. Nothing else;
  `pipeline.py` and `harness.py` needed no change.

### Mapping and method

Mapping taken from the generator, not invented: levels `[-3,-1,1,3]` / sqrt(5),
**natural binary MSB-first** (NOT Gray). Axis estimated blind via
`angle(mean(x^2))/2`; scale rescaled to the grid's average power (avoiding the Entry 007
QAM defect). Timing reuses the existing `_linear_timing_offset` - no new framework.

### Measured (10 seeds, sps=8)

| SNR | demod ok | median BER |
|---|---|---|
| clean | 10/10 | **0.00000** |
| 20 dB | 10/10 | **0.00000** |
| 15 dB | 10/10 | 0.00024 |
| 10 dB | 10/10 | 0.02051 |
| 5 dB | 10/10 | 0.13330 |
| 0 dB | 10/10 | 0.26343 |

Estimated symbol rate gives **identical** BER to the true one. Amplitude-scale delta
**0.000000** over gains 1e-4 to 1e+4. Cross-check at 20 dB: PAM4 0.00000, on par with
BPSK/QPSK/QAM16 and better than QAM64 (0.01709).

### Tests

- `tests/test_pam4_demodulation.py` (new) - **44 passed**
- Full suite - **775 passed, 1 failed, 1 skipped** (pre-existing `reedsolo`;
  `test_model_report.py` ignored for missing `h5py`)
- One test corrected during development: it asserted full rotation invariance, which
  measurement showed is physically impossible for a symmetric PAM constellation. It now
  asserts the axis is recovered, with a separate test pinning correct polarity when
  unrotated.

### Notes

- **180-degree polarity ambiguity is fundamental** and shared with BPSK; documented, not
  solved. Fixing it means adding phase/differential reference handling across all digital
  demodulators.
- This fixes demodulation, not classification - PAM4 top-1 is still 6/25 (Entry 033).
- V1 hash `d6d3f918687d0700a43e46211c3f04b9` unchanged.

---

## Run 028

- **Date:** 2026-09-09
- **Task:** GFSK demodulation (BANK.md Entry 036).
- **Result:** **NEGATIVE RESULT - no production change.** The Entry 033 attribution was
  wrong; the ~0.47 BER is symbol-rate estimation, not GFSK demodulation, and it affects
  CPFSK identically.

### What the investigation found

- **GFSK demodulation with the true symbol rate is already good**: 0.0044-0.0314
  noiseless, 0.0049-0.0275 at 20 dB across samples-per-symbol 4/8/16/32. Noiseless at
  sps=8, GFSK makes 12 bit errors against CPFSK's 56 - GFSK is *better*.
- **Root cause is symbol-rate estimation.** Same capture, same demodulator: true Rs gives
  0.0137, estimated Rs gives 0.5108 (estimator returned 1880 Hz for a true 25000 Hz).
  **BFSK/CPFSK collapses identically** above sps=8. The only working configuration is
  BFSK sps=8 at high SNR - exactly V1, which is why it stayed hidden.
- **A small genuine GFSK timing weakness exists**: `_fsk_timing_offset` picks a
  straddling offset in 3/40 GFSK cases (1 at sps=8, 2 at sps=32) and 0/40 CPFSK cases,
  because the Gaussian pulse flattens the variance minimum. Brute-forcing the offset
  recovers those seeds to 0.039-0.043.
- **A candidate replacement criterion was implemented and rejected on measurement**:
  it failed 5-8/10 against the existing criterion's 0-2/10.

### Why nothing was changed

A GFSK-specific fix cannot address a defect that is neither GFSK-specific nor in the
demodulator, the demodulator is already within ~2x of CPFSK, and the only justifiable
timing change measured worse. Chasing a 3/40 edge case while the real 0.47 sits in the
estimator would be motion without progress.

### Generator facts (understanding only, never runtime)

BT default 0.3, unit-area Gaussian pulse spanning 4 symbols (33 taps at sps=8), h=0.5,
deviation = h*Rs/2. ISI is real: per-symbol phase-step separation 13.6 sigma (BFSK) ->
4.8 sigma (GFSK). BT sweep: 0.2 -> 0.1095, 0.3 -> 0.0137, 0.5 -> 0.0117.
Amplitude-scale: exactly invariant (0.0137 at gains 1e-4 / 1 / 1e4).

### Routing vs demodulation

Entry 034 holds - no GFSK capture reached an analog demodulator in any end-to-end trace;
all reached `2FSK`. Routing is fixed; demodulation is capable; the symbol rate is what is
missing.

### Verification

- Full suite **775 passed, 1 failed, 1 skipped** - unchanged.
- V1 hash `d6d3f918687d0700a43e46211c3f04b9` - MATCH.
- `git diff --stat -- src/` identical to the Entry 035 state.

### Recommendation

Next task: **FSK symbol-rate estimation** for CPFSK *and* GFSK across sps 4-32. Entry 014
closed this as a low-SNR-only negative result; that conclusion needs revisiting, because
the failure is now measured at 20 dB.

---

## Run 029

- **Date:** 2026-09-09
- **Task:** FSK symbol-rate estimation investigation and targeted fix (BANK.md Entry 037).
- **Result:** **NEGATIVE RESULT - no production change.** The symbol-rate line is at the
  noise floor for GFSK at all oversampling and for CPFSK above sps=8.

### Findings

- **The harmonic hypothesis is wrong.** 25000/1880 = 13.3, 12500/854 = 14.6,
  6250/415 = 15.1 - not integer ratios. `phase_second_difference` finds the **exact**
  true rate; `envelope_power` wins on confidence with an unrelated low-frequency bump.
- **Root cause, quantified**: symbol-boundary line over noise floor is **42.4x / 16.7x**
  for CPFSK at sps 4/8 (where the estimator works) and **2.9-4.9x** for CPFSK at sps
  16/32 and **3.2-4.2x for GFSK at every sps** (where it fails). Gaussian frequency
  shaping exists to suppress transition energy, so it suppresses the feature the
  estimator needs.
- **Candidate fix tested and rejected**: gating out `envelope_power` for constant-modulus
  signals gave 80.6% -> 81.9% within 10%, improved only BFSK sps=16 (1/5 -> 3/5), and
  left **GFSK at 0/5 everywhere**. It would not have changed a single failing downstream
  BER row.
- **Linear modulations are already correct**: BPSK/QPSK/8PSK 20/20, PAM4 19/20,
  16QAM 19/20, 64QAM 16/20 across sps 4-32. The defect is FSK-only.
- **No usable rejection signal**: spurious estimates carry confidence 0.44-0.55, the same
  band as correct ones, so the estimator cannot flag its own failure.

### Downstream BER (20 dB, median)

BFSK sps 4/8: 0.0049/0.0117 with either rate. BFSK sps 16/32 and GFSK sps 8/16/32:
0.0157-0.0275 with the true rate, **0.50-0.51 with the estimated rate**.

### Verification

- Full suite **775 passed, 1 failed, 1 skipped** - unchanged.
- V1 hash `d6d3f918687d0700a43e46211c3f04b9` MATCH; `git diff --stat -- src/ tests/`
  identical to the Entry 035/036 state.
- Generator used only as an evaluation oracle; no ground truth inside estimation.

### Next step

Entry 038: scope a **cyclostationary** symbol-rate estimator for FSK (cyclic
autocorrelation / SCD at alpha = Rs), with an explicit go/no-go on lifting GFSK
downstream BER from ~0.50 toward ~0.015. If that fails too, document FSK above sps=8 as
an accepted capability limit.

---

## Run 030

- **Date:** 2026-09-09
- **Task:** Cyclostationary FSK symbol-rate feasibility and go/no-go (BANK.md Entry 038).
- **Result:** **GO for CPFSK/BFSK (narrow), NO-GO for GFSK.** Gated cyclic estimator
  implemented.

### Method

Cyclic autocorrelation of the instantaneous frequency, computed as an FFT of the lag
product so the whole cycle-frequency scan is a handful of FFTs. Theory predicted the
outcome in advance: the alpha = Rs term needs pulse excess bandwidth, which rectangular
CPFSK has and Gaussian GFSK (BT 0.3) does not.

### Feasibility (peak-to-background at true Rs, 20 dB)

CPFSK 15.5x / 23.9x / 12.9x / 2.8x at sps 4/8/16/32 - blind pick error **0.00 everywhere**.
GFSK 3.6x / 4.6x / 1.9x / 1.3x - blind pick wrong everywhere. Prediction confirmed.

### Observation length

BFSK sps 32 goes 4/5 -> 5/5 -> 5/5 and 2.6x -> 3.8x -> 5.2x at N 8192/16384/32768;
GFSK sps 16 reaches 4/5 only at N=32768; **GFSK sps 32 never recovers**.

### Gate

Ratio sweep over 120 FSK captures: gate 3.0 gives +9 correct but 4 regressions; **gate
6.0 gives +4 with ZERO regressions** and was chosen.

### Production change

`dsp/parameter_estimation.py` only: `_cyclic_symbol_rate()`, `CYCLIC_PEAK_RATIO_MIN=6.0`,
gated override reporting `symbol_rate_feature="cyclic_autocorrelation"`.

### Downstream BER - narrow but complete

**BFSK sps=16 @ 20 dB: 0.5135 -> 0.0176**, matching the true-Rs BER. **1 of 24 conditions
improved**; everything else unchanged. Stated plainly - the gate ships fewer fixes than
the raw estimator finds, by design.

### Regression

- Full suite **800 passed, 1 failed, 1 skipped** - failure is the pre-existing `reedsolo`
  gap, `test_model_report.py` ignored for missing `h5py`. No new failures.
- New `tests/test_cyclic_symbol_rate.py` - **25 passed**.
- **Cyclic gate fired on 0/120 linear-modulation captures** - linear path provably
  untouched.
- One prior test updated: it pinned the feature name to the three pre-FFT nonlinearities;
  a fourth legitimate source now exists.
- V1 hash `d6d3f918687d0700a43e46211c3f04b9` MATCH.

### Incidental finding

Linear modulations at sps=32 are weaker than Entry 037 implied (16QAM 1/5, PAM4 2/5 on
seeds 503-541). **Verified pre-existing** by stashing this change - identical numbers.
Previously unrecorded.

### Next

**Entry 039: the final synthetic benchmark**, not another estimator family. GFSK
symbol-rate estimation above sps 4 is now an accepted capability limitation.

---

## Run 031

- **Date:** 2026-09-09
- **Task:** Final synthetic end-to-end benchmark and freeze decision (BANK.md Entry 039).
- **Result:** **TARGETED BLOCKER.** Zero production changes.

### Benchmark

570 captures (480 digital = 8 classes x sps 4/8/16/32 x SNR 20/15/10/5/0 x 3 seeds;
90 analog incl. SSB with and without SigMF metadata). Fresh seeds **601/607/613**.
Config + results in `reports/benchmark_v039/` (gitignored). Leakage audit clean: no
truth references in any production module.

### Headline

- CNN top-1 **38.5%**, top-3 **73.5%**, fused top-1 **29.2%**, rejection **45.0%**,
  confident-wrong **25.8%**.
- **Rs correct -> median BER 0.0010; Rs wrong -> median BER 0.4881.** Symbol-rate
  estimation is the dominant failure mode; everything downstream works.
- By sps: **8 -> 59.2%**, 16 -> 36.7%, 4 -> 14.2%, **32 -> 6.7%**.
- Safety excellent: digital->analog **1/480 (0.21%)**, analog->digital **0/90**. The
  Entry 034 gate converted 87 confident-wrong analog labels into rejections.
- Analog: reliable at >=10 dB. **SSB recovery 0.995 with metadata vs 0.013 without** -
  classification identical either way.
- Report validation PASS; `.iq` and `.wav` give identical conclusions.

### New finding - corrects Entry 033

At sps 32, **CNN top-1 is 0/15 for five of six linear classes** and top-3 falls to 0/15
(8PSK). Oracle BER at the same cells is 0.0000-0.0928, so the information is present.
Entry 033 concluded "do not retrain" from a **sps=8-only** measurement; that evidence does
not hold across sps 4-32.

### Decision and next step

**TARGETED BLOCKER**, one entry: **Entry 040 - retrain the V2 CNN with samples-per-symbol
augmentation (4-32)**, and check whether the 128-sample inference frame is adequate at
high oversampling (4 symbols per window at sps 32 vs 32 at sps 4). Target: lift sps 4/32
fused accuracy from 14.2%/6.7% toward the 59.2% already achieved at sps 8. Do not reopen
symbol-rate estimation.

### Verification

Full suite **800 passed, 1 failed, 1 skipped** (pre-existing `reedsolo`;
`test_model_report.py` ignored for missing `h5py`). V1 hash
`d6d3f918687d0700a43e46211c3f04b9` MATCH. Zero production changes.

---

## Run 032

- **Date:** 2026-09-10
- **Task:** CNN SPS generalisation + inference window (BANK.md Entry 040).
- **Result:** **SHIP.** End-to-end fused accuracy **35.4% -> 99.5%**; sps 32
  **10.4% -> 97.9%**; confident-wrong **23.4% -> 0.5%**.

### Audit

Production `modulation_cnn.pt` = RML2016.10a, generated at a **single 8 sps**.
`train_v2_synthetic.py` had `SAMPLES_PER_SYMBOL = 8`. `ModulationCNN` ends in
`AdaptiveAvgPool1d(1)`, so input length is not fixed - the window hypothesis was testable
with zero architecture change.

### Two hypotheses, separated

- **Training distribution: confirmed.** But Entry 039's attribution was only half right -
  the larger step (43.3% -> 86.2%) is **domain mismatch**: the already-existing 8-class V2
  synthetic checkpoint beats the RadioML default by itself. SPS augmentation adds
  86.2% -> 99.2%.
- **Window length: REJECTED.** 128/256/512 give 99.2 / 99.2 / 98.8% test accuracy.
  Longer windows lower validation loss but do not help. **128 kept.**

### Production changes

- `training/train_v2_synthetic.py`: `SAMPLES_PER_SYMBOL_SWEEP = (4, 8, 16, 32)`,
  `CaptureSpec.samples_per_symbol`, constant capture sample count across the sweep.
- `pipeline.py`: `DEFAULT_MODULATION_MODEL` -> `modulation_cnn_v3_spsaug.pt`. RadioML
  checkpoint retained so the Entry 039 baseline stays reproducible.
- New checkpoint via the project's own pipeline: best epoch 35, val loss 0.2469,
  val acc 0.8972, 44800/16000/16000 frames.

### Verification

- Analog unaffected by the digital-only checkpoint: AM-DSB 6/6, AM-SSB 6/6, WBFM 6/6 -
  Entry 032's classical gate never consulted the CNN.
- BER conditional on correct classification: **median 0.0008 with a correct Rs**; 0.5058
  when Rs is wrong (the untouched Entry 037/038 limitation).
- Digital -> analog false routing 0/192 before and after.
- Suite **826 passed, 1 failed, 1 skipped** (pre-existing `reedsolo`;
  `test_model_report.py` ignored for missing `h5py`). New
  `tests/test_sps_generalisation.py` - 26 tests. V1 hash MATCH.
- Leakage: six disjoint seed blocks, none overlapping Entry 039's 601/607/613;
  capture-level splits; selection on validation loss only.

### Caveats recorded

- **99.5% is synthetic-to-synthetic** - not a real-world accuracy claim.
- `models_saved/` is gitignored, so the new default checkpoint is untracked; a fresh clone
  yields `Unclassified`. Pre-existing, but now on the critical path.
- `train_v2` does not write the `_metrics.json` sibling the loader requires; that is a
  separate `write_metrics()` call. Hit during this entry.

### Next

Re-run the Entry 039 benchmark against the new default and, if confirmed, declare
**SYNTHETIC FREEZE** and move to real-world data.
