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
