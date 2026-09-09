# Current State

_Last updated: 2026-09-10, after BANK.md Entry 040 (CNN SPS generalisation)._

## Status by component

- **Synthetic V1 — FROZEN.** 40 captures, byte-identical since the freeze. Integrity check:
  SHA-256 over the concatenated, name-sorted `data/synthetic_v1/captures/*.iq` bytes =
  `d6d3f918687d0700a43e46211c3f04b9ef74be9d8b232eac5d0e4cf4bf2390ab`. BANK entries quote
  the **first 32 characters** of that digest as the short form. Do not regenerate.
- **V2.0 digital generator — 8 classes.** `config.MODULATIONS` =
  `16QAM, 64QAM, 8PSK, BFSK, BPSK, GFSK, PAM4, QPSK`. PAM4 and GFSK are generated and
  verified (Entry 016); PAM4 is classifiable but has **no dispatch route**, so it cannot
  be demodulated.
- **V2.0 8-class CNN — trained and evaluated (Entry 017).** Current baseline checkpoint:
  `models_saved/modulation_cnn_v2_8class.pt`. Inference uses 4-window mean-softmax.
- **AM-DSB — generated + harness-safe (Entries 020, 021).** Independent envelope oracle
  passes. The harness no longer crashes on a null `bits` block and reports
  `ber_status="unavailable"`, `ber_reason="analog_no_transmitted_bits"`.
- **AM-SSB — generated + oracle-validated (Entry 022).** Analytic-signal (Hilbert)
  construction, **USB is the default**, LSB configurable, recorded in ground truth as
  `analog.sideband`. Unwanted-sideband suppression measured at 140–176 dB per tone.
- **WBFM — generated + oracle-validated (Entry 024).** Phase-integral FM, constant
  envelope, configured peak deviation reproduced exactly (15000.000 Hz measured against
  15000.0 configured), noiseless recovery correlation 1.000000. Analog registry is now
  `['AM-DSB', 'AM-SSB', 'WBFM']` — complete, and deliberately separate from the digital
  `MODULATIONS` registry.

## Benchmark status (Entry 039, 570 captures, seeds 601/607/613)

- CNN top-1 **38.5%**, top-3 **73.5%**, fused top-1 **29.2%**, rejection **45.0%**,
  confident-wrong **25.8%**.
- **The master variable: Rs correct -> median BER 0.0010; Rs wrong -> median BER 0.4881.**
  Every demodulator works when handed a correct symbol rate.
- By samples-per-symbol: **8 -> 59.2%**, 16 -> 36.7%, 4 -> 14.2%, **32 -> 6.7%**.
- Safety is the strongest area: digital->analog **1/480 (0.21%)**, analog->digital
  **0/90**. The Entry 034 gate converted 87 confident-wrong analog labels into rejections.
- Analog reliable at >=10 dB; **SSB recovery 0.995 with SigMF metadata vs 0.013 without**,
  with identical classification either way.
- Report structure validates; `.iq` and `.wav` give identical conclusions.
- **DECISION was TARGETED BLOCKER; Entry 040 CLOSED it.** Production default is now
  `models_saved/modulation_cnn_v3_spsaug.pt` (8-class digital, SPS-augmented 4/8/16/32).
  End-to-end fused **35.4% -> 99.5%**, sps 32 **10.4% -> 97.9%**, rejection 41.1% -> 0.0%,
  confident-wrong **23.4% -> 0.5%**. Analog unaffected (6/6 each). **These Entry 039
  numbers are now stale** - re-run the benchmark against the new default.

## Current blockers

1. **CNN analog quality is now the dominant blocker** (Entry 027). The classical
   detector was fixed and reports `analog-like` 5/5 for AM-DSB@20k, AM-SSB and WBFM at
   >=10 dB, but AM-SSB is confidently mislabelled `WBFM` (0.32-0.49) and WBFM's own label
   rarely ranks first. End-to-end correct analog labels: **4 of 12** (Entry 028 trace). Wrong analog labels
   now reach the wrong analog demodulator cleanly - a classification problem, not a
   routing one.
   *Note:* the default checkpoint `modulation_cnn.pt` **does** have `AM-DSB`/`AM-SSB`/
   `WBFM` (11 classes); only the V2 checkpoints lack them.
2. ~~dispatch decimates analog~~ **FIXED in Entry 028.** Analog now bypasses the digital
   symbol-rate path entirely: no samples-per-symbol, no timing search, no decimation.
   WBFM recovery through dispatch went 0.1454 -> 1.0000; AM-DSB tones no longer alias.
   `symbol_rate_hz=None` works for analog and still fails for digital.
3. `preprocess` strips the baseband AM-DSB carrier (DC = the carrier). Entry 027 showed
   this is also fatal to *detection*: `frequency_cv` 0.0000 -> 7.0457, more impulsive
   than any digital control, so AM-DSB@0 is unrecoverable at the detector layer.
4. **SSB carrier estimation — accepted NEGATIVE RESULT (Entry 029).** Usable SSB
   recovery needs the carrier to **~1-2 Hz**; the best signal-only estimate is ~500 Hz
   noiseless and −16 kHz at 20 dB. The carrier is suppressed and sits below the lowest
   message tone, a per-capture gap (536–1053 Hz) that is not observable. **Do not
   re-attempt blind estimation.** **RESOLVED for real captures in Entry 030**: a SigMF
   `.sigmf-meta` sidecar now populates `metadata["center_frequency_hz"]` through
   `ingestion/sidecar.py`, giving **0.0 Hz carrier error and 0.995–1.000 SSB recovery**
   across USB/LSB, two seeds, noiseless and 20 dB. Captures *without* a sidecar are
   unchanged and still unrecoverable. The current `welch_psd_centroid`
   is provably biased for one-sided spectra but was left alone: the alternative is worse
   under noise and `estimate_parameters` is shared with digital.
5. Above-threshold **wrong** analog labels are unhandled: AM-SSB is called `WBFM` at
   0.416-0.476 and routed to `demodulate_fm`.
6. Analog is not wired into any dataset builder, manifest or training set.
7. QAM64 collapses at low SNR (0.334 accuracy at 0 dB).
8. Fusion's 0.4 threshold is uncalibrated; analog CNN confidence sits astride it, so
   analog routing is seed-dependent.
9. Low-SNR FSK symbol-rate estimation is an accepted **negative result** (Entry 014) —
   the spectral line sits below the noise floor. Do not re-attempt without new evidence.

## Recommended next direction

Analog **generation** is done, and Entry 026 opened a fusion path for analog labels —
but it fires in only 1 of 15 analog captures because the gate needs `analog-like` and the
classical detector rarely says it. Entry 027 fixed the detector (0/800 digital
false positives) and AM-DSB@20k now routes correctly 4/5. Entries 027-028 fixed the detector and dispatch;
**AM-DSB at a non-zero carrier offset now recovers its message end to end at 0.94
correlation.** **Next: CNN analog quality** — AM-SSB is confidently mislabelled `WBFM`,
which no downstream layer can undo. Entry 032 implemented the analog-type gate: analog subtype routing is now **55/60**
(55/55 wherever the classical gate engages), and AM-SSB LSB went from 0/8 to 10/15.
**Entry 033 says do NOT retrain** - CNN top-3 is 100% at >=10 dB, so the model already
carries the answer and retraining would not fix a missing dispatch route, a fusion
override rule, or a detector threshold. Item (1) is **done (Entry 034)**. Items (1) and (2) are **done**
(Entries 034, 035). Entry 040 is **done and shipped**. **Next: re-run the
Entry 039 benchmark against the new default checkpoint** and, if it confirms these
numbers, declare **SYNTHETIC FREEZE** and move to real-world data. **Do NOT reopen
symbol-rate estimation** (Entries 037/038, closed both ways). Still open: analog detection
at 10 dB for AM-SSB LSB. Then re-benchmark and reconsider the freeze. Then the `preprocess` DC issue.
Formerly next: SSB carrier
estimation is closed: Entry 029 as a negative result for blind estimation, Entry 030 by
supplying real metadata through ingestion. Do not add more analog waveforms.

## Standing constraints

- V1 is frozen; do not regenerate or modify it.
- Keep the analog registry separate from `config.MODULATIONS` — three call sites default
  to `tuple(MODULATIONS)` and must not silently widen.
- Production ingestion must never read the generator's ground-truth JSON; only a
  `.sigmf-meta` sidecar may supply `center_frequency_hz` (three tests enforce this).
- Digital carrier-estimate baseline (true 0 Hz, 20 dB, seed 101): BPSK −19.7,
  QPSK −167.5, 8PSK +32.7, 16QAM +123.3, 64QAM −44.4, BFSK +106.4, GFSK +113.5 Hz.
  Measure any estimator change against these.
- The classical detector's `analog-like` is now a **positive** verdict
  (`frequency_cv < 0.9` and no fourth-power line), not a fallback; its final `else`
  returns `"unknown"`. Do not restore analog as the catch-all.
- **The production checkpoint is `modulation_cnn_v3_spsaug.pt` and is DIGITAL-ONLY.**
  Analog labels come from the Entry 032 classical gate, which never consulted the CNN.
  Do not "restore" analog CNN classes without re-measuring - the digital-only model
  scores 99.5% fused where the 11-class RadioML model scored 35.4%.
- **`models_saved/` is gitignored, so the default checkpoint is untracked.** A fresh clone
  returns `Unclassified`. Also: `train_v2` does not write the `_metrics.json` sibling the
  loader requires - call `write_metrics()` separately or the checkpoint will not load.
- **99.5% is synthetic-to-synthetic** and is NOT a real-world accuracy claim.
- Fusion has TWO symmetric guards that must both keep working: Entry 032 (classical
  `analog-like` beats the CNN) and Entry 034 (a positive classical DIGITAL family blocks
  a CNN analog label). Never let one be added back without the other.
- **sps=32 is broken at BOTH layers (Entry 039).** Rs within 10%: PAM4 0/15, QAM64 2/15,
  QAM16 3/15, 8PSK 7/15. AND CNN top-1 0/15 for five of six linear classes even where Rs
  is recovered. The classifier failure is the larger of the two.
- Analog subtype routing (Entry 032) is gated on `classical_family == "analog-like"`.
  Never let it run on a digital family; the gate is what keeps digital false positives at
  zero. Its `trust_score` is the classical FAMILY confidence, not a subtype probability.
- ~~GFSK -> WBFM blocks freeze~~ **FIXED in Entry 034.** A positive classical digital
  verdict (`DIGITAL_FAMILIES`, confidence >= 0.5) now blocks a CNN analog label; the
  fallback keeps the CNN's own highest-ranked digital alternative if it clears 0.4, else
  `Unclassified`. **27/50 -> 0/50**, false-analog routing **128/600 -> 1/600**. Residual
  1/600 is a `unknown`-family capture, deliberately not a blocking verdict.
  Cost: digital rejection rose 21.8% -> 41.8%.
- ~~PAM4 never demodulates~~ **FIXED in Entry 035.** New `pam_demod.py` + dispatch route:
  BER **0.00000 at 20 dB**, 10/10 demodulated at every SNR, exactly scale-invariant.
  PAM4 *classification* is still weak (6/25 top-1) - this fixed demodulation only.
- **CORRECTED by Entry 036: the ~0.47 FSK BER is SYMBOL-RATE ESTIMATION, not GFSK
  demodulation.** With the true symbol rate GFSK gives 0.0044-0.0314 (noiseless) and
  0.0049-0.0275 (20 dB) across sps 4-32 - and noiseless at sps=8 GFSK beats CPFSK
  (12 vs 56 errors). With the *estimated* rate both GFSK and CPFSK collapse to ~0.50
  once sps > 8 or SNR drops; the estimator returned 1880 Hz for a true 25000 Hz. The only
  working configuration is BFSK sps=8 at high SNR - exactly V1, which is why it stayed
  hidden. **Entry 037 investigated it and returned a NEGATIVE RESULT.** The symbol-boundary
  spectral line is at the noise floor: **42.4x / 16.7x** line-to-floor for CPFSK at
  sps 4/8 (where the estimator works) vs **2.9-4.9x** for CPFSK at sps 16/32 and
  **3.2-4.2x for GFSK at every sps**. Gaussian shaping exists to suppress transition
  energy, so it suppresses the feature the estimator needs. The harmonic hypothesis was
  disproved (ratios 13.3/14.6/15.1, not integers). A candidate fix (gate out
  `envelope_power` for constant-modulus signals) gave only 80.6% -> 81.9% and left GFSK
  at 0/5 - rejected. **Linear modulations are already essentially perfect** (BPSK/QPSK/
  8PSK 20/20 across sps 4-32), so this is FSK-only, not a general estimator defect.
  **Entry 038 built and gated that cyclostationary estimator. Split verdict:**
  **GO for CPFSK/BFSK** - a real signature (12.9-23.9x peak/background), and
  **BFSK sps=16 @ 20 dB downstream BER 0.5135 -> 0.0176**. **NO-GO for GFSK** - the
  alpha=Rs feature is absent by construction (Gaussian shaping removes the excess
  bandwidth it needs), 1.3-4.6x, wrong at every sps and SNR. **GFSK symbol-rate
  estimation above sps 4 is now an ACCEPTED CAPABILITY LIMITATION** - do not try another
  estimator family. Gate set at 6.0 for zero regressions; the raw estimator fixes more
  (incl. CPFSK sps=32) than the gate ships, deliberately. There is also **no usable rejection signal**: spurious
  estimates carry confidence 0.44-0.55, same band as correct ones.
- Minor, quantified, NOT fixed: `_fsk_timing_offset` picks a straddling offset in 3/40
  GFSK cases (0/40 CPFSK) because the Gaussian pulse flattens the variance minimum. One
  candidate replacement was tested and regressed badly (5-8/10 failures), so the existing
  criterion was kept.
- **180-degree polarity ambiguity is architectural**, shared by PAM4 and BPSK: a blind
  receiver cannot resolve the constellation sign without a differential/pilot/preamble
  reference. Documented in Entry 035, not solved.
- **AM-SSB LSB is a detector problem**: reliable to 15 dB, cliff to 0/10 at 10 dB where
  classical returns `QAM-like`. USB has one SNR step more margin.
- Analog dispatch must never use a symbol rate; `dispatch.ANALOG_LABELS` and
  `fusion.ANALOG_LABELS` must stay equal (asserted by test).
- Analog captures carry no bits, no symbol rate, no Es/N0 and no Eb/N0. Never publish a
  BER for one. WBFM's deviation lives in the `analog` block, never in the digital-FSK
  `signal.fsk_*` fields.
- `test_decoding_correlation.py::test_reed_solomon_round_trip` fails and
  `test_model_report.py` fails to collect — both are **environmental** (`reedsolo`,
  `h5py` not installed), not regressions.
