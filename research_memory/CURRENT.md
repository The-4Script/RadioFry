# Current State

_Last updated: 2026-09-09, after BANK.md Entry 031 (analog routing dominance)._

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
which no downstream layer can undo. **Next: implement the Entry 031 analog-type gate** (the smallest change that makes
analog demodulators receive the right label without retraining). Then the `preprocess` DC
issue, then retraining the CNN with analog classes - the real fix.
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
- Analog dispatch must never use a symbol rate; `dispatch.ANALOG_LABELS` and
  `fusion.ANALOG_LABELS` must stay equal (asserted by test).
- Analog captures carry no bits, no symbol rate, no Es/N0 and no Eb/N0. Never publish a
  BER for one. WBFM's deviation lives in the `analog` block, never in the digital-FSK
  `signal.fsk_*` fields.
- `test_decoding_correlation.py::test_reed_solomon_round_trip` fails and
  `test_model_report.py` fails to collect — both are **environmental** (`reedsolo`,
  `h5py` not installed), not regressions.
