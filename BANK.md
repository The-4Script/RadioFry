# BANK.md — RadioFry Engineering Change Log

Permanent, append-only record of Claude Code engineering work on RadioFry
(SIH 2026 problem statement **SIH26147 / NTRO** — automated analysis of `.IQ` and
`.wav` files with signal-parameter extraction).

**Rules for this file**
- Append new entries at the bottom. Never delete or rewrite earlier entries.
- Record what actually happened, with file paths and measured numbers.
- Distinguish measured results from hypotheses. Do not record a claim as
  validated without evidence.

---

## Entry 001 — 2026-09-07 — Synthetic Dataset V1, evaluation harness, and baseline findings

### Context

A prior forensic audit of the existing RadioFry prototype established that almost
every pipeline output was unvalidated: no ground-truth benchmark existed for SNR,
bandwidth, symbol rate, demodulation BER, or modulation classification. Entry 001
covers the two pieces of work that closed that gap — building a controlled dataset
with known ground truth, and building a harness that scores the **unmodified**
pipeline against it.

No RadioFry algorithm, model, demodulator, fusion rule, FEC/interleaver path, or
preprocessing step was modified in this entry.

### 1. Synthetic Dataset V1 generator — completed

New self-contained subpackage `src/radiofry/synthetic_gen/v1/`:

| File | Purpose |
|---|---|
| `config.py` | `SampleSpec`, `MODULATIONS`, `V1_IMPAIRMENTS`, rate resolution/validation |
| `modulation.py` | bits → natural-binary symbol indices → baseband waveform |
| `channel.py` | `add_awgn` + `NoiseReport` |
| `writers.py` | headerless interleaved `.iq` and stereo `.wav` writers |
| `generator.py` | capture/dataset orchestration, ground-truth records, manifest |
| `cli.py`, `__main__.py`, `__init__.py` | CLI entry point and public API |

Chain implemented: `known bits → modulation → controlled baseband → AWGN → IQ/WAV → ground truth`.

- **Modulations:** BPSK, QPSK, 8PSK, BFSK (continuous-phase), 16QAM, 64QAM.
- **Configurable:** sample rate, symbol rate, samples/symbol (any two supplied, the
  third derived and cross-validated), symbol count, SNR sweep, seed, formats, dtypes,
  byte order.
- **Default SNR sweep:** 20 / 15 / 10 / 5 / 0 dB.
- **V1 pinned controlled:** CFO 0, timing offset 0, phase offset 0, fading /
  multipath / interference disabled, FEC none, interleaving none — all recorded
  explicitly in metadata rather than left implicit.
- **Reproducibility:** two independent RNG streams
  (`default_rng([bits_seed, 1])` for payload, `default_rng([seed, 2])` for noise);
  child seeds derived via `blake2b(base_seed | modulation | snr | replicate)` so they
  do not depend on iteration order. The bits seed deliberately **excludes** SNR, so a
  sweep shares one payload and differs only in noise.
- **Reuse:** the ingestion `IQFormat` contract is imported directly rather than duplicated.

**Ground truth per capture:** `<id>.json` plus `<id>.source_bits.npy` and
`<id>.transmitted_bits.npy`, covering modulation (name, family, order, bits/symbol,
`radiofry_label`, bit mapping, normalization, pulse shape), signal (both rates, sps,
counts, duration, FSK deviation, centre frequency), noise (type, target and realized
SNR, explicit `snr_definition`, derived Es/N0 and Eb/N0, signal/noise powers),
impairments, bits (counts, filenames, SHA-256 digests), seeds, and per-file writer
records. Dataset level: `manifest.csv` (19 columns) + `dataset.json`.

**Generated artifact:** `data/synthetic_v1/` — 30 captures (6 modulations × 5 SNRs),
4096 symbols each, 8 samples/symbol, 200 kHz sample rate, 8.8 MB.

**Verification (measured, not asserted):**
- 77 new tests, all passing; full suite 40 → 117 passing.
- AWGN realizes the requested SNR within **0.1 dB** at every sweep point.
- Noiseless waveforms de-map bit-exactly for all six modulations against an
  independent oracle.
- Files round-trip through RadioFry's real `read_iq` / `read_wav` / `load_capture`
  for int16, float32, and big-endian.
- Two full generation runs into different directories produced a **byte-identical**
  aggregate SHA-256 (`de63dd64ae5cdf5bc2f08589d9ccf86751429c1838e93fc8f39e510a069248d9`).
- Independent decode of the written files shows BER falling monotonically with SNR
  for every modulation, reaching exactly 0 at high SNR for BPSK/QPSK/8PSK/16QAM/BFSK.

**Recorded conventions (decisions, not defects):** natural-binary MSB-first bit
mapping (matches the existing de-mappers); rectangular pulses, no RRC; SNR defined
over the full sampled band (Es/N0 is 9.03 dB higher at sps=8); QAM normalized to unit
average symbol power; BFSK tagged `radiofry_label: "CPFSK"` because dispatch has no
`BFSK` route.

### 2. V1 evaluation harness — completed

New subpackage `src/radiofry/evaluation/`:

| File | Purpose |
|---|---|
| `metrics.py` | BER (strict + bounded-shift aligned), family mapping, error helpers, confusion matrices |
| `harness.py` | per-capture pipeline run and stage scoring, `UNAVAILABLE_METRICS`, `RECORD_FIELDS` |
| `report.py` | aggregation by modulation / SNR / format, CSV + JSON + markdown output |
| `cli.py`, `__main__.py`, `__init__.py` | CLI entry point |

The harness calls only the public entry points `load_capture` and `analyze_capture`
and reads what the report already contains — no pipeline stage is re-implemented, and
no known mismatch (QAM normalization, BFSK routing, SNR definition) is compensated.
Seven metrics are recorded as explicitly **unavailable with reasons** rather than
invented, most notably occupied bandwidth (V1 ground truth defines no 99%-power
bandwidth for rectangular pulses).

31 new tests added; full suite 117 → **148 passing**.

Run with:
```
PYTHONPATH=src python -m radiofry.evaluation --dataset data/synthetic_v1 --output reports/v1_baseline
```

### 3. Baseline findings — unmodified RadioFry on V1

Main run: `reports/v1_baseline/` (30 captures × IQ and WAV = 60 records).

| Stage | Baseline result |
|---|---|
| Ingestion | **100%** — IQ and WAV both clean, zero errors |
| Pipeline completes without exception | **100%** |
| Classical family accuracy | 40% (100% on QAM, **0% on all PSK**) |
| CNN top-1 accuracy | **0%** — predicted `AM-SSB` for 26/30 captures |
| CNN top-3 accuracy | 20% |
| Fusion | rejected **96.7%**; review recommended 100% |
| Demodulation reached | **3.3%** (2 of 60) |
| BER | 0.508 on the only scored case — chance level |
| Symbol rate | truth 25 kHz; estimates 977 Hz – 29 kHz; **0% within 1%** |
| SNR estimate | tracks truth at 0–10 dB, saturates ~11–12 dB above that (−8.2 dB median error at 20 dB) |
| Carrier estimate | **good** — median error ~1–100 Hz against 0 Hz truth |
| Interleaver reported as `none` | 0% (of 2 scored) |
| Sync-word false positives | 100% (of 2 scored) |
| IQ vs WAV | no measurable difference — 0/30 captures disagreed on CNN label or end-to-end status |

**What works:** ingestion (both formats, identical results), pipeline robustness (no
crashes), and carrier-frequency estimation.
**What fails:** modulation classification, fusion gating, symbol-rate estimation,
demodulation, and every downstream bit-level stage.

### 4. CNN long-capture / 128-sample input issue — identified

`src/radiofry/models/modulation_inference.py:21-30` (`_fixed_iq`) reduces **any**
capture to `sample_length` (128) samples with
`np.interp(np.linspace(0, size-1, 128), ...)`. For a 32768-sample V1 capture that is
**256× decimation with no anti-aliasing filter** — the sample positions are ~258
samples apart, so the result is point-sampling of an aliased signal, not a
representative frame.

**Controlled experiment isolating the confound.** A second dataset was generated with
16 symbols × 8 sps = exactly 128 samples, so the adapter performs no decimation
(`data/synthetic_v1_shortframe/`, evaluated to `reports/v1_baseline_shortframe/`):

| Metric | 4096-symbol captures (32768 samples) | 128-sample control |
|---|---|---|
| CNN top-1 accuracy | 0.0% | **33.3%** |
| CNN top-3 accuracy | 20.0% | **80.0%** |
| Demodulation reached | 3.3% | **80.0%** |
| BER scored | 2 / 60 | 48 / 60 |
| Mean CNN softmax | 0.331 | 0.584 |

**Conclusion (measured):** the CNN's 0% top-1 accuracy on normal-length captures is
substantially caused by the input adapter destroying the signal, not solely by model
quality. This is a confirmed defect in `_fixed_iq`, and is the highest-value target
identified so far.

**Still failing even with the confound removed** (short-frame control): BER stays
0.2–1.0 with **no SNR trend** — BPSK sits at exactly 0.200 at every SNR, a structural
error rather than noise. Consistent with the audit findings of no carrier/phase
recovery and symbol-rate-driven decimation errors. The interleaver classifier reported
`none` on **0 of 48** uncoded streams; sync words were falsely detected on 50%.

### Files added in this entry

```
src/radiofry/synthetic_gen/v1/{__init__,config,modulation,channel,writers,generator,cli,__main__}.py
src/radiofry/evaluation/{__init__,metrics,harness,report,cli,__main__}.py
tests/test_synthetic_v1_{modulation,io,dataset}.py
tests/test_v1_harness.py, tests/test_v1_harness_metrics.py
data/synthetic_v1/            (30 captures, 8.8 MB)
data/synthetic_v1_shortframe/ (30 control captures)
reports/v1_baseline/{results.csv,results.json,summary.md}
reports/v1_baseline_shortframe/{results.csv,results.json,summary.md}
```

No existing RadioFry source file was modified in this entry.

### Open items carried forward

- `reports/*.md|csv|json` and `data/synthetic_v1*` are not covered by `.gitignore`
  (which only ignores `reports/*.png`) — commit-vs-ignore decision pending.
- Pre-existing unrelated test failure: `tests/test_decoding_correlation.py::test_reed_solomon_round_trip`
  fails because `reedsolo` is not installed in this environment. `commpy`, `pyldpc`
  and `h5py` are also absent.
- V1 ground truth does not define an occupied bandwidth, so bandwidth accuracy cannot
  be scored.

---

## Entry 002 — 2026-09-07 — Fix CNN input adapter: window long captures instead of decimating them

### Context

Entry 001 identified and experimentally confirmed that `_fixed_iq` in
`src/radiofry/models/modulation_inference.py` reduced **any** capture to 128 samples
by interpolating across the whole capture — 256× decimation with no anti-alias filter
for a 32768-sample V1 capture. This entry fixes that single defect.

Scope was deliberately limited: the CNN architecture, the trained model, the dataset,
and every other component were left untouched. No retraining was performed.

### Change

`src/radiofry/models/modulation_inference.py` — `_fixed_iq` only:

- **Capture longer than the frame length:** now takes a **contiguous window at the
  native sample rate**, centred in the capture (`start = (size - length) // 2`). This
  matches what the CNN was trained on (RML2016.10a frames are 128 contiguous samples)
  and preserves the true per-sample phase progression.
- **Capture shorter than the frame length:** unchanged — still stretched by linear
  interpolation.
- **Capture exactly the frame length:** unchanged (no-op).
- RMS normalization, output shape `(2, length)`, `float32` dtype, and the empty-signal
  `ValueError` are all unchanged.

This is a windowing-policy change only. Nothing else in the file, and no other file,
was modified.

### Tests

New file `tests/test_modulation_inference_input.py` (9 tests), written before the fix:

- `test_long_capture_preserves_the_signal_frequency_instead_of_aliasing_it` — a tone at
  fs/16 must retain 2π/16 rad per-sample phase advance. **Failed before the fix**
  (produced ~0.785 rad instead of 0.393), passes after.
- `test_long_capture_frame_is_a_contiguous_window_of_the_capture` — **failed before**,
  passes after.
- Seven contract-preservation tests (shape, dtype, RMS normalization, exact-length
  capture, short capture, single sample, empty capture, all-zero capture) — these
  **passed both before and after**, guarding against regression.

Full suite: **157 passed**, up from 148. The single failure
(`test_reed_solomon_round_trip`) is the pre-existing missing-`reedsolo` environment
gap, unrelated to this change.

### Measured before/after on Synthetic V1

Same dataset (`data/synthetic_v1`, 30 captures × IQ and WAV = 60 records), same
harness, unmodified pipeline otherwise.
Before: `reports/v1_baseline/`. After: `reports/v1_after_cnn_input_fix/`.

| Metric | Before | After |
|---|---|---|
| CNN top-1 accuracy | 0.0% | **40.0%** |
| CNN top-3 accuracy | 20.0% | **73.3%** |
| Mean CNN softmax (uncalibrated) | 0.331 | 0.722 |
| Fusion accuracy | 0.0% | **40.0%** |
| Fusion rejection rate | 96.7% | **13.3%** |
| Mean fusion trust score | 0.179 | 0.405 |
| Demodulation reached | 3.3% | **86.7%** |
| End-to-end (bits recovered) | 3.3% | **86.7%** |
| BER scored | 2 / 60 | **52 / 60** |
| Median BER (strict) | 0.508 | 0.500 |

Per-modulation CNN top-1 after the fix: BPSK 100%, QPSK 80%, 8PSK 60%,
16QAM 0%, 64QAM 0%, BFSK 0%.

**Isolation evidence.** Every metric that does not depend on the CNN adapter is
**bit-identical** before and after, confirming the change did not leak into other
stages:

| Metric | Before | After |
|---|---|---|
| Ingestion success rate | 1.0 | 1.0 |
| Classical family accuracy | 0.4 | 0.4 |
| Median SNR error (dB) | 0.2203 | 0.2203 |
| Median carrier error (Hz) | 3.872 | 3.872 |
| Median symbol-rate relative error | −0.7302 | −0.7302 |

The 128-sample control dataset (`data/synthetic_v1_shortframe/`) was re-evaluated to
`reports/v1_shortframe_after_cnn_input_fix/` and is **identical** before and after
(CNN top-1 33.3%, top-3 80%, end-to-end 80%, median BER 0.5) — captures at exactly the
frame length take the unchanged code path, as intended.

### Remaining limitations (not addressed here, by instruction)

- **Demodulation still does not work.** Median strict BER is ~0.50 (chance) at every
  SNR for every modulation, with no SNR trend. The fix opened the gate to
  demodulation; it did not make demodulation correct. Root causes remain those
  identified in the audit: symbol-rate estimation is badly wrong (median relative
  error −0.73, 0% within 1%), and there is no carrier or phase recovery anywhere in
  the demodulator path.
- **16QAM, 64QAM and BFSK remain at 0% CNN top-1.** 64QAM top-3 is only 40%. BFSK is
  never in the top 3 — it is scored against the `CPFSK` label, and the QAM
  normalization mismatch noted in Entry 001 is still present and uncorrected.
- **A single centred window is used.** Only 128 of 32768 samples now reach the CNN, so
  ~99.6% of each capture is unused. Averaging or voting predictions over multiple
  windows would likely improve accuracy further, but that changes `predict_modulation`
  semantics and was deliberately out of scope for a minimal fix.
- **Softmax is still uncalibrated.** Mean confidence rose from 0.331 to 0.722, which
  moves many captures past the hand-set 0.4 fusion threshold. This changes the
  operating point of an unvalidated gate; the fusion constants remain empirically
  unjustified.
- **Downstream stages degraded in appearance only because more captures now reach
  them:** interleaver reported as `none` on 7.7% of 52, FEC `none` on 3.8% of 52, and
  sync-word false positives at 100% of 52. These now have a meaningful denominator for
  the first time and confirm the audit's concerns about those stages.

---

## Entry 003 — 2026-09-08 — Investigation: why symbol-rate estimation is 0% accurate on V1

**Investigation only. No RadioFry behaviour was modified.** `dsp/parameter_estimation.py`
is byte-for-byte unchanged; the only file added is a diagnostic test module.

### The algorithm as actually implemented

`src/radiofry/dsp/parameter_estimation.py:91-102`, then
`_select_symbol_rate` at lines 32-61:

1. `centered = signal.iq - np.mean(signal.iq)` — DC removal.
2. `powered = np.abs(centered) ** 2` — **envelope power**. Despite the reported
   `method="welch_nth_power"`, the exponent is hard-coded to 2.
3. `spectrum = np.abs(np.fft.rfft(powered - np.mean(powered)))`.
4. `rates = np.fft.rfftfreq(powered.size, d=1/fs)`; `spectrum[0] = 0`.
5. `find_peaks(spectrum, prominence=max(2*median(spectrum), 1e-12))`.
6. `_select_symbol_rate`: `score = peak_value + 0.25 * harmonic_support`, where
   harmonic support sums interpolated amplitudes at 2f/3f/4f that reach ≥5% of the
   peak; `argmax(scores)` wins.

**What it therefore detects:** the strongest spectral line in the *amplitude/envelope
fluctuation* of the waveform. This is the classical squaring timing-tone method.

### Root cause — two independent and individually fatal problems

**(a) For constant-modulus modulations the feature is identically zero.**
Measured variance of `|x|²` on noiseless V1 captures:

| BPSK | QPSK | 8PSK | BFSK | 16QAM | 64QAM |
|---|---|---|---|---|---|
| 0.000e+00 | 0.000e+00 | 7.11e-15 | 3.55e-15 | 3.17e-01 | 3.77e-01 |

With rectangular pulses and unit-magnitude constellations the envelope is perfectly
flat; all symbol information lives in the phase, which `|·|²` discards. For 4 of the 6
V1 modulations the estimator is peak-picking noise.

**(b) For every modulation the true symbol rate sits on an exact spectral null.**
Measured `spectrum` at 25 000 Hz = **0.0000e+00** for 16QAM, 64QAM and QPSK, against a
local median of ~5.8. Reason: full-duty rectangular pulses do not overlap, so
`|x(t)|² = Σ |a_n|² p(t − nT)` is exactly a piecewise-constant NRZ waveform built from
the same rectangular pulse. Its spectrum is `|P(f)| = T·|sinc(fT)|`, which has **exact
zeros at f = k/T = k·Rs** — for both the continuous part and any discrete line. The
estimator searches for a peak precisely where the feature is mathematically guaranteed
to have none, at Rs and at every harmonic.

The surviving energy is the low-frequency main lobe (0 … Rs), so the selected peak is
strongly biased **low**: measured median relative error −0.51 to −0.94.

### Modulation-dependent or SNR-dependent? Neither — it is structural

From `reports/v1_after_cnn_input_fix/results.csv` (IQ records):

| grouped by modulation | median rel err | within 1% | | grouped by SNR | median rel err | within 1% |
|---|---|---|---|---|---|---|
| BPSK | −0.508 | 0/10 | | 20 dB | −0.851 | 0/12 |
| QPSK | −0.779 | 0/10 | | 15 dB | −0.828 | 0/12 |
| 8PSK | −0.531 | 0/10 | | 10 dB | −0.358 | 0/12 |
| BFSK | −0.117 | 0/10 | | 5 dB | −0.628 | 0/12 |
| 16QAM | −0.877 | 0/10 | | 0 dB | −0.672 | 0/12 |
| 64QAM | −0.941 | 0/10 | | | | |

It fails for every modulation and at every SNR. Decisively, it **also fails with no
noise at all** — on noiseless captures the estimates were BPSK 2069 Hz, QPSK 1440 Hz,
8PSK 1953 Hz, BFSK 1294 Hz, 16QAM 299 Hz, 64QAM 12.2 Hz (relative errors −0.92 to
−1.00). Noise is not the cause.

### Rectangular pulse shaping and harmonics: decisive contributors

Rectangular pulses are the direct cause of the null in (b). With RRC shaping
(rolloff > 0) adjacent pulses overlap and a genuine timing tone does appear at Rs —
which is why this textbook method normally works. The harmonic-support scoring cannot
rescue it, because Rs **and all of its harmonics** are nulls; harmonic support only
reinforces low-frequency noise lobes.

**BFSK is a separate confound.** Its envelope tone sits at **2 × deviation**, not at the
symbol rate. It coincides with Rs in V1 only because the generator defaults
Δf = Rs/2. Verified by sweeping deviation: at Δf = 3125 Hz the line appears at 6250 Hz
(88× the noise floor) while 25 kHz stays at ~3× floor. BFSK's smaller apparent error is
therefore partly coincidence, and is a **V1 dataset confound to fix in V2** (vary the
deviation independently of the symbol rate).

### Downstream interpretation: correct units, but no plausibility guard

`src/radiofry/decoding/demodulators/dispatch.py:35`:
`samples_per_symbol = max(1, round(sample_rate / symbol_rate_hz))` — dimensionally
correct, no unit bug. Two observations:

- True sps = 8 requires the estimate to land in **[23 529, 26 667] Hz (±6.7%)**, which
  is *looser* than the harness's 1% criterion. Even so only **1 of 30** IQ captures
  lands there (BPSK @ 0 dB, 24 164 Hz → sps 8). Measured sps ranged **8 to 205**
  (64QAM @ 20 dB: 1477 Hz → sps 135).
- No plausibility check exists: an estimate of 12.2 Hz yields sps = 16 384, i.e. two
  symbols from a 32 768-sample capture, and the demodulator proceeds and reports
  success regardless.

### Candidate-feature experiment

Only the pre-FFT nonlinearity was varied; the FFT, `find_peaks` and RadioFry's own
`_select_symbol_rate` were imported and used unchanged. 36 cases (6 modulations ×
noiseless + 20/15/10/5/0 dB), truth 25 000 Hz:

| pre-FFT feature | within 1% of truth |
|---|---|
| `np.abs(x)**2` (current) | **0 / 36** |
| `np.abs(np.diff(x))**2` | 30 / 36 (fails all BFSK) |
| `np.diff(instantaneous_frequency)**2` | **35 / 36** (fails only BFSK @ 0 dB) |

The winning feature is the squared second difference of the unwrapped phase. It
returned exactly 25 000.0 Hz — not approximately — for every modulation at every SNR
except BFSK at 0 dB. Symbol boundaries are phase discontinuities, so this feature is an
impulse train at Rs for both the linear modulations and CPFSK.

### Recommended smallest fix (NOT applied)

Replace the single nonlinearity line `parameter_estimation.py:92` with the squared
second difference of unwrapped phase. Keep the FFT, `rfftfreq`, `spectrum[0] = 0`,
`find_peaks` and `_select_symbol_rate` exactly as they are.

**Caveat that must be resolved before committing:** this is validated **only** on V1's
rectangular-pulse signals. The existing `|x|²` method is the textbook approach for
RRC-shaped signals, and we currently hold **no** band-limited/RRC test data. Swapping
it outright risks regressing on realistic captures. Recommended sequencing: add
RRC-shaped captures to V2 first, then either commit the swap or evaluate both features
and select by spectral-line quality.

### Diagnostic artefacts

- `tests/test_symbol_rate_diagnostics.py` — **added**, 19 tests, all passing. Asserts
  properties of the *signals and features*, never the estimator's current output, so it
  stays valid after any future fix. Covers: zero envelope variance for constant-modulus
  V1 signals; non-zero variance for QAM; the exact spectral null at Rs; the BFSK
  2×deviation confound; and that the phase-step feature peaks at the true symbol rate.
- Full suite after this entry: **176 passed** (up from 157), same single pre-existing
  `reedsolo` environment failure.
- The exploratory sweep scripts were run from the session scratchpad and are not
  retained; `tests/test_symbol_rate_diagnostics.py` is the durable reproduction.

### Not touched (and should stay untouched until the fix is agreed)

`dsp/parameter_estimation.py`, `dsp/preprocessing.py`, `dsp/cyclostationary.py`, all
demodulators and `dispatch.py`, `fusion/confidence_fusion.py`, the CNN and its
checkpoint, the FEC/interleaver paths, and the V1 generator and dataset.

---

## Entry 004 — 2026-09-08 — Resolving the RRC caveat: controlled feature comparison on rect vs RRC

**Investigation only. No RadioFry behaviour was modified.**
`dsp/parameter_estimation.py` remains byte-for-byte unchanged (103 lines, line 92 still
`powered = np.abs(centered) ** 2`). The V1 generator and dataset are untouched; RRC
shaping is applied only inside the new diagnostic module.

### Purpose

Entry 003 recommended replacing the `|x|²` nonlinearity but flagged an unresolved
caveat: the phase-step feature was validated only on V1's rectangular pulses, while
`|x|²` is the textbook method for band-limited (RRC) signals, and we held no RRC test
data. This entry builds that data and settles the question.

### Experiment design

New module `src/radiofry/evaluation/symbol_rate_experiment.py` (diagnostic only; it
imports RadioFry's `_select_symbol_rate` unchanged and re-uses the V1 generator's bit
and constellation helpers).

- **Pulse shapes:** `rect` (produced by the unmodified V1 generator) and RRC at
  **β = 0.20** and **β = 0.35** (unit-energy RRC, 10-symbol span, odd tap count so
  there is no net delay). For CPFSK the RRC is applied to the *frequency* pulse, which
  keeps the signal constant-modulus by construction.
- **Modulations:** all six supported (BPSK, QPSK, 8PSK, BFSK, 16QAM, 64QAM).
- **SNR:** noiseless, 20, 15, 10, 5, 0 dB (AWGN via the V1 channel).
- **Seeds:** 3 per cell. **Total 324 captures**, 2048 symbols each, 8 sps, 200 kHz,
  ground-truth symbol rate 25 000 Hz.
- **Scoring:** only the pre-FFT nonlinearity is varied; the FFT, `find_peaks` and
  `_select_symbol_rate` are RadioFry's own code. Hit = estimate within 1% of truth.
  (`demod_usable`, within ±6.7%, was also recorded and tracked the 1% figure closely.)

**Diagnostic-signal validity is itself tested** — `tests/test_symbol_rate_feature_comparison.py`
(25 tests, all passing) checks that the RRC taps are symmetric/unit-energy/finite at the
singular points, that the shaped signals are band-limited to ≈(1+β)·Rs, that they are
narrower than the rect equivalent, that they **recover their source symbols through a
matched filter with >0.99 correlation**, that RRC gives constant-modulus PSK a varying
envelope while shaped CPFSK stays constant-modulus, and that the rect arm is
byte-identical to the V1 generator's output.

### Results — hit rate within 1% of the true symbol rate

| Group (n) | `abs(x)²` (current) | `abs(diff x)²` | phase 2nd diff² |
|---|---|---|---|
| **Overall (324)** | 178 (55%) | 241 (74%) | **283 (87%)** |
| rect (108) | **1** | 90 | **102** |
| rrc β=0.20 (108) | 87 | 72 | 86 |
| rrc β=0.35 (108) | 90 | 79 | **95** |
| rect, linear only (90) | **1** | 90 | **90** |
| rrc β=0.20, linear only (90) | **87** | 60 | 82 |
| rrc β=0.35, linear only (90) | **90** | 65 | 89 |
| BFSK, all shapes (54) | **0** | 26 | 22 |

**The caveat was justified and is now resolved.** On RRC-shaped linear signals the
current `|x|²` feature is genuinely strong (87/90 and 90/90) — it is the correct tool
for band-limited signals, not a broken one. It is catastrophically blind on rectangular
pulses (**1/90** linear) and on CPFSK at every pulse shape (**0/54**), exactly as
Entry 003 predicted.

**The phase feature does not collapse on RRC.** It scores 82/90 (β=0.20) and 89/90
(β=0.35) on linear signals versus the current 87/90 and 90/90 — a bounded loss of ~7
cases out of 324 — while going from 1/90 to 90/90 on rect.

**SNR robustness differs sharply.** `abs(diff x)²` collapses on shaped signals at low
SNR (β=0.20: 18/18 at ≥10 dB but **0/18** at 5 dB and 0/18 at 0 dB). The phase feature
degrades gracefully (rect: 18/18 noiseless → 15/18 at 0 dB). The current feature is flat
across SNR because its failures are structural, not noise-driven.

**Modulation dependence:** every feature is weakest on BFSK. The current feature never
succeeds on it (0/54). The other two are complementary — `abs(diff x)²` scores 0/18 on
rect BFSK but 12–14/18 on shaped BFSK, while the phase feature is the reverse (12/18
rect, 4–6/18 shaped).

### Combination strategies measured

Selection rule = run the candidates and keep the one whose `_select_symbol_rate`
confidence is highest. This uses the existing confidence value, and its usefulness as a
selector is itself measured: when `abs(diff x)²` wins the 3-way selection it is correct
**85/85** times, the current feature 75/83, the phase feature 136/156.

| Strategy | Overall | rect·linear | rrc0.20·linear | rrc0.35·linear | BFSK (54) |
|---|---|---|---|---|---|
| Keep `abs(x)²` | 178/324 | 1/90 | 87/90 | 90/90 | 0 |
| Replace with phase 2nd diff² | 283/324 | 90/90 | 82/90 | 89/90 | 22 |
| Confidence-select (phase, absdiff) | 295/324 | 90/90 | 82/90 | 89/90 | 34 |
| **Confidence-select (phase, absdiff, current)** | **296/324** | **90/90** | **88/90** | **90/90** | **28** |
| Same, gated by envelope CV | 300/324 | 90/90 | 88/90 | 90/90 | 32 |
| Oracle (best of all three) | 306/324 | 90/90 | 88/90 | 90/90 | 38 |

The envelope-CV gate (drop `|x|²` when the envelope is effectively constant, since it
then carries no information by construction) adds 4 cases, but measured envelope CV is
noise-contaminated — constant-modulus captures show CV ≈ 0.237 once AWGN is present, so
the gate only fires at high SNR. Not worth the extra branch as a first change.

### Recommendation: **adaptive/combined — replace the single feature with a confidence-selected set of three**

Evaluate all three nonlinearities through the existing `find_peaks` +
`_select_symbol_rate` path and keep the candidate with the highest returned confidence.

Justification, entirely from the measured table:

- **No regression anywhere on linear modulations**: rect 1/90 → 90/90,
  β=0.20 87/90 → 88/90, β=0.35 90/90 → 90/90. Retaining `|x|²` as a candidate is what
  protects the band-limited case, which is the exact risk Entry 003 raised.
- **BFSK goes from 0/54 to 28/54**, the only strategy other than the 2-way combination
  that moves it at all.
- **Overall 178/324 → 296/324**, within 10 of the 306/324 oracle, so the selection rule
  is capturing most of what is achievable.
- A straight replacement (283/324) is simpler but regresses β=0.20 linear from 87/90 to
  82/90 — i.e. it trades away the case the caveat was about. The combination costs two
  extra FFTs per capture and avoids that trade.

Rejected alternatives: **keep current** (fails V1 entirely — 1/90 on rect linear, 0/54
on BFSK); **straight replace** (works, but gives up the band-limited case for no reason
now that the combination is measured to be cheap and strictly better).

### Remaining limitations of this experiment

- Only AWGN. No CFO, timing offset, fading, or multipath — those are V2 items and were
  deliberately not added.
- Only one symbol rate (25 kHz), one sps (8) and one capture length (2048 symbols).
  The features were not tested against a symbol-rate sweep.
- Only RRC and rectangular shaping; no Gaussian/GMSK, and the CPFSK arm uses an
  RRC-shaped frequency pulse, which is a reasonable band-limited CPM analogue but is not
  standard GFSK.
- BFSK remains poor under every strategy (best 34/54). The V1 Δf = Rs/2 confound noted
  in Entry 003 is still present in these signals.
- The confidence value used as the selector is itself one of the audit's flagged
  unvalidated heuristics. Its selector precision is measured here (85/85, 136/156,
  75/83) but it has not been calibrated.

### Artefacts added

```
src/radiofry/evaluation/symbol_rate_experiment.py     diagnostic module (RRC builder, features, sweep)
tests/test_symbol_rate_feature_comparison.py          25 tests validating the RRC signals
reports/symbol_rate_feature_comparison/               feature_comparison.csv + .json (324 records)
```

Full suite after this entry: **201 passed** (up from 176), same single pre-existing
`reedsolo` environment failure.

### Still not touched

`dsp/parameter_estimation.py` (the fix is recommended but **not applied**),
`dsp/preprocessing.py`, `dsp/cyclostationary.py`, all demodulators and `dispatch.py`,
`fusion/confidence_fusion.py`, the CNN and checkpoint, the FEC/interleaver paths, and
the V1 generator and dataset.

---

## Entry 005 — 2026-09-08 — Implement adaptive symbol-rate feature selection

Implements the strategy approved in Entry 004. Scope was limited to the symbol-rate
block of `dsp/parameter_estimation.py`; no demodulator, fusion rule, CNN, FEC/interleaver
path, V1 generator, or other parameter estimator was touched.

### Production change

`src/radiofry/dsp/parameter_estimation.py` (103 → 140 lines), three additive edits:

1. **`ParameterEstimate` gains `symbol_rate_feature: str | None = None`** — records which
   candidate won. Appended with a default, so every existing positional and keyword
   construction still works, and `report_builder._json_safe` picks it up automatically.
2. **`SYMBOL_RATE_FEATURES` registry** with the three measured nonlinearities:
   `envelope_power` (`abs(x)**2`, the original), `transition_power`
   (`abs(diff(x))**2`), and `phase_second_difference` (squared second difference of
   unwrapped phase).
3. **`_symbol_rate_candidate(powered, sample_rate)`** — the pre-existing FFT →
   `spectrum[0]=0` → `find_peaks` → `_select_symbol_rate` chain, extracted verbatim so
   it can be called per candidate. `estimate_parameters` now loops the registry and
   keeps the candidate with the highest returned confidence.

**`_select_symbol_rate` was not modified at all** — no change was required to support
the three candidates. The `method` strings were also left alone (they are asserted by
existing tests, and correcting their misleading "nth_power" naming is a separate
concern noted in the original audit).

### Tests

New `tests/test_symbol_rate_estimator_regression.py` — **38 tests**, written before the
change and failing on import beforehand:

- Rectangular linear captures (BPSK/QPSK/8PSK/16QAM/64QAM × noiseless/20/10 dB) recover
  the symbol rate within 1% — the case the original feature scored 1/90 on.
- Rectangular CPFSK recovers it without noise.
- RRC-shaped linear captures at **β = 0.20 and β = 0.35** still recover it, including
  under 20 dB noise — the no-regression guard for the Entry 003 caveat.
- The original envelope feature remains registered and numerically unchanged.
- The selected feature name is reported.
- Preserved contract: short capture, missing sample rate, and constant capture still
  return safe empty/valid estimates; bandwidth, SNR, carrier and `method` unchanged.

Full suite: **239 passed** (up from 201), same single pre-existing `reedsolo`
environment failure. No previously passing test changed behaviour.

### V1 evaluation — before/after

`reports/v1_after_cnn_input_fix/` → `reports/v1_after_symbol_rate_fix/`
(same dataset, same harness, 30 captures × IQ and WAV).

| Metric | Before | After |
|---|---|---|
| Symbol rate within 1% | **0%** | **96.7%** |
| Median symbol-rate relative error | −0.7302 | **−1.5e-16** (exact) |
| Median BER (strict, all modulations) | 0.4999 | 0.4944 |
| Ingestion / classical family / SNR / carrier | unchanged | unchanged |
| CNN, fusion, demod-reached, end-to-end | unchanged | unchanged |

The aggregate median BER barely moves, and that number is misleading. **Per modulation
the change is decisive** (strict BER, IQ records, before → after):

| Modulation | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| BPSK | 0.481 → **0.000** | 0.466 → **0.000** | 0.502 → **0.000** | 0.527 → **0.007** | 0.080 → 0.080 |
| QPSK | 0.496 → **0.000** | 0.504 → **0.000** | 0.502 → **0.001** | 0.488 → **0.055** | not demodulated |
| 8PSK | 0.507 → 0.493 | 0.501 → **0.002** | 0.494 → **0.045** | 0.498 → **0.196** | not demodulated |
| BFSK | 0.508 → 0.505 | 0.494 → 0.506 | 0.498 → 0.506 | 0.502 → 0.510 | not demodulated |
| 16QAM | 0.520 → 0.504 | 0.531 → 0.505 | 0.497 → 0.505 | 0.503 → 0.504 | not demodulated |
| 64QAM | 0.493 → 0.500 | 0.524 → 0.499 | 0.499 → 0.496 | 0.503 → 0.497 | 0.491 → 0.504 |

**This is the first time RadioFry has demodulated anything correctly on V1.** BPSK and
QPSK now reach exactly zero bit errors at high SNR and show a proper monotonic
BER-versus-SNR curve; 8PSK reaches 0.002 at 15 dB.

### Feature selection on the 30 V1 captures

| Selected feature | Times chosen | Correct when chosen |
|---|---|---|
| `phase_second_difference` | 25 / 30 | **25 / 25** |
| `transition_power` | 4 / 30 | **4 / 4** |
| `envelope_power` | 1 / 30 | **0 / 1** |

29/30 correct, matching the harness's 96.7%. The single failure is the one capture where
the original envelope feature won the confidence contest and was wrong — the known
weakness of confidence-based selection, carried over from Entry 004.

### Remaining limitations

- **QAM still fails (~0.50 BER) for an unrelated reason.** `qam_demod.py:13` slices
  against the unnormalized integer grid (±1, ±3, …) while `preprocess` RMS-normalizes
  the capture. Symbol timing is now correct; the constellation scaling is not. Untouched
  by instruction.
- **BFSK still fails (~0.51 BER).** The symbol rate is now recovered, but
  `fsk_demod.py` thresholds the instantaneous frequency of symbol-decimated samples,
  which is a separate defect. Also still subject to the V1 Δf = Rs/2 confound.
- **8PSK at 20 dB shows 0.493 for a classification reason, not a timing one** — the CNN
  and fusion both labelled it `QPSK`, so `dispatch` ran the 4PSK demapper (recovered
  8192 bits against 12288 expected).
- **QPSK at 0 dB is still never demodulated** — fusion rejected it (CNN predicted
  `CPFSK`).
- The selector is the existing unvalidated confidence heuristic. It is right 29/30 here
  but has not been calibrated, and `envelope_power` winning incorrectly once shows the
  failure mode is real.
- Validated only on AWGN, one symbol rate (25 kHz), one sps (8), rect and RRC shaping.
  No CFO, timing offset, fading or multipath — all V2 items.
- Cost: three FFTs per capture instead of one.

### Files changed

```
src/radiofry/dsp/parameter_estimation.py            modified (103 -> 140 lines)
tests/test_symbol_rate_estimator_regression.py      added, 38 tests
reports/v1_after_symbol_rate_fix/                   added (results.csv/json, summary.md)
```

---

## Entry 006 — 2026-09-08 — Investigation: why 16QAM and 64QAM still sit at ~0.5 BER

**Investigation only. No production code was modified.** Full suite re-run unchanged at
**251 passed** (up from 239 by the added diagnostic tests), same single pre-existing
`reedsolo` failure.

### Correction to Entry 005

Entry 005 recorded the remaining QAM failure as an amplitude-scale mismatch. That was
**incomplete, and not the dominant cause.** A pure scale collapse predicts a BER of
0.25 (16QAM) / 0.33 (64QAM), not the ~0.50 that is measured. Chasing that discrepancy
found a second, larger defect. Both are real and independent.

### Defect 1 (dominant) — the correct-order QAM demodulator is never invoked

From `reports/v1_after_symbol_rate_fix/results.csv` (IQ records):

| Capture | Truth | CNN | Fusion | Demodulator actually run | Bits out / expected | BER |
|---|---|---|---|---|---|---|
| 16QAM @ 20 dB | QAM16 | QAM64 | QAM64 | **QAM64** | 24576 / 16384 | 0.504 |
| 16QAM @ 15 dB | QAM16 | QAM64 | QAM64 | **QAM64** | 24576 / 16384 | 0.505 |
| 16QAM @ 10 dB | QAM16 | QAM64 | QAM64 | **QAM64** | 24576 / 16384 | 0.505 |
| 16QAM @ 5 dB | QAM16 | QAM64 | QAM64 | **QAM64** | 24576 / 16384 | 0.504 |
| 16QAM @ 0 dB | QAM16 | CPFSK | Unclassified | none | 0 / 16384 | — |
| 64QAM @ 20 dB | QAM64 | 8PSK | 8PSK | **8PSK** | 12288 / 24576 | 0.500 |
| 64QAM @ 15 dB | QAM64 | 8PSK | 8PSK | **8PSK** | 12288 / 24576 | 0.499 |
| 64QAM @ 10 dB | QAM64 | 8PSK | 8PSK | **8PSK** | 12288 / 24576 | 0.496 |
| 64QAM @ 5 dB | QAM64 | 8PSK | 8PSK | **8PSK** | 12288 / 24576 | 0.497 |
| 64QAM @ 0 dB | QAM64 | QPSK | QPSK | **4PSK** | 8192 / 24576 | 0.504 |

**Not one V1 QAM capture reaches the QAM demodulator at its correct order.** 16QAM is
always demapped as 64QAM (6 bits/symbol instead of 4), and 64QAM is demapped as 8PSK —
a PSK demodulator that ignores amplitude entirely. The wrong bit counts confirm it. The
measured ~0.50 is therefore primarily a **modulation-classification** failure, consistent
with Entry 002's finding that CNN top-1 accuracy on 16QAM and 64QAM is 0%.

### Defect 2 — amplitude-scale mismatch, present even with correct routing

Routing by hand (preprocess → decimate at the true sps → `demodulate_qam` at the correct
order), bypassing CNN and fusion entirely:

| Modulation | SNR | BER | Per-bit-position error rate (MSB…LSB) |
|---|---|---|---|
| 16QAM | noiseless | **0.240** | 0.00, 0.49, 0.00, 0.47 |
| 16QAM | 20 dB | 0.240 | 0.00, 0.49, 0.00, 0.47 |
| 64QAM | noiseless | **0.330** | 0.00, 0.48, 0.50, 0.00, 0.50, 0.51 |
| 64QAM | 20 dB | 0.334 | 0.00, 0.48, 0.50, 0.00, 0.50, 0.51 |

**The failure exists on noiseless captures**, so it is deterministic, not a noise effect.
The per-bit-position pattern is the signature: the sign bits (positions 0 and 2 for
16QAM; 0 and 3 for 64QAM) are recovered **perfectly**, while every magnitude bit is at
chance. 16QAM loses 2 of 4 bits → 0.25; 64QAM loses 4 of 6 → 0.333. Both match.

Cause, measured directly:

| Modulation | Symbol power delivered | Grid power assumed by `qam_demod.py:13` |
|---|---|---|
| 16QAM | 1.000 | 10.0 |
| 64QAM | 1.000 | 42.0 |

V1 normalizes its constellation to unit average symbol power and `preprocess`
RMS-normalizes to unit power, so `demodulate_qam` receives 16QAM levels of ±0.316/±0.949
and slices them against `np.arange(-(side-1), side, 2)` = ±1, ±3. Every level falls
nearest the innermost pair, so all four amplitude levels collapse onto two and only the
sign survives.

**`preprocess` is not the culprit** — it is required. Raw int16 IQ samples arrive at
~±16 000, which would collapse everything onto ±3. The defect is a *convention* mismatch:
the demodulator assumes a constellation of average power 10/42 while the entire pipeline
delivers unit average power, and `demodulate_qam` performs no normalization of its own.
This is precisely why the Entry 005 symbol-rate fix rescued PSK but not QAM —
`demodulate_psk` decides on `np.angle` only and is inherently scale-invariant.

### What is *not* wrong

- **Symbol timing**: decimated samples correlate **0.9987** (16QAM) and **0.9995**
  (64QAM) with the transmitted symbols.
- **Decision grid and bit mapping**: rescaling the symbols by `sqrt(grid_power)` before
  the existing demodulator gives **BER exactly 0.00000** at noiseless and 20 dB for
  16QAM, and 0.00000 noiseless / 0.0145 at 20 dB for 64QAM.
- **Bit mapping convention**: natural-binary MSB-first round-trips exactly through the
  grid, matching the V1 generator.

### Smallest proposed fix (NOT applied)

In `src/radiofry/decoding/demodulators/qam_demod.py`, normalize the incoming symbols to
the average power of the decision grid before slicing — one statement, e.g. scale
`values` by `sqrt(mean(|grid|^2) / mean(|values|^2))`. This makes `demodulate_qam`
scale-invariant, matching the scale-invariance `demodulate_psk` already has, and touches
no other component.

Measured target once applied: 16QAM 0.000 BER at ≥20 dB, 64QAM 0.000 noiseless /
0.0145 at 20 dB. The residual 64QAM error at lower SNR is a genuine Es/N0 limit, not a
defect.

### Expected impact — important sequencing note

**Fixing Defect 2 alone will not move any V1 end-to-end metric.** Because of Defect 1,
16QAM captures are demapped as QAM64 and 64QAM captures as 8PSK, so a corrected QAM
normalization is still applied at the wrong order or bypassed entirely. Its benefit is
only visible in isolation, or after classification is fixed. Defect 1 lives in the CNN
and fusion, which were explicitly out of scope for this investigation.

### Diagnostic artefacts

`tests/test_qam_demodulation_diagnostics.py` — added, **12 tests**, all passing. Asserts
durable conventions rather than current broken outputs, so it survives the fix: V1
transmits QAM at unit average power; the demodulator grid assumes power 10/42; the
pipeline delivers unit-power symbols; decimation recovers the transmitted symbols; the
demodulator is exact once the scales agree; and the bit mapping round-trips.

### Not touched

`decoding/demodulators/qam_demod.py` (fix proposed, not applied), the V1 generator and
dataset, the CNN and checkpoint, `fusion/confidence_fusion.py`,
`dsp/parameter_estimation.py`, `dsp/preprocessing.py`, the FSK/BFSK path, and the
FEC/interleaver components.

---

## Entry 007 — 2026-09-08 — Make QAM demodulation scale-invariant

Implements the fix proposed for Defect 2 in Entry 006. Scope limited to one function;
the CNN and checkpoint, fusion, symbol-rate estimator, preprocessing, V1
generator/dataset, FSK/BFSK path and FEC/interleaver components were not touched.

### Production change

`src/radiofry/decoding/demodulators/qam_demod.py` (20 → 27 lines). One normalization
step inserted between building the decision grid and the existing slicing:

```python
grid_power = 2.0 * float(np.mean(levels.astype(np.float64) ** 2))
power = float(np.mean(np.abs(values.astype(np.complex128)) ** 2))
if power > 0:
    values = (values * np.sqrt(grid_power / power)).astype(np.complex64)
```

`grid_power` evaluates to 10.0 for 16QAM and 42.0 for 64QAM — the average power of the
existing `np.arange(-(side-1), side, 2)` grid. The received symbols are rescaled to that
same average power, after which the **existing grid, `argmin` slicing, index packing and
natural-binary MSB-first bit mapping run unchanged**. The `power > 0` guard keeps
all-zero input safe. The demodulator is now scale-invariant in the same way
`demodulate_psk` already was, which is why PSK never showed this defect.

### Tests

New `tests/test_qam_demodulation_scale.py` — **28 tests**, written before the change;
**20 of them failed beforehand**, all pass now:

- Noiseless 16QAM and 64QAM through the real pipeline path (V1 generator → `preprocess`
  → decimate at true sps) recover the **exact** transmitted bits.
- **Arbitrary positive amplitude scaling**: input multiplied by 1e-4, 0.01, 0.3162, 1.0,
  3.1623, 100, 1e4 yields byte-identical bits, for both orders; separately, scaling a
  20 dB capture by 12 345 and by 1e-6 gives identical output.
- Backward compatibility: symbols already at the historical ±1/±3 grid scale still
  decode exactly.
- Noisy cases: 16QAM exact at 20 dB, 64QAM below 0.05 at 20 dB, 16QAM below 0.15 at
  10 dB, and BER decreasing from 5 dB to 20 dB for both orders.
- Preserved contract: bit/symbol counts, `modulation` label, unsupported-order rejection,
  and all-zero input without a divide-by-zero.

Full suite: **279 passed** (up from 251), same single pre-existing `reedsolo`
environment failure. No previously passing test changed behaviour — in particular the
existing `test_decoding_correlation.py` QAM test still passes.

### Measured BER with correct routing (CNN/fusion bypassed)

| Modulation | SNR | Before (Entry 006) | After |
|---|---|---|---|
| 16QAM | noiseless | 0.2400 | **0.00000** |
| 16QAM | 20 dB | 0.2400 | **0.00000** |
| 16QAM | 15 dB | — | 0.00647 |
| 16QAM | 10 dB | 0.2800 | 0.07556 |
| 16QAM | 5 dB | — | 0.21472 |
| 64QAM | noiseless | 0.3302 | **0.00000** |
| 64QAM | 20 dB | 0.3344 | 0.01465 |
| 64QAM | 15 dB | — | 0.09717 |
| 64QAM | 10 dB | 0.3866 | 0.21175 |
| 64QAM | 5 dB | — | 0.31095 |

Both orders now show a proper monotonic BER-versus-SNR curve and reach exactly zero
errors without noise. The residual error at low SNR is a genuine Es/N0 limit for
higher-order QAM, not a defect.

### V1 end-to-end: unchanged, as predicted

`reports/v1_after_symbol_rate_fix/` → `reports/v1_after_qam_scale_fix/`:

| Metric | Before | After |
|---|---|---|
| Symbol rate within 1% | 0.9667 | 0.9667 |
| CNN accuracy / fusion accuracy | 0.4 / 0.4 | 0.4 / 0.4 |
| Demodulation reached / end-to-end | 0.8667 | 0.8667 |
| Median BER (strict) | 0.4944 | 0.4944 |
| BER scored | 52 | 52 |

Per QAM capture the routing defect is still in control:

| Capture | Demodulator actually run | BER before | BER after |
|---|---|---|---|
| 16QAM @ 20/15/10/5 dB | QAM64 (wrong order) | ~0.504 | ~0.505 |
| 64QAM @ 20/15/10/5 dB | 8PSK (wrong family) | ~0.499 | ~0.499 (identical) |
| 64QAM @ 0 dB | 4PSK | 0.5043 | 0.5043 (identical) |

The 64QAM rows are bit-identical because those captures never enter the QAM path at all.
The 16QAM rows move only in the noise because the QAM64 demapper now normalizes but is
still applied at the wrong order.

### Remaining QAM routing problem (Defect 1, unchanged)

Not one V1 QAM capture reaches `demodulate_qam` at its correct order. CNN top-1 accuracy
on 16QAM and 64QAM is 0%: 16QAM is labelled `QAM64` and 64QAM is labelled `8PSK` (or
`QPSK` at 0 dB), and fusion passes those labels straight through to `dispatch`. Until
classification is fixed, the corrected QAM demodulator cannot demonstrate its measured
capability end-to-end. That work lives in the CNN and fusion, both explicitly out of
scope here.

### Files changed

```
src/radiofry/decoding/demodulators/qam_demod.py     modified (20 -> 27 lines)
tests/test_qam_demodulation_scale.py                added, 28 tests
reports/v1_after_qam_scale_fix/                     added (results.csv/json, summary.md)
```

---

## Entry 008 — 2026-09-08 — Investigation: the QAM classification / routing failure

**Investigation only. No production code, CNN, checkpoint, fusion or dataset was
modified.** Full suite **289 passed** (up from 279 by the added diagnostic tests), same
single pre-existing `reedsolo` failure.

### Path traced

V1 signal → `preprocess` → `_fixed_iq` window → `add_signal_features` (iqap, 4 channels)
→ `ModulationCNN` → softmax → `fuse_modulation` → `demodulate_capture` label→order.

### Ruled out, with evidence

- **Label ordering / checkpoint mapping.** `train_modulation.py:66-67` builds
  `labels = sorted({...})` and indexes targets from that same dict; the checkpoint stores
  that list, and it matches `modulation_cnn_metrics.json["classes"]` exactly
  (`['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']`,
  already sorted). No permutation bug.
- **Checkpoint metadata.** `input_channels=4`, `sample_length=128`, `features='iqap'` —
  and inference uses exactly those values via the same `add_signal_features` helper used
  in training.
- **Fusion.** `fuse_modulation` either forwards the CNN label unchanged or replaces it
  with `"Unclassified"` when confidence < 0.4. It can never turn QAM16 into QAM64.
  Confirmed in a staged trace: 64QAM → CNN said `QAM16` → fused `QAM16`; 16QAM → CNN said
  `8PSK` @0.355 → fused `Unclassified`.
- **Dispatch routing.** `demodulate_capture` maps the label faithfully
  (`QAM16`→order 16, `QAM64`→order 64). It demodulates exactly what the label says.
- **Preprocessing / normalization.** Not implicated — BPSK and QPSK reach 0.987 and
  0.993 confidence through the identical path.

### Root cause 1 (primary, and fixable now) — single-window sampling of a near-chance decision

`_fixed_iq` feeds the CNN **one centred 128-sample window out of 32 768** (0.4% of the
capture). The QAM16-vs-QAM64 decision is close to a coin flip per window, so that single
draw decides the outcome.

Probabilities on the real V1 capture files (centred window, via the actual ingestion
path):

| Capture | P(QAM16) | P(QAM64) | top-1 | conf |
|---|---|---|---|---|
| 16QAM @ 20 dB | 0.419 | 0.566 | QAM64 | 0.566 |
| 16QAM @ 15 dB | 0.428 | 0.556 | QAM64 | 0.556 |
| 16QAM @ 10 dB | 0.448 | 0.539 | QAM64 | 0.539 |
| 16QAM @ 5 dB | 0.430 | 0.521 | QAM64 | 0.521 |
| 16QAM @ 0 dB | 0.199 | 0.191 | CPFSK | 0.229 |
| 64QAM @ 20 dB | 0.163 | 0.092 | 8PSK | 0.716 |
| 64QAM @ 15 dB | 0.314 | 0.210 | 8PSK | 0.424 |
| 64QAM @ 10 dB | 0.042 | 0.023 | 8PSK | 0.793 |

The 16QAM rows are not confidently wrong — they are ~0.43 vs ~0.55 splits that land on
the wrong side. **Sliding the window proves the information is present:** across 64
windows of the *same* 16QAM @ 20 dB capture the argmax is QAM16 **34/64** and QAM64 the
rest. The centred window simply drew a loser.

Aggregating across windows (majority vote of per-window argmax), on the real captures:

| | Centred window (current) | Majority vote, 64 windows | Mean probability, 64 windows |
|---|---|---|---|
| QAM captures | **0 / 10** | **9 / 10** | 7 / 10 |
| All 30 V1 captures | **12 / 30** | **22 / 30** | 21 / 30 |

Majority vote beats probability averaging, and both beat a single window.

**Window-count sweep is noisy and non-monotonic**, which matters for sizing the fix:

| Windows | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| All 30 | 12 | 20 | 20 | **16** | 19 | **22** |
| QAM only (10) | 0 | 5 | 6 | 3 | 5 | **9** |

Any count ≥4 clearly beats a single window, but 16 underperforming 8 and 32 shows the
estimate is high-variance at n=30. The exact window count must be validated on more
captures and seeds before being fixed in code.

### Root cause 2 (intrinsic to the checkpoint, not fixable without retraining)

The model separates QAM16 from QAM64 poorly **on its own RML test set**. From
`modulation_cnn_metrics.json`, high-SNR bucket:

| Truth | → QAM16 | → QAM64 | accuracy |
|---|---|---|---|
| QAM16 | 310 | 262 | **0.52** |
| QAM64 | 227 | 357 | **0.59** |

Every other class in that bucket scores ≥0.92 (BPSK 0.98, QPSK 0.98, 8PSK 0.97, CPFSK
1.00). QAM16↔QAM64 is by a wide margin the model's worst pair, and it is an intrinsic
property of this checkpoint, not a V1 artefact. This puts a ceiling on what any
inference-side change can achieve.

### Root cause 3 (contributing) — pulse-shape domain gap

RML2016.10a is RRC-shaped; V1 uses rectangular pulses. Feeding RRC-shaped equivalents
(the Entry 004 builder) through the same path, noiseless:

| Modulation | Pulse | P(QAM16) | P(QAM64) | top-1 |
|---|---|---|---|---|
| 16QAM | rect | 0.319 | 0.226 | 8PSK (0.355) |
| 16QAM | rrc 0.35 | **0.593** | 0.276 | **QAM16** |
| 64QAM | rect | 0.499 | 0.450 | QAM16 |
| 64QAM | rrc 0.35 | 0.534 | 0.464 | QAM16 |

Over 20 seeds, single-window 16QAM top-1 is 5/20 on rect versus 10/20 on RRC. Real, but
a smaller effect than the windowing one. BPSK and QPSK are unaffected by pulse shape
(0.987/0.989 and 0.993/0.998).

### Which layer is at fault

**Multiple layers, but not the ones previously suspected.** In order of impact:
inference-side **windowing** (fixable now, 12/30 → 22/30), **model quality** on the
QAM16/QAM64 pair (needs retraining), and the **dataset domain gap** (a V2 item).
**Fusion, dispatch routing, label ordering, checkpoint metadata and preprocessing are all
correct** and were positively ruled out.

### Smallest proposed fix (NOT applied)

In `src/radiofry/models/modulation_inference.py` only: evaluate several evenly spaced
128-sample windows instead of one and aggregate by **majority vote** of per-window
argmax, reporting the winner. No architecture change, no retraining, no fusion change,
no dataset change.

Two things must be decided before implementing: the window count (the sweep above is too
noisy at n=30 to pick one — validate on more captures/seeds first), and what
`confidence` should mean afterwards, since fusion gates on it at 0.4 and vote share is a
different quantity from softmax probability.

### Expected V1 impact

Measured on the real captures: CNN top-1 **12/30 → 22/30** overall and **0/10 → 9/10**
on QAM. End-to-end BER improvement is *not* predictable from this alone, because the
downstream result then depends on fusion's 0.4 threshold interacting with whatever
confidence definition replaces the softmax value. With Entry 007's scale fix already in
place, correctly routed QAM demodulation is independently verified at 0.000 BER
noiseless and at 20 dB (16QAM), so correct routing is the last missing link for QAM.

### Diagnostic artefacts

`tests/test_qam_routing_diagnostics.py` — added, **10 tests**, all passing. Pins the
structural contracts that were ruled out: checkpoint labels match the metrics classes and
are sorted, both QAM classes exist, fusion only forwards or rejects a label (never
substitutes another modulation), fusion forwards a confident label even on family
disagreement, and dispatch maps each QAM label to its own order.

### Must remain untouched

The CNN architecture and checkpoint (no retraining), `fusion/confidence_fusion.py`, the
V1 generator and dataset, `dsp/preprocessing.py`, `dsp/parameter_estimation.py`,
`decoding/demodulators/qam_demod.py` (already fixed in Entry 007), the FSK/BFSK path,
and the FEC/interleaver components. The proposed change is confined to
`models/modulation_inference.py`.

---

## Entry 009 — 2026-09-08 — Verification run and commit handoff

**No production code was modified.** Verification and packaging only.

### Tests

`python -m pytest -q --ignore=tests/test_model_report.py` at 00:42:46 IST, 290 collected:

**289 passed, 1 failed.**

Per-module, covering all work from Entries 001–008:

| Module | Result | | Module | Result |
|---|---|---|---|---|
| test_synthetic_v1_modulation | 47 passed | | test_symbol_rate_diagnostics | 19 passed |
| test_synthetic_v1_io | 10 passed | | test_symbol_rate_feature_comparison | 25 passed |
| test_synthetic_v1_dataset | 20 passed | | test_symbol_rate_estimator_regression | 38 passed |
| test_v1_harness | 17 passed | | test_qam_demodulation_diagnostics | 12 passed |
| test_v1_harness_metrics | 14 passed | | test_qam_demodulation_scale | 28 passed |
| test_modulation_inference_input | 9 passed | | test_qam_routing_diagnostics | 10 passed |

The single failure, `tests/test_decoding_correlation.py::test_reed_solomon_round_trip`,
is the pre-existing environmental gap tracked since Entry 001 — `reedsolo` is not
installed. `commpy`, `pyldpc` and `h5py` are also absent (the last skips
`tests/test_model_report.py`).

### Observations

- Production files confirmed unchanged since Entry 008: `modulation_inference.py`
  78 lines, `parameter_estimation.py` 140 lines, `qam_demod.py` 27 lines.
- The multi-window CNN inference change recommended in Entry 008 is **not started**.
- Generated artefacts total ~11 MB (`data/synthetic_v1` 8.8 MB,
  `data/synthetic_v1_shortframe` 316 KB, seven `reports/` directories ~2.2 MB) and are
  **not** matched by the current `.gitignore`. They must be excluded manually when
  staging; `.gitignore` was deliberately left unmodified.

### Files added in this entry

```
OP.md                       operational run log (Run 001)
ANTIGRAVITY_COMMIT.txt      commit handoff for branch feature/synthetic-dataset-v1
```

---

## Entry 010 — 2026-09-08 — Multi-window CNN inference: experiment, decision, implementation

Follows the Entry 008 recommendation. Entry 008's sweep was n=30 with one seed per
condition and was explicitly flagged as noisy; this entry replaces it with a properly
powered paired experiment before changing anything.

### Phase 1 — experiment (MEASURED)

In-memory only, no artefacts written. 300 captures (6 modulations × 5 SNRs × 10 seeds),
4096 symbols at 8 sps / 200 kHz — the same parameters as `data/synthetic_v1`. The
**same** capture is scored under every window count and strategy, so all comparisons are
paired. Window count 1 uses the centred window, reproducing production exactly.

Strategies compared: `vote_share` (label by majority vote, confidence = share of
windows), `vote_meanprob` (label by vote, confidence = mean softmax of that label),
`mean_prob` (label and confidence both from the averaged softmax vector).

Overall top-1 accuracy (n=300):

| Strategy | 1w | 4w | 8w | 16w | 32w | 64w |
|---|---|---|---|---|---|---|
| vote_share | 0.557 | 0.643 | 0.613 | **0.673** | 0.650 | 0.663 |
| vote_meanprob | 0.557 | 0.643 | 0.613 | 0.673 | 0.650 | 0.663 |
| mean_prob | 0.557 | 0.650 | 0.623 | 0.663 | 0.653 | 0.653 |

Top-3 accuracy barely moves (0.797 → 0.797–0.817): the gain is in ranking the correct
class first, not in candidate coverage.

**Paired McNemar tests (mean_prob) settle the window count:**

| Comparison | better | worse | net | p |
|---|---|---|---|---|
| 1w → 4w | 43 | 15 | **+28** | **0.0003** |
| 4w → 8w | 15 | 23 | −8 | 0.26 |
| 4w → 16w | 18 | 14 | +4 | 0.60 |
| 4w → 32w | 19 | 18 | +1 | 1.00 |
| 4w → 64w | 18 | 17 | +1 | 1.00 |
| 16w → 64w | 5 | 8 | −3 | 0.58 |

**Interpretation:** the entire statistically significant gain is realised at four
windows. No count above four is distinguishable from four. Entry 008's apparent
preference for 64 windows was sampling noise.

Strategy choice, paired at every count: `vote_meanprob` vs `mean_prob` differ on at most
17 of 300 captures, p ≥ 0.55 everywhere — **statistically indistinguishable**, so the
choice must be made on semantics rather than accuracy.

**Confidence-contract check at count = 1 (MEASURED):**

| Strategy | mean confidence | rejection rate at fusion's 0.4 |
|---|---|---|
| mean_prob / vote_meanprob | 0.699 | 0.133 |
| vote_share | **1.000** | **0.000** |

`vote_share` is degenerate at one window — a single window always votes 100% for its own
argmax — so adopting it would silently disable fusion's rejection path. It is rejected
despite scoring marginally higher on raw accuracy. `mean_prob` at count 1 is identical to
current production in both label and confidence (verified element-wise).

Per-modulation top-1 (mean_prob, n=50 each): BPSK 1.00 → 1.00, QPSK 0.78 → 0.88,
8PSK 0.88 → 0.96, BFSK 0.00 → 0.00, 16QAM 0.32 → 0.56, 64QAM 0.36 → 0.52 (1w → 4w).

Cost of multi-window: rejection rises (0.133 → 0.193 at 4w) because averaging pulls peak
probabilities down. The metric that accounts for both effects — **correct and not
rejected** — still improves, 0.517 → 0.580 for mean_prob. Accuracy among accepted
captures rises 0.596 → 0.719, so the rejection path is filtering usefully.

### Phase 2 — decision

**Implement**, with **4 windows** and **mean-softmax aggregation**.

- 4 windows because it captures the whole measured gain (p = 0.0003 against 1 window)
  while nothing larger is statistically better — the smallest count the data supports.
- mean-softmax because the average of probability vectors is itself a probability
  vector, so `confidence` keeps the scale fusion's 0.4 threshold was written against, and
  at one window it is bit-identical to previous behaviour. Vote share is not a
  probability and is degenerate at one window.

Fusion was **not** touched. The rejection-rate increase is reported as a real
consequence, not tuned away.

### Phase 3 — production change

`src/radiofry/models/modulation_inference.py` (78 → 106 lines), three additive edits:

1. `DEFAULT_INFERENCE_WINDOWS = 4`.
2. `_window_frames(signal, length, windows)` — evenly spaced contiguous native-rate
   frames, each RMS-normalised exactly as `_fixed_iq` does. Captures at or below the
   frame length, and `windows <= 1`, return `[_fixed_iq(...)]`, so short and
   exact-length behaviour is unchanged.
3. `predict_modulation(..., windows=DEFAULT_INFERENCE_WINDOWS)` batches the frames
   through one forward pass and averages the softmax vectors.

**`_fixed_iq` is unchanged**, so its nine existing tests still pass unmodified. The
`ModulationPrediction` contract is unchanged.

### Phase 3 — tests

New `tests/test_modulation_inference_windows.py`, **21 tests**, written before the
change (failed on import beforehand): window count/shape, contiguous native-rate
extraction verified by per-window phase advance, distinct windows spanning the capture,
single-window equivalence to the centred frame, short and exact-length captures yielding
exactly one unchanged frame at any window count, empty-capture rejection, single-window
inference matching the centred frame scored alone, output contract, confidence equal to
the top aggregate probability and on the softmax scale, determinism, configurability,
non-positive counts falling back to one window, and missing-checkpoint handling.

Full suite: **310 passed** (up from 289), same single pre-existing `reedsolo` failure.

### Phase 4 — V1 evaluation, before → after

`reports/v1_after_qam_scale_fix/` → `reports/v1_after_multiwindow/`.

| Metric | Before | After |
|---|---|---|
| CNN top-1 accuracy | 0.400 | **0.733** |
| CNN top-3 accuracy | 0.733 | **0.833** |
| Fusion accuracy | 0.400 | **0.633** |
| **Median BER (strict)** | 0.4944 | **0.0293** |
| Mean CNN confidence | 0.722 | 0.644 |
| Fusion rejection rate | 0.133 | 0.200 |
| Demodulation reached | 0.867 | 0.800 |
| BER scored | 52 | 48 |
| Symbol rate within 1% | 0.967 | 0.967 |

Per-modulation CNN top-1: BPSK 1.00 → 1.00, QPSK 0.80 → 1.00, 8PSK 0.60 → 1.00,
BFSK 0.00 → 0.00, **16QAM 0.00 → 0.80**, **64QAM 0.00 → 0.60**.

Per-modulation strict BER (IQ), before → after:

| Modulation | 20 dB | 15 dB | 10 dB | 5 dB |
|---|---|---|---|---|
| BPSK | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.007 → 0.007 |
| QPSK | 0.000 → 0.000 | 0.000 → 0.000 | 0.001 → 0.001 | 0.055 → 0.055 |
| 8PSK | 0.493 → **0.000** | 0.002 → 0.002 | 0.045 → 0.045 | 0.196 → 0.196 |
| BFSK | 0.505 → 0.505 | 0.506 → 0.506 | 0.506 → 0.506 | 0.510 → 0.510 |
| 16QAM | 0.506 → **0.000** | 0.504 → **0.005** | 0.507 → **0.080** | 0.510 → rejected |
| 64QAM | 0.500 → **0.014** | 0.499 → **0.106** | 0.496 → **0.226** | 0.497 → rejected |

**QAM demodulates correctly end-to-end for the first time.** Combined with Entries 005
and 007, 16QAM reaches 0.000 BER at 20 dB through the unmodified pipeline.

### Regression, reported in full

Demodulation reached fell 0.867 → 0.800 and BER-scored 52 → 48. Exactly three captures
lost demodulation because their confidence dropped below fusion's 0.4 threshold:
16QAM @ 5 dB (0.521 → 0.313), 64QAM @ 5 dB (0.677 → 0.336), 64QAM @ 0 dB (0.421 →
0.261). **All three were previously demodulated under a wrong label and produced
chance-level BER (~0.50)**, so nothing of value was lost — they are now honestly
rejected instead of silently producing garbage. One capture moved the other way,
BFSK @ 0 dB (Unclassified → 8PSK, BER 0.512), which is a wrong label that is now
demodulated; BFSK remains broken for reasons unrelated to this change.

Mean CNN confidence fell 0.722 → 0.644. This is expected from averaging and was
predicted by the experiment; it is a real change to the operating point of an
unvalidated threshold and is left uncorrected by design.

### Remaining limitations

- **BFSK is still 0% and ~0.51 BER.** Unaffected by this change; separate defect.
- The 0.4 fusion threshold was tuned against single-window confidences and is now
  applied to averaged ones. It was deliberately not adjusted. Whether 0.4 is still the
  right operating point is an open question that needs its own calibration study.
- The checkpoint's intrinsic QAM16/QAM64 weakness (Entry 008: 0.52/0.59 on its own RML
  test set) still caps accuracy; this change recovers what windowing was throwing away,
  it does not improve the model.
- Cost is four forward passes per capture instead of one.
- Validated on V1 only: AWGN, rectangular pulses, one symbol rate, one sps.

---

## Entry 011 — 2026-09-08 — Investigation: the BFSK failure

**Investigation only. No production code was modified.** Full suite unchanged at
**310 passed**, same single pre-existing `reedsolo` failure. `modulation_inference.py`
116 lines, `fsk_demod.py` 15, `dispatch.py` 64, `synthetic_gen/v1/modulation.py` 72 —
all unchanged.

### Failure chain (MEASURED, real V1 capture files, matched Fs/Rs/sps/length/format)

| Stage | BFSK | Evidence |
|---|---|---|
| Ingestion | **PASS** | 32 768 samples read, matches ground truth, all SNRs |
| Preprocessing | **PASS** | per-sample phase increment median identical raw vs preprocessed (0.3927 rad); tone estimates identical to 1 Hz |
| Parameter estimation | **PASS** | estimated Rs = 25 000 Hz exactly at 20/15/10/5 dB (28 979 at 0 dB) |
| Classical family | **PARTIAL** | `FSK-like` at 20/15 dB, `QAM-like` at 10/5/0 dB |
| CNN | **FAIL** | predicts `8PSK` at **0.956 / 0.953 / 0.966 / 0.889** confidence |
| Fusion | **PASS (forwards)** | confidence > 0.4, so it faithfully forwards the wrong label |
| Demodulator | **FAIL (never invoked)** | routed to 8PSK: 12 288 bits out vs 4 096 expected |
| BER | **FAIL** | 0.5046 / 0.5059 / 0.5063 / 0.5103 |

BPSK and QPSK on identical settings reach 0.0000 BER, so nothing in the shared path is
at fault.

### The V1 BFSK waveform is correct — measured, not assumed

Ground truth: Rs 25 000 Hz, sps 8, fs 200 000 Hz, deviation 12 500 Hz. Measured
per-sample phase advance 0.3927 rad = exactly pi/8, spectrum peaks at ±12 500 Hz,
envelope CV 0.0033. The generator produces exactly what it claims. **But that means
modulation index h = 2*dev/Rs = 1.000, and the phase advance per symbol is exactly
±pi.**

### Root cause 1 — CNN: h = 1 is off-distribution and looks like 8PSK

Per-window probabilities are **confidently and unanimously wrong**, unlike QAM which was
a coin flip: `['8PSK:0.99','8PSK:1.00','8PSK:0.99','8PSK:0.85']`, 4-window mean
`8PSK:0.959`. This is why multi-window inference (Entry 010) did not move BFSK — every
window agrees on the wrong answer.

**Deviation probe (noiseless, generator's existing `fsk_deviation_hz` parameter):**

| deviation | h | phase/symbol | CNN top-1 | confidence |
|---|---|---|---|---|
| 12 500 (V1 default) | 1.00 | 3.142 | `8PSK` | 0.959 |
| 6 250 | 0.50 | 1.571 | **`CPFSK`** | **1.000** |
| 3 125 | 0.25 | 0.785 | **`GFSK`** | **1.000** |
| 25 000 | 2.00 | 6.283 | `AM-SSB` | 0.395 |

**The CNN classifies FSK essentially perfectly at h = 0.5 and h = 0.25 and fails only at
h = 1.** Distinct IQ phases on the unit circle: BFSK@h=1 lands on exactly **16** phases
with constant envelope (CV 0.0033) — a dense PSK-like phase grid — versus 32 at h=0.5,
8 for 8PSK, 4 for QPSK, 2 for BPSK. RML2016.10a's CPFSK uses the standard h = 0.5.

### Root cause 2 — demodulator: ±pi per symbol is ambiguous after decimation

`dispatch.py:43` decimates to one sample per symbol before calling `demodulate_fsk`,
which then takes `diff(unwrap(angle(...)))`. At h = 1 the per-symbol advance is ±pi,
which is congruent mod 2*pi — the discriminator cannot tell the two tones apart.

| Path | noiseless | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| Production (decimated), h=1 | 0.4984 | 0.4884 | 0.4884 | 0.4894 | — |
| Decimated, best over all 8 timing offsets × 3 bit shifts, h=1 | **0.2512** | — | — | — | — |
| **Same demodulator at full rate, per-symbol decision, h=1** | **0.0000** | **0.0000** | 0.0034 | 0.0625 | 0.2269 |
| Production (decimated), h=0.5 | 0.2530 | 0.2584 | 0.2549 | 0.4392 | — |
| Decimated, best over offsets × shifts, h=0.5 | **0.0000** | — | — | — | — |

**Measured facts:** (a) at h = 1 the decimated path is irrecoverable — 0.2512 even after
searching every timing offset and bit shift, noiseless; (b) `demodulate_fsk`'s own
discriminator logic is **correct** — 0.0000 BER at full rate with a proper BER-vs-SNR
curve; (c) at h = 0.5 the decimated path *is* recoverable (0.0000 with the right offset
and shift), so decimation is not fundamentally broken for FSK, only for h ≥ 1.

### Interpretation

Both failures trace to the single generator choice **Δf = Rs/2 (h = 1)** made in Entry
001 and already flagged as a confound in Entry 003. It simultaneously puts the waveform
off the CNN's training distribution and drives the decimated discriminator to its
ambiguity limit. `fsk_demod.py` and `preprocessing` are not defective;
`dispatch`'s decimation is a genuine latent limitation that only bites at h ≥ 1.

### Hypotheses (NOT established)

- The residual gap at h = 0.5 (dispatch 0.2530 vs 0.0000 with the right offset/shift) is
  most likely a timing-offset plus off-by-one effect: `np.diff` returns N−1 values, so
  bit k corresponds to a symbol transition rather than symbol k, and `dispatch`'s
  smoothness-based offset search is not choosing the offset that aligns them. The exact
  mechanism was **not** pinned down.
- That the CNN's 8PSK answer is caused specifically by the 16-point phase grid is a
  plausible reading of the phase-count measurement, not a proven attribution.

### Proposed next fix (NOT applied)

Make the FSK deviation an explicitly swept V1 dimension with the standard **h = 0.5**
(Δf = Rs/4) as the default, retaining h = 1.0 as a labelled known-hard case rather than
deleting it. This is the only change that unblocks the whole FSK path: the CNN then
classifies `CPFSK` at 1.000 confidence and the existing demodulator becomes recoverable.
It does not hide the h = 1 limitation — it makes it explicit and measurable.

The alignment defect above should then be re-measured; it is expected to surface as the
next BFSK blocker once routing works.

### Not touched

CNN architecture and checkpoint (no retraining), fusion, symbol-rate estimator,
`fsk_demod.py`, `dispatch.py`, preprocessing, QAM path, multi-window inference, and the
V1 generator and dataset.

---

## Entry 012 - 2026-09-08 - Make FSK deviation an explicit swept V1 dimension

Implements the single next task recommended by Entry 011. Changes are confined to the
V1 generator package. The CNN, checkpoint, fusion, `modulation_inference.py`,
`fsk_demod.py`, `dispatch.py`, QAM path and symbol-rate estimator were **not** modified.

### Generator change (PRODUCTION)

`src/radiofry/synthetic_gen/v1/config.py` (140 -> 179 lines):

- `DEFAULT_FSK_MODULATION_INDEX = 0.5`, `KNOWN_HARD_FSK_MODULATION_INDICES = (1.0,)`,
  `DEFAULT_FSK_MODULATION_INDICES = (0.5, 1.0)`, and a `KNOWN_HARD_FSK_REASON` string.
- `SampleSpec` gains `fsk_modulation_index`. Deviation and index resolve against each
  other (h = 2*deviation/Rs); supplying both inconsistently raises. **The default is now
  h = 0.5 (deviation Rs/4)**, previously an implicit Rs/2. The index resolves to `None`
  for non-FSK modulations.

`src/radiofry/synthetic_gen/v1/generator.py` (~350 -> 386 lines):

- `DatasetSpec.fsk_modulation_indices` defaults to `(0.5, 1.0)`; FSK modulations are
  swept over it, every other modulation runs exactly once as before.
- `capture_id_for(..., fsk_modulation_index=...)` appends `_h0.5` / `_h1.0` **for FSK
  only**; non-FSK capture ids are unchanged.
- Ground truth records `signal.fsk_modulation_index` alongside the deviation, plus
  top-level `known_hard` and `known_hard_reason`.
- Manifest gains `fsk_modulation_index`, `fsk_deviation_hz` and `known_hard` columns;
  `dataset.json` records the swept indices.
- The per-capture seed deliberately **excludes** the index, so the two FSK arms share one
  payload and one noise realisation and differ only in h - a properly paired comparison.

`cli.py` gains `--fsk-modulation-index` (and its capture-count printout now accounts for
the sweep). `__init__.py` re-exports the new constants.

### Dataset regeneration - non-FSK provably unchanged

`data/synthetic_v1` regenerated: 30 -> 40 captures, 8.8 -> 12 MB.

- **25/25 non-FSK captures byte-identical** to before (verified by SHA-256).
- **5/5 previous BFSK captures byte-identical**, now carried as the `_h1.0` arm.
- 5 new `_h0.5` captures added.

### MEASURED - h = 0.5 (new standard default)

| SNR | deviation | measured tone sep | est Rs | classical | CNN | conf | fused | dispatch BER | full-rate BER |
|---|---|---|---|---|---|---|---|---|---|
| 20 dB | 6250 | 12 629 Hz | 25 000 | FSK-like | **CPFSK** | **1.000** | CPFSK | 0.2437 | **0.0000** |
| 15 dB | 6250 | 14 106 Hz | 25 000 | FSK-like | **CPFSK** | **1.000** | CPFSK | 0.2623 | **0.0063** |
| 10 dB | 6250 | 19 426 Hz | 25 000 | QAM-like | **CPFSK** | **1.000** | CPFSK | 0.2530 | 0.0637 |
| 5 dB | 6250 | 33 206 Hz | 7 574 | QAM-like | **CPFSK** | 0.999 | CPFSK | 0.4909 | 0.1875 |
| 0 dB | 6250 | 59 705 Hz | 23 486 | QAM-like | **CPFSK** | 0.935 | CPFSK | 0.4967 | 0.3397 |

Tone separation measures 12 629 Hz against the expected 2*6250 = 12 500 Hz at 20 dB;
the growth at low SNR is noise in the instantaneous-frequency estimate, not signal.

### MEASURED - h = 1.0 (known-hard, retained)

| SNR | CNN | conf | fused | dispatch BER | full-rate BER |
|---|---|---|---|---|---|
| 20 dB | 8PSK | 0.956 | 8PSK | 0.5046 | **0.0000** |
| 15 dB | 8PSK | 0.953 | 8PSK | 0.5059 | **0.0000** |
| 10 dB | 8PSK | 0.966 | 8PSK | 0.5063 | 0.0029 |
| 5 dB | 8PSK | 0.889 | 8PSK | 0.5103 | 0.0520 |
| 0 dB | 8PSK | 0.532 | 8PSK | 0.5122 | 0.2147 |

Reproduces Entry 011 exactly, on byte-identical captures.

### Result: classification is fixed, production demodulation is not

**h = 0.5 fixes classification and routing completely.** The CNN returns `CPFSK` at
1.000 confidence at 20/15/10 dB and fusion forwards it, so the FSK demodulator is now
correctly invoked for the first time. **Production dispatch BER nevertheless stays at
~0.25**, so per the task instruction the demodulator was left untouched.

### Exact location of the remaining failure (MEASURED, not hypothesised)

Entry 011 recorded the off-by-one explanation as an unproven hypothesis. It is now
measured. Sweeping every timing offset and bit shift on `BFSK_h0.5_snr20dB_r000`:

| offset | shift -1 | shift 0 | shift +1 | shift +2 |
|---|---|---|---|---|
| 0 | 0.5005 | **0.0049** | 0.4957 | 0.4971 |
| 1 | 0.5024 | **0.0049** | 0.4957 | 0.4971 |
| 2 | 0.5010 | **0.0049** | 0.4957 | 0.4990 |
| **3 (dispatch picks this)** | 0.5039 | **0.2437** | 0.2569 | 0.5020 |
| 4 | 0.4976 | 0.4957 | **0.0049** | 0.5022 |
| 5-7 | ~0.496 | ~0.496 | **0.0049** | ~0.501 |

**Seven of the eight timing offsets reach 0.0049 BER. The smoothness heuristic in
`dispatch.py` - minimise `mean|diff(iq[offset::sps])|` over offsets - selects offset 3,
the single worst one.** The heuristic was designed for linear modulations, where
mid-symbol sampling minimises transitions; for constant-envelope CPFSK it lands on a
boundary-straddling offset.

There is a second, smaller effect: `np.diff` in `fsk_demod.py` returns N-1 values
(4095 vs 4096 expected bits), which shifts the correct bit alignment from 0 to +1 for
offsets >= 4. The harness `ber_aligned` search covers that one.

**Failure location: `src/radiofry/decoding/demodulators/dispatch.py:36-42`, the timing
offset selection - not `fsk_demod.py`, whose discriminator reaches 0.0000 BER.**

### V1 evaluation, 30 -> 40 captures

| Metric | Before (30) | After (40) |
|---|---|---|
| CNN top-1 | 0.7333 | **0.7714** |
| CNN top-3 | 0.8333 | **0.8571** |
| Fusion accuracy | 0.6333 | **0.6857** |
| Rejection rate | 0.200 | 0.171 |
| Demod reached | 0.800 | 0.829 |
| Median BER (strict) | 0.0293 | 0.0796 |
| BER scored | 48 | 58 |

Per-modulation CNN top-1: BPSK 1.00, QPSK 1.00, 8PSK 1.00, 16QAM 0.80, 64QAM 0.60,
**BFSK 0.50** (5/5 correct at h=0.5, 0/5 at h=1.0 - exactly the intended split).

The median BER rise from 0.0293 to 0.0796 is a **denominator effect, not a regression**:
ten BFSK captures at 0.24-0.51 BER now sit in a 40-capture set instead of five in a
30-capture set. No individual capture got worse; the non-FSK captures are byte-identical
and produced identical results.

### Tests

13 new/updated tests across `tests/test_synthetic_v1_modulation.py` and
`tests/test_synthetic_v1_dataset.py`: default h=0.5, explicit h=1.0, explicit deviation
reporting its index, inconsistent deviation+index rejected, non-FSK reporting no index,
FSK swept over both arms while non-FSK is not, capture-id tagging for FSK only, non-FSK
ids unchanged, ground truth recording deviation/index/hardness, non-FSK not marked hard,
shared payload across the two arms, configurable sweep, and reproducible generation.

Full suite: **323 passed** (up from 310), same single pre-existing `reedsolo` failure.

### Remaining blocker

BFSK is **not** fully working. Classification is solved at h=0.5; production
demodulation is blocked by the timing-offset heuristic in `dispatch.py`. The next task
would be to make that heuristic pick a valid offset for constant-envelope FSK - but that
is a demodulator-path change, deliberately out of scope here.

h=1.0 remains classified as 8PSK. That is now an explicitly labelled `known_hard`
condition rather than a silent failure, and would require CNN retraining to address.

---

## Entry 013 - 2026-09-08 - FSK-aware timing-offset selection in dispatch

Implements V1 Task #13, the blocker identified in Entry 012. The change is confined to
`dispatch.py`. The CNN, checkpoint, `modulation_inference.py`, fusion,
symbol-rate estimator, `qam_demod.py`, `fsk_demod.py` and the V1 generator were **not**
modified.

### The old criterion and why it fails for CPFSK

`dispatch.py` selected the symbol-sampling offset by minimising
`mean|diff(iq[offset::sps])|` over all offsets - "pick the alignment whose decimated
samples move least". That is sound for linear modulations, where mid-symbol sampling
minimises transitions.

It is **actively wrong** for constant-envelope FSK. For offset k the phase advance
between consecutive decimated samples spans `sps-1-k` samples of symbol n and `k+1`
samples of symbol n+1. When those two symbols carry opposite tones the contributions
partly cancel, so a boundary-straddling offset produces the *smallest* movement and wins
the minimisation. At sps=8 the exact tie is k=3 (4 samples of each symbol), which is
precisely the offset the heuristic selected and the one Entry 012 measured at 0.2437 BER.

### Candidate criteria tested (MEASURED)

Signal-derived only; ground truth was used solely to score them offline. h=0.5 BFSK,
5 seeds x 6 SNRs = 30 cases, 2048 symbols each.

| Criterion | Offset chosen | Mean strict BER |
|---|---|---|
| `current` - min mean\|diff\| (existing) | 3 | 0.2677 |
| **`within_var` - min instantaneous-frequency variance inside each symbol window** | **0** | **0.0508** |
| `f_ratio` - max between/within frequency variance | 0 | 0.0508 |
| `max|diff|` - inverse of the existing heuristic (control) | 7 | 0.5001 |
| *oracle (best offset per case, not implementable)* | - | *0.0494* |

`within_var` and `f_ratio` select identically. `within_var` was chosen: it is a single
statistic with no ratio or epsilon guard. At 0.0508 against an oracle of 0.0494 it is
essentially optimal. `max|diff|` confirms that simply inverting the old heuristic is not
a fix - it selects offset 7, which is a valid alignment but attributes each decision to
symbol n+1, so the unshifted bitstream is wrong.

### Production change

`src/radiofry/decoding/demodulators/dispatch.py` (64 -> 99 lines):

- `_linear_timing_offset(iq, sps)` - the existing heuristic, extracted verbatim.
- `_fsk_timing_offset(iq, sps)` - new. Computes the full-rate instantaneous frequency,
  reshapes it into candidate symbol windows starting at each offset, and returns the
  offset minimising the mean within-window variance. Guards `sps <= 1` and captures
  shorter than two symbols by returning 0.
- `demodulate_capture` picks `_fsk_timing_offset` when the label is in
  `_FSK_LABELS = {"CPFSK", "GFSK"}` and `_linear_timing_offset` otherwise.

The criterion reads only the received IQ and the samples-per-symbol derived from the
estimated symbol rate. No transmitted bits, no BER, no ground truth, no hard-coded
offset.

### MEASURED - h = 0.5 (standard), 5 seeds per SNR

| SNR | offset chosen | NEW dispatch | OLD heuristic | full-rate reference |
|---|---|---|---|---|
| noiseless | 0 | **0.0181** | 0.2502 | 0.0000 |
| 20 dB | 0 | **0.0180** | 0.2517 | 0.0002 |
| 15 dB | 0 | **0.0180** | 0.2525 | 0.0058 |
| 10 dB | 0 | **0.0180** | 0.2521 | 0.0599 |
| 5 dB | 0 | 0.0360 | 0.2585 | 0.1904 |
| 0 dB | 0 | 0.1970 | 0.3412 | 0.3324 |

Offset 0 is selected for every seed and every SNR. The residual ~0.018 at high SNR is
inherent to deciding from one sample per symbol rather than averaging the whole symbol;
the full-rate reference reaches 0.0000. That gap is a property of the existing
decimate-then-discriminate design, not of the timing criterion.

### MEASURED - h = 1.0 (known-hard, NOT solved)

| SNR | NEW dispatch | OLD heuristic |
|---|---|---|
| noiseless | 0.2774 | 0.5018 |
| 20 dB | 0.2523 | 0.4950 |
| 10 dB | 0.2634 | 0.4928 |

**This fix does not solve h=1.0 and does not claim to.** The +/-pi per-symbol advance
remains ambiguous regardless of alignment. Note also that in the full pipeline h=1.0
captures never reach this code path at all: the CNN labels them 8PSK, so dispatch routes
them to the PSK demodulator. Their V1 numbers are therefore completely unchanged.

### V1 evaluation, 40 captures

| Metric | Before | After |
|---|---|---|
| **Median BER (strict)** | 0.0796 | **0.0138** |
| CNN top-1 / fusion accuracy | 0.7714 / 0.6857 | 0.7714 / 0.6857 (unchanged) |
| Rejection / demod reached | 0.1714 / 0.8286 | 0.1714 / 0.8286 (unchanged) |

Per BFSK capture, strict BER before -> after:

| Capture | Before | After |
|---|---|---|
| BFSK_h0.5_snr20dB | 0.2437 | **0.0049** |
| BFSK_h0.5_snr15dB | 0.2623 | **0.0049** |
| BFSK_h0.5_snr10dB | 0.2530 | **0.0054** |
| BFSK_h0.5_snr5dB | 0.4909 | 0.4909 |
| BFSK_h0.5_snr0dB | 0.4967 | 0.5077 |
| BFSK_h1.0_* (all five) | ~0.505 | ~0.505 (unchanged, routed to PSK) |

The 20/15/10 dB captures land exactly on the ~0.0049 target Entry 012 predicted from the
offset sweep.

### Non-FSK regression check

**Zero non-FSK captures changed by even one bit** - every BPSK, QPSK, 8PSK, 16QAM and
64QAM capture produced a byte-identical strict BER. Confirmed by direct comparison of
the two results CSVs. Classification, fusion, rejection and demodulation-reached rates
are all unchanged, as expected for a timing-only change.

### Remaining limitation (measured, not fixed here)

h=0.5 at 5 dB and 0 dB is still ~0.49, and the cause is **not** timing:

| Capture | estimated Rs | sps = round(fs/est) | BER |
|---|---|---|---|
| BFSK_h0.5_snr20dB | 25 000 | 8 | 0.0049 |
| BFSK_h0.5_snr10dB | 25 000 | 8 | 0.0054 |
| BFSK_h0.5_snr5dB | **7 574** | **26** | 0.4909 |
| BFSK_h0.5_snr0dB | **23 486** | **9** | 0.5077 |

The symbol-rate estimator fails on FSK below ~10 dB, so the wrong samples-per-symbol is
handed to dispatch and no timing offset can recover it. That is a symbol-rate estimation
issue and was deliberately left untouched.

### Tests

New `tests/test_fsk_timing_dispatch.py`, **21 tests**: CPFSK and GFSK take the FSK path;
BPSK/QPSK/QAM16 keep the linear path; the two criteria demonstrably disagree on CPFSK;
the FSK criterion depends only on signal and sps; selection is deterministic (function
and dispatch level); h=0.5 reaches BER < 0.10 through real dispatch across SNRs and
seeds; the new path beats the old heuristic by more than 4x; h=1.0 still exceeds 0.15
(known-hard guard); BPSK/QPSK still reach exactly 0.0 BER; and degenerate inputs
(sps=1, capture shorter than one symbol) return offset 0.

Full suite: **344 passed** (up from 323), same single pre-existing `reedsolo` failure.

---

## Entry 014 - 2026-09-08 - Low-SNR FSK symbol-rate estimation: no safe small fix (NEGATIVE RESULT)

**Investigation only. No production code was modified.** `parameter_estimation.py`
140 lines, `dispatch.py` 99, `fsk_demod.py` 15 - all unchanged. Full suite unchanged at
**344 passed**, same single pre-existing `reedsolo` failure.

### Question

Entry 013 left FSK h=0.5 failing at 5 dB and 0 dB because the symbol-rate estimator
returned 7 574 Hz and 23 486 Hz instead of 25 000 Hz, producing samples-per-symbol of 26
and 9 instead of 8. Is there a small, safe fix using the features the adaptive estimator
already has?

### MEASURED - each existing feature, BFSK h=0.5, within 1% of 25 000 Hz, 5 seeds

| SNR | `envelope_power` | `transition_power` | `phase_second_difference` | production |
|---|---|---|---|---|
| 20 dB | 0/5 | 0/5 | **5/5** | 5/5 |
| 15 dB | 0/5 | 0/5 | **5/5** | 5/5 |
| 10 dB | 0/5 | 0/5 | **4/5** | 2/5 |
| 5 dB | 0/5 | 0/5 | **0/5** | 0/5 |
| 0 dB | 0/5 | 0/5 | **0/5** | 0/5 |

**At 5 dB and 0 dB no existing feature finds the symbol rate at all.** This rules out the
cheapest candidate fix - changing which feature wins the confidence contest - because
there is no correct candidate to select.

### MEASURED - the spectral line is below the noise floor, not merely mis-ranked

`phase_second_difference` spectrum at the true 25 kHz, seed 3:

| SNR | S(25 kHz) | noise floor | ratio | rank among detected peaks |
|---|---|---|---|---|
| 20 dB | 313.4 | 8.7 | **35.8x** | 3 of 491 |
| 10 dB | 182.7 | 61.7 | 3.0x | 32 of 676 |
| **5 dB** | 140.6 | 273.5 | **0.5x** | 68 of 706 |
| **0 dB** | 1202.3 | 727.0 | 1.7x | 78 of 574 |

At 5 dB the line sits at **half the median noise floor** - it is not a peak that a
better selection rule could recover. At 0 dB it is buried under 77 larger peaks. The
feature applies a second difference to the unwrapped phase, which is a double high-pass:
it amplifies phase noise far faster than it amplifies the timing tone, so the feature's
own SNR collapses well before the signal's does.

### Interpretation

The failure is a genuine loss of information in the existing features at low SNR, not a
selection, threshold or ranking defect. Every fix that could plausibly work would be a
redesign of the estimator - spectral averaging across segments, a matched pre-filter, or
a non-peak-picking estimator - each of which changes behaviour for every modulation and
is explicitly out of scope for this task.

A secondary, separate observation: at **10 dB** `phase_second_difference` is correct 4/5
while production is correct only 2/5, so the confidence-based selection does lose a
correct candidate there. That is a selection issue rather than an information issue, but
fixing it means changing the selection rule for all modulations, which is also out of
scope. Recorded for a future task, not acted on.

### Decision

**No safe small fix exists. V1 is frozen with this known limitation.**

FSK h=0.5 works through the full production path at 20/15/10 dB (BER 0.0049 / 0.0049 /
0.0054 per Entry 013) and fails at 5 dB and 0 dB because the symbol rate cannot be
recovered from the current features at those SNRs.

### V1 final state at freeze (40 captures, `reports/v1_after_fsk_timing/`)

| Metric | Value |
|---|---|
| CNN top-1 accuracy | 0.7714 |
| Fusion accuracy | 0.6857 |
| Demodulation reached | 0.8286 |
| Median BER (strict) | 0.0138 |
| Symbol rate within 1% | 0.9143 |

### Known limitations carried into V2

1. FSK symbol-rate estimation fails below ~10 dB (this entry).
2. FSK h=1.0 is classified as 8PSK and its +/-pi per-symbol advance is ambiguous after
   decimation - labelled `known_hard` in the dataset (Entries 011-012).
3. The checkpoint separates QAM16/QAM64 poorly, 0.52/0.59 on its own RML test set
   (Entry 008); V1 QAM top-1 is 0.80/0.60.
4. Fusion's 0.4 threshold was tuned against single-window confidences and has not been
   recalibrated for the 4-window mean (Entry 010).
5. Decimate-then-discriminate FSK costs ~0.018 BER at high SNR versus 0.0000 at full
   rate (Entry 013).

---

# V2 CARRY-FORWARD PROBLEMS / REQUIRED INVESTIGATIONS

**Status: V1 is FROZEN as of 2026-09-08.**

This section is a **V2 backlog**, not unfinished V1 work. Every item below was
investigated, measured and deliberately left in place during V1, with the evidence
recorded in the entry cited against each one. None of them is a defect introduced by V1;
they are the boundaries of what the frozen V1 baseline can do.

Nothing in this section should be implemented against V1.

## Final V1 baseline (frozen)

Measured on `data/synthetic_v1` (40 captures), reported in
`reports/v1_after_fsk_timing/`:

| Metric | Value |
|---|---|
| Captures | 40 |
| CNN top-1 accuracy | 0.7714 |
| Fusion accuracy | 0.6857 |
| Demodulation reached | 0.8286 |
| Median BER (strict) | 0.0138 |
| Symbol rate within 1% | 0.9143 |

- FSK h=0.5 works through the full production path at **20 / 15 / 10 dB**
  (BER 0.0049 / 0.0049 / 0.0054).
- FSK h=0.5 remains **unreliable at 5 dB and 0 dB**.

## 1. Low-SNR FSK symbol-rate estimation

*Evidence: Entry 014.*

- Applies to FSK h=0.5.
- At 20 / 15 / 10 dB symbol-rate estimation generally works.
- At 5 dB and 0 dB the existing features cannot recover the 25 kHz symbol-rate line.
  Measured: 0/5 seeds for all three features at both SNRs; the line sits at 0.5x the
  noise floor at 5 dB and is ranked 78th of 574 peaks at 0 dB.
- **This is an information-loss problem, not merely a feature-ranking problem.** No
  change to selection, thresholds or ranking can recover a peak that is not present.
- Likely V2 directions: spectral averaging across segments, matched or pre-filtering, or
  a more robust non-peak-based estimator.
- **Do NOT implement now.**

## 2. FSK h=1.0 known-hard case

*Evidence: Entries 011, 012.*

- h=1.0 causes CNN confusion with 8PSK (predicted 8PSK at 0.956 confidence).
- The +/-pi per-symbol phase advance creates ambiguity in the current
  decimate-then-discriminate path, unrecoverable at any timing offset.
- Already labelled `known_hard` with a `known_hard_reason` in the V1 dataset ground
  truth, so it is explicit rather than a silent failure.
- **Retain as a labelled known-hard / OOD-style robustness case for V2.** Do not delete
  it from the dataset to make results look better.

## 3. CNN QAM16 / QAM64 separation

*Evidence: Entries 008, 010.*

- The current checkpoint remains weak at distinguishing QAM16 from QAM64: 0.52 and 0.59
  accuracy on its own RML test set at high SNR, against >= 0.92 for every other class.
- Multi-window inference substantially recovered V1 performance (QAM16 0.00 -> 0.80,
  QAM64 0.00 -> 0.60) but **does not fundamentally improve the checkpoint** - it only
  recovers information that single-window sampling was discarding.
- Candidate for future model training / validation work. Out of scope for V1, which did
  no CNN training by design.

## 4. Fusion confidence calibration

*Evidence: Entry 010.*

- The 0.4 fusion threshold was designed around single-window CNN confidence.
- Multi-window mean-softmax changed the confidence distribution (mean confidence
  0.722 -> 0.644; rejection rate 0.133 -> 0.200).
- Threshold calibration should be revisited with a proper validation / calibration study,
  not tuned to make a metric look better.
- **The threshold was deliberately NOT changed in V1**, so the measured V1 numbers
  reflect the original operating point.

## 5. FSK decimate-then-discriminate BER cost

*Evidence: Entry 013.*

- At h=0.5 and high SNR, full-rate FSK demodulation reaches **0.0000** BER.
- Production decimation to one sample per symbol leaves approximately **0.018** BER even
  after correct timing selection.
- This is a **structural limitation of the current production path**, not a timing or
  discriminator defect: deciding from one sample per symbol discards the averaging that
  the full-rate path gets for free.
- V2 may investigate symbol-wise or full-rate discrimination, or better symbol
  integration.

---

*End of V1. All five items above are V2 backlog.*

---

## Entry 015 - 2026-09-08 - V2.0: train a modulation CNN on our own synthetic distribution

First V2.0 model task. The frozen V1 dataset and its baseline results were **not**
modified and were used only as a held-out regression benchmark. No fusion threshold,
dispatch timing, symbol-rate, demodulator, preprocessing or multi-window change was made.

### Phase 1 - audit of the existing training path (MEASURED / READ)

1. **Architecture**: `ModulationCNN` - three Conv1d blocks (64/128/128, k=8/4/4, BN+ReLU,
   dropout 0.3) -> AdaptiveAvgPool1d(1) -> Linear(128,256) -> dropout 0.5 -> Linear(256,C).
   `input_channels` must be 2 or 4.
2. **Input representation**: `(batch, channels, 128)`. The shipped checkpoint declares
   `input_channels=4`, `sample_length=128`, `features="iqap"` - I, Q, amplitude and
   phase-difference.
3. **Training preprocessing** (`train_modulation.py`): `_canonicalize_frames` only
   *resamples length* - it does **not** normalise - then `add_signal_features_batch`.
4. **Inference preprocessing**: `preprocess` (DC removal + RMS) -> `_fixed_iq`
   (per-window RMS normalise) -> `add_signal_features`.
5. **Are they identical? Not literally.** Training omits the `_fixed_iq` normalisation.
   For `iqap` this is harmless because `add_signal_features` re-normalises every channel
   to unit RMS, making the path scale-invariant; a regression test now asserts the two
   produce equal frames. V2.0 training uses the **inference path verbatim** so the
   question cannot resurface.
6. **Label mapping**: `labels = sorted({...})` in training, stored in the checkpoint, and
   verified equal to `modulation_cnn_metrics.json["classes"]` (Entry 008). No permutation
   bug.
7. **Leakage**: the existing script splits **per frame** with `train_test_split`, with no
   grouping by originating capture - a real leakage risk, flagged since Entry 001.
8. **Generator capacity**: yes. One 1024-symbol capture yields 64 non-overlapping
   128-sample windows, so a few hundred captures give tens of thousands of frames.
9. **RML assumptions**: yes - the loader is RML2016 pickle / RML2018 HDF5 specific, and
   the shipped checkpoint carries 11 RML classes, of which only 6 are V1 classes.

### Phase 2 - dataset design

Built in memory by `training/train_v2_synthetic.py`; **no dataset artefacts written**.

| Property | Value |
|---|---|
| Captures | 840 (1024 symbols each, 8 sps, 200 kHz) |
| Classes | 8PSK, BPSK, CPFSK, QAM16, QAM64, QPSK (project `radiofry_label` convention) |
| Per modulation | 120 captures; CPFSK 240 (swept over h=0.5 and h=1.0) |
| SNR distribution | 20/15/10/5/0 dB, uniform (168 captures per SNR) |
| Frames | 53 760 total: train 32 256 / validation 10 752 / test 10 752 |
| Split | **capture level**, 60/20/20, stratified by (label, SNR, h) |
| Split seed | 20 260 908 |
| Torch/numpy seed | 7 |
| Class balance | equal per modulation; CPFSK deliberately 2x because it spans two indices |

**Anti-leakage.** The split assigns whole captures, so windows of one generated signal
can never straddle it - asserted by tests at both capture-id and seed level. Training
seeds come from a range disjoint from the frozen V1 capture seeds, and `build_dataset`
raises if any collision exists.

**Generalisation set.** A further 210 captures (13 440 frames) from seed base 9 000 000,
never used for training or model selection, kept separate from the primary test split.

### Phase 3 - input representation

Raw IQ, IQ+amplitude and IQ+phase-difference are all supported via
`add_signal_features(include_engineered=...)`. V2.0 uses **iqap (4 channels)**, matching
the production checkpoint contract exactly. Frames are built with the same normalisation
`_fixed_iq` applies, so the trained model drops into the existing inference path with no
architectural change.

### Phase 4 - training

Adam lr 1e-3, ReduceLROnPlateau (patience 2, factor 0.5), CrossEntropyLoss, batch 256,
max 60 epochs, early stopping patience 6, validation-loss model selection. CPU only
(torch 2.13.0+cpu, no CUDA available). Training took 450 s.

**Best epoch 18**, validation loss 0.12028, validation accuracy 0.9443. Early stopped at
epoch 24. Train loss at best epoch 0.1337 against validation 0.1203 - no train/validation
gap, so no overfitting signal.

### Phase 5A - held-out synthetic test set (10 752 frames, 168 captures)

| Metric | Value |
|---|---|
| Top-1 | **0.9382** |
| Top-3 | **0.9992** |
| Validation top-1 | 0.9443 |
| **Independent generalisation set top-1** | **0.9406** |

Per-class test accuracy: BPSK 1.0000, CPFSK 0.9991, QPSK 0.9788, 8PSK 0.9569,
QAM16 0.9313, **QAM64 0.7025**.

Confusion matrix (truth rows):

| | 8PSK | BPSK | CPFSK | QAM16 | QAM64 | QPSK |
|---|---|---|---|---|---|---|
| 8PSK | 1531 | 0 | 4 | 36 | 9 | 20 |
| BPSK | 0 | 1600 | 0 | 0 | 0 | 0 |
| CPFSK | 3 | 0 | 3197 | 0 | 0 | 0 |
| QAM16 | 39 | 0 | 1 | 1490 | 69 | 1 |
| **QAM64** | 26 | 0 | 0 | **450** | 1124 | 0 |
| QPSK | 25 | 3 | 0 | 5 | 1 | 1566 |

Per-SNR: 20 dB 0.9951, 15 dB 0.9938, 10 dB 0.9848, 5 dB 0.9210, 0 dB 0.7964.
Confidence: mean 0.9397, median 0.9999, p10 0.6896, fraction below the 0.4 fusion
threshold 0.0042.

### Phase 5B - frozen V1 benchmark, OLD vs NEW checkpoint

Identical captures (`data/synthetic_v1`, 40 captures) and identical evaluation settings;
only the checkpoint differs.

| Metric | OLD | NEW |
|---|---|---|
| CNN top-1 | 0.7714 | **0.9714** |
| CNN top-3 | 0.8571 | **1.0000** |
| Mean CNN confidence | 0.6932 | 0.9309 |
| Fusion accuracy | 0.6857 | **0.9714** |
| Fusion rejection rate | 0.1714 | **0.0000** |
| Demodulation reached | 0.8286 | **1.0000** |
| End-to-end success | 0.8286 | **1.0000** |
| BER scored | 58 | **70** |
| Median strict BER (all scored) | 0.01383 | 0.07959 |
| **Median strict BER (paired subset, n=58)** | **0.01383** | **0.01383** |

Per-modulation CNN top-1 (OLD -> NEW): BPSK 1.00 -> 1.00, QPSK 1.00 -> 1.00,
8PSK 1.00 -> 1.00, **BFSK 0.50 -> 1.00**, **16QAM 0.80 -> 1.00**, **64QAM 0.60 -> 0.80**.
Fusion accuracy: QPSK 0.80 -> 1.00, 8PSK 0.80 -> 1.00, BFSK 0.50 -> 1.00,
16QAM 0.60 -> 1.00, 64QAM 0.60 -> 0.80.

**The median-BER move is a population effect, not a regression.** Per capture:

| Outcome | Count |
|---|---|
| Improved | 5 |
| **Worse** | **0** |
| Unchanged | 24 |
| Newly demodulated | 6 |
| Lost demodulation | 0 |

On the 58 records both models scored, the median strict BER is **identical**. The full-set
median rises only because 12 further records entered the scored set - six captures the old
model's fusion had rejected, all at 0 or 5 dB, with BER 0.21-0.50. The five improvements
are the BFSK h=1.0 captures, now correctly labelled CPFSK (0.505 -> 0.25).

### Answers to the comparison questions

1. **Better on our synthetic distribution?** Yes - 0.9382 test top-1 on a capture-level
   held-out split.
2. **Better end-to-end on frozen V1?** Yes - CNN 0.77 -> 0.97, fusion 0.69 -> 0.97,
   rejection 0.17 -> 0.00, demodulation reached 0.83 -> 1.00, with zero captures worse.
3. **QAM16/QAM64?** QAM16 0.80 -> 1.00 on V1. QAM64 0.60 -> 0.80 on V1 but only 0.7025 on
   the synthetic test set, where 450 of 1600 QAM64 frames go to QAM16. **QAM64 remains the
   weakest class** and is now the dominant error mode.
4. **BFSK/CPFSK?** Yes, decisively - 0.50 -> 1.00 on V1 and 0.9991 on the test set, at
   **both** modulation indices. h=1.0 is no longer a classification failure. Its
   demodulation is still limited by the +/-pi ambiguity (0.25 BER), which is a DSP issue,
   not a model one.
5. **Regressions?** None measured. Zero captures worse, zero lost demodulation, full
   suite green.
6. **Overfitting?** No evidence. Train 0.1337 vs validation 0.1203 at the best epoch;
   validation 0.9443, test 0.9382, independent generalisation 0.9406.
7. **Learning modulation or artefacts?** The independent seed set (0.9406) matches the
   test split (0.9382), and the frozen V1 benchmark - different seeds, 4x longer captures,
   int16 file round-trip, 4-window inference - reaches 0.9714. That is consistent with
   learning modulation structure. **It does not prove generalisation beyond this
   generator**: everything here is rectangular-pulse AWGN from one synthetic source.

### Limitations

- Trained and tested entirely on one synthetic generator: rectangular pulses, AWGN only,
  one symbol rate, one sps, no CFO/timing/fading/multipath. Real-capture performance is
  **unmeasured**.
- QAM64 at 0.7025 is the clear weak point; 28% of its frames are called QAM16.
- 0 dB accuracy is 0.7964 and drives most residual error.
- The new model covers only the 6 V1 classes. The old checkpoint carried 11 RML classes
  including AM-DSB/AM-SSB/WBFM/PAM4/GFSK; **those five classes are gone**. If any
  downstream behaviour depends on them, this checkpoint is not a drop-in replacement.
- Mean confidence rose 0.69 -> 0.93 and rejection fell to zero, so fusion's 0.4 threshold
  now rejects nothing on V1. That threshold remains uncalibrated (carry-forward item 4)
  and was deliberately left untouched.
- CPU-only training; no hyperparameter search was performed.

### Conclusion

**B - the new model improves most areas but needs further training/data work before it
replaces the shipped checkpoint.** The V1 end-to-end gains are large and free of
regressions, but the reduced class set (6 vs 11) and QAM64 at 0.70 mean this is a strong
V2.0 baseline rather than a finished production checkpoint.

### Files changed

```
src/radiofry/training/train_v2_synthetic.py   added (training pipeline)
src/radiofry/evaluation/harness.py            modified (optional model_path passthrough)
src/radiofry/evaluation/cli.py                modified (--model flag)
tests/test_v2_training_pipeline.py            added, 14 tests
models_saved/modulation_cnn_v2_synth.pt       added (~193 KB checkpoint)
models_saved/modulation_cnn_v2_synth_metrics.json  added (required by the loader)
reports/v2_newmodel/                          added (frozen V1 evaluation with the new model)
```

No DSP, fusion, demodulator, generator or frozen-V1 artefact was modified. Full suite:
**358 passed** (up from 344), same single pre-existing `reedsolo` failure.

---

## Entry 016 - 2026-09-08 - V2.0: add PAM4 and GFSK generator support (6 -> 8 classes)

Scoped follow-up to the 11-class audit. No training run was launched. The frozen V1
dataset is byte-identical (aggregate .iq SHA-256 `d6d3f918687d0700a43e46211c3f04b9`,
40 captures - the same value recorded in Entry 015). `dispatch.py` (99 lines) and
`fsk_demod.py` (15 lines) were **not** touched, and no PAM4 demodulator was added.

### PAM4 - fits the existing constellation abstraction cleanly

No blocker. PAM4 is a memoryless real-axis constellation, so it drops into the existing
`constellation_for` / `modulate` path with one new family branch.

- `MODULATIONS["PAM4"] = ModulationSpec("PAM4", "pam", 4, 2, "PAM4")`.
- `constellation_for` gains a `pam` branch: levels `[-3,-1,1,3]` normalised to unit
  average symbol power, i.e. divided by sqrt(5).
- `modulate()` needed **no change** - the non-FSK branch already routes through
  `constellation_for`.
- Bit mapping is the existing natural-binary MSB-first convention, so the
  `source_bits` / `transmitted_bits` / BER contract is untouched.

**Measured:** the constellation is four real levels at unit average power, index k maps
to ascending level k, and a noiseless capture recovers the **exact** source bits through
an independent nearest-point oracle.

### GFSK - Gaussian frequency pulse on the existing FSK path

- `ModulationSpec` gains `default_pulse_shape` (default `"rect"`).
- `MODULATIONS["GFSK"] = ModulationSpec("GFSK", "fsk", 2, 1, "GFSK",
  default_pulse_shape="gaussian")` - same family as BFSK, differing only in pulse shape.
- `SampleSpec.pulse_shape` becomes `str | None`, resolving to the modulation's default;
  `SUPPORTED_PULSE_SHAPES = ("rect", "gaussian")`. Gaussian is rejected for non-FSK
  families. New `gaussian_bt` field resolves to `DEFAULT_GAUSSIAN_BT = 0.3` and is
  rejected when the pulse is rectangular.
- New `gaussian_frequency_pulse(bt, sps, span_symbols=4)`: sigma = sqrt(ln2)/(2*pi*BT),
  taps normalised to **unit area**.
- `_modulate_cpfsk` convolves the rectangular frequency-pulse train with those taps
  **only** when `pulse_shape == "gaussian"`, so CPFSK generation is bit-for-bit unchanged.

**Measured properties:**

| Property | CPFSK | GFSK | Note |
|---|---|---|---|
| Pulse tap sum | - | **1.000000** | unit area preserves the modulation index |
| Total accumulated phase (rad) | -6.087 | -6.180 | 1.5% apart - finite `same`-convolution edge effect |
| Peak instantaneous freq (Hz) | 6250.0 | **6250.0** | deviation never exceeded, no overshoot |
| Mean absolute freq (Hz) | 6250.0 | 4020.3 | **not** an invariant - smoothing passes through zero |
| Occupied bandwidth | wider | narrower | the defining GFSK property |
| Envelope | constant | constant | |

An initial test asserted mean absolute frequency as the invariant. That was **wrong** -
Gaussian shaping preserves the *area* of the frequency pulse, not the mean of its
magnitude. The test was corrected to assert unit pulse area, preserved accumulated phase
and un-exceeded peak deviation, which is the actual physics.

### Ground truth

`signal.gaussian_bt` added. `modulation.pulse_shape` already existed and now records
`"gaussian"` for GFSK. `radiofry_label` resolves to `GFSK` and `PAM4` directly from the
registry, so the label vocabulary needs no special-casing. `fsk_deviation_hz` and
`fsk_modulation_index` populate for GFSK exactly as for CPFSK.

### Training pipeline - 8 classes, dataset build validated (NOT trained)

`build_capture_specs` iterates `MODULATIONS`, so PAM4 and GFSK were picked up with no
change to the training module itself. Validated build (4 replicates, split seed
20 260 908, V1 seed-collision check enabled):

```
labels (8): ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK']
captures  : train=100  validation=50  test=50
per class : CPFSK/GFSK 20-10-10, all others 10-5-5
frames    : (1024, 4, 128) float32, all finite
```

Labels match the production vocabulary exactly (8 of RadioFry's 11). A full 24-replicate
build would be **1200 captures / ~76 800 frames**, up from 840 / 53 760 at 6 classes.

**Class-balance note for the eventual training run:** CPFSK and GFSK each receive twice
the captures of the other six, because both are FSK and are swept over h = 0.5 and
h = 1.0. This is deliberate coverage, not an accident, but it should be a conscious
decision before training.

### Tests

New `tests/test_synthetic_pam4_gfsk.py`, **30 tests**: registry/label coverage, PAM4
constellation geometry and mapping, PAM4 bit-exact oracle recovery, PAM4 carrying no FSK
parameters, GFSK defaults and configurable BT, constant modulus, unit pulse area,
accumulated-phase preservation, frequency smoothing, narrower occupied bandwidth,
determinism, peak-deviation bound, rejection of gaussian shaping on non-FSK and of
unknown pulse shapes, the existing six modulations keeping rectangular pulses, and - the
strongest guard - **six frozen V1 captures re-derived from their own ground truth and
matched against the on-disk files sample by sample**.

`tests/test_v2_training_pipeline.py` updated for 8 classes (+3 tests: both FSK
modulations swept over their indices, PAM4 label coverage, FSK-aware index assertion).

Full suite: **390 passed** (up from 358), same single pre-existing `reedsolo` failure.

### Files changed

```
src/radiofry/synthetic_gen/v1/config.py       modified (179 -> 204) registry, pulse shape, gaussian_bt
src/radiofry/synthetic_gen/v1/modulation.py   modified (72 -> 97)  pam branch, gaussian_frequency_pulse
src/radiofry/synthetic_gen/v1/generator.py    modified (386 -> 387) gaussian_bt in ground truth
src/radiofry/synthetic_gen/v1/__init__.py     modified (47 -> 54)  export gaussian_frequency_pulse
tests/test_synthetic_pam4_gfsk.py             added, 30 tests
tests/test_v2_training_pipeline.py            modified for 8 classes
```

### Not done, by instruction

No PAM4 demodulator or dispatch route (PAM4 is generatable and classifiable but still
**not demodulatable** - `dispatch` returns "No demodulator is registered for PAM4").
No FSK demodulator or timing change. No analog classes. No training run. V1 untouched.

### Next step

Decide the class-balance question above, then launch the 8-class training run and
benchmark it against both the 6-class checkpoint and the frozen V1 set.

---

## Entry 017 - 2026-09-08 - V2.0: train and evaluate the 8-class modulation CNN

Training and evaluation only. **No production code was changed** in this entry - the
suite is unchanged at 390 passed. The frozen V1 dataset is byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures). `dispatch.py`, `fsk_demod.py`,
fusion, symbol-rate estimation, preprocessing and the V1 generator were not touched.

### Pre-training verification (all five checks passed)

1. **Labels**: exactly `['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK']`.
2. **Class counts**: 1200 captures / 76 800 frames. CPFSK and GFSK carry 240 captures
   each because both sweep h = 0.5 and h = 1.0; the other six carry 120. **The h = 1.0
   hard cases were retained, not dropped.**
3. **Leakage**: capture-id and seed overlap is zero across all three split pairs; every
   capture appears in exactly one split; the generalisation seed range is disjoint; the
   V1 seed-collision check passed.
4. **Ground truth**: BFSK -> `CPFSK` (rect, h=0.5), GFSK -> `GFSK` (gaussian, BT=0.3,
   h=0.5), PAM4 -> `PAM4` (pam family, 2 bits/symbol), 16QAM -> `QAM16`.
5. **Composition**: 1024 symbols @ 8 sps / 200 kHz, SNR sweep 20/15/10/5/0 dB with 240
   captures at each SNR.

### Dataset and split

| Property | Value |
|---|---|
| Captures / frames | 1200 / 76 800 |
| Split (capture level) | train 700 / validation 250 / test 250 |
| Frames | train 44 800 / validation 16 000 / test 16 000 |
| Stratification | (label, SNR, FSK modulation index) |
| Split seed / torch seed | 20 260 908 / 7 |
| Input | 4-channel iqap, 128 samples, inference-identical preprocessing |

### Training configuration - unchanged from Entry 015

Adam lr 1e-3, ReduceLROnPlateau (patience 2, factor 0.5), CrossEntropyLoss, batch 256,
max 60 epochs, early-stopping patience 6, validation-loss selection, CPU.

**Best epoch 25**, validation loss 0.09626, early stopped at epoch 31, 584 s.

### Held-out test results (16 000 frames / 250 captures)

| Metric | 6-class (Entry 015) | **8-class** |
|---|---|---|
| Top-1 | 0.9382 | **0.9583** |
| Top-3 | 0.9992 | **0.9995** |
| Validation top-1 | 0.9443 | 0.9561 |
| Independent generalisation top-1 | 0.9406 | **0.9545** |

Independent generalisation used 300 captures / 19 200 frames from the disjoint seed base
9 000 000, never used for training or model selection.

Per-class test accuracy (6-class value in brackets where comparable):

| Class | Accuracy | vs Entry 015 |
|---|---|---|
| BPSK | 0.9975 | 1.0000 (−0.0025) |
| PAM4 | **0.9950** | new |
| GFSK | **0.9881** | new |
| CPFSK | 0.9816 | 0.9991 (**−0.0175**) |
| QPSK | 0.9775 | 0.9788 (−0.0013) |
| 8PSK | 0.9569 | 0.9569 (unchanged) |
| QAM16 | 0.9119 | 0.9313 (−0.0194) |
| **QAM64** | **0.8050** | 0.7025 (**+0.1025**) |

Confusion matrix (truth rows):

| | 8PSK | BPSK | CPFSK | GFSK | PAM4 | QAM16 | QAM64 | QPSK |
|---|---|---|---|---|---|---|---|---|
| 8PSK | 1531 | 0 | 1 | 5 | 0 | 34 | 10 | 19 |
| BPSK | 0 | 1596 | 0 | 0 | 4 | 0 | 0 | 0 |
| CPFSK | 1 | 0 | 3141 | **58** | 0 | 0 | 0 | 0 |
| GFSK | 1 | 0 | **37** | 3162 | 0 | 0 | 0 | 0 |
| PAM4 | 0 | 7 | 0 | 0 | 1592 | 1 | 0 | 0 |
| QAM16 | 35 | 0 | 0 | 0 | 0 | 1459 | **106** | 0 |
| QAM64 | 24 | 0 | 0 | 2 | 1 | **285** | 1288 | 0 |
| QPSK | 29 | 0 | 0 | 0 | 1 | 2 | 4 | 1564 |

Per-SNR: 20 dB 0.9997, 15 dB 0.9981, 10 dB 0.9947, 5 dB 0.9575, **0 dB 0.8416**.
Confidence: mean 0.9512, median 0.9999, p10 0.8030, fraction below 0.4 = 0.0034.

### Required class-specific analysis

**CPFSK vs GFSK - well separated but not free.** CPFSK->GFSK 1.81%, GFSK->CPFSK 1.16%.
This is the only new confusion pair of consequence and it explains CPFSK's drop from
0.9991 to 0.9816: the model now has a genuinely similar neighbour. Both remain above
0.98, so Gaussian shaping is a learnable discriminator.

**PAM4 - highly learnable.** 0.9950, with only 8 errors in 1600 frames (7 to BPSK, 1 to
QAM16). BPSK confusion is expected - both are real-axis constellations.

**QAM16 vs QAM64 - still the dominant error mode, but improved.** QAM64->QAM16 17.81%
(285 frames) and QAM16->QAM64 6.63%. QAM64 nonetheless **improved** from 0.7025 to
0.8050; adding two classes did not degrade it.

**Low-SNR behaviour by class (the collapse is concentrated, not general):**

| Class | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| BPSK | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 |
| PAM4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 |
| GFSK | 1.000 | 1.000 | 0.998 | 0.988 | 0.955 |
| CPFSK | 1.000 | 1.000 | 1.000 | 0.994 | 0.914 |
| QPSK | 1.000 | 1.000 | 1.000 | 0.997 | 0.891 |
| 8PSK | 1.000 | 1.000 | 1.000 | 0.981 | 0.803 |
| QAM16 | 1.000 | 0.991 | 0.991 | 0.891 | 0.688 |
| **QAM64** | 0.997 | 0.991 | 0.959 | 0.744 | **0.334** |

**QAM64 collapses at 0 dB to 0.334** and is already degraded at 5 dB (0.744). Every other
class stays above 0.80 at 0 dB. This is the single clearest weakness and is reported as
measured, not smoothed over.

### Frozen V1 regression - no change whatsoever

`reports/v2_newmodel/` (6-class) vs `reports/v2_8class/` (8-class), identical captures
and settings:

| Metric | OLD 11-class | 6-class | **8-class** |
|---|---|---|---|
| CNN top-1 | 0.7714 | 0.9714 | **0.9714** |
| CNN top-3 | 0.8571 | 1.0000 | **1.0000** |
| Fusion accuracy | 0.6857 | 0.9714 | **0.9714** |
| Rejection rate | 0.1714 | 0.0000 | **0.0000** |
| Demodulation reached | 0.8286 | 1.0000 | **1.0000** |
| BER scored | 58 | 70 | **70** |
| Median strict BER | 0.01383 | 0.07959 | **0.07959** |
| Mean CNN confidence | 0.6932 | 0.9309 | 0.9273 |

Per-modulation CNN top-1 is identical between the two V2 models: BPSK/QPSK/8PSK/BFSK/
16QAM all 1.00, 64QAM 0.80.

**Per-capture: 0 improved, 0 worse, 35 unchanged, 0 newly demodulated, 0 lost.** Adding
PAM4 and GFSK cost the original six classes **nothing** on the frozen benchmark.

**Caveat:** the frozen V1 set contains no PAM4 or GFSK captures, so this regression test
exercises only the original six classes. The new classes are evidenced solely by the
synthetic held-out and generalisation sets.

### Did any of the original six become worse?

On the **frozen V1 benchmark, no** - every metric and every capture is identical.
On the **synthetic test set**, three of the six moved slightly:
CPFSK −0.0175 (the GFSK neighbour), QAM16 −0.0194, BPSK −0.0025, QPSK −0.0013,
8PSK unchanged, and QAM64 **+0.1025**. Net overall accuracy rose 0.9382 -> 0.9583, so the
small per-class costs are outweighed, but the CPFSK and QAM16 dips are real and caused by
the added classes.

### Limitations

- QAM64 at 0 dB is 0.334 and at 5 dB is 0.744; 285 of 1600 QAM64 test frames are called
  QAM16. This is the dominant residual error.
- CPFSK lost 1.75 points to the new GFSK neighbour.
- 3 of RadioFry's 11 production classes are still missing (AM-DSB, AM-SSB, WBFM); the
  generator cannot produce them because they carry no bits.
- **PAM4 is classifiable but still not demodulatable** - `dispatch` has no PAM4 route, so
  a PAM4 label would be forwarded by fusion and then fail at `demodulate_capture`.
- Everything is one synthetic generator: rectangular/Gaussian pulses, AWGN only, one
  symbol rate, one sps. Real-capture performance remains unmeasured.
- CPFSK/GFSK carry twice the captures of the other six classes (the h sweep). This was
  retained deliberately; no loss weighting was applied.
- Fusion rejection is 0.0000 on V1 with both V2 models, so the uncalibrated 0.4 threshold
  currently rejects nothing - carry-forward item 4 is now more pressing, not less.

### Recommendation

**B - the 8-class model is a strict improvement on the 6-class one and should become the
V2.0 working baseline, but it is still not a drop-in replacement for the shipped
checkpoint.**

It beats the 6-class model on held-out test (0.9583 vs 0.9382), on independent
generalisation (0.9545 vs 0.9406) and on QAM64 (0.8050 vs 0.7025), while being exactly
equal on the frozen V1 benchmark with zero regressions. The blockers to promotion are
unchanged and concrete: three missing production classes, QAM64's low-SNR collapse, and
PAM4 having no demodulation route.

### Artifacts

```
models_saved/modulation_cnn_v2_8class.pt            ~193 KB checkpoint
models_saved/modulation_cnn_v2_8class_metrics.json  metrics required by the loader
reports/v2_8class/                                  frozen V1 evaluation with the 8-class model
```

---

## Entry 018 - 2026-09-08 - Analog ground-truth design investigation (AM-DSB / AM-SSB / WBFM)

**Design investigation only. Nothing was implemented.** No generator, harness,
demodulator, dispatch, fusion, CNN or dataset file was modified; the suite is unchanged
at 390 passed and the frozen V1 dataset is untouched.

### EXISTING CODE FACTS (read from source)

- `decoding/demodulators/analog_demod.py` provides three functions:
  `demodulate_am` (envelope minus mean), `demodulate_ssb` (product detector taking
  `sample_rate` and `carrier_frequency`), `demodulate_fm` (`diff(unwrap(angle))`).
- `dispatch.py` routes `AM-DSB` -> `demodulate_am`, `AM-SSB` -> `demodulate_ssb`,
  `WBFM` -> `demodulate_fm`.
- `fusion/confidence_fusion.py` already maps all three labels to `analog-like`, and
  `dsp/cyclostationary.py` can already emit `analog-like`. **No label or fusion work is
  needed.**
- The V1 ground-truth contract is bit-centric: `generate_source_bits` draws
  `num_symbols * bits_per_symbol` bits, and the record carries a mandatory `bits` block
  plus `bits_per_symbol`, `num_symbols`, `symbol_rate_hz`, `samples_per_symbol`.
- `writers.py` operates on arbitrary complex baseband and needs **no change** for analog.

### MEASURED FACTS (probed, read-only)

Four concrete blockers were verified rather than assumed:

1. **The harness would crash.** `evaluation/metrics.py:expected_family_for("analog")`
   raises `KeyError: 'analog'`. It is called unconditionally in `evaluate_capture`
   *before* the ingestion try/except, so the first analog capture aborts the sweep.
2. **Ground truth would be poisoned.** `generator.py` computes
   `eb_n0_db = es_n0_db - 10*log10(bits_per_symbol)`; with `bits_per_symbol = 0` this
   evaluates to **-inf**.
3. **`dispatch` synthesises fake bits for analog**:
   `DemodulationResult(analog, np.asarray(analog > np.median(analog), dtype=np.uint8), ...)`.
   An analog capture therefore **would** produce a numeric BER, and that number would be
   meaningless - a median threshold of an audio waveform.
4. **`dispatch` decimates analog by the estimated symbol rate.** `symbol_samples =
   signal.iq[timing_offset::samples_per_symbol]` runs before the analog branch, and
   AM-SSB compensates with `sample_rate / samples_per_symbol`. Analog signals have no
   symbol rate, so the analog path currently depends on an estimate with no physical
   meaning for these classes.

Also measured: `score_bits(empty_truth, recovered, available=True)` returns
`status="ok"` with `strict=None` - a silent hole - whereas
`score_bits(..., available=False, reason=...)` correctly returns `status="unavailable"`.
The mechanism to mark BER unavailable already exists and should be used explicitly.

### PROPOSED DESIGN - message signal

Use a **deterministic multi-tone message**, not recorded audio. Rationale: no external
assets, exactly reproducible from a seed, and it yields crisp analytic oracles (known
tone frequencies produce known spectral structure). This matches V1's known-ground-truth
discipline.

Message = sum of K tones with seeded frequencies, amplitudes and phases, peak-normalised
to 1.0, recorded in ground truth as an explicit tone list plus an `.npy` of the sampled
message and its SHA-256.

### PROPOSED DESIGN - per-class generation parameters

| Class | Baseband model | Required parameters |
|---|---|---|
| AM-DSB | `x(t) = 1 + m*s(t)` (real, carrier present so the envelope detector works) | modulation depth `m` (default 0.5) |
| AM-SSB | `x(t) = s(t) + j*hilbert(s(t))` for USB, conjugate for LSB | `sideband` ("upper"/"lower") |
| WBFM | `x(t) = exp(j*2*pi*deviation*cumsum(s)/fs)` | `frequency_deviation_hz`, derived `modulation_index = deviation / max_tone_hz` |

All three share: `sample_rate_hz`, `duration_sec` / `num_samples`, message tone list,
SNR. None of them has a symbol rate, samples-per-symbol, constellation or bit mapping.

### PROPOSED DESIGN - one shared analog schema

All three classes fit **one** schema; no per-class schema is needed.

Reused unchanged: `schema`, `capture_id`, `generated_utc`, `generator`,
`modulation.{name, family, radiofry_label}`, `signal.{sample_rate_hz, num_samples,
duration_sec, center_frequency_hz}`, the whole `noise` block except the two derived
energy ratios, `impairments`, `seeds`, `files`, `known_hard`.

Must become **explicitly null** for analog (not absent, not zero):
`modulation.{order, bits_per_symbol, bit_mapping, constellation_normalization}`,
`signal.{symbol_rate_hz, samples_per_symbol, fsk_deviation_hz, fsk_modulation_index,
gaussian_bt}`, `noise.{es_n0_db, eb_n0_db}`, and the entire `bits` block.

New, analog-only:

```
"message": {"type": "multitone", "tones": [{"frequency_hz", "amplitude", "phase_rad"}],
            "peak_normalised": true, "message_file": "<id>.message.npy",
            "message_sha256": "..."}
"analog":  {"scheme": "am_dsb" | "am_ssb" | "wbfm",
            "modulation_depth": ..., "sideband": ..., "frequency_deviation_hz": ...,
            "modulation_index": ...}
```

A `capture_kind: "digital" | "analog"` discriminator at the top level is proposed so
consumers can branch without inspecting families.

### PROPOSED DESIGN - evaluation metrics

- **BER: explicitly unavailable.** Add an `UNAVAILABLE_METRICS` entry
  (`bit_error_rate_analog`) and pass `available=False,
  reason="analog_no_transmitted_bits"` to `score_bits`. The harness must **not** score
  the median-threshold bits `dispatch` synthesises (measured fact 3); doing so would
  publish a meaningless number.
- **Positive metric: message-recovery correlation.** Correlate the demodulated analog
  output against the known message from ground truth. This is the analog analogue of
  BER and needs no new architecture - it is one extra record field.
  **Caveat to record with it:** `dispatch` decimates by the estimated symbol rate
  (measured fact 4), so the correlation is confounded by a quantity that has no meaning
  for analog. It must be reported as a diagnostic with that caveat, not as a clean score.

### PROPOSED DESIGN - generator-side oracles

Each class gets a deterministic oracle that proves the waveform is the intended
modulation, **independent of RadioFry's demodulators**:

| Class | Oracle |
|---|---|
| AM-DSB | spectrum shows a carrier at 0 Hz and symmetric sidebands at +/- each tone; `abs(x) - mean` correlates > 0.99 with the known message |
| AM-SSB | sideband suppression: energy at the unwanted sideband is >= 30 dB below the wanted one; real part of the product-detected signal correlates > 0.99 with the message |
| WBFM | `diff(unwrap(angle(x))) * fs / 2pi` correlates > 0.99 with `deviation * s(t)`; occupied bandwidth matches Carson's rule `2*(deviation + f_max)` |

### Do the analog classes break the existing digital contract?

**No, provided the bit fields become explicitly nullable rather than being removed or
zero-filled.** Digital captures keep every field they have today, unchanged. The four
measured blockers are all small and localised, and three of them are latent bugs that
would bite the moment any analog capture appeared - they are worth fixing regardless.

### Minimum implementation files (NOT modified)

```
src/radiofry/synthetic_gen/v1/config.py        analog MODULATIONS entries, analog family,
                                               message/analog parameters, nullable digital fields
src/radiofry/synthetic_gen/v1/modulation.py    multitone message + three analog modulators
src/radiofry/synthetic_gen/v1/generator.py     schema: null bits block, analog/message blocks,
                                               guard the eb_n0_db computation
src/radiofry/synthetic_gen/v1/__init__.py      exports
src/radiofry/evaluation/metrics.py             expected_family_for: "analog" -> "analog-like"
src/radiofry/evaluation/harness.py             BER unavailable for analog, message-correlation
                                               field, new UNAVAILABLE_METRICS entry
src/radiofry/evaluation/symbol_rate_experiment.py  pin its modulation default to digital only
src/radiofry/training/train_v2_synthetic.py    11-class coverage, analog captures have no symbols
tests/                                          oracle + schema + harness-compatibility tests
```

### Risks / compatibility concerns

1. **`tuple(MODULATIONS)` is the default in three places** (`generator.DatasetSpec`,
   `training.build_capture_specs`, `evaluation.symbol_rate_experiment.run_experiment`).
   Adding analog entries silently widens all three. The symbol-rate experiment would
   break outright, since it assumes digital signals - its default must be pinned first.
2. **End-to-end analog demodulation will look bad for reasons unrelated to the
   generator** (measured fact 4: symbol-rate decimation). This must be expected and
   documented up front, not "fixed" by touching `dispatch`.
3. `num_symbols` is load-bearing across the generator, harness and training builder.
   Analog needs `num_samples` / `duration_sec` as the primary length parameter.
4. The frozen V1 dataset must not be regenerated. It is on disk with 40 captures and
   stays that way.
5. Class balance: adding three analog classes to an 11-class training run changes the
   mix again; CPFSK/GFSK already carry 2x.

### Recommended implementation sequence

1. **Fix the two latent crash/poison points first** - `expected_family_for` and the
   `eb_n0_db` guard. Tiny, independently testable, and no behaviour change for digital.
2. Pin `symbol_rate_experiment`'s modulation default to the digital set.
3. Add the multi-tone message generator with its own tests.
4. Add AM-DSB + oracle. Then AM-SSB + sideband-suppression oracle. Then WBFM + Carson /
   discriminator oracle. One class per step, each validated before the next.
5. Extend the ground-truth schema (nullable digital fields, `analog`/`message` blocks,
   `capture_kind`).
6. Harness: BER explicitly unavailable, message-correlation diagnostic with its caveat.
7. Only then extend training to 11 classes and re-benchmark against frozen V1.

Steps 1-2 are safe to do immediately and independently of the analog work.

### Open question for the team

Should analog captures be added to the **frozen V1 benchmark** (they cannot - V1 is
frozen), or should a separate **V2 analog benchmark set** be created? The current
recommendation is a separate V2 set, leaving V1 as the untouched digital regression
benchmark.

---

## Entry 019 - 2026-09-09 - Analog prerequisite safety fixes (no analog generation)

The two latent defects measured in Entry 018, plus a guard on the symbol-rate
experiment. **No analog class was implemented.** The registry is still the eight digital
modulations, and the frozen V1 dataset is byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures). No demodulator, `dispatch.py`, fusion,
CNN, PAM4/GFSK or V1 dataset change.

### Task 1 - analog family lookup (FIXED)

`evaluation/metrics.py` (140 -> 149 lines). `_FAMILY_BY_V1_FAMILY` gained two entries:

```
"pam":    "QAM-like"       <- matches fusion's own PAM4 mapping
"analog": "analog-like"    <- matches fusion's AM-DSB/AM-SSB/WBFM mapping
```

The three existing digital mappings are untouched, and an unknown family still raises
`KeyError`.

**Correction to Entry 018:** the crash was reported there as analog-only. It is not.
`expected_family_for("pam")` raised `KeyError` as well, and **PAM4 already exists in the
registry** (Entry 016), so the crash was live today for any V2 dataset containing PAM4 -
not merely hypothetical. Both families are now covered.

No analog BER is invented anywhere; this change only makes the family label resolvable.

### Task 2 - Eb/N0 guard (FIXED)

`synthetic_gen/v1/generator.py` (387 -> 400 lines). The inline expression
`es_n0_db - 10*np.log10(spec.bits_per_symbol)` is replaced by a small helper:

```python
def _eb_n0_db(es_n0_db, bits_per_symbol):
    if es_n0_db is None or bits_per_symbol < 1:
        return None
    return float(es_n0_db - 10 * np.log10(bits_per_symbol))
```

Unavailable is represented explicitly as `None`, never as zero and never as `-inf`.

**Digital behaviour verified unchanged:** the frozen V1 QPSK capture records
`eb_n0_db = 26.0206` on disk, and recomputing it through the guard returns
**26.0206** - identical. Every current modulation (including PAM4 and GFSK) still yields
a finite value.

### Task 3 - symbol-rate experiment pin (NO CHANGE NEEDED)

**Entry 018's risk #1 was partly wrong and is corrected here.**
`evaluation/symbol_rate_experiment.py` does **not** import `MODULATIONS` from the
registry. It defines its own list at line 39:

```python
MODULATIONS = ["BPSK", "QPSK", "8PSK", "BFSK", "16QAM", "64QAM"]
```

which shadows nothing and is already digital-only - verified at runtime: it contains
neither PAM4 nor GFSK even though both are in the registry today. It therefore **cannot**
silently widen when analog classes are added, and no code change was made.

Instead, five guard tests now lock that behaviour: the list equals the six digital names,
excludes PAM4/GFSK, is a strict subset of the registry (so it is provably not
`tuple(MODULATIONS)`), `run_experiment`'s default matches the pinned list, and requesting
an analog modulation raises.

Existing digital experiment results remain reproducible - nothing in the experiment
changed.

### Tests

New `tests/test_analog_safety_guards.py`, **25 tests**: digital family mappings
unchanged, analog and pam resolve, every registry family resolves, the resolved
vocabulary is a subset of fusion's, unknown families still raise; Eb/N0 normal for
digital, `None` at zero bits, `None` when Es/N0 is `None`, never infinite, finite for all
eight current modulations; and the five symbol-rate pin guards.

**One pre-existing test was updated, deliberately.**
`tests/test_v1_harness_metrics.py::test_expected_family_rejects_an_unknown_family` used
`"analog"` as its example of an unknown family - exactly the behaviour Task 1 was asked
to change. It now uses a genuinely unknown value. This is a behaviour change, recorded
rather than hidden.

Full suite: **415 passed** (up from 390), same single pre-existing `reedsolo` failure.

### Confirmations

1. Focused tests for all three tasks: 25 passed.
2. Full suite: 415 passed, 1 known failure.
3. Digital behaviour unchanged: frozen V1 `eb_n0_db` identical; family mappings identical.
4. No analog generation: registry is
   `['16QAM','64QAM','8PSK','BFSK','BPSK','GFSK','PAM4','QPSK']`; none of AM-DSB, AM-SSB,
   WBFM is present.
5. `symbol_rate_experiment.MODULATIONS == ['BPSK','QPSK','8PSK','BFSK','16QAM','64QAM']`.
6. Frozen V1 SHA-256 unchanged.

### Files changed

```
src/radiofry/evaluation/metrics.py          modified (140 -> 149) pam + analog families
src/radiofry/synthetic_gen/v1/generator.py  modified (387 -> 400) _eb_n0_db guard
tests/test_analog_safety_guards.py          added, 25 tests
tests/test_v1_harness_metrics.py            modified (one test, deliberate)
src/radiofry/evaluation/symbol_rate_experiment.py   NOT changed - already pinned
```

The infrastructure is now safe for analog captures. Analog generation itself remains
unimplemented and is the next task when approved.

---

## Entry 020 - 2026-09-09 - AM-DSB analog synthetic generation + independent oracle

First analog scheme implemented, following the Entry 018 design. **AM-SSB and WBFM are
NOT implemented.** No CNN, training, dispatch, fusion, analog demodulator, PAM4/GFSK or
frozen V1 change.

### Implementation - one new file, zero existing production files modified

`src/radiofry/synthetic_gen/v1/analog.py` (240 lines) is the only production file
touched. `config.py`, `modulation.py`, `generator.py`, `analog_demod.py` and
`dispatch.py` are all byte-for-byte unchanged at 204 / 97 / 400 / 30 / 99 lines.

**Key architectural decision: a separate analog registry.** `ANALOG_MODULATIONS` is
deliberately *not* `config.MODULATIONS`. Three call sites default to
`tuple(MODULATIONS)` - the V1 `DatasetSpec`, the V2 training capture builder, and the
symbol-rate experiment - so keeping analog out of that registry means none of them can
silently widen. This resolves Entry 018's risk #1 structurally rather than by patching
three defaults.

Analog captures cannot use `SampleSpec`, whose entire contract is
`num_symbols * bits_per_symbol`. They get `AnalogSampleSpec`, which has **no**
`symbol_rate_hz`, `samples_per_symbol`, `num_symbols` or `bits_seed` field at all -
asserted by test. Everything else is reused unchanged: the existing `add_awgn` channel,
the existing `write_iq_file` / `write_wav_file` writers, and `_file_entry` / `_sha256`
from the digital generator.

### AM-DSB parameter choices

| Parameter | Value | Rationale |
|---|---|---|
| Model | `x(t) = (1 + m*s(t)) * exp(j*2*pi*fc*t)` | complex baseband; carrier retained so envelope detection is valid |
| Modulation depth `m` | 0.5 | Entry 018 default; keeps `1 + m*s(t)` strictly positive (measured min envelope 0.4+) |
| Carrier offset `fc` | 0.0 Hz default, configurable | matches the V1 baseband convention; tests cover 0 and 12 kHz so the oracle genuinely exercises carrier placement |
| Message | 3 tones, 300-3000 Hz | deterministic multi-tone per Entry 018, no audio assets |
| Message normalisation | peak-normalised to 1.0, zero mean | bounds the envelope and makes correlation well-conditioned |
| Sample rate / length | 200 kHz / 32 768 samples (0.164 s) | matches the digital V1 convention |
| Noise | existing `add_awgn`, same total-band SNR definition | consistent with every digital capture |

With `fc = 0` the waveform is real-valued, which is the correct baseband DSB form.

### Ground truth

Adds `capture_kind: "analog"`, an `analog` block (`scheme`, `carrier_offset_hz`,
`modulation_depth`) and a `message` block (`type`, `num_tones`, `tone_band_hz`, the
explicit tone list with frequency/amplitude/phase, `peak_normalised`, `message_file`,
`message_sha256`). The message is written as an `.npy` beside the capture so validation
can be done independently of the generator.

**Every bit-derived field is explicitly null, never zero and never fabricated:**
`bits` (the whole block), `modulation.{order, bits_per_symbol, bit_mapping,
constellation_normalization, pulse_shape}`, `signal.{symbol_rate_hz, samples_per_symbol,
fsk_deviation_hz, fsk_modulation_index, gaussian_bt}`, `noise.{es_n0_db, eb_n0_db}`.

### The oracle - independent, not a re-run of the generator

The oracle re-derives expected structure from the recorded tone list and recovers the
message with its own detector, rather than calling generator code.

| Oracle check | Result |
|---|---|
| Carrier present at the expected frequency | PASS at fc = 0 Hz and fc = 12 kHz (spectral peak within 40 Hz) |
| Symmetric sidebands at fc +/- each tone | PASS - both sidebands > 20x the noise floor, and upper/lower match within 25% |
| Sideband amplitude tracks modulation depth | PASS - the m=0.8 sideband/carrier ratio exceeds m=0.2 by > 3x |
| Envelope recovery on a clean capture | **correlation > 0.99** with the known message |
| Envelope recovery with a carrier offset | > 0.99 at both fc = 0 and fc = 12 kHz |
| Recovery across the SNR range | > 0.95 at 30 dB, > 0.90 at 20 dB, > 0.60 at 10 dB |
| Message spectrum shows exactly the declared tones | PASS - every declared tone > 20x floor |

**This validates the generator, not RadioFry.** No claim is made about production
demodulation: `dispatch` still decimates analog by an estimated symbol rate and
synthesises median-threshold bits (Entry 018 measured facts 3 and 4), and none of that
was touched here.

### Tests

New `tests/test_synthetic_am_dsb.py`, **31 tests**: registry separation and the explicit
absence of AM-SSB/WBFM, scheme validation, deterministic message generation, seed
sensitivity, peak normalisation, tone-band compliance, message spectrum, deterministic
AM-DSB generation, envelope positivity, shape/dtype, the seven oracle checks above,
ground-truth contract, message block with SHA-256 verification, null bit-derived fields,
absence of any symbol-rate field, capture file writing, byte-level reproducibility, and
readability through the existing `load_capture`.

- Focused: **31 passed**
- Full suite: **446 passed** (up from 415), same single pre-existing `reedsolo` failure

### Digital / V1 preservation - verified

| Check | Result |
|---|---|
| Frozen V1 aggregate .iq SHA-256 | `d6d3f918687d0700a43e46211c3f04b9` (40 captures) - **unchanged** |
| Digital registry | `['16QAM','64QAM','8PSK','BFSK','BPSK','GFSK','PAM4','QPSK']` - unchanged, no analog |
| `symbol_rate_experiment.MODULATIONS` | still the six digital names |
| Existing digital generator files | byte-for-byte unchanged |

### Artifacts

**None written to the repository.** All tests generate captures into pytest `tmp_path`.
`data/` still holds only `synthetic_v1` (12 MB) and `synthetic_v1_shortframe`. No analog
dataset was generated.

### Known limitations

- Only AM-DSB. AM-SSB and WBFM remain unimplemented by instruction.
- Analog captures are not wired into any dataset builder, manifest, harness path or
  training set - this is generation plus oracle only. The harness has never seen an
  analog capture, so the Entry 019 `analog -> analog-like` family fix is still untested
  end-to-end.
- `analog.py` is not exported from the package `__init__.py`; it is imported by its own
  path. That was deliberate, to keep the change to exactly one new file.
- Entry 018's measured facts 3 and 4 still stand: production analog demodulation would
  decimate by a meaningless symbol rate and publish a fake BER. Not addressed here.
- The oracle validates the generator only; it says nothing about RadioFry's ability to
  demodulate AM-DSB.
- `research_memory/CURRENT.md` is stale - it still lists the 8-class training run as
  pending, which Entry 017 completed.

### Recommended next step

Wire a small AM-DSB capture set through the evaluation harness to exercise the Entry 019
family fix end-to-end and confirm the harness marks BER unavailable rather than scoring
dispatch's synthesised bits. That is a harness-side task and is the smallest way to
validate the remaining half of the Entry 018 design before implementing AM-SSB.

---

## Entry 021 - 2026-09-09 - AM-DSB through the evaluation harness: analog safety validated

End-to-end safety validation only. **No production analog demodulation was implemented**,
and no claim is made about analog demodulation quality. AM-SSB and WBFM remain
unimplemented. Frozen V1 is byte-identical (`d6d3f918687d0700a43e46211c3f04b9`,
40 captures).

### MEASURED - the actual harness path, which contradicts Entry 018

Entry 018 predicted the first analog blocker would be
`evaluation/metrics.py:expected_family_for("analog")`. **It is not.** Traced with a real
AM-DSB capture:

`evaluate_capture` calls `load_ground_truth` on **line 111**, which does
`np.load(base / metadata["bits"]["source_bits_file"])`. An analog record has
`bits: null`, so this raises **`TypeError: 'NoneType' object is not subscriptable`** and
aborts *before* `expected_family_for` on line 123 is ever reached.

The Entry 019 family fix is correct and necessary, but it was not the first thing an
analog capture would hit. This is exactly why the path was traced rather than assumed.

### Production changes - two, both required to prevent a crash or a fake metric

**1. `synthetic_gen/v1/generator.py` (400 -> 405 lines), `load_ground_truth`.**
Returns `source_bits`/`transmitted_bits` as `None` when the `bits` block is null,
instead of subscripting `None`. Chosen over returning an empty array so callers must
decide explicitly rather than silently scoring zero bits. Digital captures are
unaffected - `QPSK_snr20dB_r000` still loads an 8192-element array.

**2. `evaluation/harness.py` (327 -> 358 lines).** A `has_source_bits` flag threaded
through `evaluate_capture` and `_score_report`:
- `expected_bits` is `None` rather than `0` for a bit-less capture.
- `_score_report(..., has_bits=False)` forces `score_bits(available=False,
  reason="analog_no_transmitted_bits")`, so recovered bits are never compared.
- `bit_count_ratio` is `None` rather than a ratio against a zero denominator.
- New `UNAVAILABLE_METRICS["bit_error_rate_analog"]` documents why.

No dispatch, fusion, demodulator, generator-waveform, CNN or V1 change.

### MEASURED - observed harness record for an AM-DSB capture

| Field | Value |
|---|---|
| `ingestion_ok` / `pipeline_ok` | True / True |
| `expected_family` | **`analog-like`** |
| `expected_cnn_label` | `AM-DSB` |
| `classical_family` | `QAM-like` |
| `cnn_label` / `cnn_confidence` | `PAM4` / 0.462 |
| `fusion_label` | `PAM4` |
| `demod_available` / `demod_scheme` | **False** / `None` |
| `ber_status` / `ber_reason` | **`unavailable`** / **`analog_no_transmitted_bits`** |
| `ber_strict` / `ber_aligned` | **None** / **None** |
| `expected_bits` / `compared_bits` | **None** / **0** |
| `es_n0_db` / `bits_per_symbol` | **None** / **None** |
| `truth_symbol_rate_hz` | **None** |
| `est_symbol_rate_hz` | 1806.6 |

All seven required validations hold: analog family resolved, no crash, BER explicitly
unavailable, no threshold-derived BER, Eb/N0 and Es/N0 null, no symbol-rate requirement,
ground truth unmutated (byte-compared before and after).

### An honest caveat about what this run did *not* exercise

The shipped 11-class checkpoint labelled this AM-DSB capture **`PAM4`** at 0.462
confidence, and `dispatch` has **no PAM4 route**, so demodulation never ran and the
analog fake-bit branch was never reached in the end-to-end record above. The guard
therefore passed without being exercised by that path.

Rather than claim coverage it did not have, two further tests were added:
- `test_dispatch_really_does_synthesise_analog_bits` confirms Entry 018 measured fact 3
  still holds - `demodulate_capture(..., "AM-DSB", ...)` returns a non-empty bit array
  built by thresholding the waveform at its median.
- `test_the_guard_refuses_those_bits_even_when_they_are_present` calls `_score_report`
  directly with `has_bits=False` and non-empty recovered bits, proving the guard returns
  `unavailable` regardless.

Two incidental observations, recorded but **not** acted on: parameter estimation still
emits a meaningless `est_symbol_rate_hz` of 1806.6 Hz for an analog capture (harmless -
there is no truth value to compare against, so no error is computed), and the classical
detector called this capture `QAM-like` rather than `analog-like`.

### Tests

New `tests/test_analog_harness_safety.py`, **15 tests**: null-bits ground-truth loading,
digital loading unchanged, no crash on IQ and WAV, analog family resolution, capture
identity, BER unavailable, no fake BER, dispatch genuinely synthesising analog bits, the
guard refusing them directly, the metric-registry entry, bit-derived fields staying null,
no symbol-rate requirement, ground truth not mutated, and a digital capture still scoring
a real BER of 0.0.

- Focused: **15 passed**
- Full suite: **461 passed** (up from 446), same single pre-existing `reedsolo` failure

### Digital regression - verified

| Check | Result |
|---|---|
| Frozen V1 aggregate .iq SHA-256 | `d6d3f918687d0700a43e46211c3f04b9` (40 captures) - unchanged |
| Digital `MODULATIONS` | `['16QAM','64QAM','8PSK','BFSK','BPSK','GFSK','PAM4','QPSK']` - no analog |
| Digital family mappings | psk->PSK-like, qam->QAM-like, fsk->FSK-like, pam->QAM-like - unchanged |
| Digital BER behaviour | `QPSK_snr20dB_r000` still scores `ber_status="ok"`, `ber_strict=0.0` |
| `symbol_rate_experiment` | still the six digital names |
| Analog artifacts in repo | none - all tests use pytest `tmp_path` |

### Remaining analog blockers

1. The classical detector calls AM-DSB `QAM-like`, and the shipped checkpoint calls it
   `PAM4`. Neither routes to an analog demodulator, so analog captures currently never
   reach `demodulate_am` through the production path.
2. Entry 018 measured fact 4 is untouched: `dispatch` still decimates by an estimated
   symbol rate before the analog branch.
3. Analog is still not wired into any dataset builder, manifest or training set.
4. `AM-SSB` and `WBFM` remain unimplemented.

### Known documentation issue

`research_memory/CURRENT.md` is still stale - it describes the Entry 017 8-class training
run as pending. Reported only; no memory-system cleanup was mixed into this task.

### Recommended next step

Implement AM-SSB generation plus its sideband-suppression oracle, following the same
pattern as Entry 020. The harness safety path is now proven, so subsequent analog schemes
inherit it without further harness work.

---

## Entry 022 - 2026-09-09 - AM-SSB synthetic generation and independent sideband oracle

Signal generation only. **No production AM-SSB demodulation was implemented**, and the
oracle below is evidence about the generator, not about RadioFry's demodulation quality.
WBFM remains unimplemented. Frozen V1 is byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

### Sideband convention - decided and documented

**Upper sideband (USB) is the default.** `DEFAULT_SIDEBAND = "upper"`; LSB is available
via `AnalogSampleSpec(scheme="am_ssb", sideband="lower")`. The choice is recorded in
ground truth as `analog.sideband`, so a capture is never ambiguous.

### Production change - one file

`src/radiofry/synthetic_gen/v1/analog.py` (240 -> 293 lines):

- `ANALOG_MODULATIONS` gains `"AM-SSB"` -> `ModulationSpec("AM-SSB", "analog", 0, 0, "AM-SSB")`.
  Order and bits_per_symbol stay 0/null: an analog capture has neither.
- `SUPPORTED_ANALOG_SCHEMES = ("am_dsb", "am_ssb")`, `SCHEME_TO_MODULATION`,
  `SUPPORTED_SIDEBANDS = ("upper", "lower")`, `DEFAULT_SIDEBAND = "upper"`.
- `AnalogSampleSpec.sideband: str | None = None` plus `_resolve_sideband()`, which
  rejects a sideband on a non-SSB scheme and fills in the USB default for SSB. This keeps
  the "each parameter is null for the scheme it does not apply to" rule from Entry 020.
- `modulate_am_ssb()` - analytic signal via `scipy.signal.hilbert`, conjugated for LSB,
  then shifted to the carrier offset. Suppressed carrier by construction: no `1 +` term.
- `modulate_analog()` dispatcher; `modulation_name` / `modulation_spec` properties.
- Ground truth generalised: `modulation.name` from the scheme, and the `analog` block now
  carries `sideband` (SSB only), `modulation_depth` (DSB only) and
  `carrier: "present" | "suppressed"`.

The digital registry `config.MODULATIONS` was **not** touched, so the three call sites
that default to `tuple(MODULATIONS)` still see eight digital classes only.

### MEASURED - the independent oracle

The oracle derives the expected spectrum analytically from the ground-truth tone list and
computes an FFT of the emitted waveform. It never calls `modulate_am_ssb`.

| Quantity (USB, noiseless) | Measured |
|---|---|
| Suppression of the image of the 304.0 Hz tone | **140.1 dB** |
| Suppression of the image of the 2592.3 Hz tone | **171.1 dB** |
| Suppression of the image of the 2904.3 Hz tone | **175.7 dB** |
| Total energy above carrier vs below carrier | **72.6 dB** |
| Requirement | >= 30 dB - met with large margin |

Each wanted tone is present at its predicted bin; the carrier bin is suppressed; the
occupied bandwidth is measurably narrower than AM-DSB for the same message. An
independent product detector (down-convert, take the real part - no generator code)
recovers the message with correlation **> 0.99** for both USB and LSB. Adding AWGN
degrades that correlation monotonically, as it must.

### Tests

New `tests/test_synthetic_am_ssb.py`, **28 tests**: registry membership, WBFM absence,
digital registry unchanged, sideband validation and defaulting, deterministic generation,
USB != LSB, the five oracle assertions above, independent recovery for both sidebands,
SNR degradation, ground-truth contract, message block + SHA-256, null bit-derived fields,
reproducibility, harness pass-through, and an AM-DSB regression.

**Three Entry 020 tests were deliberately updated**, because they asserted the previous
state of the world rather than a physical property:
`test_analog_registry_contains_only_am_dsb_for_now` (now
`..._holds_the_implemented_analog_schemes`, expecting `["AM-DSB", "AM-SSB"]`),
`test_am_ssb_and_wbfm_are_not_implemented` (now `test_wbfm_is_still_not_implemented`),
and `test_unsupported_analog_scheme_is_rejected` (now uses `"wbfm"` as its unsupported
scheme). The `test_synthetic_am_dsb.py` module docstring was corrected for the same
reason. No assertion was loosened; no failing test was deleted.

- Focused (both analog generation files): **59 passed**
- Full suite: **489 passed, 1 failed** - the same pre-existing `reedsolo` gap

### Regression - verified

| Check | Result |
|---|---|
| Frozen V1 aggregate .iq SHA-256 | `d6d3f918687d0700a43e46211c3f04b9` (40 captures) - unchanged |
| Digital `MODULATIONS` | `['16QAM','64QAM','8PSK','BFSK','BPSK','GFSK','PAM4','QPSK']` - 8, no analog |
| Analog registry | `['AM-DSB','AM-SSB']` only |
| Supported schemes | `('am_dsb','am_ssb')` |
| `symbol_rate_experiment.MODULATIONS` | six digital names - unchanged |
| WBFM implemented | **False** |
| Analog artifacts in repo | none - tests use pytest `tmp_path` |

### Scope boundaries observed

`dispatch.py`, fusion thresholds, CNN architecture, checkpoints, training, PAM4/GFSK,
the analog demodulators and `evaluation/harness.py` were **not** modified. No Git
operation was performed.

### What this entry does NOT establish

AM-SSB captures still do not reach `demodulate_ssb` through the production path: the
classical detector and the shipped checkpoint both label analog captures as digital
classes (Entry 021), and `dispatch` still decimates by a meaningless estimated symbol
rate. The >0.99 recovery figure comes from the test-local oracle detector, not from
RadioFry.

### Known documentation issue

`research_memory/CURRENT.md` remains stale - it still lists the Entry 017 8-class
training run as pending. Reported only; not modified, per task scope.

---

## Entry 023 - 2026-09-09 - Post-AM-SSB documentation and repository audit (no behaviour change)

Housekeeping only. **No DSP, ML, generator or harness behaviour was changed.** Entries
021 and 022 were audited against the actual repository; the memory system was brought
back in line with reality.

### Entries 021 and 022 - audited against the code

Every file named in both entries exists at the stated size:
`synthetic_gen/v1/analog.py` 293 lines, `evaluation/harness.py` 358 lines,
`tests/test_synthetic_am_dsb.py`, `tests/test_synthetic_am_ssb.py`,
`tests/test_analog_harness_safety.py`. The Entry 021 guard claims are present in the
source (`has_source_bits`, `analog_no_transmitted_bits`, `UNAVAILABLE_METRICS
["bit_error_rate_analog"]`). Re-measured test counts match what the entries recorded:
**28** AM-SSB, **31** AM-DSB (59 together, as Entry 022 states), **15** analog harness
safety. Registry state matches: digital 8 classes, analog `['AM-DSB','AM-SSB']`,
`symbol_rate_experiment` still six digital names, WBFM absent. Nothing was fabricated,
and no historical result was rewritten.

### CORRECTION - the frozen V1 "SHA-256" is a truncated digest

Entries 016, 017, 019, 020, 021 and 022 all quote
`d6d3f918687d0700a43e46211c3f04b9` as the frozen V1 "aggregate .iq SHA-256". That is 32
hex characters - MD5 length - which reads as a mislabel. It is not.

Reproduced this audit: SHA-256 over the concatenated, name-sorted
`data/synthetic_v1/captures/*.iq` bytes (40 files) is
`d6d3f918687d0700a43e46211c3f04b9ef74be9d8b232eac5d0e4cf4bf2390ab`. The recorded value
is the **first 32 characters** of that digest. (MD5 over the same bytes is
`ff7b703a31b80fb3d1f9c0a320938e79` - a different value, so the short form was never an
MD5.)

The historical entries are therefore **correct but under-specified**, and are left as
written. The full digest and the truncation rule are now recorded in
`research_memory/CURRENT.md` so future checks cannot go wrong.

### CORRECTION - what "489 passed" excludes

Entry 022 reports a full suite of "489 passed, 1 failed". That run used
`--ignore=tests/test_model_report.py`. Without the ignore, pytest **aborts during
collection**: `test_model_report.py` imports `h5py`, which is not installed, and a
collection error interrupts the whole session. So there are two environmental gaps, not
one:

- `test_decoding_correlation.py::test_reed_solomon_round_trip` - `reedsolo` missing (a
  failure, already recorded).
- `tests/test_model_report.py` - `h5py` missing (a **collection error**, not previously
  recorded).

Both pre-date this work. Neither is a regression.

### Memory system - brought up to date

- `research_memory/CURRENT.md` **rewritten.** It had been stale since Entry 016: it still
  described the 8-class training run as pending, listed class balance as the blocker, and
  said nothing about analog. It now states V1/V2/PAM4-GFSK/AM-DSB/AM-SSB status, the six
  live blockers, the recommended direction, and the standing constraints. Kept short and
  operational; history stays in BANK.md.
- `research_memory/INDEX.md` **extended.** It indexed Entries 001-016 only. Added Entry
  017 and a new **Analog** section covering Entries 018-022, plus this entry. Also records
  the two facts a reader most needs up front: WBFM is unimplemented, and analog captures
  do not reach an analog demodulator.

### `ANTIGRAVITY_COMMIT.txt` - was materially wrong, rewritten

The handoff file described a commit that has since happened. Measured against the working
tree:

| Claim in the file | Reality |
|---|---|
| `Branch: feature/synthetic-dataset-v1` | current branch is **`main`**; that branch was merged in `0d0d9d7` |
| ~30 files listed as "commit these" | all but a handful are **already committed** |
| "these paths are NOT matched by the current .gitignore" | `.gitignore` now has `/data/`, `/reports/`, `/models_saved/` - they **are** matched |
| checkpoints listed as commit candidates | `models_saved/` is gitignored |
| "Tests: 489 passed" | true only with `--ignore=tests/test_model_report.py` |

It was rewritten to describe only the actual uncommitted delta (5 modified, 6 untracked)
and no longer implies anything has been committed or pushed. It is still useful, so it was
kept rather than deleted.

### MEASURED - repository state (read-only `git status`)

Modified, tracked: `ANTIGRAVITY_COMMIT.txt`, `BANK.md`, `OP.md`,
`src/radiofry/evaluation/harness.py`, `src/radiofry/synthetic_gen/v1/generator.py`.
Untracked: `src/radiofry/synthetic_gen/v1/analog.py`,
`tests/test_analog_harness_safety.py`, `tests/test_synthetic_am_dsb.py`,
`tests/test_synthetic_am_ssb.py`, plus `scratch.py` and `commit_msg.txt`.

`data/`, `reports/` and `models_saved/` do not appear at all - `.gitignore` covers them.
No generated artifact is staged or modified. `scratch.py` (a one-off script that parsed
`ANTIGRAVITY_COMMIT.txt` to build a file list) and `commit_msg.txt` (a copy of the
already-committed V1 message) are both spent; they were **left in place**, flagged rather
than deleted, since deletion is the user's call.

### Tests - unchanged by this entry

- Analog focused: **74 passed** (28 AM-SSB + 31 AM-DSB + 15 harness safety)
- Full suite: **489 passed, 1 failed** (`reedsolo`), with `test_model_report.py` ignored
  (`h5py`)

Identical to the Entry 022 numbers, as expected for a documentation-only change.

---

## Entry 024 - 2026-09-09 - WBFM synthetic generation and independent FM oracle

Signal generation only. **No production WBFM demodulation, classification, fusion or
routing was implemented or modified.** Frozen V1 is byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures). The analog registry is now complete:
AM-DSB, AM-SSB, WBFM.

### FM formulation - phase integral, not an FM-looking approximation

The message `s(t)` is the existing deterministic multi-tone signal, peak-normalised to 1,
so a configured peak deviation `df` is realised exactly:

    f_i(t) = f_c + df * s(t)                     |f_i - f_c| <= df
    phi(t) = 2*pi * integral_0^t f_i(u) du
    x(t)   = exp(j * phi(t))                     constant unit envelope

Sampled at Fs the integral is a cumulative sum with step 1/Fs:

    phi[n] = (2*pi / Fs) * sum_{k<=n} (f_c + df * s[k])

so `angle(x[n+1] * conj(x[n])) * Fs / (2*pi)` returns `f_i[n+1]` exactly. Amplitude is
deliberately not modulated - FM carries information in phase alone.

### Production change - one file

`src/radiofry/synthetic_gen/v1/analog.py` (293 -> 367 lines):

- `ANALOG_MODULATIONS["WBFM"]`, `SUPPORTED_ANALOG_SCHEMES = ("am_dsb","am_ssb","wbfm")`,
  `DEFAULT_FREQUENCY_DEVIATION_HZ = 15_000.0`.
- `AnalogSampleSpec.frequency_deviation_hz` + `_resolve_frequency_deviation()`, which
  rejects the field on the AM schemes, rejects a non-positive value, and rejects a
  configuration whose `|f_c| + df` would reach Nyquist and alias.
- `modulate_wbfm()` and a `modulate_analog()` route.
- Ground truth `analog` block gains `frequency_deviation_hz`, `modulation_index`,
  `carson_bandwidth_hz`; `carrier` is now `"suppressed"` for SSB and `"present"`
  otherwise (WBFM transmits a carrier, it just does not modulate its amplitude).

Digital `config.MODULATIONS` untouched - still the eight digital classes.

### Ground truth - analog semantics preserved

`modulation.name = "WBFM"`, `family = "analog"`. `frequency_deviation_hz` and
`modulation_index` live in the **analog** block, deliberately **not** in the existing
`signal.fsk_deviation_hz` / `signal.fsk_modulation_index`, which stay null: those are
digital-FSK descriptors and reusing them would be exactly the fake-digital-semantics trap
this task forbids. `bits`, `symbol_rate_hz`, `samples_per_symbol`, `bits_per_symbol`,
`order`, `es_n0_db`, `eb_n0_db` all remain null; FEC and interleaving stay `"none"`.
`modulation_depth` (DSB) and `sideband` (SSB) are null for WBFM.

`modulation_index` is the single-tone definition applied to the widest component,
`beta = df / f_max`; it is recorded, not asserted to be the only valid definition for a
multi-tone message.

### MEASURED - the independent oracle

The oracle recovers instantaneous frequency from the emitted samples' phase progression
and rebuilds the expected message from the recorded **tone list**. It never calls
`modulate_wbfm` and never reuses the generator's message array.

**Instantaneous frequency / deviation** (Fs 200 kHz, f_c 20 kHz, df 15 kHz, N 32768):

| Quantity | Measured |
|---|---|
| Mean instantaneous frequency | **19999.954 Hz** (carrier 20000.0) |
| Peak `abs(f_i - f_c)` | **15000.000 Hz** (configured 15000.0) |
| f_i range | 5000.0 - 34974.4 Hz |
| RMS deviation | 6151.3 Hz |
| Envelope | mean **1.00000000**, std 4.2e-08 |

Deviation also tracks a changed configuration: `df = 5 kHz` measures 5000.0 Hz.

**Message recovery** (noiseless, discriminator `s_hat = (f_i - f_c)/df`):
correlation **1.000000**, NRMSE **8.7e-08** - i.e. at the complex64 storage limit.

**Recovered tones:**

| Expected | Recovered | Error |
|---|---|---|
| 499.95 Hz | 500.50 Hz | 0.55 Hz |
| 1359.21 Hz | 1361.13 Hz | 1.92 Hz |
| 2472.25 Hz | 2472.00 Hz | 0.25 Hz |

**Spectrum.** beta = 15000/2472.2 = **6.07** (genuinely wideband). Carson's rule is a
**theoretical approximation**, `2*(df + f_max) = 34944.5 Hz`; the **measured** 99%
occupied bandwidth is **36926.3 Hz**, a ratio of **1.057**. The test asserts only the
same order of magnitude (0.5x - 2x), not equality, because a finite deterministic
multi-tone signal has no reason to match Carson exactly. For the same message, AM-DSB
occupies 4950.0 Hz and AM-SSB 2063.0 Hz - WBFM is ~7.5x wider than DSB, as expected.

**SNR sweep** - monotone degradation, no thresholds tuned to pass:

| Target SNR | Realized | Recovery correlation |
|---|---|---|
| 40 dB | 40.02 | **0.998672** |
| 30 dB | 30.02 | **0.986952** |
| 20 dB | 20.02 | **0.888326** |
| 10 dB | 10.02 | **0.511968** |
| 5 dB | 5.02 | **0.282209** |
| 0 dB | 0.02 | **0.116717** |
| -5 dB | -4.98 | **0.048065** |

Honest caveat: the oracle is a **bare** discriminator with no limiter and no
post-detection filtering, so it falls off faster than a real FM receiver would. The
sharp drop below ~20 dB is the FM threshold effect plus the absence of any de-emphasis or
audio-band filter - a property of the oracle, not a defect in the generated waveform.

### PIPELINE OBSERVATION - recorded as evidence, deliberately not patched

A WBFM capture (8192 samples, 20 dB) through the **unmodified** harness:

| Field | Value |
|---|---|
| `expected_family` / `expected_cnn_label` | `analog-like` / `WBFM` |
| `classical_family` | **`analog-like`** - correct |
| `cnn_label` / `cnn_confidence` | **`BPSK`** / 0.407 |
| `fusion_label` | `BPSK` |
| `demod_available` / `demod_scheme` | **True** / **`2PSK`** |
| `ber_status` / `ber_reason` | `unavailable` / `analog_no_transmitted_bits` |
| `ber_strict` / `expected_bits` / `compared_bits` | None / None / 0 |
| `est_symbol_rate_hz` | 1367.2 (meaningless for analog) |

Two things worth recording:

1. **The classical detector gets WBFM right** (`analog-like`), unlike AM-DSB, which it
   called `QAM-like` in Entry 021. Constant envelope evidently helps it. The CNN still
   does not - it has no analog class - and fusion follows the CNN.
2. **This is the first analog capture that actually reaches a demodulator.** Because
   fusion said `BPSK`, dispatch ran the 2PSK demodulator and returned bits. The Entry 021
   guard therefore held **end-to-end** for the first time: despite real recovered bits
   being present, `ber_status` is `unavailable` and `compared_bits` is 0. Entry 021 could
   only exercise this by calling `_score_report` directly, because AM-DSB was labelled
   PAM4 and had no dispatch route.

No classifier, fusion threshold, dispatch route or demodulator was changed to improve any
of this. It belongs to a later integration task.

### Tests

New `tests/test_synthetic_wbfm.py`, **34 tests**: registry, digital-registry isolation,
deviation defaulting/validation/aliasing, sideband rejection, constant envelope,
determinism, measured peak deviation (two configurations), mean instantaneous frequency,
non-constant frequency, independent message recovery, recovered tones, monotone SNR
degradation, high-SNR usability, Carson-order bandwidth, wider-than-AM-DSB, ground-truth
contract, FM parameters, AM-only fields null, digital fields null, message + SHA-256,
byte reproducibility, IQ write/read integrity (deviation preserved through the int16
round trip within 5%), WAV output, AM-DSB waveform regression, AM-SSB waveform
regression, AM ground truth still nulling the FM fields, and the frozen V1 hash.

**Five prior tests were deliberately updated** - all of them asserted "WBFM is not
implemented", a statement about world state rather than a physical property:
three in `test_synthetic_am_dsb.py` (registry contents; the WBFM-absence test, replaced
with a constant-envelope distinction; and the unsupported-scheme test, which used
`"wbfm"` as its negative case and now uses `"nbfm"`) and two in
`test_synthetic_am_ssb.py` (registry contents; WBFM absence, replaced with a test that
WBFM stays a distinct scheme and rejects `sideband`). Nothing was loosened or deleted.

- Analog focused (4 files): **108 passed**
- Full suite: **523 passed, 1 failed** - the pre-existing `reedsolo` gap, with
  `tests/test_model_report.py` ignored (pre-existing missing `h5py`)

### Regression - verified

| Check | Result |
|---|---|
| Frozen V1 aggregate .iq SHA-256[:32] | `d6d3f918687d0700a43e46211c3f04b9` (40 captures) - unchanged |
| Digital `MODULATIONS` | `['16QAM','64QAM','8PSK','BFSK','BPSK','GFSK','PAM4','QPSK']` - unchanged |
| Analog registry / schemes | `['AM-DSB','AM-SSB','WBFM']` / `('am_dsb','am_ssb','wbfm')` |
| `symbol_rate_experiment.MODULATIONS` | six digital names - unchanged |
| AM-DSB waveform | asserted equal to `1 + 0.5*s(t)`, imag 0 |
| AM-SSB waveform | asserted equal to `hilbert(s) * exp(j2*pi*fc*t)` |
| PAM4 / GFSK / symbol-rate estimator / CNN / fusion / dispatch / FEC | not touched |
| Analog artifacts in repo | none - tests use pytest `tmp_path` |

### Remaining limitations

1. WBFM captures are classified `BPSK` by the CNN and routed to the 2PSK demodulator.
   The CNN has no analog class; that is the integration task, not this one.
2. `dispatch` still decimates analog captures by a meaningless estimated symbol rate.
3. Analog is still not wired into any dataset builder, manifest or training set.
4. The oracle's discriminator has no limiter or post-detection filter, so its SNR curve
   is pessimistic relative to a real FM receiver.
5. Only WBFM is implemented - NBFM is not, and no analog schemes beyond these three are
   planned in the current scope.

---

## Entry 025 - 2026-09-09 - Forensic investigation: where analog identity is lost (READ-ONLY)

No production code, test, dataset or model was modified. `git status` was byte-identical
before and after. Recorded here because Entry 026 depends on these measurements.

### Two prior claims corrected against the current code

1. **"The CNN has no analog class" is false for the default checkpoint.**
   `models_saved/modulation_cnn.pt` carries 11 labels including `AM-DSB`, `AM-SSB`,
   `WBFM`. Only the V2 checkpoints (`_v2_8class` 8 labels, `_v2_synth` 6) lack them, and
   the harness uses the default when `model_path is None`.
2. **"Analog never reaches an analog demodulator" is false.** AM-SSB is labelled `WBFM`
   at 0.476 - above threshold - and routed to `demodulate_fm`. It reaches the *wrong*
   analog demodulator, not none.

### MEASURED - three analog traces (Fs 200 kHz, N 8192, 20 dB, seed 7, default checkpoint)

| Field | AM-DSB | AM-SSB | WBFM |
|---|---|---|---|
| `classical_family` | QAM-like | QAM-like | **analog-like** |
| `cnn_label` / conf | PAM4 / 0.394 | WBFM / 0.476 | BPSK / 0.267 |
| CNN top-3 | PAM4 .394, AM-DSB .331, WBFM .180 | WBFM .476, AM-DSB .368, GFSK .080 | BPSK .267, AM-SSB .152, 8PSK .129 |
| `fusion_label` | Unclassified | WBFM | Unclassified |
| `demod_available` / scheme | False / None | True / WBFM | False / None |
| `est_symbol_rate_hz` | 3857.4 | 2002.0 | 2929.7 |
| `est_carrier_hz` (true) | 11.0 (0) | 22147.2 (20000) | 19957.6 (20000) |
| `ber_status` | unavailable | unavailable | unavailable |

Entry 024 recorded WBFM -> BPSK 0.407 -> 2PSK at seed 23; at seed 7 the same
configuration gives 0.267 -> Unclassified. Both are real: **analog CNN confidence sits
astride the 0.4 threshold**, so routing flips between rejection and a wrong demodulator
depending on the capture. Any analog work must be validated across seeds.

### Three distinct losses of analog identity

**(a) `dsp/cyclostationary.py:44-51` - `analog-like` is a fallback, not a detection.**
It is the bare `else` after three digital tests, at a hard-coded confidence of 0.45.
Measured `amplitude_cv`: AM-DSB **0.2068** against the `amplitude_cv_qam = 0.2`
threshold - it misses `analog-like` by **0.007**. AM-SSB 0.4609 -> QAM-like. WBFM 0.0698
reaches `analog-like` only by elimination.

**(b) `fusion/confidence_fusion.py` - the decisive transition.** `classical_family` feeds
only `agrees`, which scales `trust_score`; the returned `label` is `ml_label` or
`"Unclassified"` and is never derived from the classical family. So `analog-like` can
drive trust down to 0.133 and set `review_recommended=True` while the label stays `BPSK`.
Notably `_FAMILY_BY_LABEL` **already maps all three analog labels** - fusion was built
expecting analog labels from the CNN. The gap is that a classical analog verdict has no
path to an analog label when the CNN's top-1 disagrees.

**(c) `dsp/preprocessing.py:17` - `iq -= np.mean(iq)` deletes the AM-DSB carrier.**
For baseband DSB the DC mean *is* the carrier, measured at exactly `1.000000+0.000000j`:

| carrier offset | amplitude_cv raw -> preprocessed | envelope recovery raw -> preprocessed |
|---|---|---|
| **0 Hz** | 0.2068 -> 0.6984 | 1.0000 -> **0.0004** |
| 20 kHz | 0.2068 -> 0.2068 | 1.0000 -> 1.0000 |

DSB-with-carrier is silently turned into DSB-suppressed-carrier. Offset-0 specific; the
Entry 020/021 captures used offset 0.

### MEASURED - dispatch decimates analog by a digital concept

`dispatch.py:73-78` runs before the analog branch at line 86: a symbol rate is
*required*, then the waveform is decimated by it with no anti-alias filter. AM-DSB, true
tones 926.3 / 2688.3 / 2932.6 Hz:

| sps | Fs' | Nyquist | tones above Nyquist | true tones fold to | measured peaks |
|---|---|---|---|---|---|
| 1 | 200 kHz | 100 kHz | 0/3 | 926.3, 2688.3, 2932.6 | 927.7, 2685.5, 2929.7 |
| 52 | 3846 Hz | 1923 Hz | **2/3** | 926.3, 1157.8, 913.6 | 852.0, 925.0, 1168.5 |
| 68 | 2941 Hz | 1471 Hz | **2/3** | 926.3, 252.8, 8.6 | 24.3, 243.1, 923.7 |
| 100 | 2000 Hz | 1000 Hz | **2/3** | 926.3, 688.3, 932.6 | 682.9, 853.7, 926.8 |

Measured peaks match the predicted fold positions, and only 82-158 of 8192 samples
survive. A correctly-null analog symbol rate would *block* demodulation at line 73, so
dispatch must learn to skip decimation for analog **before** estimation may return null.

### MEASURED - analog demodulator interfaces

| Function | Needs symbol rate? | Meaningful output | Limitation |
|---|---|---|---|
| `demodulate_am(samples)` | No | corr **1.0000** on raw DSB | envelope detector; **0.0004** once `preprocess` strips DC |
| `demodulate_ssb(samples, fs, carrier)` | No (needs fs + carrier) | true carrier 20000 -> **1.0000** | estimated carrier 22147 -> **-0.0119**; `None` -> 0.0000 |
| `demodulate_fm(samples)` | No | yes | bare discriminator; no limiter, de-emphasis or audio filter |

Cross-checks: `demodulate_fm` on SSB -> 0.0002, `demodulate_am` on SSB -> -0.0003. Wrong
analog routing yields noise.

### Recommendation carried into Entry 026

Smallest safe change: let a *rejected* digital CNN decision with `analog-like` classical
evidence recover an analog label from the CNN's own ranked alternatives. Touches no
digital path, no threshold, no dispatch.

---

## Entry 026 - 2026-09-09 - Analog-aware fusion fallback (minimal integration patch)

Implements the Entry 025 recommendation. The global 0.4 threshold, the CNN, all
checkpoints, dispatch, preprocessing, the BER guard and the analog generators are
**unchanged**. Frozen V1 is byte-identical (`d6d3f918687d0700a43e46211c3f04b9`, 40
captures).

### Production changes - two files

**1. `fusion/confidence_fusion.py`.** New `ANALOG_FAMILY`, `ANALOG_LABELS`
(`{"AM-DSB","AM-SSB","WBFM"}`), `_recover_analog_alternative()`, a new optional keyword
`ranked_alternatives: tuple[tuple[str, float], ...] = ()`, and one new
`FusionResult.analog_fallback: bool = False` field (defaulted, so existing readers are
unaffected). The legacy `alternatives` argument and every positional call are unchanged.

The new path, and only this path:

    if rejected and classical_family == "analog-like" and ml_label not in ANALOG_LABELS:
        recover the highest-ranked analog label among ranked_alternatives

Trust for a recovered label is `min(1.0, alternative_confidence * 1.1)` - the
alternative's own softmax value with the existing agreement bonus, since it agrees with
the classical family by construction. **No confidence is invented**, and the recovered
trust is necessarily below the rejected primary's. `review_recommended` stays `True`.

**2. `pipeline.py:73-76`.** Previously discarded CNN confidences
(`tuple(label for label, _ in prediction.top_k[1:])`). Now builds `ranked` once and
passes both `alternatives` (unchanged shape) and `ranked_alternatives`. No second
inference pass.

### Decision table - before -> after

| Situation | Before | After |
|---|---|---|
| Digital CNN above threshold | label kept | **unchanged** |
| Digital CNN above threshold + `analog-like` evidence | label kept, review=True | **unchanged** - a confident CNN is never overridden |
| Digital CNN **below** threshold + `analog-like` + analog alternative | Unclassified | **recovered analog label**, `analog_fallback=True` |
| Digital CNN below threshold + `analog-like` + no analog alternative | Unclassified | **unchanged** |
| Digital CNN below threshold + digital evidence | Unclassified | **unchanged** - gated on the family, not on low confidence |
| Analog CNN above threshold (right or wrong) | label kept | **unchanged** (see limitation below) |
| Analog CNN below threshold | Unclassified | unchanged unless another analog label ranks below it |

### MEASURED - multi-seed validation (5 seeds x 3 schemes, 20 dB, default checkpoint)

| Case | Unclassified before -> after | analog label before -> after | analog demod reached | exactly correct |
|---|---|---|---|---|
| AM-DSB | 1/5 -> 1/5 | 0/5 -> 0/5 | 0/5 | 0/5 |
| AM-SSB | 1/5 -> 1/5 | 4/5 -> 4/5 | 4/5 | 0/5 |
| WBFM | 4/5 -> **3/5** | 0/5 -> **1/5** | **1/5** | 0/5 |

**The fallback fires in 1 of 15 cases, and the one time it fires it picks the wrong
analog label** (WBFM seed 7 -> `AM-SSB`, from the CNN's own top-2 at 0.152). This is
reported as measured, not presented as a success.

Why it fires so rarely - the gate state per capture:

| Case | gate open | analog label present in top-k | blocker |
|---|---|---|---|
| AM-DSB, all 5 seeds | **No** | Yes (AM-DSB ranks 2nd, .107-.331) | `classical_family == "QAM-like"` |
| AM-SSB, all 5 seeds | **No** | Yes (AM-DSB ranks 2nd, .273-.368) | `classical_family == "QAM-like"` |
| WBFM seeds 3, 11, 17 | Yes | **No** | CNN top-3 is entirely digital |
| WBFM seed 7 | Yes | Yes (AM-SSB .152) | fires - wrong label |
| WBFM seed 23 | No (0.407 > 0.4) | Yes (WBFM .184) | above threshold, unchanged by design |

So for AM-DSB and AM-SSB the remaining blocker is **not** fusion - the analog evidence is
already in the CNN top-k in 10/10 cases, but the classical detector reports `QAM-like`
and the gate never opens. That is Entry 025 finding (a), and fixing it means touching
`cyclostationary.py`, which was explicitly out of scope here.

A second structural limiter: `modulation_inference.predict_modulation` defaults to
`top_k=3`, so fusion only ever sees **two** alternatives. Widening it would change
`prediction.top_k` for every capture and every harness record, so it was not done here.

### Digital regression - verified

| Check | Result |
|---|---|
| Frozen V1 aggregate .iq SHA-256[:32] | `d6d3f918687d0700a43e46211c3f04b9` (40 captures) - unchanged |
| Frozen V1: fallback gate open | **0 of 40** |
| Frozen V1: fallback fired | **0 of 40** |
| Frozen V1: fusion label changed | **0 of 40** |
| `threshold` default | still `0.4` |
| Existing fusion/QAM-routing/FSK-timing/QAM-scale tests | pass |

### Tests

New `tests/test_fusion_analog_fallback.py`, **19 tests** covering the required cases A-F
plus API compatibility: accepted digital unchanged; accepted digital with analog evidence
unchanged; the fallback itself; trust not fabricated and bounded by the rejected primary;
review still recommended; no analog alternative -> rejection; no alternatives -> rejection;
digital evidence -> untouched; the existing `test_dsp_fusion` rejection case preserved;
accepted analog preserved; low-confidence analog primary not self-rescued; highest-ranked
analog wins; rank order beats alphabetical; `ANALOG_LABELS` asserted against the actual
dispatch source; a recovered label reaching `demodulate_am` for real; legacy
`alternatives` still accepted; positional signature unchanged; threshold default pinned;
and `pipeline` asserted to pass `ranked_alternatives`.

- Focused (fusion + dispatch + analog + QAM/FSK regression, 9 files): **193 passed**
- Full suite: **542 passed, 1 failed** - the pre-existing `reedsolo` gap, with
  `tests/test_model_report.py` ignored (pre-existing missing `h5py`)

### Analog cases still failing - stated plainly

1. **AM-DSB and AM-SSB are not helped at all.** Blocked upstream by the classical
   detector reporting `QAM-like` (AM-DSB misses the analog branch by 0.007 of
   `amplitude_cv`).
2. **The one firing case selects the wrong analog label.** WBFM -> AM-SSB.
3. **Above-threshold wrong analog predictions are left unchanged**, as instructed:
   AM-SSB is labelled `WBFM` at 0.416-0.476 in 4/5 seeds and routed to `demodulate_fm`.
   The current architecture has no evidence that could distinguish a right analog label
   from a wrong one at that confidence, and inventing a heuristic would have changed
   unrelated behaviour.
4. **Dispatch still decimates analog by an estimated symbol rate** (Entry 025 table);
   not touched here.
5. **`preprocess` still strips the baseband AM-DSB carrier**, so `demodulate_am` would
   receive a carrier-less signal even when routed correctly.
6. **`demodulate_ssb` still receives an inaccurate carrier** (22147 Hz vs 20000 Hz true),
   which alone takes its recovery from 1.0000 to -0.0119.

### Honest assessment

The patch does exactly what it was scoped to do and provably breaks nothing: 0 of 40
frozen V1 captures change. But it is **not** an analog integration on its own - measured
end-to-end benefit is 1 firing in 15 analog captures, with the wrong label. The dominant
blocker has moved upstream to `cyclostationary.py`, which is the natural next task.

---

## Entry 027 - 2026-09-09 - Positive family-level analog evidence in the classical detector

Replaces `analog-like`-by-elimination with a measured positive test. No CNN,
checkpoint, fusion threshold, dispatch, preprocessing, symbol-rate or generator change.
Frozen V1 is byte-identical (`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

### Original detector behaviour, reproduced

`estimate_modulation_family` computes three features and tests them in order:

    amplitude_cv      = std(|s|) / mean(|s|)
    frequency_cv      = std(diff(unwrap(angle(s)))) / mean(|diff(unwrap(angle(s)))|)
    fourth_power_line = |mean(exp(4j*phase))|

    1. amplitude_cv < 0.15 and frequency_cv > 0.8   -> FSK-like
    2. amplitude_cv < 0.2  and fourth_power > 0.2   -> PSK-like
    3. amplitude_cv >= 0.2                          -> QAM-like
    4. else                                         -> analog-like   (confidence 0.45, flat)

Reproduced, not inherited (preprocessed, 20 dB, seed 7): AM-DSB@0 `amplitude_cv`
**0.6019** -> QAM-like; AM-SSB **0.4609** -> QAM-like; WBFM **0.0698** with
`frequency_cv` 0.349 -> falls to branch 4 by elimination. Entry 025 quoted AM-DSB at
0.2068 - that is the **raw** waveform; after `preprocess` it is 0.60, so branch 3 fires
even harder than previously recorded.

### Diagnostic matrix - no single existing feature separates

72 captures (4 analog configurations + 8 digital, 2 SNR, 3 seeds). Min/median/max:

| Feature | ANALOG | DIGITAL | Separable |
|---|---|---|---|
| `amplitude_cv` | 0.0698 / 0.3752 / 0.6085 | 0.0709 / 0.2168 / 0.5121 | no |
| **`frequency_cv`** | **0.175 / 0.572 / 1.436** | **1.107 / 1.688 / 2.907** | **nearly - overlap only from AM-DSB@0** |
| `fourth_power_line` | 0.0006 / 0.0046 / 0.278 | 0.0011 / 0.282 / 0.961 | no |
| envelope spectral flatness | 0.0167 / 0.240 / 0.554 | 0.117 / 0.537 / 0.574 | no |
| IF spectral flatness | 0.059 / 0.416 / 0.575 | 0.244 / 0.557 / 0.605 | no |
| IF low-band fraction | 0.0017 / 0.075 / 0.807 | 0.072 / 0.099 / 0.640 | no |
| envelope low-band fraction | 0.103 / 0.635 / 0.975 | 0.082 / 0.218 / 0.662 | no |

Three candidate features were built and rejected on measurement (envelope spectral
peakiness, IF spectral flatness, low-band energy fraction) - each overlapped. Only
`frequency_cv`, a feature the detector **already computes**, separates.

**Why it is physical, not a fitted threshold.** `frequency_cv` measures how *impulsive*
the instantaneous frequency is. A digital signal jumps at symbol boundaries and sits
still between them, so phase increments are heavy-tailed and `std >> mean|.|`. An analog
message moves the instantaneous frequency smoothly, so the two are comparable. It
detects **symbol-transition structure**, which is exactly the digital/analog family
distinction - and it names no modulation.

Explicitly **not** done, per Entry 025: `amplitude_cv_qam` was not raised from 0.2, and
"constant envelope = analog" was not assumed (WBFM, PSK and FSK are all constant
envelope; the matrix shows `amplitude_cv` does not separate).

### Threshold selection - measured, on a held-out set

Tuning used seeds 3/7/11; validation used **held-out seeds 101/103/107/109(/113)**.
Digital controls: 8 modulations x samples-per-symbol 4/8/16/32 x SNR 0/5/10/15/20 dB.

**Lowest digital `frequency_cv` observed anywhere: 1.032** (BFSK, sps=4), stable across
oversampling (1.03-1.28) and SNR. Threshold sweep:

| T | analog detected | excl. AM-DSB@0 | digital false positives |
|---|---|---|---|
| 0.80 | 41.2% | 55.0% | **0 / 640** |
| **0.90** | **47.5%** | **63.3%** | **0 / 640** |
| 1.00 | 55.0% | 73.3% | 0 / 640 |
| 1.05 | 55.0% | 73.3% | 4 / 640 (0.62%) |

**0.9 chosen** over the higher-yield 1.00: it keeps ~13% margin below the lowest observed
digital value, where 1.00 leaves only 3%. Conservative, as instructed.

### Production change - one file, one branch

`src/radiofry/dsp/cyclostationary.py`: new module constant `ANALOG_FREQUENCY_CV_MAX =
0.9` (plus two confidence constants), and a new branch inserted **after** FSK and PSK and
**before** QAM:

    elif frequency_cv < ANALOG_FREQUENCY_CV_MAX and fourth_power_line < 0.2:
        analog-like, confidence = 0.5 + 0.4 * (1 - frequency_cv / 0.9)

Placement is deliberate: FSK and PSK keep absolute priority, so the new test can only
take captures that would have been **QAM-like**, which is exactly the observed failure.
The fourth-power veto reuses the existing `fourth_power_psk` constant - no second magic
number - and blocks real-valued/carrier-bearing signals whose `frequency_cv` is ~0.

Confidence is now evidence-based (0.5-0.9, scaling with margin) instead of a flat 0.45.

The final `else` returns **`"unknown"`** (confidence 0.2) instead of `analog-like`, so
analog is a positive verdict only. Measured impact: that branch fired **0 times in 900
captures**, and it makes the Entry 026 fusion fallback strictly more conservative, never
less.

### MEASURED - validation, 5 held-out seeds

Analog detection (new rule), by scheme and SNR:

| Scheme | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| AM-DSB @ 0 Hz | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| AM-DSB @ 20 kHz | **5/5** | **5/5** | **5/5** | 0/5 | 0/5 |
| AM-SSB | **5/5** | **5/5** | **5/5** | 0/5 | 0/5 |
| WBFM | **5/5** | **5/5** | **5/5** | 2/5 | 0/5 |

Digital controls, 800 captures: **0 analog false positives (0.00%)**. Family transitions
old -> new: QAM-like -> QAM-like 600, FSK-like -> FSK-like 199, PSK-like -> PSK-like 1.
**Not one digital capture changed family.**

Frozen V1, all 40 captures: QAM-like 28, FSK-like 12, **analog-like 0, unknown 0** - no
family changed. Lowest `frequency_cv` on the real frozen dataset is **1.032**, matching
the synthetic controls exactly.

Below ~10 dB all analog schemes converge on `frequency_cv` ~1.16 as noise makes the phase
impulsive, so low-SNR analog is not detectable by this feature. Stated, not hidden.

### AM-DSB at carrier offset 0 - answered explicitly, not worked around

`preprocess` removes DC, and for baseband AM-DSB the DC mean **is** the carrier
(`1.000000+0.000000j`). Measured on the same capture:

| | `amplitude_cv` | `frequency_cv` |
|---|---|---|
| raw waveform | 0.2036 | **0.0000** |
| after `preprocess` | 0.6990 | **7.0457** |

The carrier-bearing waveform has a perfectly static phase. Removing DC leaves a bipolar
residual whose phase flips by pi at every message zero crossing, making it **more
impulsive than any digital control measured** (worst digital: 2.91). **No family-level
rule can recover analog identity from that**, and none was invented. Pinned by
`test_baseband_am_dsb_is_documented_as_undetectable_after_preprocessing`. Fixing it means
changing `preprocess`, which is out of scope and would alter every digital capture.

### End-to-end interaction with the Entry 026 fusion fallback

Traced classical -> CNN -> fusion -> dispatch, 5 held-out seeds at 20 dB, default
11-class checkpoint:

| Case | classical analog | fallback fired | analog label | analog demod reached | **label CORRECT** | unclassified |
|---|---|---|---|---|---|---|
| AM-DSB @ 0 Hz | 0/5 | 0/5 | 2/5 | 2/5 | **2/5** | 1/5 |
| AM-DSB @ 20 kHz | **5/5** | **5/5** | **5/5** | **5/5** | **4/5** | 0/5 |
| AM-SSB | **5/5** | 0/5 | 3/5 | 3/5 | **0/5** | 2/5 |
| WBFM | **5/5** | 1/5 | 1/5 | 1/5 | **0/5** | 3/5 |

- **AM-DSB @ 20 kHz is the real win.** Classical 0/5 -> 5/5, the Entry 026 fallback fires
  5/5, and **4/5 are labelled `AM-DSB` and reach `demodulate_am`**. Entry 026 alone
  achieved 0/5 here.
- **AM-SSB is now recognised as analog 5/5 but still ends up wrong.** The CNN's top-1 is
  `WBFM`; in 3/5 it is above threshold (accepted, routed to `demodulate_fm` - wrong
  analog demodulator) and in 2/5 below threshold with no *other* analog alternative in
  the top-3, so it stays Unclassified. The fallback never fires. The blocker has moved
  from the detector to **CNN quality**.
- **WBFM is unchanged** (it was already `analog-like`). 1/5 fallback fired and chose
  `AM-SSB` - wrong.
- **AM-DSB @ 0 Hz** is not helped by the detector, though the CNN independently produced
  `AM-DSB` above threshold in 2/5.

`analog-like` appearing is **not** claimed as success: the correct-label column above is
the honest measure, and it is 6 of 20.

### Tests

New `tests/test_classical_analog_detection.py`, **40 tests**: threshold published and
reusing the existing fourth-power constant; the three analog schemes detected at 20 dB;
AM-SSB across 20/15/10 dB; evidence-based (non-constant) confidence bounded to 0.5-0.9;
the evidence dict unchanged; **8 digital controls pinned to their exact current family**
(including the pre-existing quirk that high-SNR PSK reads FSK-like); 14 parametrised
"never analog" control cases at 20 and 10 dB; oversampling 4/16/32 not making digital
look analog; FSK priority preserved; a strong fourth-power line vetoing the analog
verdict; `analog-like` no longer in the final `else`; short input still `unknown`; the
API contract; and the baseband AM-DSB limitation.

- Focused detector: **40 passed**
- Full suite: **582 passed, 1 failed** - the pre-existing `reedsolo` gap, with
  `tests/test_model_report.py` ignored (pre-existing missing `h5py`)

### Remaining blockers, separated by layer

**Detector (this layer, remaining):** analog below ~10 dB is indistinguishable by this
feature; baseband AM-DSB is unrecoverable after preprocessing.

**Preprocessing:** `preprocessing.py:17` DC removal destroys the AM-DSB@0 carrier. Not
touched - it would change every digital capture and invalidate the Entry 017 baseline.

**CNN:** now the dominant analog blocker. AM-SSB is confidently mislabelled `WBFM`
(0.32-0.49); WBFM's own label rarely ranks first. No retraining was done.

**Fusion:** an above-threshold *wrong* analog label is still accepted (Entry 026
limitation, unchanged).

**Dispatch:** analog is still decimated by an estimated symbol rate, and a null analog
symbol rate would block demodulation at `dispatch.py:73`.

**SSB carrier estimation:** 22147 Hz vs 20000 Hz true takes `demodulate_ssb` from 1.0000
to -0.0119.

**Demodulator quality:** no audio filtering, no FM limiter or de-emphasis.

---

## Entry 028 - 2026-09-09 - Analog dispatch bypasses the digital symbol-rate path

Dispatch only. No CNN, checkpoint, fusion threshold, classical detection, preprocessing,
symbol-rate estimation, generator, BER-guard, FSK-timing or FEC change. Frozen V1 is
byte-identical (`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

### The failure, traced in the current code

`demodulate_capture` before this entry:

| Line | What it did |
|---|---|
| 71-72 | skip `Unclassified` / `unknown` / `""` |
| **73-74** | **`if parameters.symbol_rate_hz is None ... return unavailable`** - a correctly-null analog symbol rate was rejected here |
| **75** | `samples_per_symbol = round(sample_rate / symbol_rate_hz)` |
| **76-77** | digital timing-offset search (`_linear_timing_offset`) |
| **78** | **`symbol_samples = signal.iq[timing_offset::samples_per_symbol]`** - decimation |
| 86-94 | analog branch, reached only *after* all of the above |

So analog demodulators received a decimated, timing-shifted signal, and
`demodulate_ssb` was additionally handed `sample_rate / samples_per_symbol` as its
down-conversion rate.

### Production change - one file, one helper

`src/radiofry/decoding/demodulators/dispatch.py`:

- New module constant `ANALOG_LABELS = frozenset({"AM-DSB", "AM-SSB", "WBFM"})`,
  confirmed from the code and asserted equal to `fusion.ANALOG_LABELS` by test.
- New `_demodulate_analog(signal, modulation, parameters)`: full-rate, no
  samples-per-symbol, no timing search, no decimation. `demodulate_ssb` now receives
  `signal.sample_rate` itself.
- `demodulate_capture` gains one line, placed **before** the symbol-rate requirement:

      if modulation in ANALOG_LABELS:
          return _demodulate_analog(signal, modulation, parameters)

- The dead analog branch was removed from the digital `try` block. **Every remaining
  digital line is unchanged**, including the symbol-rate requirement, both timing-offset
  heuristics and the `samples per symbol ... offset` message.
- Analog still emits median-threshold bits so the Entry 021 BER guard keeps refusing
  them; analog reporting behaviour is unchanged.

Symbol rate is now optional **for the analog branch only**. `AM-SSB` still requires a
sample rate (its product detector needs one) and says so explicitly.

### MEASURED - before vs after

Fs 200 kHz, N 8192, noiseless, seed 101, using each scheme's own estimated symbol rate:

| Case | est. Rs | sps | BEFORE samples | AFTER samples | BEFORE corr | AFTER corr |
|---|---|---|---|---|---|---|
| AM-DSB | 3857 Hz | 52 | 157 | **8192** | 1.0000* | 1.0000 |
| AM-SSB | 2002 Hz | 100 | 82 | **8192** | 1.0000* | 1.0000 |
| WBFM | 2930 Hz | 68 | 120 | **8191** | **0.1454** | **1.0000** |

*An honest caveat about the AM "before" column:* those 1.0000 values are correlations
against the **decimated** message `msg[offset::sps]`, and envelope detection commutes
with decimation, so the metric is trivially satisfied and **misleading**. It is shown to
make that explicit, not as evidence the old path was fine. FM discrimination does *not*
commute with decimation, which is why WBFM exposes the damage directly: **0.1454 ->
1.0000**.

The real damage for AM is aliasing. Recovered AM-DSB tone content:

| | tone 1 | tone 2 | tone 3 |
|---|---|---|---|
| true message | 753.4 Hz | 2326.3 Hz | 2835.6 Hz |
| **BEFORE** (sps=52, Fs'=3846 Hz) | 759.4 | **1004.4** | **1518.9** |
| **AFTER** (full rate) | 756.8 | **2319.3** | **2832.0** |

Two of three tones folded before; all three are correct now.

### MEASURED - AM-SSB: dispatch corruption separated from carrier error

At full rate, with dispatch fixed:

| Carrier passed to `demodulate_ssb` | Recovery correlation |
|---|---|
| true 20000.0 Hz | **+1.0000** |
| currently estimated 22147.2 Hz | **+0.0115** |

Dispatch is no longer the problem for SSB; **carrier estimation is**, and it is not
solved here. Pinned by `test_am_ssb_still_fails_with_the_currently_estimated_carrier`
so the two can never be conflated again.

### MEASURED - end to end (classical -> CNN -> fusion -> dispatch -> demod), 20 dB

| Case | seed | classical | fusion | demod | est Rs | **Rs used?** | message corr | label |
|---|---|---|---|---|---|---|---|---|
| AM-DSB@0 | 103 | QAM-like | AM-DSB | AM-DSB | 977 | **no** | 0.0000 | correct |
| AM-DSB@0 | 107 | QAM-like | AM-DSB | AM-DSB | 244 | **no** | 0.0084 | correct |
| **AM-DSB@20k** | 103 | analog-like | AM-DSB | AM-DSB | 2173 | **no** | **0.9414** | **correct** |
| **AM-DSB@20k** | 107 | analog-like | AM-DSB | AM-DSB | 1050 | **no** | **0.9431** | **correct** |
| AM-DSB@20k | 101 | analog-like | WBFM | WBFM | 757 | no | -0.0000 | wrong |
| AM-SSB | 101/103 | analog-like | WBFM | WBFM | 513/2466 | no | ~0.000 | wrong |
| WBFM | 107 | analog-like | AM-SSB | AM-SSB | 1050 | no | -0.0015 | wrong |
| WBFM | 101/103 | analog-like | Unclassified | - | 757/14526 | - | - | - |

**The symbol rate was used in zero analog cases**, including estimates as absurd as
244 Hz and 14526 Hz - the point of this entry.

**AM-DSB at a non-zero carrier offset now recovers its message end to end through
production at 0.94 correlation** where the pre-Entry-025 path could not. AM-DSB@0 is
labelled correctly but recovers ~0.00 because `preprocess` already removed its carrier -
a preprocessing limitation, not a dispatch one. Every ~0.00 row with a *wrong* analog
label is the wrong demodulator running, which is a CNN problem.

Correct labels remain **4 of 12**: this entry fixed dispatch, not classification, and no
classification success is claimed.

Harness record for a WBFM capture: `ber_status="unavailable"`,
`ber_reason="analog_no_transmitted_bits"`, `ber_strict=None`, `compared_bits=0` - the
Entry 021 guard is intact.

### Tests

New `tests/test_analog_dispatch_bypass.py`, **32 tests**: the published analog label set
and its agreement with fusion's; all three schemes demodulating with
`symbol_rate_hz=None`; **six digital labels still failing on a null symbol rate**;
analog output byte-identical across symbol rates of None / 137 Hz / 91 kHz; analog
demodulators receiving >= N-1 samples; the dispatch message no longer mentioning samples
per symbol; AM-DSB, AM-SSB and WBFM recovery > 0.99; SSB failing with the estimated
carrier; SSB using the full rate; PSK/QAM/FSK dispatch unchanged and still reporting the
timing search; digital still requiring a sample rate; `Unclassified` still skipped;
unknown labels still rejected; `DispatchResult` semantics; and analog still emitting the
threshold bits the BER guard refuses.

**Two prior tests were deliberately updated** - both asserted the behaviour this entry
removes:
- `test_decoding_correlation.py::test_dispatch_routes_ssb_and_reports_timing_search` ->
  `..._at_the_full_capture_rate`. It required `"coarse timing search"` in the message and
  compared against a **decimated** reference `audio[offset::8]`. It now asserts no
  decimation (`symbols.size == audio.size`) and compares against the **full-rate**
  message - a strictly stronger assertion, still at > 0.99.
- `test_fusion_analog_fallback.py::test_the_analog_label_set_matches_the_dispatch_routes`
  scraped `demodulate_capture`'s source for label literals; it now compares the two
  `ANALOG_LABELS` sets directly.

- Focused dispatch: **32 passed**
- Full suite: **614 passed, 1 failed** - the pre-existing `reedsolo` gap, with
  `tests/test_model_report.py` ignored (pre-existing missing `h5py`)

### Remaining blockers, by layer - none of them dispatch

**Preprocessing:** `preprocessing.py:17` DC removal still destroys the baseband AM-DSB
carrier, so AM-DSB@0 recovers ~0.00 even when labelled and routed correctly.

**CNN classification - dominant:** AM-SSB is confidently mislabelled `WBFM`; WBFM is
labelled `AM-SSB` or a digital class. Wrong analog labels now reach the wrong analog
demodulator cleanly, which is a routing-quality problem, not a dispatch one.

**SSB carrier estimation:** 22147 Hz vs 20000 Hz true, measured above as +1.0000 ->
+0.0115.

**Demodulator quality:** no audio filtering, no FM limiter, no de-emphasis.

**NBFM:** not implemented.

---

## Entry 029 - 2026-09-09 - AM-SSB carrier estimation: no safe blind fix (NEGATIVE RESULT)

**No production code was changed.** The investigation established that the carrier
estimator is not the fixable part of this problem: blind SSB carrier recovery needs
~1 Hz accuracy, and the best signal-only candidate reaches ~500 Hz noiseless and
collapses entirely under noise. Forcing a change would have been metric theatre.
Frozen V1 byte-identical (`d6d3f918687d0700a43e46211c3f04b9`, 40 captures); full suite
unchanged at 614 passed.

### 1. Failure reproduced against current code

AM-SSB USB, Fs 200 kHz, N 8192, true carrier 20000.0 Hz, seed 101, noiseless:

| | Value |
|---|---|
| estimated carrier | **21859.5 Hz** (error **+1859.5 Hz**) |
| method | `welch_psd_centroid+nth_power` |
| recovery with **true** carrier | **+1.0000** |
| recovery with **estimated** carrier | **+0.0066** |

(Entry 028 quoted 22147 Hz for seed 7; the offset is message-dependent, see below.)

### 2. Cause - identified exactly, not guessed

`parameter_estimation.py:119` computes the **PSD centroid** over the occupied band:

    centroid = sum(f * P(f)) / sum(P(f))

Line 121 returns that as `carrier_frequency_hz` whenever no hardware
`center_frequency_hz` metadata is present. The centroid equals the carrier **only for a
spectrum that is symmetric about it**. Every digital modulation and AM-DSB are
double-sideband and therefore symmetric; **AM-SSB is one-sided by construction**, so the
centroid lands at carrier + the message's own power-weighted centroid:

| | Hz |
|---|---|
| message tones | 753.4, 2326.3, 2835.6 |
| power-weighted tone mean | **1859.8** |
| carrier + that | **21859.8** |
| actual estimate | **21859.5** |

A match to 0.3 Hz. The estimator has no notion of modulation family or sideband, and
implicitly assumes a symmetric, carrier-centred spectrum.

### 3. Signal evidence - the carrier is genuinely not in the spectrum

Measured from the emitted USB waveform (Welch, nperseg 1024, the estimator's own view):

| Quantity | Value |
|---|---|
| 99% occupied band | 20507.8 .. 23046.9 Hz |
| true carrier vs low edge | 20000.0 vs 20507.8 (**-507.8 Hz**) |
| strongest bins | 20703, 20898, 22266, 22461, 22656, 22852 Hz |
| power above vs below true carrier | **23984 : 1** |

There is no spectral line at 20000 Hz - the carrier is suppressed, as the generator
intends. For LSB the mirror holds: band 16992.2..19531.2 Hz, carrier at the **upper**
edge (+468.8 Hz).

### 4. MEASURED - how much carrier error SSB can tolerate

This is the number that decides the task. USB, noiseless:

| carrier error | 0 | 1 | 2 | 5 | 10 | 20 | 50 | 500 | 1860 Hz |
|---|---|---|---|---|---|---|---|---|---|
| recovery corr | 1.0000 | 0.9891 | 0.9568 | 0.7478 | 0.2128 | -0.1822 | 0.0156 | 0.1887 | 0.0054 |

**Usable SSB demodulation requires the carrier to ~1-2 Hz** - 0.01% of the carrier. A
product detector translates the whole message by the error, so there is no graceful
degradation.

### 5. Candidate estimators - all tested, all rejected on measurement

| Candidate | Measured behaviour | Verdict |
|---|---|---|
| **PSD centroid** (current) | biased by exactly the message centroid: +1859 / +1367 / +1139 Hz across seeds | rejected - provably wrong for one-sided spectra |
| **Occupied-band edge** | noiseless +507.8 / +507.8 / +898.4 Hz (USB); **-16094 Hz at 20 dB, -108477 Hz at 10 dB** as the 99% band swallows noise | **rejected - worse than the centroid under noise** |
| **Finer FFT resolution** | 195 Hz bin -> +507.8; 48.8 Hz -> +703.1; 24.4 Hz -> +727.5 | rejected - resolution makes it *worse*, it resolves the first tone more sharply |
| **Spectral skew for sideband** | USB: -0.29, +0.30, +0.28; LSB: +0.29, -0.30, -0.24 - sign inconsistent across seeds | rejected - not a discriminator |
| **Sideband symmetry / peak-pair geometry** | SSB has no symmetric partner and no pilot | not applicable |
| **Assume a fixed message low-cutoff** | would encode our own generator's 300 Hz tone band into production | **rejected as disguised ground-truth leakage** |

### 6. Why no blind estimator can work - the physical argument

The carrier sits below the lowest message tone and **carries no power itself**. That gap
is a property of the message, not the signal format, and it varies per capture:

| seed | 101 | 103 | 107 | 109 |
|---|---|---|---|---|
| lowest tone (= unknowable gap) | 753.4 | 681.1 | 1052.5 | 536.5 Hz |

To hit 1 Hz an estimator would have to know that gap to 1 Hz, from a signal that contains
no information about it. **The requirement is ~500x tighter than the best achievable
blind estimate, and the gap is not a resolution problem.** This is the well-known reason
real SSB receivers require manual tuning or a transmitted pilot.

### 7. The legitimate path already exists and already works

`parameter_estimation.py:120-121` already prefers
`signal.metadata["center_frequency_hz"]` (or `carrier_frequency_hz`) over the centroid,
and `preprocess` preserves metadata. Verified end to end:

| | Value |
|---|---|
| estimated carrier | **20000.0 Hz** (error **0.0**) |
| method | `hardware_center_frequency+welch_nth_power` |
| SSB recovery | **1.0000** |

For a real SDR capture the tuner's centre frequency is legitimate hardware information,
not ground truth, and it is the correct source. **No code change is needed to use it** -
the ingestion path simply has to populate that metadata. Making the *synthetic* generator
write it would be ground-truth leakage into production and was not done.

### 8. Why no production change was made

The centroid is provably wrong for one-sided spectra, so a change was considered. It was
rejected because:

1. The only signal-only alternative (band edge) is ~500 Hz noiseless - still ~500x short
   of usable - and **catastrophically worse under noise** (-16 kHz at 20 dB).
2. It would therefore improve a reported number while leaving SSB demodulation exactly as
   broken, which is the metric gaming this project has avoided throughout.
3. `estimate_parameters` is shared infrastructure. The current digital carrier estimates
   sit at -19.7 to +123.3 Hz around a true 0 Hz (BPSK -19.7, QPSK -167.5, 8PSK +32.7,
   16QAM +123.3, 64QAM -44.4, BFSK +106.4, GFSK +113.5). Risking those for no functional
   gain is not justified.

### 9. Digital baseline recorded for whoever picks this up

Current estimator, 20 dB, seed 101, true carrier 0 Hz: BPSK **-19.7**, QPSK **-167.5**,
8PSK **+32.7**, 16QAM **+123.3**, 64QAM **-44.4**, BFSK **+106.4**, GFSK **+113.5** Hz.
Any future change must be measured against these.

### 10. Recommended next step

Populate `metadata["center_frequency_hz"]` from real capture headers in the ingestion
layer (SigMF, WAV chunk, SDR sidecar). That makes SSB work for real captures without a
single line of estimator change. For synthetic evaluation, accept that blind SSB carrier
recovery is out of reach and score SSB on the demodulator given a supplied carrier.

### Repository state

Read-only investigation. `git status` and `git diff --stat` are identical to the
pre-investigation state; full suite **614 passed, 1 failed** (pre-existing `reedsolo`,
with `test_model_report.py` ignored for the pre-existing missing `h5py`).

### Remaining blockers - unchanged by this entry

**Preprocessing:** DC removal still destroys the baseband AM-DSB carrier (AM-DSB@0
recovers ~0.00 even when routed correctly).
**CNN classification - dominant:** AM-SSB is confidently mislabelled `WBFM`, so most SSB
captures never reach `demodulate_ssb` regardless of carrier accuracy.
**Demodulator quality:** no audio filtering, FM limiter or de-emphasis.
**NBFM:** not implemented.

---

## Entry 030 - 2026-09-09 - SigMF sidecar centre-frequency ingestion (closes the Entry 029 gap)

Ingestion only. No estimator, demodulator, CNN, fusion, dispatch, preprocessing,
generator, FSK-timing or detector change. Frozen V1 byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

### Current metadata flow, traced

| Stage | File / function | Metadata it sets | Carrier present? |
|---|---|---|---|
| IQ read | `ingestion/iq_parser.py:read_iq` | `dtype`, `byte_order`, `path` | **no** |
| WAV read | `ingestion/wav_parser.py:read_wav` | `channel_mode`, `path` | **no** |
| Container | `contracts.py:UnifiedSignalContainer` | free-form `metadata: dict` | n/a |
| Preprocess | `dsp/preprocessing.py:35-36` | copies the dict, adds `preprocessed` | **preserved** |
| Estimate | `dsp/parameter_estimation.py:120-121` | **reads** `center_frequency_hz`, else `carrier_frequency_hz` | **consumed** |

So the consumer existed and the preservation path existed; **no parser ever populated
the field**. That was the entire gap. No header, chunk or sidecar parsing existed
anywhere - `grep -rni sigmf` over the source tree returned nothing.

### Format survey - what is legitimately available

- **Headerless IQ**: by definition carries no tuner frequency. A sidecar is the only route.
- **WAV**: the standard header has **no** RF tuning field. (SDR recorders such as SDR#
  write a non-standard `auxi` chunk, but `scipy.io.wavfile` does not expose chunks and
  parsing them was judged disproportionate here.) A sidecar is the reliable route, and
  this entry does not pretend otherwise.
- **SigMF**: the one widely-used standard that fits, needs only `json` from the standard
  library, and stores exactly what is needed.

**Option A was chosen.** Options B (a generic metadata interface) and C (no change) were
rejected: B is speculative architecture with no second consumer today, and C would leave
a working `estimate_parameters` branch permanently unreachable for real captures.

### Production change - one new file, two three-line parser hooks

**New `src/radiofry/ingestion/sidecar.py` (76 lines).** Deliberately **not** a SigMF
implementation - it reads two values and ignores the rest of the standard:

- `SIGMF_FREQUENCY_KEY = "core:frequency"` from the first entry of `captures`
- `SIGMF_SAMPLE_RATE_KEY = "core:sample_rate"` from `global`
- `sidecar_path_for()`: `cap.iq` / `cap.sigmf-data` -> `cap.sigmf-meta`
- `read_sigmf_sidecar()` returns `{}` on a missing, unreadable, non-JSON, non-object or
  `captures`-less file. **A missing value is never replaced by a default.**
- `_finite_non_negative()` rejects non-numeric values, `bool` (an `int` subclass that is
  never a frequency), NaN, infinity, negatives and anything above 1e15 Hz. **0.0 is
  accepted** - baseband is legitimate.

**`iq_parser.read_iq`**: consults the sidecar; sets `center_frequency_hz` and
`center_frequency_source="sigmf_sidecar"` only when a valid value is present, and adopts
the sidecar sample rate **only** when the caller supplied none.

**`wav_parser.read_wav`**: same, except the WAV's own header sample rate stays
authoritative and is never overridden.

Both leave their existing metadata keys untouched, so a capture without a sidecar
produces byte-identical behaviour to before.

### Anti-leakage - enforced, not just intended

The synthetic generator writes `.iq`, `.wav`, `.json` and `.message.npy`. It does **not**
write `.sigmf-meta`, and production ingestion never reads the generator's ground-truth
JSON. Three tests enforce this:

- `test_the_synthetic_generator_does_not_write_a_sigmf_sidecar` - asserts
  `list(tmp_path.glob("*.sigmf-meta")) == []` after generation.
- `test_ingestion_ignores_the_generators_ground_truth_json` - confirms `c.json` holds the
  true carrier and that `read_iq` still exposes no `center_frequency_hz`.
- `test_the_sidecar_reader_never_looks_at_a_plain_json_file` - a `cap.json` containing
  `core:frequency` is ignored; only `cap.sigmf-meta` is read.

Every sidecar in the test suite is written by the test itself, representing what a real
recorder emits. Where the true carrier appears it is an **independent expected value**,
never a production input.

### MEASURED - SSB end to end (ingestion -> preprocess -> estimator -> demodulator)

| Sideband | seed | SNR | sidecar | estimated carrier | error | method | recovery |
|---|---|---|---|---|---|---|---|
| USB | 101 | - | no | 21859.5 | +1859.5 | centroid | +0.0066 |
| USB | 101 | - | **yes** | **20000.0** | **+0.0** | hardware | **+1.0000** |
| USB | 101 | 20 dB | no | 21849.8 | +1849.8 | centroid | -0.0061 |
| USB | 101 | 20 dB | **yes** | **20000.0** | **+0.0** | hardware | **+0.9951** |
| USB | 103 | - | no | 21367.5 | +1367.5 | centroid | -0.0955 |
| USB | 103 | - | **yes** | **20000.0** | **+0.0** | hardware | **+1.0000** |
| USB | 103 | 20 dB | **yes** | **20000.0** | **+0.0** | hardware | **+0.9948** |
| LSB | 101 | - | no | 18140.6 | -1859.4 | centroid | +0.0068 |
| LSB | 101 | - | **yes** | **20000.0** | **+0.0** | hardware | **+1.0000** |
| LSB | 101 | 20 dB | **yes** | **20000.0** | **+0.0** | hardware | **+0.9951** |
| LSB | 103 | - | **yes** | **20000.0** | **+0.0** | hardware | **+1.0000** |
| LSB | 103 | 20 dB | **yes** | **20000.0** | **+0.0** | hardware | **+0.9948** |

**Carrier error 0.0 Hz and recovery 0.9948-1.0000 in all eight sidecar cases**, USB and
LSB alike. Without a sidecar the error stays at 1353-1860 Hz and recovery at -0.20..+0.14:
**Entry 029's negative result is unchanged and is pinned by a test**
(`test_ssb_recovery_still_fails_without_the_sidecar`). This entry did not make blind
estimation work - it supplied the information blind estimation cannot obtain.

### Digital regression - identical to the Entry 029 baseline

True carrier 0 Hz, 20 dB, seed 101:

| | BPSK | QPSK | 8PSK | 16QAM | 64QAM | BFSK | GFSK |
|---|---|---|---|---|---|---|---|
| Entry 029 baseline | -19.7 | -167.5 | +32.7 | +123.3 | -44.4 | +106.4 | +113.5 |
| now | -19.7 | -167.5 | +32.7 | +123.3 | -44.4 | +106.4 | +113.5 |
| delta (Hz) | -0.05 | -0.01 | +0.01 | +0.03 | +0.04 | +0.05 | +0.02 |

All within floating-point noise. Frozen V1 has no sidecars, so it still takes the
centroid path - pinned by `test_frozen_v1_estimation_is_unaffected_by_this_entry`.

### Tests

New `tests/test_ingestion_sigmf_metadata.py`, **32 tests**: the standard key; sidecar
discovery from `.iq` and from `.sigmf-data`; a missing sidecar yielding nothing;
**seven parametrised malformed values rejected** (string, null, NaN, infinity, negative,
bool, list); 0.0 accepted as baseband; unparseable JSON ignored rather than raised; a
`captures`-less document ignored; `read_iq` / `read_wav` with and without a sidecar; an
explicit sample-rate argument still winning; the sidecar supplying one when the caller
does not; survival through `preprocess`; the estimator consuming it; the blind fallback
still measurably wrong; SSB recovery > 0.99 with the sidecar and < 0.2 without; the three
anti-leakage tests; headerless V1 ingestion unchanged; V1 still on the centroid path;
hand-built containers unchanged; and caller-supplied metadata still honoured.

- Focused: **32 passed**
- Full suite: **646 passed, 1 failed** - the pre-existing `reedsolo` gap, with
  `tests/test_model_report.py` ignored (pre-existing missing `h5py`)

No existing test needed modification.

### Remaining blockers - unchanged by this entry

**CNN analog classification - dominant.** AM-SSB is still confidently mislabelled `WBFM`,
so in production most SSB captures never reach `demodulate_ssb` no matter how accurate
the carrier now is. This entry fixes the carrier, not the routing.

**Baseband AM-DSB preprocessing.** `preprocessing.py:17` DC removal still destroys the
carrier of a carrier-offset-0 AM-DSB capture; a sidecar does not help, because the
information is destroyed after ingestion.

**Blind SSB estimation** remains impossible (Entry 029) - captures without a sidecar are
no better off.

**WAV `auxi` chunks** are not parsed; a real SDR# recording would need a sidecar.

**Demodulator quality**: no audio filtering, FM limiter or de-emphasis. **NBFM**: not
implemented.

---

## Entry 031 - 2026-09-09 - Analog routing dominance: forensic investigation (READ-ONLY)

**No production code, test, dataset or model was changed.** `git status` and
`git diff --stat` are identical before and after. Frozen V1 untouched
(`d6d3f918687d0700a43e46211c3f04b9`).

Matrix: 4 analog configurations (AM-DSB, AM-SSB USB, AM-SSB LSB, WBFM) x 8 digital
controls x 2 SNR x 4 seeds = 96 captures, traced through the full path with the default
11-class checkpoint. A separate held-out run used unseen seeds 211-233 over four SNRs.

### Q1/Q2 - how the correct analog label is lost (32 analog captures)

| Mechanism | Count |
|---|---|
| **True label ABSENT from the CNN's top-3 entirely** | **11/32** |
| True label in top-3 but out-ranked and below 0.4 | 10/32 |
| **Wrong label at >= 0.4 accepted; fallback blocked by design** | **8/32** |
| Rescued by the Entry 026 analog fallback | 3/32 |

**Final correct: 3/32**, all AM-DSB at 20 dB.

Per case, whether the true label appears in the CNN top-3 at all:

| Case | present | ranks | final correct |
|---|---|---|---|
| AM-DSB | 7/8 | 3,2,2,2,-,3,1,3 | **3/8** |
| AM-SSB USB | 5/8 | -,-,3,-,3,3,3,1 | 0/8 |
| **AM-SSB LSB** | **0/8** | all absent | 0/8 |
| WBFM | 2/8 | -,1,-,-,-,1,-,- | 0/8 |

**The decisive finding: for AM-SSB LSB the CNN never emits `AM-SSB` at all** - not as
top-1, not anywhere in the top-3, in any of 8 captures. LSB is instead called `PAM4`
(0.36-0.59) or `QPSK`.

**The exact dominance condition** (from `confidence_fusion.py`): fusion accepts
`ml_label` whenever `probability >= threshold` (0.4), and the Entry 026 fallback is
gated on `rejected and classical_family == "analog-like" and ml_label not in
ANALOG_LABELS`. So a **wrong analog label above threshold is accepted unconditionally** -
AM-SSB USB is called `WBFM` at 0.437-0.486 in 3/8 captures and routed to
`demodulate_fm`. The classical `analog-like` verdict, at confidence 0.73-0.76, has no
influence whatsoever on the label. That is by design (Entry 026 deliberately left this
case alone) and it is the dominance problem.

### Q3 - the three analog types ARE separable by features already in the pipeline

Measured over the 32 analog captures:

| Feature | AM-DSB (min/med/max) | AM-SSB | WBFM | separable pairs |
|---|---|---|---|---|
| `amp_cv` | 0.22/0.26/0.30 | **0.45/0.46/0.48** | 0.07/0.14/0.22 | **DSB/SSB, SSB/WBFM** |
| `env_flat` | 0.06/0.18/0.33 | 0.02/0.07/0.15 | **0.55/0.56/0.58** | **DSB/WBFM, SSB/WBFM** |
| `fr_cv` | 0.17/0.37/0.57 | 0.32/0.61/0.92 | 0.34/0.47/0.60 | none |
| `if_std` (Hz) | 3465/7554/11686 | 6784/13560/20852 | 6875/9496/12227 | none |
| `asym_db` | 11.65/13.72/15.15 | -1.85/12.25/17.56 | 0.16/0.84/1.59 | DSB/WBFM only |

Instantaneous-frequency statistics (`fr_cv`, `if_std`) **do not separate the analog
types** - they overlap heavily. Spectral asymmetry is unstable for SSB (LSB is negative,
USB positive, and it collapses at low SNR).

Two features suffice, and both are already computed by the classical detector or trivially
derived from it:

- **`env_flat`** (envelope spectral flatness) separates **WBFM** from both AM types:
  WBFM 0.54-0.58 vs AM <= 0.33. WBFM is constant-envelope, so `|s|` carries no message.
- **`amp_cv`** separates **AM-DSB (0.22-0.30) from AM-SSB (0.45-0.48)**.

### Q4/Q5 - a conservative gate, validated on held-out seeds

Candidate, evaluated **only** when the classical detector already says `analog-like`:

    if env_flat > 0.40:  WBFM
    elif amp_cv >= 0.37: AM-SSB
    else:                AM-DSB

Held-out validation, unseen seeds 211/223/227/229/233, SNR 20/15/10/5 dB:

| Case | 20 dB | 15 dB | 10 dB | 5 dB |
|---|---|---|---|---|
| AM-DSB | 5/5 | 5/5 | 5/5 | (0 reached the gate) |
| AM-SSB USB | 5/5 | 5/5 | 5/5 | (0 reached) |
| AM-SSB LSB | 5/5 | 5/5 | 1/1 | (0 reached) |
| WBFM | 5/5 | 5/5 | 5/5 | 4/4 |

**60/60 = 100% correct** on every held-out capture the classical detector admitted.

**Digital exposure: 0 of 480** digital captures (8 modulations x samples-per-symbol
4/8/16 x 4 SNR x 5 seeds) were called `analog-like`, so **not one could reach the gate**.
Digital false-positive analog routing is **0.00%** - the Entry 027 detector is the
protective outer gate, and the type gate never sees digital signals at all.

At 5 dB the classical detector rejects AM-DSB and AM-SSB entirely, so they are simply not
routed. That is safe behaviour (no routing beats wrong routing), not a gate failure.

### MEASURED LIMITATION - the AM-DSB/AM-SSB boundary depends on modulation depth

`amp_cv` for AM-DSB is a direct function of modulation depth, which is a free generator
parameter:

| depth | 0.2 | 0.3 | 0.5 (default) | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| `amp_cv` | 0.108 | 0.143 | **0.219** | 0.299 | **0.379** | **0.419** |
| gate says | AM-DSB | AM-DSB | AM-DSB | AM-DSB | **AM-SSB (wrong)** | **AM-SSB (wrong)** |

**The gate is valid for modulation depth up to roughly 0.8 and misclassifies
heavily-modulated AM-DSB as AM-SSB above that.** The project default is 0.5, comfortably
inside the safe region, but this is an implicit assumption the rule encodes and it must
be stated in any implementation. A depth-independent DSB/SSB discriminator would need
spectral symmetry about the carrier, and Entries 029/030 established the carrier is only
reliably known from a SigMF sidecar.

### Recommended minimum safe production change - NOT implemented here

**Must change (one place):** an analog-type gate consulted **only** when
`classical_family == "analog-like"`, that supplies the analog label when the CNN cannot.
The natural seam is `fuse_modulation`, extending the Entry 026 fallback so it fires not
only when the CNN's top-1 is a rejected digital label, but also when the classical
detector asserts analog and the CNN's analog evidence is absent or contradicted. The
gate's verdict comes from the signal, not from the CNN's ranking - which is required,
because for AM-SSB LSB the CNN produces no analog label to re-rank.

**Could change later:** retraining the CNN with analog classes (the real fix - the gate
is a workaround for a model that cannot see LSB); a depth-independent DSB/SSB
discriminator using sidecar carrier information; extending analog detection below 10 dB.

**Should NOT change in the first patch:** the 0.4 threshold; the classical detector
(0/480 digital false positives - it is what makes the gate safe); dispatch; preprocessing;
the CNN or its checkpoints; the BER guard; the generators.

**Risk if the seam is chosen wrongly:** placing the gate before the threshold check, or
outside the `analog-like` guard, would let it override confident digital predictions.
Every measurement above depends on the gate being strictly downstream of the Entry 027
verdict.

### Honest summary

Entries 026-030 fixed detection (30/32 analog-like), dispatch, and the SSB carrier, but
end-to-end correctness is still **3/32** because the CNN cannot produce the right analog
label. The gate is not a tuning exercise: it exists because **no amount of re-ranking can
recover a label the model never emits** (AM-SSB LSB, 0/8). It is a stopgap until the CNN
is retrained with analog classes.

### Note on the task prompt

The instruction list was truncated mid-sentence at "instantaneous-frequency". The
investigation covered instantaneous-frequency statistics (`fr_cv`, `if_std`) among the
candidate features and found they do **not** separate the analog types; envelope
statistics do. No production change was made, since the task asked to determine the
minimum safe change and to run the forensic experiment first.

---

## Entry 032 - 2026-09-09 - Conservative analog subtype routing

Implements the Entry 031 recommendation. **No CNN retraining, no checkpoint change, no
preprocessing change, no dispatch change, no threshold change.** Frozen V1 byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

**The claim is not that the CNN got better at analog.** It did not. The claim is that
RadioFry now uses a conservative classical analog gate plus deterministic subtype routing,
and stops asking the CNN a question Entry 031 proved it cannot answer.

### Exact routing rule

Outer safety gate - unchanged Entry 027 classical detector:

    classical_family != "analog-like"  ->  existing behaviour, byte for byte

Inner subtype rule, reached only behind that gate:

    if envelope_flatness > 0.40:   AM label = WBFM
    elif amplitude_cv    >= 0.37:  AM label = AM-SSB
    else:                          AM label = AM-DSB

The CNN gets no vote once the gate opens: a confident wrong digital label (`PAM4` 0.851)
and a confident wrong analog label (`WBFM` 0.545) are both overridden, and the true label
is **not required to appear in the CNN top-k at all** - which is the point, since Entry
031 measured that AM-SSB LSB never appears.

### Files changed

| File | Change |
|---|---|
| `dsp/cyclostationary.py` | new `_envelope_flatness()`; `envelope_flatness` added to the evidence dict. **The family decision is untouched** - it is evidence only. |
| `fusion/confidence_fusion.py` | `ANALOG_ENVELOPE_FLATNESS_MAX = 0.40`, `ANALOG_AMPLITUDE_CV_SSB_MIN = 0.37`, `select_analog_subtype()`, two new optional keywords (`classical_evidence`, `classical_confidence`), new defaulted `FusionResult.analog_route` field. |
| `pipeline.py` | passes `getattr(classical, "evidence", None)` and `classical.confidence` into fusion. |

`amplitude_cv` was already computed by the detector under that exact name and formula -
it was **not** duplicated. `envelope_flatness` did not exist in production and had to be
added; its formula is the one validated in Entry 031.

**Backward compatibility:** the gate activates only when `classical_evidence` is supplied.
Every existing call site that omits it keeps its previous behaviour exactly, which is why
the Entry 026 fallback tests still pass unchanged.

### Confidence - what it actually means

`trust_score` for a subtype-routed label is the **classical detector's analog-FAMILY
confidence** (0.50-0.82 measured). It is explicitly **not** a calibrated probability for
the subtype: the subtype is a threshold decision and carries no probability. No ML
calibration system was invented. `review_recommended` is always `True` for this path - it
is a rule, not a trained model.

### MEASURED - analog validation, unseen seeds 307-331, SNR 20/15/10 dB

**55/60 = 91.7% correct end to end**, and **55/55 wherever the gate actually engaged.**

| Case | 20 dB | 15 dB | 10 dB | route |
|---|---|---|---|---|
| AM-DSB | 5/5 | 5/5 | 5/5 | `classical_subtype` |
| AM-SSB USB | 5/5 | 5/5 | 5/5 | `classical_subtype` |
| AM-SSB LSB | 5/5 | 5/5 | **0/5** | gate never opened |
| WBFM | 5/5 | 5/5 | 5/5 | `classical_subtype` |

The 5 failures are **all AM-SSB LSB at 10 dB**, where the Entry 027 classical detector
reports `QAM-like` (confidence 0.68-0.69) rather than `analog-like`, so the gate never
engages and the pre-existing CNN path runs. That is a **detector** limitation, not a
subtype-rule failure - the envelope evidence on those same captures
(`amplitude_cv` 0.468-0.479, `envelope_flatness` 0.131-0.137) would have routed them
correctly had the gate opened.

Before/after on the same class of captures (Entry 031 -> Entry 032): AM-SSB LSB was
**0/8** correct with the true label absent from every CNN top-3; it is now **10/15**
correct, entirely because the CNN is no longer consulted.

Illustrative dominance cases now corrected:
- AM-SSB LSB, seed 331, 20 dB: CNN says `PAM4` at **0.851**, no analog label anywhere in
  top-k -> routed **AM-SSB** -> `demodulate_ssb`.
- AM-SSB USB, seed 331, 20 dB: CNN says `WBFM` at **0.545** -> routed **AM-SSB**.
- WBFM, seed 331, 20 dB: CNN says `BPSK` at 0.460 -> routed **WBFM** -> `demodulate_fm`.

### MEASURED - digital false-positive analog routing

360 digital control captures (8 modulations x samples-per-symbol 4/8/16 x SNR 20/15/10 x
5 seeds):

| Metric | Result |
|---|---|
| `analog_route` values seen across all 360 | **`{"": 360}`** - the gate fired **zero** times |
| Captures routed to an analog demodulator | **6/360 = 1.67%** |

**The 6 are NOT caused by this entry, and this is stated rather than buried.** They are
GFSK at samples-per-symbol 16, where the classical detector correctly says `FSK-like`
but the **CNN emits `WBFM` at 0.52-0.90** and fusion's original rule accepts any label at
or above threshold. Verified by running the same captures with and without
`classical_evidence`: **identical labels in both columns**. This is pre-existing Entry
026-era behaviour on the digital path, which this task explicitly required not to change.

So: **the Entry 032 gate contributes 0/360 digital false positives.** The system-level
figure is 1.67% and comes from a path this entry was forbidden to touch. Pinned by
`test_the_gate_never_fires_when_the_classical_family_is_digital`.

### Tests

New `tests/test_analog_subtype_routing.py`, **57 tests, 1 skipped** (the skip is the
AM-SSB LSB 10 dB case the detector declines - it self-documents rather than pretending).
Covers all 12 required cases: the four analog subtypes end to end into their
demodulators; digital controls never routed; both threshold boundaries from either side;
flatness checked before amplitude_cv; confident wrong digital labels (4 parametrised) and
confident wrong analog labels (2) unable to override; the true label absent from top-k;
full-rate routing preserved; null symbol rate still working; analog BER still
unavailable; the new evidence field present without changing the family decision;
backward compatibility with and without evidence; and the pre-existing GFSK leak pinned
as not ours.

**Three prior tests were deliberately updated**, all because the evidence dict
legitimately grew a documented fourth key:
- `test_dsp_fusion.py::test_classical_estimator_returns_a_structured_result` and
  `test_classical_analog_detection.py::test_the_evidence_dictionary_still_exposes_the_
  three_features` asserted an exact 3-key set; both now assert the original three are a
  **subset** - a check about nothing being *removed*, which is what they actually guard.
- `test_pipeline_integration.py::test_pipeline_open_set_rejection_skips_demodulation`
  failed because it monkeypatches the classical result with a stub lacking `.evidence`.
  **That was fixed in production, not in the test**: `pipeline.py` now uses
  `getattr(classical, "evidence", None)`, so a classical result without evidence simply
  does not trigger subtype routing. The test passes unmodified.

- Focused: **57 passed, 1 skipped**
- Full suite: **703 passed, 1 failed, 1 skipped** - the one failure is the pre-existing
  `reedsolo` gap, with `tests/test_model_report.py` ignored for the pre-existing missing
  `h5py`. Both environmental, both pre-date this work, neither hidden.

### Known limitations

1. **AM-SSB LSB below ~15 dB is not routed** - the classical detector calls it
   `QAM-like`. Detector work, not routing work.
2. **The AM-DSB/AM-SSB boundary rides on modulation depth** (Entry 031): correct to a
   depth of about 0.8; at depth >= 0.9 an AM-DSB capture reaches `amplitude_cv` 0.379 and
   is read as AM-SSB. The project default is 0.5. Documented in `select_analog_subtype`.
3. **6/360 digital captures still route to WBFM** via the pre-existing confident-CNN
   path (GFSK at sps=16). Not introduced here, not fixed here.
4. Analog detection still fails below ~10 dB for the AM schemes.
5. `preprocess` still removes the baseband AM-DSB carrier, so AM-DSB at carrier offset 0
   recovers ~0.00 even when routed correctly.
6. The gate is a stopgap. The real fix is retraining the CNN with analog classes.

### V1 integrity

`d6d3f918687d0700a43e46211c3f04b9` (first 32 characters of the full SHA-256 over the
concatenated name-sorted 40 `.iq` captures) - **unchanged**.

---

## Entry 033 - 2026-09-09 - Final synthetic end-to-end benchmark (MEASUREMENT ONLY)

**No production code was changed.** Frozen V1 verified byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures). Full suite unchanged at 703 passed.

### Methodology

350 captures through the complete unmodified pipeline: ingestion -> preprocessing ->
parameter estimation -> classical detector -> CNN -> fusion -> dispatch -> demodulation.

- **Digital:** 8 classes x 5 SNR (20/15/10/5/0 dB) x 5 seeds = **200**, samples-per-symbol 8.
- **Analog:** AM-DSB, AM-SSB USB, AM-SSB LSB, WBFM x 5 SNR x 5 seeds, with SSB run
  **twice** (with and without legitimate SigMF recorder metadata) = **150**.
- **Seeds 401/409/419/421/431 - fresh, unused by any prior entry.**
- **No leakage:** the production checkpoint `modulation_cnn.pt` is the 11-class RadioML
  model, trained on RadioML 2016.10a, never on RadioFry synthetic captures. No ground
  truth, expected modulation, or post-hoc correction enters the pipeline; ground truth is
  used only to score afterwards. SigMF metadata is supplied only in the variant that
  explicitly represents an external recorder.

### 1. Digital classification (200)

| Metric | Result |
|---|---|
| fused top-1 | **118/200 = 59.0%** |
| CNN top-3 | **192/200 = 96.0%** |
| top-1 at >= 10 dB | 85/120 = 70.8% |
| **top-3 at >= 10 dB** | **120/120 = 100.0%** |
| family level (PSK/FSK/QAM groups) | 153/200 = 76.5% |

Per class (correct/5 at each SNR, 20 -> 0 dB):

| Class | 20 | 15 | 10 | 5 | 0 | total |
|---|---|---|---|---|---|---|
| BPSK | 5 | 5 | 5 | 5 | 5 | **25/25** |
| CPFSK | 5 | 5 | 5 | 5 | 5 | **25/25** |
| QPSK | 5 | 5 | 5 | 5 | 0 | 20/25 |
| 8PSK | 5 | 5 | 5 | 5 | 0 | 20/25 |
| QAM64 | 3 | 2 | 2 | 1 | 0 | 8/25 |
| GFSK | 2 | 2 | 2 | 1 | 0 | 7/25 |
| QAM16 | 2 | 2 | 2 | 1 | 0 | 7/25 |
| PAM4 | 3 | 2 | 1 | 0 | 0 | 6/25 |

Dominant confusions: **PAM4 -> Unclassified 19**, **GFSK -> CPFSK 18**, QAM16 ->
Unclassified 11, QAM64 -> Unclassified 11, QAM16 <-> QAM64 13.

### 2. Analog classification (150)

| Case | 20 | 15 | 10 | 5 | 0 | total |
|---|---|---|---|---|---|---|
| AM-DSB | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 15/25 |
| AM-SSB USB | 10/10 | 10/10 | 10/10 | 0/10 | 0/10 | 30/50 |
| AM-SSB LSB | 10/10 | 10/10 | **0/10** | 0/10 | 0/10 | 20/50 |
| WBFM | 5/5 | 5/5 | 5/5 | 1/5 | 0/5 | 16/25 |

Overall **81/150 = 54.0%**; **at >= 10 dB, 80/90 = 88.9%**. Below 10 dB: 1/60 correct
but **56/60 rejected** rather than misclassified.

Metadata does **not** change classification (25/50 with, 25/50 without) - as expected,
carrier metadata affects demodulation, not routing.

### 3. Routing

| Metric | Result |
|---|---|
| analog routed via `classical_subtype` | 81/150 |
| **correct label when the gate engaged** | **81/81 = 100.0%** |
| analog sent to the WRONG analog demodulator | 1 |
| **DIGITAL routed as analog (sps=8)** | **0/200 = 0.0%** |
| digital -> wrong digital label | 37/200 = 18.5% |

Every analog failure is a *gate-did-not-engage* failure, never a wrong subtype.

### 4. GFSK -> WBFM: SYSTEMATIC, and driven by oversampling

Entry 032 saw 6 cases at samples-per-symbol 16. Swept properly (25 captures per sps,
5 SNR x 5 seeds):

| samples/symbol | GFSK accepted as WBFM | CNN confidence range |
|---|---|---|
| 4 | **0/25** | - |
| 8 | **0/25** | - |
| 16 | **8/25 = 32%** | 0.53 - 0.97 |
| 32 | **19/25 = 76%** | **0.57 - 1.00** |

**This is systematic, not isolated**, and it worsens monotonically with oversampling. In
every case `analog_route` is `""` - the Entry 032 gate is **not** involved. The classical
detector correctly reports `FSK-like` in the high-SNR cases; fusion's original rule
accepts any CNN label at or above 0.4, so a **0.999-confidence** WBFM prediction wins.

Physically the CNN is not being absurd: heavily oversampled GFSK *is* a smooth
continuous-phase frequency modulation, which is what WBFM is. The defect is that fusion
lets a confident CNN override a confident, correct, independent family verdict.

V1 uses samples-per-symbol 8, where the rate is 0/25 - which is why the frozen benchmark
never exposed this. Real captures are often heavily oversampled.

### 5. AM-SSB LSB: a classification (detector) problem, not demodulation

| SNR | USB gate engaged | USB correct | LSB gate engaged | LSB correct | LSB classical family |
|---|---|---|---|---|---|
| 20 dB | 10/10 | 10/10 | 10/10 | 10/10 | `analog-like` x10 |
| 15 dB | 10/10 | 10/10 | 10/10 | 10/10 | `analog-like` x10 |
| **10 dB** | **10/10** | **10/10** | **0/10** | **0/10** | **`QAM-like` x10** |
| 5 dB | 0/10 | 0/10 | 0/10 | 0/10 | `QAM-like` x10 |
| 0 dB | 0/10 | 0/10 | 0/10 | 0/10 | `QAM-like` x10 |

- **Highest SNR where LSB reliably reaches analog routing: 15 dB.** Transition between
  15 and 10 dB, and it is sharp (10/10 -> 0/10) rather than gradual.
- **USB behaves differently**: it survives to 10 dB, failing between 10 and 5 dB. USB
  therefore has roughly one SNR step more margin than LSB.
- **It is a classification problem.** Whenever the gate engaged, LSB was labelled and
  demodulated correctly (recovery 0.9951 / 0.9847 with metadata). The Entry 027
  `frequency_cv` detector is what fails at 10 dB, not the subtype rule or the
  demodulator.

### 6. Digital demodulation / BER

Demodulation available for 149/200 (74.5%); median BER over all demodulated 0.0762.

Median BER **where the label was correct** (isolates demodulator quality from routing):

| Class | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| BPSK | **0.0000** | **0.0000** | **0.0000** | 0.0068 | 0.0781 |
| QPSK | **0.0000** | **0.0000** | 0.0020 | 0.0522 | - |
| 8PSK | **0.0000** | 0.0020 | 0.0479 | 0.1833 | - |
| QAM16 | **0.0000** | 0.0076 | 0.0786 | 0.2258 | - |
| QAM64 | 0.0156 | 0.1051 | 0.2298 | 0.3205 | - |
| CPFSK | 0.0117 | 0.0117 | **0.4963** | 0.4801 | 0.4803 |
| GFSK | **0.4772** | **0.4730** | **0.4155** | 0.4646 | - |
| PAM4 | **-** | **-** | **-** | **-** | **-** |

Two demodulation defects independent of classification:
- **GFSK sits at chance (~0.47) even when correctly labelled** - it is routed to the
  order-2 FSK demodulator, which does not account for the Gaussian frequency pulse.
- **PAM4 never demodulates at all** - it still has no dispatch route (open since Entry 019).
- CPFSK collapses at 10 dB and below, consistent with the Entry 014 negative result on
  low-SNR FSK symbol-rate estimation.

### 7. Analog message recovery (median correlation, analog demod only)

| Case | metadata | 20 dB | 15 dB | 10 dB | 5 dB |
|---|---|---|---|---|---|
| AM-DSB | - | 0.9456 | 0.8513 | 0.6675 | - |
| AM-SSB USB | **no** | **-0.0028** | **0.0008** | **-0.0162** | - |
| AM-SSB USB | **yes** | **0.9951** | **0.9847** | **0.9536** | - |
| AM-SSB LSB | **no** | **-0.0030** | **0.0029** | - | - |
| AM-SSB LSB | **yes** | **0.9951** | **0.9847** | - | - |
| WBFM | - | 0.8943 | 0.7446 | 0.5215 | 0.2859 |

This is the cleanest confirmation yet of Entries 029/030: **SSB recovery is ~0.00 blind
and ~0.99 with legitimate recorder metadata.** The demodulator is correct; the blind
carrier is not recoverable.

**No BER was fabricated for any analog capture** (0/150); every analog row carries
`unavailable_analog_no_transmitted_bits`.

### 8. Parameter estimation

- **Digital symbol rate within 1%: 161/200 = 80.5%** (35/40 at 20 dB, 30/40 at 0 dB);
  median relative error 0.000.
- **Digital carrier** (true 0 Hz): median |error| **147.2 Hz**, max 1051.2 Hz.
- **Analog carrier**, no metadata: median |error| **2147.0 Hz** (AM-DSB 1655.8,
  USB 1904.0, LSB 3463.4, WBFM 1676.6). **With metadata: 0.0 Hz.**
- **SNR estimate is biased low and compressed**: true 20 -> 12.2, 15 -> 11.6, 10 -> 9.8,
  5 -> 6.4, 0 -> 3.2 dB. Usable as an ordering, not as an absolute.

### 9. Rejection / unknown behaviour

| | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB | total |
|---|---|---|---|---|---|---|
| digital Unclassified | 4/40 | 5/40 | 6/40 | 11/40 | 19/40 | 45/200 = 22.5% |
| analog Unclassified | 0/30 | 0/30 | 4/30 | 27/30 | 29/30 | 60/150 = 40.0% |

System-wide over 350 captures: **correct + confident 199 (56.9%)**, **rejected 105
(30.0%)**, **false confident 46 (13.1%)** - 37 digital, 9 analog. Rejection rises with
noise as it should; the system prefers refusing to guessing at low SNR.

### Assessment

Headline "correct and confident" is 56.9%, which looks poor, but it decomposes into a
small number of **specific, individually addressable defects** rather than a model that
cannot see the signal:

- **CNN top-3 at >= 10 dB is 100%.** The representation carries the right answer; the
  loss is in the decision layers.
- PAM4 (-19 captures) and QAM (-22) are lost mostly to the 0.4 rejection threshold, and
  PAM4 additionally has no demodulator route at all.
- GFSK -> CPFSK (-18) is a *within-family* confusion.
- Every analog failure is the classical detector declining at low SNR - the subtype rule
  was 81/81 whenever it ran.

### RECOMMENDATION: TARGETED FIX (not FREEZE, not RETRAIN)

**Do not retrain.** Retraining is only justified when the model lacks the information, and
top-3 = 100% at >= 10 dB proves it does not. A retraining cycle would not fix a missing
dispatch route, a fusion override rule, or a detector threshold.

**Do not freeze either.** One defect is severe enough to block a freeze: **GFSK -> WBFM
at 76% with 0.999 confidence under heavy oversampling**, which routes a digital capture
into an analog demodulator and produces a confidently wrong answer on a real-world
capture geometry.

Four targeted fixes, in priority order, none requiring training:

1. **Fusion: refuse a CNN analog label when the classical detector asserts a digital
   family.** Fixes GFSK -> WBFM (76% -> expected 0% at sps=32). Symmetric with the Entry
   032 gate, which already refuses the CNN in the opposite direction. **This is the one
   that must land before any freeze.**
2. **Add a PAM4 dispatch route.** 0/25 demodulated today; purely a missing route.
3. **GFSK demodulation.** ~0.47 BER even when correctly labelled - the order-2 FSK
   demodulator ignores the Gaussian pulse.
4. **Extend classical analog detection to 10 dB for AM-SSB LSB.** Would recover 10/50 LSB
   captures; the sharp 15 -> 10 dB cliff suggests a threshold, not a fundamental limit.

Items 2-4 are quality improvements; item 1 is a safety defect. Re-benchmark after item 1
before deciding on a freeze.

---

## Entry 034 - 2026-09-09 - Fusion safety gate: a digital family blocks CNN analog labels

Fixes the Entry 033 defect that blocked a freeze. **No CNN retraining, no checkpoint,
preprocessing, detector-threshold, symbol-rate, subtype-threshold or demodulator change.**
Frozen V1 byte-identical (`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

**The claim is not that the CNN improved.** It did not. The claim is that fusion now
prevents a CNN analog prediction from overriding an independent classical digital-family
verdict.

### Exact rule

Mirror of the Entry 032 gate, inserted **after** it and **before** the Entry 026 fallback:

    if ml_label in ANALOG_LABELS
       and classical_family in DIGITAL_FAMILIES          # {PSK-like, FSK-like, QAM-like}
       and classical_confidence >= 0.5:
           block the analog label

`DIGITAL_FAMILIES` deliberately **excludes `"unknown"`** - that is the detector's
non-verdict, not evidence, and must not override anything. The rule fires only on analog
labels, so digital/digital disagreements (e.g. `QAM16` under an `FSK-like` verdict) are
untouched: this is not "classical always wins".

`DIGITAL_FAMILY_MIN_CONFIDENCE = 0.5` is measured, not guessed. Over 144 digital captures
(8 classes x samples-per-symbol 8/16/32 x SNR 20/10/0 dB) the detector scored
**FSK-like 0.828-1.000** and **QAM-like 0.557-0.709**, so every real digital verdict
clears 0.5 while `unknown` (0.2) does not.

### Fallback when the analog label is blocked

Two-step, in this order:

1. **Retain the CNN's own highest-ranked digital alternative** - but only if it clears the
   *same* 0.4 acceptance threshold every other prediction must clear. No new threshold is
   introduced and no label is invented; the ranking is the CNN's.
2. **Otherwise return `Unclassified`.**

The family is **never** turned into a subtype: an `FSK-like` verdict does not become
`CPFSK`. Rationale, stated plainly: a wrong analog demodulation puts the capture in the
wrong signal domain entirely, whereas a rejection costs only an answer. Rejection is the
safe default, and step 1 only applies when the CNN independently supports a digital label
at normal strength.

The choice was evidence-driven: in **all 31** measured GFSK captures where the CNN's top-1
was analog, `GFSK` was already present in the CNN's own top-3 - so step 1 has real
material to work with rather than being theoretical.

### Files changed

| File | Change |
|---|---|
| `fusion/confidence_fusion.py` | `DIGITAL_FAMILIES`, `DIGITAL_FAMILY_MIN_CONFIDENCE`, `_highest_ranked_digital()`, the guard block, new defaulted `FusionResult.digital_family_block`. |

`pipeline.py` needed **no change** - it already passes `classical_confidence` and
`ranked_alternatives` (Entries 026, 032).

### MEASURED - the Entry 033 regression is closed

600 digital captures: 8 classes x samples-per-symbol 8/16/32 x 5 SNR x 5 seeds
(401-431).

| GFSK -> WBFM | BEFORE | AFTER |
|---|---|---|
| samples-per-symbol 16 | 8/25 (conf 0.53-0.97) | **0/25** |
| samples-per-symbol 32 | 19/25 (conf 0.57-1.00) | **0/25** |
| **combined high-oversampling** | **27/50** | **0/50** |

Target was 0/50. **Met.**

### MEASURED - digital control matrix, before vs after

| Metric | BEFORE | AFTER |
|---|---|---|
| digital top-1 | 33.5% | **34.3%** |
| digital top-3 (CNN, untouched) | 71.7% | 71.7% |
| **digital false-analog routing** | **128/600** | **1/600** |
| digital rejection rate | 21.8% | **41.8%** |

- **False-analog routing fell 128 -> 1 (99.2% reduction).**
- Top-1 went slightly **up**: 127 fused labels changed, **all 127 were analog before**;
  5 became correct and **0 became wrong**.
- Rejection rose 20 percentage points. That is the intended price and it is the right
  trade: of the 136 blocked captures, 136 -> `Unclassified` (and 7 -> a digital label:
  GFSK 3, BPSK 2, PAM4 2). Every one of those was previously heading for an analog
  demodulator.

**The single residual false positive**: CPFSK, sps=32, 15 dB, seed 421 - the classical
detector returned **`unknown` @ 0.200**, which is deliberately not a blocking verdict, so
the CNN's `WBFM` @ 0.685 stands. Blocking on `unknown` would mean "no evidence beats the
CNN", which is outside this rule's scope. Recorded, not patched.

### MEASURED - analog regression: Entry 032 fully preserved

100 analog captures, before vs after:

| Case | BEFORE | AFTER |
|---|---|---|
| AM-DSB | 15/25 | **15/25** |
| AM-SSB USB | 15/25 | **15/25** |
| AM-SSB LSB | 10/25 | **10/25** |
| WBFM | 16/25 | **16/25** |
| overall | 56/100 | **56/100** |

Gate engaged 56/100; **correct when engaged 56/56**. Byte-for-byte identical - the two
protections coexist, as required.

### Tests

New `tests/test_fusion_digital_family_guard.py`, **28 tests**: the explicit family set and
measured confidence floor; the full 3x3 blocking matrix (FSK/PSK/QAM-like x
WBFM/AM-DSB/AM-SSB); blocking at maximum CNN confidence; weak classical verdict and
`unknown` not blocking; digital alternative retained above threshold and rejected below
it; no digital alternative -> rejection; family never turned into a subtype; the retained
alternative must itself be digital; existing digital accept/reject unchanged;
digital/digital cross-family disagreement untouched; **Entry 032 gate still firing in both
its directions**; Entry 026 fallback intact; and the GFSK sps=16/32 regression driven
through the real classical detector and CNN.

**One prior test was deliberately updated.**
`test_analog_subtype_routing.py::test_the_gate_never_fires_when_the_classical_family_is_digital`
was written in Entry 032 to record the GFSK -> WBFM leak as pre-existing and
*deliberately unfixed*. Entry 034 fixes it, so the test would otherwise assert the bug. It
keeps its original purpose (the subtype gate must not fire on a digital family) and now
additionally pins the Entry 034 outcome.

- Focused: **28 passed**
- Full suite: **731 passed, 1 failed, 1 skipped** - the failure is the pre-existing
  `reedsolo` gap, with `tests/test_model_report.py` ignored for the pre-existing missing
  `h5py`. Both environmental, neither hidden.

### Remaining limitations

1. **`unknown` classical verdicts do not block** - 1/600 residual (CPFSK sps=32).
   Deliberate: a non-verdict must not override the CNN.
2. **Rejection rose to 41.8%** on the sps 8/16/32 matrix. Safe, but it means more captures
   return no answer; the underlying cause is CNN accuracy at high oversampling, not
   fusion.
3. Entry 033's other three items are untouched and still open: **PAM4 has no dispatch
   route**, **GFSK demodulates at ~0.47 BER even when correctly labelled**, and
   **AM-SSB LSB is not detected at 10 dB**.
4. The guard cannot recover a correct label the CNN never ranks; it can only prevent the
   wrong domain.

### V1 integrity

`d6d3f918687d0700a43e46211c3f04b9` - **unchanged**, 40 captures.

---

## Entry 035 - 2026-09-09 - PAM4 demodulation and dispatch

Closes Entry 033 item 2. **No CNN retraining, no checkpoint, fusion, preprocessing,
symbol-rate, analog or GFSK change.** Frozen V1 byte-identical
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

### Did a PAM4 demodulator already exist?

**No.** `src/radiofry/decoding/demodulators/` held `psk_demod`, `qam_demod`, `fsk_demod`
and `analog_demod` only; no PAM implementation existed anywhere, reachable or otherwise.
`dispatch.demodulate_capture` had no PAM4 branch, so a PAM4 label fell through to
`"No demodulator is registered for PAM4."` - exactly what Entry 033 measured as 0/25
demodulated.

### Files changed

| File | Change |
|---|---|
| `decoding/demodulators/pam_demod.py` | **new** - `demodulate_pam()`, 50 lines |
| `decoding/demodulators/dispatch.py` | one import and one `elif modulation == "PAM4"` branch |

Nothing else. `pipeline.py` and `evaluation/harness.py` needed no change - PAM4 now flows
through the existing digital path and the existing BER scorer.

### Symbol/bit mapping - taken from the generator, not invented

`synthetic_gen/v1/modulation.py:constellation_for` builds PAM4 as
`levels = [-3, -1, 1, 3]` normalised to unit average power (divide by sqrt(5)), and
`bits_to_symbol_indices` packs bits as **natural binary, MSB-first** - the project-wide
convention since Entry 001.

**PAM4 is NOT Gray coded in this project.** Index 0..3 maps to levels -3, -1, +1, +3 in
ascending order, so the bit pairs are 00, 01, 10, 11 respectively. The demodulator
inverts precisely that, and a test round-trips it against `constellation_for` and
`bits_to_symbol_indices` directly rather than restating the assumption.

### Demodulation approach

Two quantities are estimated **from the samples alone** - neither is supplied and no
ground truth is consulted (the function signature is exactly `(samples, order)`):

1. **Constellation axis.** PAM is a real one-dimensional constellation, so its samples
   lie on a line through the origin. That line has 180-degree symmetry, which makes
   `angle(mean(x^2)) / 2` a valid estimate of its rotation - squaring folds the two
   opposite lobes onto one. The samples are derotated by it before projection.
2. **Scale.** Upstream stages deliver unit-RMS symbols while the grid has average power
   `mean(levels^2) = 5`, so the samples are rescaled onto the grid. This is the same
   defect that made QAM slice everything onto the innermost pair before Entry 007; it was
   avoided here rather than repeated.

Then nearest-level slicing, and the index emitted MSB-first as two bits.

### Timing

**No new timing framework.** PAM4 is not in `_FSK_LABELS` and not in `ANALOG_LABELS`, so
it uses the existing `_linear_timing_offset` search in `demodulate_capture`, unchanged.
No PAM4-specific timing was needed.

Observed: the chosen offset varies across seeds (0, 2, 3, 5, 7) yet BER stays 0.00000 at
20 dB, because rectangular pulses make any offset inside the symbol equivalent.

### MEASURED - BER by SNR (10 seeds, samples-per-symbol 8)

| SNR | demod available | median BER (true Rs) | median BER (**estimated** Rs) |
|---|---|---|---|
| noiseless | 10/10 | **0.00000** | 0.00000 |
| 20 dB | 10/10 | **0.00000** | 0.00000 |
| 15 dB | 10/10 | 0.00024 | 0.00024 |
| 10 dB | 10/10 | 0.02051 | 0.02051 |
| 5 dB | 10/10 | 0.13330 | 0.13330 |
| 0 dB | 10/10 | 0.26343 | 0.26343 |

**Demodulation success 10/10 at every SNR** (was 0/25 in Entry 033). BER is monotone in
SNR, and the estimated symbol rate produces **identical** BER to the true one - the
existing estimator already handles PAM4.

**Amplitude-scale robustness**: gains of 1e-4 to 1e+4 give a maximum BER delta of
**0.000000** - exactly scale-invariant across eight orders of magnitude.

### MEASURED - no regression against the other digital classes (20 dB, 10 seeds)

| Class | median BER |
|---|---|
| BPSK | 0.00000 |
| QPSK | 0.00000 |
| QAM16 | 0.00000 |
| **PAM4** | **0.00000** |
| QAM64 | 0.01709 |

PAM4 lands on par with the best-performing existing demodulators.

### Dispatch verification

`PAM4 -> dispatch -> demodulate_pam -> bits -> harness BER scorer` is exercised
end-to-end by test, including `_score_report` returning `ber_status="ok"` with
`ber_strict <= 0.01`. PAM4 still requires a symbol rate like every other digital label,
BPSK/QPSK/QAM16/CPFSK still reach their own demodulators, an unregistered label
(`PAM8`) is still rejected, and a misrouted non-PAM4 capture degrades to a bad BER rather
than raising.

### Tests

New `tests/test_pam4_demodulation.py`, **44 tests**: mapping pinned against the generator;
the shared `DemodulationResult` contract; noiseless through 5 dB with per-SNR bounds;
five seeds; BER monotonicity; scale invariance over six gains; rotation behaviour;
the dispatch route; symbol-rate requirement; other labels not leaking into PAM4;
misrouting not crashing; harness BER integration; and the Entry 034 guard still blocking
analog labels while still accepting a PAM4 prediction under a `QAM-like` verdict.

**One test was corrected during development, and the correction matters.** It initially
asserted full rotation invariance. Measurement showed that is physically impossible:
`index == true` or `index == 3 - true`, always exactly one of them, never anything
between. The test now asserts what is actually true - the axis is recovered under any
rotation - and a separate test pins that polarity is correct when the capture is not
rotated, so the ambiguity cannot become an excuse for a sign error.

- Focused: **44 passed**
- Full suite: **775 passed, 1 failed, 1 skipped** - the failure is the pre-existing
  `reedsolo` gap, with `tests/test_model_report.py` ignored for the pre-existing missing
  `h5py`. Neither hidden.

### Known limitations

1. **180-degree polarity ambiguity.** A blind receiver cannot distinguish level -3 from
   +3 without an external reference (differential coding, a pilot, or a known preamble).
   Measured: under rotation the levels are recovered exactly, but possibly inverted
   (BER 1.0 rather than 0.0). BPSK has the identical ambiguity and `demodulate_psk` does
   not resolve it either - this is a **shared architectural gap**, documented rather than
   solved here, since fixing it properly means adding phase/differential reference
   handling across every digital demodulator.
2. Only `order=4` is supported; other PAM orders raise. No PAM2/PAM8 exists in the
   generator, so a wider implementation would be untested speculation.
3. PAM4 classification accuracy is unchanged - Entry 033 measured 6/25 fused top-1, with
   19/25 lost to the 0.4 rejection threshold. **This entry fixes demodulation, not
   classification**: PAM4 now demodulates correctly *when it is correctly labelled*.
4. Entry 033 items 3 and 4 remain open: GFSK demodulates at ~0.47 BER even when correctly
   labelled, and AM-SSB LSB is not detected at 10 dB.

### V1 integrity

`d6d3f918687d0700a43e46211c3f04b9` - **unchanged**, 40 captures.

---

## Entry 036 - 2026-09-09 - GFSK demodulation: the premise was wrong (NEGATIVE RESULT)

**No production code was changed.** The forensic investigation the task required found
that the ~0.47 GFSK BER is **not a GFSK demodulation defect**. Frozen V1 verified
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures); suite unchanged at 775 passed.

### CORRECTION to the Entry 033 attribution

Entry 033 recorded "GFSK sits at ~0.47 BER even when correctly labelled - the order-2 FSK
demodulator ignores the Gaussian pulse". **Measured against the actual implementation,
that attribution is wrong on both halves.**

**Measured GFSK BER with the TRUE symbol rate** (median, 5 seeds):

| samples/symbol | noiseless | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|---|
| 4 | 0.0044 | **0.0049** | 0.0088 | 0.0420 | 0.1148 | 0.3273 |
| 8 | 0.0117 | **0.0137** | 0.0137 | 0.0430 | 0.1408 | 0.4360 |
| 16 | 0.0157 | **0.0157** | 0.0157 | 0.0274 | 0.1096 | 0.4403 |
| 32 | 0.0314 | **0.0275** | 0.0275 | 0.0510 | 0.0902 | 0.4314 |

CPFSK/BFSK through the identical code path, same conditions: 0.0000-0.0440 noiseless,
0.0049-0.0275 at 20 dB. **GFSK is in the same band as CPFSK, and noiseless at
samples-per-symbol 8 GFSK is actually better** - 12 bit errors against CPFSK's 56.

The existing discriminator is therefore adequate for GFSK. Nothing about it "ignores the
Gaussian pulse" in a way that costs accuracy at these settings.

### ROOT CAUSE - symbol-rate estimation, and it is not GFSK-specific

Same captures, same demodulator, only the symbol rate differing:

| Modulation | sps | SNR | BER with **true** Rs | BER with **estimated** Rs | estimated Rs | true Rs |
|---|---|---|---|---|---|---|
| GFSK | 8 | 20 dB | **0.0137** | **0.5108** | 1880 | 25000 |
| GFSK | 16 | 20 dB | **0.0157** | **0.5000** | 854 | 12500 |
| GFSK | 32 | 20 dB | **0.0275** | **0.5000** | 415 | 6250 |
| **BFSK** | 16 | 20 dB | **0.0157** | **0.5000** | 903 | 12500 |
| **BFSK** | 32 | 20 dB | **0.0275** | **0.5000** | 415 | 6250 |
| BFSK | 8 | 20 dB | 0.0117 | 0.0117 | 25000 | 25000 |

**CPFSK/BFSK collapses identically** once samples-per-symbol rises above 8 or SNR drops.
The only configuration where the estimator works is BFSK at samples-per-symbol 8 and high
SNR - which is exactly the V1 configuration, and exactly why this was invisible until the
Entry 033 benchmark widened the matrix.

End-to-end confirmation (GFSK, correctly routed to `2FSK`, Entry 034 holding):

| sps | SNR | seed | estimated Rs | true Rs | BER (pipeline) | BER (true Rs) |
|---|---|---|---|---|---|---|
| 16 | 20 dB | 401 | 854 | 12500 | **0.5294** | **0.0020** |
| 16 | 10 dB | 401 | 8594 | 12500 | 0.4986 | 0.0215 |
| 32 | 20 dB | 401 | 342 | 6250 | 0.6154 | 0.0157 |
| 32 | 10 dB | 401 | 342 | 6250 | 0.6154 | 0.0235 |

This is the Entry 014 negative result ("low-SNR FSK symbol-rate estimation: no safe small
fix") shown to be **far broader than recorded**: it is not confined to low SNR, and not
confined to one FSK variant.

### A real but small GFSK-specific timing weakness - quantified, not fixed

`_fsk_timing_offset` minimises the variance of instantaneous frequency inside each symbol
window. For rectangular-pulse BFSK the frequency is piecewise constant, so the correct
offset gives a sharp, near-zero minimum. GFSK's Gaussian pulse makes the frequency vary
continuously, so the minimum is shallow and can land on a symbol-straddling offset.

Measured at 20 dB with the true symbol rate, 10 seeds:

| sps | GFSK catastrophic failures | CPFSK/BFSK failures | offsets chosen on the failures |
|---|---|---|---|
| 4 | 0/10 | 0/10 | - |
| 8 | **1/10** | 0/10 | 7 (best is 0) |
| 16 | 0/10 | 0/10 | - |
| 32 | **2/10** | 0/10 | 28 and 31 (best is 0) |

Brute-forcing the offset recovers those seeds to BER 0.039-0.043, so the information is
present and only the offset choice is wrong. **3 of 40 GFSK cases; 0 of 40 CPFSK cases.**

**A candidate replacement was implemented and rejected on measurement.** The idea was to
pick the offset whose decimated frequency samples have the widest spread (the eye
opening). Measured, it is far worse:

| sps | existing criterion fails | candidate fails (GFSK) | candidate fails (BFSK) |
|---|---|---|---|
| 4 | 0/10 | **5/10** | 5/10 |
| 8 | 1/10 | **7/10** | 6/10 |
| 16 | 0/10 | **8/10** | 4/10 |
| 32 | 2/10 | **5/10** | 3/10 |

The existing criterion is better everywhere. It was kept.

### Why no production change was made

1. The headline defect (~0.47 BER) is **symbol-rate estimation**, which this task's scope
   is GFSK *demodulation*, and which affects CPFSK identically - a GFSK-specific change
   would fix nothing.
2. The GFSK demodulator is already within a factor of ~2 of CPFSK and better than it
   noiseless. There is no factor-of-40 defect to remove.
3. The one timing improvement that could be justified was tested and **regresses badly**.
   Building a proper matched-filter or Gardner timing loop is "a general-purpose modem",
   which the task explicitly forbids.

Changing production to chase a 3/40 timing edge case while the real 0.47 sits in the
symbol-rate estimator would have been motion without progress.

### Generator facts recorded (used for understanding only, never at runtime)

`gaussian_frequency_pulse(bt, sps, span_symbols=4)`, `sigma = sqrt(ln2)/(2*pi*BT)`,
normalised to **unit area** so the accumulated phase per symbol - and therefore the
modulation index - matches unshaped CPFSK. Default `DEFAULT_GAUSSIAN_BT = 0.3`, span
4 symbols (33 taps at sps=8), h = 0.5, deviation = h*Rs/2. Bits map to tones
`(2k - (M-1)) * deviation`, phase by cumulative sum. The Gaussian filter **does**
introduce inter-symbol memory across roughly 4 symbols; measured per-symbol phase-step
separation drops from 13.6 sigma (BFSK) to 4.8 sigma (GFSK), which is the ISI cost and is
still comfortably decodable.

**BT sensitivity** (sps=8, 20 dB, true Rs): BT 0.2 -> 0.1095, BT 0.3 -> 0.0137,
BT 0.5 -> 0.0117. Degrades smoothly with tighter filtering, as expected. No BT value is
hard-coded anywhere in production; the discriminator is BT-agnostic.

**Amplitude-scale**: BER 0.0137 at gains 1e-4, 1.0 and 1e+4 - exactly invariant, because
the discriminator uses phase only.

### Classification/routing success vs demodulation success - the distinction that matters

- **Routing (Entry 034): working.** In every end-to-end GFSK case traced here, no capture
  was routed to an analog demodulator; all reached `2FSK`. Verified explicitly.
- **Classification: partly working.** The CNN still calls GFSK `WBFM` at high
  oversampling; Entry 034 converts that into `GFSK` or `Unclassified`, never `WBFM`.
- **Demodulation: capable but starved.** Given a correct symbol rate the demodulator
  returns 0.002-0.031 BER. The pipeline does not give it one, so it returns ~0.50.

**Fixing classification and routing did not and cannot fix this**, and no GFSK
demodulation change will either.

### Recommendation

The next task should be **FSK symbol-rate estimation**, scoped to CPFSK *and* GFSK across
samples-per-symbol 4-32, not GFSK demodulation. Evidence: with the true rate the
demodulator already delivers 0.002-0.031 BER; with the estimated rate it delivers ~0.50.
Entry 014 closed this as a negative result at low SNR only - that conclusion needs
revisiting with this wider evidence, because the failure is now shown at 20 dB.

### Repository state

Read-only. `git diff --stat -- src/` shows the four Entry 035 files unchanged at
154 insertions; full suite **775 passed, 1 failed, 1 skipped** (pre-existing `reedsolo`;
`test_model_report.py` ignored for the pre-existing missing `h5py`).
V1 hash `d6d3f918687d0700a43e46211c3f04b9` - **MATCH**.

---

## Entry 037 - 2026-09-09 - FSK symbol-rate estimation (NEGATIVE RESULT)

**No production code was changed.** The investigation found the failure is not a
peak-selection bug that a small patch can fix: for GFSK, and for CPFSK above
samples-per-symbol 8, the symbol-rate line is genuinely at the noise floor. Frozen V1
verified (`d6d3f918687d0700a43e46211c3f04b9`, 40 captures); suite unchanged at 775 passed.

### Problem and evidence from Entry 036

Entry 036 showed the ~0.47 FSK BER is a symbol-rate failure, not a demodulation failure:
with the true rate GFSK gives 0.002-0.031 BER, with the estimated rate ~0.50.

### Investigation - the estimator path

`estimate_parameters` evaluates three pre-FFT nonlinearities
(`SYMBOL_RATE_FEATURES`: `envelope_power`, `transition_power`,
`phase_second_difference`), runs `find_peaks` on each feature's spectrum, scores peaks in
`_select_symbol_rate` as `peak_value + 0.25 * harmonic_support`, and **keeps whichever
feature reports the highest confidence**.

### FIRST FINDING - the harmonic hypothesis is wrong

The task flagged the estimates (1880, 854, 415 Hz for true 25000, 12500, 6250) as
"suspiciously harmonic". They are **not** subharmonics. Per-feature dump, BFSK
samples-per-symbol 16, true Rs 12500 Hz:

| feature | selected | confidence | true-line level | selected-line level |
|---|---|---|---|---|
| `envelope_power` | **854.5** | **0.441** | 25.2 | 48.2 |
| `transition_power` | 41796.9 | 0.385 | 4.6 | 7.0 |
| `phase_second_difference` | **12500.0 (exact)** | 0.467 | 14.5 | 14.5 |

`phase_second_difference` finds the **exact** true rate. The spurious 854.5 Hz is a
low-frequency bump in the envelope spectrum with no harmonic relationship to Rs -
25000/1880 = 13.3, 12500/854 = 14.6, 6250/415 = 15.1, not integer ratios. Applying any
multiplier would have been curve-fitting to a coincidence.

### SECOND FINDING - it is a feature-comparability problem, but fixing that is not enough

`envelope_power` is meaningless for a constant-modulus signal - the source comment says
so - yet its confidence (0.49-0.545) outranks the correct feature (0.426-0.467), because
confidence is computed *within* a feature from prominence over that feature's own noise
floor and is not comparable *across* features.

**Candidate fix tested:** drop `envelope_power` when the envelope is essentially constant
(`amplitude_cv < 0.25`) - a signal-only gate, no generator knowledge. Measured across 8
modulations x samples-per-symbol 4/8/16/32 x 5 seeds at 20 dB, fraction within 10% of
true Rs:

| | OLD | NEW |
|---|---|---|
| overall | 129/160 = **80.6%** | 131/160 = **81.9%** |
| BFSK sps=16 | 1/5 | **3/5** |
| **GFSK sps 8/16/32** | **0/5 each** | **0/5 each** |

**+1.3% overall, and GFSK is completely unimproved.** Rejected: it does not fix the
downstream BER, which is the only thing that matters here.

### ROOT CAUSE - the line is at the noise floor, by design

Strength of the symbol-boundary impulse line in the `phase_second_difference` spectrum,
measured as line level over the spectrum's median noise floor (20 dB, seed 401):

| | sps 4 | sps 8 | sps 16 | sps 32 |
|---|---|---|---|---|
| **CPFSK/BFSK** | **42.37x** | **16.70x** | 4.88x | 2.88x |
| **GFSK** | 4.15x | 3.26x | 3.22x | 3.44x |

The estimator succeeds **exactly** where this ratio is large (BFSK sps 4 and 8) and fails
everywhere it is ~3-5x. That is the entire pattern, with no exceptions.

GFSK's Gaussian frequency pulse exists precisely to suppress spectral content at symbol
transitions - so it suppresses the very feature a transition-based timing estimator needs.
The information is not mis-selected; it is not measurably there. For CPFSK the line
weakens with oversampling because a fixed 8192-sample capture holds proportionally fewer
symbol transitions.

**No peak-selection rule can reliably pick a 3x line out of a spectrum whose noise peaks
are of comparable height.** This is Entry 014's conclusion, now quantified across
samples-per-symbol and both FSK variants, and shown at 20 dB rather than only low SNR.

### MEASURED - symbol-rate accuracy, "within 10% / within 20% (median relative error)"

**CPFSK/BFSK:**

| sps | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| 4 | **5/5,5/5 (0.00)** | **5/5,5/5 (0.00)** | **5/5,5/5 (0.00)** | 1/5,1/5 (0.87) | 0/5,0/5 (0.90) |
| 8 | **5/5,5/5 (0.00)** | **5/5,5/5 (0.00)** | 1/5,1/5 (0.66) | 0/5,0/5 (0.87) | 0/5,0/5 (0.79) |
| 16 | 1/5,1/5 (0.99) | 0/5,0/5 (0.66) | 0/5,1/5 (0.75) | 0/5,0/5 (0.68) | 0/5,0/5 (0.52) |
| 32 | 0/5,0/5 (0.93) | 0/5,0/5 (0.92) | 0/5,0/5 (0.92) | 0/5,0/5 (0.64) | 0/5,0/5 (0.50) |

**GFSK:**

| sps | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| 4 | **4/5,4/5 (0.00)** | 1/5,1/5 (0.82) | 0/5,0/5 (0.56) | 0/5,0/5 (0.80) | 0/5,0/5 (0.87) |
| 8 | 0/5,0/5 (0.92) | 0/5,0/5 (0.90) | 0/5,0/5 (0.90) | 0/5,0/5 (0.87) | 0/5,0/5 (0.55) |
| 16 | 0/5,0/5 (0.93) | 0/5,0/5 (0.93) | 0/5,0/5 (0.55) | 0/5,1/5 (0.55) | 0/5,0/5 (0.70) |
| 32 | 0/5,0/5 (0.93) | 0/5,0/5 (0.93) | 0/5,0/5 (0.61) | 0/5,1/5 (0.92) | 0/5,0/5 (0.63) |

Confidence never falls low enough to reject: the spurious estimates carry 0.44-0.55, in
the same band as the correct ones. **There is no usable rejection signal**, which is
itself an important finding - the estimator cannot currently tell the caller it failed.

### MEASURED - linear modulations are already correct, and this is NOT a general defect

Within 10% at 20 dB, 5 seeds, samples-per-symbol 4/8/16/32:

| Modulation | sps 4 | sps 8 | sps 16 | sps 32 |
|---|---|---|---|---|
| BPSK, QPSK, 8PSK | 5/5 | 5/5 | 5/5 | 5/5 |
| PAM4 | 5/5 | 5/5 | 5/5 | 4/5 |
| 16QAM | 5/5 | 5/5 | 5/5 | 4/5 |
| 64QAM | 5/5 | 5/5 | 4/5 | 2/5 |

**Every linear modulation is essentially perfect.** The defect is confined to FSK, and
within FSK to GFSK at all oversampling and CPFSK above samples-per-symbol 8. A general
estimator redesign is not warranted.

### Downstream BER - true Rs vs production-estimated Rs (median, 5 seeds, 20 dB)

| Modulation | sps | BER true Rs | BER estimated Rs |
|---|---|---|---|
| BFSK | 4 | 0.0049 | **0.0049** |
| BFSK | 8 | 0.0117 | **0.0117** |
| BFSK | 16 | 0.0157 | **0.5000** |
| BFSK | 32 | 0.0275 | **0.5000** |
| GFSK | 4 | 0.0049 | **0.0098** |
| GFSK | 8 | 0.0137 | **0.5108** |
| GFSK | 16 | 0.0157 | **0.5000** |
| GFSK | 32 | 0.0275 | **0.5000** |

The candidate fix would not have moved a single one of the failing rows, which is why it
was rejected rather than shipped for its +1.3% estimator score.

### Decision

**NEGATIVE RESULT - no production change.** A patch that improves an internal metric by
1.3% while leaving every failing downstream case at BER 0.50 would be exactly the metric
gaming this project has avoided since Entry 001.

The real fix is a **cyclostationary estimator** - cyclic autocorrelation or spectral
correlation density evaluated at cycle frequency alpha = Rs - which extracts a timing
tone that survives Gaussian shaping because it exploits correlation between spectral
components separated by Rs rather than a transition impulse. That is a new subsystem, and
this task explicitly forbade redesigning parameter estimation.

### Limitations of this investigation

- Fixed capture length of 8192 samples throughout; longer captures would raise the line
  above the floor for CPFSK at high oversampling and may partly rescue it. Not tested,
  and not something the estimator controls.
- Only h = 0.5 and BT = 0.3 were exercised - the project defaults.
- The candidate `amplitude_cv` gate was measured at 20 dB only, since it already failed
  on GFSK there.

### Next step

Entry 038 should scope a **cyclostationary symbol-rate estimator for FSK**, evaluated
against the tables above, with an explicit go/no-go on whether it lifts the GFSK
downstream BER from ~0.50 toward the ~0.015 the demodulator achieves with a correct rate.
If that also fails, FSK above samples-per-symbol 8 should be documented as an accepted
capability limit rather than retried.

### Regression

Read-only. `git diff --stat -- src/ tests/` identical to the Entry 035/036 state
(6 files, 162 insertions). Full suite **775 passed, 1 failed, 1 skipped** - the failure is
the pre-existing `reedsolo` gap, with `tests/test_model_report.py` ignored for the
pre-existing missing `h5py`. No V1 modification; hash
`d6d3f918687d0700a43e46211c3f04b9` MATCH. No ground-truth leakage: the generator was used
only as an evaluation oracle, never inside estimation.

---

## Entry 038 - 2026-09-09 - Cyclostationary FSK symbol-rate estimation (SPLIT VERDICT)

**GO for CPFSK/BFSK - narrowly. NO-GO for GFSK.** A gated cyclic estimator was
implemented. Frozen V1 unchanged (`d6d3f918687d0700a43e46211c3f04b9`, 40 captures).

### Hypothesis and rationale

A signal built from symbols at rate Rs is cyclostationary with cycle frequencies at
`k*Rs`, so the cyclic autocorrelation

    R^alpha(tau) = (1/N) sum_n p(n) p*(n-tau) exp(-j 2 pi alpha n / Fs)

is non-zero at `alpha = Rs`. Computed as an FFT of the lag product, the whole
cycle-frequency scan costs a handful of FFTs - no SCD framework needed. `p(n)` is the
instantaneous frequency, which for FSK is a PAM-like waveform at the symbol rate.

**Theory made a falsifiable prediction before any measurement.** For a linearly modulated
signal the `alpha = Rs` term is proportional to the pulse-spectrum overlap
`sum_f G(f) G*(f - Rs)`, so it exists **only when the pulse has excess bandwidth**. A
rectangular CPFSK frequency pulse has it; a Gaussian GFSK pulse (BT 0.3) deliberately does
not. Prediction: works for CPFSK, fails for GFSK. Both halves were confirmed.

### Was a real cyclic signature observed? Yes for CPFSK, no for GFSK

Peak-to-background at the true Rs (20 dB, seed 503), IF domain:

| | sps 4 | sps 8 | sps 16 | sps 32 |
|---|---|---|---|---|
| **CPFSK/BFSK** | 15.52x | **23.93x** | **12.91x** | 2.82x |
| **GFSK** | 3.64x | 4.63x | 1.88x | 1.33x |

Blind pick error: CPFSK **0.00 at every sps**; GFSK 0.88-2.14 at every sps. The IQ domain
was also tested and is worse than the IF domain everywhere (6.2x vs 15.5x at best).

### Rs accuracy - existing estimator vs raw cyclic (within 5/10/20% of 5 seeds)

| mod | sps | SNR | PROD | CYCLO |
|---|---|---|---|---|
| BFSK | 8 | 10 dB | 0/1/1 | **5/5/5** |
| BFSK | 16 | 20 dB | 1/1/1 | **5/5/5** |
| BFSK | 16 | 15 dB | 0/1/1 | **5/5/5** |
| BFSK | 32 | 20 dB | 0/0/0 | **4/4/4** |
| BFSK | 16 | 10 dB | 1/1/2 | 0/0/0 |
| BFSK | 32 | 15/10 dB | 0/0/0 | 0/0/0 |
| **GFSK** | **4** | **20 dB** | **4/4/4** | **0/0/0** |
| GFSK | all others | all | 0-1 of 5 | **0/0/0** |

The cyclic method is decisively better for CPFSK above sps 8, useless for GFSK, and
**actively worse** for the one GFSK case the existing estimator gets right.

### Observation length (20 dB, 5 seeds within 10%; median peak ratio)

| | N=8192 | N=16384 | N=32768 |
|---|---|---|---|
| BFSK sps 16 | 5/5, 12.9x | 5/5, 18.2x | 5/5, 25.8x |
| BFSK sps 32 | 4/5, 2.6x | **5/5, 3.8x** | **5/5, 5.2x** |
| GFSK sps 16 | 0/5, 1.9x | 0/5, 2.6x | **4/5, 3.9x** |
| GFSK sps 32 | 0/5, 1.3x | 0/5, 1.7x | 0/5, 1.7x |

Observation length helps - the feature is real and integrates coherently - but GFSK
sps 32 does not recover even at 4x the capture length. Capture length is not something the
estimator controls, so this is recorded as evidence, not exploited.

### The gate, and why it is set where it is

The peak-to-background ratio overlaps between correct and incorrect cyclic estimates
(correct: min 2.25, median 12.95; wrong: max 9.49), so it is an imperfect discriminator.
Swept over all 120 FSK captures:

| gate | uses cyclic | correct | vs prod-only | **regressions** |
|---|---|---|---|---|
| 0.0 | 120/120 | 44/120 | +8 | 8 |
| 3.0 | 65/120 | 45/120 | +9 | 4 |
| **6.0** | **37/120** | **40/120** | **+4** | **0** |
| 10.0 | 26/120 | 40/120 | +4 | 0 |

**6.0 chosen: the largest gain available at zero regressions.** Lower gates buy +9 instead
of +4 but break 4-8 previously-correct captures, which is not a trade this project makes.

### Production change

`src/radiofry/dsp/parameter_estimation.py`: `CYCLIC_LAGS`, `CYCLIC_PEAK_RATIO_MIN = 6.0`,
`_cyclic_symbol_rate(iq, sample_rate)` returning `(rate, ratio)`, and a gated override in
`estimate_parameters` that reports `symbol_rate_feature="cyclic_autocorrelation"` and a
confidence derived from the ratio. Signature takes samples and sample rate only - no
modulation, no sps, no ground truth.

### Downstream BER - the honest, narrow result

Production before vs after, median over 5 seeds:

| mod | sps | SNR | true Rs | **before** | **after** |
|---|---|---|---|---|---|
| **BFSK** | **16** | **20 dB** | 0.0176 | **0.5135** | **0.0176** |
| BFSK | 16 | 15/10 dB | 0.0176 | 0.51 | 0.51 (unchanged) |
| BFSK | 32 | all | 0.0196 | 0.50 | 0.50 (unchanged) |
| GFSK | all | all | 0.004-0.03 | 0.38-0.53 | unchanged |

**Exactly one of 24 conditions improved, and it improved completely** (0.5135 -> 0.0176,
matching the true-Rs BER to four decimals). This is stated plainly rather than dressed up:
the raw estimator fixes more cases than the gated production path ships, because the gate
was set for zero regressions.

CPFSK sps 32 is the clearest casualty: the raw estimator finds its rate (error 0.00) but
the ratio is ~2.8x, below the gate. Pinned by
`test_cpfsk_sps_32_is_recovered_by_the_estimator_but_not_by_the_gate` so it is not
mistaken for a win.

### GO / NO-GO

- **GO for CPFSK/BFSK, narrowly**: a real signature (12.9-23.9x), correct across 5 seeds,
  no generator constants, zero measured regressions, and one complete downstream BER fix.
- **NO-GO for GFSK**: the feature is absent by construction (1.3-4.6x, indistinguishable
  from background), the blind pick is wrong at every sps and SNR, and downstream BER does
  not move. **GFSK symbol-rate estimation above sps 4 is a capability limitation of
  RadioFry**, and no further estimator family should be attempted for it without changing
  the signal model (longer captures, or a known preamble).

### Regression

- Full suite **800 passed, 1 failed, 1 skipped**. The failure is the pre-existing
  `reedsolo` gap; `tests/test_model_report.py` is ignored for the pre-existing missing
  `h5py`. No new failures.
- New `tests/test_cyclic_symbol_rate.py`, **25 tests**, including the gate threshold, the
  signature taking only samples and sample rate, CPFSK sps 16 fixed in production, CPFSK
  sps 4/8 unchanged, GFSK sps 4 not broken further, **12 parametrised linear-modulation
  non-regression cases**, scale invariance, and the sps 32 limitation.
- **The cyclic gate fired on 0 of 120 linear-modulation captures at 20 dB** - BPSK, QPSK,
  8PSK, PAM4, QAM16 and QAM64 are provably untouched there.
- **One prior test deliberately updated**:
  `test_symbol_rate_estimator_regression.py::test_estimate_reports_which_feature_was_selected`
  asserted the feature name is one of the three pre-FFT nonlinearities. A fourth
  legitimate source now exists; the test's purpose - that the estimate says which source
  produced it - is unchanged.
- Analog routing, the Entry 034 digital-family gate and Entry 035 PAM4 demodulation all
  pass unchanged.

### A pre-existing weakness found while checking regressions

Linear modulations at sps 32 are **not** as solid as Entry 037 suggested. On seeds
503-541 at 20 dB: BPSK 4/5, 8PSK 3/5, PAM4 2/5, 16QAM 1/5, 64QAM 3/5 within 5%.
**Verified pre-existing** by re-running with this entry's change stashed - identical
numbers, and the cyclic gate never fires there. Entry 037's seeds 401-431 were simply
luckier. This is a separate, previously unrecorded defect.

### Limitations

- The gate is a blunt instrument: the ratio overlaps between correct and wrong estimates,
  so it buys safety by declining most of the cases it could fix.
- Only h = 0.5 and BT = 0.3 were exercised.
- The 6.0 gate was tuned on the same 120-capture sweep used to report the gain; it has not
  been validated on a fully held-out seed set.

### Next recommendation

**Entry 039 should be the final synthetic benchmark**, not another estimator family. The
FSK estimator question is now answered in both directions, and Entry 037's advice to stop
after a second negative stands for GFSK. The newly found linear-modulation sps 32 weakness
should be quantified in that benchmark rather than chased separately.

---

## Entry 039 - 2026-09-09 - Final synthetic end-to-end benchmark + freeze decision

**Audit only - zero production changes.** Frozen V1 verified
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures); suite unchanged at 800 passed.

**DECISION: TARGETED BLOCKER.** One narrowly scoped Entry 040 is justified; see the end.

### Benchmark design

**570 captures**, config and results saved to `reports/benchmark_v039/`
(`config.json`, `results.pkl`) - gitignored, untracked.

| | |
|---|---|
| Digital | 8 generator classes x samples-per-symbol 4/8/16/32 x SNR 20/15/10/5/0 dB x 3 seeds = **480** |
| Analog | AM-DSB, AM-SSB USB, AM-SSB LSB, WBFM x 5 SNR x 3 seeds, SSB run **with and without** SigMF metadata = **90** |
| Seeds | **601, 607, 613** - unused in Entries 033-038 |
| Fs / N | 200 kHz / 8192 samples |
| Checkpoint | `models_saved/modulation_cnn.pt` (11-class RadioML) |

Note recorded in the config: **BFSK (generator) and CPFSK (production label) are the same
signal**, not two classes.

### Leakage audit

- Seeds 601/607/613 are new; the production checkpoint was trained on RadioML 2016.10a,
  never on RadioFry synthetic captures.
- Source scan of `pipeline`, `parameter_estimation`, `cyclostationary`,
  `confidence_fusion`, `modulation_inference`: **no reference to ground truth,
  `load_ground_truth`, `SampleSpec`, true Rs or samples-per-symbol.**
- Ground truth is used only to score afterwards; metadata is supplied **only** in the two
  conditions that explicitly represent an external recorder.
- The oracle true-Rs demodulation is recorded as a separate diagnostic column and never
  fed back into the pipeline.

### 2. Overall (480 digital)

| Metric | Result |
|---|---|
| CNN top-1 | 185/480 = **38.5%** |
| CNN top-3 | 353/480 = **73.5%** |
| fused top-1 | 140/480 = **29.2%** |
| family-level | 215/480 = 44.8% |
| rejection | 216/480 = **45.0%** |
| confident-wrong | 124/480 = **25.8%** |

Fused top-1 is *below* CNN top-1 by design: the 0.4 threshold and the Entry 034 gate
convert low-confidence and unsafe decisions into rejection.

### 3. Per modulation (fused correct / CNN top-3, 15 per cell)

| class | sps 4 | sps 8 | sps 16 | sps 32 | total |
|---|---|---|---|---|---|
| BPSK | 4 (t3 15) | **15** (15) | 9 (14) | 0 (9) | 28/60 |
| QPSK | 4 (15) | **12** (15) | 9 (14) | 0 (6) | 25/60 |
| 8PSK | 9 (15) | **12** (15) | 0 (9) | **0 (0)** | 21/60 |
| CPFSK | 0 (10) | **15** (15) | 0 (5) | **0 (0)** | 15/60 |
| GFSK | 0 (0) | 2 (15) | **13** (15) | 1 (15) | 16/60 |
| PAM4 | 0 (9) | 6 (13) | 11 (13) | 7 (12) | 24/60 |
| QAM16 | 0 (14) | 1 (15) | 0 (13) | 0 (4) | **1/60** |
| QAM64 | 0 (13) | 8 (14) | 2 (13) | 0 (3) | 10/60 |

**By SNR:** 20 dB 40.6%, 15 dB 38.5%, 10 dB 34.4%, 5 dB 20.8%, 0 dB 11.5%.
**By SPS:** sps 4 **14.2%**, sps 8 **59.2%**, sps 16 36.7%, sps 32 **6.7%**.

### 4. Parameter estimation

Rs within 10%, per class x samples-per-symbol (of 15):

| class | sps 4 | sps 8 | sps 16 | sps 32 |
|---|---|---|---|---|
| BPSK/QPSK/8PSK | 15/15 | 15/15 | 12-13/15 | **7-10/15** |
| PAM4 | 15/15 | 15/15 | 11/15 | **0/15** |
| QAM16/QAM64 | 15/15 | 15/15 | 12-13/15 | **2-3/15** |
| CPFSK | 9/15 | 8/15 | 4/15 | **0/15** |
| GFSK | 3/15 | **0/15** | **0/15** | **0/15** |

Overall **308/480 = 64.2%** within 10%. Carrier |error| median 92.2 Hz (max 606.8).
SNR estimate biased low (true 20 -> 14.6 dB, 10 -> 10.2, 0 -> 3.4).

### 5. Digital BER - and the single most important number in this benchmark

Conditional on correct classification, production Rs vs oracle Rs:

| class | sps | 20 dB | 15 dB | 10 dB | 5 dB |
|---|---|---|---|---|---|
| BPSK | 8/16 | 0.0000 | 0.0000 | 0.0000 | 0.0039 |
| QPSK | 8 | 0.0000 | 0.0000 | 0.0005 | 0.0625 |
| 8PSK | 8 | 0.0000 | 0.0013 | 0.0521 | 0.1969 |
| PAM4 | 8/16 | 0.0000 | 0.0000-0.0010 | 0.0215-0.0234 | 0.3082 |
| QAM64 | 8 | 0.0218 | 0.1063 | 0.2263 | - |
| CPFSK | 8 | 0.0078 | 0.0078 | 0.0078 | 0.4884 (oracle **0.0371**) |
| GFSK | 16 | 0.4414 | 0.3913 | 0.4414 | (oracle **0.0098-0.0411**) |

**The master variable:**

| | n | median BER |
|---|---|---|
| **Rs estimate correct (within 10%)** | 108 | **0.0010** |
| **Rs estimate wrong** | 32 | **0.4881** |

Everything downstream works. **Symbol-rate estimation is the system's dominant failure
mode**, and it accounts for essentially every non-classification BER failure.

### 6. FSK limitation (all SNR pooled)

| class | sps | Rs within 10% | production BER | **oracle BER** |
|---|---|---|---|---|
| CPFSK | 4 | 9/15 | 0.4993 | **0.0156** |
| CPFSK | 8 | 8/15 | **0.0205** | 0.0205 |
| CPFSK | 16 | 4/15 | 0.4740 | **0.0215** |
| CPFSK | 32 | 0/15 | 0.5001 | **0.0235** |
| GFSK | 4 | 3/15 | 0.4939 | **0.0454** |
| GFSK | 8 | 0/15 | 0.5061 | 0.2532 |
| GFSK | 16 | 0/15 | 0.4414 | **0.0411** |
| GFSK | 32 | 0/15 | 0.5273 | **0.0627** |

The gap between production and oracle columns is entirely the estimator. This is the
Entry 037/038 limitation, now fully quantified and **left as-is**.

### 7. SPS=32 - the weakness IS end-to-end significant, and it is two defects not one

| class | Rs ok | fused ok | **CNN top-1** | production BER | oracle BER |
|---|---|---|---|---|---|
| BPSK | 10/15 | 0/15 | **0/15** | - | **0.0000** |
| QPSK | 8/15 | 0/15 | **0/15** | 0.5015 | **0.0020** |
| 8PSK | 7/15 | 0/15 | **0/15** | 0.4805 | 0.0560 |
| PAM4 | 0/15 | 7/15 | 7/15 | 0.5000 | **0.0156** |
| QAM16 | 3/15 | 0/15 | **0/15** | 0.5000 | 0.0928 |
| QAM64 | 2/15 | 0/15 | **0/15** | 0.5068 | 0.2428 |

Entry 038 flagged the Rs weakness at sps 32. This benchmark shows it is **not the whole
story**: even where Rs is recovered (BPSK 10/15, QPSK 8/15), **CNN top-1 is 0/15**. The
oracle BER column proves the information is present in the samples. So sps 32 fails at
*both* the estimator and the classifier, and the classifier failure is the larger of the
two.

### 8. Analog (3 captures per SNR cell)

| condition | metadata | 20 | 15 | 10 | 5 | 0 dB | total | median message correlation @20/15/10 |
|---|---|---|---|---|---|---|---|---|
| AM-DSB | - | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 9/15 | 0.9424 / 0.8438 / 0.6557 |
| AM-SSB USB | **yes** | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 9/15 | **0.9950 / 0.9843 / 0.9525** |
| AM-SSB USB | **no** | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 9/15 | **0.0132 / -0.0029 / 0.0167** |
| AM-SSB LSB | **yes** | 3/3 | 3/3 | 1/3 | 0/3 | 0/3 | 7/15 | **0.9950 / 0.9843 / 0.9524** |
| AM-SSB LSB | **no** | 3/3 | 3/3 | 1/3 | 0/3 | 0/3 | 7/15 | **-0.0062 / -0.0034 / 0.0040** |
| WBFM | - | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 12/15 | 0.8901 / 0.7373 / 0.5153 |

Classification is **identical** with and without metadata, as it should be - metadata
affects demodulation only. The recovery difference is total: **0.995 vs 0.013**. Analog
classification is reliable at >= 10 dB and declines (into rejection, not error) below.

### 9. Safety - the strongest area of the system

| Metric | Result |
|---|---|
| **digital -> analog false routing** | **1/480 = 0.21%** |
| **analog -> digital false routing** | **0/90 = 0.00%** |
| Entry 034 digital-family gate fired | 118/570 |
| Entry 032 analog subtype route fired | 53/570 |
| Entry 026 CNN-alternative route fired | 0/570 |
| system correct | 193/570 = 33.9% |
| system rejected | 253/570 = 44.4% |
| system confident-wrong | 124/570 = 21.8% |

The Entry 034 gate turned confident-wrong analog labels into **87 rejections**, 4 GFSK and
1 BPSK. That is the gate doing exactly its job: it converts a wrong-domain answer into an
honest "unclassified", which is why fused accuracy is lower than CNN accuracy.

### 12/13. Multi-format and report validation

The same QPSK capture through `.iq` and `.wav` gives identical fused label (QPSK),
identical BER (0.0) and identical Rs (25000) - **conclusions are not format-dependent**.

Report structure **PASS**: top-level `generated_at`/`schema_version`/`source`/`stages`;
all seven stages present; JSON-serialisable; analog record carries `ber_status`,
`ber_reason`, `fusion_label`, `est_symbol_rate_hz`, `classical_family`, with
`ber_strict=None` and `es_n0_db=None` as required.

### Regression

Full suite **800 passed, 1 failed, 1 skipped**. The failure is the pre-existing `reedsolo`
gap; `tests/test_model_report.py` ignored for the pre-existing missing `h5py`. No new
failures. V1 hash MATCH. **Zero production changes.** Benchmark artifacts live under
`reports/benchmark_v039/`, confirmed gitignored.

### DECISION: TARGETED BLOCKER

Not a freeze, and not merely a fundamental limitation. The reasoning:

- **No implementation defect exists.** Every route works, reports validate, formats agree,
  safety is excellent (0.21% / 0.00% false routing), and demodulators achieve
  **median BER 0.0010** when handed a correct symbol rate.
- **Two separable causes explain the low end-to-end numbers.** One is the FSK/high-sps
  symbol-rate limitation, already investigated to exhaustion in Entries 037-038 and
  correctly closed as a fundamental limitation of the signal model.
- **The other is new, and it is not fundamental**: the CNN is strong at samples-per-symbol
  8 (59.2% fused) and collapses either side of it - 14.2% at sps 4, **6.7% at sps 32**,
  with CNN top-1 **0/15** for five of six linear classes at sps 32 and top-3 as low as
  0/15. The oracle BER column proves the information is in the samples. The 11-class
  RadioML checkpoint was trained around 8 samples-per-symbol; sps 4 and 32 are simply
  out of distribution.

**This corrects Entry 033.** That entry concluded "do not retrain - top-3 is 100% at
>= 10 dB", but it measured only samples-per-symbol 8. Across sps 4-32 top-3 is 73.5%, and
0/15 in the worst cell. The evidence that justified deferring retraining does not hold on
the wider matrix.

### Recommended Entry 040 - exactly one, narrowly scoped

**Retrain the V2 CNN with samples-per-symbol augmentation (4-32)** and, in the same entry,
check whether the 128-sample inference frame is adequate at high oversampling - at sps 32
a 128-sample window holds only 4 symbols, versus 32 at sps 4, so the frame length may be a
second, architectural half of the same problem. The training pipeline
(`training/train_v2_synthetic.py`) and a generator supporting arbitrary sps already exist,
so this is bounded work against a clear, measured target: lift sps 4 and 32 fused accuracy
from 14.2% and 6.7% toward the 59.2% the model already achieves at sps 8.

Do **not** reopen symbol-rate estimation. Entries 037 and 038 answered that question in
both directions.

### Limitations of this benchmark

- 3 seeds per cell: enough to see 0/15 and 15/15 patterns clearly, not enough for tight
  confidence intervals on mid-range cells.
- Fixed 8192-sample captures and a single 200 kHz sample rate.
- Only h = 0.5 and BT = 0.3 for FSK; only modulation depth 0.5 for AM-DSB.
- WAV was cross-checked on one capture rather than the full matrix, since ingestion has
  dedicated focused tests.

---

## Entry 040 - 2026-09-10 - CNN SPS generalisation + inference window: SHIP

Closes the Entry 039 TARGETED BLOCKER. Frozen V1 unchanged
(`d6d3f918687d0700a43e46211c3f04b9`, 40 captures). Suite 800 -> **826 passed**.

### Audit - verified, not assumed

| | |
|---|---|
| `modulation_cnn.pt` (production default) | trained on **RML2016.10a**, which is generated at a **single 8 samples/symbol** |
| `train_v2_synthetic.py` | `SAMPLES_PER_SYMBOL = 8` hard-coded, `FRAME_LENGTH = 128` |
| `ModulationCNN` | ends in **`AdaptiveAvgPool1d(1)` - input length is NOT fixed** |

**Neither checkpoint had ever seen an oversampling factor other than 8.** The pooling
layer meant the window hypothesis could be tested at 256 and 512 samples with **zero
architecture change**, so the two hypotheses separate cleanly instead of being confounded.

Symbols per inference frame:

| | sps 4 | sps 8 | sps 16 | sps 32 |
|---|---|---|---|---|
| 128 samples | 32 | 16 | 8 | **4** |
| 256 samples | 64 | 32 | 16 | 8 |
| 512 samples | 128 | 64 | 32 | 16 |

### CORRECTION to the Entry 039 attribution

Entry 039 attributed the blocker to SPS training distribution. Measured on one unseen test
set, that is only **half** of it:

| checkpoint | overall | sps 4 | sps 8 | sps 16 | sps 32 |
|---|---|---|---|---|---|
| 11-class RadioML (**production default**) | 43.3% | 36.7% | 65.0% | 56.7% | **15.0%** |
| 8-class V2 synthetic (**already in the repo**) | **86.2%** | 78.3% | 100% | 90.0% | **76.7%** |
| sps-augmented experimental | **99.2%** | 96.7% | 100% | 100% | **100%** |

**The larger single step is domain mismatch, not oversampling**: 43.3% -> 86.2% comes from
simply using a checkpoint trained on RadioFry's own distribution, which already existed.
SPS augmentation then adds 86.2% -> 99.2%. Entry 039 measured only the production default
and so attributed the whole gap to SPS.

### Hypothesis 2 (window too short) - REJECTED

Same sps-augmented training, three window lengths:

| window | overall top-1 | sps 32 | best validation loss |
|---|---|---|---|
| **128** | **99.2%** | 100% | 0.2504 |
| 256 | 99.2% | 100% | 0.1788 |
| 512 | 98.8% | 100% | 0.1579 |

Longer windows lower validation loss but do **not** improve test accuracy. **4 symbols per
frame is sufficient once the model has seen that regime in training.** The 128-sample
window was therefore kept, and the production inference path is unchanged.

### Production changes

**1. `training/train_v2_synthetic.py`** - `SAMPLES_PER_SYMBOL_SWEEP = (4, 8, 16, 32)`,
`CaptureSpec.samples_per_symbol`, capture id now encodes sps, and `render_capture` holds
the **capture sample count constant** (`NUM_SYMBOLS * 8 // sps`), so higher oversampling
means proportionally fewer symbols - matching what the runtime actually sees. The sweep is
balanced: equal capture counts per oversampling factor, asserted by test.

**2. `pipeline.py`** - new `DEFAULT_MODULATION_MODEL` pointing at
`models_saved/modulation_cnn_v3_spsaug.pt`. The RadioML checkpoint is **retained**, so the
Entry 039 baseline stays reproducible.

**3. New checkpoint**, produced by the project's own pipeline (not the ad-hoc experiment
script): best epoch 35, validation loss 0.2469, frame-level validation accuracy 0.8972,
44800/16000/16000 train/validation/test frames, capture-level split.

### End-to-end result - production fusion and dispatch

| | fused | sps 4 | sps 8 | sps 16 | sps 32 | rejection | confident-wrong |
|---|---|---|---|---|---|---|---|
| **before** (11-class) | 35.4% | 16.7% | 68.8% | 45.8% | **10.4%** | 41.1% | **23.4%** |
| **after** (v3 sps-aug) | **99.5%** | **100%** | **100%** | **100%** | **97.9%** | **0.0%** | **0.5%** |

**Confident-wrong fell 23.4% -> 0.5%**, so this is not a model that improved by becoming
confidently wrong - the safety property improved alongside accuracy. Digital -> analog
false routing stayed **0/192** before and after.

BER, conditional on correct classification: **median 0.0008 when the symbol rate is
correct**, 0.5058 when it is wrong (n=46). That residual is entirely the Entry 037/038
symbol-rate limitation, deliberately untouched here.

### The analog path is unaffected by dropping the CNN's analog classes

The new checkpoint is **digital-only** (8 classes). Verified end to end:

| | 11-class | v3 digital-only |
|---|---|---|
| AM-DSB | 6/6 | **6/6** |
| AM-SSB USB | 6/6 | **6/6** |
| WBFM | 6/6 | **6/6** |

Entry 032's classical subtype gate decides the analog label from envelope evidence and
**never consulted the CNN**; Entry 039 measured Entry 026's CNN-alternative route firing
**0/570**. So an 8-class digital CNN plus the classical analog gate is architecturally
cleaner than before, not a regression. Entry 034's guard becomes structurally unreachable
for real captures - no analog labels exist to block - which can only reduce digital ->
analog false routing.

### Leakage audit

Four disjoint seed blocks: train 2,000,000+, validation 3,000,000+, test 4,000,000+,
end-to-end 5,000,000+, analog 6,000,000+, focused tests 7,000,000+. **None overlap Entry
039's 601/607/613.** Splits are capture-level, so no frame of a capture straddles splits.
Model selection used validation loss only; the reported numbers are from unseen seeds. No
ground truth, filename or metadata influences inference.

### Regression

**826 passed, 1 failed, 1 skipped** (was 800 passed). The failure is the pre-existing
`reedsolo` gap; `tests/test_model_report.py` ignored for the pre-existing missing `h5py`.
**No new failures** - Entry 034 digital-family gate, Entry 035 PAM4 demodulation, Entry
038 cyclostationary estimator, analog routing, report generation and IQ/WAV consistency
all pass unchanged. New `tests/test_sps_generalisation.py`, **26 tests**. V1 hash MATCH.

### An operational caveat that must not be lost

`models_saved/` is gitignored, so **the new default checkpoint is untracked**. A fresh
clone will not have it, and `predict_modulation` will return `Unclassified` with
"Checkpoint not found" rather than failing loudly. This was already true of the previous
default; the change does not make it worse, but it is now on the critical path for the
headline result and should be handled by whatever artifact-distribution mechanism the
project adopts.

`train_v2` also does **not** write the `_metrics.json` sibling that the inference loader
requires - `write_metrics()` is a separate call. A checkpoint trained without it loads as
`Unclassified`. Encountered during this entry and worth fixing in a later cleanup.

### Limitations

- **99.5% is synthetic-to-synthetic.** Trained and tested on the same generator with
  disjoint seeds and capture-level splits. It measures SPS generalisation *within* our
  signal model and is **not** a real-world accuracy claim.
- Only the 8 digital classes; no analog classes were added, by design.
- Only samples-per-symbol 4/8/16/32 and SNR 20/15/10 dB were evaluated end to end.
- The single remaining sps-32 error (1 of 48) was not investigated.

### DECISION: SHIP

Both experiments answered cleanly, the improvement is large and measured on unseen seeds,
safety improved rather than degraded, analog is unaffected, and the full suite is green.

### Next step

The Entry 039 blocker is closed. **Recommend returning to the Entry 039 freeze question**:
re-run that benchmark against the new default and, if it confirms these numbers, declare
SYNTHETIC FREEZE and move to real-world data. The remaining known limitations - FSK
symbol-rate estimation above sps 8 and metadata-free SSB recovery - are both already
closed as fundamental for this development cycle.
