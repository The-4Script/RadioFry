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
