# Advanced analysis: cyclostationary SCD and the capability envelope

Two investigator-facing features that sit *beside* the analysis pipeline rather than
inside it. Neither changes ingestion, preprocessing, parameter estimation, the CNN,
classical detection, fusion, demodulation, FEC, correlation or report generation. Both can
be removed by deleting their page and module without affecting normal analysis.

| | Feature 1 | Feature 2 |
|---|---|---|
| Core | `src/radiofry/dsp/spectral_correlation.py` | `src/radiofry/evaluation/capability_surface.py` |
| Page | `gui/pages/10_cyclostationary.py` | `gui/pages/11_capability.py` |
| Tests | `tests/test_spectral_correlation.py` (42), `tests/test_conjugate_scd.py` (21) | `tests/test_capability_surface.py` (27) |
| Shared page tests | `tests/test_advanced_pages.py` (16) | |
| Trigger | Explicit **Run Cyclostationary SCD** button | Loads recorded artifacts only |
| Input | The capture already analyzed | `reports/benchmark_v039/results.pkl` |

---

## Feature 1 — Spectral Correlation Density

### What it does

Estimates the spectral correlation density of the loaded capture over a
frequency x cyclic-frequency grid, and reports the cyclic frequencies where the signal
correlates with itself.

### Why it exists

The power spectrum of a digitally modulated signal is smooth and says nothing about the
symbol clock. The *spectral correlation* is not smooth: a signal built from symbols at
rate `Rs` correlates the spectral component at `f + alpha/2` with the one at
`f - alpha/2` whenever `alpha = k*Rs`. That gives an independent, non-ML view of the
symbol rate — the quantity Entry 039 identified as RadioFry's dominant end-to-end failure
mode — and a way to see that a capture carries a symbol clock at all.

### Mathematical basis

The symmetric spectral correlation density is

```
S_x^alpha(f) = lim E[ X_T(f + alpha/2) * conj(X_T(f - alpha/2)) ]
```

`alpha = 0` reduces to the ordinary PSD. The estimator is Gardner's **time-smoothed
cyclic periodogram**, implemented with exact time-domain frequency shifts:

```
a_alpha(n) = x(n) * exp(-j*pi*alpha*n/fs)     ->  spectrum X(f + alpha/2)
b_alpha(n) = x(n) * exp(+j*pi*alpha*n/fs)     ->  spectrum X(f - alpha/2)

S^alpha(f_k) = (1/P) * sum_p A_p(k) * conj(B_p(k)) / (fs * sum(w^2))
```

Shifting in the time domain (rather than re-indexing FFT bins) makes the `alpha/2` shift
exact for *any* `alpha`, so both axes stay uniform and nothing is interpolated. The
per-frame phase `exp(-j*2*pi*alpha*p*H/fs)` that survives the `A*conj(B)` product is the
`exp(-j*2*pi*alpha*n)` of the definition; averaging it over frames is what makes the
vertical axis a cyclic frequency. **Without that term this would be an ordinary
cross-spectrum, and the feature would be a spectrogram with a relabelled axis.**

Only `alpha >= 0` is computed: `S^(-alpha)(f) = conj(S^alpha(f))`, so `|S|` is even in
`alpha` and the negative half is a mirror, not new evidence.

### Two normalizations

- **coherence** (default) — `|S^alpha(f)| / sqrt(S^0(f+alpha/2) * S^0(f-alpha/2))`,
  dimensionless and bounded by 1, so values at different frequencies are comparable.
- **density** — raw `|S^alpha(f)|`, in power per Hz, dominated by wherever the power is.

### Inputs and outputs

Input: complex baseband IQ, a positive sample rate, an optional hardware centre
frequency, and an `SCDConfig`. Output: an `SCDResult` carrying `frequencies_hz`,
`alphas_hz`, `magnitude[alpha, frequency]`, the sample rate, samples used vs available,
frames, hop, effective averages, both resolutions, the normalization, the config, the
centre frequency, the method name, warnings and elapsed time.

### Assumptions and limits

**The hop is derived, not chosen.** The per-frame phase is sampled once per hop `H`, so
it cannot distinguish `alpha` from `alpha + k*fs/H`; at those cyclic frequencies the phase
is identically 1 and any correlation between the two bands survives in full. The
estimator is therefore unambiguous only for `|alpha| < fs/(2H)`, and

```
H = floor(fs / (2 * alpha_max))
```

This is a real trade: a wider cyclic range forces a shorter hop, which shortens the span
the frames cover and coarsens cyclic resolution `1/T`.

**This trap is easy to miss.** For a capture at `fs` with integer samples-per-symbol,
`Rs = fs/sps` frequently lands *exactly* on a multiple of `fs/H`, so an aliasing estimator
appears to recover the symbol rate perfectly. The tests deliberately use symbol rates that
do not divide the sample rate.

**Coherence needs power.** In a band holding no signal, coherence divides leakage by
leakage and saturates towards 1 — reporting perfect cyclostationarity exactly where there
is no signal. Bins whose sideband power is below `coherence_floor_db` (default -60 dB
relative to the strongest bin) are reported as zero.

**Resolution.** Cyclic resolution is `1/T` over the span actually analyzed; frequency
resolution is `fs/FFT`. A peak is never sharper than those. The cyclic grid is derived
from the resolution by default, because a coarser grid steps straight over real peaks.

**Statistical reliability.** Frames overlap heavily once the hop is short, so the frame
count overstates the averaging. `effective_averages = span / fft_size` is the
independent-look count; coherence between uncorrelated bands has an expected magnitude of
about `1/sqrt(N)`, which is the floor the peak picker uses. Below about 8 looks the page
warns.

**Scope.** Baseband complex IQ only. The frequency axis is a baseband offset unless the
capture carried a hardware centre frequency — no absolute RF frequency is ever invented.
This is non-conjugate SCD only; conjugate cyclostationarity (which distinguishes BPSK from
QPSK) is not computed.

### Performance

Cost is `O(alpha_bins * frames * fft_size * log fft_size)` and is **bounded independently
of capture length**: `max_samples` caps what is read and `max_frames` caps the work.
Measured on a laptop at fs = 200 kHz, defaults (fft 128, alpha_max fs/8, 768 frames):

| Capture | Time | Frames | Span read | Surface |
|---|---|---|---|---|
| 8 k samples | ~2.9 s | 768 | 3,196 | 401 x 128 |
| 32 k samples | ~3.0 s | 768 | 3,196 | 401 x 128 |
| 131 k samples | ~3.0 s | 768 | 3,196 | 401 x 128 |
| 524 k samples | ~3.2 s | 768 | 3,196 | 401 x 128 |

Surface memory is ~0.4 MB. The 3D view is decimated by striding (never interpolated) to
about 160 x 240 cells; the 2D heatmap uses the full surface. Results are cached on the
samples plus the exact configuration, so changing views is free.

### Validation performed

Synthetic BPSK and QPSK at symbol rates chosen **not** to divide the sample rate
(9,300 / 17,000 / 23,100 / 31,250 / 41,700 Hz) produce a peak at `alpha = Rs` within a few
cyclic-resolution units, plus genuine harmonics at multiples of `Rs`. Three negative
controls — complex AWGN, a CW tone, and a constant — produce **no** cyclic features.

On a real frozen-V1 capture (`QPSK_snr20dB_r000.wav`): the strongest cyclic feature was
`alpha = 25,000.0 Hz` at coherence 0.988, against a pipeline-reported symbol rate of
25,000.0 Hz — agreement to 0.0 Hz, from an entirely independent computation.

### How to read it

A horizontal band away from `alpha = 0` is a cyclic feature. The band at `alpha = 0` is
the power spectrum and is present for every signal including noise. Harmonics at multiples
of a symbol rate are genuine, not disagreements. The page reports candidates; it does not
claim a symbol rate and does not change the reported one.

---

## Feature 2 — Capability-limitation surface

### What it does

Plots the recorded end-to-end benchmark as an SNR x samples-per-symbol surface per
modulation, so an investigator can answer: *under what conditions should I trust RadioFry
for this modulation?*

### Provenance

`reports/benchmark_v039/results.pkl` (BANK.md Entry 039): 480 digital captures over 8
production classes x samples-per-symbol 4/8/16/32 x SNR 20/15/10/5/0 dB x 3 seeds, at
200 kHz with 8192 samples per capture, scored against generator ground truth *after* the
pipeline ran. **That directory is gitignored**, so it exists only where the benchmark has
been run locally; its absence is reported as missing evidence, never as an error.

### Metrics

| Metric | Aggregation | Population |
|---|---|---|
| `ber_prod` | median | every capture in the cell that produced bits |
| `ber_oracle` | median | every capture (the true symbol rate is supplied) |
| `classification_accuracy` | mean | every capture |
| `symbol_rate_accuracy` | mean | every capture, within 10% of truth |

Median for BER because it is heavily skewed and one outright failure at ~0.5 would drag a
mean away from what the cell typically delivers. Mean for fractions because an accuracy is
the fraction that succeeded — a median of per-capture booleans is only a majority vote.

**The production BER population includes misclassified captures.** `ber_prod` exists
whenever the pipeline emitted bits, including the 124 captures it had misclassified;
demodulating with the wrong scheme really does produce garbage bits, and excluding those
would flatter the surface. The count classified correctly is shown alongside every cell.

The **oracle** surface replaces the estimated symbol rate with the true one. The gap
between the two is the symbol-rate estimator's contribution — Entry 039's central finding.

### Missing data

Three distinct states, never merged:

- **measured** — the cell holds the median of its measurements.
- **no output** — captures were run and the pipeline emitted no bits to score. This is a
  limitation of the pipeline at that condition, not a gap in the experiment, and is
  coloured separately.
- **not benchmarked** — the condition was never run.

Nothing is interpolated into an empty cell, and the 3D mesh is drawn with
`connectgaps=False` so it is visibly not closed across them. `BER = 0` is drawn at an
explicit `1e-4` floor on the log axis, labelled as a bound: it means no errors were seen
in the bits compared, not that the link is error-free.

### Thresholds

The BER threshold colours cells as a **reading aid only** and is configurable
(1e-4 .. 1e-1, default 1e-2). RadioFry defines no pass/fail BER, and the page says so.

### Coverage as measured

All eight classes carry 60 captures each (20 cells x 3 seeds). Production-BER coverage is
genuinely incomplete, and that incompleteness is the point:

| Class | Cells measured | At or below 1e-2 | No output |
|---|---|---|---|
| BPSK | 14/20 | 12 | 6 |
| QPSK | 17/20 | 8 | 3 |
| 8PSK | 15/20 | 4 | 5 |
| CPFSK | 17/20 | 3 | 3 |
| GFSK | 16/20 | 0 | 4 |
| PAM4 | 11/20 | 4 | 9 |
| QAM16 | 14/20 | 0 | 6 |
| QAM64 | 12/20 | 0 | 8 |

### Limits

An experimental envelope over **synthetic** captures at one sample rate and one capture
length, with 3 seeds per cell. Real signals with different impairments may behave
differently, and conditions outside the tested grid carry no evidence either way. A
3-sample median is a coarse statistic. The surface interpolates between measured points
for display only.

---

## Regression safety

Neither feature is reachable from `analyze_capture`, the report builder, or any pipeline
stage. The SCD page computes nothing until its button is pressed — asserted directly by
`test_opening_the_scd_page_does_not_compute_an_scd`. The capability page needs no loaded
capture at all. Both fail closed: degenerate input yields an empty result carrying a
warning, and no code path in either module raises into the caller.


---

## Feature 1b - Conjugate SCD (refinement)

### What it adds

A second, explicitly selected cyclostationary mode on the same page, sharing the framing,
frequency-shift, windowing, power-floor, caching, cost-estimation and plotting machinery.
The ordinary surface is unchanged and remains the default.

### Mathematical definition

    S*_x^alpha(f) = E[ X(f + alpha/2) X(alpha/2 - f) ]

One conjugation short of the ordinary definition, and that difference is the whole point.
The ordinary spectrum relates a component to the *conjugate* of another; this one relates
it to the component itself. The result is sensitive to **impropriety**: a complex process
is proper when `E[x(t)x(t+tau)] = 0`, and a linear modulation inherits that from its
constellation. `E[s^2] != 0` -> improper -> a conjugate spectrum exists.

| constellation | `E[s^2]` | proper? | conjugate surface |
|---|---|---|---|
| BPSK (+/-1) | 1 | no | present |
| PAM4 (real levels) | non-zero | no | present |
| QPSK `exp(j*pi*(2k+1)/4)` | 0 | yes | absent |
| 8PSK, QAM16, QAM64 | 0 | yes | absent |

### Implementation

The second factor is read off the **same** shifted transform as the first:

    a_alpha(n) = x(n) exp(-j*pi*alpha*n/fs)   ->  A(f) = X(f + alpha/2)
    A(-f) = X(alpha/2 - f)

so `S*^alpha(f) = mean_p[ A_p(f) A_p(-f) ]` needs **one** FFT per cyclic frequency instead
of two. `-f` is an index permutation on the unshifted grid, computed once.

Normalization divides by the geometric mean of the ordinary PSD at the two frequencies
being related, keeping the result in `[0, 1]` and comparable across the band. The same
-60 dB power floor applies to both bands.

**`alpha = 0` is the measurement, not a ridge.** In the ordinary surface `alpha = 0` is
the PSD and is always maximal, so the peak picker steps past it. In the conjugate surface
`alpha = 0` *is* the impropriety, so it is included. This is the one behavioural
difference between the modes, and it is deliberate.

### Assumptions and limitations

- **A frequency offset moves everything.** Conjugate features sit at `2*f_offset + k*Rs`,
  so they are not directly comparable with a reported symbol rate unless the capture is
  centred. The page says so rather than drawing a comparison.
- **It is a propriety test, not a modulation detector.** Real-valued noise and a constant
  both read as maximally improper despite having no symbol clock at all. A complex tone at
  `f0` is improper at `alpha = 2*f0`.
- **It separates proper from improper, not BPSK from PAM4.** Both are improper; the
  conjugate surface does not distinguish them.
- Everything the ordinary surface is limited by - cyclic aliasing, resolution, averaging -
  applies unchanged, because the framing is identical.

### Validation performed

Synthetic, `Rs = 23,100 Hz` (deliberately not a divisor of `fs`), 25 dB:

| signal | measured `E[s^2]`/power | conjugate at `alpha=0` | features |
|---|---|---|---|
| BPSK | 0.997 | **0.9997** | 3 |
| PAM4 | 0.997 | **0.9996** | 3 |
| QPSK | 0.002 | 0.357 | **0** |
| 8PSK | 0.003 | 0.353 | **0** |
| QAM16 | 0.009 | 0.346 | **0** |

The ordinary surface gives 0.985-0.989 for **all five**, confirming it is not a
discriminator and the conjugate mode adds genuinely new information.

**Real captures (frozen V1, 10 captures at 20 dB and 10 dB): 10/10 classified correctly
by impropriety alone.** BPSK 0.999 / 0.989; QPSK 0.236 / 0.336; 8PSK 0.271 / 0.274;
QAM16 0.403 / 0.397; QAM64 0.308 / 0.225 - against a decision floor of 0.661. On the same
captures the ordinary surface gives BPSK 0.987 and QPSK 0.988, both at 25 kHz.

Controls: proper complex noise -> no features (3 seeds). Complex tone at `f0` -> a feature
at `2*f0` (14,054 / 21,982 / 27,027 Hz for `f0` = 7 / 11 / 13.5 kHz). Real noise and DC ->
correctly improper.

### Performance

Measured at fs = 200 kHz, 32 k samples, defaults: **conjugate 2.07 s vs non-conjugate
3.16 s** - about 35% faster, consistent with one transform per cyclic frequency instead of
two (not 50%, because the framing multiplies and PSD reductions are shared cost). Memory,
surface size and caching are unchanged.

### How an investigator should read it

A large value at `alpha = 0` says the constellation is real-valued or the signal itself is
real - consistent with BPSK or PAM, inconsistent with QPSK, 8PSK or QAM. It is evidence
about constellation symmetry, not a modulation decision, and a frequency offset invalidates
the direct reading. It does not feed the pipeline and changes no reported result.
