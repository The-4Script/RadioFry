"""Cyclostationary Spectral Correlation Density (SCD) for complex baseband IQ.

A wide-sense cyclostationary signal has an autocorrelation that is periodic in time.
Digitally modulated signals are cyclostationary because the symbol clock imposes that
periodicity, which is why the SCD is the natural place to look for a symbol rate that no
ordinary power spectrum can show: the PSD of a modulated signal is smooth, but its
*spectral correlation* is not.

The quantity computed here is the symmetric spectral correlation density

    S_x^alpha(f) = lim  E[ X_T(f + alpha/2) conj(X_T(f - alpha/2)) ]

where `alpha` is the cyclic frequency and `f` the spectral frequency. `alpha = 0` reduces
to the ordinary power spectral density; `alpha != 0` measures how strongly the spectral
component at `f + alpha/2` is correlated with the one at `f - alpha/2`. A linearly
modulated signal at symbol rate Rs has a feature at `alpha = +/- Rs`; stationary noise has
none anywhere except `alpha = 0`.

Estimator
---------
Gardner's *time-smoothed cyclic periodogram*, implemented with exact time-domain
frequency shifts:

    a_alpha(n) = x(n) exp(-j pi alpha n / fs)    ->  spectrum X(f + alpha/2)
    b_alpha(n) = x(n) exp(+j pi alpha n / fs)    ->  spectrum X(f - alpha/2)

Both are framed with a common window and FFT'd, then averaged over frames:

    S^alpha(f_k) = (1/P) sum_p A_p(k) conj(B_p(k)) / (fs * sum(w^2))

Shifting in the time domain rather than re-indexing FFT bins is what keeps this honest.
The half-bin shift `alpha/2` is exact for *any* alpha, so the frequency grid is identical
for every cyclic frequency and both axes stay uniform. Re-indexing bins instead would
force alpha onto even multiples of the bin spacing, or silently interpolate.

The per-frame phase `exp(-j 2 pi alpha p H / fs)` that survives the `A conj(B)` product is
not incidental: averaging it over frames is precisely what makes `alpha` a cyclic
frequency rather than a second time axis. This is why the result is an SCD and not a
spectrogram with a relabelled axis.

Cyclic ambiguity, and why the frame hop is derived rather than chosen
--------------------------------------------------------------------
That same per-frame phase is sampled only once per hop `H`, so it cannot distinguish
`alpha` from `alpha + k fs/H`: at those cyclic frequencies the phase is identically 1,
the temporal average stops discriminating, and *any* correlation between the two shifted
bands survives in full. A pure CW tone then reports coherence 1.000 at every multiple of
`fs/H` - spurious cyclic features in a signal that has none.

The estimator is therefore unambiguous only for `|alpha| < fs / (2H)`, so `H` is
**derived** from the requested cyclic range rather than from an overlap fraction:

    H = floor(fs / (2 * alpha_max))

This is a real trade, not a free choice. With `alpha_max` large the hop is short, frames
overlap heavily, and the span they cover - hence the cyclic resolution `1/T` - is coarse.
The usable number of independent cyclic bins is about `frames / 2` whatever the settings,
which is why `alpha_count` above that is reported as oversampled rather than silently
accepted.

This trap is easy to miss in testing: for a capture at `fs` with an integer
samples-per-symbol, `Rs = fs/sps` frequently lands *exactly* on a multiple of `fs/H`, so a
symbol rate can appear to be recovered perfectly by the very aliasing that would
invalidate it. The tests therefore use symbol rates that are deliberately not multiples of
the hop frequency.

Honesty rules this module follows
---------------------------------
* Only `alpha >= 0` is computed. `S^(-alpha)(f) = conj(S^alpha(f))`, so the magnitude is
  even in alpha and the negative half would be a mirrored copy, not new evidence.
* The frequency axis is a *baseband offset* unless a hardware centre frequency was
  supplied by the caller. This module never invents an absolute RF frequency.
* Cyclic resolution is bounded by the observation length (`1/T`), and frequency resolution
  by the FFT length. Both are reported so a peak is never read as more precise than the
  transform that produced it.
* Degenerate input returns an empty result carrying a warning, never a fabricated
  surface. Nothing here raises into the caller's analysis pipeline.
"""

from dataclasses import dataclass, field

import numpy as np

# Defaults chosen to stay interactive on an ordinary laptop. Cost is
# O(alpha_count * frames * fft_size * log fft_size); see `estimate_cost` below.
#
# The FFT is deliberately short. Independent looks are span/fft_size, and the span is
# frames*hop, so halving the FFT both halves the work and doubles the averaging for the
# same frame budget. Coherence needs that averaging far more than the SCD needs fine
# frequency resolution: the evidence this surface exists to show lives on the cyclic
# axis, and a 128-point FFT still resolves the frequency axis to fs/128.
DEFAULT_FFT_SIZE = 128
MAX_ALPHA_BINS = 512
DEFAULT_MAX_SAMPLES = 262_144
DEFAULT_MAX_FRAMES = 768

# Coherence is a ratio, so in a band holding no signal it divides leakage by leakage and
# saturates towards 1 - which reads as "perfectly cyclostationary" exactly where there is
# nothing to be cyclostationary. A pure CW tone made every cyclic frequency report 1.000
# before this floor existed. Bins whose sideband power sits below this level relative to
# the strongest bin in the capture carry no measurable power, so their coherence is
# reported as zero rather than as a ratio of noise. -60 dB sits far below any occupied
# band while still discarding Hann leakage skirts.
DEFAULT_COHERENCE_FLOOR_DB = -60.0

MIN_FRAMES = 4
MIN_FFT_SIZE = 16
_EPSILON = 1e-30

NORMALIZATIONS = ("coherence", "density")


@dataclass(frozen=True)
class SCDConfig:
    """Bounded computation budget for one SCD estimate.

    The frame hop is not a parameter: it is derived from `alpha_max_hz`, because the hop
    sets the cyclic frequency beyond which the estimate aliases (see the module
    docstring). `max_frames` then bounds the work, and `max_samples` bounds how much of a
    large capture is touched at all. Both truncate the observation window, which coarsens
    cyclic resolution, so both are reported as warnings rather than applied silently.
    """

    fft_size: int = DEFAULT_FFT_SIZE
    alpha_count: int | None = None
    alpha_max_hz: float | None = None
    max_samples: int = DEFAULT_MAX_SAMPLES
    max_frames: int = DEFAULT_MAX_FRAMES
    normalization: str = "coherence"
    coherence_floor_db: float = DEFAULT_COHERENCE_FLOOR_DB
    conjugate: bool = False

    def __post_init__(self) -> None:
        if self.fft_size < MIN_FFT_SIZE:
            raise ValueError(f"fft_size must be >= {MIN_FFT_SIZE}")
        if self.alpha_count is not None and self.alpha_count < 2:
            raise ValueError("alpha_count must be >= 2 when supplied")
        if self.alpha_max_hz is not None and self.alpha_max_hz <= 0:
            raise ValueError("alpha_max_hz must be positive when supplied")
        if self.max_samples < MIN_FFT_SIZE:
            raise ValueError("max_samples is too small to form a single frame")
        if self.max_frames < MIN_FRAMES:
            raise ValueError(f"max_frames must be >= {MIN_FRAMES}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
        if self.coherence_floor_db > 0:
            raise ValueError("coherence_floor_db is relative to the peak, so it must "
                             "be <= 0 dB")


@dataclass(frozen=True)
class SCDResult:
    """A computed SCD surface plus everything needed to interpret it.

    `magnitude` is indexed `[alpha, frequency]`. It is `|S^alpha(f)|` for the "density"
    normalization and the dimensionless spectral coherence for "coherence".
    """

    frequencies_hz: np.ndarray
    alphas_hz: np.ndarray
    magnitude: np.ndarray
    sample_rate: float | None
    samples_used: int
    samples_available: int
    frames: int
    hop: int
    effective_averages: float
    frequency_resolution_hz: float
    alpha_resolution_hz: float
    normalization: str
    config: SCDConfig
    center_frequency_hz: float | None = None
    conjugate: bool = False
    method: str = "time_smoothed_cyclic_periodogram"
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float | None = None

    @property
    def ok(self) -> bool:
        return self.magnitude.size > 0

    @property
    def duration_seconds(self) -> float | None:
        if not self.sample_rate:
            return None
        return self.samples_used / self.sample_rate

    @property
    def frequency_axis_label(self) -> str:
        """Absolute only when a hardware centre frequency was actually supplied."""
        if self.center_frequency_hz is None:
            return "Baseband frequency offset (Hz)"
        return "RF frequency (Hz)"

    @property
    def magnitude_label(self) -> str:
        star = "*" if self.conjugate else ""
        if self.normalization == "coherence":
            return f"Conjugate coherence |C{star}(alpha, f)|" if self.conjugate else                 "Spectral coherence |C(alpha, f)|"
        return f"Conjugate correlation magnitude |S{star}(alpha, f)|" if self.conjugate             else "Spectral correlation magnitude |S(alpha, f)|"

    @property
    def mode_label(self) -> str:
        """Which cyclostationarity this surface measures. Never abbreviated away."""
        return "conjugate SCD" if self.conjugate else "non-conjugate SCD"

    def absolute_frequencies_hz(self) -> np.ndarray:
        """Frequency axis shifted onto the RF grid when, and only when, that is known."""
        if self.center_frequency_hz is None:
            return self.frequencies_hz
        return self.frequencies_hz + float(self.center_frequency_hz)


def _empty_result(
    sample_rate: float | None,
    samples_available: int,
    config: SCDConfig,
    warning: str,
    center_frequency_hz: float | None = None,
) -> SCDResult:
    """A refusal, carrying the reason. Callers must never get a fabricated surface."""

    return SCDResult(
        frequencies_hz=np.empty(0, dtype=np.float64),
        alphas_hz=np.empty(0, dtype=np.float64),
        magnitude=np.empty((0, 0), dtype=np.float64),
        sample_rate=sample_rate,
        samples_used=0,
        samples_available=int(samples_available),
        frames=0,
        hop=0,
        effective_averages=0.0,
        frequency_resolution_hz=float("nan"),
        alpha_resolution_hz=float("nan"),
        normalization=config.normalization,
        config=config,
        center_frequency_hz=center_frequency_hz,
        conjugate=config.conjugate,
        warnings=(warning,),
    )


def estimate_cost(samples: int, sample_rate: float | None, config: SCDConfig) -> dict:
    """Predict the size and work of an SCD before paying for it.

    The GUI uses this to show what a chosen configuration will cost, so an expensive
    setting is a visible decision rather than a hang.
    """

    used = min(int(samples), int(config.max_samples))
    alpha_max = config.alpha_max_hz or ((sample_rate / 8.0) if sample_rate else 0.0)
    hop = (max(1, int(np.floor(sample_rate / (2.0 * alpha_max))))
           if sample_rate and alpha_max else config.fft_size)
    frames = 0 if used < config.fft_size else 1 + (used - config.fft_size) // hop
    if frames > config.max_frames:
        frames = int(config.max_frames)
        used = config.fft_size + (frames - 1) * hop
    if config.alpha_count is None:
        resolution = (sample_rate / used) if (sample_rate and used) else 0.0
        alpha_bins = min(max(2, int(np.ceil(alpha_max / resolution)) + 1),
                         MAX_ALPHA_BINS) if resolution else MAX_ALPHA_BINS
    else:
        alpha_bins = int(config.alpha_count)
    # Two framed FFTs per cyclic frequency for the non-conjugate surface (+alpha/2 and
    # -alpha/2); the conjugate surface reads both factors off one transform.
    transforms = 1 if config.conjugate else 2
    flops = (transforms * alpha_bins * frames * config.fft_size
             * np.log2(max(2, config.fft_size)))
    cells = alpha_bins * config.fft_size
    return {
        "samples_used": used,
        "frames": frames,
        "hop": hop,
        "alpha_bins": alpha_bins,
        "transforms_per_alpha": transforms,
        "surface_cells": int(cells),
        "surface_megabytes": cells * 8 / 1e6,
        "hop": hop,
        "approx_flops": float(flops),
        "duration_seconds": (used / sample_rate) if sample_rate else None,
    }


def compute_scd(
    iq: np.ndarray,
    sample_rate: float | None,
    config: SCDConfig | None = None,
    center_frequency_hz: float | None = None,
) -> SCDResult:
    """Estimate the spectral correlation density of a complex baseband capture.

    Returns an empty `SCDResult` carrying a warning for any input the estimator cannot
    honestly describe, rather than raising. Degenerate captures are a normal condition
    here, not an error the caller should have to guard.
    """

    import time as _time

    started = _time.perf_counter()
    config = config or SCDConfig()
    samples = np.asarray(iq)
    available = int(samples.size)
    warnings: list[str] = []

    if available == 0:
        return _empty_result(sample_rate, 0, config, "The capture contains no samples.",
                             center_frequency_hz)
    if not sample_rate or not np.isfinite(sample_rate) or sample_rate <= 0:
        return _empty_result(
            sample_rate, available, config,
            "SCD needs a positive sample rate to place the frequency and cyclic axes.",
            center_frequency_hz)

    samples = samples.astype(np.complex128, copy=False)
    if not np.all(np.isfinite(samples)):
        # Keep going on a partially corrupt capture, but say so - silently zeroing
        # samples would move energy onto every cyclic frequency at once.
        finite = np.isfinite(samples)
        bad = int(finite.size - int(np.count_nonzero(finite)))
        samples = np.where(finite, samples, 0.0)
        warnings.append(
            f"{bad:,} non-finite sample(s) were replaced with zero before analysis.")

    if available > config.max_samples:
        samples = samples[: config.max_samples]
        warnings.append(
            f"Analyzed the first {config.max_samples:,} of {available:,} samples "
            f"({config.max_samples / available:.0%}). Cyclic resolution is set by this "
            "shorter window, not by the whole capture.")
    used = int(samples.size)

    fft_size = int(config.fft_size)
    if used < fft_size:
        return _empty_result(
            sample_rate, available, config,
            f"Needs at least {fft_size:,} samples for one FFT frame; the capture has "
            f"{used:,}.", center_frequency_hz)

    # The cyclic range is chosen first, because it fixes the hop. Everything downstream
    # (frames, span, cyclic resolution) follows from that choice.
    nyquist_alpha = float(sample_rate) / 2.0
    alpha_max = float(config.alpha_max_hz) if config.alpha_max_hz else float(sample_rate) / 8.0
    if alpha_max > nyquist_alpha:
        alpha_max = nyquist_alpha
        warnings.append(
            f"Cyclic frequency was capped at fs/2 = {nyquist_alpha:,.0f} Hz; beyond that "
            "the +/-alpha/2 shifts leave the sampled band.")

    # H = floor(fs / (2 alpha_max)) keeps every requested alpha below the fs/H aliasing
    # point. Without this the surface grows spurious unit-coherence ridges at multiples
    # of fs/H, which a CW carrier alone is enough to produce.
    hop = max(1, int(np.floor(sample_rate / (2.0 * alpha_max))))
    unambiguous_alpha = sample_rate / (2.0 * hop)
    if alpha_max > unambiguous_alpha:
        # Only reachable when hop has bottomed out at 1 sample.
        alpha_max = unambiguous_alpha
        warnings.append(
            f"Cyclic frequency was capped at fs/2 = {unambiguous_alpha:,.0f} Hz, the "
            "highest this estimator can resolve without aliasing.")

    frames = 1 + (used - fft_size) // hop
    if frames > config.max_frames:
        # Bound the work by frames, not by samples: cost is linear in frames, while the
        # cyclic resolution that matters is set by the span the frames cover. Keeping the
        # frame count and stretching the hop would alias the across-frame phase, so the
        # span is trimmed instead and the coarser resolution is reported.
        frames = int(config.max_frames)
        used = fft_size + (frames - 1) * hop
        samples = samples[:used]
        warnings.append(
            f"Limited to {frames:,} frames ({used:,} samples, "
            f"{used / sample_rate * 1e3:,.1f} ms). Raise the frame budget for finer "
            "cyclic resolution at proportionally higher cost.")
    if frames < MIN_FRAMES:
        return _empty_result(
            sample_rate, available, config,
            f"Only {frames} frame(s) fit in this capture; cyclic averaging needs at "
            f"least {MIN_FRAMES}. Use a shorter FFT or a longer capture.",
            center_frequency_hz)

    # Frames overlap heavily once the hop is short, so the frame count overstates how
    # much averaging actually happened. The independent-look count is the number of
    # non-overlapping window lengths inside the analyzed span; coherence estimated from
    # only a handful of looks is biased high and noisy, so it is reported and warned on.
    effective_averages = float(used) / float(fft_size)
    if effective_averages < 8.0:
        warnings.append(
            f"Only about {effective_averages:.1f} independent looks were averaged. "
            "Coherence is biased high and noisy at this setting - lower the cyclic range "
            "(which lengthens the hop) or raise the frame budget before trusting weak "
            "features.")

    power = float(np.mean(np.abs(samples) ** 2))
    if not np.isfinite(power) or power <= _EPSILON:
        return _empty_result(
            sample_rate, available, config,
            "The capture carries no measurable power, so spectral correlation is "
            "undefined.", center_frequency_hz)

    # --- axes ----------------------------------------------------------------------
    # Frequency: the standard fftshifted grid, so index 0 is the most negative offset.
    frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    frequency_resolution = float(sample_rate) / fft_size

    # Cyclic: range already fixed above by the hop; resolved no finer than 1/T, where T
    # is the span the frames actually cover.
    observation_seconds = used / float(sample_rate)
    alpha_resolution = 1.0 / observation_seconds

    # The grid is derived from the resolution by default. A coarser grid does not merely
    # look blocky - a cyclic feature is only about `1/T` wide, so a grid stepping more
    # than that walks straight past real peaks and reports whatever noise sits on the
    # sample points instead. Undersampling here cost a genuine 23.1 kHz symbol-rate line
    # during development.
    if config.alpha_count is None:
        wanted = int(np.ceil(alpha_max / alpha_resolution)) + 1
        alpha_count = min(max(2, wanted), MAX_ALPHA_BINS)
        if wanted > MAX_ALPHA_BINS:
            warnings.append(
                f"The cyclic grid was capped at {MAX_ALPHA_BINS:,} points; fully "
                f"resolving 0-{alpha_max:,.0f} Hz at {alpha_resolution:,.1f} Hz would "
                f"need {wanted:,}. Narrow features between grid points can be missed - "
                "lower the cyclic range to see them.")
    else:
        alpha_count = int(config.alpha_count)

    alphas = np.linspace(0.0, alpha_max, alpha_count)
    grid_step = float(alphas[1] - alphas[0]) if alphas.size > 1 else alpha_max
    if grid_step > alpha_resolution * 1.5:
        warnings.append(
            f"The cyclic grid steps {grid_step:,.1f} Hz but this capture resolves "
            f"{alpha_resolution:,.1f} Hz, so features narrower than the step can fall "
            "between rows and be missed.")
    elif alphas.size > 1 and grid_step < alpha_resolution * 0.9:
        # A derived grid targets the resolution exactly, so it lands a fraction of a
        # percent either side of it. Only a meaningfully finer grid is worth mentioning.
        warnings.append(
            f"The cyclic grid is finer ({grid_step:,.1f} Hz) than this capture can "
            f"resolve ({alpha_resolution:,.1f} Hz). Neighbouring alpha rows are not "
            "independent measurements.")

    # --- framing -------------------------------------------------------------------
    window = np.hanning(fft_size).astype(np.float64)
    window_power = float(np.sum(window ** 2))
    starts = np.arange(frames, dtype=np.int64) * hop
    frame_index = starts[:, None] + np.arange(fft_size, dtype=np.int64)[None, :]
    framed = samples[frame_index]                                   # (frames, fft_size)

    # The frequency shift splits into a within-frame ramp and a per-frame phase. The
    # per-frame phase is the exp(-j 2 pi alpha n) of the SCD definition and must be kept
    # on the absolute sample index - dropping it would average away the cyclic structure
    # and leave an ordinary cross-spectrum.
    within = np.arange(fft_size, dtype=np.float64)[None, :]         # (1, fft_size)
    across = starts.astype(np.float64)[:, None]                     # (frames, 1)

    scale = float(sample_rate) * window_power
    magnitude = np.empty((alphas.size, fft_size), dtype=np.float64)

    # Reference PSD (the alpha = 0 row) sets the power floor below which coherence is
    # not a measurement. Computed once, from the same frames and scaling as every row.
    reference = np.fft.fft(framed * window, axis=1)
    reference_psd = np.mean(np.abs(reference) ** 2, axis=0) / scale
    peak_psd = float(np.max(reference_psd))
    psd_floor = peak_psd * (10.0 ** (config.coherence_floor_db / 10.0))
    occupied = int(np.count_nonzero(reference_psd >= psd_floor))
    if config.normalization == "coherence" and occupied < 2:
        warnings.append(
            "Almost no frequency bin carries measurable power, so spectral coherence "
            "is reported as zero nearly everywhere.")

    # Single precision for the transform stage: the inner loop is memory-bound, not
    # precision-bound, and every quantity here is a ratio or a magnitude of averaged
    # spectra rather than a long accumulation. The reductions below promote back to
    # double, so the reported surface keeps full precision.
    framed = framed.astype(np.complex64, copy=False)
    buffer = np.empty_like(framed)

    # Index of -f for each f on the unshifted FFT grid, so X(alpha/2 - f) can be read
    # off the same transform as X(f + alpha/2). Used by the conjugate estimator only.
    mirror = (-np.arange(fft_size)) % fft_size

    for row, alpha in enumerate(alphas):
        # a -> X(f + alpha/2), b -> X(f - alpha/2). Exact for any alpha.
        ramp = np.exp(-1j * np.pi * alpha * within / sample_rate)
        phase = np.exp(-1j * np.pi * alpha * across / sample_rate)

        np.multiply(framed, (window * ramp).astype(np.complex64), out=buffer)
        upper = np.fft.fft(buffer, axis=1) * phase
        upper_psd = np.mean(np.abs(upper) ** 2, axis=0) / scale

        if config.conjugate:
            # S*^alpha(f) = E[X(f + alpha/2) X(alpha/2 - f)]. The second factor is the
            # SAME shifted transform read at -f, so the conjugate surface costs one FFT
            # per cyclic frequency instead of two. Note there is no conjugation on the
            # second factor - that is exactly what makes this sensitive to impropriety.
            partner = upper[:, mirror]
            partner_psd = upper_psd[mirror]
            cross = np.mean(upper * partner, axis=0) / scale
        else:
            np.multiply(framed, (window * np.conj(ramp)).astype(np.complex64),
                        out=buffer)
            partner = np.fft.fft(buffer, axis=1) * np.conj(phase)
            partner_psd = np.mean(np.abs(partner) ** 2, axis=0) / scale
            cross = np.mean(upper * np.conj(partner), axis=0) / scale

        if config.normalization == "coherence":
            # |S^alpha(f)| divided by the geometric mean of the ordinary PSD at the two
            # frequencies being related: dimensionless and bounded by 1, which is what
            # makes rows at different f comparable. The denominator uses the ordinary
            # PSD in both modes, because that is the power actually available to
            # correlate.
            values = np.abs(cross) / np.maximum(
                np.sqrt(upper_psd * partner_psd), _EPSILON)
            np.clip(values, 0.0, 1.0, out=values)
            # Both bands must actually carry power. Without this, an empty band divides
            # leakage by leakage and saturates to 1, reporting perfect cyclostationarity
            # precisely where there is no signal to be cyclostationary.
            values[(upper_psd < psd_floor) | (partner_psd < psd_floor)] = 0.0
        else:
            values = np.abs(cross)
        magnitude[row] = np.fft.fftshift(values)

    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

    return SCDResult(
        frequencies_hz=frequencies,
        alphas_hz=alphas,
        magnitude=magnitude,
        sample_rate=float(sample_rate),
        samples_used=used,
        samples_available=available,
        frames=int(frames),
        hop=int(hop),
        effective_averages=effective_averages,
        frequency_resolution_hz=frequency_resolution,
        alpha_resolution_hz=float(alpha_resolution),
        normalization=config.normalization,
        config=config,
        center_frequency_hz=center_frequency_hz,
        conjugate=config.conjugate,
        warnings=tuple(warnings),
        elapsed_seconds=_time.perf_counter() - started,
    )


@dataclass(frozen=True)
class CycleFeature:
    """One candidate cyclic frequency read off the SCD surface."""

    alpha_hz: float
    magnitude: float
    peak_to_background: float
    frequency_hz: float = field(default=float("nan"))


def cycle_profile(result: SCDResult) -> np.ndarray:
    """Maximum correlation at each cyclic frequency: the alpha-domain evidence.

    Reducing over `f` with a maximum rather than a mean keeps a narrowband feature
    visible; averaging would dilute it across an otherwise empty band.
    """

    if not result.ok:
        return np.empty(0, dtype=np.float64)
    return np.max(result.magnitude, axis=1)


def _dc_ridge_end(profile: np.ndarray) -> int:
    """First index where the cyclic profile stops falling away from `alpha = 0`.

    The `alpha = 0` ridge is the PSD and is as wide as the signal's occupied bandwidth,
    which no fixed bin count can bracket: a narrowband tone's ridge spans a few bins, a
    wideband signal's spans many. Searching only beyond the ridge's own foot adapts to
    whatever the capture contains. A profile that never stops falling - a CW tone, say -
    has no region left to search, which is the right answer for a signal with no symbol
    clock.
    """

    if profile.size < 2:
        return profile.size
    # A ridge that plateaus - a CW tone saturates coherence across its whole leakage
    # width - wobbles by an ULP or two, which a strict comparison reads as the ridge
    # ending and a new peak starting. The tolerance travels with the profile's own
    # range, so it stays far below any genuine rise towards a symbol-rate feature.
    span = float(np.max(profile) - np.min(profile))
    tolerance = max(span * 0.01, 1e-12)
    index = 1
    while index < profile.size and profile[index] <= profile[index - 1] + tolerance:
        index += 1
    return min(index, profile.size)


def dominant_cycle_frequencies(
    result: SCDResult,
    count: int = 5,
    dc_guard_resolutions: float = 8.0,
    min_ratio: float = 1.5,
    bias_multiple: float = 1.5,
) -> list[CycleFeature]:
    """Strongest genuine cyclic-frequency peaks, strongest first.

    A candidate must be a *local maximum* of the cyclic profile, not merely a large
    value. That distinction matters: the `alpha = 0` ridge is as wide as the signal's
    occupied bandwidth, so a plain "largest value away from DC" search walks down its
    shoulder and reports a cycle frequency where there is only the PSD. A pure CW tone
    produces a monotonically decaying shoulder and therefore no peaks at all, which is
    the correct answer for a signal carrying no symbol clock.

    Each feature also carries its peak-to-background ratio; near 1 means the surface is
    flat there and the peak is not evidence. This reports candidates only - it does not
    claim a symbol rate, which remains the job of `parameter_estimation`.
    """

    from scipy.signal import find_peaks

    profile = cycle_profile(result)
    if profile.size < 3:
        return []

    if result.conjugate:
        # alpha = 0 is not a ridge to step over here, it is the measurement. The
        # conjugate spectrum at alpha = 0 is the signal's own impropriety: it vanishes
        # for any circularly symmetric (proper) signal - QPSK, QAM, complex noise - and
        # is non-zero for a real-valued constellation such as BPSK or PAM. Excluding it
        # would discard the single most useful thing this surface offers.
        start = 0
    else:
        # The alpha = 0 response has the width of the analysis window's cyclic main
        # lobe, a few multiples of 1/T, so its skirt is excluded the way a DC bin is
        # excluded from a spectrum. A constant signal and a CW tone - which are the same
        # waveform up to a frequency shift, and neither of which is cyclostationary -
        # both produce artifacts inside exactly this band.
        guard_hz = dc_guard_resolutions * float(result.alpha_resolution_hz)
        guard_bins = int(np.searchsorted(result.alphas_hz, guard_hz)) if np.isfinite(
            guard_hz) else 0
        start = max(_dc_ridge_end(profile), min(guard_bins, profile.size))
    searchable = profile[start:]
    if searchable.size < 2:
        return []

    # A feature sitting on the last cyclic bin is real but has no right-hand neighbour to
    # be a local maximum against. Pad below the minimum so the edge can win, and drop the
    # sentinel again when mapping back to alpha indices.
    sentinel = float(np.min(searchable)) - 1.0
    padded = np.concatenate(([sentinel], searchable, [sentinel]))
    span = float(np.max(searchable) - np.min(searchable))
    peaks, _ = find_peaks(padded, prominence=max(span * 0.1, 1e-12))
    if peaks.size == 0:
        return []
    indices = peaks - 1 + start

    background = float(np.median(searchable))
    if background <= _EPSILON:
        # Most of the band was masked for want of power, so a median of zero would make
        # every ratio infinite. Fall back to the smallest real value present.
        positive = searchable[searchable > 0]
        background = float(np.min(positive)) if positive.size else 0.0

    # Coherence estimated from N independent looks has an expected magnitude of about
    # 1/sqrt(N) even between completely uncorrelated bands. Anything at or below that is
    # the estimator's own bias, not a cyclic feature, so the floor is derived from the
    # averaging actually achieved rather than fixed.
    #
    # The profile is a MAXIMUM over F frequency bins, which is a multiple comparison: the
    # largest of F noise draws sits well above their mean, growing as sqrt(ln F). Ignoring
    # that let proper complex noise clear a 1/sqrt(N) floor and be reported as a cyclic
    # feature. Folding it in costs nothing and removes that whole class of false positive.
    floor = 0.0
    if result.normalization == "coherence" and result.effective_averages > 0:
        bins = max(2, int(result.frequencies_hz.size))
        floor = (bias_multiple * np.sqrt(np.log(bins))
                 / np.sqrt(result.effective_averages))

    # Ranked by correlation strength rather than prominence: harmonics of a symbol rate
    # are all genuine features, and the strongest is the more useful one to show first.
    order = indices[np.argsort(profile[indices])[::-1]]
    features: list[CycleFeature] = []
    for index in order:
        if len(features) >= count:
            break
        index = int(index)
        if profile[index] < floor:
            continue
        ratio = float(profile[index] / (background + _EPSILON))
        if ratio < min_ratio:
            continue
        row = result.magnitude[index]
        features.append(CycleFeature(
            alpha_hz=float(result.alphas_hz[index]),
            magnitude=float(profile[index]),
            peak_to_background=ratio,
            frequency_hz=float(result.frequencies_hz[int(np.argmax(row))]),
        ))
    return features


def decimate_surface(
    result: SCDResult,
    max_alpha: int = 160,
    max_frequency: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin the surface to a size a browser can render, keeping real sampled values.

    Decimation by striding, not smoothing or interpolation: every point still plotted is
    a computed value. Peaks can be missed by striding, which is why the 2D heatmap uses
    the full surface and only the 3D view is thinned.
    """

    if not result.ok:
        return (np.empty(0), np.empty(0), np.empty((0, 0)))
    alpha_step = max(1, int(np.ceil(result.alphas_hz.size / max(1, max_alpha))))
    frequency_step = max(1, int(np.ceil(result.frequencies_hz.size / max(1, max_frequency))))
    return (
        result.frequencies_hz[::frequency_step],
        result.alphas_hz[::alpha_step],
        result.magnitude[::alpha_step, ::frequency_step],
    )
