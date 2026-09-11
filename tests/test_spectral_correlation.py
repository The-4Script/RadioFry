"""The SCD estimator measures cyclic frequency, not a relabelled time axis.

The dangerous failure for this feature is not a crash - it is a surface that looks like
spectral correlation, carries confident-looking peaks, and is actually an artifact. Three
such artifacts were found and fixed during development, and each has a regression test
here:

* coherence saturating to 1.0 in bands holding no power (a CW carrier reported perfect
  cyclostationarity everywhere),
* cyclic aliasing at multiples of `fs/hop`, which a symbol rate of `fs/sps` lands on
  exactly - so an aliased estimator appears to recover the symbol rate perfectly,
* a cyclic grid coarser than the cyclic resolution, which steps straight over real peaks.

The tests therefore use symbol rates chosen NOT to divide the sample rate.
"""

import numpy as np
import pytest

from radiofry.dsp.spectral_correlation import (
    MAX_ALPHA_BINS,
    SCDConfig,
    SCDResult,
    compute_scd,
    cycle_profile,
    decimate_surface,
    dominant_cycle_frequencies,
    estimate_cost,
)

FS = 200_000.0
FAST = SCDConfig(fft_size=64, max_frames=256, alpha_max_hz=30_000.0)


def _bpsk(symbol_rate_hz: float, samples: int = 16_384, snr_db: float = 25.0,
          seed: int = 11) -> np.ndarray:
    """BPSK at an arbitrary symbol rate, so `sps` need not be an integer.

    Deliberately not `np.repeat(bits, sps)`: an integer `sps` puts Rs exactly on the
    cyclic aliasing grid, where a broken estimator still appears to find it.
    """
    rng = np.random.default_rng(seed)
    sps = FS / symbol_rate_hz
    symbols = rng.integers(0, 2, int(np.ceil(samples / sps)) + 2) * 2 - 1
    wave = symbols[np.floor(np.arange(samples) / sps).astype(int)].astype(np.complex128)
    noise = (rng.normal(size=samples) + 1j * rng.normal(size=samples)) / np.sqrt(2)
    return (wave + 10 ** (-snr_db / 20) * noise).astype(np.complex64)


def _noise(samples: int = 16_384, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.normal(size=samples) + 1j * rng.normal(size=samples))
            / np.sqrt(2)).astype(np.complex64)


# --- shape, axes and units -------------------------------------------------------------


def test_a_valid_capture_produces_a_surface_with_matching_axes() -> None:
    result = compute_scd(_bpsk(23_100.0), FS, FAST)

    assert result.ok
    assert result.magnitude.shape == (result.alphas_hz.size, result.frequencies_hz.size)
    assert result.magnitude.ndim == 2


def test_the_frequency_axis_spans_the_sampled_band() -> None:
    result = compute_scd(_bpsk(23_100.0), FS, FAST)

    assert result.frequencies_hz[0] == pytest.approx(-FS / 2)
    assert result.frequencies_hz[-1] < FS / 2
    assert result.frequency_resolution_hz == pytest.approx(FS / FAST.fft_size)
    assert np.all(np.diff(result.frequencies_hz) > 0), "must be monotonic for plotting"


def test_the_cyclic_axis_is_non_negative_and_bounded_by_the_requested_range() -> None:
    result = compute_scd(_bpsk(23_100.0), FS, FAST)

    assert result.alphas_hz[0] == 0.0
    assert result.alphas_hz[-1] <= FAST.alpha_max_hz + 1e-9
    assert np.all(np.diff(result.alphas_hz) > 0)


def test_the_frequency_axis_scales_with_the_sample_rate() -> None:
    """The axis is physical, not bin indices."""
    slow = compute_scd(_bpsk(23_100.0), FS, FAST)
    fast = compute_scd(_bpsk(46_200.0), 2 * FS, FAST)

    assert fast.frequencies_hz[0] == pytest.approx(2 * slow.frequencies_hz[0])
    assert fast.frequency_resolution_hz == pytest.approx(
        2 * slow.frequency_resolution_hz)


def test_baseband_captures_are_not_given_an_invented_carrier() -> None:
    baseband = compute_scd(_bpsk(23_100.0), FS, FAST)
    tuned = compute_scd(_bpsk(23_100.0), FS, FAST, center_frequency_hz=433.92e6)

    assert baseband.center_frequency_hz is None
    assert "Baseband" in baseband.frequency_axis_label
    assert np.array_equal(baseband.absolute_frequencies_hz(), baseband.frequencies_hz)
    assert "RF" in tuned.frequency_axis_label
    assert tuned.absolute_frequencies_hz()[0] == pytest.approx(
        433.92e6 + tuned.frequencies_hz[0])


# --- the alpha axis is genuinely cyclic frequency ---------------------------------------


@pytest.mark.parametrize("symbol_rate", [9_300.0, 17_000.0, 23_100.0])
def test_a_digitally_modulated_signal_peaks_at_its_symbol_rate(symbol_rate: float) -> None:
    """The core scientific claim: alpha = Rs carries a spectral-correlation feature.

    Harmonics at k*Rs are equally genuine, so the assertion is that Rs is among the
    detected peaks rather than that it is the single largest.
    """
    result = compute_scd(_bpsk(symbol_rate), FS)
    features = dominant_cycle_frequencies(result, count=6)

    assert features, "a modulated signal must show cyclic structure"
    tolerance = max(3 * result.alpha_resolution_hz, 0.03 * symbol_rate)
    assert any(abs(f.alpha_hz - symbol_rate) <= tolerance for f in features), (
        f"expected a peak at Rs={symbol_rate:,.0f} Hz, got "
        f"{[round(f.alpha_hz) for f in features]}")


def test_the_symbol_rate_peak_is_not_an_artifact_of_the_frame_hop() -> None:
    """Rs = fs/sps lands exactly on the fs/hop aliasing grid for integer sps.

    An estimator that aliases reports that rate perfectly while getting every other rate
    wrong, so this pins the off-grid case that the aliasing cannot fake.
    """
    off_grid = 23_100.0
    result = compute_scd(_bpsk(off_grid), FS)

    assert FS / result.hop >= 2 * result.alphas_hz[-1] - 1e-6, (
        "the hop must keep every reported alpha below the aliasing point")
    features = dominant_cycle_frequencies(result, count=6)
    assert any(abs(f.alpha_hz - off_grid) <= 3 * result.alpha_resolution_hz
               for f in features)


def test_stationary_noise_shows_no_cyclic_feature() -> None:
    result = compute_scd(_noise(), FS)

    assert result.ok
    assert dominant_cycle_frequencies(result) == [], (
        "white noise is stationary and must not produce cyclic peaks")


def test_a_pure_tone_shows_no_cyclic_feature() -> None:
    """A CW carrier is not cyclostationary.

    Before the power floor existed this returned coherence 1.000 at every multiple of
    fs/hop - the single most misleading output this module could produce.
    """
    n = np.arange(16_384)
    tone = np.exp(2j * np.pi * 12_000 * n / FS).astype(np.complex64)

    result = compute_scd(tone, FS)

    assert result.ok
    assert dominant_cycle_frequencies(result) == []


def test_a_constant_signal_shows_no_cyclic_feature() -> None:
    result = compute_scd(np.ones(16_384, dtype=np.complex64), FS)

    assert dominant_cycle_frequencies(result) == []


def test_an_empty_band_reports_zero_coherence_not_saturated_coherence() -> None:
    """Where there is no power there is nothing to correlate."""
    n = np.arange(16_384)
    tone = np.exp(2j * np.pi * 5_000 * n / FS).astype(np.complex64)

    result = compute_scd(tone, FS, SCDConfig(fft_size=64, max_frames=256,
                                             alpha_max_hz=40_000.0))

    profile = cycle_profile(result)
    far = result.alphas_hz > 25_000.0
    # Most bins are masked outright; the residue is coherence between the window's own
    # leakage tails, which a 64-point Hann spreads widely. The point of the test is that
    # this stays negligible instead of saturating to 1.
    assert np.max(profile[far]) < 0.05, (
        "cyclic frequencies far from the tone must be masked, not saturated")
    assert dominant_cycle_frequencies(result) == []


# --- normalization ----------------------------------------------------------------------


def test_coherence_is_bounded_and_unity_at_zero_cyclic_frequency() -> None:
    result = compute_scd(_bpsk(23_100.0), FS, FAST)

    assert result.normalization == "coherence"
    assert np.all(result.magnitude >= 0.0)
    assert np.all(result.magnitude <= 1.0 + 1e-9)
    # alpha = 0 is the PSD correlated with itself.
    assert np.max(result.magnitude[0]) == pytest.approx(1.0, abs=1e-6)


def test_density_normalization_returns_unbounded_magnitudes() -> None:
    config = SCDConfig(fft_size=64, max_frames=256, alpha_max_hz=30_000.0,
                       normalization="density")

    result = compute_scd(_bpsk(23_100.0) * 1_000.0, FS, config)

    assert result.normalization == "density"
    assert np.max(result.magnitude) > 1.0, "a density is not a bounded ratio"
    assert "magnitude" in result.magnitude_label.lower()


def test_every_value_is_finite() -> None:
    for signal in (_bpsk(23_100.0), _noise(), np.ones(16_384, dtype=np.complex64)):
        result = compute_scd(signal, FS, FAST)
        assert np.all(np.isfinite(result.magnitude))


# --- degenerate input is refused, never fabricated ----------------------------------------


def test_an_empty_capture_is_refused_with_a_reason() -> None:
    result = compute_scd(np.empty(0, dtype=np.complex64), FS)

    assert not result.ok
    assert result.magnitude.size == 0
    assert result.warnings and "no samples" in result.warnings[0].lower()


@pytest.mark.parametrize("rate", [None, 0.0, -1.0, float("nan")])
def test_an_unusable_sample_rate_is_refused(rate) -> None:
    result = compute_scd(_bpsk(23_100.0), rate)

    assert not result.ok
    assert result.warnings


def test_a_capture_shorter_than_one_frame_is_refused() -> None:
    result = compute_scd(_bpsk(23_100.0)[:32], FS, FAST)

    assert not result.ok
    assert "frame" in result.warnings[0].lower()


def test_a_capture_with_too_few_frames_is_refused() -> None:
    """Long enough for one frame, too short for the cyclic average to mean anything."""
    result = compute_scd(_bpsk(23_100.0)[:70], FS, FAST)

    assert not result.ok
    assert result.warnings and "frame" in result.warnings[0].lower()


def test_a_zero_power_capture_is_refused() -> None:
    result = compute_scd(np.zeros(16_384, dtype=np.complex64), FS, FAST)

    assert not result.ok
    assert "power" in result.warnings[0].lower()


def test_non_finite_samples_are_replaced_and_reported() -> None:
    signal = _bpsk(23_100.0).astype(np.complex128)
    signal[100] = np.nan
    signal[200] = np.inf

    result = compute_scd(signal, FS, FAST)

    assert result.ok
    assert np.all(np.isfinite(result.magnitude))
    assert any("non-finite" in w for w in result.warnings)


def test_nothing_raises_for_any_degenerate_input() -> None:
    """The normal pipeline must never break because SCD could not be computed."""
    for signal in (np.empty(0, dtype=np.complex64),
                   np.zeros(4, dtype=np.complex64),
                   np.ones(1, dtype=np.complex64),
                   np.full(1_000, np.nan, dtype=np.complex128)):
        for rate in (None, 0.0, FS):
            result = compute_scd(signal, rate)
            assert isinstance(result, SCDResult)


# --- bounded computation ------------------------------------------------------------------


def test_a_large_capture_is_truncated_and_says_so() -> None:
    config = SCDConfig(fft_size=64, max_frames=64, max_samples=4_096,
                       alpha_max_hz=30_000.0)

    result = compute_scd(_bpsk(23_100.0, samples=65_536), FS, config)

    assert result.samples_used <= 4_096
    assert result.samples_available == 65_536
    assert any("Analyzed the first" in w or "Limited to" in w for w in result.warnings)


def test_the_cost_estimate_matches_what_the_estimator_actually_does() -> None:
    config = SCDConfig(fft_size=64, max_frames=256, alpha_max_hz=30_000.0)
    signal = _bpsk(23_100.0, samples=65_536)

    predicted = estimate_cost(signal.size, FS, config)
    result = compute_scd(signal, FS, config)

    assert predicted["frames"] == result.frames
    assert predicted["hop"] == result.hop
    assert predicted["samples_used"] == result.samples_used


def test_the_cyclic_grid_is_derived_from_the_resolution() -> None:
    """A grid coarser than 1/T steps over real peaks; a finer one is not independent."""
    result = compute_scd(_bpsk(23_100.0), FS)

    step = float(result.alphas_hz[1] - result.alphas_hz[0])
    assert step <= result.alpha_resolution_hz * 1.5
    assert result.alphas_hz.size <= MAX_ALPHA_BINS


def test_an_explicit_alpha_count_is_honoured() -> None:
    result = compute_scd(_bpsk(23_100.0), FS,
                         SCDConfig(fft_size=64, max_frames=256, alpha_count=33,
                                   alpha_max_hz=30_000.0))

    assert result.alphas_hz.size == 33


def test_the_hop_is_derived_from_the_requested_cyclic_range() -> None:
    narrow = compute_scd(_bpsk(23_100.0), FS,
                         SCDConfig(fft_size=64, max_frames=256, alpha_max_hz=5_000.0))
    wide = compute_scd(_bpsk(23_100.0), FS,
                       SCDConfig(fft_size=64, max_frames=256, alpha_max_hz=50_000.0))

    assert narrow.hop > wide.hop, "a wider cyclic range needs a shorter hop"
    for result in (narrow, wide):
        assert FS / result.hop >= 2 * result.alphas_hz[-1] - 1e-6


def test_low_averaging_is_warned_about() -> None:
    config = SCDConfig(fft_size=256, max_frames=8, alpha_max_hz=90_000.0)

    result = compute_scd(_bpsk(23_100.0), FS, config)

    assert result.effective_averages < 8
    assert any("independent looks" in w for w in result.warnings)


# --- configuration validation --------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"fft_size": 4},
    {"alpha_count": 1},
    {"alpha_max_hz": -5.0},
    {"max_frames": 1},
    {"normalization": "nonsense"},
    {"coherence_floor_db": 6.0},
])
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SCDConfig(**kwargs)


# --- presentation helpers --------------------------------------------------------------------


def test_decimation_keeps_real_sampled_values_and_shrinks_the_surface() -> None:
    result = compute_scd(_bpsk(23_100.0), FS)

    frequencies, alphas, magnitude = decimate_surface(result, max_alpha=16,
                                                      max_frequency=16)

    assert magnitude.shape == (alphas.size, frequencies.size)
    assert alphas.size <= result.alphas_hz.size
    assert frequencies.size <= result.frequencies_hz.size
    assert alphas[0] == result.alphas_hz[0]
    assert np.isin(magnitude, result.magnitude).all(), (
        "decimation must subsample real values, never interpolate new ones")


def test_decimating_an_empty_result_is_safe() -> None:
    frequencies, alphas, magnitude = decimate_surface(
        compute_scd(np.empty(0, dtype=np.complex64), FS))

    assert magnitude.size == 0 and alphas.size == 0 and frequencies.size == 0


def test_the_cycle_profile_matches_the_surface() -> None:
    result = compute_scd(_bpsk(23_100.0), FS, FAST)

    profile = cycle_profile(result)

    assert profile.size == result.alphas_hz.size
    assert profile == pytest.approx(np.max(result.magnitude, axis=1))


def test_features_carry_interpretable_metadata() -> None:
    result = compute_scd(_bpsk(23_100.0), FS)

    for feature in dominant_cycle_frequencies(result, count=3):
        assert 0.0 <= feature.alpha_hz <= result.alphas_hz[-1] + 1e-9
        assert feature.magnitude > 0
        assert feature.peak_to_background > 1.0
        assert result.frequencies_hz[0] <= feature.frequency_hz <= result.frequencies_hz[-1]
