"""Conjugate spectral correlation measures impropriety, not just a symbol clock.

The conjugate cyclic spectrum

    S*^alpha(f) = E[ X(f + alpha/2) X(alpha/2 - f) ]

differs from the ordinary one by a single missing conjugation, and that difference is
what makes it sensitive to *impropriety*: a constellation with `E[s^2] != 0` has a
conjugate spectrum, a circularly symmetric one does not. BPSK (+/-1, `E[s^2] = 1`) is
improper; QPSK symbols are `exp(j*pi*(2k+1)/4)`, so `s^2` walks `{j,-1,-j,1}` and averages
to zero. That is the discriminator these tests pin, and it is the thing the ordinary
surface cannot do - both constellations carry a symbol-rate line.

Two cases here would be *wrong* answers for the ordinary surface and are correct for this
one: a real-valued signal and a constant both have no symbol clock at all, yet both are
maximally improper. The conjugate surface is a propriety test, not a modulation detector.
"""

import numpy as np
import pytest

from radiofry.dsp.spectral_correlation import (
    SCDConfig,
    compute_scd,
    cycle_profile,
    dominant_cycle_frequencies,
    estimate_cost,
)

FS = 200_000.0
CONJ = SCDConfig(alpha_max_hz=30_000.0, conjugate=True)
NONCONJ = SCDConfig(alpha_max_hz=30_000.0, conjugate=False)


def _symbols(kind: str, count: int, rng) -> np.ndarray:
    if kind == "BPSK":
        return (rng.integers(0, 2, count) * 2 - 1).astype(complex)
    if kind == "QPSK":
        return np.exp(1j * np.pi * (2 * rng.integers(0, 4, count) + 1) / 4)
    if kind == "8PSK":
        return np.exp(2j * np.pi * rng.integers(0, 8, count) / 8)
    if kind == "PAM4":
        return (2 * rng.integers(0, 4, count) - 3).astype(complex) / np.sqrt(5)
    if kind == "QAM16":
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        return rng.choice(levels, count) + 1j * rng.choice(levels, count)
    raise ValueError(kind)


def _modulated(kind: str, symbol_rate_hz: float = 23_100.0, samples: int = 16_384,
               snr_db: float = 25.0, seed: int = 17) -> np.ndarray:
    """A linear modulation at a symbol rate that does not divide the sample rate."""
    rng = np.random.default_rng(seed)
    sps = FS / symbol_rate_hz
    symbols = _symbols(kind, int(np.ceil(samples / sps)) + 2, rng)
    wave = symbols[np.floor(np.arange(samples) / sps).astype(int)]
    noise = (rng.normal(size=samples) + 1j * rng.normal(size=samples)) / np.sqrt(2)
    return (wave + 10 ** (-snr_db / 20) * noise).astype(np.complex64)


def _noise(samples: int = 16_384, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.normal(size=samples) + 1j * rng.normal(size=samples))
            / np.sqrt(2)).astype(np.complex64)


# --- the two modes stay separate -----------------------------------------------------


def test_the_conjugate_mode_is_labelled_and_never_silently_mixed() -> None:
    conjugate = compute_scd(_modulated("BPSK"), FS, CONJ)
    ordinary = compute_scd(_modulated("BPSK"), FS, NONCONJ)

    assert conjugate.conjugate is True
    assert ordinary.conjugate is False
    assert conjugate.mode_label == "conjugate SCD"
    assert ordinary.mode_label == "non-conjugate SCD"
    assert "Conjugate" in conjugate.magnitude_label
    assert "Conjugate" not in ordinary.magnitude_label
    assert not np.allclose(conjugate.magnitude, ordinary.magnitude), (
        "the two modes must not return the same surface")


def test_the_conjugate_surface_shares_the_axes_of_the_ordinary_one() -> None:
    conjugate = compute_scd(_modulated("BPSK"), FS, CONJ)
    ordinary = compute_scd(_modulated("BPSK"), FS, NONCONJ)

    assert np.array_equal(conjugate.frequencies_hz, ordinary.frequencies_hz)
    assert np.array_equal(conjugate.alphas_hz, ordinary.alphas_hz)
    assert conjugate.magnitude.shape == ordinary.magnitude.shape
    assert conjugate.frequency_resolution_hz == ordinary.frequency_resolution_hz
    assert conjugate.hop == ordinary.hop


def test_the_ordinary_surface_is_unchanged_by_the_refinement() -> None:
    """Default construction must still produce the non-conjugate estimate."""
    assert SCDConfig().conjugate is False
    assert compute_scd(_modulated("BPSK"), FS, SCDConfig()).conjugate is False


# --- impropriety is what it measures ---------------------------------------------------


@pytest.mark.parametrize("kind", ["BPSK", "PAM4"])
def test_an_improper_constellation_shows_conjugate_cyclostationarity(kind: str) -> None:
    result = compute_scd(_modulated(kind), FS, CONJ)

    profile = cycle_profile(result)
    assert profile[0] > 0.9, (
        f"{kind} is improper; alpha=0 conjugate coherence was {profile[0]:.3f}")
    assert dominant_cycle_frequencies(result), "improper signals carry conjugate features"


@pytest.mark.parametrize("kind", ["QPSK", "8PSK", "QAM16"])
def test_a_proper_constellation_shows_no_conjugate_cyclostationarity(kind: str) -> None:
    result = compute_scd(_modulated(kind), FS, CONJ)

    profile = cycle_profile(result)
    assert profile[0] < 0.6, (
        f"{kind} is proper; alpha=0 conjugate coherence was {profile[0]:.3f}")
    assert dominant_cycle_frequencies(result) == []


def test_conjugate_separates_bpsk_from_qpsk_where_the_ordinary_surface_cannot() -> None:
    """The point of the refinement, asserted as a comparison rather than a claim.

    The ordinary SCD sees a symbol-rate line for both, because both have one. Only the
    conjugate surface distinguishes the real-valued constellation from the circular one.
    """
    bpsk_conjugate = cycle_profile(compute_scd(_modulated("BPSK"), FS, CONJ))
    qpsk_conjugate = cycle_profile(compute_scd(_modulated("QPSK"), FS, CONJ))
    bpsk_ordinary = cycle_profile(compute_scd(_modulated("BPSK"), FS, NONCONJ))
    qpsk_ordinary = cycle_profile(compute_scd(_modulated("QPSK"), FS, NONCONJ))

    assert bpsk_conjugate[0] > 2 * qpsk_conjugate[0], (
        f"conjugate alpha=0 must separate them: BPSK {bpsk_conjugate[0]:.3f} vs "
        f"QPSK {qpsk_conjugate[0]:.3f}")
    assert max(bpsk_ordinary[1:]) > 0.8 and max(qpsk_ordinary[1:]) > 0.8, (
        "both carry a symbol-rate line, so the ordinary surface is not a discriminator")


def test_a_complex_tone_is_improper_at_twice_its_frequency() -> None:
    """exp(j*2*pi*f0*t) squared is a tone at 2*f0, so the conjugate cycle sits there.

    The sharpest positive control available: it predicts not merely that a feature
    exists but exactly where, for a signal with no symbol clock at all.
    """
    f0 = 11_000.0
    n = np.arange(16_384)
    tone = np.exp(2j * np.pi * f0 * n / FS).astype(np.complex64)

    result = compute_scd(tone, FS, SCDConfig(alpha_max_hz=40_000.0, conjugate=True))

    features = dominant_cycle_frequencies(result, count=3)
    assert features, "a complex tone is improper and must show a conjugate feature"
    assert any(abs(f.alpha_hz - 2 * f0) <= max(3 * result.alpha_resolution_hz, 500.0)
               for f in features), (
        f"expected a conjugate feature at 2*f0 = {2 * f0:,.0f} Hz, got "
        f"{[round(f.alpha_hz) for f in features]}")


# --- negative and inverted controls ------------------------------------------------------


def test_proper_complex_noise_shows_no_conjugate_feature() -> None:
    for seed in (1, 2, 3):
        result = compute_scd(_noise(seed=seed), FS, CONJ)
        assert dominant_cycle_frequencies(result) == [], f"seed {seed}"


def test_a_real_valued_signal_is_correctly_reported_as_improper() -> None:
    """Where "no features" would be the WRONG answer: real noise has no symbol clock
    but is maximally improper."""
    rng = np.random.default_rng(9)
    real_noise = rng.normal(size=16_384).astype(np.complex64)

    result = compute_scd(real_noise, FS, CONJ)

    assert cycle_profile(result)[0] > 0.9
    assert dominant_cycle_frequencies(result)


def test_a_constant_signal_is_improper_in_conjugate_mode() -> None:
    """DC is real-valued, so unlike the ordinary surface this one correctly fires."""
    result = compute_scd(np.ones(16_384, dtype=np.complex64), FS, CONJ)

    assert cycle_profile(result)[0] > 0.9
    assert dominant_cycle_frequencies(result) != []


# --- numerical behaviour ------------------------------------------------------------------


def test_conjugate_coherence_is_bounded_and_finite() -> None:
    for kind in ("BPSK", "QPSK"):
        result = compute_scd(_modulated(kind), FS, CONJ)
        assert np.all(np.isfinite(result.magnitude))
        assert np.all(result.magnitude >= 0.0)
        assert np.all(result.magnitude <= 1.0 + 1e-9)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_conjugate_coherence_is_scale_invariant(scale: float) -> None:
    reference = compute_scd(_modulated("BPSK"), FS, CONJ)

    scaled = compute_scd((_modulated("BPSK") * scale).astype(np.complex64), FS, CONJ)

    assert np.allclose(scaled.magnitude, reference.magnitude, atol=1e-5)


def test_the_conjugate_power_floor_still_applies() -> None:
    """A band with no power must not saturate the conjugate ratio either."""
    n = np.arange(16_384)
    tone = np.exp(2j * np.pi * 3_000 * n / FS).astype(np.complex64)

    result = compute_scd(tone, FS, SCDConfig(fft_size=64, max_frames=256,
                                             alpha_max_hz=40_000.0, conjugate=True))

    assert np.any(result.magnitude == 0.0), (
        "bins below the power floor must be masked, not left as a leakage ratio")


def test_the_axes_are_physical_in_conjugate_mode() -> None:
    result = compute_scd(_modulated("BPSK"), FS, CONJ)

    assert result.frequencies_hz[0] == pytest.approx(-FS / 2)
    assert result.frequency_resolution_hz == pytest.approx(FS / result.config.fft_size)
    assert result.alphas_hz[0] == 0.0
    assert result.alphas_hz[-1] <= CONJ.alpha_max_hz + 1e-9
    assert FS / result.hop >= 2 * result.alphas_hz[-1] - 1e-6, "no cyclic aliasing"


def test_degenerate_input_is_refused_in_conjugate_mode_too() -> None:
    for signal, rate in ((np.empty(0, dtype=np.complex64), FS),
                         (np.zeros(16_384, dtype=np.complex64), FS),
                         (_modulated("BPSK"), None)):
        result = compute_scd(signal, rate, CONJ)
        assert not result.ok
        assert result.warnings
        assert result.conjugate is True, "a refusal must still say which mode it was"


def test_the_conjugate_cost_estimate_reports_one_transform_per_alpha() -> None:
    """Both factors are read off a single shifted transform."""
    ordinary = estimate_cost(16_384, FS, NONCONJ)
    conjugate = estimate_cost(16_384, FS, CONJ)

    assert ordinary["transforms_per_alpha"] == 2
    assert conjugate["transforms_per_alpha"] == 1
    assert conjugate["approx_flops"] < ordinary["approx_flops"]
    assert conjugate["frames"] == ordinary["frames"], "same framing, different product"
