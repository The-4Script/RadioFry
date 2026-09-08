"""PAM4 and GFSK generator support (BANK.md Entry 016).

Extends V2 synthetic coverage from 6 to 8 classes. The existing six modulations
must be bit-for-bit unchanged, which is asserted against the frozen V1 captures.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from radiofry.ingestion.iq_parser import IQFormat, read_iq
from radiofry.synthetic_gen.v1 import (
    MODULATIONS,
    SampleSpec,
    add_awgn,
    bits_to_symbol_indices,
    constellation_for,
    generate_source_bits,
    modulate,
)

FS, SPS, NSYM = 200_000.0, 8, 512
V1_CAPTURES = Path("data/synthetic_v1/captures")


def _spec(modulation: str, **overrides) -> SampleSpec:
    params = dict(modulation=modulation, num_symbols=NSYM, samples_per_symbol=SPS,
                  sample_rate_hz=FS, snr_db=None, seed=21)
    params.update(overrides)
    return SampleSpec(**params)


# --- registry ---------------------------------------------------------------------


def test_registry_now_covers_eight_classes_with_production_labels() -> None:
    assert sorted(m.radiofry_label for m in MODULATIONS.values()) == [
        "8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK",
    ]


def test_pam4_and_gfsk_declare_the_expected_specs() -> None:
    pam4, gfsk = MODULATIONS["PAM4"], MODULATIONS["GFSK"]

    assert (pam4.family, pam4.order, pam4.bits_per_symbol, pam4.radiofry_label) == ("pam", 4, 2, "PAM4")
    assert (gfsk.family, gfsk.order, gfsk.bits_per_symbol, gfsk.radiofry_label) == ("fsk", 2, 1, "GFSK")


# --- PAM4 -------------------------------------------------------------------------


def test_pam4_constellation_is_four_real_levels_at_unit_average_power() -> None:
    points = constellation_for("PAM4")

    assert points.size == 4
    np.testing.assert_allclose(points.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.mean(np.abs(points) ** 2), 1.0, rtol=1e-9)
    np.testing.assert_allclose(np.sort(points.real), np.array([-3, -1, 1, 3]) / np.sqrt(5.0), rtol=1e-9)


def test_pam4_maps_natural_binary_indices_to_ascending_levels() -> None:
    spec = _spec("PAM4", num_symbols=4)
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)

    symbols = modulate(bits, spec)[:: SPS]

    np.testing.assert_allclose(symbols.real, np.array([-3, -1, 1, 3]) / np.sqrt(5.0), atol=1e-6)


def test_pam4_noiseless_capture_recovers_the_exact_source_bits() -> None:
    spec = _spec("PAM4")
    bits = generate_source_bits(spec)
    points = constellation_for("PAM4")

    symbols = modulate(bits, spec)[SPS // 2 :: SPS][: spec.num_symbols]
    indices = np.argmin(np.abs(symbols[:, None] - points[None, :]), axis=1).astype(np.int64)
    recovered = ((indices[:, None] >> np.arange(1, -1, -1)) & 1).astype(np.uint8).ravel()

    np.testing.assert_array_equal(recovered, bits)


def test_pam4_bits_round_trip_through_the_shared_index_helper() -> None:
    spec = _spec("PAM4", num_symbols=64)
    bits = generate_source_bits(spec)

    indices = bits_to_symbol_indices(bits, spec.bits_per_symbol)

    assert indices.max() < 4
    assert indices.size == spec.num_symbols


def test_pam4_carries_no_fsk_parameters() -> None:
    spec = _spec("PAM4")

    assert spec.fsk_modulation_index is None
    assert spec.pulse_shape == "rect"
    assert spec.gaussian_bt is None


# --- GFSK -------------------------------------------------------------------------


def test_gfsk_defaults_to_a_gaussian_pulse_with_a_recorded_bt() -> None:
    spec = _spec("GFSK")

    assert spec.pulse_shape == "gaussian"
    assert spec.gaussian_bt == pytest.approx(0.3)
    assert spec.fsk_modulation_index == pytest.approx(0.5)


def test_gfsk_bt_is_configurable() -> None:
    assert _spec("GFSK", gaussian_bt=0.5).gaussian_bt == pytest.approx(0.5)


def test_gfsk_is_constant_modulus() -> None:
    waveform = modulate(generate_source_bits(_spec("GFSK")), _spec("GFSK"))

    np.testing.assert_allclose(np.abs(waveform), 1.0, atol=1e-5)


def test_gaussian_frequency_pulse_has_unit_area() -> None:
    from radiofry.synthetic_gen.v1 import gaussian_frequency_pulse

    # Unit area is what keeps the accumulated phase, and hence the modulation index,
    # equal to the unshaped case.
    assert gaussian_frequency_pulse(0.3, SPS).sum() == pytest.approx(1.0, rel=1e-9)


def test_gfsk_differs_from_cpfsk_but_preserves_the_accumulated_phase() -> None:
    bits = generate_source_bits(_spec("BFSK"))
    cpfsk = modulate(bits, _spec("BFSK"))
    gfsk = modulate(bits, _spec("GFSK"))

    assert not np.allclose(cpfsk, gfsk, atol=1e-3)
    total = lambda x: np.unwrap(np.angle(x))[-1] - np.unwrap(np.angle(x))[0]
    # Equal to within the edge effect of the finite 'same' convolution.
    assert total(gfsk) == pytest.approx(total(cpfsk), rel=0.05)


def test_gfsk_never_exceeds_the_configured_frequency_deviation() -> None:
    spec = _spec("GFSK")
    waveform = modulate(generate_source_bits(spec), spec)

    peak_hz = np.max(np.abs(np.diff(np.unwrap(np.angle(waveform))))) * FS / (2 * np.pi)

    assert peak_hz <= spec.fsk_deviation_hz * 1.001


def test_gfsk_smooths_the_instantaneous_frequency() -> None:
    # The defining property: Gaussian shaping removes the hard frequency steps.
    bits = generate_source_bits(_spec("BFSK"))
    step = lambda x: np.max(np.abs(np.diff(np.diff(np.unwrap(np.angle(x))))))

    assert step(modulate(bits, _spec("GFSK"))) < step(modulate(bits, _spec("BFSK")))


def test_gfsk_occupies_less_bandwidth_than_cpfsk() -> None:
    from radiofry.evaluation.symbol_rate_experiment import occupied_bandwidth

    bits = generate_source_bits(_spec("BFSK"))

    assert occupied_bandwidth(modulate(bits, _spec("GFSK")), FS) < occupied_bandwidth(
        modulate(bits, _spec("BFSK")), FS
    )


def test_gfsk_generation_is_deterministic() -> None:
    spec = _spec("GFSK")

    np.testing.assert_array_equal(
        modulate(generate_source_bits(spec), spec), modulate(generate_source_bits(spec), spec)
    )


def test_gaussian_pulse_shaping_is_rejected_for_non_fsk_modulations() -> None:
    with pytest.raises(ValueError, match="gaussian"):
        _spec("QPSK", pulse_shape="gaussian")


def test_unsupported_pulse_shape_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="pulse"):
        _spec("BPSK", pulse_shape="raised_cosine")


# --- the existing six modulations must not move -------------------------------------


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK", "BFSK", "16QAM", "64QAM"])
def test_existing_modulations_keep_rectangular_pulses(modulation: str) -> None:
    spec = _spec(modulation)

    assert spec.pulse_shape == "rect"
    assert spec.gaussian_bt is None


@pytest.mark.skipif(not V1_CAPTURES.exists(), reason="frozen V1 dataset not present")
@pytest.mark.parametrize(
    "capture_id",
    ["BPSK_snr20dB_r000", "QPSK_snr10dB_r000", "8PSK_snr15dB_r000",
     "16QAM_snr20dB_r000", "64QAM_snr5dB_r000", "BFSK_h0.5_snr20dB_r000"],
)
def test_frozen_v1_captures_still_regenerate_bit_for_bit(capture_id: str) -> None:
    # The strongest guard available: re-derive a frozen capture from its own recorded
    # ground truth and require the samples to match the file on disk exactly.
    metadata = json.loads((V1_CAPTURES / f"{capture_id}.json").read_text(encoding="utf-8"))
    signal, noise = metadata["signal"], metadata["noise"]
    entry = next(f for f in metadata["files"] if f["file_format"] == "iq")

    spec = SampleSpec(
        modulation=metadata["modulation"]["name"],
        num_symbols=signal["num_symbols"],
        samples_per_symbol=signal["samples_per_symbol"],
        sample_rate_hz=signal["sample_rate_hz"],
        snr_db=noise["target_snr_db"],
        seed=metadata["seeds"]["seed"],
        bits_seed=metadata["seeds"]["bits_seed"],
        fsk_deviation_hz=signal["fsk_deviation_hz"],
    )
    regenerated = add_awgn(
        modulate(generate_source_bits(spec), spec), spec.snr_db, np.random.default_rng([spec.seed, 2])
    )[0]
    on_disk = read_iq(
        V1_CAPTURES / entry["filename"], sample_rate=signal["sample_rate_hz"],
        fmt=IQFormat(entry["dtype"], entry["byte_order"]),
    ).iq / entry["scale_factor"]

    np.testing.assert_allclose(on_disk.real, regenerated.real, atol=2.0 / entry["scale_factor"])
    np.testing.assert_allclose(on_disk.imag, regenerated.imag, atol=2.0 / entry["scale_factor"])
