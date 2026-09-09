"""Analog synthetic capture generation: AM-DSB, AM-SSB, WBFM (BANK.md Entries 018, 020, 022, 024).

Analog captures carry no bits, so they cannot use `SampleSpec`, whose whole contract is
`num_symbols * bits_per_symbol`. They get their own spec and their own generation entry
point, while reusing the existing AWGN channel, the existing IQ/WAV writers and the same
ground-truth record shape, with every bit-derived field explicitly null.

`ANALOG_MODULATIONS` is deliberately a **separate** registry from `config.MODULATIONS`.
Three call sites default to `tuple(MODULATIONS)` (the V1 `DatasetSpec`, the V2 training
capture builder, and the symbol-rate experiment); keeping analog out of that registry
means none of them silently widens.

Generation only. None of these schemes is wired into production classification or
routing; see BANK.md Entry 024 for the measured pipeline behaviour.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radiofry.ingestion.iq_parser import IQFormat

from .channel import add_awgn
from .config import V1_IMPAIRMENTS, ModulationSpec
from .generator import (
    CAPTURE_SCHEMA,
    GENERATOR_NAME,
    GENERATOR_VERSION,
    SUPPORTED_FORMATS,
    _file_entry,
    _sha256,
)
from .writers import DEFAULT_FULL_SCALE_FRACTION, write_iq_file, write_wav_file

ANALOG_MODULATIONS: dict[str, ModulationSpec] = {
    # order and bits_per_symbol are 0: an analog capture has neither.
    "AM-DSB": ModulationSpec("AM-DSB", "analog", 0, 0, "AM-DSB"),
    "AM-SSB": ModulationSpec("AM-SSB", "analog", 0, 0, "AM-SSB"),
    "WBFM": ModulationSpec("WBFM", "analog", 0, 0, "WBFM"),
}

SUPPORTED_ANALOG_SCHEMES = ("am_dsb", "am_ssb", "wbfm")
SCHEME_TO_MODULATION = {"am_dsb": "AM-DSB", "am_ssb": "AM-SSB", "wbfm": "WBFM"}
SUPPORTED_SIDEBANDS = ("upper", "lower")
DEFAULT_SIDEBAND = "upper"
DEFAULT_FREQUENCY_DEVIATION_HZ = 15_000.0
DEFAULT_MODULATION_DEPTH = 0.5
DEFAULT_TONE_BAND_HZ = (300.0, 3_000.0)
DEFAULT_NUM_TONES = 3


@dataclass(frozen=True)
class AnalogSampleSpec:
    """One analog capture. Deliberately has no symbol rate, sps or bit fields."""

    scheme: str = "am_dsb"
    sample_rate_hz: float = 200_000.0
    num_samples: int = 32_768
    carrier_offset_hz: float = 0.0
    modulation_depth: float = DEFAULT_MODULATION_DEPTH
    sideband: str | None = None
    frequency_deviation_hz: float | None = None
    snr_db: float | None = None
    seed: int = 0
    message_seed: int | None = None
    num_tones: int = DEFAULT_NUM_TONES
    tone_band_hz: tuple[float, float] = DEFAULT_TONE_BAND_HZ

    def __post_init__(self) -> None:
        if self.scheme not in SUPPORTED_ANALOG_SCHEMES:
            raise ValueError(
                f"unsupported analog scheme {self.scheme!r}; supported: {SUPPORTED_ANALOG_SCHEMES}"
            )
        if self.sample_rate_hz <= 0 or self.num_samples < 1:
            raise ValueError("sample_rate_hz and num_samples must be positive")
        if not 0.0 < self.modulation_depth <= 1.0:
            raise ValueError("modulation_depth must be in (0, 1]")
        self._resolve_sideband()
        self._resolve_frequency_deviation()
        if self.num_tones < 1:
            raise ValueError("num_tones must be positive")
        low, high = self.tone_band_hz
        if not 0 < low < high < self.sample_rate_hz / 2:
            raise ValueError("tone_band_hz must be inside (0, sample_rate/2)")

    def _resolve_sideband(self) -> None:
        """Sideband is meaningful for SSB only; it defaults to the documented USB."""

        if self.scheme != "am_ssb":
            if self.sideband is not None:
                raise ValueError(f"sideband is only meaningful for am_ssb, not {self.scheme}")
            return
        sideband = DEFAULT_SIDEBAND if self.sideband is None else self.sideband
        if sideband not in SUPPORTED_SIDEBANDS:
            raise ValueError(f"sideband must be one of {SUPPORTED_SIDEBANDS}")
        object.__setattr__(self, "sideband", sideband)

    def _resolve_frequency_deviation(self) -> None:
        """Peak deviation is meaningful for WBFM only, and must stay inside the band."""

        if self.scheme != "wbfm":
            if self.frequency_deviation_hz is not None:
                raise ValueError(
                    f"frequency_deviation_hz is only meaningful for wbfm, not {self.scheme}"
                )
            return
        deviation = (
            DEFAULT_FREQUENCY_DEVIATION_HZ
            if self.frequency_deviation_hz is None
            else float(self.frequency_deviation_hz)
        )
        if deviation <= 0:
            raise ValueError("frequency_deviation_hz must be positive")
        # The message is peak-normalised to 1, so the instantaneous frequency spans
        # f_c +/- deviation. Both edges must stay below Nyquist or the capture aliases.
        if abs(self.carrier_offset_hz) + deviation >= self.sample_rate_hz / 2:
            raise ValueError(
                "carrier_offset_hz +/- frequency_deviation_hz must stay inside the "
                f"Nyquist band (+/-{self.sample_rate_hz / 2} Hz)"
            )
        object.__setattr__(self, "frequency_deviation_hz", deviation)

    @property
    def modulation_name(self) -> str:
        return SCHEME_TO_MODULATION[self.scheme]

    @property
    def modulation_spec(self) -> ModulationSpec:
        return ANALOG_MODULATIONS[self.modulation_name]

    @property
    def duration_sec(self) -> float:
        return self.num_samples / self.sample_rate_hz

    @property
    def effective_message_seed(self) -> int:
        return self.seed if self.message_seed is None else self.message_seed


def generate_message(spec: AnalogSampleSpec) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Deterministic multi-tone message, peak-normalised to 1.0.

    A tone sum rather than recorded audio: reproducible from a seed, needs no external
    asset, and puts known lines in the spectrum so the oracle can be analytic.
    """

    rng = np.random.default_rng([spec.effective_message_seed, 3])
    low, high = spec.tone_band_hz
    frequencies = np.sort(rng.uniform(low, high, size=spec.num_tones))
    amplitudes = rng.uniform(0.5, 1.0, size=spec.num_tones)
    phases = rng.uniform(0.0, 2 * np.pi, size=spec.num_tones)

    time = np.arange(spec.num_samples, dtype=np.float64) / spec.sample_rate_hz
    message = np.zeros(spec.num_samples, dtype=np.float64)
    for frequency, amplitude, phase in zip(frequencies, amplitudes, phases):
        message += amplitude * np.sin(2 * np.pi * frequency * time + phase)
    message -= message.mean()
    peak = float(np.max(np.abs(message)))
    if peak > 0:
        message /= peak

    tones = [
        {"frequency_hz": float(f), "amplitude": float(a), "phase_rad": float(p)}
        for f, a, p in zip(frequencies, amplitudes, phases)
    ]
    return message, tones


def modulate_am_dsb(message: np.ndarray, spec: AnalogSampleSpec) -> np.ndarray:
    """x(t) = (1 + m*s(t)) * exp(j*2*pi*fc*t), complex baseband.

    The carrier term is retained so that envelope detection is valid; with the default
    zero carrier offset the waveform is real-valued, which is the baseband DSB form.
    """

    values = np.asarray(message, dtype=np.float64)
    envelope = 1.0 + spec.modulation_depth * values
    if spec.carrier_offset_hz == 0.0:
        return envelope.astype(np.complex64)
    time = np.arange(values.size, dtype=np.float64) / spec.sample_rate_hz
    return (envelope * np.exp(2j * np.pi * spec.carrier_offset_hz * time)).astype(np.complex64)


def modulate_am_ssb(message: np.ndarray, spec: AnalogSampleSpec) -> np.ndarray:
    """Single-sideband, suppressed carrier, via the analytic signal.

    `hilbert(s) = s + j*H{s}` has energy only at positive frequencies, so shifting it up
    by the carrier places the whole message above it (**upper** sideband, the documented
    default). Conjugating first mirrors the spectrum and yields the lower sideband.
    """

    from scipy.signal import hilbert

    values = np.asarray(message, dtype=np.float64)
    analytic = hilbert(values)
    if spec.sideband == "lower":
        analytic = np.conj(analytic)
    time = np.arange(values.size, dtype=np.float64) / spec.sample_rate_hz
    return (analytic * np.exp(2j * np.pi * spec.carrier_offset_hz * time)).astype(np.complex64)


def modulate_wbfm(message: np.ndarray, spec: AnalogSampleSpec) -> np.ndarray:
    """Wideband FM by explicit phase integration.

    The message `s(t)` is peak-normalised to 1, so with peak deviation `df` the
    instantaneous frequency is

        f_i(t) = f_c + df * s(t),                       |f_i - f_c| <= df

    and the transmitted phase is its integral,

        phi(t) = 2*pi * integral_0^t f_i(u) du.

    Sampled at `Fs`, that integral becomes a cumulative sum with step `1/Fs`:

        phi[n] = (2*pi / Fs) * sum_{k<=n} (f_c + df * s[k])

    so `angle(x[n+1] * conj(x[n])) * Fs / (2*pi)` returns `f_i[n+1]` exactly - which is
    what the test oracle measures. The emitted waveform is

        x[n] = exp(j * phi[n])

    i.e. constant unit envelope: FM carries information in phase alone, so amplitude is
    deliberately not modulated.
    """

    values = np.asarray(message, dtype=np.float64)
    deviation = float(spec.frequency_deviation_hz or 0.0)
    instantaneous = spec.carrier_offset_hz + deviation * values
    phase = 2.0 * np.pi * np.cumsum(instantaneous) / spec.sample_rate_hz
    return np.exp(1j * phase).astype(np.complex64)


def modulate_analog(message: np.ndarray, spec: AnalogSampleSpec) -> np.ndarray:
    """Route a message to the modulator for the configured scheme."""

    if spec.scheme == "am_dsb":
        return modulate_am_dsb(message, spec)
    if spec.scheme == "am_ssb":
        return modulate_am_ssb(message, spec)
    if spec.scheme == "wbfm":
        return modulate_wbfm(message, spec)
    raise ValueError(f"unsupported analog scheme {spec.scheme!r}")


def generate_analog_sample(
    spec: AnalogSampleSpec,
    output_dir: str | Path,
    capture_id: str,
    *,
    formats: tuple[str, ...] = SUPPORTED_FORMATS,
    iq_format: IQFormat | None = None,
    wav_dtype: str = "int16",
    full_scale_fraction: float = DEFAULT_FULL_SCALE_FRACTION,
) -> dict[str, Any]:
    """Generate one analog capture plus its ground-truth record; returns the record."""

    unsupported = set(formats) - set(SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(f"unsupported output formats: {sorted(unsupported)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    message, tones = generate_message(spec)
    clean = modulate_analog(message, spec)
    deviation = spec.frequency_deviation_hz if spec.scheme == "wbfm" else None
    highest_tone_hz = max(tone["frequency_hz"] for tone in tones)
    noisy, noise = add_awgn(clean, spec.snr_db, np.random.default_rng([spec.seed, 2]))

    message_name = f"{capture_id}.message.npy"
    np.save(output / message_name, message)

    files: list[dict[str, Any]] = []
    if "iq" in formats:
        files.append(_file_entry(write_iq_file(
            output / f"{capture_id}.iq", noisy, fmt=iq_format or IQFormat(),
            full_scale_fraction=full_scale_fraction)))
    if "wav" in formats:
        files.append(_file_entry(write_wav_file(
            output / f"{capture_id}.wav", noisy, spec.sample_rate_hz,
            dtype=wav_dtype, full_scale_fraction=full_scale_fraction)))

    truth = {
        "schema": CAPTURE_SCHEMA,
        "schema_version": 1,
        "capture_kind": "analog",
        "capture_id": capture_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "modulation": {
            "name": spec.modulation_name,
            "family": "analog",
            "radiofry_label": spec.modulation_spec.radiofry_label,
            # Bit-derived descriptors do not exist for an analog capture.
            "order": None,
            "bits_per_symbol": None,
            "bit_mapping": None,
            "constellation_normalization": None,
            "pulse_shape": None,
        },
        "signal": {
            "sample_rate_hz": spec.sample_rate_hz,
            "num_samples": spec.num_samples,
            "duration_sec": spec.duration_sec,
            "center_frequency_hz": spec.carrier_offset_hz,
            "symbol_rate_hz": None,
            "samples_per_symbol": None,
            "fsk_deviation_hz": None,
            "fsk_modulation_index": None,
            "gaussian_bt": None,
        },
        "analog": {
            "scheme": spec.scheme,
            "carrier_offset_hz": spec.carrier_offset_hz,
            # Each parameter is null for the scheme it does not apply to.
            "modulation_depth": spec.modulation_depth if spec.scheme == "am_dsb" else None,
            "sideband": spec.sideband if spec.scheme == "am_ssb" else None,
            "frequency_deviation_hz": deviation,
            # beta = peak deviation / highest message tone, the single-tone definition
            # applied to the widest component of this multi-tone message.
            "modulation_index": None if deviation is None else deviation / highest_tone_hz,
            # Carson's rule is an approximation, recorded as such - the measured
            # occupied bandwidth is not required to equal it.
            "carson_bandwidth_hz": (
                None if deviation is None else 2.0 * (deviation + highest_tone_hz)
            ),
            "carrier": "suppressed" if spec.scheme == "am_ssb" else "present",
        },
        "message": {
            "type": "multitone",
            "num_tones": spec.num_tones,
            "tone_band_hz": list(spec.tone_band_hz),
            "tones": tones,
            "peak_normalised": True,
            "message_file": message_name,
            "message_sha256": _sha256(message),
        },
        "noise": {
            "noise_type": noise.noise_type,
            "target_snr_db": noise.target_snr_db,
            "realized_snr_db": noise.realized_snr_db,
            "snr_definition": noise.snr_definition,
            "signal_power": noise.signal_power,
            "target_noise_power": noise.target_noise_power,
            "realized_noise_power": noise.realized_noise_power,
            # Both are per-symbol / per-bit energy ratios and have no analog meaning.
            "es_n0_db": None,
            "eb_n0_db": None,
        },
        "known_hard": False,
        "known_hard_reason": "",
        "impairments": dict(V1_IMPAIRMENTS),
        # Analog carries no transmitted bits; null rather than an empty bit block.
        "bits": None,
        "seeds": {
            "seed": spec.seed,
            "message_seed": spec.effective_message_seed,
            "message_stream": "numpy.random.default_rng([message_seed, 3])",
            "noise_stream": "numpy.random.default_rng([seed, 2])",
        },
        "files": files,
    }
    (output / f"{capture_id}.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth
