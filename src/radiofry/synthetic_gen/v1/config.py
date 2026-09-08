"""Configuration contracts for Synthetic Dataset V1.

V1 is deliberately impairment-free: it exists to measure the existing RadioFry
pipeline against known ground truth, so every impairment knob is pinned off and
recorded as such rather than being left implicit.
"""

from dataclasses import dataclass

# Impairments intentionally disabled in V1; recorded verbatim in ground truth so
# that a V2 capture is distinguishable from a V1 capture by metadata alone.
V1_IMPAIRMENTS: dict[str, object] = {
    "carrier_frequency_offset_hz": 0.0,
    "timing_offset_samples": 0.0,
    "phase_offset_rad": 0.0,
    "fading": "disabled",
    "multipath": "disabled",
    "interference": "disabled",
    "fec": "none",
    "interleaving": "none",
}


@dataclass(frozen=True)
class ModulationSpec:
    """Static description of a supported V1 modulation."""

    name: str
    family: str
    order: int
    bits_per_symbol: int
    radiofry_label: str
    default_pulse_shape: str = "rect"


MODULATIONS: dict[str, ModulationSpec] = {
    "BPSK": ModulationSpec("BPSK", "psk", 2, 1, "BPSK"),
    "QPSK": ModulationSpec("QPSK", "psk", 4, 2, "QPSK"),
    "8PSK": ModulationSpec("8PSK", "psk", 8, 3, "8PSK"),
    "BFSK": ModulationSpec("BFSK", "fsk", 2, 1, "CPFSK"),
    "16QAM": ModulationSpec("16QAM", "qam", 16, 4, "QAM16"),
    "64QAM": ModulationSpec("64QAM", "qam", 64, 6, "QAM64"),
    "PAM4": ModulationSpec("PAM4", "pam", 4, 2, "PAM4"),
    # GFSK is CPFSK with a Gaussian frequency pulse, so it shares the fsk family and
    # differs only in its default pulse shape.
    "GFSK": ModulationSpec("GFSK", "fsk", 2, 1, "GFSK", default_pulse_shape="gaussian"),
}

SUPPORTED_PULSE_SHAPES = ("rect", "gaussian")
# Bandwidth-time product of the Gaussian frequency pulse. 0.3 is the common GFSK value.
DEFAULT_GAUSSIAN_BT = 0.3

DEFAULT_SNR_SWEEP_DB: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0)

# FSK modulation index h = 2 * deviation / symbol_rate.
# h = 0.5 (deviation Rs/4) is the standard CPFSK/MSK index and matches the training
# distribution of the shipped checkpoint. h = 1.0 (deviation Rs/2) is retained as an
# explicitly labelled hard case: BANK.md Entry 011 measured that it is classified as
# 8PSK at 0.96 confidence and that its +/-pi per-symbol phase advance is ambiguous once
# the dispatcher decimates to one sample per symbol.
DEFAULT_FSK_MODULATION_INDEX = 0.5
KNOWN_HARD_FSK_MODULATION_INDICES: tuple[float, ...] = (1.0,)
KNOWN_HARD_FSK_REASON = (
    "modulation index h=1.0 advances the phase by +/-pi per symbol, which is ambiguous "
    "after symbol-rate decimation, and is classified as 8PSK by the shipped checkpoint"
)
DEFAULT_FSK_MODULATION_INDICES: tuple[float, ...] = (
    DEFAULT_FSK_MODULATION_INDEX,
) + KNOWN_HARD_FSK_MODULATION_INDICES


@dataclass(frozen=True)
class SampleSpec:
    """Fully resolved description of one synthetic capture.

    Exactly two of ``sample_rate_hz``, ``symbol_rate_hz`` and
    ``samples_per_symbol`` need to be supplied; the third is derived. Supplying
    all three is allowed but must be self-consistent.
    """

    modulation: str
    num_symbols: int = 4096
    samples_per_symbol: int | None = 8
    sample_rate_hz: float | None = None
    symbol_rate_hz: float | None = None
    snr_db: float | None = None
    seed: int = 0
    bits_seed: int | None = None
    fsk_deviation_hz: float | None = None
    fsk_modulation_index: float | None = None
    pulse_shape: str | None = None
    gaussian_bt: float | None = None
    bit_mapping: str = "natural"

    def __post_init__(self) -> None:
        if self.modulation not in MODULATIONS:
            raise ValueError(
                f"unsupported modulation {self.modulation!r}; V1 supports {sorted(MODULATIONS)}"
            )
        self._resolve_pulse_shape()
        if self.bit_mapping != "natural":
            raise ValueError("V1 only generates natural-binary ('natural') symbol mapping")
        if self.num_symbols < 1:
            raise ValueError("num_symbols must be positive")
        self._resolve_rates()
        self._resolve_fsk_deviation()

    def _resolve_pulse_shape(self) -> None:
        spec = MODULATIONS[self.modulation]
        shape = spec.default_pulse_shape if self.pulse_shape is None else self.pulse_shape
        if shape not in SUPPORTED_PULSE_SHAPES:
            raise ValueError(f"pulse_shape must be one of {SUPPORTED_PULSE_SHAPES}")
        if shape == "gaussian" and spec.family != "fsk":
            raise ValueError("gaussian pulse shaping is only defined for frequency modulations")
        object.__setattr__(self, "pulse_shape", shape)
        if shape == "gaussian":
            bt = DEFAULT_GAUSSIAN_BT if self.gaussian_bt is None else float(self.gaussian_bt)
            if bt <= 0:
                raise ValueError("gaussian_bt must be positive")
            object.__setattr__(self, "gaussian_bt", bt)
        elif self.gaussian_bt is not None:
            raise ValueError("gaussian_bt is only meaningful with gaussian pulse shaping")

    def _resolve_fsk_deviation(self) -> None:
        index = self.fsk_modulation_index
        deviation = self.fsk_deviation_hz
        if index is not None and index <= 0:
            raise ValueError("fsk_modulation_index must be positive")
        if deviation is not None and index is not None:
            if abs(deviation - index * self.symbol_rate_hz / 2.0) > 1e-6:
                raise ValueError(
                    "inconsistent FSK settings: fsk_deviation_hz must equal "
                    "fsk_modulation_index * symbol_rate_hz / 2"
                )
        elif deviation is None:
            resolved = DEFAULT_FSK_MODULATION_INDEX if index is None else index
            deviation = resolved * self.symbol_rate_hz / 2.0
        if deviation <= 0:
            raise ValueError("fsk_deviation_hz must be positive")
        object.__setattr__(self, "fsk_deviation_hz", float(deviation))
        # Only frequency modulations carry a meaningful index.
        resolved_index = 2.0 * deviation / self.symbol_rate_hz
        object.__setattr__(
            self,
            "fsk_modulation_index",
            resolved_index if MODULATIONS[self.modulation].family == "fsk" else None,
        )

    def _resolve_rates(self) -> None:
        sample_rate = self.sample_rate_hz
        symbol_rate = self.symbol_rate_hz
        sps = self.samples_per_symbol
        supplied = sum(value is not None for value in (sample_rate, symbol_rate, sps))
        if supplied < 2:
            raise ValueError(
                "at least two of sample_rate_hz, symbol_rate_hz, samples_per_symbol must be supplied"
            )
        for name, value in (("sample_rate_hz", sample_rate), ("symbol_rate_hz", symbol_rate), ("samples_per_symbol", sps)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if sps is None:
            ratio = sample_rate / symbol_rate
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    "sample_rate_hz / symbol_rate_hz must be a whole integer number of samples per symbol"
                )
            sps = int(round(ratio))
        elif sample_rate is None:
            sample_rate = symbol_rate * sps
        elif symbol_rate is None:
            symbol_rate = sample_rate / sps
        elif abs(sample_rate - symbol_rate * sps) > 1e-6:
            raise ValueError(
                "inconsistent rates: sample_rate_hz must equal symbol_rate_hz * samples_per_symbol"
            )
        if int(sps) != sps:
            raise ValueError("samples_per_symbol must be an integer in V1")
        object.__setattr__(self, "sample_rate_hz", float(sample_rate))
        object.__setattr__(self, "symbol_rate_hz", float(symbol_rate))
        object.__setattr__(self, "samples_per_symbol", int(sps))

    @property
    def modulation_spec(self) -> ModulationSpec:
        return MODULATIONS[self.modulation]

    @property
    def bits_per_symbol(self) -> int:
        return self.modulation_spec.bits_per_symbol

    @property
    def num_bits(self) -> int:
        return self.num_symbols * self.bits_per_symbol

    @property
    def num_samples(self) -> int:
        return self.num_symbols * self.samples_per_symbol

    @property
    def duration_sec(self) -> float:
        return self.num_samples / self.sample_rate_hz

    @property
    def effective_bits_seed(self) -> int:
        return self.seed if self.bits_seed is None else self.bits_seed
