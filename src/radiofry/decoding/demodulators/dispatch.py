"""Runtime dispatch from a modulation label to a baseline demodulator."""

from dataclasses import dataclass

import numpy as np

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.parameter_estimation import ParameterEstimate

from .analog_demod import demodulate_am, demodulate_fm
from .common import DemodulationResult
from .fsk_demod import demodulate_fsk
from .psk_demod import demodulate_psk
from .qam_demod import demodulate_qam


@dataclass(frozen=True)
class DispatchResult:
    result: DemodulationResult | None
    available: bool
    message: str = ""


def demodulate_capture(
    signal: UnifiedSignalContainer,
    modulation: str,
    parameters: ParameterEstimate,
) -> DispatchResult:
    """Demodulate a capture using the estimated symbol rate when possible."""

    if modulation in {"Unclassified", "unknown", ""}:
        return DispatchResult(None, False, "Demodulation skipped because modulation is unclassified.")
    if parameters.symbol_rate_hz is None or signal.sample_rate is None:
        return DispatchResult(None, False, "Demodulation requires both sample rate and symbol-rate estimates.")
    samples_per_symbol = max(1, round(signal.sample_rate / parameters.symbol_rate_hz))
    symbol_samples = signal.iq[::samples_per_symbol]
    try:
        if modulation in {"BPSK", "QPSK", "8PSK"}:
            result = demodulate_psk(symbol_samples, {"BPSK": 2, "QPSK": 4, "8PSK": 8}[modulation])
        elif modulation in {"CPFSK", "GFSK"}:
            result = demodulate_fsk(symbol_samples)
        elif modulation in {"QAM16", "QAM64"}:
            result = demodulate_qam(symbol_samples, int(modulation[3:]))
        elif modulation in {"AM-DSB", "AM-SSB", "WBFM"}:
            analog = demodulate_fm(symbol_samples) if modulation == "WBFM" else demodulate_am(symbol_samples)
            result = DemodulationResult(analog, np.asarray(analog > np.median(analog), dtype=np.uint8), modulation)
        else:
            return DispatchResult(None, False, f"No demodulator is registered for {modulation}.")
        return DispatchResult(result, True, f"Used approximately {samples_per_symbol} samples per symbol.")
    except ValueError as error:
        return DispatchResult(None, False, f"Demodulation failed: {error}")
