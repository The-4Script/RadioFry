"""Optional GNU Radio waveform-generation adapter."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GnuradioStatus:
    available: bool
    message: str


def check_gnuradio() -> GnuradioStatus:
    """Report whether the system GNU Radio Python bindings are importable."""

    try:
        import gnuradio  # type: ignore
    except ImportError:
        return GnuradioStatus(False, "GNU Radio bindings unavailable; use the NumPy synthetic path.")
    return GnuradioStatus(True, f"GNU Radio {getattr(gnuradio, '__version__', 'available')}")


def generate_bpsk(bits: np.ndarray, samples_per_symbol: int = 8) -> np.ndarray:
    """Generate a normalized BPSK baseband fallback without GNU Radio."""

    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be positive")
    symbols = 1.0 - 2.0 * values.astype(np.float32)
    return np.repeat(symbols, samples_per_symbol).astype(np.complex64)


def generate_bpsk_gnuradio(
    bits: np.ndarray,
    *,
    samples_per_symbol: int = 8,
    snr_db: float | None = None,
    frequency_offset: float = 0.0,
) -> np.ndarray:
    """Generate BPSK through GNU Radio's channel model when available."""

    status = check_gnuradio()
    if not status.available:
        raise RuntimeError(status.message)
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be positive")
    try:
        from gnuradio import blocks, channels, gr
    except ImportError as error:
        raise RuntimeError(f"GNU Radio Python bindings unavailable: {error}") from error

    waveform = generate_bpsk(bits, samples_per_symbol)
    signal_power = float(np.mean(np.abs(waveform) ** 2))
    noise_voltage = 0.0 if snr_db is None else float(np.sqrt(signal_power / (10 ** (snr_db / 10))))
    flowgraph = gr.top_block()
    source = blocks.vector_source_c(waveform.tolist(), False)
    channel = channels.channel_model(
        noise_voltage=noise_voltage,
        frequency_offset=frequency_offset,
        epsilon=1.0,
        taps=[1.0 + 0.0j],
        noise_seed=7,
    )
    limiter = blocks.head(gr.sizeof_gr_complex, waveform.size)
    sink = blocks.vector_sink_c()
    flowgraph.connect(source, channel, limiter, sink)
    flowgraph.run()
    output = np.asarray(sink.data(), dtype=np.complex64)
    if output.size < waveform.size:
        output = np.pad(output, (0, waveform.size - output.size), mode="edge")
    return output[: waveform.size]
