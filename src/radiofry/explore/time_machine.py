"""Signal Time Machine: window selection and view data for interactive exploration.

This module is the single source of truth for "which part of the recording am I looking
at". Every view - waveform, spectrum, spectrogram, IQ scatter - is derived from one
`WindowSelection`, so moving the timeline moves all of them together. It holds no UI
state and imports no UI library, which is what makes it testable.

Honesty rules this module follows, because the analyst has to be able to trust what is
on screen:

* Everything returned here is computed from the real samples handed in. Nothing is
  synthesised to make a plot look better.
* `window_measurements` reports only quantities that are genuinely measurable from the
  selected window alone. It does NOT recompute modulation, symbol rate, SNR or carrier
  frequency - those come from the whole-capture pipeline and are the caller's job to
  label as such.
* Downsampling for display keeps real sample values (it decimates, it does not smooth or
  interpolate), so what is plotted is a subset of the actual signal.
"""

from dataclasses import dataclass

import numpy as np

# Plot budgets. A capture can hold millions of samples; browsers cannot.
DEFAULT_DISPLAY_POINTS = 2_000
DEFAULT_SPECTRUM_POINTS = 1_024
DEFAULT_SPECTROGRAM_ROWS = 96
DEFAULT_WINDOW_SECONDS = 0.02
MIN_WINDOW_SAMPLES = 8


@dataclass(frozen=True)
class WindowSelection:
    """A half-open sample range [start_sample, start_sample + length_samples)."""

    start_sample: int
    length_samples: int
    total_samples: int
    sample_rate: float | None

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.length_samples

    @property
    def start_seconds(self) -> float | None:
        if not self.sample_rate:
            return None
        return self.start_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float | None:
        if not self.sample_rate:
            return None
        return self.length_samples / self.sample_rate

    @property
    def end_seconds(self) -> float | None:
        """Derived from `end_sample`, not from start + duration.

        Those two agree mathematically but not in binary: 6000/fs + 11000/fs lands a
        half-ULP below 17000/fs, which is enough to display a region ending at 0.085 s
        as 0.084. Reading the boundary off the sample it actually is avoids that.
        """
        if not self.sample_rate:
            return None
        return self.end_sample / self.sample_rate

    @property
    def center_seconds(self) -> float | None:
        start = self.start_seconds
        duration = self.duration_seconds
        return None if start is None or duration is None else start + duration / 2


def total_duration_seconds(total_samples: int, sample_rate: float | None) -> float | None:
    """Recording length in seconds, or None when the sample rate is unknown."""

    if not sample_rate or sample_rate <= 0 or total_samples <= 0:
        return None
    return total_samples / sample_rate


def select_window(
    total_samples: int,
    sample_rate: float | None,
    cursor_seconds: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> WindowSelection:
    """Map a timeline position to a safe sample range.

    The cursor is the START of the window, which keeps scrubbing predictable: the
    displayed region begins where the timeline marker sits. The range is always clamped
    inside the recording, so a cursor near the end yields a shorter window rather than an
    out-of-bounds slice.
    """

    total = max(0, int(total_samples))
    if total == 0:
        return WindowSelection(0, 0, 0, sample_rate)
    if not sample_rate or sample_rate <= 0:
        # Without a sample rate there is no time axis; fall back to the whole capture
        # rather than inventing one.
        return WindowSelection(0, total, total, sample_rate)

    length = int(round(max(0.0, float(window_seconds)) * sample_rate))
    length = max(MIN_WINDOW_SAMPLES, length)
    length = min(length, total)

    start = int(round(max(0.0, float(cursor_seconds)) * sample_rate))
    start = max(0, min(start, total - length))
    return WindowSelection(start, length, total, sample_rate)


def extract_window(iq: np.ndarray, selection: WindowSelection) -> np.ndarray:
    """The real samples the selection points at. Never raises on a short capture."""

    samples = np.asarray(iq)
    if samples.size == 0 or selection.length_samples <= 0:
        return samples[:0]
    start = max(0, min(selection.start_sample, samples.size))
    end = max(start, min(selection.end_sample, samples.size))
    return samples[start:end]


def downsample_for_display(
    values: np.ndarray, max_points: int = DEFAULT_DISPLAY_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate to a plottable size, returning (original_indices, values).

    Decimation, not interpolation: every returned value is a real sample. The indices let
    a caller map a plotted point back to its position in the capture.
    """

    samples = np.asarray(values)
    if samples.size == 0:
        return np.empty(0, dtype=int), samples[:0]
    budget = max(1, int(max_points))
    if samples.size <= budget:
        return np.arange(samples.size), samples
    step = int(np.ceil(samples.size / budget))
    indices = np.arange(0, samples.size, step)
    return indices, samples[indices]


def window_spectrum(
    samples: np.ndarray,
    sample_rate: float | None,
    max_points: int = DEFAULT_SPECTRUM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed magnitude spectrum of the selection, in dB.

    Frequencies are baseband offsets around 0 Hz for complex input. Returns empty arrays
    for an unusable window rather than a fabricated spectrum.
    """

    values = np.asarray(samples)
    if values.size < 4 or not sample_rate or sample_rate <= 0:
        return np.empty(0), np.empty(0)
    count = min(int(max_points) * 2, values.size)
    block = values[:count].astype(np.complex128)
    spectrum = np.fft.fftshift(np.fft.fft(block * np.hanning(block.size)))
    frequencies = np.fft.fftshift(np.fft.fftfreq(block.size, d=1.0 / sample_rate))
    magnitude = np.abs(spectrum)
    reference = magnitude.max()
    if reference <= 0:
        return frequencies, np.full(frequencies.size, -120.0)
    return frequencies, 20.0 * np.log10(np.maximum(magnitude / reference, 1e-6))


def window_spectrogram(
    samples: np.ndarray,
    sample_rate: float | None,
    rows: int = DEFAULT_SPECTROGRAM_ROWS,
    bins: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Waterfall for the selection: (times_s, frequencies_hz, power_db[row, bin]).

    A plain STFT with non-overlapping blocks - enough to show where energy sits in time
    and frequency, and cheap enough to recompute on every scrub. Returns empty arrays
    when the window is too short to form even one block.
    """

    values = np.asarray(samples)
    if values.size < 8 or not sample_rate or sample_rate <= 0:
        return np.empty(0), np.empty(0), np.empty((0, 0))
    block_size = max(8, int(bins))
    available = values.size // block_size
    if available == 0:
        block_size = max(8, values.size // 2)
        available = values.size // block_size
        if available == 0:
            return np.empty(0), np.empty(0), np.empty((0, 0))
    step = max(1, available // max(1, int(rows)))
    starts = list(range(0, available * block_size, block_size * step))[: int(rows)]

    window = np.hanning(block_size)
    power = np.empty((len(starts), block_size), dtype=np.float64)
    for row, start in enumerate(starts):
        block = values[start:start + block_size].astype(np.complex128)
        spectrum = np.fft.fftshift(np.fft.fft(block * window))
        power[row] = np.abs(spectrum) ** 2
    reference = power.max()
    power_db = 10.0 * np.log10(np.maximum(power / reference, 1e-9)) if reference > 0 \
        else np.full_like(power, -90.0)
    frequencies = np.fft.fftshift(np.fft.fftfreq(block_size, d=1.0 / sample_rate))
    times = np.asarray(starts, dtype=np.float64) / sample_rate
    return times, frequencies, power_db


def symbol_rate_decimated(
    samples: np.ndarray, sample_rate: float | None, symbol_rate_hz: float | None
) -> np.ndarray:
    """One sample per symbol period, for an IQ scatter that is closer to a constellation.

    This does NOT re-estimate symbol timing for the window - it simply decimates at the
    whole-capture symbol rate the pipeline already reported. The caller must label it
    accordingly; a correct rate with an unknown timing phase still produces a smeared
    cloud, which is honest but is not a decoded constellation.
    """

    values = np.asarray(samples)
    if values.size == 0 or not sample_rate or not symbol_rate_hz or symbol_rate_hz <= 0:
        return values[:0]
    samples_per_symbol = int(round(sample_rate / symbol_rate_hz))
    if samples_per_symbol < 1 or samples_per_symbol >= values.size:
        return values[:0]
    return values[::samples_per_symbol]


def window_measurements(
    samples: np.ndarray,
    sample_rate: float | None,
    spectrum_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Quantities genuinely measurable from this window alone.

    Deliberately excludes modulation, symbol rate, SNR, carrier frequency and occupied
    bandwidth: those are whole-capture pipeline results and are NOT recomputed per
    window. Anything a caller wants that is missing here should be shown as unavailable
    rather than filled in from the global analysis.
    """

    values = np.asarray(samples)
    if values.size == 0:
        return {}
    magnitude = np.abs(values.astype(np.complex128))
    measurements = {
        "samples": float(values.size),
        "rms_amplitude": float(np.sqrt(np.mean(magnitude**2))),
        "peak_amplitude": float(magnitude.max()),
        "mean_amplitude": float(magnitude.mean()),
    }
    mean = magnitude.mean()
    if mean > 0:
        measurements["amplitude_cv"] = float(magnitude.std() / mean)
    if sample_rate and sample_rate > 0 and values.size >= 4:
        if spectrum_data is not None and spectrum_data[0].size > 0:
            frequencies, magnitude_db = spectrum_data
        else:
            frequencies, magnitude_db = window_spectrum(values, sample_rate)
        if frequencies.size:
            measurements["peak_offset_hz"] = float(frequencies[int(np.argmax(magnitude_db))])
    return measurements


def advance_cursor(
    cursor_seconds: float,
    duration_seconds: float | None,
    step_seconds: float,
    speed: float = 1.0,
    loop: bool = True,
) -> float:
    """Move the playback cursor one tick. Playback is timeline movement, not audio."""

    if duration_seconds is None or duration_seconds <= 0:
        return 0.0
    advanced = float(cursor_seconds) + float(step_seconds) * max(0.0, float(speed))
    if advanced < duration_seconds:
        return max(0.0, advanced)
    return 0.0 if loop else float(duration_seconds)


# --------------------------------------------------------------------- region analysis
#
# Selecting a region and re-running the pipeline is a different action from moving the
# timeline: scrubbing is visualisation only, while a region analysis is the full,
# expensive `analyze_capture` call. The region reuses `WindowSelection` so there is one
# selection concept in the feature, not two.

# One CNN inference frame. Shorter than this and the classifier has nothing to look at,
# so the pipeline would return an answer built from padding rather than signal.
MIN_ANALYSIS_SAMPLES = 128


@dataclass(frozen=True)
class RegionValidation:
    """The outcome of asking for a region. `selection` is None when it is unusable."""

    selection: "WindowSelection | None"
    ok: bool
    reason: str = ""


def select_region(
    total_samples: int,
    sample_rate: float | None,
    start_seconds: float,
    end_seconds: float,
) -> RegionValidation:
    """Validate a start/end pair and turn it into a sample range.

    Refuses rather than guesses: a reversed, empty, out-of-range or too-short region
    returns `ok=False` with a reason the UI can show. Silently "fixing" a reversed
    selection would analyse a region the analyst did not choose.
    """

    total = max(0, int(total_samples))
    if total == 0:
        return RegionValidation(None, False, "The capture contains no samples.")
    if not sample_rate or sample_rate <= 0:
        return RegionValidation(
            None, False,
            "This capture has no sample rate, so a time region cannot be located. "
            "Re-analyze the file with its sample rate to enable region analysis.")

    duration = total / sample_rate
    if end_seconds <= start_seconds:
        return RegionValidation(
            None, False,
            f"The region ends at or before it starts ({start_seconds:.4f} s to "
            f"{end_seconds:.4f} s).")
    if start_seconds >= duration:
        return RegionValidation(
            None, False,
            f"The region starts past the end of the {duration:.4f} s recording.")
    if end_seconds <= 0:
        return RegionValidation(None, False, "The region ends before the recording starts.")

    start = max(0, int(round(start_seconds * sample_rate)))
    end = min(total, int(round(end_seconds * sample_rate)))
    length = end - start
    if length < MIN_ANALYSIS_SAMPLES:
        return RegionValidation(
            None, False,
            f"The region holds {length:,} samples; at least {MIN_ANALYSIS_SAMPLES} are "
            "needed for the analysis pipeline to have a full frame to work with.")
    return RegionValidation(WindowSelection(start, length, total, sample_rate), True)


def region_container(signal: object, selection: "WindowSelection") -> object:
    """Build a `UnifiedSignalContainer` holding exactly the selected samples.

    Sample rate and source format carry over unchanged because they still describe the
    slice. Metadata carries over too, plus explicit region bounds, so a reader can see
    which part of the original file this is - the `path` entry still names the real
    source, and would otherwise be quietly misleading. Nothing is invented: if the
    original had no centre frequency, the region has none either.
    """

    from radiofry.contracts import UnifiedSignalContainer

    samples = np.asarray(signal.iq)
    start = max(0, min(selection.start_sample, samples.size))
    end = max(start, min(selection.end_sample, samples.size))
    metadata = dict(getattr(signal, "metadata", {}) or {})
    metadata.update({
        "region_of_capture": True,
        "region_start_sample": int(start),
        "region_end_sample": int(end),
    })
    if selection.sample_rate:
        metadata["region_start_seconds"] = start / selection.sample_rate
        metadata["region_end_seconds"] = end / selection.sample_rate
    return UnifiedSignalContainer(
        iq=samples[start:end],
        sample_rate=signal.sample_rate,
        source_format=getattr(signal, "source_format", "unknown"),
        metadata=metadata,
    )


# Fields worth putting side by side. Each entry is (label, stage, key, formatter).
# Only fields the existing pipeline genuinely returns appear here - nothing is computed
# specially for the comparison.
_COMPARISON_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("Modulation", "fusion", "label", "text"),
    ("Fusion trust", "fusion", "trust_score", "ratio"),
    ("Classical family", "classical_modulation", "family", "text"),
    ("CNN label", "cnn_modulation", "label", "text"),
    ("CNN confidence", "cnn_modulation", "confidence", "ratio"),
    ("Carrier frequency", "parameters", "carrier_frequency_hz", "hz"),
    ("Occupied bandwidth", "parameters", "occupied_bandwidth_hz", "hz"),
    ("Estimated SNR", "parameters", "snr_db", "db"),
    ("Symbol rate", "parameters", "symbol_rate_hz", "hz"),
)


def _format_value(value: object, kind: str) -> str | None:
    if value is None or value == "":
        return None
    if kind == "text":
        return str(value)
    if not isinstance(value, (int, float)):
        return str(value)
    if kind == "ratio":
        return f"{value:.1%}"
    if kind == "hz":
        return f"{value:,.1f} Hz"
    if kind == "db":
        return f"{value:.1f} dB"
    return str(value)


def compare_reports(
    whole_report: dict | None, selection_report: dict | None
) -> list[tuple[str, str, str]]:
    """Rows of (field, whole-capture value, selection value) for fields present in both.

    A field missing from either side is dropped rather than padded, so the table never
    implies a measurement that was not made.
    """

    if not whole_report or not selection_report:
        return []
    rows: list[tuple[str, str, str]] = []
    for label, stage, key, kind in _COMPARISON_FIELDS:
        whole_stage = (whole_report.get("stages", {}) or {}).get(stage) or {}
        selection_stage = (selection_report.get("stages", {}) or {}).get(stage) or {}
        if not isinstance(whole_stage, dict) or not isinstance(selection_stage, dict):
            continue
        left = _format_value(whole_stage.get(key), kind)
        right = _format_value(selection_stage.get(key), kind)
        if left is None or right is None:
            continue
        rows.append((label, left, right))
    return rows


def region_from_drag(
    total_samples: int,
    sample_rate: float | None,
    x0: float | None,
    x1: float | None,
) -> RegionValidation:
    """Turn a drag across a time axis into the same validated region a form produces.

    A drag has no inherent direction, so the pair is ordered here rather than being
    rejected as "reversed" - dragging right-to-left is a normal gesture, unlike typing
    an end before a start. Everything after ordering goes through `select_region`, so a
    dragged region and a typed one are validated by exactly the same rules and there is
    still only one selection concept.
    """

    if x0 is None or x1 is None:
        return RegionValidation(None, False, "The drag did not produce a time range.")
    start, end = sorted((float(x0), float(x1)))
    return select_region(total_samples, sample_rate, max(0.0, start), max(0.0, end))


def capture_overview(
    iq: np.ndarray,
    sample_rate: float | None,
    rows: int = 160,
    bins: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-recording spectrogram used as the region-selection surface.

    Same computation as `window_spectrogram`, applied to the entire capture: the block
    count is capped by `rows`, so the cost is bounded by the row budget rather than the
    capture length. A region cannot be chosen sensibly from a 20 ms detail view, so the
    analyst needs a view of the whole recording to drag on.
    """

    return window_spectrogram(iq, sample_rate, rows=rows, bins=bins)


def format_timestamp(seconds: float | None) -> str:
    """A position in the recording as mm:ss.mmm, or hh:mm:ss.mmm past an hour.

    Locked-region bounds are read off the screen and compared by eye, so they are shown
    as clock positions rather than raw float seconds. The value is truncated, never
    rounded up, so a displayed timestamp never points past the sample it describes. The
    nanosecond of slack absorbs binary representation error without ever reaching the
    next millisecond for a value genuinely below it.
    """

    if seconds is None:
        return "n/a"
    value = max(0.0, float(seconds))
    milliseconds = int(value * 1_000 + 1e-6)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    if hours:
        return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def analysis_for_region(
    investigations: list[dict] | None,
    start_sample: int,
    end_sample: int,
) -> dict | None:
    """The most recent investigation that ran on exactly these samples, if any.

    Bounds must match exactly. A region that merely overlaps an earlier analysis is a
    different region, and reporting that analysis as its own would be attributing a
    measurement to samples it was never computed from - the same class of dishonesty as
    labelling whole-capture parameters as window parameters.
    """

    if not investigations:
        return None
    for investigation in reversed(investigations):
        if (investigation.get("start_sample") == start_sample
                and investigation.get("end_sample") == end_sample):
            return investigation
    return None


def investigation_name(raw: str | None, index: int) -> str:
    """The analyst's label for an investigation, or a numbered default.

    A name is a note to the analyst and nothing else: it is never read by the pipeline
    and never influences which samples are analyzed. Whitespace-only input counts as
    empty, so a stray space does not become the label.
    """

    cleaned = (raw or "").strip()
    return cleaned or f"Investigation {index}"


# -------------------------------------------------------- axis formatting & readouts


def format_time_axis_unit(duration_s: float | np.ndarray | None) -> tuple[float, str]:
    """Return (multiplier, unit_label) for context-aware time axes.

    e.g. 2.5 s -> (1.0, "s"), 0.05 s -> (1000.0, "ms"), 0.00002 s -> (1e6, "µs")
    """
    if duration_s is None:
        return 1.0, "s"
    if isinstance(duration_s, np.ndarray) or hasattr(duration_s, "__len__"):
        dur = float(np.max(np.abs(duration_s))) if len(duration_s) > 0 else 0.0
    else:
        dur = abs(float(duration_s))
    if dur >= 1.0:
        return 1.0, "s"
    if dur >= 0.001:
        return 1_000.0, "ms"
    return 1_000_000.0, "\u00b5s"


def format_frequency_axis_unit(freq: float | np.ndarray | None) -> tuple[float, str]:
    """Return (multiplier, unit_label) for human-friendly frequency axes.

    e.g. 2_500_000 Hz -> (1e-6, "MHz"), 25_000 Hz -> (1e-3, "kHz")
    """
    if freq is None:
        return 1e-3, "kHz"
    if isinstance(freq, np.ndarray) or hasattr(freq, "__len__"):
        abs_freq = float(np.max(np.abs(freq))) if len(freq) > 0 else 0.0
    else:
        abs_freq = abs(float(freq))
    if abs_freq >= 1_000_000.0:
        return 1e-6, "MHz"
    if abs_freq >= 1_000.0:
        return 1e-3, "kHz"
    return 1.0, "Hz"


def format_smart_time(seconds: float | None, span_seconds: float | None = None) -> str:
    """Format a timestamp without floating point representation artifacts.

    e.g. 0.0300000000001 -> '30.00 ms' or '0.0300 s' depending on span.
    """
    if seconds is None:
        return "n/a"
    val = float(seconds)
    ref_span = span_seconds if span_seconds is not None else abs(val)
    if ref_span >= 1.0:
        return f"{val:.4f} s"
    if ref_span >= 0.001:
        return f"{val * 1_000.0:.2f} ms"
    return f"{val * 1_000_000.0:.1f} \u00b5s"


def format_smart_frequency(freq_hz: float | None) -> str:
    """Format a frequency with appropriate sign and units without raw float noise."""
    if freq_hz is None:
        return "n/a"
    val = float(freq_hz)
    sign = "+" if val > 0 else ""
    abs_val = abs(val)
    if abs_val >= 1_000_000.0:
        return f"{sign}{val / 1e6:.3f} MHz"
    if abs_val >= 1_000.0:
        return f"{sign}{val / 1e3:.2f} kHz"
    return f"{sign}{val:.1f} Hz"


# ---------------------------------------- RF Markers & Differential Measurements


def calculate_marker_delta(
    time_a: float | None = None,
    time_b: float | None = None,
    freq_a: float | None = None,
    freq_b: float | None = None,
    power_a: float | None = None,
    power_b: float | None = None,
    **kwargs: float | None,
) -> dict[str, float | str | None]:
    """Compute exact delta between Marker A and Marker B.

    Accepts (time_a, time_b, freq_a, freq_b, power_a, power_b) or legacy (t1, f1, p1, t2, f2, p2).
    """
    t1 = kwargs.get("t1", time_a)
    t2 = kwargs.get("t2", time_b)
    f1 = kwargs.get("f1", freq_a)
    f2 = kwargs.get("f2", freq_b)
    p1 = kwargs.get("p1", power_a)
    p2 = kwargs.get("p2", power_b)

    deltas: dict[str, float | str | None] = {
        "delta_time_s": None,
        "delta_time_str": "Unset",
        "pri_freq_hz": None,
        "pri_freq_str": "Unset",
        "time_frequency_hz": None,
        "delta_freq_hz": None,
        "delta_freq_str": "Unset",
        "delta_power_db": None,
        "delta_power_str": "Unset",
    }
    if t1 is not None and t2 is not None:
        dt = abs(float(t2) - float(t1))
        deltas["delta_time_s"] = dt
        deltas["delta_time_str"] = format_smart_time(dt, dt)
        if dt > 0:
            deltas["pri_freq_hz"] = 1.0 / dt
            deltas["time_frequency_hz"] = 1.0 / dt
            deltas["pri_freq_str"] = format_smart_frequency(1.0 / dt).lstrip("+")
    if f1 is not None and f2 is not None:
        df = abs(float(f2) - float(f1))
        deltas["delta_freq_hz"] = df
        deltas["delta_freq_str"] = format_smart_frequency(df).lstrip("+")
    if p1 is not None and p2 is not None:
        dp = float(p2) - float(p1)
        deltas["delta_power_db"] = dp
        deltas["delta_power_str"] = f"{dp:+.1f} dB"
    return deltas


# ------------------------------------- Candidate Active Region (Burst) Detection


def detect_candidate_bursts(
    iq: np.ndarray,
    sample_rate: float | None,
    threshold_db_above_noise: float = 6.0,
    min_duration_s: float = 0.002,
    max_bursts: int = 8,
) -> list[dict]:
    """Conservative, explainable energy-threshold activity detector.

    Divides the capture into blocks, estimates background noise floor, and identifies
    candidate active regions where power exceeds noise floor by the threshold.
    Returns a list of candidate active regions with start/end bounds.
    """
    samples = np.asarray(iq)
    if samples.size < 128 or not sample_rate or sample_rate <= 0:
        return []

    block_size = max(64, int(round(sample_rate * 0.0005)))
    num_blocks = samples.size // block_size
    if num_blocks < 4:
        return []

    trimmed = samples[: num_blocks * block_size].reshape(num_blocks, block_size)
    block_power = np.mean(np.abs(trimmed.astype(np.complex128)) ** 2, axis=1)
    noise_floor = float(np.median(block_power))
    if noise_floor <= 1e-15:
        noise_floor = float(np.mean(block_power)) * 0.1
    if noise_floor <= 1e-15:
        return []

    threshold_linear = noise_floor * (10.0 ** (threshold_db_above_noise / 10.0))
    active = block_power >= threshold_linear

    bursts: list[dict] = []
    in_burst = False
    burst_start_block = 0
    min_blocks = max(1, int(round(min_duration_s * sample_rate / block_size)))

    for i, is_act in enumerate(active):
        if is_act and not in_burst:
            in_burst = True
            burst_start_block = i
        elif not is_act and in_burst:
            in_burst = False
            burst_length_blocks = i - burst_start_block
            if burst_length_blocks >= min_blocks:
                start_sample = burst_start_block * block_size
                end_sample = min(samples.size, i * block_size)
                start_s = start_sample / sample_rate
                end_s = end_sample / sample_rate
                dur_s = end_s - start_s
                pk_pwr = float(np.max(block_power[burst_start_block:i]))
                snr_db = 10.0 * np.log10(max(1.0, pk_pwr / noise_floor))
                bursts.append({
                    "index": len(bursts) + 1,
                    "start_seconds": start_s,
                    "end_seconds": end_s,
                    "duration_seconds": dur_s,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "peak_snr_db": float(snr_db),
                })
                if len(bursts) >= max_bursts:
                    break

    if in_burst and len(bursts) < max_bursts:
        burst_length_blocks = num_blocks - burst_start_block
        if burst_length_blocks >= min_blocks:
            start_sample = burst_start_block * block_size
            end_sample = samples.size
            start_s = start_sample / sample_rate
            end_s = end_sample / sample_rate
            dur_s = end_s - start_s
            pk_pwr = float(np.max(block_power[burst_start_block:num_blocks]))
            snr_db = 10.0 * np.log10(max(1.0, pk_pwr / noise_floor))
            bursts.append({
                "index": len(bursts) + 1,
                "start_seconds": start_s,
                "end_seconds": end_s,
                "duration_seconds": dur_s,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "peak_snr_db": float(snr_db),
            })

    return bursts


# --------------------------------------------------------- A/B Region Comparison


def compare_two_regions(
    arg1: object,
    arg2: object,
    arg3: object | None = None,
) -> list[tuple[str, str, str, str]]:
    """Compare measurable signal parameters between Region A and Region B.

    Can be called with:
      - (signal, selection_a, selection_b)
      - (investigation_dict_a, investigation_dict_b)
    """
    rows: list[tuple[str, str, str, str]] = []

    if arg3 is not None:
        signal = arg1
        selection_a = arg2
        selection_b = arg3
        samples_a = extract_window(signal.iq, selection_a)
        samples_b = extract_window(signal.iq, selection_b)
        meas_a = window_measurements(samples_a, signal.sample_rate)
        meas_b = window_measurements(samples_b, signal.sample_rate)

        dur_a = selection_a.duration_seconds or 0.0
        dur_b = selection_b.duration_seconds or 0.0
        diff_dur = dur_b - dur_a
        rows.append((
            "Duration",
            format_smart_time(dur_a, max(dur_a, dur_b)),
            format_smart_time(dur_b, max(dur_a, dur_b)),
            f"{diff_dur * 1000:+.2f} ms",
        ))

        rows.append((
            "Samples",
            f"{selection_a.length_samples:,}",
            f"{selection_b.length_samples:,}",
            f"{selection_b.length_samples - selection_a.length_samples:+,}",
        ))

        rms_a = meas_a.get("rms_amplitude", 0.0)
        rms_b = meas_b.get("rms_amplitude", 0.0)
        diff_rms = rms_b - rms_a
        rows.append((
            "RMS Amplitude",
            f"{rms_a:.4f}",
            f"{rms_b:.4f}",
            f"{diff_rms:+.4f}",
        ))

        pk_a = meas_a.get("peak_amplitude", 0.0)
        pk_b = meas_b.get("peak_amplitude", 0.0)
        diff_pk = pk_b - pk_a
        rows.append((
            "Peak Amplitude",
            f"{pk_a:.4f}",
            f"{pk_b:.4f}",
            f"{diff_pk:+.4f}",
        ))

        if "peak_offset_hz" in meas_a and "peak_offset_hz" in meas_b:
            f_a = meas_a["peak_offset_hz"]
            f_b = meas_b["peak_offset_hz"]
            diff_f = f_b - f_a
            rows.append((
                "Peak Offset Frequency",
                format_smart_frequency(f_a),
                format_smart_frequency(f_b),
                f"{diff_f / 1000:+.2f} kHz",
            ))

        if "amplitude_cv" in meas_a and "amplitude_cv" in meas_b:
            cv_a = meas_a["amplitude_cv"]
            cv_b = meas_b["amplitude_cv"]
            diff_cv = cv_b - cv_a
            rows.append((
                "Amplitude CV",
                f"{cv_a:.4f}",
                f"{cv_b:.4f}",
                f"{diff_cv:+.4f}",
            ))

        return rows

    # Investigation dict comparison
    inv_a = arg1 if isinstance(arg1, dict) else {}
    inv_b = arg2 if isinstance(arg2, dict) else {}

    dur_a = float(inv_a.get("end_s", 0.0) - inv_a.get("start_s", 0.0))
    dur_b = float(inv_b.get("end_s", 0.0) - inv_b.get("start_s", 0.0))
    diff_dur = dur_b - dur_a
    rows.append((
        "Duration",
        format_smart_time(dur_a, max(dur_a, dur_b)),
        format_smart_time(dur_b, max(dur_a, dur_b)),
        f"{diff_dur * 1000:+.2f} ms",
    ))

    s_a = int(inv_a.get("samples", 0))
    s_b = int(inv_b.get("samples", 0))
    rows.append((
        "Samples",
        f"{s_a:,}",
        f"{s_b:,}",
        f"{s_b - s_a:+,}",
    ))

    rep_a = inv_a.get("report", {}) or {}
    rep_b = inv_b.get("report", {}) or {}
    stages_a = rep_a.get("stages", {}) or {}
    stages_b = rep_b.get("stages", {}) or {}
    fus_a = stages_a.get("fusion", {}) or {}
    fus_b = stages_b.get("fusion", {}) or {}

    lbl_a = fus_a.get("label", "Unclassified")
    lbl_b = fus_b.get("label", "Unclassified")
    rows.append((
        "Modulation",
        lbl_a,
        lbl_b,
        "Identical" if lbl_a == lbl_b else "Different",
    ))

    tr_a = fus_a.get("trust_score")
    tr_b = fus_b.get("trust_score")
    if isinstance(tr_a, (int, float)) and isinstance(tr_b, (int, float)):
        rows.append((
            "Modulation Trust",
            f"{tr_a:.3f}",
            f"{tr_b:.3f}",
            f"{tr_b - tr_a:+.3f}",
        ))

    p_a = stages_a.get("parameters", {}) or {}
    p_b = stages_b.get("parameters", {}) or {}

    snr_a = p_a.get("snr_db")
    snr_b = p_b.get("snr_db")
    if isinstance(snr_a, (int, float)) and isinstance(snr_b, (int, float)):
        rows.append((
            "Estimated SNR",
            f"{snr_a:.1f} dB",
            f"{snr_b:.1f} dB",
            f"{snr_b - snr_a:+.1f} dB",
        ))

    obw_a = p_a.get("occupied_bandwidth_hz")
    obw_b = p_b.get("occupied_bandwidth_hz")
    if isinstance(obw_a, (int, float)) and isinstance(obw_b, (int, float)):
        rows.append((
            "Occupied Bandwidth",
            format_smart_frequency(obw_a),
            format_smart_frequency(obw_b),
            f"{(obw_b - obw_a) / 1000:+.2f} kHz",
        ))

    return rows

