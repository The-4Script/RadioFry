"""Where RadioFry has actually been measured, and where it degrades.

This module turns the recorded end-to-end benchmark into an SNR x samples-per-symbol
operating envelope. It is a *capability and limitation* view, not a performance claim:
every value plotted is a measurement that was taken, and every combination that was not
measured stays visibly absent.

Provenance
----------
The source is `reports/benchmark_v039/results.pkl` (BANK.md Entry 039): 480 digital
captures over 8 production classes x samples-per-symbol 4/8/16/32 x SNR 20/15/10/5/0 dB x
3 seeds, at 200 kHz with 8192 samples, scored against generator ground truth *after* the
pipeline ran. That directory is gitignored, so it may simply be absent - which this module
reports as missing evidence rather than treating as an error.

Honesty rules this module follows
---------------------------------
* Nothing is interpolated, smoothed or extrapolated. A cell holds the median of the
  measurements actually recorded in it, or nothing at all.
* "No measurement" is separated from "measured and bad". A cell where the pipeline never
  emitted bits is reported as `NO_OUTPUT`, because a refusal to demodulate is a real
  limitation and must not be mistaken for an absent experiment - nor averaged in as if it
  were a BER.
* The BER population is stated rather than assumed. `ber_prod` exists for every capture
  the pipeline demodulated, *including* the 124 it had misclassified; demodulating with
  the wrong scheme really does produce garbage bits, so those belong in an end-to-end
  view. The count of correctly classified captures is carried alongside so the reader can
  separate the two.
* Thresholds used to colour a cell are display aids, stated explicitly and configurable.
  They are not a project-defined guarantee.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BENCHMARK = Path("reports/benchmark_v039")

# A cell is coloured against this only as a reading aid. 1e-2 is the conventional point
# below which a practical FEC can close the link; RadioFry itself defines no such
# guarantee, which is why the threshold is a parameter and is always shown next to the
# surface.
DEFAULT_BER_THRESHOLD = 1e-2

# BER = 0 means "no errors in the bits that were compared", not "error free". On a log
# axis it has no position, so it is drawn at this floor and labelled as a bound. The
# benchmark compared a few thousand bits per capture, so the true rate is somewhere below
# roughly 1e-3; 1e-4 keeps those points visibly distinct without implying more.
DEFAULT_LOG_FLOOR = 1e-4

VALIDATED = "validated"
DEGRADED = "degraded"
NO_OUTPUT = "no_output"
NOT_BENCHMARKED = "not_benchmarked"

METRICS: dict[str, dict[str, Any]] = {
    "ber_prod": {
        "label": "Median end-to-end BER (production)",
        "field": "ber_prod",
        "kind": "ber",
        "description": "Bit error rate from the pipeline's own decisions, over every "
                       "capture in the cell that produced bits.",
    },
    "ber_oracle": {
        "label": "Median BER with oracle symbol rate",
        "field": "ber_oracle",
        "kind": "ber",
        "description": "Bit error rate when the true symbol rate is supplied instead of "
                       "estimated. The gap to the production surface is the symbol-rate "
                       "estimator's contribution.",
    },
    "classification_accuracy": {
        "label": "Fused classification accuracy",
        "field": "correct",
        "kind": "fraction",
        "description": "Fraction of captures in the cell whose fused modulation "
                       "decision matched ground truth.",
    },
    "symbol_rate_accuracy": {
        "label": "Symbol rate within 10%",
        "field": None,
        "kind": "fraction",
        "description": "Fraction of captures whose estimated symbol rate landed within "
                       "10% of truth.",
    },
}


@dataclass(frozen=True)
class BenchmarkSource:
    """Loaded benchmark records plus where they came from and what is wrong with them."""

    records: list[dict]
    config: dict
    path: Path
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.records)

    @property
    def description(self) -> str:
        entry = self.config.get("entry")
        date = str(self.config.get("date", ""))[:10]
        parts = [f"{len(self.records):,} digital captures"]
        if entry:
            parts.append(f"benchmark entry {entry}")
        if date:
            parts.append(date)
        return " · ".join(parts)


@dataclass(frozen=True)
class CapabilitySurface:
    """One modulation's measured operating envelope.

    `values` is indexed `[sps, snr]` and holds NaN where nothing was measured. `status`
    carries the same shape and distinguishes an unmeasured cell from a measured-and-poor
    one.
    """

    modulation: str
    metric: str
    metric_label: str
    snr_axis: np.ndarray
    sps_axis: np.ndarray
    values: np.ndarray
    captures: np.ndarray
    measurements: np.ndarray
    classified_correct: np.ndarray
    status: np.ndarray
    ber_threshold: float
    kind: str = "ber"
    warnings: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.values.size > 0

    @property
    def measured_cells(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.values)))

    @property
    def total_cells(self) -> int:
        return int(self.values.size)

    @property
    def total_captures(self) -> int:
        return int(np.sum(self.captures))

    @property
    def total_measurements(self) -> int:
        return int(np.sum(self.measurements))

    def coverage_note(self) -> str:
        """One line an investigator can read without cross-checking the arrays."""
        missing = self.total_cells - self.measured_cells
        note = (f"{self.measured_cells} of {self.total_cells} cells carry a measurement "
                f"({self.total_measurements:,} of {self.total_captures:,} captures "
                "produced one)")
        if missing:
            note += f"; {missing} cell(s) have no measurement and are left empty"
        return note + "."


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK) -> BenchmarkSource:
    """Read the recorded benchmark, reporting rather than raising on any problem.

    The artifacts live under a gitignored directory, so absence is an ordinary outcome
    and the caller gets an empty source carrying the reason.
    """

    import json
    import pickle

    root = Path(path)
    results = root / "results.pkl"
    if not results.is_file():
        return BenchmarkSource(
            [], {}, root,
            (f"No benchmark artifact at {results}. This directory is gitignored, so it "
             "is present only where the benchmark has been run locally.",))

    try:
        with results.open("rb") as handle:
            raw = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
            ImportError, ValueError) as error:
        return BenchmarkSource([], {}, root,
                               (f"Could not read {results}: {error}",))

    if not isinstance(raw, list):
        return BenchmarkSource([], {}, root,
                               (f"{results} does not hold a list of records.",))

    warnings: list[str] = []
    config: dict = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"Could not read {config_path.name}: {error}")

    required = {"kind", "true", "sps", "snr"}
    digital, skipped = [], 0
    for record in raw:
        if not isinstance(record, dict) or not required <= record.keys():
            skipped += 1
            continue
        if record.get("kind") != "digital":
            continue
        digital.append(record)
    if skipped:
        warnings.append(
            f"{skipped} record(s) were missing required fields and were ignored.")
    if not digital:
        warnings.append("The artifact holds no digital captures to build a surface from.")

    return BenchmarkSource(digital, config, root, tuple(warnings))


def available_modulations(records: list[dict]) -> list[str]:
    """Modulations the benchmark actually contains, in a stable order."""

    return sorted({str(record["true"]) for record in records if record.get("true")})


def _metric_values(records: list[dict], metric: str) -> list[float]:
    """The measurements a cell contributes, dropping captures that produced none."""

    spec = METRICS[metric]
    if metric == "symbol_rate_accuracy":
        values = []
        for record in records:
            true_rs, estimated = record.get("true_rs"), record.get("est_rs")
            if not isinstance(true_rs, (int, float)) or not true_rs:
                continue
            if not isinstance(estimated, (int, float)):
                values.append(0.0)
                continue
            values.append(float(abs(estimated - true_rs) / abs(true_rs) <= 0.10))
        return values

    field_name = spec["field"]
    if spec["kind"] == "fraction":
        return [float(bool(record.get(field_name)))
                for record in records if field_name in record]

    values = []
    for record in records:
        value = record.get(field_name)
        # None means the pipeline emitted no bits for this capture. That is a limitation,
        # counted separately - never folded in as a BER of 0 or of 0.5.
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


def build_surface(
    records: list[dict],
    modulation: str,
    metric: str = "ber_prod",
    ber_threshold: float = DEFAULT_BER_THRESHOLD,
) -> CapabilitySurface:
    """Aggregate the recorded captures for one modulation into an SNR x SPS surface."""

    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    spec = METRICS[metric]

    selected = [r for r in records if str(r.get("true")) == str(modulation)]
    warnings: list[str] = []
    if not selected:
        return CapabilitySurface(
            modulation=str(modulation), metric=metric, metric_label=spec["label"],
            snr_axis=np.empty(0), sps_axis=np.empty(0),
            values=np.empty((0, 0)), captures=np.empty((0, 0), dtype=int),
            measurements=np.empty((0, 0), dtype=int),
            classified_correct=np.empty((0, 0), dtype=int),
            status=np.empty((0, 0), dtype=object), ber_threshold=ber_threshold,
            kind=spec["kind"],
            warnings=(f"The benchmark holds no captures for {modulation}.",))

    # Axes come from the data, so a benchmark run over a different grid still plots.
    snr_axis = np.array(sorted({float(r["snr"]) for r in selected}))
    sps_axis = np.array(sorted({int(r["sps"]) for r in selected}))

    shape = (sps_axis.size, snr_axis.size)
    values = np.full(shape, np.nan, dtype=float)
    captures = np.zeros(shape, dtype=int)
    measurements = np.zeros(shape, dtype=int)
    correct = np.zeros(shape, dtype=int)
    status = np.full(shape, NOT_BENCHMARKED, dtype=object)

    for row, sps in enumerate(sps_axis):
        for column, snr in enumerate(snr_axis):
            cell = [r for r in selected
                    if int(r["sps"]) == int(sps) and float(r["snr"]) == float(snr)]
            captures[row, column] = len(cell)
            correct[row, column] = sum(1 for r in cell if r.get("correct"))
            if not cell:
                continue
            measured = _metric_values(cell, metric)
            measurements[row, column] = len(measured)
            if not measured:
                # Captures were run here and the pipeline produced nothing to score.
                status[row, column] = NO_OUTPUT
                continue
            if spec["kind"] == "ber":
                # Median: BER is heavily skewed, and one capture that failed outright at
                # ~0.5 would drag a mean away from what the cell typically delivers.
                value = float(np.median(measured))
                status[row, column] = (VALIDATED if value <= ber_threshold else DEGRADED)
            else:
                # Mean: an accuracy is the fraction of captures that succeeded. A median
                # of per-capture booleans is only a majority vote and would report 3
                # captures out of 4 correct as 1.0.
                value = float(np.mean(measured))
                status[row, column] = (VALIDATED if value >= 0.5 else DEGRADED)
            values[row, column] = value

    unmeasured = int(np.count_nonzero(status == NO_OUTPUT))
    if unmeasured and spec["kind"] == "ber":
        warnings.append(
            f"{unmeasured} cell(s) were exercised but produced no bits to score, so they "
            "carry no BER. They are shown as 'no output', which is a limitation of the "
            "pipeline at those conditions rather than a gap in the experiment.")

    return CapabilitySurface(
        modulation=str(modulation), metric=metric, metric_label=spec["label"],
        snr_axis=snr_axis, sps_axis=sps_axis, values=values, captures=captures,
        measurements=measurements, classified_correct=correct, status=status,
        ber_threshold=ber_threshold, kind=spec["kind"], warnings=tuple(warnings))


def log_values(values: np.ndarray, floor: float = DEFAULT_LOG_FLOOR) -> np.ndarray:
    """log10 of a BER surface, with zeros placed at an explicit floor.

    BER = 0 has no logarithm. Clamping is the only honest option that keeps the point
    visible, so the floor is a named constant the caller shows in the legend rather than
    a hidden fudge. NaN (no measurement) stays NaN and therefore stays unplotted.
    """

    clamped = np.where(np.isnan(values), np.nan, np.maximum(values, floor))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(clamped)


def coverage_table(surface: CapabilitySurface) -> list[dict]:
    """Per-cell provenance: what was run, what was measured, what it says."""

    rows = []
    for row, sps in enumerate(surface.sps_axis):
        for column, snr in enumerate(surface.snr_axis):
            value = surface.values[row, column]
            rows.append({
                "sps": int(sps),
                "snr_db": float(snr),
                "captures": int(surface.captures[row, column]),
                "measurements": int(surface.measurements[row, column]),
                "classified_correct": int(surface.classified_correct[row, column]),
                "value": None if np.isnan(value) else float(value),
                "status": str(surface.status[row, column]),
            })
    return rows


def compare_modulations(
    records: list[dict],
    metric: str = "ber_prod",
    ber_threshold: float = DEFAULT_BER_THRESHOLD,
) -> list[dict]:
    """One summary row per modulation, for choosing which surface to inspect."""

    summary = []
    for modulation in available_modulations(records):
        surface = build_surface(records, modulation, metric, ber_threshold)
        finite = surface.values[np.isfinite(surface.values)]
        summary.append({
            "modulation": modulation,
            "captures": surface.total_captures,
            "measurements": surface.total_measurements,
            "measured_cells": surface.measured_cells,
            "total_cells": surface.total_cells,
            "median": float(np.median(finite)) if finite.size else None,
            "best": float(np.min(finite)) if finite.size else None,
            "validated_cells": int(np.count_nonzero(surface.status == VALIDATED)),
            "no_output_cells": int(np.count_nonzero(surface.status == NO_OUTPUT)),
        })
    return summary
