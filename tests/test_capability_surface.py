"""The capability surface plots measurements, and leaves gaps as gaps.

The dangerous failure here is not a crash - it is a smooth, confident surface implying
RadioFry was validated at conditions where nothing was ever measured. These tests pin
the distinctions that prevent that: measured vs unmeasured, measured-and-bad vs
produced-no-output, and a median taken over the population actually stated.

Most tests build synthetic records so they run without the benchmark artifact, which
lives under a gitignored directory. The few that read the real artifact skip when it is
absent.
"""

from pathlib import Path

import numpy as np
import pytest

from radiofry.evaluation.capability_surface import (
    DEGRADED,
    NO_OUTPUT,
    NOT_BENCHMARKED,
    VALIDATED,
    available_modulations,
    build_surface,
    compare_modulations,
    coverage_table,
    load_benchmark,
    log_values,
)

BENCHMARK = Path(__file__).resolve().parents[1] / "reports" / "benchmark_v039"
requires_benchmark = pytest.mark.skipif(
    not (BENCHMARK / "results.pkl").is_file(),
    reason="benchmark artifact is gitignored and absent")


def _record(modulation="QPSK", sps=8, snr=20.0, ber_prod=0.0, ber_oracle=0.0,
            correct=True, true_rs=25_000.0, est_rs=25_000.0, **extra):
    record = {
        "kind": "digital", "true": modulation, "sps": sps, "snr": snr,
        "ber_prod": ber_prod, "ber_oracle": ber_oracle, "correct": correct,
        "true_rs": true_rs, "est_rs": est_rs,
    }
    record.update(extra)
    return record


# --- aggregation is over the population that was stated -----------------------------------


def test_a_cell_holds_the_median_of_its_measurements() -> None:
    records = [_record(ber_prod=b) for b in (0.10, 0.20, 0.90)]

    surface = build_surface(records, "QPSK")

    assert surface.values[0, 0] == pytest.approx(0.20), "median, not mean"
    assert surface.measurements[0, 0] == 3
    assert surface.captures[0, 0] == 3


def test_captures_that_produced_no_bits_are_excluded_from_the_median() -> None:
    """A refusal to demodulate is not a BER of zero, nor of one half."""
    records = [_record(ber_prod=0.01), _record(ber_prod=None),
               _record(ber_prod=0.03)]

    surface = build_surface(records, "QPSK")

    assert surface.values[0, 0] == pytest.approx(0.02)
    assert surface.measurements[0, 0] == 2, "only the two that produced bits"
    assert surface.captures[0, 0] == 3, "but all three were run"


def test_a_cell_with_no_measurement_at_all_is_marked_no_output() -> None:
    records = [_record(ber_prod=None), _record(ber_prod=None)]

    surface = build_surface(records, "QPSK")

    assert np.isnan(surface.values[0, 0])
    assert surface.status[0, 0] == NO_OUTPUT
    assert surface.captures[0, 0] == 2
    assert surface.measurements[0, 0] == 0
    assert any("no bits" in w for w in surface.warnings)


def test_no_output_is_distinct_from_never_benchmarked() -> None:
    """Two different absences that must never be shown as the same thing."""
    records = [_record(sps=4, snr=20.0, ber_prod=None),
               _record(sps=8, snr=20.0, ber_prod=0.01),
               _record(sps=8, snr=10.0, ber_prod=0.02)]

    surface = build_surface(records, "QPSK")

    sps4, sps8 = list(surface.sps_axis).index(4), list(surface.sps_axis).index(8)
    snr20 = list(surface.snr_axis).index(20.0)
    snr10 = list(surface.snr_axis).index(10.0)
    assert surface.status[sps4, snr20] == NO_OUTPUT, "run, produced nothing"
    assert surface.status[sps4, snr10] == NOT_BENCHMARKED, "never run"
    assert surface.status[sps8, snr20] == VALIDATED


def test_nothing_is_interpolated_into_an_empty_cell() -> None:
    records = [_record(sps=4, snr=20.0, ber_prod=0.001),
               _record(sps=16, snr=20.0, ber_prod=0.001)]

    surface = build_surface(records, "QPSK")

    middle = list(surface.sps_axis).index(8) if 8 in surface.sps_axis else None
    assert middle is None, "an unmeasured sps must not appear on the axis at all"
    assert surface.measured_cells == 2
    assert np.count_nonzero(np.isnan(surface.values)) == surface.total_cells - 2


def test_the_correctly_classified_count_is_carried_alongside_the_ber() -> None:
    """`ber_prod` includes misclassified captures, so the reader needs both numbers."""
    records = [_record(ber_prod=0.001, correct=True),
               _record(ber_prod=0.500, correct=False),
               _record(ber_prod=0.480, correct=False)]

    surface = build_surface(records, "QPSK")

    assert surface.measurements[0, 0] == 3
    assert surface.classified_correct[0, 0] == 1
    assert surface.values[0, 0] == pytest.approx(0.480)


# --- axes come from the data ----------------------------------------------------------------


def test_axes_are_taken_from_the_records_and_sorted() -> None:
    records = [_record(sps=32, snr=0.0), _record(sps=4, snr=20.0),
               _record(sps=8, snr=10.0)]

    surface = build_surface(records, "QPSK")

    assert list(surface.sps_axis) == [4, 8, 32]
    assert list(surface.snr_axis) == [0.0, 10.0, 20.0]
    assert surface.values.shape == (3, 3)


def test_an_unknown_modulation_yields_an_empty_surface_with_a_reason() -> None:
    surface = build_surface([_record()], "NOT-A-MODULATION")

    assert not surface.ok
    assert surface.warnings and "no captures" in surface.warnings[0].lower()


def test_available_modulations_lists_only_what_is_present() -> None:
    records = [_record(modulation="QPSK"), _record(modulation="BPSK"),
               _record(modulation="QPSK")]

    assert available_modulations(records) == ["BPSK", "QPSK"]


def test_an_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_surface([_record()], "QPSK", metric="not_a_metric")


# --- thresholds and metrics -------------------------------------------------------------------


def test_the_threshold_only_labels_and_is_configurable() -> None:
    records = [_record(ber_prod=0.05)]

    strict = build_surface(records, "QPSK", ber_threshold=1e-2)
    lenient = build_surface(records, "QPSK", ber_threshold=1e-1)

    assert strict.status[0, 0] == DEGRADED
    assert lenient.status[0, 0] == VALIDATED
    assert strict.values[0, 0] == lenient.values[0, 0], "the measurement never moves"


def test_the_oracle_metric_uses_its_own_field() -> None:
    records = [_record(ber_prod=0.5, ber_oracle=0.001)]

    production = build_surface(records, "QPSK", metric="ber_prod")
    oracle = build_surface(records, "QPSK", metric="ber_oracle")

    assert production.values[0, 0] == pytest.approx(0.5)
    assert oracle.values[0, 0] == pytest.approx(0.001)


def test_fraction_metrics_average_booleans() -> None:
    records = [_record(correct=True), _record(correct=True), _record(correct=False),
               _record(correct=False)]

    surface = build_surface(records, "QPSK", metric="classification_accuracy")

    assert surface.kind == "fraction"
    assert surface.values[0, 0] == pytest.approx(0.5)


def test_symbol_rate_accuracy_uses_a_ten_percent_window() -> None:
    records = [_record(true_rs=25_000.0, est_rs=25_500.0),   # 2% -> within
               _record(true_rs=25_000.0, est_rs=30_000.0),   # 20% -> outside
               _record(true_rs=25_000.0, est_rs=None)]       # no estimate -> outside

    surface = build_surface(records, "QPSK", metric="symbol_rate_accuracy")

    assert surface.values[0, 0] == pytest.approx(1 / 3)


# --- log handling -------------------------------------------------------------------------------


def test_zero_ber_is_floored_rather_than_dropped_or_faked() -> None:
    values = np.array([[0.0, 0.1, np.nan]])

    logged = log_values(values, floor=1e-4)

    assert logged[0, 0] == pytest.approx(-4.0), "zero sits at the stated floor"
    assert logged[0, 1] == pytest.approx(-1.0)
    assert np.isnan(logged[0, 2]), "a missing measurement must stay missing"


def test_log_values_never_returns_infinities() -> None:
    logged = log_values(np.array([[0.0, 1e-30, 1.0]]))

    assert np.all(np.isfinite(logged))


# --- provenance and comparison -------------------------------------------------------------------


def test_the_coverage_table_reports_every_cell() -> None:
    records = [_record(sps=4, ber_prod=0.01), _record(sps=8, ber_prod=None)]

    rows = coverage_table(build_surface(records, "QPSK"))

    assert len(rows) == 2
    by_sps = {row["sps"]: row for row in rows}
    assert by_sps[4]["value"] == pytest.approx(0.01)
    assert by_sps[8]["value"] is None
    assert by_sps[8]["status"] == NO_OUTPUT


def test_comparison_summarises_each_modulation() -> None:
    records = [_record(modulation="QPSK", ber_prod=0.001),
               _record(modulation="BPSK", ber_prod=0.4)]

    summary = {row["modulation"]: row for row in compare_modulations(records)}

    assert summary["QPSK"]["validated_cells"] == 1
    assert summary["BPSK"]["validated_cells"] == 0
    assert summary["QPSK"]["median"] == pytest.approx(0.001)


# --- missing and malformed artifacts ---------------------------------------------------------------


def test_a_missing_benchmark_directory_is_reported_not_raised() -> None:
    source = load_benchmark(Path("does") / "not" / "exist")

    assert not source.ok
    assert source.records == []
    assert source.warnings and "No benchmark artifact" in source.warnings[0]


def test_a_malformed_artifact_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "results.pkl").write_bytes(b"this is not a pickle")

    source = load_benchmark(tmp_path)

    assert not source.ok
    assert source.warnings


def test_an_artifact_of_the_wrong_shape_is_reported(tmp_path: Path) -> None:
    import pickle

    with (tmp_path / "results.pkl").open("wb") as handle:
        pickle.dump({"not": "a list"}, handle)

    source = load_benchmark(tmp_path)

    assert not source.ok
    assert "list of records" in source.warnings[0]


def test_records_missing_required_fields_are_skipped(tmp_path: Path) -> None:
    import pickle

    with (tmp_path / "results.pkl").open("wb") as handle:
        pickle.dump([_record(), {"kind": "digital"}, {"junk": 1}], handle)

    source = load_benchmark(tmp_path)

    assert len(source.records) == 1
    assert any("missing required fields" in w for w in source.warnings)


def test_analog_records_are_not_treated_as_digital(tmp_path: Path) -> None:
    import pickle

    analog = {"kind": "analog", "true": "AM-DSB", "sps": 0, "snr": 20.0}
    with (tmp_path / "results.pkl").open("wb") as handle:
        pickle.dump([_record(), analog], handle)

    source = load_benchmark(tmp_path)

    assert len(source.records) == 1
    assert source.records[0]["true"] == "QPSK"


# --- against the real artifact ------------------------------------------------------------------------


@requires_benchmark
def test_the_real_benchmark_loads_with_the_documented_shape() -> None:
    source = load_benchmark(BENCHMARK)

    assert source.ok
    assert len(source.records) == 480, "Entry 039 recorded 480 digital captures"
    assert set(available_modulations(source.records)) == {
        "8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK"}


@requires_benchmark
def test_every_real_surface_is_built_from_three_seeds_per_cell() -> None:
    source = load_benchmark(BENCHMARK)

    for modulation in available_modulations(source.records):
        surface = build_surface(source.records, modulation)
        assert surface.values.shape == (4, 5), f"{modulation}: 4 sps x 5 SNR"
        assert np.all(surface.captures == 3), f"{modulation}: 3 seeds per cell"


@requires_benchmark
def test_the_real_surface_keeps_its_gaps() -> None:
    """The measured envelope is genuinely incomplete, and must stay that way."""
    source = load_benchmark(BENCHMARK)

    total_missing = 0
    for modulation in available_modulations(source.records):
        surface = build_surface(source.records, modulation)
        total_missing += surface.total_cells - surface.measured_cells
    assert total_missing > 0, (
        "the benchmark has cells where no bits were produced; a surface showing none "
        "would mean the gaps were filled in")


@requires_benchmark
def test_the_oracle_surface_is_complete_where_production_is_not() -> None:
    """Entry 039's central finding, reproduced from the artifact itself."""
    source = load_benchmark(BENCHMARK)

    production = build_surface(source.records, "GFSK", metric="ber_prod")
    oracle = build_surface(source.records, "GFSK", metric="ber_oracle")

    assert oracle.measured_cells >= production.measured_cells
    assert oracle.measured_cells == oracle.total_cells, (
        "ber_oracle was recorded for every capture")
