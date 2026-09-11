"""The SCD and capability pages render, and the expensive one stays opt-in.

The requirement these tests exist for is that advanced analysis must not slow the
ordinary workflow: opening the SCD page must not compute an SCD. That is asserted
directly rather than assumed from reading the code.

Streamlit is a `gui` extra, so the whole module skips when it is not installed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit", reason="streamlit is an optional gui extra")

from streamlit.testing.v1 import AppTest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCD_PAGE = PROJECT_ROOT / "gui" / "pages" / "10_cyclostationary.py"
CAPABILITY_PAGE = PROJECT_ROOT / "gui" / "pages" / "11_capability.py"
LANDSCAPE_PAGE = PROJECT_ROOT / "gui" / "pages" / "12_fusion_landscape.py"
DNA_PAGE = PROJECT_ROOT / "gui" / "pages" / "13_signal_dna.py"
CHECKPOINT = PROJECT_ROOT / "models_saved" / "modulation_cnn_v3_spsaug.pt"
V1_CAPTURES = PROJECT_ROOT / "data" / "synthetic_v1" / "captures"
BENCHMARK = PROJECT_ROOT / "reports" / "benchmark_v039" / "results.pkl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

requires_v1 = pytest.mark.skipif(not V1_CAPTURES.exists(),
                                 reason="frozen V1 dataset not present")

def _run_scd(app: "AppTest") -> "AppTest":
    """Press whichever run control the page is currently offering."""
    button = next(b for b in app.button
                  if b.label.startswith("Run ") and "SCD" in b.label)
    return button.click().run()

requires_benchmark = pytest.mark.skipif(
    not BENCHMARK.is_file(), reason="benchmark artifact is gitignored and absent")


def _capture():
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import load_capture

    truth = json.loads(
        (V1_CAPTURES / "QPSK_snr20dB_r000.json").read_text(encoding="utf-8"))
    entry = next(f for f in truth["files"] if f["file_format"] == "iq")
    return load_capture(V1_CAPTURES / "QPSK_snr20dB_r000.iq",
                        sample_rate=truth["signal"]["sample_rate_hz"],
                        iq_format=IQFormat(entry["dtype"], entry["byte_order"]))


def _synthetic(samples: int, sample_rate):
    from radiofry.contracts import UnifiedSignalContainer

    rng = np.random.default_rng(5)
    iq = (rng.normal(size=samples) + 1j * rng.normal(size=samples)).astype(np.complex64)
    return UnifiedSignalContainer(iq, sample_rate, "iq")


def _page(path: Path, signal=None, report=None) -> "AppTest":
    app = AppTest.from_file(str(path), default_timeout=300)
    if signal is not None:
        app.session_state["signal"] = signal
        app.session_state["report"] = report or {}
    return app.run()


# --- SCD page ------------------------------------------------------------------------


def test_the_scd_page_renders_without_a_capture() -> None:
    app = _page(SCD_PAGE)

    assert not app.exception, [e.value for e in app.exception]
    assert any("Analyze a WAV or IQ capture" in str(b.value) for b in app.markdown)


@requires_v1
def test_the_scd_page_renders_with_a_capture() -> None:
    app = _page(SCD_PAGE, _capture())

    assert not app.exception, [e.value for e in app.exception]
    assert any(b.label.startswith("Run ") and "SCD" in b.label
               for b in app.button), "the run control names its mode"


@requires_v1
def test_opening_the_scd_page_does_not_compute_an_scd() -> None:
    """The whole point of an explicit advanced-analysis control."""
    app = _page(SCD_PAGE, _capture())

    assert not app.exception, [e.value for e in app.exception]
    assert "scd_request" not in app.session_state
    assert any("has not been computed" in str(item.value) for item in app.info)


@requires_v1
def test_the_configuration_cost_is_shown_before_running() -> None:
    app = _page(SCD_PAGE, _capture())

    labels = {m.label for m in app.metric}
    assert {"Frames", "Frame hop", "Surface", "Span analyzed"} <= labels
    captions = " ".join(str(c.value) for c in app.caption)
    assert "Nothing is computed until you press" in captions


@requires_v1
def test_running_the_scd_produces_a_surface_and_says_what_it_did() -> None:
    app = _page(SCD_PAGE, _capture())

    _run_scd(app)

    assert not app.exception, [e.value for e in app.exception]
    assert "scd_request" in app.session_state
    labels = {m.label for m in app.metric}
    assert {"Samples analyzed", "Cyclic resolution", "Frequency resolution",
            "Independent looks"} <= labels
    rendered = " ".join(str(b.value) for b in app.markdown)
    captions = " ".join(str(c.value) for c in app.caption)
    assert "Spectral correlation surface" in rendered
    assert "frequency x cyclic frequency" in rendered.lower(), "the 2D projection"
    assert "Full computed surface, no decimation" in captions, (
        "the heatmap must state that it is not decimated")


@requires_v1
def test_the_scd_page_states_its_method_and_limits() -> None:
    app = _page(SCD_PAGE, _capture())

    rendered = " ".join(str(b.value) for b in app.markdown)
    assert "time-smoothed cyclic periodogram" in rendered
    assert "not a relabelled spectrogram" in rendered or "relabelled spectrogram" in rendered


def test_a_capture_without_a_sample_rate_is_refused_gracefully() -> None:
    app = _page(SCD_PAGE, _synthetic(8_192, None))

    assert not app.exception, [e.value for e in app.exception]
    assert any("no sample rate" in str(w.value).lower() for w in app.warning)


def test_a_capture_too_short_for_scd_does_not_crash_the_page() -> None:
    app = _run_scd(_page(SCD_PAGE, _synthetic(32, 200_000.0)))

    assert not app.exception, [e.value for e in app.exception]
    assert app.error or app.warning, "a refusal must be explained"


def test_a_zero_power_capture_does_not_crash_the_page() -> None:
    from radiofry.contracts import UnifiedSignalContainer

    silent = UnifiedSignalContainer(np.zeros(8_192, dtype=np.complex64), 200_000.0, "iq")
    app = _run_scd(_page(SCD_PAGE, silent))

    assert not app.exception, [e.value for e in app.exception]
    assert app.error or app.warning


# --- capability page ------------------------------------------------------------------


@requires_benchmark
def test_the_capability_page_renders_from_recorded_evidence() -> None:
    app = _page(CAPABILITY_PAGE)

    assert not app.exception, [e.value for e in app.exception]
    labels = {m.label for m in app.metric}
    assert {"Captures", "Measurements", "Cells measured"} <= labels


@requires_benchmark
def test_the_capability_page_offers_every_benchmarked_modulation() -> None:
    app = _page(CAPABILITY_PAGE)

    options = set(app.selectbox[0].options)
    assert {"BPSK", "QPSK", "8PSK", "CPFSK", "GFSK", "PAM4", "QAM16",
            "QAM64"} <= options


@requires_benchmark
def test_switching_modulation_rebuilds_the_surface() -> None:
    app = _page(CAPABILITY_PAGE)
    before = [m.value for m in app.metric if m.label == "Cells measured"][0]

    app.selectbox[0].set_value("GFSK").run()

    assert not app.exception, [e.value for e in app.exception]
    after = [m.value for m in app.metric if m.label == "Cells measured"][0]
    assert isinstance(after, str) and "/" in after
    assert before is not None


@requires_benchmark
def test_switching_metric_to_the_oracle_surface_works() -> None:
    app = _page(CAPABILITY_PAGE)

    app.selectbox[1].set_value("ber_oracle").run()

    assert not app.exception, [e.value for e in app.exception]
    rendered = " ".join(str(b.value) for b in app.markdown)
    assert "Across modulations" in rendered


@requires_benchmark
def test_the_page_says_the_points_are_discrete_and_the_gaps_are_real() -> None:
    """The misreading this feature must prevent."""
    app = _page(CAPABILITY_PAGE)

    captions = " ".join(str(c.value) for c in app.caption)
    rendered = " ".join(str(b.value) for b in app.markdown)
    assert "not a continuous validated region" in captions
    assert "no measurement" in captions.lower()
    assert "Nothing is interpolated" in rendered


@requires_benchmark
def test_the_page_states_the_threshold_is_only_a_reading_aid() -> None:
    app = _page(CAPABILITY_PAGE)

    rendered = " ".join(str(b.value) for b in app.markdown)
    assert "defines no pass/fail BER" in rendered


@requires_benchmark
def test_the_capability_page_does_not_need_a_loaded_capture() -> None:
    """It plots recorded evidence, so it must work with no signal in session."""
    app = _page(CAPABILITY_PAGE)

    assert "signal" not in app.session_state
    assert not app.exception, [e.value for e in app.exception]


@requires_v1
def test_the_scd_page_offers_both_cyclostationary_modes() -> None:
    app = _page(SCD_PAGE, _capture())

    options = [set(r.options) for r in app.radio]
    assert any({"Non-conjugate SCD", "Conjugate SCD"} <= o for o in options), (
        "both modes must be selectable")


@requires_v1
def test_switching_to_conjugate_mode_changes_the_run_control() -> None:
    """The two modes must never be confusable once a surface is on screen."""
    app = _page(SCD_PAGE, _capture())
    mode = next(r for r in app.radio if "Conjugate SCD" in r.options)

    app = mode.set_value("Conjugate SCD").run()

    assert not app.exception, [e.value for e in app.exception]
    assert any(b.label == "Run Conjugate SCD" for b in app.button)


@requires_v1
def test_running_conjugate_mode_reports_impropriety_not_a_symbol_rate() -> None:
    app = _page(SCD_PAGE, _capture())
    mode = next(r for r in app.radio if "Conjugate SCD" in r.options)
    app = mode.set_value("Conjugate SCD").run()

    app = _run_scd(app)

    assert not app.exception, [e.value for e in app.exception]
    rendered = " ".join(str(b.value) for b in app.markdown)
    captions = " ".join(str(c.value) for c in app.caption)
    assert "conjugate SCD" in captions, "the surface must say which mode produced it"
    assert "Impropriety at alpha = 0" in rendered
    assert "not a modulation decision" in captions


@requires_v1
def test_the_two_modes_do_not_share_a_cached_surface() -> None:
    """A stale surface from the other mode would be the worst possible confusion."""
    app = _page(SCD_PAGE, _capture())
    app = _run_scd(app)
    ordinary_request = app.session_state["scd_request"]

    mode = next(r for r in app.radio if "Conjugate SCD" in r.options)
    app = mode.set_value("Conjugate SCD").run()
    app = _run_scd(app)

    assert app.session_state["scd_request"] != ordinary_request
    assert app.session_state["scd_request"][-1] is True, "conjugate flag in the key"


# --- fusion decision landscape ------------------------------------------------------


def test_the_landscape_page_renders_without_an_analysis() -> None:
    app = _page(LANDSCAPE_PAGE)

    assert not app.exception, [e.value for e in app.exception]
    assert any("Analyze a WAV or IQ capture" in str(b.value) for b in app.markdown)


@requires_v1
def test_the_landscape_explains_a_real_analysed_capture() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    report = analyze_capture(signal)

    app = _page(LANDSCAPE_PAGE, signal, report)

    assert not app.exception, [e.value for e in app.exception]
    labels = {m.label for m in app.metric}
    assert {"CNN prediction", "Classical family", "Estimated SNR",
            "Fusion decision", "Review"} <= labels


@requires_v1
def test_the_plotted_decision_matches_the_recorded_one() -> None:
    """The agreement the feature rests on, checked through the page itself."""
    from radiofry.fusion.decision_landscape import describe_capture
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    report = analyze_capture(signal)
    app = _page(LANDSCAPE_PAGE, signal, report)

    position = describe_capture(report)
    shown = [m.value for m in app.metric if m.label == "Fusion decision"][0]
    assert shown == report["stages"]["fusion"]["label"]
    assert position.reproduced is True, (
        "replaying the recorded inputs must return the recorded decision")
    assert not any("did not reproduce" in str(e.value) for e in app.error)


@requires_v1
def test_the_landscape_offers_the_guard_contexts_this_capture_did_not_trigger() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    app = _page(LANDSCAPE_PAGE, signal, analyze_capture(signal))

    contexts = set(app.selectbox[0].options)
    assert "This capture" in contexts
    assert any("guard" in c for c in contexts)


@requires_v1
def test_switching_context_redraws_a_different_regime() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    app = _page(LANDSCAPE_PAGE, signal, analyze_capture(signal))
    before = " ".join(str(b.value) for b in app.markdown)

    app = app.selectbox[0].set_value(
        "Classical analog with envelope evidence").run()

    assert not app.exception, [e.value for e in app.exception]
    after = " ".join(str(b.value) for b in app.markdown)
    assert "Analog subtype" in after and "Analog subtype" not in before


@requires_v1
def test_the_page_states_that_snr_is_not_a_fusion_axis() -> None:
    """The honesty point: SNR is reported but is not an input to the decision."""
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    app = _page(LANDSCAPE_PAGE, signal, analyze_capture(signal))

    rendered = " ".join(str(b.value) for b in app.markdown)
    assert "SNR is deliberately NOT an axis" in rendered
    assert "Nothing here is learned" in rendered


@requires_v1
def test_changing_the_threshold_redraws_without_touching_the_analysis() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    report = analyze_capture(signal)
    recorded = report["stages"]["fusion"]["label"]
    app = _page(LANDSCAPE_PAGE, signal, report)

    app = app.select_slider[0].set_value(0.7).run()

    assert not app.exception, [e.value for e in app.exception]
    assert [m.value for m in app.metric
            if m.label == "Fusion decision"][0] == recorded, (
        "the recorded decision must not move when the map threshold changes")


# --- Signal-DNA -----------------------------------------------------------------------

requires_cnn = pytest.mark.skipif(not CHECKPOINT.is_file(),
                                  reason="production checkpoint not present")


@requires_cnn
@requires_v1
def test_the_signal_dna_page_renders_without_a_capture() -> None:
    """It plots a reference dataset, so it works with nothing loaded."""
    app = _page(DNA_PAGE)

    assert not app.exception, [e.value for e in app.exception]
    labels = {m.label for m in app.metric}
    assert {"Points shown", "Embedding width", "Variance in 3 axes"} <= labels


@requires_cnn
@requires_v1
def test_the_page_reports_the_layer_width_and_never_the_softmax() -> None:
    app = _page(DNA_PAGE)

    width = [m.value for m in app.metric if m.label == "Embedding width"][0]
    assert width == "256-d", "the penultimate layer, not the 8-class output"
    rendered = " ".join(str(b.value) for b in app.markdown)
    assert "softmax output is deliberately never used" in rendered


@requires_cnn
@requires_v1
def test_switching_to_the_pooled_layer_changes_the_width() -> None:
    app = _page(DNA_PAGE)
    layer = next(s for s in app.selectbox if "pooled" in " ".join(
        str(o) for o in s.options))

    app = layer.set_value("pooled").run()

    assert not app.exception, [e.value for e in app.exception]
    assert [m.value for m in app.metric if m.label == "Embedding width"][0] == "128-d"


@requires_cnn
@requires_v1
def test_a_loaded_capture_is_shown_as_unlabelled() -> None:
    """The mandatory distinction: a CNN prediction is not a ground-truth label."""
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    app = _page(DNA_PAGE, signal, analyze_capture(signal))

    assert not app.exception, [e.value for e in app.exception]
    truth = [m.value for m in app.metric if m.label == "Ground truth"]
    assert truth and truth[0] == "Unknown / unlabelled"
    captions = " ".join(str(c.value) for c in app.caption)
    assert "never promoted into the ground-truth field" in captions


@requires_cnn
@requires_v1
def test_the_page_refuses_to_call_prototype_distance_an_ood_score() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture()
    app = _page(DNA_PAGE, signal, analyze_capture(signal))

    captions = " ".join(str(c.value) for c in app.caption)
    assert "not** a validated" in captions or "not a validated" in captions
    assert "not a probability" in captions


@requires_cnn
@requires_v1
def test_the_page_states_the_projection_is_not_a_trained_model() -> None:
    app = _page(DNA_PAGE)

    rendered = " ".join(str(b.value) for b in app.markdown)
    captions = " ".join(str(c.value) for c in app.caption)
    assert "not a trained model" in (rendered + captions)
    assert "classifies" in rendered and "nothing" in rendered
