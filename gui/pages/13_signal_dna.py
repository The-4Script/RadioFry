"""Signal-DNA: the production CNN's internal representation, made visible.

"Signal-DNA" is the investigator-facing name. What is plotted is a PCA projection of an
internal activation of the current production checkpoint - a view of the space the
classifier works in, not a classifier and not a trained model. The extraction and the
projection live in `radiofry.models.embedding` / `embedding_dataset`; this page renders.
"""

import html
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gui.theme import (
    render_empty_state, render_method_note, render_page_shell, render_stage_header)
from radiofry.models.embedding import (
    DEFAULT_WINDOWS,
    EMBEDDING_LAYERS,
    PENULTIMATE,
    SOURCE_REAL,
    UNLABELLED,
    aggregate_windows,
    embed_signal,
    load_embedding_model,
)
from radiofry.models.embedding_dataset import (
    build_set,
    centroid_distances,
    class_separation,
    collect_benchmark_grid,
    collect_v1_captures,
    filter_set,
    fit_pca,
    nearest_class_centroids,
)
from radiofry.pipeline import DEFAULT_MODULATION_MODEL

CLASS_COLOURS = {
    "8PSK": "#f2c14e", "BPSK": "#3fb950", "CPFSK": "#4aa8d8", "GFSK": "#7fe7c4",
    "PAM4": "#d1495b", "QAM16": "#b07fe7", "QAM64": "#e77fb0", "QPSK": "#9ccc5a",
    UNLABELLED: "#8b949e",
}

render_page_shell(13)
signal = st.session_state.get("signal")
report = st.session_state.get("report")
render_stage_header(
    "13",
    "Signal-DNA",
    "Where captures land inside the production CNN's learned representation - which "
    "classes separate, which overlap, and where this capture sits among them.",
    "Ready", "ready")


@st.cache_resource(show_spinner=False)
def _model(checkpoint: str):
    """Loaded once per process. The checkpoint is never re-read per capture."""
    return load_embedding_model(checkpoint)


@st.cache_data(show_spinner=False, max_entries=6)
def _dataset(dataset: str, layer: str, windows: int, checkpoint: str, _loaded):
    """Cached on dataset, layer, window count and checkpoint identity.

    `_loaded` is underscored so Streamlit does not try to hash the torch module; the
    checkpoint's own SHA is in the key instead, which is the thing that actually matters.
    """
    started = time.perf_counter()
    if dataset == "Frozen V1 captures":
        collected = collect_v1_captures(_loaded, layer=layer, windows=windows)
    else:
        collected = collect_benchmark_grid(_loaded, layer=layer, windows=windows)
    return collected, time.perf_counter() - started


loaded, reason = _model(DEFAULT_MODULATION_MODEL)
if loaded is None:
    render_empty_state(
        "The production CNN checkpoint could not be loaded.",
        "Signal-DNA reads an internal activation of the production model, so without the "
        "checkpoint there is no representation to show.")
    st.warning(reason)
    st.stop()

# ---------------------------------------------------------------- configuration
st.markdown("<div class='evidence-label'>Embedding space</div>", unsafe_allow_html=True)
controls = st.columns(4)
with controls[0]:
    dataset_name = st.selectbox(
        "Reference dataset",
        ["Benchmark grid (regenerated)", "Frozen V1 captures"],
        help="The grid spans samples-per-symbol 4/8/16/32 x SNR 20..0 dB x 3 seeds, "
             "regenerated from the Entry 039 seeds. Frozen V1 is the 40 on-disk captures, "
             "all at samples-per-symbol 8.")
with controls[1]:
    layer = st.selectbox(
        "CNN layer", list(EMBEDDING_LAYERS), index=list(EMBEDDING_LAYERS).index(PENULTIMATE),
        format_func=lambda name: ("penultimate (256-d)" if name == PENULTIMATE
                                  else "pooled conv (128-d)"),
        help="penultimate is the vector the final linear layer reads, so overlap there "
             "directly explains confusion. pooled is the convolutional summary before "
             "any classification-specific projection. The softmax output is never used.")
with controls[2]:
    granularity = st.radio(
        "Granularity", ["Per capture", "Per window"], horizontal=True,
        help="Production classifies 4 windows and averages their softmax. Per-window "
             "shows each one; per-capture shows the mean of a capture's window "
             "embeddings, which is a different reduction and is labelled as such.")
with controls[3]:
    colour_by = st.selectbox(
        "Colour by",
        ["Ground truth", "CNN prediction", "Correct / incorrect", "Samples per symbol",
         "SNR", "Source"])

with st.spinner("Extracting CNN embeddings..."):
    collected, build_seconds = _dataset(dataset_name, layer, DEFAULT_WINDOWS,
                                        loaded.checkpoint_sha256, loaded)

if not collected.ok:
    st.error("No embeddings could be produced for this dataset.")
    for warning in collected.warnings:
        st.warning(warning)
    st.stop()

points = collected
if granularity == "Per capture":
    points = build_set(aggregate_windows(list(collected.records)), loaded,
                       collected.layer, collected.windows, collected.warnings)

# ---------------------------------------------------------------- filters
labels_present = sorted({r.display_true_label for r in points.records})
sps_present = sorted({r.samples_per_symbol for r in points.records
                      if r.samples_per_symbol is not None})
snr_present = sorted({r.snr_db for r in points.records if r.snr_db is not None})

filters = st.columns(4)
with filters[0]:
    chosen_labels = st.multiselect("Ground-truth classes", labels_present,
                                   default=labels_present)
with filters[1]:
    chosen_sps = st.multiselect("Samples per symbol", sps_present, default=sps_present)
with filters[2]:
    chosen_snr = st.multiselect("SNR (dB)", snr_present, default=snr_present)
with filters[3]:
    correctness = st.radio("Predictions", ["All", "Correct only", "Incorrect only"],
                           horizontal=True)

criteria = {}
if set(chosen_labels) != set(labels_present):
    criteria["true_label"] = {None if v == UNLABELLED else v for v in chosen_labels}
if chosen_sps and set(chosen_sps) != set(sps_present):
    criteria["samples_per_symbol"] = set(chosen_sps)
if chosen_snr and set(chosen_snr) != set(snr_present):
    criteria["snr_db"] = set(chosen_snr)
if correctness == "Correct only":
    criteria["correct"] = True
elif correctness == "Incorrect only":
    criteria["correct"] = False

shown = filter_set(points, **criteria) if criteria else points
if not shown.ok:
    st.warning("No embeddings match the selected filters.")
    st.stop()

# PCA is fitted to what is displayed, so the axes describe this view and nothing else.
projection = fit_pca(shown.vectors, 3, fitted_on=f"{len(shown)} displayed points")
if projection is None:
    st.warning("At least two points are needed to fit a projection.")
    st.stop()

# ---------------------------------------------------------------- summary
summary = st.columns(5)
summary[0].metric("Points shown", f"{len(shown):,}",
                  delta=f"of {len(points):,}", delta_color="off")
summary[1].metric("Embedding width", f"{shown.width}-d")
summary[2].metric("Variance in 3 axes", f"{projection.shown_variance:.1%}")
summary[3].metric("Components for 90%", f"{projection.components_for(0.90)}")
summary[4].metric("Extraction time", f"{build_seconds:.1f} s")
st.caption(
    f"Checkpoint `{Path(loaded.checkpoint_path).name}` (sha "
    f"{loaded.checkpoint_sha256[:12]}) · layer **{layer}** ({shown.width}-d) · "
    f"{DEFAULT_WINDOWS} windows per capture · PCA fitted on "
    f"{projection.fitted_on} · per-component variance "
    f"{np.round(projection.explained_variance_ratio[:3], 3).tolist()}. "
    "The projection is a linear view of the embedding, not a trained model.")
for warning in shown.warnings[:3]:
    st.warning(warning)


def _group_key(record):
    if colour_by == "Ground truth":
        return record.display_true_label
    if colour_by == "CNN prediction":
        return record.predicted_label
    if colour_by == "Correct / incorrect":
        return {True: "Correct", False: "Incorrect", None: UNLABELLED}[record.correct]
    if colour_by == "Samples per symbol":
        return f"sps {record.samples_per_symbol}" if record.samples_per_symbol else "sps ?"
    if colour_by == "SNR":
        return f"{record.snr_db:g} dB" if record.snr_db is not None else "SNR ?"
    return record.source


groups: dict[str, list[int]] = {}
for index, record in enumerate(shown.records):
    groups.setdefault(_group_key(record), []).append(index)

figure = go.Figure()
for name in sorted(groups):
    indices = groups[name]
    coordinates = projection.coordinates[indices]
    hover = []
    for i in indices:
        record = shown.records[i]
        hover.append(
            f"<b>{html.escape(record.capture_id)}</b><br>"
            f"ground truth: {html.escape(record.display_true_label)}<br>"
            f"CNN predicts: {html.escape(record.predicted_label)} "
            f"({record.confidence:.3f})<br>"
            f"sps {record.samples_per_symbol if record.samples_per_symbol else 'n/a'} · "
            f"SNR {f'{record.snr_db:g} dB' if record.snr_db is not None else 'n/a'}<br>"
            f"source: {record.source} · window "
            f"{'mean' if record.window_index < 0 else record.window_index}")
    figure.add_scatter3d(
        x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
        mode="markers", name=str(name), customdata=hover,
        marker={"size": 3.2, "opacity": 0.8,
                "color": CLASS_COLOURS.get(str(name))},
        hovertemplate="%{customdata}<extra></extra>")

# ---------------------------------------------------------------- this capture
capture_records = []
if signal is not None:
    predicted = ((report or {}).get("stages", {}).get("cnn_modulation") or {}).get("label")
    capture_records = embed_signal(
        loaded, signal, capture_id="This capture", layer=layer,
        windows=DEFAULT_WINDOWS, source=SOURCE_REAL, true_label=None)
    if granularity == "Per capture" and capture_records:
        capture_records = aggregate_windows(capture_records)
    if capture_records:
        # Projected with the SAME fitted transform, so it is comparable with the cloud.
        placed = projection.transform(
            np.stack([r.vector for r in capture_records]).astype(np.float64))
        figure.add_scatter3d(
            x=placed[:, 0], y=placed[:, 1], z=placed[:, 2], mode="markers",
            name="This capture (unlabelled)",
            marker={"size": 9, "color": "#ffffff", "symbol": "x",
                    "line": {"width": 2}},
            customdata=[f"This capture<br>ground truth: {UNLABELLED}<br>"
                        f"CNN predicts: {r.predicted_label} ({r.confidence:.3f})"
                        for r in capture_records],
            hovertemplate="%{customdata}<extra></extra>")

figure.update_layout(
    height=620, margin={"l": 0, "r": 0, "t": 10, "b": 0},
    legend={"itemsizing": "constant"},
    scene={"xaxis": {"title": {"text": "PC1"}},
           "yaxis": {"title": {"text": "PC2"}},
           "zaxis": {"title": {"text": "PC3"}},
           "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 0.9}}})
st.plotly_chart(figure, use_container_width=True, key="dna_scatter")
st.caption(
    "Hover any point for its capture identifier, ground truth, CNN prediction and "
    "confidence. Axes are principal components of the displayed embeddings: they have no "
    "physical units, and distances are distances in a 3-D shadow of a "
    f"{shown.width}-D space.")

# ---------------------------------------------------------------- this capture panel
if capture_records:
    st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
                "This capture in the embedding space</div>", unsafe_allow_html=True)
    centroids = nearest_class_centroids(points)
    distances = centroid_distances(points, capture_records[0].vector, centroids)
    detail = st.columns(4)
    detail[0].metric("Ground truth", UNLABELLED)
    detail[1].metric("CNN prediction", capture_records[0].predicted_label,
                     delta=f"{capture_records[0].confidence:.3f}", delta_color="off")
    detail[2].metric("Nearest class centroid",
                     distances[0][0] if distances else "n/a",
                     delta=f"{distances[0][1]:.2f}" if distances else None,
                     delta_color="off")
    detail[3].metric("Source", "real / uploaded")
    if distances:
        st.table({
            "Class centroid": [name for name, _ in distances],
            "Distance in embedding space": [f"{value:.2f}" for _, value in distances],
        })
    st.caption(
        "The loaded capture carries **no ground-truth label** and is shown as unlabelled: "
        "the CNN's own prediction is never promoted into the ground-truth field. "
        "Distances are Euclidean prototype distances to each class centroid in the raw "
        f"{shown.width}-D embedding - a precisely defined quantity, **not** a validated "
        "out-of-distribution score and not a probability. A large distance to every "
        "centroid is a prompt to look more closely, not a verdict.")

# ---------------------------------------------------------------- class geometry
st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "Class geometry (raw embedding space)</div>", unsafe_allow_html=True)
pairs = [("QAM16", "QAM64"), ("CPFSK", "GFSK"), ("BPSK", "QPSK"), ("BPSK", "8PSK"),
         ("QAM16", "PAM4"), ("QPSK", "8PSK")]
rows = []
for left, right in pairs:
    measured = class_separation(points, left, right)
    if measured is not None:
        rows.append((f"{left} vs {right}", measured))
if rows:
    st.table({
        "Pair": [name for name, _ in rows],
        "Centroid distance": [f"{m.centroid_distance:.2f}" for _, m in rows],
        "Separation ratio": [f"{m.separation_ratio:.2f}" for _, m in rows],
        "Centroid overlap": [f"{m.overlap:.1%}" for _, m in rows],
        "Captures": [f"{m.count_left + m.count_right}" for _, m in rows],
    })
    st.caption(
        "Measured in the raw embedding space, so these numbers do not depend on how many "
        "principal components the plot happens to show. Separation ratio is the centroid "
        "distance divided by the classes' own mean spread: below about 1 the two classes "
        "are closer to each other than they are internally consistent. Overlap is the "
        "fraction of captures whose nearest class centroid is the *other* class.")

render_method_note(
    "What this is, and what it is not",
    "Signal-DNA is an investigator-facing name for a PCA projection of an internal "
    "activation of the current production checkpoint. The default layer is the 256-d "
    "output of the classifier's first linear layer and its ReLU - the vector the final "
    "linear layer reads, which is why overlap there explains confusion directly. The "
    "8-class softmax output is deliberately never used as an embedding: it is eight "
    "numbers on a simplex and describes the decision rather than the representation. No "
    "aggregation is invented - the model already pools over time with "
    "AdaptiveAvgPool1d(1) - and the same window framing and iqap feature channels that "
    "production inference uses are called here, not a copy of them. Extraction is "
    "deterministic (eval mode, inference mode, seeded generation) and the PCA sign "
    "convention is fixed, so repeated runs are identical. PCA is fitted to the points "
    "currently displayed: it is a linear view, not a trained model, it classifies "
    "nothing, and geometric distance in the 3-D projection is not a physical similarity "
    "between signals. Captures without ground truth are shown as unlabelled and never "
    "take the CNN's prediction as a label. Nothing on this page alters the CNN, the "
    "checkpoint, fusion, or any reported result.")
