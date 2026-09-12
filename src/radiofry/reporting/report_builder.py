"""JSON-safe aggregation of pipeline results."""

from datetime import datetime, timezone
import json
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {name: _json_safe(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_report(*, source: dict[str, Any], stages: dict[str, Any]) -> dict[str, Any]:
    """Build a stable report envelope while preserving stage-level results."""

    fusion = stages.get("fusion", {})
    bitstream = stages.get("bitstream_analysis", {})
    review_required = (
        bool(fusion.get("review_recommended", True))
        if isinstance(fusion, dict)
        else bool(getattr(fusion, "review_recommended", True))
    )
    verification = bitstream.get("verification", {}) if isinstance(bitstream, dict) else {}
    fusion_dict = fusion if isinstance(fusion, dict) else {}
    parameters = stages.get("parameters", {})
    parameters_dict = parameters if isinstance(parameters, dict) else {}
    prediction = stages.get("cnn_modulation", {})
    prediction_dict = prediction if isinstance(prediction, dict) else {}
    summary = {
        "decision": fusion_dict.get("label", "Unclassified"),
        "trust_score": float(fusion_dict.get("trust_score", 0.0) or 0.0),
        "model_confidence": float(prediction_dict.get("confidence", 0.0) or 0.0),
        "review_recommended": review_required,
        "review_level": str(fusion_dict.get("review_level", "high")),
        "key_metrics": {
            "carrier_frequency_hz": parameters_dict.get("carrier_frequency_hz"),
            "occupied_bandwidth_hz": parameters_dict.get("occupied_bandwidth_hz"),
            "snr_db": parameters_dict.get("snr_db"),
            "symbol_rate_hz": parameters_dict.get("symbol_rate_hz"),
        },
    }
    return {
        "schema_version": "0.3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _json_safe(source),
        "summary": _json_safe(summary),
        "interpretation": {
            "status": "research_prototype",
            "human_review_required": review_required,
            "protocol_verified": bool(
                verification.get("protocol_verified", False)
            ),
            "confidence_is_not_field_accuracy": True,
            "model_domain": "synthetic_or_unvalidated",
        },
        "stages": _json_safe(stages),
    }


def build_pdf_report(report: dict[str, Any], output_path: str, signal: Any | None = None) -> str:
    """Create a self-contained formal PDF from the same JSON-safe report."""

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.signal import stft

    stages = report.get("stages", {})
    parameters = stages.get("parameters", {})
    fusion = stages.get("fusion", {})
    prediction = stages.get("cnn_modulation", {})
    correlation = stages.get("bitstream_analysis", {}).get("correlation", {})
    with PdfPages(output_path) as pdf:
        summary = report.get("summary", {})
        figure = plt.figure(figsize=(11.7, 8.3), facecolor="#0b1117")
        figure.text(0.08, 0.78, "RADIOFRY", color="#67e0c8", fontsize=11, weight="bold")
        figure.text(0.08, 0.68, "Signal analysis\nreport", color="white", fontsize=31, weight="bold", linespacing=1.15)
        figure.text(0.08, 0.48, "Evidence-first interpretation of a captured RF signal", color="#c9d8d2", fontsize=13)
        figure.text(0.08, 0.22, f"Generated {report.get('generated_at', 'unknown')}\n"
                    f"Source: {report.get('source', {}).get('format', 'unknown').upper()}  |  "
                    f"{report.get('source', {}).get('samples', 'unknown')} samples",
                    color="#9eafb2", fontsize=10, linespacing=1.7)
        figure.text(0.08, 0.08, "Confidence is evidence, not protocol verification. Human review remains part of the result.",
                    color="#f3bd73", fontsize=9)
        pdf.savefig(figure, facecolor=figure.get_facecolor(), bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(11.7, 8.3))
        figure.suptitle("Executive result", x=0.07, y=0.94, ha="left", fontsize=22, weight="bold")
        axis.axis("off")
        result_rows = [
            ("Decision", summary.get("decision", fusion.get("label", "Unclassified"))),
            ("Fused trust score", f"{summary.get('trust_score', fusion.get('trust_score', 0)):.1%}"),
            ("CNN model confidence", f"{summary.get('model_confidence', prediction.get('confidence', 0)):.1%}"),
            ("Review status", "REVIEW REQUIRED" if summary.get("review_recommended", True) else "Available for inspection"),
            ("Protocol verification", "Not established" if not report.get("interpretation", {}).get("protocol_verified") else "Verified"),
        ]
        table = axis.table(cellText=result_rows, colLabels=["Measure", "Result"], loc="upper left",
                           cellLoc="left", colWidths=[0.38, 0.52], bbox=[0.04, 0.43, 0.92, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d5dfdc")
            if row == 0:
                cell.set_facecolor("#173b3d")
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f2f6f4" if row % 2 else "#e5eeeb")
        axis.text(0.04, 0.27, "Interpretation", fontsize=14, weight="bold", transform=axis.transAxes)
        axis.text(0.04, 0.22, "The result packages independent measurements, model evidence, and explicit uncertainty. "
                  "Use the detailed pages to inspect waveform evidence and decoding limitations.",
                  fontsize=10, transform=axis.transAxes, wrap=True, va="top")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        if signal is not None and getattr(signal, "iq", np.array([])).size:
            samples = signal.iq[: min(signal.iq.size, 12000)]
            figure, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), constrained_layout=True)
            axes[0].plot(samples.real, label="I", linewidth=0.7)
            axes[0].plot(samples.imag, label="Q", linewidth=0.7)
            axes[0].set_title("Time-domain I/Q waveform")
            axes[0].set_xlabel("Sample index")
            axes[0].legend()
            axes[1].specgram(samples, NFFT=min(256, max(32, samples.size // 8)), Fs=getattr(signal, "sample_rate", None) or 1.0, noverlap=32, cmap="viridis")
            axes[1].set_title("Time-frequency spectrogram")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Frequency")
            pdf.savefig(figure)
            plt.close(figure)

            frequencies, times, spectrum = stft(samples, fs=getattr(signal, "sample_rate", None) or 1.0, nperseg=min(256, max(32, samples.size // 8)), noverlap=32, return_onesided=False)
            order = np.argsort(frequencies)
            power_db = 10 * np.log10(np.maximum(np.abs(spectrum[order]) ** 2, 1e-12))
            figure = plt.figure(figsize=(11.7, 8.3))
            axis = figure.add_subplot(111, projection="3d")
            mesh_times, mesh_frequencies = np.meshgrid(times, frequencies[order])
            axis.plot_surface(mesh_times[:: max(1, mesh_times.shape[0] // 40), :: max(1, mesh_times.shape[1] // 60)], mesh_frequencies[:: max(1, mesh_frequencies.shape[0] // 40), :: max(1, mesh_frequencies.shape[1] // 60)], power_db[:: max(1, power_db.shape[0] // 40), :: max(1, power_db.shape[1] // 60)], cmap="viridis", linewidth=0, antialiased=True)
            axis.set_title("3D spectral power surface")
            axis.set_xlabel("Time")
            axis.set_ylabel("Frequency")
            axis.set_zlabel("Power (dB)")
            pdf.savefig(figure)
            plt.close(figure)

            symbols = stages.get("demodulation", {}).get("result", {}).get("symbols", [])
            if symbols:
                points = np.array([[item.get("real", 0.0), item.get("imag", 0.0)] if isinstance(item, dict) else [float(item), 0.0] for item in symbols], dtype=float)
                figure = plt.figure(figsize=(11.7, 8.3))
                axis = figure.add_subplot(111, projection="3d")
                axis.scatter(points[:, 0], points[:, 1], np.arange(len(points)), c=np.arange(len(points)), cmap="turbo", s=10)
                axis.set_title("3D recovered constellation trajectory")
                axis.set_xlabel("In-phase")
                axis.set_ylabel("Quadrature")
                axis.set_zlabel("Symbol index")
                pdf.savefig(figure)
                plt.close(figure)

        figure = plt.figure(figsize=(11.7, 8.3))
        figure.text(0.07, 0.9, "RadioFry Signal Analysis Report", fontsize=24, weight="bold")
        figure.text(0.07, 0.85, "Formal evidence summary generated from the analyzed capture", fontsize=12)
        lines = [
            f"Generated: {report.get('generated_at', 'unknown')}",
            f"Source: {report.get('source', {}).get('format', 'unknown')} | samples: {report.get('source', {}).get('samples', 'unknown')}",
            f"Measured sample rate: {report.get('source', {}).get('sample_rate') if report.get('source', {}).get('sample_rate') is not None else 'unknown'}",
            f"Modulation decision: {fusion.get('label', 'Unclassified')}",
            f"CNN confidence: {prediction.get('confidence', 0):.1%}",
            f"Fused trust score: {fusion.get('trust_score', 0):.1%}",
            f"Carrier frequency: {parameters.get('carrier_frequency_hz') if parameters.get('carrier_frequency_hz') is not None else 'unknown'} Hz",
            f"Occupied bandwidth: {parameters.get('occupied_bandwidth_hz') if parameters.get('occupied_bandwidth_hz') is not None else 'unknown'} Hz",
            f"Estimated SNR: {parameters.get('snr_db') if parameters.get('snr_db') is not None else 'unknown'} dB",
            f"Estimated symbol rate: {parameters.get('symbol_rate_hz') if parameters.get('symbol_rate_hz') is not None else 'unknown'} Hz",
            f"Symbol-rate confidence: {parameters.get('symbol_rate_confidence', 0):.1%}",
            f"Sync pattern: {correlation.get('sync_pattern') or 'not detected'}",
        ]
        figure.text(0.09, 0.76, "\n".join(lines), family="monospace", fontsize=12, va="top", linespacing=1.7)
        figure.text(0.09, 0.18, "Score definitions", fontsize=14, weight="bold")
        figure.text(0.09, 0.13, "CNN confidence is the model softmax probability. Fused trust score combines CNN confidence with independent classical family agreement. These values are intentionally different.", fontsize=10, wrap=True)
        figure.text(0.09, 0.08, "Scope: the production CNN is a synthetic-domain digital classifier; real-world accuracy and out-of-distribution correctness are not established. Human review remains required.", fontsize=9, wrap=True)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(11.7, 8.3))
        top_k = prediction.get("top_k", [])
        if top_k:
            axis.bar([label for label, _ in top_k], [value for _, value in top_k], color="#28b8a6")
            axis.set_ylim(0, 1)
            axis.set_ylabel("CNN confidence")
            axis.set_title("Modulation hypothesis confidence")
        else:
            axis.text(0.5, 0.5, "CNN prediction unavailable", ha="center", va="center")
            axis.set_axis_off()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure = plt.figure(figsize=(11.7, 8.3))
        figure.text(0.07, 0.9, "Decoded Evidence", fontsize=20, weight="bold")
        analysis = stages.get("bitstream_analysis", {})
        decoded = analysis.get("fec_decoding", {})
        evidence = [
            f"Interleaver prediction: {analysis.get('interleaver', {}).get('label', 'unavailable')}",
            f"Selected interleaver: {analysis.get('selected_interleaver', 'unknown')}",
            f"FEC prediction: {analysis.get('fec', {}).get('label', 'unavailable')}",
            f"Selected FEC: {analysis.get('selected_fec', 'unknown')}",
            f"FEC success: {decoded.get('success', False)}",
            f"Recovered bits: {len(decoded.get('bits', []))}",
            f"Header bits: {len(correlation.get('header_bits', []))}",
            f"Payload bits: {len(correlation.get('payload_bits', []))}",
        ]
        figure.text(0.09, 0.78, "\n".join(evidence), family="monospace", fontsize=13, va="top", linespacing=1.8)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return output_path


def report_json(report: dict[str, Any]) -> str:
    """Serialize a report for Streamlit download or filesystem export."""

    return json.dumps(_json_safe(report), indent=2, sort_keys=True)
