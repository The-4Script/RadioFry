# RadioFry CNN Architecture - Current State Audit

*This document outlines the current state of the RadioFry Convolutional Neural Network (CNN) architecture and its production configuration as discovered during the engineering audit.*

## 1. Current CNN Architecture

**Exact Model Class and File:**
The CNN is implemented as `ModulationCNN` in [`src/radiofry/models/modulation_cnn.py`](file:///c:/Users/Kaustubh%20Bhoir/Documents/RadioFry/RadioFry/src/radiofry/models/modulation_cnn.py). It is a 1D VT-CNN2-inspired architecture designed to classify fixed-length signal windows.

**Layer-by-Layer Architecture:**
The model consists of a feature extraction sequence followed by a fully connected classifier:

*   **Feature Extractor** (`nn.Sequential`):
    *   `Conv1d` (in_channels=4, out_channels=64, kernel_size=8, padding="same")
    *   `BatchNorm1d(64)`, `ReLU()`, `Dropout(0.3)`
    *   `Conv1d` (in_channels=64, out_channels=128, kernel_size=4, padding="same")
    *   `BatchNorm1d(128)`, `ReLU()`, `Dropout(0.3)`
    *   `Conv1d` (in_channels=128, out_channels=128, kernel_size=4, padding="same")
    *   `BatchNorm1d(128)`, `ReLU()`
    *   `AdaptiveAvgPool1d(1)` (collapses the temporal dimension down to a single value per feature map)
*   **Classifier Block** (`nn.Sequential`):
    *   `Flatten()`
    *   `Linear` (in_features=128, out_features=256), `ReLU()`, `Dropout(0.5)`
    *   `Linear` (in_features=256, out_features=8) (Outputs 8 classes for the digital-only checkpoint)

**Input Shape and Channels:**
The input tensor shape is `(batch, 4, 128)`. The 4 input channels (`features="iqap"`) are:
1.  In-phase (I)
2.  Quadrature (Q)
3.  Amplitude ($|I + jQ|$)
4.  Phase Difference (differential angle between adjacent complex samples)

**Current Preprocessing and Features Fed to CNN:**
1.  **Format-independent Stage** (`dsp/preprocessing.py`): The raw capture is optionally resampled to a target sample rate, mean-centered (DC offset removed), and normalized by its global root-mean-square (RMS) power.
2.  **Window Extraction** (`models/modulation_inference.py`): 128-sample windows are extracted from the capture. If the capture is shorter than 128 samples, it is upsampled via linear interpolation. Each extracted window is RMS-normalized independently.
3.  **Feature Engineering** (`models/signal_features.py`): The I/Q channels are expanded to include Amplitude and Phase Difference channels. Finally, per-channel power normalization is applied across the 128 samples.

**Current Inference Window Length:**
The inference frame length is fixed at **128 samples** (`FRAME_LENGTH = 128`).

**Whether Variable Input Length is Supported:**
At the neural network level, variable length is **not** supported; it requires exactly 128 samples.
At the pipeline level (`modulation_inference.py`), variable length captures **are** supported. The pipeline selects 4 evenly spaced non-overlapping 128-sample frames from the capture (`DEFAULT_INFERENCE_WINDOWS = 4`), runs the model on all frames, and averages the output softmax probabilities to make the final prediction.

**Any Normalization or Resampling Performed Before CNN:**
*   Polyphase resampling to target sample rate (if requested).
*   Global DC removal (mean centering).
*   Global RMS power normalization.
*   Window-level RMS power normalization.
*   Channel-wise power normalization after the feature expansion (I, Q, Amplitude, Phase Difference).

## 2. Production Model

**Exact Weight File Name:**
The production model artifact is [`models_saved/modulation_cnn_v3_spsaug.pt`](file:///c:/Users/Kaustubh%20Bhoir/Documents/RadioFry/RadioFry/models_saved/modulation_cnn_v3_spsaug.pt). This is the default in [`src/radiofry/pipeline.py`](file:///c:/Users/Kaustubh%20Bhoir/Documents/RadioFry/RadioFry/src/radiofry/pipeline.py).
It classifies **8 digital modulation types**: 8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK. Analog types are routed through a classical DSP gate.

**Training Parameters:**
(Sourced from `src/radiofry/training/train_v2_synthetic.py` and the artifact metrics file)
*   **Optimizer**: Adam with learning rate `1e-3` (paired with `ReduceLROnPlateau` scheduler: patience 2, factor 0.5).
*   **Loss Function**: CrossEntropyLoss.
*   **Batch Size**: 256.
*   **Epochs**: Best validation loss was achieved at Epoch 35 (max epochs 60, early stopping patience 6).
*   **Dataset Augmentation**: Sweeps over SNR [20.0, 15.0, 10.0, 5.0, 0.0] dB and Samples-Per-Symbol (SPS) over [4, 8, 16, 32] to increase robustness against varying symbol rates.

**Evaluation Metrics:**
*   **Validation Loss**: The `modulation_cnn_v3_spsaug_metrics.json` file records a best validation loss of **0.2469** at epoch 35.
*   **Fused Accuracy**: As noted in pipeline documentation (Bank Entry 040), this SPS-augmented checkpoint achieved **99.5% fused accuracy** on the held-out seed test set, resolving an earlier degradation issue seen at edge oversampling factors.
