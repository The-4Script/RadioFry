# SIH Problem Statement 26147 — System Architecture & Implementation Plan
## Automated Model for Analysis of .IQ and .WAV Files with Signal Parameter Extraction

**Organization:** National Technical Research Organisation (NTRO)
**Category:** Software | **Theme:** Space Technology
**Project Root (dev machine):** `~/sih26147` (WSL2 Ubuntu, Python 3.14 venv `sih_env`, GNU Radio 3.10.12)
**Sprint Duration:** 10 days to first submission/demo, iterative improvement allowed after submission until judging.
**Hardware constraint:** No SDR/RF hardware available. All development, training and validation is done on synthetic and publicly available datasets. This is explicitly stated as a design constraint, not a gap — the system must be provably hardware-agnostic and validated purely in software.

---

## 0. How to Use This Document (Instructions for any AI agent reading this)

This file is the single source of truth for the project. It is designed so that **different sections can be handed to different AI agents/team members independently** — each module section (Section 6) is self-contained with Purpose / Input / Output / Approach / Libraries / Files-to-create / Owner-track, so an agent working on one module does not need to read the whole document to start coding, though reading Sections 1–5 first is strongly recommended for context.

Do not deviate from the folder structure (Section 11) or the naming conventions given — multiple people/agents are working in parallel and consistency is what makes integration possible on Day 10.

If a library mentioned here fails to install/build (this has already happened once with `pyldpc` on Python 3.14), do not block on it — mark that specific sub-feature as degraded/skipped and continue; see Section 14 (Risk Register) for the exact fallback for each such case.

---

## 1. Problem Statement Summary

Off-air RF signals (HF/VHF/UHF bands) are recorded as raw `.wav` or `.IQ` files by different sensors at different locations, with inconsistent/incomplete metadata. Manual analysis to extract fine-grained parameters (sampling rate, modulation type, FEC, interleaving) is slow and inconsistent. The required solution is a **GUI-based system** that, given a `.wav` or `.IQ` file, must:

1. Identify signal parameters: sampling frequency, modulation type, FEC scheme, interleaving scheme.
2. Demodulate signals (FSK, QAM, PSK).
3. De-interleave (Block, Convolutional, Diagonal, Pseudo-Random).
4. Decode FEC (short-constraint convolutional codes w/ Viterbi, RS block codes, Concatenated codes, LDPC).
5. Perform bitstream correlation to identify header vs. payload.

The PS explicitly notes that files from different sensors have inconsistent parameters — meaning **the system cannot assume any parameter is known a priori** and must be able to estimate/classify parameters blind wherever possible. This single sentence in the PS is the design anchor for most of the "uniqueness" strategy in Section 9.

---

## 2. Project Identity

- **Suggested codename:** RF-SENTINEL (team may rename; used only for internal file naming consistency where needed, not a hard requirement).
- **Primary language:** Python 3.14 (existing venv). DSP heavy-lifting via NumPy/SciPy/GNU Radio. ML via PyTorch (classifier) + scikit-learn (lightweight classical-feature classifiers).
- **GUI:** Streamlit (fastest path to a working, presentable GUI within 10 days — see Section 12 for why this beats PyQt/Electron under this timeline).
- **No real hardware.** All validation is against RadioML (public benchmark dataset) + self-generated synthetic data via scripted GNU Radio flowgraphs.

---

## 3. Development Environment (already provisioned — do not re-derive)

- OS: Windows 11 host, WSL2 (Ubuntu), WSLg enabled for GUI passthrough.
- GNU Radio 3.10.12 installed via `apt` (system-level, NOT inside the venv).
- Python virtual environment: `~/sih26147/sih_env` (Python 3.14).
- Installed in venv: `numpy, scipy, torch, torchvision, streamlit, matplotlib, plotly, scikit-learn, h5py, scikit-commpy (imports as `commpy`), reedsolo`.
- `pyldpc` failed to build on Python 3.14 (missing numpy at build-isolation time). Treat LDPC support as **best-effort/Tier-2**, not core deliverable. See Section 14.
- Dataset already downloaded: **RML2016.10a** (DeepSig), located at `~/sih26147/data/RML2016.10a_dict.pkl`. Verified structure: dict keyed by `(modulation:str, SNR:int)` → numpy array of shape `(1000, 2, 128)` (1000 frames, 2 channels = I/Q, 128 time samples per frame). 11 modulations × 20 SNR values (-20 dB to +18 dB, step 2) = 220 keys, 220,000 total labeled frames.
- Modulations present in RML2016.10a: `8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM`.

Any agent generating code must target this exact environment (paths, package names) unless explicitly told the environment has changed.

---

## 4. High-Level System Architecture

```
                         ┌───────────────────────────┐
                         │   GUI (Streamlit) Layer    │  <- Section 12
                         └─────────────▲─────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │   Orchestration Pipeline   │
                         └─────────────▲─────────────┘
                                       │
   ┌──────────────┐   ┌───────────────┴────────────────┐   ┌──────────────────┐
   │ File Ingestion│──▶│ Unified Signal Container (USC) │──▶│ Parameter         │
   │ (.wav / .IQ)  │   │ (canonical IQ array + metadata)│   │ Estimation (DSP)  │
   └──────────────┘   └───────────────┬────────────────┘   └────────┬─────────┘
                                       │                             │
                     ┌─────────────────┴───────────────┐            │
                     ▼                                  ▼            │
        ┌────────────────────────┐        ┌─────────────────────────┴──┐
        │ ML Modulation Classifier│        │ Cyclostationary / Classical │
        │ (CNN, Section 7.1)      │        │ Modulation-Family Analysis  │
        └────────────┬────────────┘        └────────────┬─────────────┘
                      └───────────────┬───────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │  Confidence Fusion Engine  │   <- Section 6.6
                         │ (agreement score, "Unknown"│
                         │  rejection if low conf.)   │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │     Demodulation Engine    │
                         └─────────────┬─────────────┘
                                       ▼
                ┌──────────────────────────────────────────┐
                │ Interleaver-Type Classifier (Sec 7.2)     │
                │        ──▶ De-interleaving Engine         │
                └───────────────────────┬────────────────────┘
                                       ▼
                ┌──────────────────────────────────────────┐
                │ FEC-Scheme Classifier (Sec 7.3)           │
                │        ──▶ FEC Decoding Engine             │
                └───────────────────────┬────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │ Bitstream Correlation Engine│
                         │ (header/payload detection)  │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   Report / Export Layer    │
                         └───────────────────────────┘
```

Offline/parallel to the above runtime pipeline is the **Synthetic Data Generation Pipeline** (Section 8.2) which feeds the training of all three ML models (Section 7) and is itself a GNU Radio + Python component, not part of the runtime inference path.

---

## 5. End-to-End Data Flow (step-by-step)

1. User uploads a `.wav` or `.IQ` file via GUI.
2. **Ingestion module** detects format by extension + header sniffing, parses into raw samples.
3. Samples converted into a **Unified Signal Container (USC)**: a Python object holding `iq: np.ndarray[complex64]`, `sample_rate: Optional[float]`, `source_format: str`, `metadata: dict`.
4. If `sample_rate` is unknown (typical for raw `.IQ` with no header): **Parameter Estimation module** estimates sample rate / occupied bandwidth from spectral characteristics (Section 6.3).
5. Preprocessing: DC-offset removal, normalization to unit average power, optional resampling to a canonical rate for the ML classifier's fixed input size.
6. **Two parallel, independent estimators** run on the same segment:
   - ML CNN classifier (Section 7.1) → predicted modulation + softmax confidence.
   - Classical cyclostationary/nonlinearity-based analysis (Section 6.4) → estimated modulation family (PSK-like / FSK-like / QAM-like / analog) + estimated symbol rate, entirely independent of the trained model.
7. **Confidence Fusion Engine** combines both outputs into a single decision + a **Trust Score**. If they disagree, both hypotheses are surfaced to the user rather than silently picking one (Section 6.6). If CNN's top-1 softmax probability is below a threshold, the system labels the signal **"Unclassified / Unknown modulation"** rather than forcing a wrong guess (open-set rejection).
8. **Demodulation Engine** is dispatched based on the fused decision, producing a raw bit/symbol stream.
9. **Interleaver-Type Classifier** (statistical/classical ML, Section 7.2) inspects the bitstream and predicts: none / block / convolutional / diagonal / pseudo-random, plus estimated parameters where feasible. **De-interleaving Engine** applies the corresponding inverse operation.
10. **FEC-Scheme Classifier** (Section 7.3) inspects the de-interleaved stream and predicts the FEC family. **FEC Decoding Engine** applies Viterbi / Reed–Solomon / Concatenated / (best-effort LDPC) decoding accordingly.
11. **Bitstream Correlation Engine** searches the decoded payload for periodic sync words / frame boundaries via autocorrelation and a small library of known sync patterns, splitting header vs. payload.
12. All intermediate outputs (constellation plot, waterfall, per-stage confidence, decoded bits) are rendered in the GUI and can be exported as a structured report (JSON + optional PDF).

---

## 6. Module Specifications

Each module below is independently implementable. "Owner track" labels (A/B/C/D) are suggested parallel work-streams — see Section 15 for the full task-division table.

### 6.1 File Ingestion & Format Parsing — *Owner track: A*
- **Purpose:** Read `.wav` and `.IQ` files into raw sample arrays.
- **Input:** File path/bytes, user-supplied hints (sample rate, byte format) if the file is a headerless `.IQ`.
- **Output:** Raw complex/real sample array + whatever metadata was recoverable.
- **Approach:**
  - `.wav`: use `scipy.io.wavfile.read`. If the wav has 2 channels, treat as (I, Q) directly. If mono, treat as a real-valued signal (Hilbert transform may be used later to derive an analytic/complex signal if needed for demodulation).
  - `.IQ`: no universal standard. Support the common conventions: interleaved `int16` or `float32`, I first then Q, little-endian. Read as raw binary via `numpy.fromfile`, reshape to `(-1, 2)`, combine into complex64. Provide GUI fields for the user to override assumed dtype/byte order if auto-detection looks wrong (e.g., visibly wrong dynamic range).
- **Libraries:** `numpy`, `scipy.io.wavfile`.
- **Files:** `src/ingestion/wav_parser.py`, `src/ingestion/iq_parser.py`, `src/ingestion/unified_container.py`.

### 6.2 Unified Signal Container (USC) & Preprocessing — *Owner track: A*
- **Purpose:** Give every downstream module one consistent data structure regardless of source format. This abstraction is itself one of the uniqueness points (Section 9.4) — most naive solutions branch format-specific logic throughout the pipeline; here it's isolated to the ingestion boundary only.
- **Output:** `USC` dataclass: `iq (complex64 ndarray)`, `sample_rate (float|None)`, `source_format (str)`, `duration_sec`, `metadata (dict)`.
- **Preprocessing steps:** DC offset removal (subtract mean), power normalization (divide by RMS), optional decimation/resampling to a canonical rate the CNN expects.
- **Files:** `src/ingestion/unified_container.py`, `src/dsp/preprocessing.py`.

### 6.3 Signal Parameter Estimation (blind sample-rate/bandwidth/SNR) — *Owner track: B*
- **Purpose:** Estimate parameters that a headerless `.IQ` file does not supply. This directly answers the PS's core pain point: "data points recorded from different sensors ... parameters may vary."
- **Approach:**
  - **Occupied bandwidth estimation:** compute Welch PSD (`scipy.signal.welch`), find the frequency band containing e.g. 99% of signal energy → gives occupied bandwidth, from which a sane minimum sample rate can be inferred/suggested if unknown.
  - **SNR estimation:** ratio of in-band signal power (from the estimated occupied band) to out-of-band noise floor power.
  - **Symbol-rate estimation (classical, no ML):** apply a nonlinearity to expose a spectral line at the symbol rate — square the signal for 2-level schemes (BPSK/2FSK), 4th-power for QPSK/4-level schemes — then FFT the result and find the strongest peak away from DC. This is the standard "delay-and-multiply"/nth-power spectral-line method used in real signal intelligence and is computationally light enough for a 10-day build (full cyclostationary FAM analysis is listed as a Tier-2 stretch goal in Section 9).
- **Libraries:** `numpy`, `scipy.signal`.
- **Files:** `src/dsp/parameter_estimation.py`.

### 6.4 Cyclostationary / Classical Modulation-Family Analysis — *Owner track: B*
- **Purpose:** An independent, non-ML second opinion on modulation family, used purely for cross-validation/trust scoring (Section 6.6), not as the primary classifier.
- **Approach:** Reuse the nth-power spectral-line technique from 6.3. Presence/absence and sharpness of specific spectral lines under squaring vs 4th-powering can distinguish: constant-envelope FSK-like signals (strong envelope-variance signature) vs PSK-like (sharp line under nth-power) vs QAM-like (weaker/broader lines, non-constant envelope) vs analog AM/FM (envelope and instantaneous-frequency statistics look very different from digital modulations). Output a coarse family label with a confidence heuristic, not a specific modulation index.
- **Files:** `src/dsp/cyclostationary.py`.

### 6.5 ML Modulation Classifier — *Owner track: C* (see full spec in Section 7.1)
- **Files:** `src/models/modulation_cnn.py`, `src/training/train_modulation.py`, `models_saved/modulation_cnn.pt`.

### 6.6 Confidence Fusion Engine — *Owner track: C*
- **Purpose:** Combine the ML classifier's output with the classical analysis's output into one **Trust Score** and a final decision, and implement **open-set rejection**.
- **Logic (concrete, implementable):**
  1. Let `p_ml` = CNN softmax max probability, `label_ml` = its argmax class.
  2. Let `family_classical` = coarse family from 6.4 (e.g. "PSK-like").
  3. Map `label_ml` to its known family (e.g. QPSK → PSK-like).
  4. If families agree: `trust_score = p_ml` (boosted slightly, e.g. `min(1.0, p_ml * 1.1)`).
  5. If families disagree: `trust_score = p_ml * 0.5`, and the GUI surfaces **both** hypotheses with a "review recommended" flag instead of hiding the disagreement.
  6. If `p_ml < threshold` (default 0.4, tunable): output `"Unclassified"` rather than forcing a class — this open-set behaviour is important because RML2016.10a only covers 11 modulations, and a real off-air capture may contain something the model never saw.
- **Files:** `src/fusion/confidence_fusion.py`.

### 6.7 Demodulation Engine — *Owner track: D*
- **Purpose:** Given a modulation decision, recover the raw bit/symbol stream.
- **Approach:** Use GNU Radio's `gr-digital` blocks scripted via the GNU Radio Python API (not just GRC canvas clicking, so it can be driven programmatically from the pipeline): `gnuradio.digital` provides PSK/QAM/GFSK demodulator hier-blocks with built-in clock recovery and Costas-loop-style carrier tracking. For a simpler, more controllable fallback that avoids fragile GNU Radio flowgraph wiring at runtime, a pure-Python/NumPy demodulator per family is acceptable and recommended for the 10-day timeline:
  - **PSK family (BPSK/QPSK/8PSK):** carrier phase estimation via nth-power method (reuses 6.3/6.4 code), phase correction, then nearest-constellation-point symbol decisions.
  - **FSK family (CPFSK/GFSK):** instantaneous frequency via derivative of unwrapped phase, then threshold/level decisions.
  - **QAM family (QAM16/QAM64):** carrier/phase correction similar to PSK, then 2D nearest-neighbour decision on the I/Q grid.
  - **Analog (AM/FM/SSB):** envelope detection (AM) / frequency discriminator (FM) — included for completeness since RML2016.10a includes these classes, but the PS's explicit ask is FSK/QAM/PSK demodulation; analog demod is a bonus, not core.
- **Output:** bit array (`np.ndarray[uint8]`) or symbol array + constellation points for GUI plotting.
- **Files:** `src/decoding/demodulators/psk_demod.py`, `fsk_demod.py`, `qam_demod.py`, `analog_demod.py`.

### 6.8 Interleaver-Type Classifier + De-interleaving Engine — *Owner track: D* (classifier spec in Section 7.2)
- **De-interleaving approach per predicted type:**
  - **Block interleaver:** parameters are (rows, cols). Brute-force a bounded search over common depths (e.g. rows/cols in {2,4,8,16,32}), applying the inverse (write row-wise, read column-wise reversed) for each candidate, and score each candidate using a "goodness" metric (see below).
  - **Convolutional interleaver (cross-interleaver):** shift-register based, parameters (I branches, J increment). Same bounded search + goodness scoring.
  - **Diagonal interleaver:** diagonal write/read over a block; search over block sizes.
  - **Pseudo-random interleaver:** blind exact-permutation recovery is not realistically solvable without the generator seed/polynomial (this is a known hard problem, not a shortcut we're missing) — for pseudo-random, the system correctly *detects* its presence (via the interleaver classifier) but explicitly reports "pseudo-random interleaving detected — exact de-interleaving requires known generator parameters" rather than pretending to solve it. This honesty is itself defensible in front of judges versus silently failing or faking a result.
  - **Goodness/scoring metric used to pick the winning candidate in the block/convolutional/diagonal search:** after tentative de-interleaving, check for (a) reduction in local bit-run randomness / increase in structure (e.g. lower windowed entropy, indicating recovered structure vs scrambled noise), and (b) whether a known sync pattern (Section 6.10) becomes detectable. Whichever candidate maximizes sync-pattern correlation / minimizes entropy wins.
- **Files:** `src/decoding/deinterleavers/block.py`, `convolutional.py`, `diagonal.py`, `src/decoding/deinterleave_search.py`.

### 6.9 FEC-Scheme Classifier + FEC Decoding Engine — *Owner track: D* (classifier spec in Section 7.3)
- **Decoding approach per predicted scheme, using the same "trial-decode + goodness check" philosophy as 6.8 to remain robust to classifier error:**
  - **Convolutional code (Viterbi):** use `commpy` (imports as `commpy`) — `commpy.channelcoding.convcode` for trellis definition + Viterbi decode. Default to constraint length 7, rate 1/2 (a very common short-constraint code, matches PS wording "short-constrained convolution codes").
  - **Reed–Solomon block code:** use `reedsolo` library. Default to RS(255,223), the most common standard configuration (also used in CCSDS/DVB), decode and report success/failure via RS's own error-correction capability check.
  - **Concatenated codes:** implement as RS(outer) + Convolutional(inner) chain — decode inner (Viterbi) first, then outer (RS), mirroring the classic CCSDS/Voyager-style concatenated scheme. This is the standard real-world meaning of "concatenated codes" and is the correct interpretation for the PS.
  - **LDPC:** best-effort via `pyldpc` if it can be made to build in the environment (Section 14); if not, implement encode-side only for synthetic training-data generation purposes (so the FEC-classifier can still learn to *recognize* LDPC-encoded statistics) and mark decode as a documented future-scope item rather than silently omitting it from the pitch.
  - **Goodness check (used to validate which candidate decode is "real"):** for RS, use its built-in error count / failure signal; for Viterbi, use path metric / residual Hamming-distance-to-nearest-codeword as a soft confidence signal; whichever scheme yields the lowest residual-error indicator is treated as correct if the classifier's top guess fails validation.
- **Files:** `src/decoding/fec/viterbi_wrapper.py`, `rs_wrapper.py`, `concatenated_wrapper.py`, `ldpc_wrapper.py` (best-effort).

### 6.10 Bitstream Correlation / Header–Payload Detection — *Owner track: D*
- **Purpose:** Identify frame/header boundaries in the decoded bitstream.
- **Approach:**
  1. **Autocorrelation-based periodicity detection:** compute the autocorrelation of the bit sequence; a strong periodic peak indicates a repeating frame/sync structure and gives an estimate of frame length.
  2. **Known sync-word library matching:** cross-correlate against a small built-in library of common sync/preamble patterns (e.g. HDLC flag `01111110`, common satellite/telemetry sync words) as a secondary signal.
  3. Once a frame boundary/sync position is found, everything up to the end of the sync pattern (or up to a fixed/estimated header length) is labeled **header**, the remainder **payload**, repeating per frame across the stream.
- **Files:** `src/correlation/bitstream_correlation.py`, `src/correlation/sync_library.py`.

### 6.11 Synthetic Data Generation Pipeline — *Owner track: B* (full spec in Section 8.2)
- **Files:** `src/synthetic_gen/generate_dataset.py`, `src/synthetic_gen/gnuradio_chain.py`.

### 6.12 GUI / Visualization Layer — *Owner track: C* (full spec in Section 12)
- **Files:** `gui/app.py`, `gui/pages/*.py`.

### 6.13 Reporting / Export Layer — *Owner track: C*
- **Purpose:** Package all extracted parameters + key plots into a downloadable structured report (JSON always; PDF/DOCX export as a nice-to-have using existing skills if time permits).
- **Files:** `src/reporting/report_builder.py`.

---

## 7. Machine Learning Model Specifications

Three trainable models total. This is intentionally more than the "one CNN" that most competing teams will build — see Section 9 for why this matters for uniqueness/judging.

### 7.1 Model 1 — Modulation Classifier (Primary, Deep Learning)

- **Framework:** PyTorch.
- **Dataset:** RML2016.10a (`data/RML2016.10a_dict.pkl`), 220,000 labeled frames, 11 classes, SNR -20 to +18 dB. Optionally augment later with synthetic data from Section 8.2 for extra robustness/regularization, but RML2016.10a alone is sufficient for an MVP.
- **Input representation:** shape `(2, 128)` per frame (I channel, Q channel, 128 time samples). Optional enhancement: augment to 4 channels `(I, Q, amplitude, phase)` — amplitude = `sqrt(I²+Q²)`, phase = `arctan2(Q,I)` unwrapped — this is a cheap, literature-supported accuracy improvement over the raw-IQ-only baseline.
- **Architecture (VT-CNN2-inspired, PyTorch `Conv1d`):**
  ```
  Input: (batch, C, 128)   where C = 2 (baseline) or 4 (enhanced)
  Conv1d(C, 64, kernel=8, padding='same') -> BatchNorm1d -> ReLU -> Dropout(0.3)
  Conv1d(64, 128, kernel=4, padding='same') -> BatchNorm1d -> ReLU -> Dropout(0.3)
  Conv1d(128, 128, kernel=4, padding='same') -> BatchNorm1d -> ReLU
  GlobalAveragePooling1d (mean over time dim)
  Linear(128, 256) -> ReLU -> Dropout(0.5)
  Linear(256, num_classes=11)  -> (softmax applied via CrossEntropyLoss, not explicitly in-model)
  ```
- **Training config:** Adam optimizer, `lr=1e-3` with `ReduceLROnPlateau`, `CrossEntropyLoss`, batch size 256, up to 50 epochs with early stopping (patience 7) on validation loss.
- **Split:** stratified 70/15/15 train/val/test across `(modulation, SNR)` pairs so every SNR level is represented in every split.
- **Required evaluation artifacts (must be produced, not optional):**
  - Overall test accuracy.
  - **Accuracy-vs-SNR curve** (line plot, one point per SNR value from -20 to +18 dB) — this single plot is the most convincing thing to show judges, since it demonstrates the system is honest about its limits rather than claiming one flat accuracy number.
  - Confusion matrix, computed separately for low-SNR (≤0 dB), mid-SNR (0–10 dB), high-SNR (>10 dB) buckets.
- **Success criteria:** >80% accuracy at SNR ≥ 10 dB (stretch: >90%); graceful degradation (not a cliff) at low SNR.
- **Output artifact:** `models_saved/modulation_cnn.pt` + `models_saved/modulation_cnn_metrics.json`.

### 7.2 Model 2 — Interleaver-Type Classifier (Classical ML, Feature-Based)

- **Why this exists:** The PS explicitly asks for identification of interleaving type (Block/Convolutional/Diagonal/Pseudo-Random), not just correction of a known type. Most competing solutions will skip *detection* entirely and only implement de-interleaving assuming the type is already known/hardcoded. Building an actual classifier for this is a genuine differentiator (Section 9.2).
- **Dataset (synthetic, generated by the team — see Section 8.2):** random bitstreams (optionally FEC-encoded first, to match realistic pipelines) passed through each interleaver type with randomized parameters, labeled by type: `{none, block, convolutional, diagonal, pseudo_random}`.
- **Feature extraction (deterministic, no learned convolution needed):**
  - Normalized autocorrelation of the bit sequence (first ~64 lags).
  - FFT magnitude spectrum of the bit sequence (captures periodicity).
  - Run-length distribution statistics (mean, variance, max run length).
  - Windowed entropy profile across multiple window sizes (e.g. 8, 16, 32, 64 bits).
  - Total feature vector length: ~60–100 values.
- **Model:** `sklearn.ensemble.RandomForestClassifier` (n_estimators=200) — chosen over a neural net because the dataset is feature-engineered/tabular, training is fast, and **feature importances give a free explainability angle** for the GUI ("the model flagged this as convolutional interleaving primarily due to periodicity signature X").
- **Training:** 80/20 split + 5-fold cross-validation, report accuracy + feature importance bar chart.
- **Output artifact:** `models_saved/interleaver_classifier.pkl`.

### 7.3 Model 3 — FEC-Scheme Classifier (Classical ML, Feature-Based)

- **Why this exists:** same rationale as 7.2 — the PS wants FEC *identification*, and blind FEC recognition is itself treated as a research problem in the literature; a lightweight, defensible classical-ML approach here is a legitimate differentiator rather than assuming the FEC type is given.
- **Dataset (synthetic):** random bitstreams encoded with each scheme: `{none, convolutional, reed_solomon, concatenated, ldpc(best-effort)}`, using `commpy` for convolutional, `reedsolo` for RS, chained for concatenated.
- **Feature extraction:** bit-balance/parity statistics, autocorrelation peak spacing (hints at block/constraint length), Hamming-weight distribution of fixed-length chunks, entropy of fixed-length chunks — same feature family style as 7.2, reused code where possible.
- **Model:** `sklearn.ensemble.RandomForestClassifier` or `GradientBoostingClassifier`.
- **Training:** same protocol as 7.2.
- **Output artifact:** `models_saved/fec_classifier.pkl`.

---

## 8. Dataset Strategy

### 8.1 Public Dataset — RML2016.10a
Already downloaded and verified (Section 3). Used exclusively to train/evaluate Model 1 (Section 7.1). Note its research-use license (see `LICENSE.TXT` bundled with the dataset) should be credited in the final report/pitch.

### 8.2 Synthetic Data Generation Pipeline (for Models 2 & 3, and optional augmentation of Model 1)

Because RML2016.10a only contains raw modulated signals (no FEC, no interleaving), the team must generate its own labeled data for Models 2 and 3. This is done with **scripted Python, using GNU Radio's Python API where signal-chain realism (channel effects) matters, and pure NumPy where only bit-level statistics matter** — the latter is faster to generate at scale and is sufficient for Models 2/3 since they operate on bit statistics, not on the RF waveform itself.

**Generation recipe:**
1. Generate random bit sequences (`numpy.random.randint(0, 2, N)`).
2. Optionally FEC-encode using `commpy` (convolutional) / `reedsolo` (RS) / both (concatenated) — label = FEC type used (feeds Model 3 training).
3. Optionally interleave the (possibly FEC-encoded) bits using one of: block / convolutional / diagonal / pseudo-random interleaver, with randomized parameters within realistic ranges — label = interleaver type used (feeds Model 2 training).
4. (Only for realism/optional Model-1 augmentation) Modulate the resulting bits using GNU Radio's `gnuradio.digital` modulator hier-blocks, pass through `gnuradio.channels.channel_model` (AWGN + optional frequency offset + optional Rician/Rayleigh fading), and sink to both `.wav` (via `blocks.wavfile_sink`) and raw `.IQ` (via `blocks.file_sink`) so both target file formats are represented in the synthetic corpus — this directly produces labeled ground truth for the exact file formats the PS specifies, which RadioML does not provide.
5. Store labels in a companion `manifest.csv`/`manifest.json` alongside the generated files: `{filename, modulation, fec_type, fec_params, interleaver_type, interleaver_params, snr_db, sample_rate}`.

**Files:** `src/synthetic_gen/generate_dataset.py` (orchestrator), `src/synthetic_gen/gnuradio_chain.py` (GNU Radio scripted flowgraph), `data/synthetic/` (output directory), `data/synthetic/manifest.csv`.

---

## 9. Uniqueness & Differentiation Strategy

Most teams attempting this PS will build: RadioML → single CNN → GNU Radio demod blocks → done, and will either fake or entirely skip interleaving/FEC *identification* (only implementing correction for a hardcoded known type). The following points are what should be emphasized in the pitch/demo, roughly in order of impact:

1. **Hybrid classical-DSP + deep-learning cross-validation with a Trust Score (Sections 6.4, 6.6).** For a defense/intelligence end-user (NTRO), a black-box CNN confidence number alone is not trustworthy. Having an independent, non-ML verification path (nonlinearity/spectral-line based family estimation) that must agree with the CNN before high confidence is reported is a directly relevant, judge-legible differentiator — it maps straight onto the PS's own language about "accuracy and confidence of interpretation."
2. **Open-set rejection ("Unclassified" output) instead of forced classification.** Nearly every competing demo will always output *some* modulation label even for garbage/unseen input. Explicitly reporting "Unknown" when confidence is low is more honest and more advanced, and is trivial to implement given the fusion engine already computes a confidence score.
3. **Actual interleaver-TYPE and FEC-TYPE classifiers (Models 2 & 3), not just hardcoded correction.** The PS explicitly asks to *identify* these; most teams will assume they're given. Building small, fast, explainable classical-ML classifiers for these (with feature-importance-based explainability) directly and visibly satisfies a requirement other teams will likely gloss over.
4. **Honest handling of pseudo-random interleaving (Section 6.8).** Rather than faking a de-interleaving result for a case that is genuinely unsolvable without a known generator seed, the system correctly detects and reports the limitation. This kind of engineering honesty reads well to technically literate judges and avoids an embarrassing "why did your undo of pseudo-random interleaving just produce garbage" question.
5. **Unified Signal Container abstraction (Section 6.2).** Clean architectural separation between "which file format did this come from" and "everything downstream" is a genuine software-engineering quality signal, not just a feature — worth mentioning explicitly when judges ask about extensibility (e.g., "what if we get a `.sigmf` file next year").
6. **Blind sample-rate/bandwidth/symbol-rate estimation (Section 6.3).** Directly answers the PS's stated pain point about inconsistent sensor metadata — most teams will assume sample rate is always known/in the header.
7. **Trial-decode-with-goodness-check fallback (Sections 6.8, 6.9).** Even if the FEC/interleaver classifier's top guess is wrong, the system doesn't just fail — it validates the result and can fall back to the next-best candidate, making the pipeline more robust than a naive "classify once, trust blindly" chain.

**Tier-2 / stretch ideas (mention in the pitch as roadmap, do not attempt to fully build in 10 days unless ahead of schedule):**
- Full cyclostationary Spectral Correlation Density analysis via the FFT Accumulation Method (FAM), superseding the simpler nth-power method in Section 6.3/6.4.
- Wideband multi-signal detection: energy-detection based channelization of a wide capture into multiple sub-band signals before per-signal analysis (useful since a real off-air capture may contain more than one emitter).
- RF-fingerprinting / specific-emitter identification.
- Domain-adaptation/fine-tuning of Model 1 on real SDR-captured data once/if hardware becomes available.

---

## 10. Technology Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| DL framework | PyTorch |
| Classical ML | scikit-learn |
| DSP / signal chain | NumPy, SciPy, GNU Radio 3.10 (`gnuradio.digital`, `gnuradio.channels`, `gnuradio.blocks`) |
| FEC libraries | `commpy` (scikit-commpy) for convolutional/Viterbi, `reedsolo` for RS, `pyldpc` best-effort for LDPC |
| GUI | Streamlit |
| Visualization | Matplotlib, Plotly |
| Dataset | RML2016.10a (DeepSig) + self-generated synthetic corpus |

---

## 11. Repository Structure

```
sih26147/
├── data/
│   ├── RML2016.10a_dict.pkl
│   ├── LICENSE.TXT
│   └── synthetic/
│       └── manifest.csv
├── src/
│   ├── ingestion/
│   │   ├── wav_parser.py
│   │   ├── iq_parser.py
│   │   └── unified_container.py
│   ├── dsp/
│   │   ├── preprocessing.py
│   │   ├── parameter_estimation.py
│   │   └── cyclostationary.py
│   ├── models/
│   │   └── modulation_cnn.py
│   ├── training/
│   │   ├── train_modulation.py
│   │   ├── train_interleaver.py
│   │   └── train_fec.py
│   ├── fusion/
│   │   └── confidence_fusion.py
│   ├── decoding/
│   │   ├── demodulators/
│   │   │   ├── psk_demod.py
│   │   │   ├── fsk_demod.py
│   │   │   ├── qam_demod.py
│   │   │   └── analog_demod.py
│   │   ├── deinterleavers/
│   │   │   ├── block.py
│   │   │   ├── convolutional.py
│   │   │   └── diagonal.py
│   │   ├── deinterleave_search.py
│   │   └── fec/
│   │       ├── viterbi_wrapper.py
│   │       ├── rs_wrapper.py
│   │       ├── concatenated_wrapper.py
│   │       └── ldpc_wrapper.py
│   ├── correlation/
│   │   ├── bitstream_correlation.py
│   │   └── sync_library.py
│   ├── synthetic_gen/
│   │   ├── generate_dataset.py
│   │   └── gnuradio_chain.py
│   ├── reporting/
│   │   └── report_builder.py
│   └── utils/
├── models_saved/
│   ├── modulation_cnn.pt
│   ├── modulation_cnn_metrics.json
│   ├── interleaver_classifier.pkl
│   └── fec_classifier.pkl
├── gui/
│   ├── app.py
│   └── pages/
│       ├── 1_upload_preview.py
│       ├── 2_parameters.py
│       ├── 3_modulation.py
│       ├── 4_demodulation.py
│       ├── 5_deinterleaving.py
│       ├── 6_fec_decoding.py
│       ├── 7_correlation.py
│       └── 8_report.py
├── notebooks/
├── reports/
├── tests/
└── requirements.txt
```

---

## 12. GUI / UX Specification

**Framework choice justification:** Streamlit was chosen over PyQt/Electron specifically because of the 10-day constraint — it renders a professional, interactive, web-based GUI with a fraction of the boilerplate, and Streamlit's multipage app structure (`gui/pages/`) maps cleanly onto the pipeline stages, so the demo naturally walks judges through the system stage by stage.

**Pages (Streamlit multipage app):**
1. **Upload & Preview** — file upload widget, auto-detected format, raw waveform plot, spectrogram/waterfall (time-frequency) plot.
2. **Parameters** — estimated sample rate, occupied bandwidth, estimated SNR, estimated symbol rate (Section 6.3), with an option for the user to manually override any estimate.
3. **Modulation** — constellation diagram, CNN top-3 predictions with confidence bars, classical cross-check result, fused Trust Score, "Unclassified" state clearly styled differently (e.g. amber warning) from a confident result.
4. **Demodulation** — recovered bit/symbol stream preview (hex/binary toggle), eye-diagram plot if feasible.
5. **De-interleaving** — predicted interleaver type + confidence + feature-importance mini-chart, before/after bitstream visualization, manual override dropdown for all 4 types + none.
6. **FEC Decoding** — predicted FEC scheme + confidence, decoded payload, pass/fail residual-error indicator, manual override dropdown.
7. **Bitstream Correlation** — autocorrelation plot with detected periodicity marked, header/payload split visualization on the bitstream.
8. **Report** — consolidated JSON view of every extracted parameter across all stages, with a download button; PDF/DOCX export is a nice-to-have if time remains (existing docx/pdf tooling can be reused for this later, not a Day 1–9 priority).

---

## 13. Evaluation Metrics & Success Criteria

| Component | Metric | Target |
|---|---|---|
| Modulation classifier | Accuracy @ SNR ≥ 10 dB | > 80% (stretch > 90%) |
| Modulation classifier | Accuracy-vs-SNR curve | Must be produced and shown, not just a single number |
| Interleaver classifier | 5-fold CV accuracy on synthetic test set | > 75% |
| FEC classifier | 5-fold CV accuracy on synthetic test set | > 75% |
| End-to-end pipeline | Live demo: upload → full report, on at least 3 distinct synthetic example files | Must run without crashing, in front of judges |
| Confidence fusion | At least one demo example deliberately shown where fusion correctly flags disagreement/low-confidence | Demonstrates the trust/explainability angle explicitly |

---

## 14. Risk Register & Fallback Plan (priority-ordered cuts if time runs short)

| Risk | Impact | Fallback |
|---|---|---|
| `pyldpc` doesn't build on Python 3.14 | LDPC decode unavailable | Ship without LDPC decode; keep LDPC *encode* (pure Python, no exotic deps) for training-data generation so the FEC classifier can still recognize LDPC statistics; state LDPC decode as documented future work. |
| Blind pseudo-random de-interleaving | Cannot exactly recover permutation | This is expected and acceptable — detect-and-report-limitation is the correct behaviour (Section 9.4), not a failure to fix. |
| GNU Radio scripted flowgraphs are slow/fragile to get right under time pressure | Delays synthetic data generation | Fall back to pure NumPy bit-level generation for Models 2/3 (Section 8.2 already recommends this as primary path); only use GNU Radio's channel/modulation blocks if time remains, for Model-1 augmentation realism. |
| CNN accuracy lower than hoped at low SNR | Weakens headline number | This is expected and literature-consistent — the accuracy-vs-SNR curve itself (showing honest degradation) is the deliverable, not a single flattering number. |
| Time runs out before Model 3 (FEC classifier) is trained | Missing one differentiator | Cut Model 3 before cutting Model 2 (interleaver classifier) — both are valuable, but interleaving detection is explicitly named first in the PS's feature list and is slightly cheaper to get right. |
| Time runs out before GUI polish (Section 12) | Weak demo impression | Prioritize pages 1, 3, 5, 6, 8 (upload, modulation, deinterleaving, FEC, report) over 2, 4, 7 if forced to cut — these five best tell the "identify → decode" story end to end. |

---

## 15. Task Division for Parallel Work Streams

| Track | Scope | Modules |
|---|---|---|
| **A** | Ingestion & data plumbing | 6.1, 6.2 |
| **B** | Classical DSP & synthetic data | 6.3, 6.4, 6.11, Section 8 |
| **C** | ML models, fusion, GUI | 7.1 (Model 1), 6.6, 6.12, 6.13 |
| **D** | Decoding chain (demod/deinterleave/FEC/correlation) | 6.7, 6.8, 6.9, 6.10, 7.2 (Model 2), 7.3 (Model 3) |

Tracks A and B can start completely in parallel on Day 1. Track C's Model 1 training depends only on the already-downloaded RML2016.10a dataset, so it can also start Day 1 independently. Track D depends on Track B's synthetic data generator (Section 8.2) being functional, so Track D should start with the demodulation sub-module (6.7, needs only Track C's Model 1 output contract, not actual trained weights) while waiting on synthetic data.

---

## 16. 10-Day Execution Timeline (reference)

- **Day 1:** Environment finalization (done), RML2016.10a verified (done), begin Track A (ingestion) and Track C (Model 1 training script) in parallel.
- **Day 2–3:** Finish Model 1 training + accuracy-vs-SNR evaluation. Finish ingestion + Unified Signal Container.
- **Day 3–4:** Track B builds parameter estimation (6.3) + synthetic data generator (8.2, NumPy-first).
- **Day 4–5:** Basic Streamlit GUI skeleton wired to Model 1 (upload → classify → display), so an end-to-end demo-able artifact exists early.
- **Day 5–6:** Track D builds demodulation (6.7) using Model 1's output contract.
- **Day 6–7:** Generate synthetic interleaved/FEC-encoded data; train Model 2 (interleaver classifier) and Model 3 (FEC classifier).
- **Day 7–8:** Build de-interleaving engine (6.8) and FEC decoding engine (6.9) around the two new classifiers.
- **Day 8–9:** Confidence fusion engine (6.6), bitstream correlation (6.10), GUI pages for all remaining stages.
- **Day 9–10:** Full pipeline integration test on multiple synthetic example files, GUI polish, prepare a pre-recorded backup demo video in case of live failure, finalize the report/export page.

---

## 17. Glossary

- **USC (Unified Signal Container):** internal canonical data structure holding IQ samples + metadata regardless of source file format.
- **Trust Score:** fused confidence metric combining ML classifier confidence and classical cross-validation agreement.
- **Open-set rejection:** deliberately outputting "Unknown/Unclassified" instead of forcing a classification when confidence is too low.
- **Goodness/scoring metric (interleaving/FEC search):** a heuristic (entropy reduction, sync-pattern correlation, residual error) used to pick the correct candidate among several brute-forced options when the classifier's top guess needs validation.
- **FAM (FFT Accumulation Method):** a full cyclostationary spectral-correlation analysis technique; listed as Tier-2/stretch, not core MVP.
