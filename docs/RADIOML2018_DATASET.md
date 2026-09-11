# RadioML 2018.01A (Dataset 2): forensics, split protocol, and training

Dataset: DeepSig RadioML 2018.01A, `GOLD_XYZ_OSC.0001_1024.hdf5` (21.4 GB), CC BY-NC-SA 4.0.
Location: `C:\Users\Kaustubh Bhoir\Documents\RadioFry\dataset 2\`.
Companion to `docs/REALWORLD_DATASET.md`, which covers Dataset 1 (Belousov & Ronkin).

Everything below was measured from the released file. **The dataset was never modified** — all
access is `h5py` mode `'r'`. Reproduction scripts live in
`research_memory/experiments/radioml2018_2026_09/`.

---

## 1. Structure

```
dataset 2/
  GOLD_XYZ_OSC.0001_1024.hdf5   21,449,148,312 B
  classes.txt                              262 B   <- WRONG ordering, see section 2
  classes-fixed.json / .txt          246 / 1632 B  <- the correct ordering
  LICENSE.TXT                           20,993 B
```

| dataset | shape | dtype | chunks | compression |
|---|---|---|---|---|
| `X` | `(2 555 904, 1024, 2)` | float32 | none (contiguous) | none |
| `Y` | `(2 555 904, 24)` | int64 one-hot | none | none |
| `Z` | `(2 555 904, 1)` | int64 (SNR dB) | none | none |

**No root attributes. No class names inside the file. No train/val/test split. No recording
or capture identifier of any kind.**

* 24 classes x 26 SNR levels x **exactly 4096 frames** = 2,555,904. Verified: every one of
  the 624 configurations has exactly 4096 frames — **perfectly balanced**, unlike Dataset 1.
* SNR spans **-20 to +30 dB in 2 dB steps**. This is the property Dataset 1 lacked entirely
  (it had only 20–30 dB), and `research_memory/CURRENT.md` had listed low-SNR labelled data as
  a specific collection requirement.
* The file is laid out as contiguous 4096-frame `(class, SNR)` blocks, class-major then SNR
  ascending. `radioml2018.block_start` encodes this and `verify_layout` asserts it against the
  stored labels before anything relies on it.
* `X` being contiguous and uncompressed makes indexed reads cheap — no decompression, direct
  seek.

---

## 2. The class ordering — the single most dangerous thing in this dataset

The download ships **two contradictory orderings** and the HDF5 carries no metadata to
arbitrate. Using the wrong one mislabels every frame while nothing crashes.

| index | `classes.txt` (shipped) | `classes-fixed` |
|---|---|---|
| 0 | 32PSK | **OOK** |
| 1 | 16APSK | **4ASK** |
| 2 | 32QAM | **8ASK** |
| 3 | FM | **BPSK** |
| … | … | … |
| 21 | AM-DSB-WC | **FM** |
| 22 | OOK | **GMSK** |

### Settled by measurement

Discriminators chosen because they are physical and unambiguous at high SNR:

| index | impropriety `\|E[x²]\|/E[\|x\|²]` | amplitude CV | asymmetry | DC | shipped says | fixed says |
|---|---|---|---|---|---|---|
| 0 | 0.993 | 0.460 | 0.004 | 53.5% | 32PSK | **OOK** |
| 1 | 0.990 | 0.602 | 0.006 | 68.1% | 16APSK | **4ASK** |
| 2 | 0.986 | 0.617 | 0.006 | 73.4% | 32QAM | **8ASK** |
| 3 | 0.996 | 0.467 | 0.001 | 0.4% | FM | **BPSK** |
| 17 | 0.896 | 0.527 | 0.145 | 99.6% | AM-SSB-WC | AM-SSB-WC |
| 19 | 0.996 | 0.492 | **0.000** | 0.0% | QPSK | **AM-DSB-WC** |
| 21 | 0.003 | **0.007** | 0.997 | 0.0% | AM-DSB-WC | **FM** |
| 22 | 0.076 | 0.107 | 0.076 | 0.8% | OOK | **GMSK** |

* **Impropriety agrees with the fixed ordering for 24 of 24 classes, and with the shipped one
  for 10 of 24.** A real-valued constellation (OOK/ASK/BPSK/AM) has `E[x²] ≠ 0`; a proper
  complex one (PSK/QAM/APSK) has `E[x²] ≈ 0`. Indices 0–3 measure ~0.99 — they cannot be
  32PSK/16APSK/32QAM/FM.
* **Index 21 has amplitude CV 0.007** — a perfectly constant envelope. That is FM. It cannot
  be the shipped ordering's AM-DSB-WC, which is amplitude modulation by definition.
* **Index 22 has CV 0.107** — near-constant envelope, i.e. GMSK. It cannot be OOK, whose
  envelope is maximally variable.
* **Indices 19/20 have spectral asymmetry exactly 0.000** — symmetric, as double-sideband must
  be.

### Independent confirmation, sharing no assumption with the above

For M-ary PSK the M-th power moment collapses the constellation to a point and peaks at M:

| index | fixed says | M=2 | M=4 | M=8 | peak |
|---|---|---|---|---|---|
| 3 | BPSK | **0.992** | 0.969 | 0.886 | **M=2** |
| 4 | QPSK | 0.073 | **0.420** | 0.233 | **M=4** |
| 5 | 8PSK | 0.066 | 0.051 | **0.150** | **M=8** |

A clean 2 → 4 → 8 progression. **16PSK and 32PSK are inconclusive, not contradictory**: each
frame holds only ~100 symbols and RadioML applies a deliberate carrier frequency offset, which
together destroy 16th- and 32nd-order moments.

**Verdict: the `classes-fixed` ordering is correct. DeepSig's shipped `classes.txt` is wrong
and must not be used.** `radioml2018.CLASSES` is the fixed ordering; the shipped one is kept
as `SHIPPED_WRONG_ORDER` purely so a test can assert it is *not* in use.

### Two of my own checks misfired and are not counted as evidence

* A zero-amplitude-fraction test was expected to isolate OOK. It did not (index 0 measured
  7.4%, index 2 measured 11.6%): pulse shaping fills the "off" symbols, so the envelope never
  reaches zero. Inconclusive, not contradictory.
* Spectral asymmetry was expected to rank the two AM-SSB classes highest. FM outranked them
  (0.997), because RadioML's FM sits off-centre and its narrowband energy lands on one side.

---

## 3. Signal characterisation

### Samples per symbol

| measurement | result |
|---|---|
| occupied bandwidth (BW99), per frame | **0.1270 fs**, 5th–95th percentile 0.119–0.135 |
| implied sps under RRC α = 0.35 | **≈ 10.6** |
| per-frame cyclic-line estimate | median 19.7–27.7, 5th–95th 13.8–46.7 |

**The bandwidth figure is the one to use.** The per-frame cyclic-line spread is estimator
variance, not real variance: occupied bandwidth cannot be constant to ±6% while the true sps
swings 3×. Bandwidth is an energy integral and robust; the `|s|²` line is a peak pick on ~100
symbols with carrier offset, and it lands on noise.

RadioML is commonly described as sps = 8. The measured occupied bandwidth implies ~10.6.
**This discrepancy is unresolved** and does not block training — what matters for the CNN is
that sps is *consistent across frames*, and it is. It is worth noting that ~10.6 is close to
Dataset 1's measured 10, and that neither is in V3's trained sweep of (4, 8, 16, 32).

### Amplitude

Most classes are power-normalised to ~1.000 (std 0.02–0.04). Five are not:

| class | mean frame power |
|---|---|
| AM-SSB-WC | 155.9 |
| AM-SSB-SC | 67.1 |
| 8ASK | 3.68 |
| 4ASK | 3.06 |
| OOK | 2.14 |
| everything else | ~1.00 |

Spread 21.9 dB, comparable to Dataset 1's 23.3 dB — so an amplitude shortcut exists here too.
**RadioFry's production contract normalises per window and cannot use it.** The AM-SSB figures
are dominated by a residual carrier (99% of their energy sits at DC).

### Unresolved: AM-DSB-WC carries no DC

Index 19 is AM-DSB-**WC** ("with carrier") yet measures 0.0% DC fraction and unit power, while
the two AM-SSB classes measure ~99% DC. This is internally odd and is **not explained**. It does
not affect the digital classes RadioFry's CNN handles, and is recorded rather than smoothed over.

---

## 4. Leakage — three mechanisms tested, all absent

RadioML publishes no split and no recording identifier, so the split is ours to design and the
only question is whether any split can be independent. Measured against a **phase-randomised
surrogate null** (identical PSD, unrelated content, DC removed) — the naive `1/√N` floor is
wrong for bandlimited frames, an error that cost real time on Dataset 1 and is not repeated.

**L1 — do the 26 SNR levels share one underlying waveform?** Frame *k* of (class, 30 dB) vs
frame *k* of (class, other dB), at matched block offsets:

| class | 30 vs 28 | 30 vs 20 | 30 vs 0 | shuffled | surrogate |
|---|---|---|---|---|---|
| BPSK | 0.065 | 0.058 | 0.048 | 0.067 | 0.076 |
| QPSK | 0.078 | 0.076 | 0.058 | 0.077 | 0.074 |
| 16QAM | 0.083 | 0.075 | 0.057 | 0.081 | 0.087 |
| 64QAM | 0.075 | 0.083 | 0.057 | 0.075 | 0.082 |
| GMSK | 0.077 | 0.081 | 0.054 | 0.076 | 0.074 |

Matched offset is **indistinguishable from shuffled and from the null**. The SNR levels are
independent realisations — RadioML did not re-noise one waveform per SNR.

**L2 — are consecutive frames slices of one continuous recording?**

| class | adjacent (k, k+1) | far apart | surrogate |
|---|---|---|---|
| BPSK | 0.058 | 0.057 | 0.068 |
| QPSK | 0.078 | 0.080 | 0.080 |
| 16QAM | 0.076 | 0.081 | 0.074 |
| GMSK | 0.084 | 0.076 | 0.073 |

Adjacent equals far-apart equals null. **Frames carry no temporal order.**

**L3 — near-duplicates within a block** (peak |corr| over all lags, DC removed):

| class | observed | null | excess | frac > 0.9 |
|---|---|---|---|---|
| BPSK | 0.391 | 0.309 | +0.082 | **0.0%** |
| QPSK | 0.306 | 0.303 | +0.003 | **0.0%** |
| 16QAM | 0.305 | 0.305 | −0.001 | **0.0%** |
| 64QAM | 0.305 | 0.304 | +0.002 | **0.0%** |
| OOK | 0.394 | 0.310 | +0.084 | **0.0%** |
| GMSK | 0.317 | 0.310 | +0.008 | **0.0%** |

Compare Dataset 1, where **48% of QAM test frames** had a near-duplicate in train. The small
BPSK/OOK excess is structural — a BPSK frame lives in a 1-D real subspace, so independent
frames resemble each other — not duplication.

**L4 — exact duplicates:** 0 in 9,000 frames sampled across 18 blocks.

### What can and cannot be claimed

**A stratified random split is defensible on this dataset, which it never was on Dataset 1.**
All three mechanisms that would make one leak are measurably absent.

**The caveat that must travel with any result**: there are no recording identifiers, so
recording-level independence is *inferred from the absence of its symptoms*, not guaranteed by
metadata. That is far stronger evidence than Dataset 1 had, and still not a proof.

---

## 5. Split protocol

Each 4096-frame `(class, SNR)` block is partitioned **contiguously**:

| split | offsets in block | frames/config | total |
|---|---|---|---|
| train | 0 – 2867 | 2867 | 1,789,008 |
| val | 2867 – 3481 | 614 | 383,136 |
| test (**sealed**) | 3481 – 4096 | 615 | 383,760 |

Contiguous rather than random **because L2 showed frames carry no temporal order**, so a
contiguous partition is equivalent to a random one and reads sequentially from a 21 GB file
instead of seeking per frame. The split is stratified over class and SNR by construction.

`subset test` is sealed: `train_radioml.load_pool` raises `PermissionError` for it, model
selection uses the validation split only, and no threshold is tuned on it.

---

## 6. Label overlap with RadioFry

RadioFry's production checkpoint is 8-class digital. **Five have an exact counterpart** — up
from three on Dataset 1:

    BPSK -> BPSK,  QPSK -> QPSK,  8PSK -> 8PSK,  16QAM -> QAM16,  64QAM -> QAM64

Deliberately **not** mapped:

* **GMSK → GFSK.** GMSK is CPM with h = 0.5; RadioFry's GFSK is a separate generator
  configuration. Same decision as Dataset 1.
* **4ASK → PAM4.** RadioML's 4ASK measures 68% DC fraction, i.e. unipolar levels; PAM4 is
  bipolar. A direct generator-to-generator comparison was attempted and **the probe was
  broken** — `level_profile` subtracted the per-frame mean, destroying the very DC under test,
  so all four classes came out symmetric. **Unresolved, therefore unmapped.**
* **CPFSK** has no counterpart here.

---

## 7. Production input contract

Unchanged and verified, not assumed. `train_realworld.build_windows` is asserted
element-for-element (`np.array_equal`) against `modulation_inference._window_frames` +
`add_signal_features` by `tests/test_realworld_training.py`, which **now passes** (it had never
been executed when `docs/REALWORLD_DATASET.md` was written).

* 128-sample windows at the four fixed starts `[0, 298, 597, 896]`
* 4 channels `iqap`, per-window power normalisation then per-channel renormalisation
* frame decision = mean of the four softmax vectors

Training uses **one random window start per frame per epoch** as timing-offset augmentation;
evaluation always uses the fixed four.

---

## 8. GPU

Training is GPU-only and refuses to fall back. `training/device.verify_cuda` proves, in order:
a CUDA build, `is_available()`, an enumerated device, **that the compiled arch list covers this
device's capability**, a real kernel launch, parameters on the device, and an optimiser step
that changes weights. It raises `CudaUnavailable` rather than returning a CPU device.

Verified on this machine:

| check | result |
|---|---|
| Python | 3.14.5 AMD64 |
| PyTorch | **2.13.0+cu130** |
| CUDA / cuDNN | 13.0 / 92000 |
| `cuda.is_available()` | True |
| device | **NVIDIA GeForce RTX 5060 Laptop GPU**, 8.55 GB, 26 SMs |
| capability | **sm_120** (Blackwell) |
| `get_arch_list()` | `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — **exact sm_120 match** |
| kernel launch | matmul verified against CPU, max abs error 6.9e-05 |
| smoke training step | loss 3.1944, weights changed, peak 235.8 MB |

Throughput: **49,767 windows/s training** (batch 512) and 148,076 windows/s inference, against
1,951 windows/s on CPU — a **25× speedup**. The CPU-side window builder runs at 64,575
windows/s, so it does not starve the GPU.

`is_available()` alone is explicitly not treated as sufficient: a pre-CUDA-12.8 build can
report True on a Blackwell card and then fail at the first kernel launch.

---

## 9. Results

Three runs, identical except for what is trainable. 374,400 train / 62,400 validation frames
(600/100 per configuration, balanced across all 24 classes and 26 SNR levels), 30 epochs,
batch 1024, Adam lr 1e-3, seed 20260911, ~14 min each on the RTX 5060.

Evaluated on the **sealed test split**, which was never read during training or model
selection.

### Headline

| model | 5 mappable classes | full 24-class |
|---|---|---|
| **frozen V3 (zero training)** | **15.97%** | n/a (8 outputs) |
| linear_probe (V3 trunk frozen) | 30.19% | 33.17% |
| **finetune (candidate)** | **44.46%** | **42.62%** |
| scratch (control) | 38.49% | 40.73% |

Each model answers across **its own full label space** — V3 among 8, the others among 24 — so
no model is given an easier task than another. Chance on the 24-class task is 4.17%.

### What the three runs establish

Against the interpretation fixed in advance (section 9 of `docs/REALWORLD_DATASET.md`):

1. **V3's features transfer partially, but are not directly reusable.** A linear head on the
   frozen trunk reaches 30–33% — well above chance and roughly double V3's own 15.97%
   end-to-end — but 10+ points below what unfreezing the trunk achieves. The synthetic
   representation contains real signal; it is not sufficient.
2. **Synthetic pre-training is a modest help, not a head start.** finetune beats scratch by
   **6.0 points** on the mappable classes and **1.9 points** on the full task. Most of the
   performance comes from training on real data, not from where training started.
3. **Real-data training does close the domain gap** that Entry 044/045 documented: 15.97% →
   44.46% on the same frames, a **2.8× improvement**.

### Accuracy against SNR (finetune, 24-class)

```
 -20dB   4.3%   -18dB   4.2%   -16dB   4.5%   -14dB   5.0%
 -12dB   5.6%   -10dB   5.9%    -8dB   7.6%    -6dB  13.5%
  -4dB  18.8%    -2dB  30.0%     0dB  39.4%     2dB  51.2%
   4dB  54.7%     6dB  60.1%     8dB  65.6%    10dB  66.7%
  12dB  64.4%    14dB  66.8%    16dB  69.3%    18dB  66.7%
  20dB  67.4%    22dB  66.7%    24dB  66.8%    26dB  67.7%
  28dB  68.2%    30dB  67.0%
```

**Mean ≥ +10 dB: 67.06%. Mean ≤ −10 dB: 4.92%, i.e. chance.** The overall 42.62% is an
average over a dataset half of which is at or below 0 dB, where 128 samples carry almost no
recoverable information. The curve is monotone through the transition and flat above +10 dB,
which is the expected shape.

### Per-class (finetune, 24-class, n = 624 each)

| | | | |
|---|---|---|---|
| FM 86.4% | GMSK 77.1% | AM-SSB-WC 72.6% | AM-DSB-WC 67.1% |
| OOK 67.1% | BPSK 66.0% | QPSK 62.0% | OQPSK 60.6% |
| 4ASK 58.0% | 32PSK 56.4% | 16APSK 54.2% | 32APSK 54.2% |
| 8ASK 52.1% | 8PSK 50.0% | 16QAM 45.7% | 128APSK 22.0% |
| 256QAM 20.4% | 32QAM 19.7% | AM-DSB-SC 14.6% | AM-SSB-SC 12.0% |
| 64APSK 3.4% | 128QAM 1.3% | 64QAM 0.2% | **16PSK 0.0%** |

The failures are structured, not random — every one collapses onto an adjacent member of its
own family:

| true | acc | goes to |
|---|---|---|
| 16PSK | 0.0% | **32PSK 362**, GMSK 82, FM 73 |
| 64QAM | 0.2% | **256QAM 128**, FM 84, 16QAM 77 |
| 128QAM | 1.3% | **256QAM 82**, FM 80, GMSK 75 |
| 64APSK | 3.4% | **128APSK 106**, 16APSK 67 |
| AM-SSB-SC | 12.0% | **AM-SSB-WC 408** |
| AM-DSB-SC | 14.6% | **AM-DSB-WC 383** |

This is the same information-limited pattern Entry 041 established for QAM16/QAM64 on
synthetic data: at 128 samples and ~10 samples/symbol a frame holds only ~12 symbols, which
cannot separate 16PSK from 32PSK or 64QAM from 256QAM regardless of the model. The
suppressed-carrier/with-carrier AM pairs collapse the same way.

### Confidence behaviour — the clearest win

| model | median conf. correct | median conf. wrong | max wrong | wrong ≥ 0.9 | inverted? |
|---|---|---|---|---|---|
| frozen V3 | 0.762 | 0.557 | **1.000** | **798** | no |
| linear_probe | 0.846 | 0.080 | 0.743 | **0** | no |
| **finetune** | 0.867 | **0.058** | **0.588** | **0** | no |
| scratch | 0.993 | 0.065 | 0.660 | 0 | no |

**All three trained models produce zero confident-wrong predictions.** The candidate's wrong
answers sit at median confidence 0.058 and never exceed 0.588, against 0.867 when correct.
That is a usable rejection threshold, and it is the property Dataset 1 most conspicuously
lacked — there the frozen checkpoint's calibration *inverted* (wrong answers more confident
than right ones, 492 above 0.99).

### Honest limits on these numbers

* **128 samples, not 1024.** Published RadioML 2018.01A results use the full 1024-sample
  frame and far larger networks; they report ~95% at high SNR and ~60% overall. This model
  sees **1/8 the context** with **140,504 parameters**, because that is the frozen production
  input contract. 67% at ≥10 dB under that constraint is a reasonable result, **not** a
  state-of-the-art one, and must not be presented as beating published figures.
* **No recording identifiers.** Independence is inferred from the measured absence of
  cross-SNR waveform reuse, temporal contiguity and near-duplicates (section 4), not
  guaranteed by metadata.
* **Not promoted.** The candidate is written to a new filename. The frozen production
  checkpoint `a7b02533a7c7129c` is unchanged, and promotion remains a separate decision —
  this model has a 24-class label space that the current 8-class pipeline does not consume.
* **Nothing was tuned on the test split**, and no threshold was fitted anywhere.

### Provenance

| run | state_dict sha256 | train frames | best epoch | val | seconds |
|---|---|---|---|---|---|
| linear_probe | `49e49224cbf686f6` | 374,400 | 20 | 0.3299 | 860 |
| finetune | `cecbd88c7a059275` | 374,400 | 27 | 0.4288 | 831 |
| scratch | `09c0d62b856ad187` | 374,400 | 27 | 0.4116 | 834 |

All three share train-index digest `b6a53edd6a9eb90d` (identical frames, so the comparison is
controlled), device sm_120, seed 20260911. Full metrics including per-epoch history are in
`models_saved/modulation_cnn_radioml_*_metrics.json` and
`reports/radioml2018_experiments.json`.
