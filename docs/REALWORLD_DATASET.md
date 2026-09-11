# Real-world IQ dataset: investigation record and training plan

**Status: investigation complete, training NOT started.**
Dataset: Belousov & Ronkin (2026), *Real-World IQ Dataset for Automatic Radio Modulation
Recognition under Multipath Channels*, Mendeley Data DOI 10.17632/tjzsbph49x.1.
Location on this machine: `C:\Users\Kaustubh Bhoir\Documents\RadioFry\Real-World IQ Dataset for Automatic Radio Modulati\dataset\`.

Everything below was measured from the released files. Where the paper and the data disagree,
the measurement is used and the disagreement is recorded rather than reconciled silently.
Reproduction scripts are in `research_memory/experiments/realworld_2026_09/`, raw outputs in
`research_memory/experiments/realworld_2026_09/measurements/`.

**The dataset was never modified, extracted over, or written to.** All access is `h5py` mode
`'r'`. The original archive remains in `~/Downloads` and is byte-unchanged.

---

## 1. Dataset structure and inventory

```
dataset/
  README.txt                                  270 B
  Real-World IQ Dataset ... .docx         241,080 B   (the paper)
  subset_train.h5                     591,177,568 B
  subset_val.h5                       121,460,677 B
  subset_test.h5                      121,573,165 B
```

| dataset | shape | dtype | chunks | compression |
|---|---|---|---|---|
| `X` | `(N, 1024, 2)` | float16 | `(4096, 1024, 2)` | gzip |
| `y_mod` | `(N,)` | int16 | `(5000,)` | gzip |
| `y_chan` | `(N,)` | int8 | `(10000,)` | gzip |
| `y_snr` | `(N,)` | int16 | `(5000,)` | gzip |

Root attributes: `frame_len=1024`, `subset_name`, `source='dataset3_iq_frames.h5'`,
`channel_meaning='0=clean, 1=multipath(ref)'`, and
`mod2id_json={"BPSK":0,"QPSK":1,"QAM":2,"GMSK":3,"OFDM":4,"NBFM":5,"WBFM":6}`.

**The label mapping is read from `mod2id_json` at load time, never assumed from the order
the paper lists classes in.**

| split | frames | share | configurations | frames/config min–max |
|---|---|---|---|---|
| train | 400,000 | 71.43% | 84 / 84 | 3,853 – 5,814 |
| val | 80,000 | 14.29% | 84 / 84 | 753 – 1,158 |
| test | 80,000 | 14.29% | 84 / 84 | 762 – 1,179 |
| **total** | **560,000** | | | |

A "configuration" is one (modulation, channel, SNR) combination = 7 × 2 × 6 = 84, and each
corresponds to one `.dat` recording under the paper's own `{MOD}_{ref}_{SNR}.dat` naming.

Per-class share ranges **13.16% (NBFM) to 15.68% (WBFM)** — a 19% relative imbalance.
SNR values present: **20, 22, 24, 26, 28, 30 dB only.** Channel: ~51% / 49% clean/multipath.
Capture: 2.4 GHz carrier, 2 MSps, HackRF One transmit and receive, ~3 m indoor separation.

---

## 2. Discrepancies between the paper and the released data

Six, of which four matter for training.

| item | paper | released data (measured) | matters? |
|---|---|---|---|
| total frames | 840,000 | **560,000** | minor |
| split sizes | 588k / 126k / 126k | **400k / 80k / 80k** | minor |
| balance | exactly 10,000 per configuration | **753 – 5,814**, not balanced | **yes** |
| symbol rate | 100 kSps | **~199.2 kHz** | **yes** |
| samples/symbol | 4, "upsampled to 2 MSps" (⇒ 20) | **10** | **yes** |
| power normalisation | each frame to unit average power | **not normalised**, 23.3 dB spread between classes | **yes** |
| multipath taps | delays [0, 100, 200] µs | **no cepstral echo detectable at those lags** | unresolved |

The paper's Table 1 is internally inconsistent on its own terms: 100 kSps at 4 samples/symbol
is a 400 kSps stream, and it then says everything was upsampled to a common 2 MSps, which
would give 20 samples/symbol. Neither 4 nor 20 matches the data; 10 does.

Two further notes:

* The paper's stated baseline is **84.3% test accuracy** with their own 1-D CNN on
  `(1024, 2)` input. That number was measured on the contaminated split described in §6 and
  on data carrying the amplitude shortcut described in §4, so **it is not a like-for-like
  target for RadioFry**, whose production contract normalises both away. It has not been
  reproduced here.
* The multipath is **simulated in software** (GNU Radio Channel Model block), as is the
  AWGN that sets SNR. The "real-world" part is the HackRF transmit/receive chain and the
  indoor over-the-air path. This is worth stating precisely: the labelled channel and SNR
  axes are synthetic overlays on a genuine hardware capture.

---

## 3. Sample rate, symbol rate, samples-per-symbol

**Result: Rs ≈ 199,219 Hz, sps = 10.0, at the stored 2 MSps.**

Method: averaged spectrum of `|s|²` over 400 clean 30 dB frames per class, noise floor taken
as the median above 20 kHz.

| class | line | dB over floor | strongest lines found |
|---|---|---|---|
| BPSK | **199,219 Hz** | 33.4 | 199219, 201172, 115234, 117188 |
| QPSK | **199,219 Hz** | 34.0 | 199219, 201172, 400391, 197266 |
| QAM | **199,219 Hz** | 31.0 | 199219, 201172, 29297, 21484 |

The candidate near 100 kHz is **not** a symbol-rate line: it is 10 dB weaker and lands at a
*different* frequency for each class (109,375 / 99,609 / 91,797 Hz), which is the signature of
spurs, not a symbol clock. The 199,219 Hz line lands on the **same FFT bin for all three**
(bin 102 of 1024 at 1953.125 Hz resolution ⇒ 199,219 ± 977 Hz, i.e. 200 kHz), and its second
harmonic is present at 400,391 Hz.

Cross-check by occupied bandwidth, which is an independent observable: BW99 is 308 kHz
(BPSK), 447 kHz (QPSK), 248 kHz (QAM). For RRC α = 0.35, `BW = Rs(1+α)` gives **269 kHz at
Rs = 200 kHz** and only 135 kHz at Rs = 100 kHz. The measurements bracket the 200 kHz
prediction and exclude the 100 kHz one.

**⇒ sps = 2,000,000 / 199,219 = 10.04.**

### Caveat, stated because the table invites misreading

The `|s|²` estimator assumes a **linearly modulated, non-constant-envelope** signal. It is
**not valid** for GMSK (constant envelope), OFDM, NBFM or WBFM. The values it returned for
those four classes (35,156 / 300,781 / 54,688 / 712,891 Hz) are **not symbol-rate estimates
and must not be quoted as such.** Only the BPSK/QPSK/QAM rows are meaningful.

A fourth-power estimator was also run and returned ~37,109 Hz for almost every class,
including the analog ones. That is a common low-frequency artifact, not a cyclic feature, and
the fourth-power column is **discarded as uninformative**.

### Why sps = 10 matters

The frozen V3 checkpoint was trained with `SAMPLES_PER_SYMBOL_SWEEP = (4, 8, 16, 32)`.
**10 was never seen.** It lies between 8 and 16, so this is interpolation rather than
extrapolation, but Entry 040 measured how sharply this architecture degrades off its trained
sps grid. This remains hypothesis #2 for the 0.20% transfer failure and is **not yet
attributed**.

---

## 4. Signal characterisation

### Amplitude: the frames are not power-normalised

Median per-frame average power, 30 dB clean frames:

| class | median power | vs quietest |
|---|---|---|
| NBFM | 0.13988 | +23.3 dB |
| GMSK | 0.09897 | +21.8 dB |
| QPSK | 0.08397 | +21.0 dB |
| BPSK | 0.07420 | +20.5 dB |
| QAM | 0.03476 | +17.2 dB |
| OFDM | 0.00067 | 0.0 dB |
| WBFM | 0.00066 | 0.0 dB |

Across all sampled frames: min 0.00021, max 0.26400 — a **1,244× spread**. PAPR is 2.3–4.3 dB.

This is a **usable shortcut**. Measured directly (§6, L3): log frame power alone predicts the
7-class label at **22.9%** against a 14.3% chance level, from a single scalar.

**RadioFry's production contract normalises per window and therefore cannot use it.** That is
the right behaviour — a classifier keyed on receiver gain does not generalise — but it means
RadioFry is handicapped relative to any model trained on the raw amplitudes, and it is a
further reason the paper's 84.3% is not a like-for-like target.

### Channel condition: real but not as documented — UNRESOLVED

Three tests, and they do not fully agree:

1. **PSD ripple.** A 3-tap channel with delays [0, 200, 400] samples at 2 MSps should produce
   a comb with nulls every 10 kHz. Measured in-band ripple is 0.7–3.8 dB and **chan=0 has
   *more* ripple than chan=1 for all seven classes** — the opposite of the attribute's
   `0=clean`. Differences are small (e.g. BPSK 1.59 vs 1.46 dB) and not decisive.
2. **Cepstrum at the documented lags.** Echo energy at quefrency 200 and 400 samples, relative
   to the local cepstral floor, is **0.54–1.79 for both labels** — i.e. no detectable echo at
   the documented delays in either class. Per-class verdicts disagree (BPSK/QPSK point to
   chan=1, QAM/GMSK to chan=0).
3. **Learned separability.** A logistic regression on a 64-bin log-PSD, trained on the train
   split and scored on the test split *within one modulation at one SNR*, separates the two
   channel labels at **BPSK 77.1%, QPSK 77.4%, QAM 83.1%, GMSK 92.0%** (chance 50%).

**Conclusion:** the channel labels encode a **genuine and learnable physical difference**
(test 3 is unambiguous), but its signature **does not match the documented 100/200 µs tap
delays** (tests 1 and 2), and **the polarity of the labels cannot be confirmed from the
data.** The attribute string `'0=clean, 1=multipath(ref)'` is itself ambiguous, since the
paper's filename convention uses `_ref` for the *clean* reference recording.

**Consequence for reporting:** clean-vs-multipath results must be quoted against the file's
own labels with the caveat that their physical meaning is unverified. Do not claim a
multipath robustness result from this axis without resolving the polarity first.

---

## 5. The SCD cross-check, and why its result was rejected

The project's own cyclostationary estimator (`dsp/spectral_correlation.compute_scd`) was run
as an independent confirmation of the symbol rate. It returned **no dominant cycle
frequencies** for BPSK, QPSK or QAM.

**That result was discarded as an invalid probe, not treated as evidence against sps = 10.**

The reason: the dataset stores 1024-sample frames that are **not contiguous in time**. To get
a record long enough for the estimator, 64 frames were concatenated. That fabricates a
discontinuity every 1024 samples. Cyclostationary estimation is precisely a measurement of
periodic correlation structure, so injecting a strong artificial periodicity at 1024 samples —
and destroying phase continuity across every boundary — makes the input something other than
the signal under test. An empty result from a mis-constructed input says nothing about the
signal.

A valid SCD cross-check would need either per-frame SCD averaged coherently over many frames,
or contiguous data that this release does not provide. **It was not re-run**, because the
`|s|²` evidence is already strong and self-consistent (33 dB line, same bin across three
classes, independently corroborated by occupied bandwidth). Listed in §16 as optional.

This is the second mis-designed probe in this project's history (see BANK Entry 041, where a
`density`-normalised probe was blamed on the estimator). The pattern is worth naming: when a
tool returns nothing, check the input construction before concluding anything about the
signal.

---

## 6. Leakage investigation — complete

The paper states the split protocol explicitly: continuous multi-minute recordings were
segmented into 1024-sample frames, and **the frames were then split train/val/test by
stratified random sampling**. Frames from one physical recording are therefore scattered
across all three splits. The structural inventory confirms the consequence: **all 84
configurations appear in all three splits.**

That is recording-level contamination *by construction*. The remaining question was whether it
is **exploitable**, since a split can be nominally contaminated and still benign.

### L1 — exact duplicate frames: negligible

| pair | identical frames |
|---|---|
| train ∩ test | **2** |
| train ∩ val | **5** |
| val ∩ test | **0** |

Internal: train has 5 duplicate frames among 400,000; val and test have none. Not a problem.

### L2 — first correlation result, and why it was wrong

First measurement: peak normalised cross-correlation between test-split and train-split frames
of the same configuration was **0.94–0.98**, against an assumed chance floor of
`1/√1024 = 0.031`. Read literally that says the splits are near-copies of each other.

**That reading was wrong, and the error was mine: the null was invalid.** Two reasons:

1. **No DC removal.** These frames carry residual local-oscillator leakage. A shared DC
   component correlates every frame with every other frame regardless of content. The
   signature was visible in the data and was the thing that prompted the re-check: WBFM —
   an FM signal with a strong residual carrier — showed a median correlation of **0.60**
   against *unrelated* frames of its own class, and OFDM 0.33. A genuine duplication effect
   does not raise the median that way.
2. **`1/√N` assumes white, full-band frames.** These occupy ~270 kHz of a 2 MHz span, so the
   effective degrees of freedom are ~138, not 1024; and taking a max over ~2048 lags and
   thousands of bank frames adds a large extreme-value inflation on top.

### L2 corrected — the phase-randomised surrogate null

Correct control: keep each bank frame's **magnitude spectrum exactly** and randomise its
phases. The surrogate has identical PSD, bandwidth, DC content and per-frame power, but
unrelated content by construction. DC is removed from both sides first.

**The measured null is ~0.28–0.30, not 0.031.**

Probe: 120 test frames vs a 1,500-frame train bank, clean channel, 30 dB.

| class | observed | surrogate null | cross-modulation | excess |
|---|---|---|---|---|
| BPSK | 0.3944 | 0.2962 | 0.3125 | **+0.098** |
| QPSK | 0.3399 | 0.2878 | 0.2964 | +0.052 |
| **QAM** | **0.7476** | 0.2925 | 0.2937 | **+0.455** |
| GMSK | 0.3084 | 0.2748 | 0.2868 | +0.034 |
| OFDM | 0.1276 | 0.1244 | 0.1493 | +0.003 |
| NBFM | 0.3052 | 0.3020 | 0.2973 | +0.003 |
| WBFM | 0.3170 | 0.3056 | 0.2029 | +0.011 |

### Within-recording duplication — the real, class-dependent finding

Fraction of test frames having a **near-identical partner (|corr| > 0.9)** in the train split,
DC removed, peak over all lags:

| class | test frames with a >0.9 train partner | surrogate |
|---|---|---|
| **QAM** | **40.3% – 48%** | 0.0% |
| **BPSK** | **10.0% – 10.7%** | 0.0% |
| QPSK | 8.0% | 0.0% |
| GMSK | 4.0% – 5.0% | 0.0% |
| NBFM | 0.0% | 0.0% |

(Two figures where two independent samples were taken; both are reported rather than averaged.)

Distribution detail — this is bimodal, not a uniform shift. For QAM: median match 0.8599,
90th percentile 0.9825, max 0.9954, **40.3% above 0.9**; the surrogate for the same frames has
median 0.3013, max 0.3756, **0.0% above 0.9**.

**⇒ For QAM, roughly half the dataset's own test split is a near-duplicate of its training
data. Any QAM accuracy measured on the released split is inflated by an unknown but large
amount.**

### Cross-recording — the result that makes a clean protocol possible

Same measurement, but the bank is drawn from a **different recording**: a different SNR level,
or a different channel condition. Probe: 150 test frames at clean 30 dB vs 1,800 train frames.

| class | same recording | different SNR (20 dB) | different SNR (24 dB) | different channel |
|---|---|---|---|---|
| BPSK | 10.0% | **0.0%** | **0.0%** | **0.0%** |
| QPSK | 8.0% | **0.0%** | **0.0%** | **0.0%** |
| QAM | 48.0% | **0.0%** | **0.0%** | **0.0%** |
| GMSK | 4.0% | **0.0%** | **0.0%** | **0.0%** |
| NBFM | 0.0% | 0.0% | 0.0% | 0.0% |
| OFDM | 28.7% | 28.7% | 28.7% | 28.7% |
| WBFM | 28.7% | 28.7% | 28.7% | 28.7% |

For QAM the median collapses from 0.8862 (same recording, null 0.3061, excess **+0.580**) to
0.3051 across SNR (null 0.3024, excess **+0.003**).

**⇒ For every digital class, duplicated content is confined to a single recording and does not
cross recording boundaries.** This is what makes the §7 protocol sound.

> **A wrong conclusion that was printed and is corrected here.** The script
> `inv7_crossrecording.py` prints a summary line reading *"content DOES repeat across
> recordings; no split of this dataset is clean."* That line is **wrong**. It is driven by a
> mean over all seven classes that is dominated by the OFDM/WBFM 28.7% figure, which the next
> section shows is not cross-recording repetition at all. The per-class table above is the
> correct result. The script is preserved unedited for provenance; **read the table, not the
> summary line.**

### OFDM and WBFM — under-driven recordings, not duplication

The identical 28.7% for OFDM and WBFM against *every* bank — same recording, different SNR,
different channel alike — is a property of the probe frames, not of what they are compared to.

**The first hypothesis was wrong.** Degenerate/near-silent frames were suspected. Measured
participation ratio (effective sample count out of 1024) refutes it: the matching frames have
*higher* participation ratio (OFDM 891 vs 481 for non-matching), and across the whole 80,000
frame test split only **10 frames (0.01%)** have a participation ratio below 50. There are
essentially no degenerate frames.

The actual cause is in the quantisation and level statistics:

| class | mean power | distinct float16 levels (median frame, real part) |
|---|---|---|
| NBFM | 0.108263 | 111 |
| BPSK | 0.044676 | 87 |
| QAM | 0.026234 | 82 |
| GMSK | 0.079751 | 82 |
| QPSK | 0.048521 | 66 |
| **OFDM** | 0.004982 | **6** |
| **WBFM** | 0.018053 | **6** |

And within OFDM and WBFM the frames split into two populations:

| class | matching frames, power | non-matching frames, power | ratio |
|---|---|---|---|
| OFDM | 0.011471 | 0.000640 | **18×** |
| WBFM | 0.063527 | 0.000639 | **99×** |
| BPSK (control) | 0.077252 | 0.075946 | 1.02× |

**⇒ The OFDM and WBFM recordings were captured badly under-driven**, exercising about six ADC
codes for the majority of frames (HackRF One has an 8-bit ADC). The ~29% that were captured at
normal level are the ones that correlate with everything. This is a **data-quality defect in
two of the seven classes**, independent of the leakage question.

Neither class maps to a RadioFry production label, so it does not affect the headline
comparison — but it must be handled explicitly in a 7-class training run (see §16).

### L3 — hardware-artifact shortcut

Can features that carry **no legitimate modulation information** predict the class across
splits? Features: DC offset (I and Q), gain imbalance, quadrature error — all zero for an
ideal signal of any of these modulations — plus log frame power. Multinomial logistic
regression trained on 28,000 train frames, scored on 14,000 test frames. Chance 14.3%.

| features | 7-class accuracy |
|---|---|
| 4 hardware artifacts (DC, gain imbalance, quadrature error) | **23.2%** |
| log frame power alone | **22.9%** |
| all 5 together | **31.4%** |
| all 5, predicting the 84-way *configuration* | 2.2% (chance 1.19%) |

**⇒ A real but modest shortcut: 2.2× chance from five scalars that should carry nothing.**
The dominant leakage channel is the within-recording duplication of §6, not this. RadioFry's
per-window normalisation removes the power component; the DC and imbalance components survive
normalisation and remain available to any model.

---

## 7. Recommended split protocol

**Primary protocol — capture-disjoint by SNR.**

```
TRAIN_SNR_DB   = (20, 24, 28)     # training and validation draw only from these
HELDOUT_SNR_DB = (22, 26, 30)     # evaluation draws only from these
SEALED_SPLIT   = "test"           # subset_test.h5, never read during training
```

| pool | file | SNR | channels | classes | per-config | frames |
|---|---|---|---|---|---|---|
| train | `subset_train.h5` | 20, 24, 28 | 0 and 1 | all 7 | 3,300 | **138,600** |
| validation | `subset_val.h5` | 20, 24, 28 | 0 and 1 | all 7 | 500 | **21,000** |
| held-out test | `subset_test.h5` | 22, 26, 30 | 0 and 1 | all 7 | — | sealed |

Per-configuration minima are **3,853 (train)** and **753 (val)**, so both limits are
achievable for all 84 configurations and **class balance is exact** — which matters because
the released data is not balanced (§1).

**Evaluation design — leakage measured, not assumed.** The same trained model is scored on two
disjoint slices of the sealed test split:

* **`subset_test.h5` at SNR {22, 26, 30}** — different recordings from training.
  **This is the honest generalisation number.**
* **`subset_test.h5` at SNR {20, 24, 28}** — the *same recordings* the model trained on.
  This is the leakage-inflated number, directly comparable to what the dataset's own
  protocol (and the paper's 84.3%) would produce.

**The difference between them is a direct measurement of the leakage inflation**, obtained
with one model, no second training run, and no confound.

---

## 8. Why this protocol is capture-disjoint for the digital classes

Each of the 84 configurations is a **separate physical recording** — the paper's own file
naming is `{MODULATION}_{ref}_{SNR}.dat`, so changing the SNR level changes the file. Holding
out SNR levels {22, 26, 30} therefore holds out **42 entire recordings** (7 modulations × 2
channel conditions × 3 SNR levels), not a random subset of frames from shared recordings.

The measurement that licenses this is the cross-recording table in §6: for **BPSK, QPSK, QAM
and GMSK alike, the near-duplicate rate across a recording boundary is 0.0%**, against 4–48%
within a recording. The contamination mechanism that ruins the released split does not operate
across this boundary.

**Two honest limits on that claim:**

1. It is **file-disjoint, not necessarily session-disjoint.** Different SNR levels of the same
   modulation were plausibly recorded back-to-back in one sitting with unchanged hardware
   state. Slowly varying artifacts (DC offset, LO drift, thermal state) may persist across the
   boundary. The L3 measurement bounds what that is worth: ~31% from artifact features alone,
   well above the 14.3% chance level. So the protocol removes the *large* leakage channel
   (duplicate content) and leaves a *smaller* one (shared hardware state) partially in place.
   It is a substantial improvement, not a guarantee.
2. It costs an SNR-generalisation assumption. The model trains at 20/24/28 dB and is evaluated
   at 22/26/30 dB. Two of those are interpolation; **30 dB is a 2 dB extrapolation** beyond
   the highest trained level. Per-SNR accuracy must be reported so this is visible.

**No claim of "no leakage" is made.** The claim is: the dominant, measured leakage channel is
removed, a smaller one is bounded, and the residual inflation is measured directly by the
two-slice evaluation in §7.

---

## 9. The three planned experiments

The frozen checkpoint scores 95.38% synthetic and 0.20% real (Entry 044). Fine-tuning until
the number improves would raise the score without explaining the gap. **Three runs that differ
only in what is trainable separate the causes.**

| # | mode | trunk (`features.*`) | head (`classifier.*`) | question it answers |
|---|---|---|---|---|
| **E1** | `linear_probe` | V3 weights, **frozen** | new, trained | Do V3's learned features transfer? Was only the head wrong? |
| **E2** | `finetune` | V3 weights, trainable | new, trained | The candidate model. |
| **E3** | `scratch` | random, trainable | new, trained | Control: if this matches E2, V3's initialisation contributed nothing. |

Interpretation, decided in advance:

* **E1 high** ⇒ the convolutional features transfer and the 0.20% was a head/label-space
  problem. That would be the most encouraging outcome and the cheapest fix.
* **E1 low, E2 ≈ E3** ⇒ V3's features do not transfer at all and synthetic pre-training bought
  nothing. The domain gap is in the learned representation, not the classifier.
* **E2 > E3** ⇒ synthetic pre-training is a useful initialisation even though its features are
  not directly usable.

### A precision about "fine-tune from V3"

V3 has **8 output classes**; this dataset has **7**, and they are not the same 7. The head
shape is therefore incompatible and cannot be transferred. `_build_model` loads **only
`features.*`** from the checkpoint and builds a fresh head. E2 is accurately described as
**trunk transfer with a new head**, not as fine-tuning an 8-class model. E1 is a linear probe
on frozen V3 features in the strict sense.

### Label space

Training uses the **dataset's own 7 classes** (`BPSK, QPSK, QAM, GMSK, OFDM, NBFM, WBFM`), not
RadioFry's 8. Reasons: it matches the dataset's own benchmark; and restricting to the 3
mappable classes would make the comparison against V3's 0.20% unfair, since V3 had to choose
among 8 while a 3-class model chooses among 3.

The comparison against the 0.20% baseline is then made on the **restricted view**: frames of
BPSK/QPSK/QAM only, with the new model free to answer any of its 7 classes and V3 free to
answer any of its 8. That is like-for-like.

**`GMSK` is still not mapped to `GFSK`.** Both use a Gaussian pulse at BT = 0.3, but GMSK is a
CPM with h = 0.5 and RadioFry's GFSK is a separate generator configuration. It enters the
7-class training label space under its own name; it does **not** become a RadioFry GFSK label.

---

## 10. Proposed training configuration

From `TrainingConfig` in `src/radiofry/training/train_realworld.py`:

| parameter | value |
|---|---|
| classes | `("BPSK","QPSK","QAM","GMSK","OFDM","NBFM","WBFM")` |
| train SNR | `(20, 24, 28)` |
| per-config train / validation | 3,300 / 500 → 138,600 / 21,000 frames |
| epochs | 25 |
| batch size | 512 |
| optimiser | Adam, lr 1e-3, weight decay 1e-4 |
| scheduler | `ReduceLROnPlateau(factor=0.5, patience=2)` on 1 − val accuracy |
| loss | `CrossEntropyLoss` |
| seed | 20260911 (numpy and torch) |
| model selection | best validation accuracy, validation split only |
| training window | **one random start per frame per epoch** (timing-offset augmentation) |
| evaluation window | the **4 fixed production starts**, mean-softmax |

Random training starts are deliberate: they make the model robust to symbol timing offset and
let 25 epochs see far more distinct windows than 4 fixed positions would. They change nothing
about the inference contract, which is evaluated on the fixed grid.

**These values are a starting point, not a tuned configuration.** Nothing has been tuned,
because tuning requires runs that have not happened. Learning rate, epoch count and per-config
depth are the first things to revisit once E1–E3 have produced a baseline.

---

## 11. Production inference / input contract

The candidate must drop into the existing runtime unchanged, so windows are built to match
`models/modulation_inference.py` exactly:

1. Window starts: `sorted({int(s) for s in np.linspace(0, 1024 - 128, 4)})` = **[0, 298, 597, 896]**.
2. Per window: `values = stack([block.real, block.imag]).astype(float32)`;
   `power = sqrt(mean(values**2))`; `values / power` when `power > 0`, else unchanged.
3. `add_signal_features(..., include_engineered=True)` → 4 channels `iqap`:
   I, Q, amplitude `|c|`, and phase difference `angle(c[1:] · conj(c[:-1]))` with 0 prepended.
4. Per-channel renormalisation: divide each channel by `max(sqrt(mean(ch**2)), 1e-6)`.
5. Frame decision: **mean of the softmax vectors over the 4 windows** (`DEFAULT_INFERENCE_WINDOWS = 4`).

`train_realworld.build_windows` implements this vectorised.
`tests/test_realworld_training.py::test_build_windows_reproduces_the_production_path_exactly`
asserts **element-for-element equality** (`np.array_equal`) against the production path for
every window position, so train/inference skew cannot creep in silently.

Checkpoint payload matches the production schema: `state_dict`, `model_sha256`, `labels`,
`input_channels=4`, `sample_length=128`, `features='iqap'`, plus seed, mode and protocol
fields. A sidecar metrics JSON is written at `metrics_path(output)` as the runtime's integrity
check requires.

**Consequence worth stating:** per-window power normalisation **discards the 23.3 dB amplitude
shortcut** of §4. This is correct behaviour and it is also why RadioFry's number here is not
comparable to a model that keeps it.

---

## 12. Sealed-test protections

`subset_test.h5` is the sealed held-out set. Enforced in code, not by intention:

* `train_realworld._forbid_sealed_split` raises `PermissionError` for `"test"`.
  `load_pool` calls it on every load; reaching the sealed split requires passing
  `allow_sealed=True` explicitly, which the training path never does.
* `train()` refuses an output path named `modulation_cnn_v3_spsaug.pt` with `PermissionError`,
  so the frozen production checkpoint cannot be overwritten even by accident.
* Model selection uses the **validation split only** (`subset_val.h5`), never the test split.
* Thresholds will **not** be tuned on the test split.

Tests covering these: `test_the_sealed_split_cannot_be_loaded_through_the_training_path`,
`test_load_pool_refuses_the_sealed_split`,
`test_training_refuses_to_overwrite_the_production_checkpoint`,
`test_the_training_and_heldout_snr_levels_are_disjoint`.

---

## 13. Reproducibility controls

* **Seeds** fixed for `torch.manual_seed`, `np.random.seed` and the `default_rng` that drives
  batch order and window starts. Default 20260911.
* **Frame selection is content-addressed.** The metrics file records
  `train_indices_sha256` and `validation_indices_sha256` — SHA-256 over the int64 frame
  positions actually used — so a run can be replayed frame for frame, and a claimed
  reproduction can be checked rather than trusted.
* **Selection is deterministic**: `select_indices` samples with a seeded `default_rng` and
  returns sorted indices. `test_selection_is_reproducible_and_stratified` asserts same-seed
  identity and different-seed difference.
* **Stratification is exact**: `per_config_limit` samples independently from each of the 84
  (class, channel, SNR) recordings, so the pool is balanced over class *and* channel *and*
  SNR. Verified achievable in §7.
* The metrics file also records mode, initialisation string, all hyper-parameters, per-epoch
  history, elapsed time, the protocol's SNR sets, and the sealed split name.

---

## 14. The production checkpoint stays frozen

**`models_saved/modulation_cnn_v3_spsaug.pt` must not be modified, retrained or replaced.**

| | |
|---|---|
| `state_dict` sha256 | `a7b02533a7c7129c5435abb7fa12f95fb1af90188115c7c3cb96d9e580ce5a40` |
| file sha256 | `1444cf667fb017a79df5c50489fe5113b8cde4e0c2640f4a0cb8f2741f52530b` |
| labels | 8 digital: `8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK` |
| config | 4-channel `iqap`, 128-sample frames, 4 inference windows |
| synthetic baseline | **95.38%** (Entry 041) |
| real-world baseline | **0.20%** (Entry 044) |

Pinned by `docs/PRODUCTION_FREEZE.md` and `tests/test_production_freeze.py`. Frozen V1 dataset
(`d6d3f918687d0700a43e46211c3f04b9…`) is likewise untouched.

Candidates from these experiments are written to **new filenames** and are **not promoted
automatically**. Promotion is a separate, explicit decision that requires beating the frozen
checkpoint on evidence, not merely producing a number.

Both baselines above remain reproducible: `research_memory/experiments/realworld_2026_09/baseline_entry044.py`
regenerates the 0.20% measurement without training.

---

## 15. Compute: benchmark, and the move to the RTX 5060

Measured on this machine, 2026-09-11:

| | |
|---|---|
| torch | **2.13.0+cpu** — CPU-only build, `torch.cuda.is_available() = False` |
| python | 3.14.5, AMD64 |
| CPU | 24 threads |
| RAM | 16.5 GB total, **2.6 GB available at benchmark time** |
| `ModulationCNN(4, 7)` | 136,135 parameters |
| training throughput | **1,951 windows/s** ⇒ 1M windows/epoch ≈ 513 s |
| inference throughput | **10,409 windows/s** |

Projected cost of the planned protocol on CPU: 138,600 windows/epoch ≈ **71 s/epoch**, so
25 epochs ≈ **30 min per run**, ≈ **1 hour** for E2 + E3 (E1 is much cheaper, since frozen
trunk features can be precomputed once). Feasible but slow, and it rules out the wider sweeps
that would actually be informative.

**Decision: the heavy training moves to the RTX 5060.** Two things must be verified first, and
neither is a formality:

1. **The RTX 5060 is Blackwell (compute capability sm_120).** PyTorch builds that predate the
   CUDA 12.8 toolchain do **not** contain sm_120 kernels; a cu121/cu124 wheel will either
   refuse the device or fall back. A **cu128 or newer** build is required. Verify with
   `torch.cuda.get_arch_list()` containing `sm_120` — not merely
   `torch.cuda.is_available() == True`, which can be true while every kernel launch fails.
2. **Python 3.14 CUDA wheel availability.** The current interpreter is 3.14.5, which is new
   enough that CUDA wheels may not yet be published for it. If not, a 3.12/3.13 environment
   for training is the pragmatic answer.

The training code is device-agnostic in structure but **currently hard-codes CPU tensors**; it
has no `.to(device)` calls and no `pin_memory`/`DataLoader` worker setup. Adding them is a
small, contained change and was deliberately **not** made now, per instruction. Memory is not
a constraint on an 8 GB card: the model is 136k parameters and a 512-window batch of
`(4, 128)` float32 is ~1 MB.

The 2.6 GB free RAM is worth noting as a host-side constraint on pool size regardless of GPU:
the 138,600-frame pool is held as float16 at ~568 MB, which is why `FramePool` stores raw
float16 and converts per batch rather than materialising complex64 (which would be ~1.1 GB).

---

## 16. Open questions and what must be verified before final training

Ordered by how much damage getting them wrong would do.

1. ~~**The new tests have never been run.**~~ **CLOSED (Entry 046).** All 34 run and pass —
   19 training, 15 adapter. Critically,
   `test_build_windows_reproduces_the_production_path_exactly` passes, so the vectorised
   window builder **is** element-for-element identical to
   `modulation_inference._window_frames` + `add_signal_features`. That assertion underpins
   every comparison against V3 and is now verified rather than claimed. Full suite:
   **1309 passed, 1 skipped, 0 failed**.
2. **A live bug found while fixing CI, not yet fixed.**
   `modulation_inference.predict_modulation` verifies the checkpoint with
   `hash_torch_state_dict`, which hashes the bytes `torch.save` produces and so varies with
   the torch version. **On a machine whose torch serialises differently — including after a
   routine `pip install -U torch` — every prediction returns `Unclassified`.** Bypassed under
   pytest and `CI=true`, so no test catches it. See BANK Entry 046; it needs a decision
   because the value inside the frozen `.pt` cannot be rewritten without breaking the file
   hash that now pins the freeze.
3. **Decide how to handle OFDM and WBFM.** Both are under-driven, ~6 ADC codes for ~71% of
   frames (§6). Options: train on all 7 and report the defect; exclude the two and train on 5;
   or keep them but report their metrics separately. **Excluding them changes the task and
   makes the number incomparable to the paper's 84.3%** — this is a judgement call that should
   be made deliberately and recorded, not defaulted into.
4. **QAM's within-recording duplication is 48%.** Even under the capture-disjoint protocol,
   the *training* pool for QAM has high internal redundancy, so its effective diversity is far
   below its frame count. Expect QAM to overfit and check for it explicitly.
5. **Channel label polarity is unresolved** (§4). Do not report a multipath-robustness claim
   from this axis until it is settled. A route: generate a known 3-tap channel synthetically,
   run the same cepstral test, and confirm the test can detect an echo it is given.
6. **The 84.3% paper baseline has not been reproduced** and is not a like-for-like target
   (contaminated split + amplitude shortcut). Do not present RadioFry's number as beating or
   trailing it without that caveat attached.
7. **The SCD cross-check of the symbol rate was never validly run** (§5). Optional — the
   `|s|²` evidence is strong — but a correct per-frame SCD would close it properly.
8. **sps = 10 is still only hypothesis #2** for the 0.20% transfer failure, alongside RRC
   pulse shaping (#1), real hardware impairments (#3) and amplitude scale (#4). None is yet
   attributed. The synthetic V2.1 experiment (add RRC and sps 10 to the generator, re-measure
   the frozen checkpoint) attributes the failure and **requires no new data**; it remains the
   scientifically cleaner next step and is independent of this training work.
9. **GPU toolchain verification** (§15): sm_120 kernel availability and Python 3.14 CUDA
   wheels.
10. **RadioML 2018.01A is still not present** on this machine; that inventory remains
    outstanding.
11. **No result has been produced yet.** E1/E2/E3 have not been run. Nothing in this document
    reports a trained-model accuracy, and none should be inferred from it.
