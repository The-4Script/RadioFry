# Signal-DNA: the production CNN's embedding space

| | |
|---|---|
| Core | `src/radiofry/models/embedding.py`, `src/radiofry/models/embedding_dataset.py` |
| Page | `gui/pages/13_signal_dna.py` (stage 13) |
| Tests | `tests/test_signal_dna.py` (37), `tests/test_advanced_pages.py` (6 page tests) |
| Reads | `models_saved/modulation_cnn_v3_spsaug.pt` — **unmodified** |

## 1. Purpose

Show where captures land inside the representation the production CNN actually uses, so
an investigator can see which classes separate, which overlap, and where an unknown
capture sits among them. "Signal-DNA" is the investigator-facing name; technically this is
**a PCA projection of an internal activation of the current production checkpoint**.

## 2. CNN layer selected

`ModulationCNN` is:

```
features   = [Conv 4->64 k8, Conv 64->128 k4, Conv 128->128 k4, AdaptiveAvgPool1d(1)]
classifier = [Flatten, Linear(128->256), ReLU, Dropout(0.5), Linear(256->8)]
```

Two internal representations are offered; neither is the softmax.

| name | tensor | width | why |
|---|---|---|---|
| **`penultimate`** (default) | `classifier[:4](features(x))` | **256** | The vector the final linear layer reads. The logits are an affine map of it, so two classes that overlap here *must* be confusable — no linear boundary separates what overlaps. That direct link to the decision is why it is the default for diagnosing confusions. |
| `pooled` | `flatten(features(x))` | **128** | The convolutional stack's own globally-average-pooled summary, before any classification-specific projection. Less class-specialised. |

`Dropout(0.5)` sits inside `classifier[:4]` but is the identity in `eval()` mode, so the
extracted 256-vector is exactly what `Linear(256->8)` receives.

**The 8-class softmax is deliberately never used.** It is eight numbers constrained to a
simplex; distance in it describes the decision, not the representation, and using it would
make "these classes overlap" circular evidence for "these classes get confused". A test
asserts no layer returns `num_classes` dimensions.

## 3. Mathematical representation

For an input frame `x` (4 x 128, the iqap channels):

```
h = AdaptiveAvgPool1d(ReLU(BN(Conv3(ReLU(BN(Conv2(ReLU(BN(Conv1(x)))))))))) in R^128
e = ReLU(W1 h + b1)                                                        in R^256   <- penultimate
logits = W2 e + b2                                                         in R^8
```

`e` is what is plotted (after PCA). `pooled` plots `h` instead.

## 4. Embedding dimensionality

256 (penultimate, default) or 128 (pooled). Reported on the page as "Embedding width".

## 5. Aggregation

**None is invented.** The model already reduces time itself with `AdaptiveAvgPool1d(1)`,
so both representations are fixed-dimensional for any input length.

Production classifies **4 evenly spaced 128-sample windows** and averages their *softmax*
vectors. This module returns **one record per window**, so averaging stays the caller's
explicit choice. The "Per capture" view uses `aggregate_windows`, the mean of the window
*embeddings* — a different operation from production's mean-softmax, marked by
`window_index = -1` and labelled as such in the UI.

## 6. Preprocessing path

The production functions are **called**, not copied: `modulation_inference._window_frames`
for framing and `signal_features.add_signal_features(include_engineered=True)` for the
iqap channels. Model config (128 samples, 4 channels, iqap) is read from the checkpoint.

Verified: re-deriving softmax from the same forward pass reproduces `predict_modulation`
exactly — label identical and confidence equal to within 1e-6 on every capture tested
(5/5 spot-checked; asserted by `test_the_embedding_path_reproduces_the_production_prediction`).

## 7. Dataset used

| dataset | size | ground truth | SPS | notes |
|---|---|---|---|---|
| **Frozen V1 captures** | 40 captures / 160 windows | yes, from sidecars | 8 only | the byte-frozen on-disk set |
| **Benchmark grid** (default) | 480 captures / 1,920 windows | yes | 4 / 8 / 16 / 32 | regenerated in memory from the Entry 039 seeds (601/607/613) x SNR 20/15/10/5/0 |

The grid is **not a new dataset**. It is produced by the same generator calls that wrote
the frozen files (`generate_source_bits` → `modulate` → `add_awgn` with
`default_rng([seed, 2])`), and regeneration reproduces a frozen V1 capture to
**|corr| = 1.0000** — the residual is the int16 quantisation of the stored file. A test
asserts this. Regenerating is what makes SPS and SNR available as axes at all, since V1 is
entirely at sps 8.

**No train/test leakage concern applies:** nothing here is trained. The embeddings are used
for **visualization and diagnosis only**. No classifier is fitted, nothing is scored
against a held-out set, and the CNN is read, never updated.

## 8. PCA / UMAP methodology

**PCA only**, by SVD on the mean-centred vectors, with the sign of each component fixed by
forcing its largest-magnitude loading positive (the same convention as scikit-learn's
`svd_flip`). Without that, SVD may return either sign and the plot would mirror itself
between runs.

Reported alongside every plot: per-component explained variance, the variance captured by
the three displayed axes, and how many components would be needed for 90%. On the
capture-level grid: **first 3 components explain 78.7%**, and **5 components** reach 90%
(of 256 available) — so the 3-D view is a good but incomplete summary, which is why the
class-geometry table is computed in the **raw** space instead.

The fit is on **the points currently displayed**; changing filters refits. The fitted
transform is reusable (`PCAProjection.transform`) and is what places the loaded capture on
the same axes as the reference cloud.

**UMAP was considered and deliberately not added.** `umap-learn` pulls in numba and
llvmlite, is declared nowhere in this project, and its default embedding is stochastic.
Nothing here needs a nonlinear projection to answer the questions asked.

## 9. Reproducibility

`model.eval()` (dropout → identity, batch-norm uses stored statistics),
`torch.inference_mode()` (no autograd), seeded generation, fixed PCA sign convention.
Repeated extraction is bit-identical; asserted by `test_embeddings_are_deterministic` and
`test_pca_is_deterministic_and_reports_its_variance`.

## 10. Performance

Measured on a laptop, production checkpoint:

| step | time | size |
|---|---|---|
| checkpoint load | 1,106 ms | once per process (`st.cache_resource`) |
| frozen V1, 40 captures | 114 ms | 160 vectors, 0.16 MB |
| benchmark grid, 480 captures | 1,352 ms | 1,920 vectors, 1.97 MB |
| PCA on (480, 256) | 40 ms | |
| PCA on (1920, 256) | 299 ms | |
| peak traced memory | 7.4 MB | |

The checkpoint is loaded once and reused. Embedding sets are cached on
(dataset, layer, window count, checkpoint SHA), so changing a **filter or a colour** never
recomputes embeddings — only the PCA refits.

## 11. What the visualization means

Points close together are captures the CNN represents similarly *at the layer that feeds
its decision*. Classes forming separate clouds are classes the final linear layer can
separate; classes interpenetrating are classes it cannot. The class-geometry table
quantifies that in the raw space: **separation ratio** is centroid distance over the
classes' own mean spread (below ~1 means the classes are closer to each other than they
are internally consistent), and **overlap** is the fraction of captures whose nearest class
centroid is the *other* class.

## 12. What it does NOT mean

- It is **not a classifier** and nothing here is trained.
- PCA is **not a model**: it is a linear change of basis fitted to the displayed points.
- Distance in the 3-D projection is **not physical similarity between signals**. It is
  distance in a 3-D shadow of a 256-D space, and 21% of the variance is not shown.
- Cluster tightness is **not confidence**, and cluster size is not class prevalence.
- The axes (PC1/PC2/PC3) have **no units and no physical meaning**.

## 13. OOD limitations

The page reports **prototype distance**: the Euclidean distance from a capture's embedding
to each class centroid in the raw embedding space, sorted nearest first. That is a
precisely defined quantity and nothing more.

It is **not** an out-of-distribution detector, not a probability, and has not been
validated as either. No threshold is offered, because none has been measured. A large
distance to every centroid is a prompt to look more closely, not a verdict. An OOD score
was deliberately *not* invented for this task.

## 14. Synthetic vs real

**There are no off-air captures in this repository.** `data/` contains only
`synthetic_v1` and `synthetic_v1_shortframe`, both from `radiofry.synthetic_gen.v1`, and
the V1 captures have every impairment disabled (no carrier offset, no timing offset, no
fading, no multipath).

The *mechanism* for real captures is implemented and tested: an uploaded capture is
embedded through the production path, projected with the same fitted transform, drawn
distinctly as a white cross, labelled **`Unknown / unlabelled`**, and given prototype
distances. The CNN's own prediction is shown separately and is **never** promoted into the
ground-truth field — asserted by
`test_an_unlabelled_capture_never_inherits_the_prediction_as_truth`.

But **no claim is made about where real signals fall relative to the synthetic
distribution, because no real signals were available to measure.** That question stays
open until off-air captures exist in the project.

---

# 15. CNN findings from Signal-DNA

Diagnostic only. **No CNN change was made, proposed as implemented, or applied.**

### Correction first: which checkpoint the older findings describe

`reports/benchmark_v039/config.json` records `checkpoint: models_saved/modulation_cnn.pt`
— the **11-class RadioML** model. Production is now `modulation_cnn_v3_spsaug.pt`
(Entry 040). The CNN observations in `docs/FUSION_LANDSCAPE.md` are read from that Entry
039 record and therefore describe the **older checkpoint**, not production.

Re-running the same grid through the production v3 checkpoint:

| | Entry 039 record (`modulation_cnn.pt`) | production v3 (measured here) |
|---|---|---|
| CNN top-1 over 480 | 38.5% | **93.3%** |
| sps 32 top-1 | 7.5% | **90.0%** |
| wrong at confidence ≥ 0.9 | 39 | **0** |
| wrong at confidence ≥ 0.99 | 13 | **0** |

Per class (v3, 60 captures each): BPSK 60, PAM4 60, QPSK 59, 8PSK 58, GFSK 58, CPFSK 54,
QAM16 52, QAM64 47.

### A. Does QAM16/QAM64 overlap explain the confusion? — **Yes, decisively**

| pair | centroid distance | separation ratio | centroid overlap |
|---|---|---|---|
| **QAM16 vs QAM64** | 2.82 | **0.46** | **25.8%** |
| CPFSK vs GFSK | 4.62 | 0.87 | 16.7% |
| QPSK vs 8PSK | 9.52 | 1.50 | 6.7% |
| QAM16 vs PAM4 | 14.18 | 2.59 | 0.0% |
| BPSK vs 8PSK | 13.97 | 2.87 | 0.8% |
| BPSK vs QPSK | 15.53 | 3.09 | 1.7% |

QAM16/QAM64 is the worst pair in the space by a wide margin: the two classes sit closer to
each other than to their own members. 20 of the 32 total errors are QAM64↔QAM16
(13 QAM64→QAM16, 7 QAM16→QAM64), and CPFSK↔GFSK accounts for 8 more — **28 of 32 errors
(87.5%) are these two adjacent pairs**.

### B. Does SPS 32 occupy a distinct representation? — **Shifted, but no longer collapsed**

SPS systematically translates the representation, in order:

```
centroid distance   sps4    sps8   sps16   sps32
        sps4       0.000   1.948   3.454   4.717
        sps8       1.948   0.000   2.115   3.618
       sps16       3.454   2.115   0.000   1.612
       sps32       4.717   3.618   1.612   0.000
```

sps 4 and sps 32 are the extremes (each 2.42 from the global centroid; sps 8 and 16 are
1.31 and 1.06). So SPS is a real axis of variation in the embedding — but v3 handles it:
**90.0% top-1 at sps 32**, against 7.5% for the pre-Entry-040 checkpoint. Entry 040's SPS
augmentation evidently worked.

Within sps 32 the classes are still separable where they are separable generally
(BPSK vs QPSK ratio 4.78, overlap 0.0%) and still collapsed where they collapse generally
(QAM16 vs QAM64 ratio 0.51, overlap 30.0%). **SPS 32 does not have its own failure mode
any more; it amplifies the existing QAM one.**

### C. Where do high-confidence errors occur? — **There are none**

Zero errors at confidence ≥ 0.9 and ≥ 0.99. The full picture:

| | min | median | max |
|---|---|---|---|
| error confidence | 0.285 | 0.473 | **0.616** |
| correct confidence | 0.357 | 0.981 | 1.000 |

Above 0.7 confidence there are **0 wrong out of 376**. All 32 errors occur at SNR 0 dB (24)
or 5 dB (8) — **none above 5 dB**. On this benchmark v3's confidence is well separated,
which is the opposite of the old checkpoint's behaviour.

### D. Are BPSK and QPSK actually separated? — **Yes**

Separation ratio 3.09, overlap 1.7%, and **zero BPSK↔QPSK cross-predictions** in 120
captures. There is no BPSK/QPSK problem to solve, and none is claimed. (This also tempers
the conjugate-SCD work from the previous task: it is correct and validated, but it
addresses a confusion the production CNN does not make.)

### E. Do real captures overlap synthetic clusters? — **Unanswerable here**

No off-air captures exist in the repository (see §14). Not claimed either way.

### F. Should the CNN architecture change? — **No evidence for that**

The representation separates six of eight classes cleanly, is well calibrated, and handles
SPS. The failure is concentrated in two adjacent pairs at low SNR.

### G. Most likely cause of the remaining errors

Ranked by the evidence above:

1. **Feature representation / class adjacency (most likely).** QAM16 and QAM64 differ only
   in constellation density; at 0–5 dB the iqap channels over 128 samples may simply not
   carry enough information to separate them. Same story for CPFSK vs GFSK, which differ
   only in pulse shaping.
2. **SNR robustness.** Every single error is at ≤ 5 dB.
3. **Not architecture**, **not SPS generalization** (fixed by Entry 040), **not confidence
   calibration** (zero confident errors), **not domain shift** (untested — no real data).

---

## 16. Known limitations

- Both datasets are synthetic, from one generator, at one sample rate, with all
  impairments disabled. Nothing here predicts behaviour on off-air signals.
- The 3-D view shows 78.7% of the variance; the class-geometry table is computed in the
  raw space precisely because the projection is incomplete.
- PCA is linear. Classes separable only by a nonlinear boundary would look overlapped.
- Per-capture aggregation is the mean of window embeddings, which is not production's
  mean-softmax reduction.
- Prototype distance is not a validated OOD detector and carries no threshold.
- The grid is regenerated each run rather than stored; that is 1.4 s and keeps the
  repository free of a derived dataset, but it does mean the page depends on the
  generator remaining deterministic (asserted by a test).

## 17. Future possibilities

- Collect genuine off-air captures and answer §14 empirically.
- A QAM16/QAM64-focused study: longer windows, or a channel carrying amplitude-histogram
  information, measured against the 0.46 separation ratio recorded here as the baseline.
- Supervised projection (LDA) as a *second* view, clearly labelled, to show the best
  linear separation rather than the highest-variance one.
- If an OOD indicator is ever wanted, validate one properly against held-out
  distributions before showing a number.
