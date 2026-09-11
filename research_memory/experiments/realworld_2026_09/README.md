# Real-world dataset investigation, 2026-09-11 (BANK Entry 045)

Reproduction scripts for the investigation written up in `docs/REALWORLD_DATASET.md`.
**Every script is read-only**: HDF5 files are opened with `h5py` mode `'r'` and nothing is
written to the dataset directory.

Run from the repository root with the dataset present:

```bash
PYTHONPATH=src python research_memory/experiments/realworld_2026_09/inv1_structure.py
```

`ROOT` is hard-coded near the top of each script and points at
`Documents/RadioFry/Real-World IQ Dataset for Automatic Radio Modulati/dataset/`.
Change it there if the dataset moves.

| script | question | headline result |
|---|---|---|
| `inv1_structure.py` | shape, labels, splits, class/SNR/channel distribution, amplitude scale | 560k frames; 84/84 configurations in **all three splits**; frames **not** power-normalised |
| `inv2_signal.py` | symbol rate, channel polarity, amplitude shortcut | `\|s\|²` line at **199,219 Hz ⇒ sps 10**; 23.3 dB power spread between classes |
| `inv3_resolve.py` | is 199 kHz the fundamental or a harmonic? which label is multipath? | fundamental (33 dB, same bin for BPSK/QPSK/QAM); polarity **unresolved** |
| `inv4_leakage.py` | exact duplicates, near-duplicates, artifact shortcut, channel separability | artifact features predict class at **31.4%** (chance 14.3%); channel labels separable at 77–92% |
| `inv5_duplication.py` | characterise the near-duplicate structure | **superseded by `inv6`** — its null is wrong, see below |
| `inv6_duplication_controlled.py` | re-measure against a correct null | true floor is **~0.30, not 0.031**; QAM excess **+0.455**, others ≤ +0.10 |
| `inv7_crossrecording.py` | does duplication cross recording boundaries? | **0.0% for every digital class** — this licenses the split protocol |
| `inv8_degenerate.py` | why do OFDM/WBFM match everything? | not degeneracy: those recordings are **under-driven, ~6 ADC levels** |
| `baseline_entry044.py` | frozen V3 on real data, zero training | **0.20%** (BANK Entry 044) |

Raw outputs are in `measurements/`.

## Two results in here are wrong, and are kept deliberately

The scripts are preserved **unedited**, so the record shows what was actually run. Two of
their outputs must not be quoted:

1. **`inv4_leakage.py` L2 and `inv5_duplication.py`** report correlations of 0.94–0.98
   against an assumed chance floor of `1/√1024 = 0.031`. **That null is invalid** — it omits
   DC removal (these frames carry LO leakage) and assumes white full-band frames when they
   occupy ~270 kHz of 2 MHz. `inv6` establishes the correct floor at ~0.30 with a
   phase-randomised surrogate. **Use `inv6` and `inv7`, not `inv4`/`inv5`, for duplication.**

2. **`inv7_crossrecording.py` prints a summary line reading *"content DOES repeat across
   recordings; no split of this dataset is clean."* That line is wrong.** It is a mean over
   seven classes dominated by the OFDM/WBFM artifact that `inv8` explains. **Read the
   per-class table above it**, which shows 0.0% for every digital class.

Similarly, `inv3_resolve.py` contains an SCD cross-check that returns nothing. That result was
**rejected as a mis-designed probe**, not treated as evidence: it concatenates non-contiguous
frames, fabricating a discontinuity every 1024 samples, which is precisely the structure a
cyclostationary estimator measures. See `docs/REALWORLD_DATASET.md` §5.
