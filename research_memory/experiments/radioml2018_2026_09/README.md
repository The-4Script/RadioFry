# RadioML 2018.01A investigation, 2026-09-11 (BANK Entry 047)

Reproduction scripts for `docs/RADIOML2018_DATASET.md`. **Every inspection script is
read-only** — the HDF5 is opened with `h5py` mode `'r'` and nothing is written to the dataset
directory.

Run from the repository root:

```bash
PYTHONPATH=src python research_memory/experiments/radioml2018_2026_09/ds2_01_structure.py
```

`PATH` / `ROOT` is hard-coded near the top of each script and points at
`Documents/RadioFry/dataset 2/`. Change it there if the dataset moves.

| script | question | headline result |
|---|---|---|
| `ds2_01_structure.py` | shape, labels, balance, and **which of the two class orderings is real** | 2,555,904 frames, perfectly balanced; impropriety agrees with the FIXED ordering **24/24** and with the shipped one 10/24 |
| `ds2_02_confirm_and_signal.py` | independent ordering confirmation; symbol rate, bandwidth, amplitude | M-th power moment peaks at **M = 2, 4, 8** for BPSK/QPSK/8PSK; 21.9 dB amplitude spread |
| `ds2_03_leakage.py` | does anything make a random split leak? | **no**: SNR levels independent, frames non-contiguous, near-duplicates **0.0%** |
| `ds2_04_sps_and_mapping.py` | per-frame sps; is 4ASK the same as PAM4? | BW99 tightly at **0.1270 fs** ⇒ sps ≈ 10.6; the PAM4 probe was broken, see below |
| `ds2_run_experiments.py` | the three GPU training runs plus evaluation | see `reports/radioml2018_experiments.json` |

Raw measurement outputs are in `measurements/`.

## Results in here that must NOT be quoted

The scripts are preserved **unedited**, so the record shows what was actually run. Three of
their outputs are wrong or misleading, and the reasoning is in
`docs/RADIOML2018_DATASET.md`:

1. **`ds2_04` section B — the 4ASK vs PAM4 level histograms are meaningless.**
   `level_profile` subtracts the per-frame mean before projecting, which removes exactly the
   DC component that distinguishes a unipolar ASK from a bipolar PAM4. Every class therefore
   came out symmetric with `mean/std = ±0.000`. The question is **unresolved**, and 4ASK is
   left unmapped rather than assumed equivalent to PAM4.

2. **`ds2_04` section A — the per-frame cyclic-line sps values (median 19.7–27.7, range
   13.8–46.7) are estimator variance, not real spread.** Occupied bandwidth over the same
   frames is constant to ±6%, which cannot happen if sps truly varied 3×. Use the bandwidth
   figure (0.1270 fs ⇒ sps ≈ 10.6).

3. **`ds2_01` section 3 — two single-class heuristics misfired** and are reported there as if
   informative. The zero-amplitude-fraction test does not isolate OOK (pulse shaping fills the
   "off" symbols), and FM outranks the AM-SSB classes on spectral asymmetry. Neither
   contradicts the ordering verdict; both are simply uninformative. The ordering rests on the
   24/24 impropriety agreement, the constant-envelope signatures at indices 21 and 22, the
   exact-zero DSB asymmetry at 19/20, and the independent M-th-power confirmation.

## Relationship to Dataset 1

`research_memory/experiments/realworld_2026_09/` holds the equivalent investigation for the
Belousov & Ronkin dataset. The leakage methodology here is the corrected one from that work:
a **phase-randomised surrogate null**, not `1/√N`, which on Dataset 1 inflated an early
correlation reading to 0.94–0.98 before the null was fixed.
