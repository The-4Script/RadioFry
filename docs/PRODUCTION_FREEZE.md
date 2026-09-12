# Production freeze — 2026-09-11

The backend is frozen here, before any real-world data touches it, so that every later
real-data result is measured against a known and unchanging system. Nothing in this file
is a quality claim; it is an **identity record**.

Enforced by `tests/test_production_freeze.py` (7 tests). If one fails, either something
drifted or the freeze needs deliberately re-cutting with a BANK entry recording why.

## Frozen identities

| | |
|---|---|
| Production checkpoint | `models_saved/modulation_cnn_v3_spsaug.pt` |
| Checkpoint file SHA-256 | `1444cf667fb017a79df5c50489fe5113b8cde4e0c2640f4a0cb8f2741f52530b` |
| **Weights** SHA-256 (`hash_state_dict_contents`) | `65bb179501f6cbea2cadaa0a64b391e452f930115c38e4e513b7a28c65ed079b` |
| `state_dict` SHA-256, legacy (`hash_torch_state_dict`) | `a7b02533a7c7129c5435abb7fa12f95fb1af90188115c7c3cb96d9e580ce5a40` — traceability only |
| Labels (8, digital only) | 8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK |
| Input representation | 4-channel `iqap`, 128-sample frames |
| Inference | 4 windows, mean-softmax (`DEFAULT_INFERENCE_WINDOWS = 4`) |
| Frozen V1 SHA-256 | `d6d3f918687d0700a43e46211c3f04b9ef74be9d8b232eac5d0e4cf4bf2390ab` (40 `.iq`) |
| Git HEAD at freeze | `c467d30` (working tree carries uncommitted Entries 041–044) |

> ### The legacy `a7b02533…` hash is environment-specific (Entry 046)
>
> `hash_torch_state_dict` hashes the bytes **`torch.save` produces**, so its result depends
> on the torch version doing the serialising, not only on the weights. CI computed
> `0365780e…` from a checkout that was **byte-identical** to the local file, and the freeze
> test failed for that reason alone.
>
> The freeze is now asserted on two identities that hold on any machine: the **file**
> SHA-256 and `hash_state_dict_contents`, which hashes sorted parameter names with their
> dtype, shape and raw little-endian bytes. `a7b02533…` is kept for traceability because
> BANK quotes it throughout, but it is **no longer asserted**.
>
> **Fixed:** inference and training now verify and publish the portable content hash.
> The environment-based pytest/CI bypass was removed. The legacy value remains only for
> historical traceability; new manifests use the portable weights hash above.

Weights, labels **and** inference configuration are all pinned: the recorded baselines were
measured with a 128-sample frame and 4 windows, so changing either invalidates them even if
the weights match.

## Baselines this freeze preserves

All synthetic-to-synthetic, all on seeds disjoint from training.

| measurement | value | source |
|---|---|---|
| CNN top-1, 800 unseen-seed captures | **95.38%** | Entry 041 |
| by SNR | 100% at 20/15/10 dB; 93.8% at 5 dB; 83.1% at 0 dB | Entry 041 |
| by samples-per-symbol | 93.5 / 97.5 / 95.0 / 95.5% at 4/8/16/32 | Entry 041 |
| confidently wrong | **0** above 0.683 confidence | Entry 041 |
| end-to-end BER, sps 8 @ 20 dB | 7 of 8 classes at 0.0000–0.027 | Entry 043 |
| test suite | **1264 passed, 0 failed, 1 skipped** | Entry 044 |

The one end-to-end failure is GFSK, attributed to symbol-rate estimation (Entries 037/038,
closed), not to the demodulator.

## Environment requirement

The `fec` extra is **required** for FEC decoding:

```bash
pip install -e ".[fec]"
```

Without it, `reed_solomon`, `convolutional` and `concatenated` return an honest
`success=False` instead of decoding. `requirements.txt` (`-e .[ml,gui,fec]`) already
includes it. Since Entry 044 the condition is reported in every analysis at
`report["stages"]["runtime"]["fec_support"]`, so a degraded deployment is visible rather
than silent.

## What is deliberately NOT frozen

Investigator-facing analysis pages (SCD, capability surface, fusion landscape, Signal-DNA,
Time Machine) sit beside the pipeline and can change without invalidating these baselines —
none of them feeds a production decision.

## Re-cutting the freeze

Only after a deliberate, recorded change:

1. Record the change and its measurements in a new BANK entry.
2. Re-measure the baselines above with the same protocol.
3. Update the hashes in `tests/test_production_freeze.py` and this file together.
4. Update `research_memory/CURRENT.md`.

Do not update the hashes to make a failing test pass.
