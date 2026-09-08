# Synthetic RF Dataset Roadmap

## V1 — Baseline Validation

Purpose: prove/test the existing RadioFry prototype and establish baseline performance before major modifications.

Includes:

* BPSK
* QPSK
* 8PSK
* BFSK/FSK
* 16-QAM
* 64-QAM
* Known Fs
* Known symbol rate
* AWGN
* Multiple SNR levels
* Known transmitted bits
* IQ + WAV
* Little/no RF impairments initially
* Full ground truth

Main evaluation:

`classification → parameter estimation → demodulation → BER`

---

# V2 — Robust / Integrated Dataset

V2 is intentionally split into progressive levels rather than generating one enormous dataset immediately.

## V2.0 — Baseline

* BPSK/QPSK/8PSK
* BFSK
* 16-QAM/64-QAM
* Controlled Fs/Rs
* AWGN
* SNR sweep
* No impairments initially
* Known bits
* IQ + WAV

## V2.1 — RF Impairments

Add:

* CFO
* Timing offset/drift
* Phase offset
* Amplitude variation

## V2.2 — Coding

Add:

* FEC
* Interleaving
* Combined end-to-end chain

## V2.3 — Hard Mode

Add:

* Fading
* Multipath
* Interference
* Multiple simultaneous signals
* Burst/intermittent signals
* Missing/incomplete metadata
* Unknown/OOD signals

---

# V2 Ground Truth

Every generated sample should have a ground-truth record containing, where applicable:

* True modulation
* Fs
* Symbol rate
* Bandwidth
* SNR
* Noise type
* CFO
* Timing offset
* Phase offset
* FEC + parameters
* Interleaver + parameters
* Original bits
* Transmitted bits
* Signal/file format

This ground truth is required for objective parameter-validation and BER experiments.

---

# Overall Progression

`V1 → V2.0 → V2.1 → V2.2 → V2.3`

V1 establishes the baseline.

V2.0 establishes the controlled integrated baseline.

V2.1 adds RF impairments.

V2.2 adds coding/interleaving.

V2.3 represents the hard/realistic RF environment.
