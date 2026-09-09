# Knowledge Index

Map of historical entries in `BANK.md`. Use `search_memory.py` to retrieve the full text.

## Datasets & Evaluation
- **Entry 001**: Synthetic Dataset V1, evaluation harness, and baseline findings
- **Entry 015**: V2.0: train a modulation CNN on our own synthetic distribution
- **Entry 016**: V2.0: add PAM4 and GFSK generator support (6 -> 8 classes)
- **Entry 017**: V2.0: train and evaluate the 8-class modulation CNN (current baseline)

## Model Training & Inference
- **Entry 002**: Fix CNN input adapter: window long captures instead of decimating them
- **Entry 008**: Investigation: the QAM classification / routing failure
- **Entry 010**: Multi-window CNN inference: experiment, decision, implementation

## DSP & Parameter Estimation
- **Entry 003**: Investigation: why symbol-rate estimation is 0% accurate on V1
- **Entry 004**: Resolving the RRC caveat: controlled feature comparison on rect vs RRC
- **Entry 005**: Implement adaptive symbol-rate feature selection
- **Entry 014**: Low-SNR FSK symbol-rate estimation: no safe small fix (NEGATIVE RESULT)

## Demodulation
- **Entry 006**: Investigation: why 16QAM and 64QAM still sit at ~0.5 BER
- **Entry 007**: Make QAM demodulation scale-invariant
- **Entry 011**: Investigation: the BFSK failure
- **Entry 012**: Make FSK deviation an explicit swept V1 dimension
- **Entry 013**: FSK-aware timing-offset selection in dispatch
- **Entry 009**: Verification run and commit handoff

## Analog (AM-DSB / AM-SSB / WBFM)
- **Entry 018**: Analog ground-truth design investigation (no implementation)
- **Entry 019**: Analog prerequisite safety fixes - family lookup, Eb/N0 guard, experiment pinning
- **Entry 020**: AM-DSB synthetic generation + independent envelope oracle
- **Entry 021**: AM-DSB through the evaluation harness; BER forced unavailable for analog
- **Entry 022**: AM-SSB synthetic generation + independent sideband oracle (USB default)
- **Entry 024**: WBFM synthetic generation + independent FM oracle (analog registry complete)
- **Entry 025**: Forensic trace - where analog identity is lost (READ-ONLY; detector/fusion/preprocess/dispatch)
- **Entry 026**: Analog-aware fusion fallback (minimal patch; limited benefit, blocker moved upstream)
- **Entry 027**: Positive analog evidence in the classical detector (frequency_cv < 0.9; 0/800 digital FP)
- **Entry 028**: Analog dispatch bypasses symbol-rate timing/decimation (WBFM recovery 0.145 -> 1.000)
- **Entry 029**: AM-SSB carrier estimation - no safe blind fix (NEGATIVE RESULT; needs ~1 Hz, best blind ~500 Hz)
- **Entry 030**: SigMF sidecar centre-frequency ingestion - SSB recovery 0.995-1.000 with real metadata
- **Entry 031**: Analog routing dominance (READ-ONLY) - CNN omits AM-SSB LSB entirely; gate validated 60/60, 0/480 digital FP
- **Entry 032**: Conservative analog subtype routing - classical gate + deterministic rule; 55/60 analog, gate adds 0 digital FP

## System benchmarks
- **Entry 033**: Final synthetic end-to-end benchmark (350 captures) - recommendation TARGETED FIX, not freeze, not retrain
- **Entry 034**: Fusion safety gate - classical digital family blocks CNN analog labels; GFSK->WBFM 27/50 -> 0/50
- **Entry 035**: PAM4 demodulation + dispatch route (new pam_demod.py); BER 0.00000 at 20 dB, 0/25 -> 10/10 demodulated
- **Entry 036**: GFSK demodulation (NEGATIVE RESULT) - the ~0.47 BER is SYMBOL-RATE ESTIMATION, not GFSK, and hits CPFSK equally
- **Entry 037**: FSK symbol-rate estimation (NEGATIVE RESULT) - line is at the noise floor (3-5x vs 17-42x where it works); needs a cyclostationary estimator
- **Entry 038**: Cyclostationary FSK symbol rate - GO for CPFSK (BFSK sps16 BER 0.51->0.018), NO-GO for GFSK; gate 6.0, zero regressions
- **Entry 039**: Final synthetic benchmark (570 captures) - TARGETED BLOCKER: CNN is out-of-distribution at sps 4/32; corrects Entry 033's do-not-retrain
- **Entry 040**: CNN SPS generalisation - SHIP. Fused 35.4% -> 99.5%; window hypothesis REJECTED (128 kept); new v3 digital-only default checkpoint
- Analog captures still do not route to an analog demodulator; the CNN has no analog class.

## Housekeeping
- **Entry 023**: Post-AM-SSB documentation and repository audit
