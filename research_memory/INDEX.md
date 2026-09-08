# Knowledge Index

Map of historical entries in `BANK.md`. Use `search_memory.py` to retrieve the full text.

## Datasets & Evaluation
- **Entry 001**: Synthetic Dataset V1, evaluation harness, and baseline findings
- **Entry 015**: V2.0: train a modulation CNN on our own synthetic distribution
- **Entry 016**: V2.0: add PAM4 and GFSK generator support (6 -> 8 classes)

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
