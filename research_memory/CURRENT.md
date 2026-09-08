# Current State

- **Current Project Phase**: V2.0 Dataset generation and CNN training.
- **Current ML/model work**: Multi-window CNN inference (implemented) & 8-class V2.0 model training (pending).
- **Active dataset version**: Synthetic V1 (frozen), preparing V2.0.
- **Latest experiment**: Verified PAM4 and GFSK generation, extending from 6 to 8 classes.
- **Latest results**: Unit pulse area and constant modulus confirmed for GFSK. Exact bit recovery confirmed for PAM4.
- **Current blockers**: Class balance for training (CPFSK/GFSK have 2x captures due to modulation index sweeping).
- **Immediate next steps**: Decide class balance, then launch the 8-class training run and benchmark against V1.
- **Important active decisions**: Dropped 'mean absolute frequency' invariant for Gaussian pulses in favor of true unit area physics.
