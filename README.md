# RadioFry

RadioFry is a hardware-agnostic pipeline for analyzing WAV and raw IQ captures,
estimating signal parameters, classifying modulation, and reporting decoding
evidence. The implementation follows the SIH26147 architecture.

## Development setup

The documented runtime is WSL2 Ubuntu with Python 3.14. Create an environment
and install the core plus development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Optional ML, GUI, and FEC dependencies can be installed with `.[ml,gui,fec]`.
The RML2016.10a dataset is intentionally not bundled. Place it at
`data/RML2016.10a_dict.pkl` before running the modulation training command.

For NVIDIA GPU training on Windows, install the CUDA wheel after the base
requirements because the default PyPI Torch package may be CPU-only:

```powershell
& .venv/Scripts/python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
$env:PYTHONPATH = "src"
& .venv/Scripts/python.exe -m radiofry.training.train_modulation --device cuda
```

The trainer selects CUDA automatically when available and falls back to CPU;
`--device cuda` makes a missing CUDA installation fail explicitly.

The synthetic classical classifiers can be trained and evaluated directly from
their manifests. Each command writes the pickle model and a companion JSON
file containing held-out accuracy, five-fold cross-validation, confusion data,
and feature importances:

```bash
PYTHONPATH=src python -m radiofry.training.train_interleaver
PYTHONPATH=src python -m radiofry.training.train_fec
```

Regenerate the synthetic manifests before training if the checked-in corpus is
missing an architecture class; the generator is the source of truth for the
five interleaver labels and five FEC labels.

## WSL2 GNU Radio workflow

The repository is available inside Ubuntu at `/mnt/c/RadioFry`. GNU Radio
3.10.12 is installed system-wide there, so run the waveform adapter with the
WSL Python environment rather than the Windows interpreter:

```bash
wsl
cd /mnt/c/RadioFry
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install -e '.[dev,ml,gui,fec]'
python -c 'import gnuradio; print(gnuradio.__version__)'
PYTHONPATH=src python -c 'import numpy as np; from radiofry.synthetic_gen.gnuradio_chain import generate_bpsk_gnuradio; print(generate_bpsk_gnuradio(np.array([0,1,1,0], dtype=np.uint8), samples_per_symbol=8, snr_db=12).shape)'
PYTHONPATH=src streamlit run gui/app.py
```

The GNU Radio adapter currently generates channel-impaired BPSK baseband
samples. Use the NumPy generator for Windows-only execution; use WSL for the
GNU Radio path and future GNU Radio digital/channel blocks.

## Free-tier hosting

The current Streamlit GUI is deployable on [Streamlit Community Cloud](https://streamlit.io/cloud), which provides a free CPU tier for public GitHub repositories. Connect the repository, select `gui/app.py` as the main file, and set the Python version to 3.11 or newer. The tracked inference artifacts in `models_saved/` are sufficient for the hosted demo; do not upload the 640 MB RML training dataset.

Hugging Face Spaces free dynamic hosting is not used for this MVP because free compute is intended for Gradio ZeroGPU, while this application is Streamlit. Migrating to Gradio would be a separate deployment project and would add risk without improving the current demo.

For a local deployment smoke test:

```bash
PYTHONPATH=src streamlit run gui/app.py --server.headless true
```

## Current pipeline boundary

`src/ingestion` converts WAV and headerless interleaved IQ into a common
`UnifiedSignalContainer`. `src/dsp` and later pipeline stages consume that
container without format-specific branches.
