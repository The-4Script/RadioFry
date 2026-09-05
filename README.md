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
RML2016.10a is already supported at `data/RML2016.10a_dict.pkl`. RML2018.01A
is also supported directly in its original HDF5 form; place the extracted file
at `data/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5` (or pass another `.h5` path to `--data`). Do not
convert it to pickle: the trainer uses bounded stratified reads so a 2-million
example archive does not need to be loaded entirely into RAM.

For NVIDIA GPU training on Windows, install the CUDA wheel after the base
requirements because the default PyPI Torch package may be CPU-only:

```powershell
& .venv/Scripts/python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
$env:PYTHONPATH = "src"
& .venv/Scripts/python.exe -m radiofry.training.train_modulation --device cuda
```

For RML2018 on the RTX 3050, start with a bounded, reproducible run and inspect
its metrics before increasing the sample count:

```powershell
$env:PYTHONPATH = "src"
& .venv/Scripts/python.exe -m radiofry.training.train_modulation --data data/RML2018.01A.h5 --output models_saved/modulation_cnn_rml2018.pt --max-samples 100000 --epochs 30 --batch-size 64 --device cuda
```

The RML2018 HDF5 loader expects the standard `X`, `Y`, and `Z` datasets: signal
data, modulation labels (one-hot or categorical), and SNR. If the archive
contains class-name metadata it is preserved; otherwise class indices remain
explicitly named `0`, `1`, and so on rather than being guessed.

Training output contains learned weights and biases in the `.pt` checkpoint;
the optimizer state and raw training data are not required for inference. The
trainer replaces the target checkpoint atomically only when validation loss
improves, so an interrupted run leaves the previous best model usable. Keep
the dataset locally for reproducibility and auditability, but it is excluded
from Git and is not needed by the hosted inference app.

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

## What to upload for a meaningful test

For modulation, symbol-rate, interleaver, and FEC analysis, upload a baseband
RF capture rather than an ordinary music recording. The best test file is a
short WAV containing complex I/Q in two channels, or a headerless interleaved
IQ file with documented dtype, byte order, sample rate, modulation, and (when
available) symbol rate. A 5-30 second capture is enough for a demo.

The repository's synthetic generators and the RML2016.10a benchmark are the
recommended reproducible sources. Public GNU Radio examples and documented
SDR datasets are also suitable when they preserve baseband samples. Normal
songs may be uploaded to test WAV ingestion, waveform, spectrogram, and the
open-set behavior, but the CNN was not trained to interpret music as an RF
modulation class, so its label must not be presented as a meaningful radio
identification.

Historical NASA/Voyager audio files can be used as an audio visualization
demo, but a rendered/sonified recording is not equivalent to raw spacecraft RF
IQ and cannot support defensible FEC or interleaver claims. Use it only when
the source documentation confirms the sample format and recording chain.

The modulation CNN is currently trained on the 11 RML2016.10a classes listed in
the training metrics. Signals outside that distribution are handled with
confidence/family disagreement and open-set warnings; this system does not
truthfully claim perfect classification of every possible signal.

## Current pipeline boundary

`src/ingestion` converts WAV and headerless interleaved IQ into a common
`UnifiedSignalContainer`. `src/dsp` and later pipeline stages consume that
container without format-specific branches.
