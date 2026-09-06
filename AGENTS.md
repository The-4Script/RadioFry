# RadioFry Agent Guidance

## Project Shape

RadioFry is a Python 3.11+ RF analysis pipeline with a Streamlit interface.
The runtime path is format-independent after ingestion:

- `src/radiofry/ingestion/` parses WAV and headerless interleaved IQ into `UnifiedSignalContainer`.
- `src/radiofry/dsp/` preprocesses captures and estimates signal parameters and classical modulation families.
- `src/radiofry/models/` performs modulation and bitstream inference using checked-in or user-supplied artifacts.
- `src/radiofry/decoding/` handles demodulation, deinterleaving, and FEC decoding.
- `src/radiofry/fusion/`, `correlation/`, and `reporting/` combine evidence and serialize reports.
- `gui/app.py` is the Streamlit entry point; `gui/pages/` contains the numbered stages and `gui/theme.py` owns shared page/sidebar rendering.

Read [README.md](README.md) for deployment, training, dataset, and GNU Radio details instead of duplicating them here.

## Development Checks

Use the repository's editable environment before running tests:

```powershell
python -m pip install -e ".[dev,ml,gui,fec]"
python -m compileall -q src gui tests
python -m pytest -q
```

The same install, compile check, and test command run in `.github/workflows/ci.yml`. For module commands on Windows, set `$env:PYTHONPATH = "src"`; the test configuration already adds `src` to `sys.path`.

Run the local UI with:

```powershell
$env:PYTHONPATH = "src"
streamlit run gui/app.py --server.headless true
```

When changing Streamlit behavior, exercise the home page and at least the affected numbered page. Preserve `render_page_shell`, `render_sidebar`, and `st.session_state` report/signal handoff unless the navigation contract is intentionally changing.

## Implementation Rules

- Keep ingestion format-specific only in `src/radiofry/ingestion/`; downstream stages consume `UnifiedSignalContainer`.
- Preserve availability and uncertainty in reports. Missing model artifacts or weak/open-set evidence must remain an unavailable or reviewable stage, never a fabricated definitive result.
- Keep public stage keys and dataclass contracts stable unless all callers and tests are updated together.
- Use `report_json()`/the existing JSON-safe conversion for NumPy, complex, and dataclass values rather than ad hoc serialization.
- Keep model training and large datasets out of the inference path and Git; training dependencies are optional and reproducibility details belong in the README.
- Add focused tests under `tests/` for pipeline contracts, graceful degradation, report serialization, and parser edge cases when changing those areas.
- Avoid broad UI rewrites for a stage-specific fix. Shared visual or navigation changes belong in `gui/theme.py` and must be checked across the multipage flow.

## Deployment Reality

Local Streamlit output is not proof that the hosted app has updated. When deployment is part of the request, verify `git status`, the pushed `main` revision, and CI before reporting that a change is live. Do not commit or push unless explicitly asked.