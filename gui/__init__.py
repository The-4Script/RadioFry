"""Streamlit application package for RadioFry.

Importing this package puts the repository's `src/` directory on `sys.path` so that
`radiofry` resolves when the app is launched straight from a checkout with
`python -m streamlit run gui/app.py`.

Streamlit executes only the script being viewed, so a page reached by a direct URL runs
without `gui/app.py` ever having run in that process. Every page imports from `gui`,
which makes this the one place the bootstrap is guaranteed to happen. Installing the
package (`pip install -e .`) also works; this only removes the requirement to have done
so, and never shadows an install that points somewhere else, because the path is only
added when this checkout actually contains `src/`.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if _PROJECT_ROOT.is_dir() and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
