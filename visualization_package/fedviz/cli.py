"""Launch the fedviz app: ``fedviz`` or ``python -m fedviz`` [extra streamlit flags].

Runs Streamlit from the package folder so its ``.streamlit/config.toml`` (the app theme)
applies wherever the command is started.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
APP = PACKAGE_ROOT / "streamlit_app.py"


def main(argv: list[str] | None = None) -> int:
    from streamlit.web import cli as stcli

    os.chdir(PACKAGE_ROOT)
    sys.argv = ["streamlit", "run", str(APP), *(sys.argv[1:] if argv is None else argv)]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
