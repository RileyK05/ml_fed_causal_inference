"""Render every page of the viewer headlessly; fails on any uncaught exception."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "frontend" / "streamlit"
# `streamlit run src/frontend/streamlit/Home.py` puts src/frontend/streamlit/ on the path
# so pages can import _common; mirror that
# here so each page is tested on its own, independent of test order.
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders(page):
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
