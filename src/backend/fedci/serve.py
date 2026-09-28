"""One-command launch: build the web app when needed, then serve API + SPA.

`fedci serve` is the whole launch (start.cmd at the repo root wraps it for double-click).
The SPA in src/frontend/dist is rebuilt with npm only when it is missing or older than
the frontend sources, so a normal launch is a no-op build-wise. If the port already
answers, the existing server is reused and the browser is just pointed at it.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

from fedci.config import ROOT

WEB_DIR = ROOT / "src" / "frontend"
WEB_DIST = WEB_DIR / "dist"

BUILD_HINT = ("the web app was not built; run `npm install && npm run build` in src/frontend "
              "(needs Node.js), or pass --no-build to serve the API only")


def web_sources() -> list[Path]:
    """Frontend inputs whose age decides whether dist/ is stale."""
    src = WEB_DIR / "src"
    files = [WEB_DIR / n for n in ("index.html", "package.json", "vite.config.ts", "tsconfig.json")]
    if src.exists():
        files += [p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return [p for p in files if p.exists()]


def _newest(paths) -> float:
    return max((p.stat().st_mtime for p in paths), default=0.0)


def needs_build(force: bool = False) -> bool:
    if force or not (WEB_DIST / "index.html").exists():
        return True
    return _newest(web_sources()) > _newest(WEB_DIST.rglob("*"))


def build_web() -> None:
    """npm install (if needed) + npm run build. Raises RuntimeError with instructions."""
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(BUILD_HINT)
    try:
        if not (WEB_DIR / "node_modules").exists():
            print("installing web app dependencies (npm install) ...", flush=True)
            subprocess.run([npm, "install"], cwd=WEB_DIR, check=True)
        print("building web app (npm run build) ...", flush=True)
        subprocess.run([npm, "run", "build"], cwd=WEB_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"web build failed ({exc}); run it manually in src/frontend") from exc


def port_in_use(host: str, port: int, timeout: float = 1.0) -> bool:
    check = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((check, port)) == 0


def _is_fedci(host: str, port: int) -> bool:
    """True when the process on the port actually answers as a fedci API."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/q1/config", timeout=1.5) as r:
            return "surprises" in json.load(r)
    except Exception:
        return False


def _open_when_ready(url: str, host: str, port: int, timeout: float = 20.0) -> None:
    def wait() -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if port_in_use(host, port, timeout=0.5):
                webbrowser.open(url)
                return
            time.sleep(0.3)
    threading.Thread(target=wait, daemon=True).start()


def launch(*, host: str = "127.0.0.1", port: int = 8000, build: bool = True,
           open_browser: bool = True, rebuild: bool = False) -> int:
    check_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{check_host}:{port}/"

    if port_in_use(check_host, port):
        if _is_fedci(check_host, port):
            print(f"fedci is already running at {url}", flush=True)
            if open_browser:
                webbrowser.open(url)
            return 0
        print(f"port {port} is in use by something else; try `fedci serve --port 8001`", flush=True)
        return 1

    if build:
        try:
            if needs_build(force=rebuild):
                build_web()
        except RuntimeError as exc:
            print(f"warning: {exc}", flush=True)
    elif not (WEB_DIST / "index.html").exists():
        print(f"note: {BUILD_HINT}", flush=True)

    print(f"serving fedci on {url}   (Ctrl+C to stop)", flush=True)
    if open_browser:
        _open_when_ready(url, check_host, port)

    import uvicorn

    from fedci.api import create_app
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    return 0


__all__ = ["launch", "needs_build", "build_web", "port_in_use", "web_sources"]

