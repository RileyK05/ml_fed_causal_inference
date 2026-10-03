"""FastAPI app: Q1 estimation playground and read-only results browser.

`fedci serve` runs this; the built SPA in dml/frontend/dist is served at "/" when it
exists (npm run build in dml/frontend). Routers are imported inside create_app to keep
the import graph acyclic (fedci.api.figures is used by the Q1 pipeline).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fedci.config import FRONTEND


def create_app() -> FastAPI:
    from fedci.api import q1, runs

    app = FastAPI(title="fedci", description="Q1 estimation playground and results browser")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(q1.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    dist = FRONTEND / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


__all__ = ["create_app"]
