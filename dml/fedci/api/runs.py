"""Read-only routes over the ResultStore (results/<question>/<run_id>/)."""
from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Response

from fedci.api.export import render_export
from fedci.api.serialize import records
from fedci.config import RESULTS
from fedcore.results import ResultStore

router = APIRouter()

_EXPORT_TABLES = {"residualized_fit": "residuals", "influence": "influence", "coefficients": "estimates"}


@router.get("/runs")
def list_runs() -> dict:
    return {"runs": records(ResultStore(RESULTS).summary())}


@router.get("/runs/{question}/{run_id}")
def get_run(question: str, run_id: str) -> dict:
    for run in ResultStore(RESULTS).runs(question=question.upper()):
        if run.run_id == run_id:
            # manifest entries are store slugs ("residualized_fit" -> "residualized-fit");
            # restore underscores so names match what pipelines saved.
            tables = {name.replace("-", "_"): records(run.table(name))
                      for name in run.manifest.get("tables", [])}
            figures = {
                name.replace("-", "_"): json.loads(
                    (run.path / "figures" / f"{name}.json").read_text(encoding="utf-8"))
                for name in run.manifest.get("figures", [])
            }
            return {"manifest": run.manifest, "tables": tables, "figures": figures}
    raise HTTPException(404, f"run {question}/{run_id} not found")


@router.get("/runs/{question}/{run_id}/export/{figure}")
def export_run_figure(question: str, run_id: str,
                      figure: Literal["residualized_fit", "influence", "coefficients"],
                      format: Literal["png", "svg"] = "png") -> Response:
    """Static seaborn render of one saved run, from its tables -- for slides."""
    for run in ResultStore(RESULTS).runs(question=question.upper()):
        if run.run_id == run_id:
            needed = _EXPORT_TABLES[figure]
            data = render_export(
                figure,
                metrics=run.manifest.get("metrics", {}),
                estimates=records(run.table("estimates")) if needed == "estimates" else [],
                residuals=records(run.table("residuals")) if needed == "residuals" else [],
                influence=records(run.table("influence")) if needed == "influence" else [],
                fmt=format,
                level=float(run.manifest.get("params", {}).get("level", 0.95)),
            )
            return Response(content=data, media_type="image/png" if format == "png" else "image/svg+xml")
    raise HTTPException(404, f"run {question}/{run_id} not found")
