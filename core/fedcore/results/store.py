"""Results store: every pipeline run is saved as a self-describing folder

    results/<question key>/<run_id>/
        manifest.json     question, name, role, params, metrics, data fingerprints, notes
        tables/*.csv
        figures/*.json    Plotly figures (rendered by the app)

Runs are append-only: nothing overwrites an earlier run, so the full specification
history (docs/question.md: "count the specifications") is always recoverable.
"""
from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from fedcore.data.loaders import fingerprint
from fedcore.questions.base import QuestionSpec

ROLES = ("primary", "robustness", "exploratory")


@dataclass(frozen=True)
class Run:
    path: Path
    manifest: dict

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path / "tables" / f"{name}.csv")

    def figure(self, name: str):
        import plotly.io as pio
        return pio.from_json((self.path / "figures" / f"{name}.json").read_text(encoding="utf-8"))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "run"


class ResultStore:
    def __init__(self, root: Path):
        """root: the owning project's results folder (e.g. dml/results). Each project keeps its own."""
        self.root = Path(root)

    def save(
        self,
        spec: QuestionSpec,
        name: str,
        *,
        role: str = "exploratory",
        params: dict | None = None,
        metrics: dict | None = None,
        datasets: list[str] | None = None,
        tables: dict[str, pd.DataFrame] | None = None,
        figures: dict | None = None,
        notes: str = "",
    ) -> Run:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        now = datetime.now()
        run_id = f"{now:%Y%m%d-%H%M%S}_{_slug(name)}"
        path = self.root / spec.key / run_id
        n = 1
        while path.exists():  # same name saved twice within one second
            n += 1
            path = self.root / spec.key / f"{run_id}-{n}"
        run_id = path.name
        (path / "tables").mkdir(parents=True)
        (path / "figures").mkdir()

        for tname, df in (tables or {}).items():
            df.to_csv(path / "tables" / f"{_slug(tname)}.csv", index=False)
        for fname, fig in (figures or {}).items():
            (path / "figures" / f"{_slug(fname)}.json").write_text(fig.to_json(), encoding="utf-8")

        manifest = dict(
            question=spec.id, question_key=spec.key, name=name, run_id=run_id, role=role,
            created=now.isoformat(timespec="seconds"),
            params=params or {}, metrics=metrics or {},
            datasets={d: fingerprint(d) for d in (datasets or [])},
            tables=[_slug(t) for t in (tables or {})],
            figures=[_slug(f) for f in (figures or {})],
            notes=notes,
            env=dict(python=platform.python_version(), pandas=pd.__version__),
        )
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return Run(path, manifest)

    def runs(self, question: str | None = None) -> list[Run]:
        """All runs, newest first. question filters by id ("Q1")."""
        out = []
        for mf in self.root.glob("*/*/manifest.json"):
            m = json.loads(mf.read_text(encoding="utf-8"))
            if question is None or m["question"] == question.upper():
                out.append(Run(mf.parent, m))
        return sorted(out, key=lambda r: r.manifest["created"], reverse=True)

    def summary(self, question: str | None = None) -> pd.DataFrame:
        rows = [
            dict(question=r.manifest["question"], run_id=r.run_id, name=r.manifest["name"],
                 role=r.manifest["role"], created=r.manifest["created"], **{f"m.{k}": v for k, v in r.manifest["metrics"].items()})
            for r in self.runs(question)
        ]
        return pd.DataFrame(rows)
