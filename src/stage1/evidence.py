from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required Stage-1 evidence file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stage1_evidence(project_root: Path) -> dict[str, Any]:
    """Load the frozen, auditable Stage-1 decision evidence.

    The competition runtime does not call the LLM or Europe PMC again.
    Raw API/LLM records remain under provenance/ for audit.
    """

    decision_dir = project_root / "data" / "stage1" / "decisions"

    pairwise = load_json(
        decision_dir / "functional_block_stage1_decision.json"
    )
    project_fit = load_json(
        decision_dir / "functional_block_project_fit_decision.json"
    )

    return {
        "pairwise": pairwise,
        "project_fit": project_fit,
    }
