from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stage2_inputs(project_root: Path) -> dict[str, Any]:
    """Load frozen mature-descriptor inputs used by the deterministic ranking.

    These files were generated upstream from AAindex, RDKit and peptides.py.
    Their raw-generation provenance is preserved separately.
    """

    frozen = project_root / "data" / "stage2" / "frozen"

    taxonomy = load_json(
        frozen / "discovery" / "domain_taxonomy.json"
    )

    criteria = pd.read_csv(
        frozen / "ranking" / "criteria_used.csv"
    )

    matrix = pd.read_csv(
        frozen / "ranking" / "decision_matrix_wide.csv"
    )

    d1_scale_results = pd.read_csv(
        frozen
        / "d1_hydropathy_audit"
        / "hydropathy_scale_results.csv"
    )

    frozen_final = pd.read_csv(
        frozen / "final" / "final_candidates.csv"
    )

    validate_taxonomy_and_matrix(taxonomy, matrix)

    return {
        "taxonomy": taxonomy,
        "criteria": criteria,
        "decision_matrix": matrix,
        "d1_scale_results": d1_scale_results,
        "frozen_final": frozen_final,
    }


def validate_taxonomy_and_matrix(
    taxonomy: dict[str, Any],
    matrix: pd.DataFrame,
) -> None:
    expected: list[str] = []

    for domain in taxonomy["domains"]:
        expected.extend(str(x) for x in domain["candidates"])

    if len(expected) != 17 or len(set(expected)) != 17:
        raise ValueError(
            "Frozen taxonomy must contain exactly 17 unique candidates."
        )

    observed = matrix["candidate"].astype(str).tolist()

    if set(observed) != set(expected):
        raise ValueError(
            "Decision matrix candidates do not match frozen taxonomy."
        )


def transform_values(
    raw: np.ndarray,
    transform: str,
) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)

    if np.isnan(raw).any():
        raise ValueError("Ranking criterion contains NaN values.")

    if transform in {"raw_benefit", "raw_cost"}:
        return raw

    if transform == "absolute_deviation_from_zero_cost":
        return np.abs(raw)

    if transform == "binary_presence_benefit":
        return (raw > 0).astype(float)

    if transform == "binary_presence_cost":
        return (raw > 0).astype(float)

    raise ValueError(f"Unsupported ranking transform: {transform}")


def prepare_domain_matrix(
    domain_id: str,
    taxonomy: dict[str, Any],
    criteria_df: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> dict[str, Any]:
    domain = next(
        item
        for item in taxonomy["domains"]
        if item["domain_id"] == domain_id
    )

    candidates = [str(x) for x in domain["candidates"]]
    names = taxonomy["candidate_names_zh"]

    rows = (
        decision_matrix[
            decision_matrix["domain_id"].astype(str) == domain_id
        ]
        .copy()
        .set_index("candidate")
        .loc[candidates]
    )

    cdf = (
        criteria_df[
            criteria_df["domain_id"].astype(str) == domain_id
        ]
        .copy()
        .sort_values("indicator_id")
    )

    if cdf.empty:
        raise ValueError(f"No qualified criteria found for {domain_id}.")

    columns: list[np.ndarray] = []
    types: list[int] = []
    criteria: list[dict[str, Any]] = []

    for _, criterion in cdf.iterrows():
        indicator_id = str(criterion["indicator_id"])
        registry_key = str(criterion["registry_key"])
        column_name = f"{indicator_id}__{registry_key}"

        if column_name not in rows.columns:
            raise ValueError(
                f"Decision-matrix column missing: {column_name}"
            )

        raw = rows[column_name].to_numpy(dtype=float)
        transform = str(criterion["transform"])
        operational = transform_values(raw, transform)

        columns.append(operational)
        types.append(int(criterion["criterion_type"]))

        criteria.append(
            {
                "indicator_id": indicator_id,
                "indicator_name": str(criterion["indicator_name"]),
                "registry_key": registry_key,
                "transform": transform,
                "criterion_type": int(
                    criterion["criterion_type"]
                ),
                "equal_weight": float(
                    criterion["equal_weight"]
                ),
                "raw_values": raw,
                "operational_values": operational,
            }
        )

    operational_matrix = np.column_stack(columns).astype(float)

    return {
        "domain_id": domain_id,
        "domain_name_zh": domain["name_zh"],
        "candidates": candidates,
        "candidate_names_zh": {
            aa: names[aa] for aa in candidates
        },
        "criteria": criteria,
        "matrix": operational_matrix,
        "types": np.asarray(types, dtype=int),
    }
