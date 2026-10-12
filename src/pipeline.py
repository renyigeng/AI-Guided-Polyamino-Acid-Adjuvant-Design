from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .stage1.evidence import load_stage1_evidence
from .stage1.selection import select_functional_block
from .stage2.descriptors import (
    load_stage2_inputs,
    prepare_domain_matrix,
)
from .stage2.ranking import rank_domain
from .stage2.robustness import (
    criterion_ablation,
    summarize_d1_hydropathy_scales,
    weight_perturbation,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _json_ready(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _json_ready(v)
            for v in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def run_pipeline(
    project_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()

    if output_dir is None:
        output_dir = project_root / "results"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    stage1_evidence = load_stage1_evidence(
        project_root
    )

    stage1 = select_functional_block(
        stage1_evidence
    )

    if stage1["selected_functional_block"] != "D-Lys":
        raise ValueError(
            "The frozen Stage-1 evidence no longer reconstructs D-Lys."
        )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    inputs = load_stage2_inputs(
        project_root
    )

    taxonomy = inputs["taxonomy"]
    names = taxonomy["candidate_names_zh"]

    domain_results: dict[str, dict[str, Any]] = {}

    ranking_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []

    for domain in taxonomy["domains"]:
        domain_id = domain["domain_id"]

        problem = prepare_domain_matrix(
            domain_id,
            taxonomy,
            inputs["criteria"],
            inputs["decision_matrix"],
        )

        result = rank_domain(problem)

        ablation = criterion_ablation(result)
        perturbation = weight_perturbation(result)

        result["criterion_ablation"] = ablation
        result["weight_perturbation"] = perturbation

        domain_results[domain_id] = result

        for i, candidate in enumerate(
            result["candidates"]
        ):
            ranking_rows.append(
                {
                    "domain_id": domain_id,
                    "domain_name_zh":
                        result["domain_name_zh"],
                    "candidate": candidate,
                    "candidate_name_zh":
                        names[candidate],
                    "primary_method":
                        result["method"],
                    "primary_score":
                        float(
                            result["primary_scores"][i]
                        ),
                    "primary_rank":
                        float(
                            result["primary_ranks"][i]
                        ),
                    "borda_rank_sum":
                        (
                            float(
                                result[
                                    "borda_rank_sum"
                                ][i]
                            )
                            if "borda_rank_sum" in result
                            else None
                        ),
                    "borda_rank":
                        (
                            float(
                                result[
                                    "borda_ranks"
                                ][i]
                            )
                            if "borda_ranks" in result
                            else None
                        ),
                }
            )

        ablation_rows.extend(
            ablation["rows"]
        )

        perturbation_rows.extend(
            perturbation["rows"]
        )

        robustness_rows.append(
            {
                "domain_id": domain_id,
                "domain_name_zh":
                    result["domain_name_zh"],
                "winner": result["winner"],
                "winner_name_zh":
                    names[result["winner"]],
                "method": result["method"],
                "borda_winner":
                    result.get("borda_winner"),
                "borda_agrees":
                    result.get("borda_agrees"),
                "criterion_ablation_stable":
                    ablation["stable_count"],
                "criterion_ablation_total":
                    ablation["total"],
                "weight_perturbation_stable":
                    perturbation["stable_count"],
                "weight_perturbation_total":
                    perturbation["total"],
                "margin_to_second":
                    float(
                        result["margin_to_second"]
                    ),
            }
        )

    # --------------------------------------------------------
    # Frozen-result regression check
    # This is post-computation validation only.
    # It is NOT an input to candidate selection.
    # --------------------------------------------------------

    frozen_map = {
        str(row["domain_id"]):
            str(row["selected_candidate"])
        for _, row
        in inputs["frozen_final"].iterrows()
    }

    generated_map = {
        domain_id: result["winner"]
        for domain_id, result
        in domain_results.items()
    }

    if generated_map != frozen_map:
        raise ValueError(
            "Generated winners disagree with the frozen validated result. "
            f"generated={generated_map}, frozen={frozen_map}"
        )

    # --------------------------------------------------------
    # D1 published hydropathy-scale sensitivity
    # --------------------------------------------------------

    d1_scale_summary = summarize_d1_hydropathy_scales(
        inputs["d1_scale_results"]
    )

    d1_all = d1_scale_summary.iloc[0]

    # --------------------------------------------------------
    # Competition candidate list
    # No cross-domain total score is created.
    # --------------------------------------------------------

    candidates_out: list[dict[str, Any]] = []

    for domain_id in ("D1", "D2", "D3"):
        result = domain_results[domain_id]
        aa = result["winner"]

        if domain_id == "D1":
            robustness_note = (
                f"F is unique winner on "
                f"{int(d1_all['F_unique_winner_count'])}/"
                f"{int(d1_all['n_scales'])} available "
                f"hydropathy scales; no cross-domain score."
            )
        else:
            ablation = result[
                "criterion_ablation"
            ]
            perturbation = result[
                "weight_perturbation"
            ]

            robustness_note = (
                f"TOPSIS/Borda agree={result['borda_agrees']}; "
                f"criterion ablation="
                f"{ablation['stable_count']}/{ablation['total']}; "
                f"weight perturbation="
                f"{perturbation['stable_count']}/"
                f"{perturbation['total']}."
            )

        candidates_out.append(
            {
                "candidate_id":
                    f"CAND-{domain_id}",
                "track":
                    "疫苗递送载体候选筛选",
                "structure_or_composition":
                    f"C6-{aa}_m-(D-Lys)_n",
                "functional_block":
                    "D-Lys",
                "X_block":
                    aa,
                "X_block_name_zh":
                    names[aa],
                "mechanism_domain":
                    result["domain_name_zh"],
                "selection_method":
                    result["method"],
                "within_domain_priority":
                    1,
                "primary_score":
                    float(
                        result["primary_scores"][
                            result["candidates"].index(
                                aa
                            )
                        ]
                    ),
                "model_pipeline_version":
                    "submission-runtime-v1",
                "robustness_summary":
                    robustness_note,
                "notes":
                    (
                        "Pre-experimental prioritization only; "
                        "m is not predicted; no cross-domain "
                        "global ranking is claimed."
                    ),
            }
        )

    results_df = pd.DataFrame(
        candidates_out
    )

    rankings_df = pd.DataFrame(
        ranking_rows
    )

    robustness_df = pd.DataFrame(
        robustness_rows
    )

    ablation_df = pd.DataFrame(
        ablation_rows
    )

    perturbation_df = pd.DataFrame(
        perturbation_rows
    )

    results_df.to_csv(
        output_dir / "results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rankings_df.to_csv(
        output_dir / "stage2_all_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    robustness_df.to_csv(
        output_dir / "robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    d1_scale_summary.to_csv(
        output_dir / "d1_hydropathy_scale_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ablation_df.to_csv(
        output_dir / "criterion_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    perturbation_df.to_csv(
        output_dir / "weight_perturbation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "runtime_version":
            "submission-runtime-v1",
        "stage1": stage1,
        "stage2_winners": generated_map,
        "frozen_result_regression_check":
            "PASS",
        "cross_domain_total_score":
            False,
        "outputs": [
            "results.csv",
            "stage2_all_rankings.csv",
            "robustness_summary.csv",
            "d1_hydropathy_scale_summary.csv",
            "criterion_ablation.csv",
            "weight_perturbation.csv",
        ],
    }

    with (
        output_dir / "run_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(summary),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return summary
