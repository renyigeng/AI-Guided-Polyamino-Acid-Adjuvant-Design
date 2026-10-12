from __future__ import annotations

from typing import Any


def _resolved_pairwise_scores(
    pairwise_results: list[dict[str, Any]],
) -> dict[str, int]:
    candidates: set[str] = set()

    for row in pairwise_results:
        candidates.update(str(x) for x in row["pair"])

    scores = {candidate: 0 for candidate in candidates}

    for row in pairwise_results:
        if row.get("status") != "resolved":
            continue

        winner = row.get("winner")
        if winner in scores:
            scores[str(winner)] += 1

    return scores


def select_functional_block(evidence: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the frozen Stage-1 decision without calling an LLM.

    Logic:
    1. General K/R/H pairwise ICL gives K/R unresolved and H weaker.
    2. Target-specific project-fit comparison resolves K vs R.
    3. Only after K is selected is D-vs-L lysine chirality resolved.
    """

    general = evidence["pairwise"]
    project_fit = evidence["project_fit"]

    pairwise_results = list(general["pairwise_results"])
    scores = _resolved_pairwise_scores(pairwise_results)

    if not scores:
        raise ValueError("Stage-1 pairwise evidence contains no candidates.")

    max_score = max(scores.values())
    general_frontier = sorted(
        candidate
        for candidate, score in scores.items()
        if score == max_score
    )

    # The general literature stage is deliberately allowed to remain unresolved.
    kr_row = next(
        (
            row
            for row in pairwise_results
            if set(row["pair"]) == {"K", "R"}
        ),
        None,
    )

    if kr_row is None:
        raise ValueError("K-vs-R pairwise comparison is missing.")

    project_identity = str(project_fit["project_specific_identity"])
    project_verdict = str(project_fit["literature_project_fit_verdict"])
    selection_basis = str(project_fit["selection_basis"])

    if project_identity != project_verdict:
        raise ValueError(
            "Project-fit identity and literature project-fit verdict disagree."
        )

    if project_identity not in {"K", "R", "H"}:
        raise ValueError(f"Unexpected Stage-1 identity: {project_identity}")

    chirality = project_fit["lys_chirality_result"]

    if project_identity != "K":
        raise ValueError(
            "Frozen chirality comparison is specific to lysine, "
            "but the reconstructed identity is not K."
        )

    if chirality.get("status") != "resolved":
        raise ValueError("D-vs-L lysine chirality is unresolved.")

    chirality_winner = str(chirality["winner"])

    if chirality_winner not in {"D", "L"}:
        raise ValueError(
            f"Unexpected lysine chirality winner: {chirality_winner}"
        )

    selected = f"{chirality_winner}-Lys"

    frozen_selected = str(project_fit["selected_functional_block"])
    if selected != frozen_selected:
        raise ValueError(
            "Reconstructed Stage-1 result disagrees with frozen decision: "
            f"{selected} != {frozen_selected}"
        )

    return {
        "general_pairwise_scores": scores,
        "general_frontier": general_frontier,
        "kr_general_status": kr_row.get("status"),
        "project_specific_identity": project_identity,
        "selection_basis": selection_basis,
        "chirality_votes": chirality.get("votes", {}),
        "chirality_winner": chirality_winner,
        "selected_functional_block": selected,
    }
