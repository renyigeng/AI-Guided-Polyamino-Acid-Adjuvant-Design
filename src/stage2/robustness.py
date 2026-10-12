from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .ranking import run_topsis


def _single_axis_winner(
    values: np.ndarray,
    criterion_type: int,
    candidates: list[str],
) -> tuple[str, np.ndarray]:
    values = np.asarray(values, dtype=float)

    if criterion_type == 1:
        ranks = rankdata(-values, method="min")
    else:
        ranks = rankdata(values, method="min")

    winner = candidates[
        int(np.argmin(ranks))
    ]

    return winner, np.asarray(ranks, dtype=float)


def criterion_ablation(
    result: dict[str, Any],
) -> dict[str, Any]:
    matrix = result["matrix"]
    types = result["types"]
    candidates = result["candidates"]
    baseline = result["winner"]
    criteria = result["criteria"]

    if matrix.shape[1] < 2:
        return {
            "stable_count": None,
            "total": None,
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    stable = 0

    for j, criterion in enumerate(criteria):
        keep = [
            k
            for k in range(matrix.shape[1])
            if k != j
        ]

        reduced = matrix[:, keep]
        reduced_types = types[keep]

        if reduced.shape[1] == 1:
            new_winner, ranks = _single_axis_winner(
                reduced[:, 0],
                int(reduced_types[0]),
                candidates,
            )
        else:
            preferences, ranks = run_topsis(
                reduced,
                reduced_types,
            )
            new_winner = candidates[
                int(np.argmax(preferences))
            ]

        same = new_winner == baseline
        stable += int(same)

        rows.append(
            {
                "domain_id": result["domain_id"],
                "omitted_indicator_id":
                    criterion["indicator_id"],
                "omitted_indicator_name":
                    criterion["indicator_name"],
                "baseline_winner": baseline,
                "new_winner": new_winner,
                "winner_changed": not same,
                "baseline_winner_rank_after_omission":
                    float(
                        ranks[
                            candidates.index(baseline)
                        ]
                    ),
            }
        )

    return {
        "stable_count": stable,
        "total": len(criteria),
        "rows": rows,
    }


def weight_perturbation(
    result: dict[str, Any],
    fraction: float = 0.20,
) -> dict[str, Any]:
    matrix = result["matrix"]
    types = result["types"]
    candidates = result["candidates"]
    criteria = result["criteria"]
    baseline = result["winner"]

    if matrix.shape[1] < 2:
        return {
            "stable_count": None,
            "total": None,
            "rows": [],
        }

    n = matrix.shape[1]
    base = np.full(
        n,
        1.0 / n,
        dtype=float,
    )

    rows: list[dict[str, Any]] = []
    stable = 0
    total = 0

    for j, criterion in enumerate(criteria):
        for label, factor in (
            ("down", 1.0 - fraction),
            ("up", 1.0 + fraction),
        ):
            weights = base.copy()
            weights[j] *= factor
            weights /= weights.sum()

            preference, ranks = run_topsis(
                matrix,
                types,
                weights,
            )

            new_winner = candidates[
                int(np.argmax(preference))
            ]

            same = new_winner == baseline

            stable += int(same)
            total += 1

            rows.append(
                {
                    "domain_id": result["domain_id"],
                    "perturbed_indicator_id":
                        criterion["indicator_id"],
                    "perturbed_indicator_name":
                        criterion["indicator_name"],
                    "perturbation": label,
                    "fraction": fraction,
                    "weights": "|".join(
                        f"{value:.8f}"
                        for value in weights
                    ),
                    "baseline_winner": baseline,
                    "new_winner": new_winner,
                    "winner_changed": not same,
                    "baseline_winner_rank_after_perturbation":
                        float(
                            ranks[
                                candidates.index(baseline)
                            ]
                        ),
                }
            )

    return {
        "stable_count": stable,
        "total": total,
        "rows": rows,
    }


def summarize_d1_hydropathy_scales(
    scale_results: pd.DataFrame,
) -> pd.DataFrame:
    """Recalculate the D1 published-scale sensitivity summary."""

    rows: list[dict[str, Any]] = []

    subsets = [
        (
            "all_available_scales",
            scale_results,
        ),
        (
            "moderately_concordant_abs_rho_ge_0.5",
            scale_results[
                scale_results[
                    "spearman_vs_kyte_doolittle"
                ].abs() >= 0.5
            ],
        ),
        (
            "strongly_concordant_abs_rho_ge_0.7",
            scale_results[
                scale_results[
                    "spearman_vs_kyte_doolittle"
                ].abs() >= 0.7
            ],
        ),
    ]

    for name, subset in subsets:
        winner_counts = (
            subset["winner"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        n = int(len(subset))
        f_count = int(winner_counts.get("F", 0))

        rows.append(
            {
                "subset": name,
                "n_scales": n,
                "F_unique_winner_count": f_count,
                "W_unique_winner_count":
                    int(winner_counts.get("W", 0)),
                "Y_unique_winner_count":
                    int(winner_counts.get("Y", 0)),
                "F_unique_winner_fraction":
                    f_count / n if n else float("nan"),
                "most_common_winner":
                    subset["winner"]
                    .astype(str)
                    .value_counts()
                    .index[0]
                    if n
                    else "",
            }
        )

    return pd.DataFrame(rows)
