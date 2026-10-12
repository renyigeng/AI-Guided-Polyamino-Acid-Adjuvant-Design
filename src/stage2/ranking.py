from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from pymcdm.methods import TOPSIS
from scipy.stats import rankdata


def run_topsis(
    matrix: np.ndarray,
    types: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    types = np.asarray(types, dtype=int)

    n_criteria = matrix.shape[1]

    if weights is None:
        weights = np.full(
            n_criteria,
            1.0 / n_criteria,
            dtype=float,
        )
    else:
        weights = np.asarray(weights, dtype=float)

        if weights.shape != (n_criteria,):
            raise ValueError(
                f"Expected {n_criteria} weights, "
                f"got shape {weights.shape}."
            )

        if np.any(weights <= 0):
            raise ValueError(
                "TOPSIS weights must be strictly positive."
            )

        weights = weights / weights.sum()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Alternatives with indices .* are "
            r"(?:dominant|dominated).*",
            category=UserWarning,
            module=r"pymcdm\.validators",
        )

        preference = np.asarray(
            TOPSIS()(matrix, weights, types),
            dtype=float,
        )

    ranks = np.asarray(
        rankdata(-preference, method="min"),
        dtype=float,
    )

    return preference, ranks


def borda_rank_sum(
    matrix: np.ndarray,
    types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    types = np.asarray(types, dtype=int)

    criterion_ranks = np.zeros_like(
        matrix,
        dtype=float,
    )

    for j in range(matrix.shape[1]):
        values = matrix[:, j]

        if types[j] == 1:
            criterion_ranks[:, j] = rankdata(
                -values,
                method="average",
            )
        else:
            criterion_ranks[:, j] = rankdata(
                values,
                method="average",
            )

    rank_sum = criterion_ranks.sum(axis=1)

    borda_rank = np.asarray(
        rankdata(rank_sum, method="min"),
        dtype=float,
    )

    return rank_sum, borda_rank


def rank_single_axis(problem: dict[str, Any]) -> dict[str, Any]:
    if problem["matrix"].shape[1] != 1:
        raise ValueError(
            "Single-axis ranking requires exactly one criterion."
        )

    values = problem["matrix"][:, 0].astype(float)
    criterion_type = int(problem["types"][0])

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi - lo <= 1e-12:
        raise ValueError("Single ranking axis is constant.")

    if criterion_type == 1:
        score = (values - lo) / (hi - lo)
    else:
        score = (hi - values) / (hi - lo)

    ranks = np.asarray(
        rankdata(-score, method="min"),
        dtype=float,
    )

    winner_index = int(np.argmax(score))
    winner = problem["candidates"][winner_index]

    ordered = np.sort(score)
    margin = float(
        ordered[-1] - ordered[-2]
    )

    return {
        **problem,
        "method": "preregistered_single_axis_rank",
        "winner": winner,
        "primary_scores": score,
        "primary_ranks": ranks,
        "margin_to_second": margin,
        "borda_winner": None,
        "borda_agrees": None,
    }


def rank_multicriteria(
    problem: dict[str, Any],
) -> dict[str, Any]:
    matrix = problem["matrix"]
    types = problem["types"]

    preferences, topsis_ranks = run_topsis(
        matrix,
        types,
    )

    borda_sum, borda_ranks = borda_rank_sum(
        matrix,
        types,
    )

    winner_index = int(np.argmax(preferences))
    winner = problem["candidates"][winner_index]

    borda_index = int(np.argmin(borda_sum))
    borda_winner = problem["candidates"][borda_index]

    ordered = np.sort(preferences)
    margin = float(
        ordered[-1] - ordered[-2]
    )

    return {
        **problem,
        "method": "TOPSIS",
        "winner": winner,
        "primary_scores": preferences,
        "primary_ranks": topsis_ranks,
        "borda_rank_sum": borda_sum,
        "borda_ranks": borda_ranks,
        "borda_winner": borda_winner,
        "borda_agrees": winner == borda_winner,
        "margin_to_second": margin,
    }


def rank_domain(
    problem: dict[str, Any],
) -> dict[str, Any]:
    if problem["matrix"].shape[1] == 1:
        return rank_single_axis(problem)

    return rank_multicriteria(problem)
