from pathlib import Path

import pandas as pd

from src.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_final_pipeline_reproduces_frozen_winners(
    tmp_path,
):
    summary = run_pipeline(
        ROOT,
        tmp_path,
    )

    assert (
        summary["stage1"][
            "selected_functional_block"
        ]
        == "D-Lys"
    )

    assert summary["stage2_winners"] == {
        "D1": "F",
        "D2": "L",
        "D3": "Q",
    }

    assert (
        summary[
            "frozen_result_regression_check"
        ]
        == "PASS"
    )

    results = pd.read_csv(
        tmp_path / "results.csv"
    )

    assert results["X_block"].tolist() == [
        "F",
        "L",
        "Q",
    ]

    assert set(
        results["within_domain_priority"]
    ) == {1}


def test_pipeline_creates_required_outputs(
    tmp_path,
):
    run_pipeline(
        ROOT,
        tmp_path,
    )

    expected = {
        "results.csv",
        "stage2_all_rankings.csv",
        "robustness_summary.csv",
        "d1_hydropathy_scale_summary.csv",
        "criterion_ablation.csv",
        "weight_perturbation.csv",
        "run_summary.json",
    }

    assert expected.issubset(
        {
            path.name
            for path in tmp_path.iterdir()
        }
    )
