from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the frozen two-stage "
            "D-Lys / X-block candidate screening."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for generated result files.",
    )

    args = parser.parse_args()

    summary = run_pipeline(
        project_root=ROOT,
        output_dir=args.output_dir,
    )

    results = pd.read_csv(
        args.output_dir / "results.csv"
    )

    print("=" * 72)
    print("Life Science Challenge - reproducible screening")
    print("=" * 72)

    print(
        "Stage 1 functional block:",
        summary["stage1"][
            "selected_functional_block"
        ],
    )

    print("")
    print("Stage 2 within-domain priorities:")

    for _, row in results.iterrows():
        print(
            f"  {row['mechanism_domain']} "
            f"-> {row['X_block_name_zh']} "
            f"({row['X_block']}) "
            f"[{row['selection_method']}]"
        )

    print("")
    print(
        "Cross-domain total score: NO"
    )
    print(
        "Frozen-result regression check:",
        summary[
            "frozen_result_regression_check"
        ],
    )
    print("")
    print(
        "Final candidate file:",
        args.output_dir / "results.csv",
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
