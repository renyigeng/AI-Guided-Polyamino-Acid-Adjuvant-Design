from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUN_SUFFIX = "__dlys-d1-hydropathy-scale-audit-v1"
AA20 = tuple("ACDEFGHIKLMNPQRSTVWY")
D1_CANDIDATES = ("F", "W", "Y")


def _as_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _common_vector(table: Mapping[str, Any], reference: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    left: list[float] = []
    right: list[float] = []
    for aa in AA20:
        x = _as_float(table.get(aa))
        y = _as_float(reference.get(aa))
        if x is not None and y is not None:
            left.append(x)
            right.append(y)
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def audit_hydropathy_tables(
    hydrophobicity_tables: Mapping[str, Mapping[str, Any]],
    *,
    reference_scale: str = "KyteDoolittle",
    candidates: tuple[str, ...] = D1_CANDIDATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit D1 winner sensitivity across published hydrophobicity scales.

    This is a *diagnostic only*. Every scale is oriented so that its residue-wide
    Spearman correlation with Kyte-Doolittle is positive. This only reconciles
    sign conventions (some published scales are hydrophilicity-like); it does not
    tune any candidate-specific weight or alter the registered D1 decision rule.
    """
    if reference_scale not in hydrophobicity_tables:
        raise KeyError(f"reference scale {reference_scale!r} not found")
    ref = hydrophobicity_tables[reference_scale]
    rows: list[dict[str, Any]] = []

    for scale_name in sorted(hydrophobicity_tables):
        table = hydrophobicity_tables[scale_name]
        if any(_as_float(table.get(aa)) is None for aa in candidates):
            continue
        vec, ref_vec = _common_vector(table, ref)
        if vec.size < 10:
            continue
        rho = float(spearmanr(vec, ref_vec).statistic)
        if not np.isfinite(rho):
            continue
        sign = 1.0 if rho >= 0 else -1.0
        oriented = {aa: sign * float(table[aa]) for aa in candidates}
        best = max(oriented.values())
        winners = [aa for aa in candidates if abs(oriented[aa] - best) <= 1e-12]
        rows.append(
            {
                "scale": scale_name,
                "spearman_vs_kyte_doolittle": rho,
                "orientation_sign": int(sign),
                "orientation_rule": "keep" if sign > 0 else "flip_sign_to_match_KD_direction",
                **{f"raw_{aa}": float(table[aa]) for aa in candidates},
                **{f"oriented_{aa}": float(oriented[aa]) for aa in candidates},
                "winner": "|".join(winners),
                "winner_is_F": winners == ["F"],
                "tie": len(winners) > 1,
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No usable hydrophobicity scales were found")

    summary_rows: list[dict[str, Any]] = []
    for label, mask in [
        ("all_available_scales", pd.Series(True, index=detail.index)),
        ("moderately_concordant_abs_rho_ge_0.5", detail["spearman_vs_kyte_doolittle"].abs() >= 0.5),
        ("strongly_concordant_abs_rho_ge_0.7", detail["spearman_vs_kyte_doolittle"].abs() >= 0.7),
    ]:
        subset = detail[mask]
        if subset.empty:
            continue
        counts = subset["winner"].value_counts(dropna=False)
        summary_rows.append(
            {
                "subset": label,
                "n_scales": int(len(subset)),
                "F_unique_winner_count": int((subset["winner"] == "F").sum()),
                "W_unique_winner_count": int((subset["winner"] == "W").sum()),
                "Y_unique_winner_count": int((subset["winner"] == "Y").sum()),
                "tie_count": int(subset["tie"].sum()),
                "F_unique_winner_fraction": float((subset["winner"] == "F").mean()),
                "most_common_winner": str(counts.index[0]),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def load_peptides_hydrophobicity_tables() -> Mapping[str, Mapping[str, Any]]:
    try:
        import peptides
    except ImportError as exc:  # pragma: no cover - exercised in user's environment
        raise RuntimeError("peptides.py is required: python -m pip install peptides==0.5.0") from exc
    tables = getattr(getattr(peptides, "tables", None), "HYDROPHOBICITY", None)
    if not isinstance(tables, Mapping):
        raise RuntimeError("peptides.tables.HYDROPHOBICITY is unavailable in this peptides.py version")
    return tables


def write_outputs(out: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=False)
    detail.to_csv(out / "hydropathy_scale_results.csv", index=False)
    summary.to_csv(out / "hydropathy_scale_summary.csv", index=False)
    all_row = summary.iloc[0]
    md = [
        "# D1 hydropathy-scale sensitivity audit",
        "",
        "> Diagnostic only: this audit does not change the preregistered D1 selector. Published hydrophobicity scales exposed by peptides.py are sign-oriented against Kyte-Doolittle using all standard residues, then only F/W/Y are compared.",
        "",
        f"- usable scales: **{int(all_row['n_scales'])}**",
        f"- F unique winner: **{int(all_row['F_unique_winner_count'])}/{int(all_row['n_scales'])}** ({float(all_row['F_unique_winner_fraction']):.1%})",
        f"- W unique winner: **{int(all_row['W_unique_winner_count'])}**",
        f"- Y unique winner: **{int(all_row['Y_unique_winner_count'])}**",
        f"- ties: **{int(all_row['tie_count'])}**",
        "",
        "Interpretation: use this as a sensitivity check on the D1 hydropathy axis, not as an ensemble vote that overrides the registered Kyte-Doolittle decision.",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "reference_scale": "KyteDoolittle",
        "candidates": list(D1_CANDIDATES),
        "role": "sensitivity_diagnostic_only",
        "used_to_tune_primary_selector": False,
        "n_scales": int(len(detail)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit D1 F/W/Y sensitivity across peptides.py hydrophobicity scales")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    tables = load_peptides_hydrophobicity_tables()
    detail, summary = audit_hydropathy_tables(tables)
    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = root / out
        out = out.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = root / "runs" / f"{stamp}{RUN_SUFFIX}"
    write_outputs(out, detail, summary)
    print("=" * 84)
    print("D1 hydropathy-scale sensitivity audit")
    print("=" * 84)
    print(f"run: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
