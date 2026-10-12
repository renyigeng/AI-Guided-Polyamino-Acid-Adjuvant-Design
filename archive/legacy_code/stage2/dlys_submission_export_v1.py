from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FINAL_SUFFIX = "__dlys-stage2-final-v3"
D1_AUDIT_SUFFIX = "__dlys-d1-hydropathy-scale-audit-v1"


def _latest_run(runs: Path, suffix: str, required: str) -> Path | None:
    matches = sorted(p for p in runs.glob(f"*{suffix}") if p.is_dir() and (p / required).exists())
    return matches[-1] if matches else None


def _resolve_recorded_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p.resolve()
    candidate = repo_root / "runs" / p.name
    if candidate.exists():
        return candidate.resolve()
    candidate = repo_root / p
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Recorded run path cannot be resolved: {raw}")


def _copy_selected(src_dir: Path, dst_dir: Path, names: list[str]) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    return copied


def export_submission_snapshot(repo_root: Path, final_run: Path, destination: Path, overwrite: bool = False) -> dict[str, Any]:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {destination}; use --overwrite")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)

    final_manifest = json.loads((final_run / "pipeline_manifest.json").read_text(encoding="utf-8"))
    discovery_run = _resolve_recorded_path(repo_root, final_manifest["discovery_run"])
    ranking_run = _resolve_recorded_path(repo_root, final_manifest["ranking_run"])
    ranking_manifest = json.loads((ranking_run / "manifest.json").read_text(encoding="utf-8"))
    intrinsic_csv = Path(ranking_manifest["intrinsic_csv"])
    if not intrinsic_csv.exists():
        intrinsic_csv = repo_root / "runs" / intrinsic_csv.parent.name / intrinsic_csv.name
    if not intrinsic_csv.exists():
        raise FileNotFoundError(f"Intrinsic CSV not found: {ranking_manifest['intrinsic_csv']}")

    copied = {
        "discovery": _copy_selected(
            discovery_run,
            destination / "discovery",
            [
                "domain_taxonomy.json",
                "discovered_indicators.csv",
                "indicator_evidence.csv",
                "indicator_validation_warnings.csv",
                "summary.md",
                "manifest.json",
            ],
        ),
        "ranking": _copy_selected(
            ranking_run,
            destination / "ranking",
            [
                "criteria_used.csv",
                "criteria_unresolved.csv",
                "decision_matrix_long.csv",
                "decision_matrix_wide.csv",
                "topsis_rankings.csv",
                "single_criterion_rankings.csv",
                "borda_rankings.csv",
                "criterion_individual_ranks.csv",
                "ranking_agreement.csv",
                "leave_one_criterion_out.csv",
                "weight_perturbation_sensitivity.csv",
                "pareto_dominance.csv",
                "decision_diagnostics.csv",
                "selected_representatives.csv",
                "summary.md",
                "manifest.json",
            ],
        ),
        "final": _copy_selected(
            final_run,
            destination / "final",
            [
                "final_candidates.csv",
                "final_diagnostics.csv",
                "criteria_used.csv",
                "criteria_unresolved.csv",
                "criterion_individual_ranks.csv",
                "leave_one_criterion_out.csv",
                "weight_perturbation_sensitivity.csv",
                "pareto_dominance.csv",
                "ranking_agreement.csv",
                "final_report.md",
                "pipeline_manifest.json",
            ],
        ),
    }
    shutil.copy2(intrinsic_csv, destination / "intrinsic_properties_candidate17.csv")

    for cfg_name in [
        "dlys_stage2_taxonomy_v3.json",
        "dlys_metric_registry_v2.json",
        "dlys_multicriteria_ranking_v2.json",
        "dlys_posthoc_reference_targets_v1.json",
    ]:
        src = repo_root / "configs" / cfg_name
        if src.exists():
            (destination / "configs").mkdir(exist_ok=True)
            shutil.copy2(src, destination / "configs" / cfg_name)

    d1_audit = _latest_run(repo_root / "runs", D1_AUDIT_SUFFIX, "hydropathy_scale_results.csv")
    if d1_audit is not None:
        copied["d1_hydropathy_audit"] = _copy_selected(
            d1_audit,
            destination / "d1_hydropathy_audit",
            ["hydropathy_scale_results.csv", "hydropathy_scale_summary.csv", "summary.md", "manifest.json"],
        )

    snapshot_manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_final_run": final_run.name,
        "source_discovery_run": discovery_run.name,
        "source_ranking_run": ranking_run.name,
        "source_intrinsic_csv": intrinsic_csv.name,
        "contains_live_api_credentials": False,
        "purpose": "frozen deterministic inputs/outputs for notebook and recording",
        "copied": copied,
    }
    (destination / "snapshot_manifest.json").write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return snapshot_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the latest final D-Lys stage-2 run for notebook/recording")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--final-run", default="latest")
    parser.add_argument("--destination", default="submission/frozen")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    runs = repo_root / "runs"
    if args.final_run == "latest":
        final_run = _latest_run(runs, FINAL_SUFFIX, "pipeline_manifest.json")
        if final_run is None:
            raise FileNotFoundError("No completed stage-2 final run found")
    else:
        final_run = Path(args.final_run)
        if not final_run.is_absolute():
            final_run = repo_root / final_run
        final_run = final_run.resolve()

    destination = Path(args.destination)
    if not destination.is_absolute():
        destination = repo_root / destination
    manifest = export_submission_snapshot(repo_root, final_run, destination.resolve(), args.overwrite)
    print("=" * 84)
    print("D-Lys submission snapshot exported")
    print("=" * 84)
    print(f"destination: {destination.resolve()}")
    print(f"final run   : {manifest['source_final_run']}")
    print(f"ranking run : {manifest['source_ranking_run']}")
    print(f"discovery   : {manifest['source_discovery_run']}")


if __name__ == "__main__":
    main()
