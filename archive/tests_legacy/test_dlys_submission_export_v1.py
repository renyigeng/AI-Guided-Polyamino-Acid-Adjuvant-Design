import json
from pathlib import Path

from src.dlys_submission_export_v1 import export_submission_snapshot


def test_export_submission_snapshot_resolves_run_basenames(tmp_path: Path):
    repo = tmp_path / "repo"
    runs = repo / "runs"
    runs.mkdir(parents=True)

    discovery = runs / "20260101T000000Z__dlys-domain-indicator-discovery-v2"
    ranking = runs / "20260101T000001Z__dlys-multicriteria-ranking-v3"
    final = runs / "20260101T000002Z__dlys-stage2-final-v3"
    intrinsic_run = runs / "20260101T000003Z__intrinsic-material-properties-v1"
    for p in (discovery, ranking, final, intrinsic_run):
        p.mkdir()

    (discovery / "discovered_indicators.csv").write_text("domain_id\nD1\n", encoding="utf-8")
    (discovery / "domain_taxonomy.json").write_text('{"domains": []}', encoding="utf-8")
    intrinsic = intrinsic_run / "intrinsic_properties_candidate17.csv"
    intrinsic.write_text("aa\nF\n", encoding="utf-8")
    (ranking / "selected_representatives.csv").write_text("domain_id\nD1\n", encoding="utf-8")
    (ranking / "manifest.json").write_text(
        json.dumps({"intrinsic_csv": str(Path("Z:/moved") / intrinsic_run.name / intrinsic.name)}),
        encoding="utf-8",
    )
    (final / "final_candidates.csv").write_text("domain_id\nD1\n", encoding="utf-8")
    (final / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "discovery_run": str(Path("Z:/moved") / discovery.name),
                "ranking_run": str(Path("Z:/moved") / ranking.name),
            }
        ),
        encoding="utf-8",
    )

    cfg = repo / "configs"
    cfg.mkdir()
    (cfg / "dlys_stage2_taxonomy_v3.json").write_text("{}", encoding="utf-8")

    dest = repo / "submission" / "frozen"
    manifest = export_submission_snapshot(repo, final, dest)
    assert manifest["source_final_run"] == final.name
    assert (dest / "intrinsic_properties_candidate17.csv").exists()
    assert (dest / "discovery" / "discovered_indicators.csv").exists()
    assert (dest / "ranking" / "selected_representatives.csv").exists()
    assert (dest / "final" / "final_candidates.csv").exists()
