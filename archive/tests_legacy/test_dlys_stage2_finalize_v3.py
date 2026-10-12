from pathlib import Path

import pandas as pd

from src.dlys_stage2_finalize_v3 import build_final_bundle


def test_final_bundle_preserves_topsis_primary_and_reports_borda_disagreement(tmp_path):
    ranking = tmp_path / "ranking"
    ranking.mkdir()
    pd.DataFrame(
        [
            {
                "domain_id": "D2", "domain_name_zh": "非芳香疏水-构象域", "status": "completed", "reason": "",
                "selected_candidate": "L", "selected_candidate_name_zh": "亮氨酸",
                "topsis_borda_agree": False, "loo_winner_stability_count": 1, "loo_total": 3,
                "weight_perturbation_stability_count": 6, "weight_perturbation_total": 6,
                "diagnostic_flags": "borda_disagrees_with_primary_topsis|criterion_ablation_changes_winner",
            }
        ]
    ).to_csv(ranking / "selected_representatives.csv", index=False)
    pd.DataFrame(
        [
            {
                "domain_id": "D2", "domain_name_zh": "非芳香疏水-构象域", "primary_method": "TOPSIS",
                "primary_winner": "L", "primary_winner_name_zh": "亮氨酸",
                "borda_diagnostic_winner": "M", "borda_agrees": False,
                "topsis_margin_to_second": 0.1, "borda_rank_sum_margin_to_second": 0.5,
                "criterion_ablation_stability": "1/3", "local_weight_perturbation_stability": "6/6",
                "diagnostic_flags": "borda_disagrees_with_primary_topsis|criterion_ablation_changes_winner",
            }
        ]
    ).to_csv(ranking / "decision_diagnostics.csv", index=False)
    out = tmp_path / "final"
    build_final_bundle(tmp_path, tmp_path / "discovery", ranking, out)
    final = pd.read_csv(out / "final_candidates.csv")
    assert final.iloc[0]["selected_candidate"] == "L"
    text = (out / "final_report.md").read_text(encoding="utf-8")
    assert "Borda为独立序数诊断" in text
    assert "不用于覆盖TOPSIS主结果" in text


def test_final_bundle_reports_single_axis_domain_without_calling_it_topsis(tmp_path):
    ranking = tmp_path / "ranking_single"
    ranking.mkdir()
    pd.DataFrame(
        [
            {
                "domain_id": "D1", "domain_name_zh": "芳香疏水-π作用域", "status": "completed", "reason": "only one independent axis",
                "primary_method": "preregistered_single_axis_rank",
                "selected_candidate": "F", "selected_candidate_name_zh": "苯丙氨酸",
                "topsis_borda_agree": "not_applicable_single_criterion",
                "loo_winner_stability_count": "", "loo_total": "",
                "weight_perturbation_stability_count": "", "weight_perturbation_total": "",
                "diagnostic_flags": "single_criterion_fallback_used",
            }
        ]
    ).to_csv(ranking / "selected_representatives.csv", index=False)
    pd.DataFrame(
        [
            {
                "domain_id": "D1", "domain_name_zh": "芳香疏水-π作用域",
                "primary_method": "preregistered_single_axis_rank",
                "primary_winner": "F", "primary_winner_name_zh": "苯丙氨酸",
                "diagnostic_flags": "single_criterion_fallback_used",
            }
        ]
    ).to_csv(ranking / "decision_diagnostics.csv", index=False)
    out = tmp_path / "final_single"
    build_final_bundle(tmp_path, tmp_path / "discovery", ranking, out)
    text = (out / "final_report.md").read_text(encoding="utf-8")
    assert "苯丙氨酸" in text
    assert "不伪造多指标TOPSIS" in text
    assert "方法=preregistered_single_axis_rank" in text
