from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .dlys_stage2_consistency_audit_v3 import audit_repo
except ImportError:  # pragma: no cover
    from dlys_stage2_consistency_audit_v3 import audit_repo

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _resolve(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()


def _final_report(selected: pd.DataFrame, diagnostics: pd.DataFrame, discovery_run: Path, ranking_run: Path) -> str:
    lines = [
        "# D-Lys 第二嵌段最终候选报告 V3.1",
        "",
        "> 方法：合格文献证据 → LLM/ICL域内指标发现 → 成熟AAindex/RDKit/peptides.py数值映射 → 自适应确定性决策。保留≥2个合格非冗余指标时采用等权TOPSIS；若预注册允许且仅剩一个独立轴，则采用单轴排序，不人为制造第二指标。Borda、criterion-ablation、±20%权重扰动与Pareto仅在真正多指标域中作为诊断。已知参考代表仅post-hoc对照，不参与筛选。",
        "",
        f"- indicator run: `{discovery_run}`",
        f"- ranking run: `{ranking_run}`",
        "",
        "## 正式域内代表",
        "",
    ]
    for _, row in selected.iterrows():
        if row.get("status") != "completed":
            lines.append(f"- **{row['domain_name_zh']}**：未完成（{row.get('reason','')}）。")
            continue
        method_raw = row.get("primary_method", "")
        method = "" if pd.isna(method_raw) else str(method_raw)
        diag = diagnostics[diagnostics["domain_id"] == row["domain_id"]]
        if not method and not diag.empty:
            inferred = diag.iloc[0].get("primary_method", "")
            method = "" if pd.isna(inferred) else str(inferred)
        flags = row.get("diagnostic_flags", "")
        if method == "TOPSIS":
            lines.append(
                f"- **{row['domain_name_zh']} → {row['selected_candidate_name_zh']}**；方法=TOPSIS；"
                f"Borda一致={row.get('topsis_borda_agree','')}；"
                f"criterion-ablation={row.get('loo_winner_stability_count','')}/{row.get('loo_total','')}；"
                f"权重扰动={row.get('weight_perturbation_stability_count','')}/{row.get('weight_perturbation_total','')}；"
                f"诊断标记={flags or 'none'}。"
            )
            if not diag.empty and str(row.get("topsis_borda_agree", "")).lower() in {"false", "0"}:
                bw = diag.iloc[0].get("borda_diagnostic_winner", "")
                lines.append(f"  - Borda为独立序数诊断，当前与TOPSIS不同（{bw}）；不用于覆盖TOPSIS主结果。")
        else:
            lines.append(
                f"- **{row['domain_name_zh']} → {row['selected_candidate_name_zh']}**；方法={method}；"
                f"资格化/冗余审计后仅保留一个独立决策轴，因此不伪造多指标TOPSIS、Borda或权重敏感性；"
                f"诊断标记={flags or 'none'}。"
            )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 这是候选优选/prioritization，不宣称是有监督机器学习对长链材料性能的直接预测。",
        "- AI的作用是从限定证据中发现域内可比较物化维度；数值来自成熟工具/数据库，最终决策是确定性的。",
        "- 只有真正存在≥2个独立合格指标的域才使用TOPSIS；单轴域强行凑第二指标会造成重复投票，因此显式降级为单轴排序。",
        "- criterion-ablation改变winner说明某个物理机制轴不可随意删除；它不是自动判定结果无效。",
        "- Borda只看名次、TOPSIS保留数值距离，因此两者不一致时必须报告，不反向调权重。",
        "- post-hoc reference match只用于结果核验，不能触发重新调指标、域划分或权重。",
    ]
    return "\n".join(lines) + "\n"


def build_final_bundle(repo_root: Path, discovery_run: Path, ranking_run: Path, out: Path) -> Path:
    selected_path = ranking_run / "selected_representatives.csv"
    diagnostics_path = ranking_run / "decision_diagnostics.csv"
    if not selected_path.exists() or not diagnostics_path.exists():
        raise FileNotFoundError("ranking run lacks selected_representatives.csv or decision_diagnostics.csv")
    selected = pd.read_csv(selected_path)
    diagnostics = pd.read_csv(diagnostics_path)
    out.mkdir(parents=True, exist_ok=False)
    selected.to_csv(out / "final_candidates.csv", index=False)
    diagnostics.to_csv(out / "final_diagnostics.csv", index=False)
    for name in [
        "criteria_used.csv",
        "criteria_unresolved.csv",
        "criterion_individual_ranks.csv",
        "leave_one_criterion_out.csv",
        "weight_perturbation_sensitivity.csv",
        "pareto_dominance.csv",
        "ranking_agreement.csv",
    ]:
        src = ranking_run / name
        if src.exists():
            shutil.copy2(src, out / name)
    (out / "final_report.md").write_text(
        _final_report(selected, diagnostics, discovery_run, ranking_run), encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "schema_version": "3.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "discovery_run": str(discovery_run),
        "ranking_run": str(ranking_run),
        "primary_decision_method": "adaptive: equal-weight TOPSIS for multicriteria domains; preregistered single-axis rank when explicitly allowed and only one independent criterion remains",
        "borda_role": "diagnostic_only_for_multicriteria_domains",
        "posthoc_reference_used_for_selection": False,
        "selected": selected[["domain_id", "selected_candidate", "selected_candidate_name_zh"]].to_dict(orient="records"),
    }
    (out / "pipeline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command D-Lys stage-2 finalization: audit -> ICL discovery -> robust MCDA -> final bundle.")
    parser.add_argument("--repo-root", default=".")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qualification-run", help="mechanism-role-refinement-v1 run; runs a fresh ICL indicator discovery")
    group.add_argument("--indicator-run", help="existing indicator-discovery run; use 'latest' to skip a new LLM call")
    parser.add_argument("--intrinsic-csv")
    parser.add_argument("--final-output-dir")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    audit = audit_repo(repo_root)
    if audit["status"] != "passed":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        raise SystemExit("Stage-2 consistency audit failed; refusing to run final ranking")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.qualification_run:
        discovery_run = repo_root / "runs" / f"{stamp}__dlys-domain-indicator-discovery-v2"
        _run(
            [
                sys.executable, "-m", "src.dlys_domain_indicator_discovery_v2",
                "--repo-root", str(repo_root),
                "--qualification-run", str(_resolve(repo_root, args.qualification_run)),
                "--output-dir", str(discovery_run),
            ],
            repo_root,
        )
    else:
        if args.indicator_run == "latest":
            matches = sorted(
                p for p in (repo_root / "runs").glob("*__dlys-domain-indicator-discovery-v2")
                if (p / "discovered_indicators.csv").exists()
            )
            if not matches:
                raise FileNotFoundError("No completed indicator-discovery run found")
            discovery_run = matches[-1]
        else:
            discovery_run = _resolve(repo_root, args.indicator_run)
            assert discovery_run is not None

    ranking_run = repo_root / "runs" / f"{stamp}__dlys-multicriteria-ranking-v3"
    cmd = [
        sys.executable, "-m", "src.dlys_multicriteria_ranking_v2",
        "--repo-root", str(repo_root),
        "--indicator-run", str(discovery_run),
        "--output-dir", str(ranking_run),
    ]
    if args.intrinsic_csv:
        cmd += ["--intrinsic-csv", str(_resolve(repo_root, args.intrinsic_csv))]
    _run(cmd, repo_root)

    if args.final_output_dir:
        final_out = _resolve(repo_root, args.final_output_dir)
        assert final_out is not None
    else:
        final_out = repo_root / "runs" / f"{stamp}__dlys-stage2-final-v3"
    build_final_bundle(repo_root, discovery_run, ranking_run, final_out)

    final = pd.read_csv(final_out / "final_candidates.csv")
    print("=" * 84)
    print("D-Lys stage-2 FINAL V3")
    print("=" * 84)
    print(f"final bundle: {final_out}")
    for _, row in final.iterrows():
        if row.get("status") == "completed":
            print(
                f"{row['domain_id']} {row['domain_name_zh']} -> {row['selected_candidate_name_zh']} "
                f"[{row.get('primary_method','')}]"
            )


if __name__ == "__main__":
    main()
