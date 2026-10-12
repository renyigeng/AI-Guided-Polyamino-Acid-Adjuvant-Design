from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from llm_compatible_client import call_chat, load_settings, parse_json_content
from functional_block_decide_llm import (
    ROOT,
    RUNS_DIR,
    RESULTS_DIR,
    LLM_CONFIG,
    clean,
    compact_record,
    load_jsonl,
    chirality_packet,
    chirality_system,
    chirality_prompt,
    validate_chirality,
    normalize_vote,
    pair_result,
    selected_label,
)

PROJECT_CONTEXT = {
    "target_system": "amphiphilic binary poly(amino acid) tumor-vaccine carrier",
    "existing_experimental_platform": "lysine-based (K) cationic block",
    "role_interchangeability_context": (
        "K and R are regarded by the project team as potentially replaceable cationic blocks; "
        "the existing experimental route uses K."
    ),
    "use_of_context": (
        "This is a project-continuity prior, not literature evidence. It is used only if the "
        "literature-based K-vs-R project-fit comparison is tied or insufficient."
    ),
}


def balanced_identity_packet(records, candidate, top_per_query=3):
    selected = []

    for suffix in ("01", "02", "03", "04"):
        qid = f"{candidate}{suffix}"

        rows = [
            compact_record(item)
            for item in records
            if clean(item.get("rag_query_id")) == qid
            and clean(item.get("summary"))
        ]

        rows.sort(
            key=lambda x: (
                x["relevance_score"],
                -x["medcpt_rank"],
            ),
            reverse=True,
        )

        selected.extend(rows[:top_per_query])

    return selected


def project_fit_system():
    return (
        "You are performing a project-fit tie-break between lysine (K) and arginine (R) "
        "after a prior general evidence comparison could not stably separate them. "
        "Use ONLY the retrieved scientific evidence supplied here. "
        "Do NOT decide universal biological superiority. "
        "Do NOT use paper count, MedCPT score, or RAG relevance score as an immune-effect score. "
        "Compare transferability to the specific target material architecture and vaccine mechanism. "
        "If the evidence does not stably support one candidate, return TIE. "
        "Return valid JSON only."
    )


def project_fit_prompt(first, second, packets):
    names = {
        "K": "Lys / lysine-based polypeptide",
        "R": "Arg / arginine-based polypeptide",
    }

    schema = {
        "preference": "K|R|TIE|INSUFFICIENT",
        "architecture_fit": {
            "verdict": "K|R|TIE|INSUFFICIENT",
            "reason": "short reason",
            "evidence_ids": ["E001"],
        },
        "vaccine_mechanism_fit": {
            "verdict": "K|R|TIE|INSUFFICIENT",
            "reason": "short reason",
            "evidence_ids": ["E001"],
        },
        "delivery_safety_balance": {
            "verdict": "K|R|TIE|INSUFFICIENT",
            "reason": "short reason",
            "evidence_ids": ["E001"],
        },
        "counter_evidence_ids": ["E002"],
        "uncertainties": ["..."],
    }

    return (
        "TARGET PROJECT:\n"
        "An amphiphilic binary poly(amino acid) tumor-vaccine carrier. "
        "The cationic block should be transferable to a polypeptide/block-polypeptide "
        "or nanoparticle architecture, support antigen/biomacromolecule delivery and "
        "intracellular handling, have relevant dendritic-cell/antigen-presentation/"
        "adjuvant evidence, and have an acceptable safety trade-off.\n\n"

        "IMPORTANT: project-history information is intentionally NOT supplied in this "
        "evidence comparison. Judge literature-based project fit only.\n\n"

        f"CANDIDATE 1 = {names[first]} ({first})\n"
        + json.dumps(packets[first], ensure_ascii=False, indent=2)
        + f"\n\nCANDIDATE 2 = {names[second]} ({second})\n"
        + json.dumps(packets[second], ensure_ascii=False, indent=2)
        + "\n\nCompare these three criteria:\n"
        "1. architecture_fit: closeness to polypeptide/block-polypeptide/"
        "nanoparticle/nanovaccine systems;\n"
        "2. vaccine_mechanism_fit: antigen delivery, DC maturation, antigen presentation/"
        "cross-presentation, adjuvant or vaccine response;\n"
        "3. delivery_safety_balance: intracellular delivery evidence together with "
        "material safety evidence.\n\n"

        "Return K only if K is better supported for THIS project, "
        "R only if R is better supported, and TIE when the evidence does not justify "
        "a stable project-fit preference.\n"

        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def validate_project_fit(parsed):
    if not isinstance(parsed, dict):
        raise ValueError("Project-fit response must be a JSON object")

    allowed = {"K", "R", "TIE", "INSUFFICIENT"}
    pref = clean(parsed.get("preference")).upper()

    if pref not in allowed:
        raise ValueError(f"Invalid project-fit preference: {pref}")

    parsed["preference"] = pref
    return parsed


def evidence_verdict(votes):
    decisive = [v for v in votes if v in {"K", "R"}]

    # 正向、逆向必须一致，才认为文献真的分出了 K/R
    if (
        decisive
        and len(decisive) == len(votes)
        and len(set(decisive)) == 1
    ):
        return decisive[0]

    return "TIE"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "K-vs-R project-fit tie-break, followed by Lys chirality if K is selected."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="functional-block-rag-context run directory",
    )

    parser.add_argument(
        "--chirality-repeats",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    source_run = Path(args.run_dir)

    if not source_run.is_absolute():
        source_run = ROOT / source_run

    evidence_path = source_run / "rag_contextual_evidence.jsonl"

    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)

    records = load_jsonl(evidence_path)

    # K01-K04 / R01-R04 每个维度各取 top 3，
    # 防止某一个维度因为文献多而淹没其他维度
    packets = {
        "K": balanced_identity_packet(records, "K", top_per_query=3),
        "R": balanced_identity_packet(records, "R", top_per_query=3),
    }

    if not packets["K"] or not packets["R"]:
        raise RuntimeError("K or R evidence packet is empty")

    settings, safe_meta = load_settings(LLM_CONFIG)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        RUNS_DIR
        / f"{timestamp}__functional-block-project-fit-tiebreak"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    raw_dir = output_dir / "calls"
    raw_dir.mkdir()

    (
        output_dir
        / "kr_project_fit_evidence_packets.json"
    ).write_text(
        json.dumps(
            packets,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # K vs R 项目适配性
    # ---------------------------------------------------------

    project_fit_calls = []
    votes = []

    orientations = (
        ("K", "R", "forward"),
        ("R", "K", "reverse"),
    )

    for i, (first, second, orientation) in enumerate(
        orientations,
        start=1,
    ):
        prompt = project_fit_prompt(
            first,
            second,
            packets,
        )

        content, raw = call_chat(
            settings,
            [
                {
                    "role": "system",
                    "content": project_fit_system(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        parsed = validate_project_fit(
            parse_json_content(content)
        )

        vote = parsed["preference"]
        votes.append(vote)

        project_fit_calls.append(
            {
                "orientation": orientation,
                "candidate_order": [
                    first,
                    second,
                ],
                "preference": vote,
                "response": parsed,
                "usage": raw.get("usage", {}),
            }
        )

        (
            raw_dir
            / f"project_fit_{i}_{orientation}_prompt.txt"
        ).write_text(
            project_fit_system()
            + "\n\n"
            + prompt,
            encoding="utf-8",
        )

        (
            raw_dir
            / f"project_fit_{i}_{orientation}_response.txt"
        ).write_text(
            content,
            encoding="utf-8",
        )

        print(
            f"K vs R project-fit {orientation}: {vote}"
        )

    lit_verdict = evidence_verdict(votes)

    # ---------------------------------------------------------
    # 文献能分开，就尊重文献；
    # 文献仍然打平，才使用明确声明的项目连续性先验
    # ---------------------------------------------------------

    if lit_verdict in {"K", "R"}:
        selected_identity = lit_verdict
        selection_basis = (
            "stable_literature_project_fit"
        )

    else:
        selected_identity = "K"
        selection_basis = (
            "literature_tie_then_declared_"
            "project_continuity_prior"
        )

    print(
        "Literature project-fit verdict:",
        lit_verdict,
    )

    print(
        "Project-specific identity:",
        selected_identity,
    )

    print(
        "Selection basis:",
        selection_basis,
    )

    # ---------------------------------------------------------
    # 如果项目最终选择 K，立即进入 D-Lys vs L-Lys
    # ---------------------------------------------------------

    chirality_evidence = chirality_packet(records)

    chirality_calls = []

    chirality_result = {
        "pair": ["D", "L"],
        "votes": {
            "D": 0,
            "L": 0,
            "INSUFFICIENT": 0,
        },
        "winner": None,
        "status": "not_run_identity_not_K",
    }

    if selected_identity == "K":

        c_votes = []

        for repeat in range(
            1,
            args.chirality_repeats + 1,
        ):

            for left, right, orientation in (
                ("D", "L", "forward"),
                ("L", "D", "reverse"),
            ):

                prompt = chirality_prompt(
                    left,
                    right,
                    chirality_evidence,
                )

                content, raw = call_chat(
                    settings,
                    [
                        {
                            "role": "system",
                            "content": chirality_system(),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                parsed = validate_chirality(
                    parse_json_content(content)
                )

                vote = normalize_vote(
                    parsed["preference"],
                    left,
                    right,
                )

                c_votes.append(vote)

                chirality_calls.append(
                    {
                        "repeat": repeat,
                        "orientation": orientation,
                        "left": left,
                        "right": right,
                        "normalized_vote": vote,
                        "response": parsed,
                        "usage": raw.get("usage", {}),
                    }
                )

                stem = (
                    f"chirality_r{repeat}_{orientation}"
                )

                (
                    raw_dir
                    / f"{stem}_prompt.txt"
                ).write_text(
                    chirality_system()
                    + "\n\n"
                    + prompt,
                    encoding="utf-8",
                )

                (
                    raw_dir
                    / f"{stem}_response.txt"
                ).write_text(
                    content,
                    encoding="utf-8",
                )

                print(
                    "D-Lys vs L-Lys "
                    f"repeat={repeat} "
                    f"{orientation}: "
                    f"{vote or 'INSUFFICIENT'}"
                )

        chirality_result = pair_result(
            c_votes,
            "D",
            "L",
        )

    selected = selected_label(
        selected_identity,
        chirality_result.get("winner"),
    )

    result = {
        "schema_version": "1.0",
        "stage": (
            "functional_block_project_fit_tiebreak"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "source_rag_context_run": str(
            source_run
        ),

        "general_stage1_status": (
            "K_R_not_stably_separated; "
            "H comparatively weaker in "
            "prior pairwise run"
        ),

        "literature_project_fit_calls":
            project_fit_calls,

        "literature_project_fit_verdict":
            lit_verdict,

        "project_context":
            PROJECT_CONTEXT,

        "project_specific_identity":
            selected_identity,

        "selection_basis":
            selection_basis,

        "lys_chirality_evidence_count":
            len(chirality_evidence),

        "lys_chirality_calls":
            chirality_calls,

        "lys_chirality_result":
            chirality_result,

        "selected_functional_block":
            selected,

        "scientific_boundaries": {
            "does_not_claim_K_universally_superior_to_R": True,
            "project_continuity_is_not_literature_evidence": True,
            "project_continuity_used_only_on_literature_tie": True,
            "medcpt_score_is_relevance_only": True,
            "rag_relevance_score_is_relevance_only": True,
            "paper_count_is_not_candidate_quality": True,
        },

        "llm": safe_meta,
    }

    result_path = (
        output_dir
        / "functional_block_project_fit_decision.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "# Functional Block Project-Fit Tie-break",
        "",
        (
            "- Literature-based K-vs-R "
            f"project-fit verdict: {lit_verdict}"
        ),
        (
            "- Project-specific identity: "
            f"{selected_identity}"
        ),
        (
            "- Selection basis: "
            f"{selection_basis}"
        ),
        (
            "- Lys chirality: "
            f"{chirality_result.get('winner') or 'UNKNOWN'}"
            if selected_identity == "K"
            else "- Lys chirality: not run"
        ),
        (
            "- Selected functional block: "
            f"{selected or 'INCONCLUSIVE'}"
        ),
        "",
        "## Interpretation",
        (
            "This does not claim that Lys is universally superior "
            "to Arg. If the retrieved literature remains tied, "
            "Lys is selected only because the existing project "
            "platform already uses a Lys cationic block; that "
            "continuity is recorded as project context rather "
            "than literature evidence."
        ),
    ]

    summary_path = (
        output_dir
        / "functional_block_project_fit_summary.md"
    )

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    shutil.copy2(
        result_path,
        RESULTS_DIR
        / "functional_block_stage1_project_fit_final.json",
    )

    shutil.copy2(
        summary_path,
        RESULTS_DIR
        / "functional_block_stage1_project_fit_final.md",
    )

    print("=" * 72)
    print("Project-fit tie-break completed")
    print("Run:", output_dir)

    print(
        "Literature project-fit verdict:",
        lit_verdict,
    )

    print(
        "Project-specific identity:",
        selected_identity,
    )

    print(
        "Lys chirality:",
        (
            chirality_result.get("winner")
            if selected_identity == "K"
            else "N/A"
        ),
    )

    print(
        "Selected:",
        selected or "INCONCLUSIVE",
    )

    print(
        "Decision:",
        result_path,
    )


if __name__ == "__main__":
    main()
