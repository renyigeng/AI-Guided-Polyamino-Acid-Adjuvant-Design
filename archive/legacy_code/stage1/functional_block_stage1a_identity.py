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
    RAG_CONFIG,
    ICL_CONFIG,
    DEMOS_PATH,
    clean,
    compact_record,
    load_jsonl,
    pairwise_system,
    pairwise_prompt,
    validate_pairwise,
    normalize_vote,
    pair_result,
    condorcet_winner,
)

PROJECT_CONTEXT = {
    "target_system": "amphiphilic binary poly(amino acid) tumor-vaccine carrier",
    "existing_experimental_platform": "lysine-based (K) cationic block",
    "role_interchangeability_context": (
        "K and R are treated by the project team as potentially replaceable cationic blocks; "
        "the existing experimental route uses K."
    ),
    "use_of_context": (
        "Project continuity is a declared project prior, not literature evidence. It is used only "
        "if literature-based project-fit evidence remains tied or insufficient."
    ),
}

PROJECT_FIT_CRITERIA = [
    "Architecture fit: closeness to polypeptide/block-polypeptide, nanoparticle, or nanovaccine systems.",
    "Vaccine-mechanism fit: antigen delivery, dendritic-cell maturation, antigen presentation/cross-presentation, adjuvant or vaccine response.",
    "Delivery-safety balance: intracellular-delivery evidence together with material-safety evidence.",
]



def general_identity_packets(records: list[dict], top_per_candidate: int = 12) -> dict[str, list[dict]]:
    """Reproduce the prior general K/R/H packet policy without admitting chirality rows."""
    packets = {"K": [], "R": [], "H": []}
    allowed = {candidate: {f"{candidate}{suffix}" for suffix in ("01", "02", "03", "04")} for candidate in packets}
    for item in records:
        qid = clean(item.get("rag_query_id"))
        if not clean(item.get("summary")):
            continue
        for candidate, query_ids in allowed.items():
            if qid in query_ids:
                packets[candidate].append(compact_record(item))
                break
    for candidate in packets:
        packets[candidate].sort(
            key=lambda x: (x["relevance_score"], -x["medcpt_rank"]),
            reverse=True,
        )
        packets[candidate] = packets[candidate][:top_per_candidate]
    return packets

def balanced_identity_packet(records: list[dict], candidate: str, top_per_query: int = 3) -> list[dict]:
    """Build a balanced K or R evidence packet from 01-04 dimensions only.

    This intentionally ignores every chirality query (LK/DK/KD), so Stage 1A cannot
    consume Lys chirality evidence even if such evidence has already been prefetched.
    """
    selected: list[dict] = []
    for suffix in ("01", "02", "03", "04"):
        qid = f"{candidate}{suffix}"
        rows = [
            compact_record(item)
            for item in records
            if clean(item.get("rag_query_id")) == qid and clean(item.get("summary"))
        ]
        rows.sort(
            key=lambda x: (x["relevance_score"], -x["medcpt_rank"]),
            reverse=True,
        )
        selected.extend(rows[:top_per_query])
    return selected


def project_fit_system() -> str:
    return (
        "You are performing a project-fit tie-break between lysine (K) and arginine (R) after "
        "a prior general evidence comparison could not stably separate them. Use ONLY the retrieved "
        "scientific evidence supplied here. Do NOT decide universal biological superiority. Do NOT "
        "use paper count, MedCPT score, or RAG relevance score as an immune-effect score. Compare "
        "transferability to the specific target material architecture and vaccine mechanism. If the "
        "evidence does not stably support one candidate, return TIE. Return valid JSON only."
    )


def project_fit_prompt(first: str, second: str, packets: dict[str, list[dict]]) -> str:
    names = {
        "K": "Lys / lysine-based polypeptide",
        "R": "Arg / arginine-based polypeptide",
    }
    schema = {
        "preference": "K|R|TIE|INSUFFICIENT",
        "architecture_fit": {
            "verdict": "K|R|TIE|INSUFFICIENT",
            "reason": "short evidence-grounded reason",
            "evidence_ids": ["E001"],
        },
        "vaccine_mechanism_fit": {
            "verdict": "K|R|TIE|INSUFFICIENT",
            "reason": "short evidence-grounded reason",
            "evidence_ids": ["E001"],
        },
        "delivery_safety_balance": {
            "verdict": "K|R|TIE|INSUFFICIENT",
            "reason": "short evidence-grounded reason",
            "evidence_ids": ["E001"],
        },
        "counter_evidence_ids": ["E002"],
        "uncertainties": ["..."],
    }
    return (
        "TARGET PROJECT:\n"
        "An amphiphilic binary poly(amino acid) tumor-vaccine carrier. The cationic block should "
        "be transferable to a polypeptide/block-polypeptide or nanoparticle architecture, support "
        "antigen/biomacromolecule delivery and intracellular handling, have relevant dendritic-cell/"
        "antigen-presentation/adjuvant evidence, and have an acceptable safety trade-off.\n\n"
        "IMPORTANT: project-history information is intentionally NOT supplied in this literature "
        "comparison. Judge literature-based project fit only.\n\n"
        f"CANDIDATE 1 = {names[first]} ({first})\n"
        + json.dumps(packets[first], ensure_ascii=False, indent=2)
        + f"\n\nCANDIDATE 2 = {names[second]} ({second})\n"
        + json.dumps(packets[second], ensure_ascii=False, indent=2)
        + "\n\nPROJECT-FIT CRITERIA:\n"
        + json.dumps(PROJECT_FIT_CRITERIA, ensure_ascii=False, indent=2)
        + "\n\nReturn K only if K is better supported for THIS project, R only if R is better "
        "supported, and TIE when the evidence does not justify a stable project-fit preference.\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def validate_project_fit(parsed: object) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("Project-fit response must be a JSON object")
    allowed = {"K", "R", "TIE", "INSUFFICIENT"}
    pref = clean(parsed.get("preference")).upper()
    if pref not in allowed:
        raise ValueError(f"Invalid project-fit preference: {pref}")
    parsed["preference"] = pref
    return parsed


def stable_two_orientation_verdict(votes: list[str]) -> str:
    if len(votes) == 2 and votes[0] in {"K", "R"} and votes[0] == votes[1]:
        return votes[0]
    return "TIE"


def pair_winner(pair_results: list[dict], a: str, b: str) -> str | None:
    target = {a, b}
    for result in pair_results:
        if set(result.get("pair", [])) == target:
            return result.get("winner")
    return None


def h_is_outcompeted(pair_results: list[dict]) -> bool:
    return pair_winner(pair_results, "K", "H") == "K" and pair_winner(pair_results, "R", "H") == "R"


def resolve_identity(
    primary: str | None,
    pair_results: list[dict],
    literature_project_fit: str | None,
    allow_project_continuity_fallback: bool,
) -> tuple[str | None, str]:
    """Pure decision policy used by main() and tests."""
    if primary in {"K", "R", "H"}:
        return primary, "stable_general_pairwise_identity"

    if not h_is_outcompeted(pair_results):
        return None, "general_identity_inconclusive"

    if literature_project_fit in {"K", "R"}:
        return literature_project_fit, "stable_literature_project_fit"

    if allow_project_continuity_fallback:
        return "K", "literature_tie_then_declared_project_continuity_prior"

    return None, "project_fit_inconclusive"


def print_stage_header(records: list[dict], repeats: int) -> None:
    print("=" * 72)
    print("Stage 1A — Cationic amino-acid identity screening")
    print("=" * 72)
    print("Candidate universe : 20 natural amino acids")
    print("Cationic/basic set : K = Lys, R = Arg, H = His")
    print("Evidence dimensions:")
    print("  01  Antigen/protein/mRNA delivery")
    print("  02  Cellular uptake / intracellular-endosomal behavior")
    print("  03  Safety / cytotoxicity / biocompatibility")
    print("  04  DC activation / antigen presentation / vaccine immunity")
    print(f"RAG evidence rows  : {len(records)}")
    print(f"Pairwise repeats   : {repeats}")
    print("Chirality evidence : LOCKED in Stage 1A")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1A: select K/R/H cationic amino-acid identity only. "
            "This program never performs Lys chirality screening."
        )
    )
    parser.add_argument("--run-dir", required=True, help="functional-block-rag-context run directory")
    parser.add_argument("--repeats", type=int, default=1, help="Pairwise repeats per orientation")
    parser.add_argument(
        "--allow-project-continuity-fallback",
        action="store_true",
        help=(
            "If literature project-fit remains tied after H is outcompeted, allow the declared "
            "existing K-based experimental platform to select K. This prior is logged separately "
            "and is never represented as literature evidence."
        ),
    )
    args = parser.parse_args()

    source_run = Path(args.run_dir)
    if not source_run.is_absolute():
        source_run = ROOT / source_run
    evidence_path = source_run / "rag_contextual_evidence.jsonl"
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)

    rag_cfg = json.loads(RAG_CONFIG.read_text(encoding="utf-8"))
    icl_cfg = json.loads(ICL_CONFIG.read_text(encoding="utf-8"))
    demos = json.loads(DEMOS_PATH.read_text(encoding="utf-8"))
    records = load_jsonl(evidence_path)
    repeats = max(1, int(args.repeats))

    print_stage_header(records, repeats)

    # General K/R/H evidence comparison. Stage 1A deliberately uses only K01-H04 queries.
    packets = general_identity_packets(records, top_per_candidate=12)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_DIR / f"{timestamp}__functional-block-stage1a-identity"
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "calls"
    raw_dir.mkdir()

    (output_dir / "candidate_evidence_packets.json").write_text(
        json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    settings, safe_meta = load_settings(LLM_CONFIG)
    dimensions = icl_cfg.get("decision_dimensions", [])
    pairs = [tuple(x) for x in rag_cfg["identity_decision"]["pairwise_pairs"]]

    print("\n[Stage 1A.1] General K/R/H evidence comparison")
    pair_calls: list[dict] = []
    pair_results: list[dict] = []
    call_index = 0

    for a, b in pairs:
        votes: list[str | None] = []
        for repeat in range(1, repeats + 1):
            for left, right, orientation in ((a, b, "forward"), (b, a, "reverse")):
                call_index += 1
                user_prompt = pairwise_prompt(left, right, packets, demos, dimensions)
                content, raw = call_chat(
                    settings,
                    [
                        {"role": "system", "content": pairwise_system()},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                parsed = validate_pairwise(parse_json_content(content))
                vote = normalize_vote(parsed["preference"], left, right)
                votes.append(vote)
                pair_calls.append(
                    {
                        "pair": [a, b],
                        "repeat": repeat,
                        "orientation": orientation,
                        "left": left,
                        "right": right,
                        "normalized_vote": vote,
                        "response": parsed,
                        "usage": raw.get("usage", {}),
                    }
                )
                stem = f"{call_index:03d}_{a}{b}_r{repeat}_{orientation}"
                (raw_dir / f"{stem}_prompt.txt").write_text(
                    pairwise_system() + "\n\n" + user_prompt, encoding="utf-8"
                )
                (raw_dir / f"{stem}_response.txt").write_text(content, encoding="utf-8")
                print(f"{a} vs {b} repeat={repeat} {orientation}: {vote or 'INSUFFICIENT'}")
        pair_results.append(pair_result(votes, a, b))

    primary = condorcet_winner(pair_results, ["K", "R", "H"])
    print("General identity verdict:", primary or "INCONCLUSIVE")

    # Only if H is outcompeted and K/R remain unresolved do we ask the project-specific question.
    project_fit_calls: list[dict] = []
    literature_project_fit: str | None = None

    if primary is None and h_is_outcompeted(pair_results):
        print("\n[Stage 1A.2] K vs R project-fit tie-break")
        print("General evidence supports K and R over H, but K/R are not stably separated.")
        print("Now comparing which candidate better fits THIS binary poly(amino acid) vaccine project.")

        project_packets = {
            "K": balanced_identity_packet(records, "K", top_per_query=3),
            "R": balanced_identity_packet(records, "R", top_per_query=3),
        }
        (output_dir / "kr_project_fit_evidence_packets.json").write_text(
            json.dumps(project_packets, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        project_votes: list[str] = []
        for idx, (first, second, orientation) in enumerate(
            (("K", "R", "forward"), ("R", "K", "reverse")), start=1
        ):
            user_prompt = project_fit_prompt(first, second, project_packets)
            content, raw = call_chat(
                settings,
                [
                    {"role": "system", "content": project_fit_system()},
                    {"role": "user", "content": user_prompt},
                ],
            )
            parsed = validate_project_fit(parse_json_content(content))
            vote = parsed["preference"]
            project_votes.append(vote)
            project_fit_calls.append(
                {
                    "orientation": orientation,
                    "candidate_order": [first, second],
                    "preference": vote,
                    "response": parsed,
                    "usage": raw.get("usage", {}),
                }
            )
            (raw_dir / f"project_fit_{idx}_{orientation}_prompt.txt").write_text(
                project_fit_system() + "\n\n" + user_prompt, encoding="utf-8"
            )
            (raw_dir / f"project_fit_{idx}_{orientation}_response.txt").write_text(
                content, encoding="utf-8"
            )
            print(f"K vs R project-fit {orientation}: {vote}")

        literature_project_fit = stable_two_orientation_verdict(project_votes)
        print("Literature project-fit verdict:", literature_project_fit)

    selected_identity, selection_basis = resolve_identity(
        primary=primary,
        pair_results=pair_results,
        literature_project_fit=literature_project_fit,
        allow_project_continuity_fallback=args.allow_project_continuity_fallback,
    )

    selected_name = {"K": "Lysine", "R": "Arginine", "H": "Histidine"}.get(
        selected_identity or "", "INCONCLUSIVE"
    )

    result = {
        "schema_version": "1.0",
        "stage": "functional_block_stage1a_identity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rag_context_run": str(source_run),
        "candidate_universe": {
            "natural_amino_acids": 20,
            "cationic_basic_candidates": ["K", "R", "H"],
        },
        "evidence_dimensions": {
            "01": "antigen/protein/mRNA delivery",
            "02": "cellular uptake and intracellular/endosomal behavior",
            "03": "safety/cytotoxicity/biocompatibility",
            "04": "DC activation/antigen presentation/vaccine immunity",
        },
        "general_pairwise_calls": pair_calls,
        "general_pairwise_results": pair_results,
        "general_identity_verdict": primary,
        "literature_project_fit_calls": project_fit_calls,
        "literature_project_fit_verdict": literature_project_fit,
        "project_context": PROJECT_CONTEXT,
        "selected_identity": selected_identity,
        "selected_identity_name": selected_name,
        "selection_basis": selection_basis,
        "next_stage": "stage1b_lys_chirality" if selected_identity == "K" else None,
        "chirality_screening_run": False,
        "llm": safe_meta,
        "scientific_boundaries": {
            "stage1a_does_not_use_chirality_evidence": True,
            "does_not_claim_K_universally_superior_to_R": True,
            "project_continuity_is_not_literature_evidence": True,
            "medcpt_score_is_relevance_only": True,
            "rag_relevance_score_is_relevance_only": True,
            "paper_count_is_not_candidate_quality": True,
        },
    }

    result_path = output_dir / "functional_block_stage1a_identity_decision.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "# Stage 1A — Cationic amino-acid identity",
        "",
        f"- General identity verdict: {primary or 'INCONCLUSIVE'}",
        f"- Literature K/R project-fit verdict: {literature_project_fit or 'NOT_RUN'}",
        f"- Selected identity: {selected_name} ({selected_identity or 'INCONCLUSIVE'})",
        f"- Selection basis: {selection_basis}",
        "- Chirality screening: NOT RUN in Stage 1A",
        "",
        "## Boundary",
        "This stage selects only the amino-acid identity. It does not evaluate L-Lys vs D-Lys.",
    ]
    summary_path = output_dir / "functional_block_stage1a_identity_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    RESULTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(result_path, RESULTS_DIR / "functional_block_stage1a_identity.json")
    shutil.copy2(summary_path, RESULTS_DIR / "functional_block_stage1a_identity.md")

    manifest = {
        "step": "functional_block_stage1a_identity",
        "version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "outputs": [str(result_path), str(summary_path)],
        "chirality_screening_run": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("Stage 1A completed")
    print("Selected identity:", f"{selected_name} ({selected_identity})" if selected_identity else "INCONCLUSIVE")
    print("Selection basis:", selection_basis)
    print("Chirality module: NOT RUN")
    print("Decision:", result_path)
    if selected_identity == "K":
        print("Next: run Stage 1B to compare L-Lys vs D-Lys.")


if __name__ == "__main__":
    main()
