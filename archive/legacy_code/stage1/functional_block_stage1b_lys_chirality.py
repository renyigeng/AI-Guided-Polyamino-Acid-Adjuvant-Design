from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from llm_compatible_client import call_chat, load_settings, parse_json_content
from functional_block_decide_llm import (
    ROOT,
    RUNS_DIR,
    RESULTS_DIR,
    LLM_CONFIG,
    clean,
    load_jsonl,
    chirality_packet,
    chirality_system,
    chirality_prompt,
    validate_chirality,
    normalize_vote,
    pair_result,
)

CHIRALITY_QUERY_LABELS = {
    "LK01": "L-Lys immune-material context/control evidence",
    "DK01": "D-Lys broad polypeptide/nanoparticle immune evidence",
    "DK02": "D-Lys DC / antigen-presentation / vaccine-adjuvant evidence",
    "DK03": "D-Lys adaptive-immunity / T-cell / antitumor evidence",
    "KD02": "Direct L-Lys vs D-Lys chirality/stereochemistry comparison",
    "KD03": "Explicit L-Lys and D-Lys co-mentioned head-to-head comparison",
}


def normalize_identity_code(value: object) -> str:
    text = clean(value).upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "K": "K",
        "LYS": "K",
        "LYSINE": "K",
        "L_LYS": "K",
        "D_LYS": "K",
        "R": "R",
        "ARG": "R",
        "ARGININE": "R",
        "H": "H",
        "HIS": "H",
        "HISTIDINE": "H",
    }
    return aliases.get(text, text)


def load_stage1a_identity(path: Path) -> tuple[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_identity = (
        data.get("selected_identity")
        or data.get("project_specific_identity")
        or data.get("functional_block_identity")
    )
    identity = normalize_identity_code(raw_identity)
    return identity, data


def require_lys_identity(identity: str) -> None:
    if identity != "K":
        raise RuntimeError(
            "Stage 1B is a Lys chirality screen and can run only after Stage 1A selects "
            f"Lysine (K). Stage 1A selected: {identity or 'UNKNOWN'}."
        )


def evidence_counts(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in records:
        qid = clean(item.get("rag_query_id"))
        if qid in CHIRALITY_QUERY_LABELS and clean(item.get("summary")):
            counts[qid] += 1
    return {qid: counts.get(qid, 0) for qid in CHIRALITY_QUERY_LABELS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1B: after Stage 1A has selected Lysine, compare L-Lys vs D-Lys only."
        )
    )
    parser.add_argument("--run-dir", required=True, help="functional-block-rag-context run directory")
    parser.add_argument(
        "--identity-decision",
        default=str(RESULTS_DIR / "functional_block_stage1a_identity.json"),
        help="Stage 1A identity decision JSON",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Chirality repeats per orientation")
    args = parser.parse_args()

    identity_path = Path(args.identity_decision)
    if not identity_path.is_absolute():
        identity_path = ROOT / identity_path
    identity, stage1a = load_stage1a_identity(identity_path)
    require_lys_identity(identity)

    source_run = Path(args.run_dir)
    if not source_run.is_absolute():
        source_run = ROOT / source_run
    evidence_path = source_run / "rag_contextual_evidence.jsonl"
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)

    records = load_jsonl(evidence_path)
    c_evidence = chirality_packet(records)
    if not c_evidence:
        raise RuntimeError(
            "No Lys chirality evidence was found in the supplied RAG run. Expected evidence from "
            "LK01/DK01/DK02/DK03/KD02/KD03."
        )

    counts = evidence_counts(records)
    repeats = max(1, int(args.repeats))

    print("=" * 72)
    print("Stage 1B — Lys chirality screening")
    print("=" * 72)
    print("Previous stage result: Lysine (K)")
    print("Stage 1A decision     :", identity_path)
    print("Starting Lys chirality screening")
    print("Candidates: L-Lys vs D-Lys")
    print("Evidence categories:")
    for qid, label in CHIRALITY_QUERY_LABELS.items():
        print(f"  {qid}: {label}  [RAG evidence rows={counts[qid]}]")
    print("Direct L-vs-D evidence has highest evidentiary priority.")
    print("Extra D-Lys retrieval improves recall only; paper count is not a quality score.")
    print("=" * 72)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_DIR / f"{timestamp}__functional-block-stage1b-lys-chirality"
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "calls"
    raw_dir.mkdir()

    (output_dir / "lys_chirality_evidence.json").write_text(
        json.dumps(c_evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    settings, safe_meta = load_settings(LLM_CONFIG)
    calls: list[dict] = []
    votes: list[str | None] = []

    print("\n[Stage 1B.1] Pairwise L-Lys vs D-Lys evidence synthesis")
    for repeat in range(1, repeats + 1):
        for left, right, orientation in (("D", "L", "forward"), ("L", "D", "reverse")):
            user_prompt = chirality_prompt(left, right, c_evidence)
            content, raw = call_chat(
                settings,
                [
                    {"role": "system", "content": chirality_system()},
                    {"role": "user", "content": user_prompt},
                ],
            )
            parsed = validate_chirality(parse_json_content(content))
            vote = normalize_vote(parsed["preference"], left, right)
            votes.append(vote)
            calls.append(
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
            stem = f"chirality_r{repeat}_{orientation}"
            (raw_dir / f"{stem}_prompt.txt").write_text(
                chirality_system() + "\n\n" + user_prompt, encoding="utf-8"
            )
            (raw_dir / f"{stem}_response.txt").write_text(content, encoding="utf-8")
            print(f"D-Lys vs L-Lys repeat={repeat} {orientation}: {vote or 'INSUFFICIENT'}")

    result_pair = pair_result(votes, "D", "L")
    winner = result_pair.get("winner")
    selected = "D-Lys" if winner == "D" else "L-Lys" if winner == "L" else None

    result = {
        "schema_version": "1.0",
        "stage": "functional_block_stage1b_lys_chirality",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage1a_identity_decision": str(identity_path),
        "stage1a_selected_identity": "K",
        "source_rag_context_run": str(source_run),
        "chirality_query_labels": CHIRALITY_QUERY_LABELS,
        "chirality_evidence_counts": counts,
        "chirality_calls": calls,
        "chirality_result": result_pair,
        "selected_chirality": winner,
        "selected_functional_block": selected,
        "llm": safe_meta,
        "scientific_boundaries": {
            "only_lys_chirality_is_evaluated": True,
            "direct_L_vs_D_evidence_has_highest_priority": True,
            "D_Lys_enriched_retrieval_is_recall_only": True,
            "medcpt_score_is_relevance_only": True,
            "rag_relevance_score_is_relevance_only": True,
            "paper_count_is_not_candidate_quality": True,
        },
    }

    result_path = output_dir / "functional_block_stage1b_chirality_decision.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "# Stage 1B — Lys chirality",
        "",
        "- Previous stage identity: Lysine (K)",
        f"- Selected chirality: {winner or 'INCONCLUSIVE'}",
        f"- Final Stage-1 functional block: {selected or 'INCONCLUSIVE'}",
        f"- Repeats per orientation: {repeats}",
        "",
        "## Boundary",
        "D-Lys-enriched retrieval is used only to improve recall. Comparative selection prioritizes direct L-vs-D evidence and does not use paper count or relevance score as immune-effect magnitude.",
    ]
    summary_path = output_dir / "functional_block_stage1b_chirality_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    RESULTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(result_path, RESULTS_DIR / "functional_block_stage1b_chirality.json")
    shutil.copy2(summary_path, RESULTS_DIR / "functional_block_stage1b_chirality.md")

    manifest = {
        "step": "functional_block_stage1b_lys_chirality",
        "version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage1a_decision": str(identity_path),
        "source_rag_context_run": str(source_run),
        "outputs": [str(result_path), str(summary_path)],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("Stage 1B completed")
    print("Previous identity: Lysine (K)")
    print("Selected chirality:", winner or "INCONCLUSIVE")
    print("Final Stage-1 functional block:", selected or "INCONCLUSIVE")
    print("Decision:", result_path)


if __name__ == "__main__":
    main()
