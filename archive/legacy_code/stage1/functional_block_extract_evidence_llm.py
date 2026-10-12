from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from llm_compatible_client import call_chat, load_settings, parse_json_content

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
LLM_CONFIG = ROOT / "configs" / "functional_block_llm_v1.json"
RAG_CONFIG = ROOT / "configs" / "functional_block_rag_v1.json"

SYSTEM_PROMPT = """You are performing contextual evidence summarization for a scientific RAG system.
This step follows the PaperQA2 RCS principle: summarize a retrieved document specifically for the current scientific question, and separately estimate relevance to that question.

Use ONLY the supplied title and abstract. Do not add outside knowledge. Do not infer missing experiments. Do not treat absence of a result as a negative result. Preserve explicit numeric results when stated. Put only results containing actual numeric values in reported_numeric_results; put non-numeric directional/comparative statements in reported_qualitative_findings.

The relevance_score is ONLY query-document relevance for RAG evidence selection. It is NOT an immune activation score, not an effect size, and not a material-quality score.

Return valid JSON only."""


def clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def prompt_for_row(row: dict) -> str:
    schema = {
        "summary": "query-specific evidence summary; empty string if not applicable",
        "relevance_score": 0,
        "reported_outcomes": ["explicit endpoints/tasks stated in the abstract"],
        "reported_numeric_results": ["exact numeric result text containing explicit numbers, if stated"],
        "reported_qualitative_findings": ["non-numeric directional or comparative findings explicitly stated"],
        "material_or_molecular_context": "what material/molecule was actually studied",
        "biological_context": "cell/animal/immune context actually studied",
        "limitations_for_current_question": ["specific transferability or missing-information limitations"],
        "supporting_text": "one short evidence-bearing phrase from the abstract, <= 25 words, or empty"
    }
    return (
        "Summarize this retrieved article to help answer the CURRENT QUESTION. "
        "Do not directly make the final candidate decision. If the article is irrelevant, "
        "set summary to an empty string and relevance_score to 0. Otherwise use an integer "
        "relevance_score from 1 to 10, following PaperQA2-style contextual relevance scoring.\n\n"
        f"CURRENT QUESTION:\n{clean(row.get('rag_question'))}\n\n"
        f"QUERY ID: {clean(row.get('rag_query_id'))}\n"
        f"CANDIDATE HINT: {clean(row.get('candidate'))}\n"
        f"CHIRALITY HINT: {clean(row.get('chirality'))}\n"
        f"PMID: {clean(row.get('pmid'))}\n"
        f"DOI: {clean(row.get('doi'))}\n\n"
        f"TITLE:\n{clean(row.get('title'))}\n\n"
        f"ABSTRACT:\n{clean(row.get('abstract_text'))}\n\n"
        "Return JSON in exactly this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def validate(parsed: object) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("Contextualization response must be a JSON object")
    score = parsed.get("relevance_score", 0)
    try:
        score = int(score)
    except Exception as exc:
        raise ValueError(f"Invalid relevance_score: {score}") from exc
    if score < 0 or score > 10:
        raise ValueError(f"relevance_score must be 0..10, got {score}")
    parsed["relevance_score"] = score
    parsed["summary"] = clean(parsed.get("summary"))
    for key in (
        "reported_outcomes",
        "reported_numeric_results",
        "reported_qualitative_findings",
        "limitations_for_current_question",
    ):
        value = parsed.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            value = [str(value)]
        parsed[key] = [clean(x) for x in value if clean(x)]

    # Backward compatibility for any cached/test payload using the old field.
    legacy = parsed.pop("reported_quantitative_results", None)
    if legacy:
        if not isinstance(legacy, list):
            legacy = [str(legacy)]
        parsed["reported_numeric_results"].extend(clean(x) for x in legacy if clean(x))

    # Numeric means numeric: if the model places a non-numeric statement in the
    # numeric field, preserve it as a qualitative finding instead of silently
    # treating it as an effect size.
    numeric = []
    demoted = []
    for item in parsed["reported_numeric_results"]:
        if any(ch.isdigit() for ch in item):
            numeric.append(item)
        else:
            demoted.append(item)
    parsed["reported_numeric_results"] = numeric
    for item in demoted:
        if item not in parsed["reported_qualitative_findings"]:
            parsed["reported_qualitative_findings"].append(item)
    parsed["material_or_molecular_context"] = clean(parsed.get("material_or_molecular_context"))
    parsed["biological_context"] = clean(parsed.get("biological_context"))
    parsed["supporting_text"] = clean(parsed.get("supporting_text"))
    if score == 0 and parsed["summary"]:
        raise ValueError("relevance_score 0 requires an empty summary")
    return parsed


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaperQA2-style contextual evidence summarization of MedCPT top-k pairs."
    )
    parser.add_argument("--run-dir", required=True, help="functional-block-medcpt-rerank run directory")
    parser.add_argument("--limit", type=int, default=0, help="Optional trial limit")
    parser.add_argument("--queries", default="", help="Optional comma-separated query IDs")
    args = parser.parse_args()

    source_run = Path(args.run_dir)
    if not source_run.is_absolute():
        source_run = ROOT / source_run
    input_path = source_run / "rag_query_document_pairs.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if args.queries:
        allowed = {x.strip() for x in args.queries.split(",") if x.strip()}
        df = df[df["rag_query_id"].isin(allowed)].copy()
    if args.limit > 0:
        df = df.head(args.limit).copy()
    if df.empty:
        raise RuntimeError("No query-document pairs selected for contextualization")

    settings, safe_meta = load_settings(LLM_CONFIG)
    rag_cfg = json.loads(RAG_CONFIG.read_text(encoding="utf-8"))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_DIR / f"{timestamp}__functional-block-rag-context"
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "llm_contextualization"
    raw_dir.mkdir()

    results: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        source = row.to_dict()
        evidence_id = f"E{i:03d}"
        user_prompt = prompt_for_row(source)
        content, raw_response = call_chat(
            settings,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = validate(parse_json_content(content))
        merged = {
            "evidence_id": evidence_id,
            "rag_query_id": clean(source.get("rag_query_id")),
            "rag_question": clean(source.get("rag_question")),
            "candidate": clean(source.get("candidate")),
            "chirality": clean(source.get("chirality")),
            "pmid": clean(source.get("pmid")),
            "pmcid": clean(source.get("pmcid")),
            "doi": clean(source.get("doi")),
            "title": clean(source.get("title")),
            "abstract_text": clean(source.get("abstract_text")),
            "medcpt_score": float(source.get("medcpt_score", 0) or 0),
            "medcpt_rank": int(float(source.get("medcpt_rank", 0) or 0)),
            **parsed,
        }
        results.append(merged)

        stem = f"{i:03d}_{clean(source.get('rag_query_id'))}_{evidence_id}"
        (raw_dir / f"{stem}_prompt.txt").write_text(
            SYSTEM_PROMPT + "\n\n" + user_prompt, encoding="utf-8"
        )
        (raw_dir / f"{stem}_response.txt").write_text(content, encoding="utf-8")
        (raw_dir / f"{stem}_meta.json").write_text(
            json.dumps(
                {
                    "model": safe_meta["model"],
                    "usage": raw_response.get("usage", {}),
                    "response_id": raw_response.get("id", ""),
                    "prompt_sha256": sha256_text(SYSTEM_PROMPT + "\n\n" + user_prompt),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{i}/{len(df)}] {merged['rag_query_id']} {evidence_id} "
            f"RCS={merged['relevance_score']} MedCPT={merged['medcpt_score']:.3f}"
        )

    jsonl_path = output_dir / "rag_contextual_evidence.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    flat_rows = []
    for item in results:
        row = dict(item)
        for key in (
            "reported_outcomes",
            "reported_numeric_results",
            "reported_qualitative_findings",
            "limitations_for_current_question",
        ):
            row[key] = " | ".join(row.get(key, []))
        flat_rows.append(row)
    csv_path = output_dir / "rag_contextual_evidence.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    manifest = {
        "step": "functional_block_rag_contextualization",
        "version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "source_file": str(input_path),
        "method": "PaperQA2-style contextual summarization/relevance scoring after MedCPT top-k reranking",
        "llm": safe_meta,
        "records": len(results),
        "rag": {
            "retrieval": "Europe PMC",
            "reranking": "MedCPT Cross-Encoder",
            "augmentation": "retrieved title/abstract plus query-specific contextual summary",
            "generation": "LLM contextual evidence generation",
            "is_rag_pipeline": True,
        },
        "scientific_boundary": {
            "relevance_score_is_not_immune_effect": True,
            "medcpt_score_is_not_immune_effect": True,
            "abstract_only": True,
            "numeric_results_require_explicit_numbers": True,
            "qualitative_findings_are_separate_from_numeric_results": True,
        },
        "paperqa2_prompt_reference": "Future-House/paper-qa src/paperqa/prompts.py summary_json_system_prompt (method adapted, not copied verbatim)",
        "rag_config": rag_cfg,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("RAG contextualization completed")
    print("Run:", output_dir)
    print("Evidence JSONL:", jsonl_path)
    print("Evidence CSV:", csv_path)


if __name__ == "__main__":
    main()
