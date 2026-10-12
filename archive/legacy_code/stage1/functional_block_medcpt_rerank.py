from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
RAG_CONFIG = ROOT / "configs" / "functional_block_rag_v1.json"


def clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def split_semicolon(value: str) -> list[str]:
    return [x.strip() for x in clean(value).split(";") if x.strip()]


def query_group(query_id: str) -> str:
    if query_id.startswith("BG"):
        return "background"
    if query_id.startswith(("LK", "DK", "KD")):
        return "lys_chirality"
    if query_id.startswith("SYN"):
        return "synthetic_feasibility"
    if query_id[:1] in {"K", "R", "H"} and query_id[1:].isdigit():
        return "candidate"
    return "other"


def article_text(row: pd.Series) -> str:
    title = clean(row.get("title"))
    abstract = clean(row.get("abstract_text"))
    if title and abstract:
        return f"{title} {abstract}"
    return title or abstract


def resolve_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_medcpt(model_id: str, device: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "MedCPT dependencies are missing. Install them with: "
            "python -m pip install -r requirements-stage1-rag.txt"
        ) from exc

    resolved = resolve_device(device, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(resolved)
    model.eval()
    return torch, tokenizer, model, resolved


def score_pairs(
    torch_module,
    tokenizer,
    model,
    device: str,
    question: str,
    articles: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(articles), batch_size):
        batch_articles = articles[start : start + batch_size]
        pairs = [[question, article] for article in batch_articles]
        encoded = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch_module.no_grad():
            logits = model(**encoded).logits
        batch_scores = logits.reshape(-1).detach().cpu().tolist()
        scores.extend(float(x) for x in batch_scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-rank Europe PMC retrieval results with the official NCBI MedCPT "
            "cross-encoder. Selection is top-k only; no score threshold is used."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="functional-block-screening-prep run directory containing screening_all.csv",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--queries",
        default="",
        help="Optional comma-separated query IDs for a trial run, e.g. K04,R04,H04,KD02",
    )
    args = parser.parse_args()

    source_run = Path(args.run_dir)
    if not source_run.is_absolute():
        source_run = ROOT / source_run
    input_path = source_run / "screening_all.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    cfg = json.loads(RAG_CONFIG.read_text(encoding="utf-8"))
    med_cfg = cfg["medcpt"]
    questions: dict[str, str] = cfg["query_questions"]
    selected_query_ids = [q.strip() for q in args.queries.split(",") if q.strip()]
    if selected_query_ids:
        unknown = sorted(set(selected_query_ids) - set(questions))
        if unknown:
            raise ValueError(f"Unknown query IDs: {unknown}")
        questions = {q: questions[q] for q in selected_query_ids}

    batch_size = args.batch_size or int(med_cfg["batch_size"])
    device_request = args.device or med_cfg.get("device", "auto")
    max_length = int(med_cfg["max_length"])
    model_id = med_cfg["model_id"]
    top_k_cfg = med_cfg["top_k"]

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if df.empty:
        raise RuntimeError("screening_all.csv is empty")

    print("Loading MedCPT model:", model_id)
    torch_module, tokenizer, model, device = load_medcpt(model_id, device_request)
    print("Device:", device)

    ranked_rows: list[dict] = []
    selected_rows: list[dict] = []

    for query_id, question in questions.items():
        mask = df["matched_queries"].map(lambda x: query_id in split_semicolon(x))
        subset = df.loc[mask].copy()
        group = query_group(query_id)
        top_k = int(top_k_cfg.get(group, 5))

        if subset.empty:
            print(f"[{query_id}] 0 candidates; skipped")
            continue

        articles = [article_text(row) for _, row in subset.iterrows()]
        valid_mask = [bool(x) for x in articles]
        subset = subset.loc[valid_mask].copy()
        articles = [x for x in articles if x]
        if subset.empty:
            print(f"[{query_id}] records existed but all lacked title/abstract; skipped")
            continue

        scores = score_pairs(
            torch_module,
            tokenizer,
            model,
            device,
            question,
            articles,
            batch_size=batch_size,
            max_length=max_length,
        )
        subset["medcpt_score"] = scores
        subset = subset.sort_values(
            ["medcpt_score", "best_retrieval_rank"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        subset["medcpt_rank"] = range(1, len(subset) + 1)
        subset["rag_query_id"] = query_id
        subset["rag_query_group"] = group
        subset["rag_question"] = question
        subset["selected_top_k"] = subset["medcpt_rank"] <= min(top_k, len(subset))

        ranked_rows.extend(subset.to_dict("records"))
        selected_rows.extend(subset.loc[subset["selected_top_k"]].to_dict("records"))
        best = float(subset.iloc[0]["medcpt_score"])
        print(
            f"[{query_id}] candidates={len(subset)} top_k={min(top_k, len(subset))} "
            f"best_score={best:.4f}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_DIR / f"{timestamp}__functional-block-medcpt-rerank"
    output_dir.mkdir(parents=True, exist_ok=False)

    ranked_df = pd.DataFrame(ranked_rows)
    selected_df = pd.DataFrame(selected_rows)
    ranked_path = output_dir / "medcpt_ranked_pairs.csv"
    selected_path = output_dir / "rag_query_document_pairs.csv"
    ranked_df.to_csv(ranked_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    if selected_df.empty:
        dedup_df = pd.DataFrame()
    else:
        dedup_rows = []
        for dedup_key, group_df in selected_df.groupby("dedup_key", sort=False):
            group_df = group_df.sort_values(
                ["medcpt_score", "medcpt_rank"], ascending=[False, True]
            )
            first = group_df.iloc[0].to_dict()
            first["selected_queries"] = ";".join(
                dict.fromkeys(group_df["rag_query_id"].astype(str).tolist())
            )
            first["selected_query_count"] = int(group_df["rag_query_id"].nunique())
            first["best_medcpt_score"] = float(group_df["medcpt_score"].max())
            dedup_rows.append(first)
        dedup_df = pd.DataFrame(dedup_rows)
    dedup_path = output_dir / "rag_unique_documents.csv"
    dedup_df.to_csv(dedup_path, index=False)

    manifest = {
        "step": "functional_block_medcpt_rerank",
        "version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "source_file": str(input_path),
        "model_id": model_id,
        "device": device,
        "selection_policy": "top_k_only_no_score_threshold",
        "top_k": top_k_cfg,
        "query_ids": list(questions),
        "ranked_query_document_pairs": int(len(ranked_df)),
        "selected_query_document_pairs": int(len(selected_df)),
        "selected_unique_documents": int(len(dedup_df)),
        "scientific_boundary": {
            "medcpt_score_meaning": "biomedical query-document relevance only",
            "medcpt_score_is_not_immune_effect": True,
            "no_hard_relevance_threshold": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("MedCPT re-ranking completed")
    print("Run:", output_dir)
    print("Selected query-document pairs:", len(selected_df))
    print("Selected unique documents:", len(dedup_df))
    print("Pairs:", selected_path)
    print("All rankings:", ranked_path)


if __name__ == "__main__":
    main()
