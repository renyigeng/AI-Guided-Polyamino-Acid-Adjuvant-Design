from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "functional_block_search.json"
RUNS_DIR = ROOT / "runs"

USER_AGENT = (
    "tiaozhansai-functional-block-retrieval/1.0 "
    "(competition reproducibility workflow)"
)

TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def api_get_json(base_url: str, params: dict) -> dict:
    url = f"{base_url}?{urlencode(params)}"

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
            )

            with urlopen(
                request,
                timeout=TIMEOUT_SECONDS,
            ) as response:
                payload = response.read()

            return json.loads(
                payload.decode("utf-8")
            )

        except Exception as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(
        f"Europe PMC request failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


def load_expansion_plan(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    if not isinstance(plan.get("rendered_candidate_queries"), dict):
        raise ValueError("Expansion plan lacks rendered_candidate_queries")
    return plan


def with_variant(item: dict, *, group: str, variant_id: str = "v0", variant_source: str = "canonical_v1_3", query: str | None = None, candidate: str | None = None, chirality: str | None = None) -> dict:
    result = {
        **item,
        "query_group": group,
        "query_variant_id": variant_id,
        "query_variant_source": variant_source,
    }
    if query is not None:
        result["query"] = query
    if candidate is not None:
        result["candidate"] = candidate
    if chirality is not None:
        result["chirality"] = chirality
    return result


def collect_queries(
    config: dict,
    expansion_plan: dict | None = None,
    selected_query_ids: set[str] | None = None,
) -> list[dict]:
    queries: list[dict] = []
    selected = selected_query_ids or set()

    def wanted(query_id: str) -> bool:
        return not selected or query_id in selected

    for item in config.get("background_queries", []):
        if wanted(item["id"]):
            queries.append(with_variant(item, group="background", candidate=""))

    rendered = (expansion_plan or {}).get("rendered_candidate_queries", {})
    for item in config.get("candidate_queries", []):
        if not wanted(item["id"]):
            continue
        # Preserve the frozen v1.3 query as v0 so multi-query expansion cannot
        # silently remove evidence that the original protocol could retrieve.
        queries.append(with_variant(item, group="candidate"))
        for variant in rendered.get(item["id"], []):
            variant_id = str(variant.get("variant_id", "")).strip()
            query_text = str(variant.get("query", "")).strip()
            if not variant_id or variant_id == "v0" or not query_text:
                raise ValueError(f"Invalid expansion variant for {item['id']}: {variant}")
            queries.append(
                with_variant(
                    item,
                    group="candidate",
                    variant_id=variant_id,
                    variant_source=str(variant.get("source", "llm_candidate_neutral_dimension_expansion")),
                    query=query_text,
                )
            )

    for item in config.get("lys_chirality_queries", []):
        if wanted(item["id"]):
            queries.append(with_variant(item, group="lys_chirality"))

    for item in config.get("synthetic_feasibility_queries", []):
        if wanted(item["id"]):
            queries.append(
                with_variant(
                    item,
                    group="synthetic_feasibility",
                    candidate="",
                    chirality="",
                )
            )

    return queries


def fetch_query(
    config: dict,
    query_spec: dict,
    raw_dir: Path,
) -> tuple[list[dict], dict]:
    database = config["database"]
    policy = config["retrieval_policy"]

    base_url = database["api"]
    page_size = int(database.get("page_size", 100))
    max_results = int(
        policy.get("max_results_per_query", 200)
    )

    query_id = query_spec["id"]
    query_text = query_spec["query"]

    cursor_mark = "*"
    all_results = []
    page_number = 0
    total_hit_count = None

    query_variant_id = query_spec.get("query_variant_id", "v0")
    raw_query_dir = raw_dir / query_id / query_variant_id
    raw_query_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    while len(all_results) < max_results:
        remaining = max_results - len(all_results)

        current_page_size = min(
            page_size,
            remaining,
        )

        params = {
            "query": query_text,
            "format": database.get(
                "format",
                "json",
            ),
            "resultType": database.get(
                "result_type",
                "core",
            ),
            "pageSize": current_page_size,
            "cursorMark": cursor_mark,
            "synonym": str(
                database.get(
                    "synonym_expansion",
                    False,
                )
            ).lower(),
        }

        page_number += 1

        response = api_get_json(
            base_url,
            params,
        )

        raw_path = (
            raw_query_dir
            / f"page_{page_number:03d}.json"
        )

        with raw_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                response,
                f,
                ensure_ascii=False,
                indent=2,
            )

        if total_hit_count is None:
            total_hit_count = int(
                response.get(
                    "hitCount",
                    0,
                )
            )

        result_list = (
            response
            .get("resultList", {})
            .get("result", [])
        )

        if not result_list:
            break

        all_results.extend(result_list)

        if len(all_results) >= max_results:
            break

        next_cursor = response.get(
            "nextCursorMark"
        )

        if not next_cursor:
            break

        if next_cursor == cursor_mark:
            break

        cursor_mark = next_cursor

        # 保守限速，避免连续请求过快
        time.sleep(0.25)

    all_results = all_results[:max_results]

    summary = {
        "query_id": query_id,
        "query_variant_id": query_variant_id,
        "query_variant_source": query_spec.get("query_variant_source", "canonical_v1_3"),
        "query_group": query_spec["query_group"],
        "candidate": query_spec.get(
            "candidate",
            "",
        ),
        "chirality": query_spec.get(
            "chirality",
            "",
        ),
        "purpose": query_spec.get(
            "purpose",
            "",
        ),
        "evidence_role": query_spec.get("evidence_role", ""),
        "exact_query": query_text,
        "total_hit_count": total_hit_count or 0,
        "retrieved_count": len(all_results),
        "page_count": page_number,
    }

    return all_results, summary


def flatten_record(
    query_spec: dict,
    record: dict,
) -> dict:
    return {
        "query_id": query_spec["id"],
        "query_variant_id": query_spec.get("query_variant_id", "v0"),
        "query_variant_source": query_spec.get("query_variant_source", "canonical_v1_3"),
        "query_text": query_spec.get("query", ""),
        "query_group": query_spec["query_group"],
        "candidate": query_spec.get(
            "candidate",
            "",
        ),
        "chirality": query_spec.get(
            "chirality",
            "",
        ),
        "evidence_role": query_spec.get("evidence_role", ""),

        "source": record.get("source", ""),
        "id": record.get("id", ""),
        "pmid": record.get("pmid", ""),
        "pmcid": record.get("pmcid", ""),
        "doi": record.get("doi", ""),

        "title": record.get("title", ""),
        "author_string": record.get(
            "authorString",
            "",
        ),
        "journal_title": record.get(
            "journalTitle",
            "",
        ),
        "pub_year": record.get(
            "pubYear",
            "",
        ),
        "first_pub_date": record.get(
            "firstPublicationDate",
            "",
        ),
        "pub_type": "; ".join(
            record.get("pubTypeList", {}).get(
                "pubType",
                [],
            )
        ),

        "cited_by_count": record.get(
            "citedByCount",
            "",
        ),

        "is_open_access": record.get(
            "isOpenAccess",
            "",
        ),
        "in_epmc": record.get(
            "inEPMC",
            "",
        ),
        "in_pmc": record.get(
            "inPMC",
            "",
        ),

        "abstract_text": record.get(
            "abstractText",
            "",
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve functional-block literature from Europe PMC. Optionally apply a frozen "
            "candidate-neutral multi-query expansion plan to K/R/H candidate dimensions."
        )
    )
    parser.add_argument(
        "--expansion-plan",
        default="",
        help="Path to query_expansion_plan.json; omit for historical v1.3 single-query behavior",
    )
    parser.add_argument(
        "--queries",
        default="",
        help="Optional comma-separated canonical query IDs for a trial run",
    )
    args = parser.parse_args()

    config = load_config()
    expansion_path = Path(args.expansion_plan) if args.expansion_plan else None
    if expansion_path is not None and not expansion_path.is_absolute():
        expansion_path = ROOT / expansion_path
    expansion_plan = load_expansion_plan(expansion_path)
    selected_query_ids = {x.strip() for x in args.queries.split(",") if x.strip()}

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    run_dir = (
        RUNS_DIR
        / f"{timestamp}__functional-block-literature-retrieval"
    )

    raw_dir = run_dir / "raw_api"

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_snapshot = (
        run_dir
        / "functional_block_search_snapshot.json"
    )

    shutil.copy2(
        CONFIG_PATH,
        config_snapshot,
    )

    expansion_snapshot = None
    if expansion_path is not None:
        expansion_snapshot = run_dir / "functional_block_query_expansion_snapshot.json"
        shutil.copy2(expansion_path, expansion_snapshot)

    queries = collect_queries(config, expansion_plan, selected_query_ids)

    all_metadata = []
    query_summaries = []

    print("=" * 72)
    print("Functional-block literature retrieval")
    print("=" * 72)
    print(f"Run: {run_dir}")
    print(f"Queries: {len(queries)}")
    print()

    for query_spec in queries:
        print(
            f"[{query_spec['id']}:{query_spec.get('query_variant_id', 'v0')}] "
            f"{query_spec.get('purpose', '')}"
        )

        records, summary = fetch_query(
            config,
            query_spec,
            raw_dir,
        )

        query_summaries.append(summary)

        for record in records:
            all_metadata.append(
                flatten_record(
                    query_spec,
                    record,
                )
            )

        print(
            f"  total hits : "
            f"{summary['total_hit_count']}"
        )
        print(
            f"  retrieved  : "
            f"{summary['retrieved_count']}"
        )
        print()

    metadata_path = (
        run_dir
        / "retrieved_metadata.csv"
    )

    write_csv(
        metadata_path,
        all_metadata,
    )

    query_summary_path = (
        run_dir
        / "query_summary.csv"
    )

    write_csv(
        query_summary_path,
        query_summaries,
    )

    manifest = {
        "schema_version": "1.0",
        "step": "functional_block_literature_retrieval",

        "started_from_config": str(
            CONFIG_PATH.relative_to(ROOT)
        ),

        "config_sha256": sha256_file(
            config_snapshot
        ),

        "runtime_utc": timestamp,

        "retrieval_backend": {
            "name": config["database"]["name"],
            "api": config["database"]["api"],
            "format": config["database"].get(
                "format",
                "json",
            ),
            "result_type": config["database"].get(
                "result_type",
                "core",
            ),
        },

        "query_variant_count": len(queries),
        "canonical_query_count": len({q["id"] for q in queries}),
        "multi_query_expansion": {
            "enabled": expansion_plan is not None,
            "plan_snapshot": (
                "functional_block_query_expansion_snapshot.json"
                if expansion_snapshot is not None else None
            ),
            "policy": "canonical_v0_plus_candidate-neutral dimension variants; union/dedup occurs during screening preparation",
        },

        "queries": query_summaries,

        "outputs": {
            "metadata_csv": (
                "retrieved_metadata.csv"
            ),
            "query_summary_csv": (
                "query_summary.csv"
            ),
            "raw_api_directory": "raw_api",
            "config_snapshot": (
                "functional_block_search_snapshot.json"
            ),
            "query_expansion_snapshot": (
                "functional_block_query_expansion_snapshot.json"
                if expansion_snapshot is not None else None
            ),
        },
    }

    manifest_path = (
        run_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("Retrieval completed")
    print("=" * 72)
    print(
        f"Metadata : {metadata_path}"
    )
    print(
        f"Summary  : {query_summary_path}"
    )
    print(
        f"Manifest : {manifest_path}"
    )
    print(
        f"Raw JSON : {raw_dir}"
    )


if __name__ == "__main__":
    main()