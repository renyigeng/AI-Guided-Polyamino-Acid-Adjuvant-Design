from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .llm_compatible_client import call_chat, load_settings, parse_json_content
except ImportError:  # pragma: no cover
    from llm_compatible_client import call_chat, load_settings, parse_json_content

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dlys_domain_indicator_discovery_v2.json"
DEFAULT_PROMPT = ROOT / "prompts" / "dlys_domain_indicator_discovery_v2.txt"
RUN_SUFFIX = "__dlys-domain-indicator-discovery-v2"
MAX_LLM_ATTEMPTS = 3
EXPECTED_CANDIDATES = set("ACDEFGILMNPQSTVWY")

ROLE_ALIASES = {
    "primary": "primary_rank",
    "primary_indicator": "primary_rank",
    "primary_criterion": "primary_rank",
    "ranking": "primary_rank",
    "rank": "primary_rank",
    "gate": "hard_gate",
    "hard": "hard_gate",
    "hardgate": "hard_gate",
    "tie": "tie_break",
    "tiebreak": "tie_break",
    "tie-break": "tie_break",
    "validation": "validation_only",
    "validate": "validation_only",
}
DIRECTION_ALIASES = {
    "maximize": "higher",
    "benefit": "higher",
    "high": "higher",
    "minimize": "lower",
    "cost": "lower",
    "low": "lower",
    "near_zero": "closer_to_zero",
    "near-neutral": "closer_to_zero",
    "near_neutral": "closer_to_zero",
    "neutral": "closer_to_zero",
    "present": "presence",
    "exists": "presence",
    "absent": "absence",
    "none": "absence",
    "context": "context_dependent",
    "depends": "context_dependent",
}
BASIS_ALIASES = {
    "evidence": "literature",
    "literature_evidence": "literature",
    "system": "system_constraint",
    "design_constraint": "system_constraint",
    "literature+system": "both",
    "system+literature": "both",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y", "t"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_qualification_run(runs_dir: Path) -> Path:
    matches = sorted(
        p
        for p in runs_dir.glob("*__mechanism-role-refinement-v1")
        if p.is_dir() and (p / "evidence_qualification.csv").exists()
    )
    if not matches:
        raise FileNotFoundError(
            "No mechanism-role-refinement-v1 run containing evidence_qualification.csv found"
        )
    return matches[-1]


def latest_v2_run(runs_dir: Path) -> Path:
    matches = sorted(p for p in runs_dir.glob(f"*{RUN_SUFFIX}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"No prior {RUN_SUFFIX} run found")
    return matches[-1]


def _taxonomy_contract(config: dict[str, Any]) -> dict[str, Any] | None:
    rel = clean(config.get("taxonomy_contract"))
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"taxonomy contract not found: {path}")
    return load_json(path)


def validate_taxonomy(config: dict[str, Any]) -> None:
    domains = config["preregistered_domains"]
    domain_ids = [d["domain_id"] for d in domains]
    if domain_ids != ["D1", "D2", "D3"]:
        raise ValueError("Stage-2 taxonomy must contain D1, D2, D3 in that order")
    assigned = [aa for d in domains for aa in d["candidates"]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Each stage-2 candidate must belong to exactly one domain")
    if set(assigned) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"Stage-2 taxonomy must cover all 17 candidates exactly once; "
            f"missing={sorted(EXPECTED_CANDIDATES-set(assigned))}, "
            f"extra={sorted(set(assigned)-EXPECTED_CANDIDATES)}"
        )
    names = set(config["candidate_names_zh"])
    if names != EXPECTED_CANDIDATES:
        raise ValueError("candidate_names_zh must contain exactly the frozen 17 candidates")

    contract = _taxonomy_contract(config)
    if contract is not None:
        frozen = {d["domain_id"]: list(d["candidates"]) for d in contract["domains"]}
        current = {d["domain_id"]: list(d["candidates"]) for d in domains}
        if current != frozen:
            raise ValueError(
                "Discovery taxonomy drifted from the frozen stage-2 taxonomy contract: "
                f"current={current}; frozen={frozen}"
            )
        contract_names = set(contract.get("candidate_names_zh", {}))
        if contract_names and contract_names != names:
            raise ValueError("candidate_names_zh drifted from the frozen stage-2 taxonomy contract")


def build_domain_packets(qdf: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_taxonomy(config)
    required = {
        "item_id",
        "candidate",
        "candidate_name_zh",
        "interaction_type",
        "support_level",
        "role_summary",
        "evidence_summary",
        "title",
        "abstract",
        "qualification",
        "native_x_attribution",
        "primary_eligible",
    }
    missing = required - set(qdf.columns)
    if missing:
        raise ValueError(f"qualification table missing columns: {sorted(missing)}")

    work = qdf.copy()
    if config["evidence"].get("use_primary_eligible_only", True):
        work = work[work["primary_eligible"].map(truthy)].copy()

    max_items = int(config["evidence"]["max_items_per_candidate"])
    max_chars = int(config["evidence"]["max_abstract_chars"])
    names = config["candidate_names_zh"]
    packets: list[dict[str, Any]] = []

    for domain in config["preregistered_domains"]:
        evidence: list[dict[str, Any]] = []
        evidence_counts: dict[str, int] = {}
        for aa in domain["candidates"]:
            sub = work[work["candidate"].astype(str) == aa].head(max_items)
            evidence_counts[aa] = int(len(sub))
            for _, row in sub.iterrows():
                evidence.append(
                    {
                        "item_id": clean(row["item_id"]),
                        "candidate": aa,
                        "candidate_name_zh": names[aa],
                        "interaction_type": clean(row["interaction_type"]),
                        "support_level": clean(row["support_level"]),
                        "role_summary": clean(row["role_summary"]),
                        "evidence_summary": clean(row["evidence_summary"]),
                        "title": clean(row["title"]),
                        "abstract": clean(row["abstract"])[:max_chars],
                        "qualification": clean(row["qualification"]),
                        "native_x_attribution": clean(row["native_x_attribution"]),
                    }
                )

        packets.append(
            {
                "domain_id": domain["domain_id"],
                "domain_name_zh": domain["name_zh"],
                "candidate_codes": list(domain["candidates"]),
                "candidate_names_zh": [names[aa] for aa in domain["candidates"]],
                "design_goal_zh": domain["design_goal_zh"],
                "focus_questions_zh": list(domain["focus_questions_zh"]),
                "decision_dimensions_zh": list(domain.get("decision_dimensions_zh", [])),
                "direction_guardrails_zh": list(domain.get("direction_guardrails_zh", [])),
                "interpretation_flags": list(domain.get("interpretation_flags", [])),
                "evidence_counts": evidence_counts,
                "evidence": evidence,
            }
        )
    return packets


def discovery_schema(domain_id: str) -> dict[str, Any]:
    return {
        "domain_id": domain_id,
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "one concise measurable physicochemical descriptor",
                "measurement_target": "standard scientific property name; one property only",
                "scientific_definition": "what is measured",
                "why_relevant_to_dlys_system": "why it helps compare candidates in this domain after fixing D-Lys",
                "role": "primary_rank/hard_gate/tie_break/validation_only",
                "preferred_direction": "higher/lower/closer_to_zero/presence/absence/target/context_dependent",
                "target_value": None,
                "basis": "literature/system_constraint/both",
                "supporting_evidence_ids": ["item_id from supplied evidence when applicable"],
                "quantifiable_single_residue": True,
                "redundancy_note": "why this is distinct from other returned indicators",
                "notes": "scope/limitations",
            }
        ],
        "knowledge_gaps": ["scientific gaps or unavailable mature single-residue metrics"],
        "notes": ["domain-level notes"],
    }


def build_discovery_prompt(packet: dict[str, Any], config: dict[str, Any], prompt_path: Path) -> str:
    base = prompt_path.read_text(encoding="utf-8")
    context = {
        "stage1_completed": config["stage_context"]["stage1_completed"],
        "fixed_functional_block": config["stage_context"]["fixed_functional_block"],
        "fixed_role": config["stage_context"]["fixed_role"],
        "stage2_target": config["stage_context"]["stage2_target"],
    }
    domain = {
        "domain_id": packet["domain_id"],
        "domain_name_zh": packet["domain_name_zh"],
        "candidate_codes": packet["candidate_codes"],
        "candidate_names_zh": packet["candidate_names_zh"],
        "design_goal_zh": packet["design_goal_zh"],
        "focus_questions_zh": packet["focus_questions_zh"],
        "decision_dimensions_zh": packet.get("decision_dimensions_zh", []),
        "direction_guardrails_zh": packet.get("direction_guardrails_zh", []),
        "interpretation_flags": packet["interpretation_flags"],
        "evidence_counts_for_audit_only": packet["evidence_counts"],
    }
    policy = {
        "aim_primary_rank_count": [
            config["indicator_policy"]["aim_primary_rank_count_min"],
            config["indicator_policy"]["aim_primary_rank_count_max"],
        ],
        "allow_fewer_if_evidence_insufficient": config["indicator_policy"]["allow_fewer_if_evidence_insufficient"],
        "require_nonredundant": config["indicator_policy"]["require_nonredundant"],
        "require_single_property_per_indicator": config["indicator_policy"]["require_single_property_per_indicator"],
        "allowed_roles": config["indicator_policy"]["allowed_roles"],
        "allowed_directions": config["indicator_policy"]["allowed_directions"],
        "allowed_basis": config["indicator_policy"]["allowed_basis"],
        "primary_rank_requires_actionable_direction": config["indicator_policy"].get(
            "primary_rank_requires_actionable_direction", True
        ),
        "context_dependent_primary_policy": config["indicator_policy"].get(
            "context_dependent_primary_policy", "downgrade_to_validation_only"
        ),
    }
    return (
        base
        + "\n\nFIXED SYSTEM CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\nPREREGISTERED DOMAIN AND HUMAN SCIENTIFIC GUIDANCE:\n"
        + json.dumps(domain, ensure_ascii=False, indent=2)
        + "\n\nINDICATOR POLICY:\n"
        + json.dumps(policy, ensure_ascii=False, indent=2)
        + "\n\nQUALIFIED NATIVE-X EVIDENCE:\n"
        + json.dumps(packet["evidence"], ensure_ascii=False, indent=2)
        + "\n\nRETURN SCHEMA:\n"
        + json.dumps(discovery_schema(packet["domain_id"]), ensure_ascii=False, indent=2)
    )


def _normalize_enum(
    value: Any,
    allowed: set[str],
    aliases: dict[str, str],
    *,
    default: str,
    field_name: str,
    indicator_id: str,
    warnings: list[dict[str, str]],
) -> str:
    raw = clean(value).lower().replace(" ", "_")
    raw = aliases.get(raw, raw)
    if raw in allowed:
        return raw
    warnings.append(
        {
            "indicator_id": indicator_id,
            "field": field_name,
            "issue": "missing_or_invalid_enum",
            "original_value": clean(value),
            "action": f"normalized_to_{default}",
        }
    )
    return default


def validate_discovery(obj: Any, packet: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("indicator discovery must return a JSON object")
    if clean(obj.get("domain_id")) != packet["domain_id"]:
        raise ValueError("indicator discovery domain_id mismatch")

    allowed_roles = set(config["indicator_policy"]["allowed_roles"])
    allowed_directions = set(config["indicator_policy"]["allowed_directions"])
    allowed_basis = set(config["indicator_policy"]["allowed_basis"])
    valid_evidence_ids = {x["item_id"] for x in packet["evidence"]}
    allow_system_only = bool(config["evidence"].get("allow_system_constraint_indicator_without_literature_id", True))

    indicators: list[dict[str, Any]] = []
    validation_warnings: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in obj.get("indicators", []) or []:
        if not isinstance(raw, dict):
            raise ValueError("each indicator must be a JSON object")
        indicator_id = clean(raw.get("indicator_id"))
        if not indicator_id or indicator_id in seen:
            raise ValueError("indicator_id must be unique and non-empty")
        seen.add(indicator_id)

        role = _normalize_enum(
            raw.get("role"), allowed_roles, ROLE_ALIASES,
            default="validation_only", field_name="role", indicator_id=indicator_id,
            warnings=validation_warnings,
        )
        direction = _normalize_enum(
            raw.get("preferred_direction"), allowed_directions, DIRECTION_ALIASES,
            default="context_dependent", field_name="preferred_direction", indicator_id=indicator_id,
            warnings=validation_warnings,
        )
        basis = _normalize_enum(
            raw.get("basis", "literature"), allowed_basis, BASIS_ALIASES,
            default="literature", field_name="basis", indicator_id=indicator_id,
            warnings=validation_warnings,
        )

        raw_evidence_ids = [clean(x) for x in (raw.get("supporting_evidence_ids") or []) if clean(x)]
        evidence_ids = [x for x in raw_evidence_ids if x in valid_evidence_ids]
        unknown = sorted(set(raw_evidence_ids) - valid_evidence_ids)
        if unknown:
            validation_warnings.append(
                {
                    "indicator_id": indicator_id,
                    "field": "supporting_evidence_ids",
                    "issue": "unknown_evidence_ids_filtered",
                    "original_value": "|".join(unknown),
                    "action": "filtered_unknown_ids",
                }
            )

        if basis == "literature" and not evidence_ids:
            if role != "validation_only":
                role = "validation_only"
            validation_warnings.append(
                {
                    "indicator_id": indicator_id,
                    "field": "supporting_evidence_ids",
                    "issue": "literature_indicator_without_valid_evidence",
                    "original_value": "|".join(raw_evidence_ids),
                    "action": "downgraded_to_validation_only",
                }
            )
        elif basis == "both" and not evidence_ids:
            if allow_system_only:
                basis = "system_constraint"
                validation_warnings.append(
                    {
                        "indicator_id": indicator_id,
                        "field": "basis",
                        "issue": "both_without_valid_literature_evidence",
                        "original_value": "both",
                        "action": "downgraded_basis_to_system_constraint",
                    }
                )
            else:
                role = "validation_only"
                validation_warnings.append(
                    {
                        "indicator_id": indicator_id,
                        "field": "supporting_evidence_ids",
                        "issue": "both_without_valid_evidence_under_config",
                        "original_value": "|".join(raw_evidence_ids),
                        "action": "downgraded_to_validation_only",
                    }
                )
        elif basis == "system_constraint" and not evidence_ids and not allow_system_only:
            role = "validation_only"
            validation_warnings.append(
                {
                    "indicator_id": indicator_id,
                    "field": "supporting_evidence_ids",
                    "issue": "system_constraint_without_evidence_disallowed_by_config",
                    "original_value": "",
                    "action": "downgraded_to_validation_only",
                }
            )

        indicator_name = clean(raw.get("indicator_name"))
        measurement_target = clean(raw.get("measurement_target")) or indicator_name
        definition = clean(raw.get("scientific_definition"))
        relevance = clean(raw.get("why_relevant_to_dlys_system"))
        if not indicator_name or not measurement_target or not definition or not relevance:
            raise ValueError(f"indicator {indicator_id} missing required scientific text")

        target_value = raw.get("target_value")
        if direction == "target":
            try:
                if target_value is None or clean(target_value) == "":
                    raise ValueError
                target_value = float(target_value)
            except (TypeError, ValueError):
                direction = "context_dependent"
                role = "validation_only"
                target_value = None
                validation_warnings.append(
                    {
                        "indicator_id": indicator_id,
                        "field": "target_value",
                        "issue": "target_direction_without_numeric_target",
                        "original_value": clean(raw.get("target_value")),
                        "action": "downgraded_to_validation_only_and_context_dependent",
                    }
                )
        else:
            target_value = None

        if (
            role == "primary_rank"
            and direction == "context_dependent"
            and config["indicator_policy"].get("primary_rank_requires_actionable_direction", True)
        ):
            role = "validation_only"
            validation_warnings.append(
                {
                    "indicator_id": indicator_id,
                    "field": "preferred_direction",
                    "issue": "primary_rank_without_actionable_direction",
                    "original_value": clean(raw.get("preferred_direction")),
                    "action": "downgraded_to_validation_only",
                }
            )

        quantifiable = raw.get("quantifiable_single_residue")
        if isinstance(quantifiable, str):
            quantifiable = truthy(quantifiable)
        quantifiable = bool(quantifiable)

        indicators.append(
            {
                "indicator_id": indicator_id,
                "indicator_name": indicator_name,
                "measurement_target": measurement_target,
                "scientific_definition": definition,
                "why_relevant_to_dlys_system": relevance,
                "role": role,
                "preferred_direction": direction,
                "target_value": target_value,
                "basis": basis,
                "supporting_evidence_ids": evidence_ids,
                "quantifiable_single_residue": quantifiable,
                "redundancy_note": clean(raw.get("redundancy_note")),
                "notes": clean(raw.get("notes")),
            }
        )

    if not indicators:
        raise ValueError("indicator discovery returned no indicators")

    return {
        "domain_id": packet["domain_id"],
        "domain_name_zh": packet["domain_name_zh"],
        "candidate_codes": packet["candidate_codes"],
        "candidate_names_zh": packet["candidate_names_zh"],
        "indicators": indicators,
        "validation_warnings": validation_warnings,
        "knowledge_gaps": [clean(x) for x in (obj.get("knowledge_gaps") or []) if clean(x)],
        "notes": [clean(x) for x in (obj.get("notes") or []) if clean(x)],
    }


def _messages(system_text: str, user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def call_and_validate(
    settings: Any,
    packet: dict[str, Any],
    config: dict[str, Any],
    prompt_text: str,
    raw_dir: Path,
    *,
    max_attempts: int = MAX_LLM_ATTEMPTS,
) -> dict[str, Any]:
    """Call the LLM for one domain with transport/JSON retries only.

    Every retry sends the *same* scientific system/user messages. We retry only
    when the provider returns no content, the provider call itself fails, or the
    returned text cannot be parsed as JSON. Once JSON parses, scientific/schema
    validation runs exactly once and any remaining semantic error is surfaced
    instead of asking the LLM to rewrite the science.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    system_text = (
        "Return strict JSON only. Preserve scientific uncertainty. "
        "Never rank candidates, reveal a winner, assign numerical weights, or invent custom metrics."
    )
    messages = _messages(system_text, prompt_text)
    stem = f"{packet['domain_id']}_indicator_discovery"
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_error = "unknown error"

    for attempt in range(1, max_attempts + 1):
        attempt_stem = f"{stem}_attempt_{attempt:02d}"
        try:
            content, raw = call_chat(settings, messages)
        except Exception as exc:
            last_error = f"provider_call_error: {type(exc).__name__}: {exc}"
            write_json(
                raw_dir / f"{attempt_stem}_status.json",
                {
                    "domain_id": packet["domain_id"],
                    "attempt": attempt,
                    "status": "provider_call_error",
                    "error": last_error,
                },
            )
            if attempt < max_attempts:
                print(
                    f"[LLM_RETRY] {packet['domain_id']} attempt {attempt}/{max_attempts} "
                    f"provider call failed; retrying same prompt"
                )
                continue
            break

        # Preserve every provider response for audit, including empty/invalid text.
        response_text = "" if content is None else str(content)
        (raw_dir / f"{attempt_stem}_response.txt").write_text(response_text, encoding="utf-8")
        write_json(raw_dir / f"{attempt_stem}_raw.json", raw)

        if not response_text.strip():
            last_error = "empty_or_whitespace_response"
            write_json(
                raw_dir / f"{attempt_stem}_status.json",
                {
                    "domain_id": packet["domain_id"],
                    "attempt": attempt,
                    "status": "empty_response",
                    "error": last_error,
                },
            )
            if attempt < max_attempts:
                print(
                    f"[LLM_RETRY] {packet['domain_id']} attempt {attempt}/{max_attempts} "
                    f"returned empty content; retrying same prompt"
                )
                continue
            break

        try:
            # parse_json_content may perform local syntax repair. We never send the
            # scientific answer back to the LLM for a semantic rewrite.
            obj = parse_json_content(response_text)
        except Exception as exc:
            last_error = f"json_parse_error: {type(exc).__name__}: {exc}"
            write_json(
                raw_dir / f"{attempt_stem}_status.json",
                {
                    "domain_id": packet["domain_id"],
                    "attempt": attempt,
                    "status": "json_parse_error",
                    "error": last_error,
                },
            )
            if attempt < max_attempts:
                print(
                    f"[LLM_RETRY] {packet['domain_id']} attempt {attempt}/{max_attempts} "
                    f"returned unparsable JSON; retrying same prompt"
                )
                continue
            break

        # Compatibility files point to the accepted provider response.
        (raw_dir / f"{stem}_response.txt").write_text(response_text, encoding="utf-8")
        write_json(raw_dir / f"{stem}_raw.json", raw)
        write_json(
            raw_dir / f"{attempt_stem}_status.json",
            {
                "domain_id": packet["domain_id"],
                "attempt": attempt,
                "status": "json_parsed",
            },
        )

        # Scientific/schema validation is deliberately NOT a retry trigger.
        return validate_discovery(obj, packet, config)

    raise RuntimeError(
        f"LLM indicator discovery failed for domain {packet['domain_id']} after "
        f"{max_attempts} attempts; last_error={last_error}"
    )

def flatten_indicators(discoveries: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for discovery in discoveries:
        for indicator in discovery["indicators"]:
            rows.append(
                {
                    "domain_id": discovery["domain_id"],
                    "domain_name_zh": discovery["domain_name_zh"],
                    "indicator_id": indicator["indicator_id"],
                    "indicator_name": indicator["indicator_name"],
                    "measurement_target": indicator["measurement_target"],
                    "scientific_definition": indicator["scientific_definition"],
                    "why_relevant_to_dlys_system": indicator["why_relevant_to_dlys_system"],
                    "role": indicator["role"],
                    "preferred_direction": indicator["preferred_direction"],
                    "target_value": indicator["target_value"],
                    "basis": indicator["basis"],
                    "quantifiable_single_residue": indicator["quantifiable_single_residue"],
                    "supporting_evidence_ids": "|".join(indicator["supporting_evidence_ids"]),
                    "redundancy_note": indicator["redundancy_note"],
                    "notes": indicator["notes"],
                }
            )
    return pd.DataFrame(rows)



def flatten_validation_warnings(discoveries: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for discovery in discoveries:
        for warning in discovery.get("validation_warnings", []):
            rows.append(
                {
                    "domain_id": discovery["domain_id"],
                    "domain_name_zh": discovery["domain_name_zh"],
                    "indicator_id": warning.get("indicator_id", ""),
                    "field": warning.get("field", ""),
                    "issue": warning.get("issue", ""),
                    "original_value": warning.get("original_value", ""),
                    "action": warning.get("action", ""),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "domain_id",
            "domain_name_zh",
            "indicator_id",
            "field",
            "issue",
            "original_value",
            "action",
        ],
    )

def build_indicator_evidence(discoveries: list[dict[str, Any]], qdf: pd.DataFrame) -> pd.DataFrame:
    qindex = {clean(row["item_id"]): row for _, row in qdf.iterrows()}
    rows: list[dict[str, Any]] = []
    for discovery in discoveries:
        for indicator in discovery["indicators"]:
            if not indicator["supporting_evidence_ids"]:
                rows.append(
                    {
                        "domain_id": discovery["domain_id"],
                        "indicator_id": indicator["indicator_id"],
                        "indicator_name": indicator["indicator_name"],
                        "basis": indicator["basis"],
                        "item_id": "",
                        "candidate": "",
                        "candidate_name_zh": "",
                        "title": "",
                        "doi": "",
                        "pmid": "",
                        "evidence_summary": "",
                    }
                )
                continue
            for item_id in indicator["supporting_evidence_ids"]:
                row = qindex[item_id]
                rows.append(
                    {
                        "domain_id": discovery["domain_id"],
                        "indicator_id": indicator["indicator_id"],
                        "indicator_name": indicator["indicator_name"],
                        "basis": indicator["basis"],
                        "item_id": item_id,
                        "candidate": clean(row.get("candidate")),
                        "candidate_name_zh": clean(row.get("candidate_name_zh")),
                        "title": clean(row.get("title")),
                        "doi": clean(row.get("doi")),
                        "pmid": clean(row.get("pmid")),
                        "evidence_summary": clean(row.get("evidence_summary")),
                    }
                )
    return pd.DataFrame(rows)


def create_output_dir(repo_root: Path, explicit: str | None, resume: str | None) -> Path:
    if resume:
        if resume == "latest":
            out = latest_v2_run(repo_root / "runs")
        else:
            out = Path(resume)
            if not out.is_absolute():
                out = repo_root / out
        if not out.exists():
            raise FileNotFoundError(out)
        return out.resolve()
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = repo_root / out
        out = out.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = repo_root / "runs" / f"{stamp}{RUN_SUFFIX}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="D-Lys stage-2 three-domain indicator discovery V2")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT.relative_to(ROOT)))
    parser.add_argument("--qualification-run")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-run", help="existing V2 run path or 'latest'")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    prompt_path = Path(args.prompt)
    if not prompt_path.is_absolute():
        prompt_path = repo_root / prompt_path
    config = load_json(config_path)
    validate_taxonomy(config)
    taxonomy_contract_path = Path(config.get("taxonomy_contract", ""))
    if taxonomy_contract_path and not taxonomy_contract_path.is_absolute():
        taxonomy_contract_path = repo_root / taxonomy_contract_path
    taxonomy_contract_path = taxonomy_contract_path.resolve() if str(taxonomy_contract_path) else None

    if args.qualification_run:
        qualification_run = Path(args.qualification_run)
        if not qualification_run.is_absolute():
            qualification_run = repo_root / qualification_run
        qualification_run = qualification_run.resolve()
    else:
        qualification_run = latest_qualification_run(repo_root / "runs")
    qualification_csv = qualification_run / "evidence_qualification.csv"
    if not qualification_csv.exists():
        raise FileNotFoundError(qualification_csv)
    qdf = pd.read_csv(qualification_csv)
    packets = build_domain_packets(qdf, config)

    out = create_output_dir(repo_root, args.output_dir, args.resume_run)
    prompts_dir = out / "prepared_prompts"
    raw_dir = out / "llm_raw"
    checkpoints_dir = out / "checkpoints"
    for directory in (prompts_dir, raw_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    taxonomy = {
        "schema_version": config["schema_version"],
        "stage1_fixed_functional_block": config["stage_context"]["fixed_functional_block"],
        "domains": config["preregistered_domains"],
        "candidate_names_zh": config["candidate_names_zh"],
        "all_17_candidates_ranked": True,
        "domain_membership_source": "human_preregistered_material_mechanism_taxonomy",
    }
    write_json(out / "domain_taxonomy.json", taxonomy)

    prompt_map: dict[str, str] = {}
    for packet in packets:
        text = build_discovery_prompt(packet, config, prompt_path)
        prompt_map[packet["domain_id"]] = text
        (prompts_dir / f"indicator_discovery_{packet['domain_id']}.txt").write_text(text, encoding="utf-8")

    if args.prepare_only:
        manifest = {
            "schema_version": config["schema_version"],
            "status": "prepared_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "qualification_run": str(qualification_run),
            "qualification_sha256": sha256_file(qualification_csv),
            "config_sha256": sha256_file(config_path),
            "taxonomy_contract": str(taxonomy_contract_path) if taxonomy_contract_path else None,
            "taxonomy_contract_sha256": sha256_file(taxonomy_contract_path) if taxonomy_contract_path else None,
            "prompt_template_sha256": sha256_file(prompt_path),
            "stage1_dlys_already_selected": True,
            "all_17_candidates_ranked": True,
            "domain_taxonomy_preregistered_by_humans": True,
            "llm_classifies_domains": False,
            "llm_ranks_candidates": False,
            "llm_assigns_numeric_weights": False,
            "llm_tool_mapping": False,
            "posthoc_reference_targets_exposed_to_llm": False,
            "main_scientific_llm_calls_planned": 3,
        }
        write_json(out / "manifest.json", manifest)
        print(f"Prepared only: {out}")
        return

    settings, safe_meta = load_settings(config_path)
    discoveries: list[dict[str, Any]] = []
    for index, packet in enumerate(packets, start=1):
        checkpoint = checkpoints_dir / f"{packet['domain_id']}_validated.json"
        if checkpoint.exists():
            discovery = load_json(checkpoint)
            discovery = validate_discovery(discovery, packet, config)
            print(f"[RESUME] {packet['domain_id']} reused validated checkpoint")
        else:
            discovery = call_and_validate(
                settings=settings,
                packet=packet,
                config=config,
                prompt_text=prompt_map[packet["domain_id"]],
                raw_dir=raw_dir,
            )
            write_json(checkpoint, discovery)
        discoveries.append(discovery)
        primary_n = sum(x["role"] == "primary_rank" for x in discovery["indicators"])
        print(
            f"[{index}/3] {packet['domain_id']} {packet['domain_name_zh']}: "
            f"{len(discovery['indicators'])} indicators; primary_rank={primary_n}"
        )

    write_json(out / "discovered_indicators.json", {"domains": discoveries})
    flat = flatten_indicators(discoveries)
    flat.to_csv(out / "discovered_indicators.csv", index=False)
    build_indicator_evidence(discoveries, qdf).to_csv(out / "indicator_evidence.csv", index=False)
    warning_df = flatten_validation_warnings(discoveries)
    warning_df.to_csv(out / "indicator_validation_warnings.csv", index=False)
    write_json(
        out / "indicator_validation_warnings.json",
        {"warnings": warning_df.to_dict(orient="records")},
    )

    summary_lines = [
        "# D-Lys 第二嵌段三域指标发现 V3",
        "",
        "> 第一阶段已固定D-赖氨酸。本阶段人工预注册三个可比机制域，17种X全部进入某一域；LLM只根据D-Lys体系约束与合格文献证据发现域内应比较的物化指标，不分类、不排序候选、不赋权。",
        "",
    ]
    for discovery in discoveries:
        summary_lines.append(f"## {discovery['domain_id']} {discovery['domain_name_zh']}")
        summary_lines.append("")
        for indicator in discovery["indicators"]:
            summary_lines.append(
                f"- **{indicator['indicator_name']}** — role={indicator['role']}, "
                f"direction={indicator['preferred_direction']}, basis={indicator['basis']}."
            )
        warning_count = len(discovery.get("validation_warnings", []))
        if warning_count:
            summary_lines.append(
                f"- Validation warnings: {warning_count}（不合格字段已按保守规则降级，不会自动进入主排名）。"
            )
        if discovery["knowledge_gaps"]:
            summary_lines.append(f"- Knowledge gaps: {'; '.join(discovery['knowledge_gaps'])}")
        summary_lines.append("")
    (out / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    manifest = {
        "schema_version": config["schema_version"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification_run": str(qualification_run),
        "qualification_sha256": sha256_file(qualification_csv),
        "config_sha256": sha256_file(config_path),
        "taxonomy_contract": str(taxonomy_contract_path) if taxonomy_contract_path else None,
        "taxonomy_contract_sha256": sha256_file(taxonomy_contract_path) if taxonomy_contract_path else None,
        "prompt_template_sha256": sha256_file(prompt_path),
        "llm_settings": safe_meta,
        "stage1_dlys_already_selected": True,
        "all_17_candidates_ranked": True,
        "domain_taxonomy_preregistered_by_humans": True,
        "llm_classifies_domains": False,
        "llm_ranks_candidates": False,
        "llm_assigns_numeric_weights": False,
        "llm_tool_mapping": False,
        "posthoc_reference_targets_exposed_to_llm": False,
        "main_scientific_llm_calls": 3,
        "second_llm_format_repair_calls": 0,
        "local_json_syntax_repair_may_be_used": True,
        "short_peptide_ap_used_for_final_selection": False,
        "cpp_used_for_final_selection": False,
        "camsol_used_for_final_selection": False,
        "molecular_dynamics_used_for_final_selection": False,
        "outputs": {
            "taxonomy": "domain_taxonomy.json",
            "indicators_json": "discovered_indicators.json",
            "indicators_csv": "discovered_indicators.csv",
            "indicator_evidence": "indicator_evidence.csv",
            "validation_warnings_csv": "indicator_validation_warnings.csv",
            "validation_warnings_json": "indicator_validation_warnings.json",
            "summary": "summary.md",
        },
    }
    write_json(out / "manifest.json", manifest)
    print("Completed:", out)


if __name__ == "__main__":
    main()
