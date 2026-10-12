from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dlys_multicriteria_ranking_v2.json"
DEFAULT_REGISTRY = ROOT / "configs" / "dlys_metric_registry_v2.json"
DEFAULT_REFERENCE = ROOT / "configs" / "dlys_posthoc_reference_targets_v1.json"
DISCOVERY_SUFFIX = "__dlys-domain-indicator-discovery-v2"
INTRINSIC_SUFFIX = "__intrinsic-material-properties-v1"
RUN_SUFFIX = "__dlys-multicriteria-ranking-v2"


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


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


def latest_run(runs_dir: Path, suffix: str, required: str) -> Path:
    matches = sorted(
        p for p in runs_dir.glob(f"*{suffix}") if p.is_dir() and (p / required).exists()
    )
    if not matches:
        raise FileNotFoundError(f"No {suffix} run containing {required} found")
    return matches[-1]


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def require_versions(config: dict[str, Any]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package, expected in config.get("dependencies", {}).items():
        actual = package_version(package)
        if actual is None:
            if package == "pymcdm":
                raise RuntimeError(
                    "Missing mature TOPSIS dependency pymcdm. Install with: "
                    "python -m pip install pymcdm==1.4.0"
                )
            if package == "aaindex":
                raise RuntimeError(
                    "Missing mature AAindex dependency. Install with: "
                    "python -m pip install aaindex==1.3.2"
                )
            raise RuntimeError(f"Missing required package: {package}")
        if str(expected).strip() and actual != str(expected):
            raise RuntimeError(f"{package} version mismatch: expected {expected}, got {actual}")
        versions[package] = actual
    for package in ("scipy", "pandas", "numpy", "rdkit", "peptides"):
        versions[package] = package_version(package)
    return versions


def load_inputs(repo_root: Path, args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    runs = repo_root / "runs"
    if args.indicator_run:
        if args.indicator_run == "latest":
            discovery_run = latest_run(runs, DISCOVERY_SUFFIX, "discovered_indicators.csv")
        else:
            discovery_run = Path(args.indicator_run)
            if not discovery_run.is_absolute():
                discovery_run = repo_root / discovery_run
            discovery_run = discovery_run.resolve()
    else:
        discovery_run = latest_run(runs, DISCOVERY_SUFFIX, "discovered_indicators.csv")

    if args.intrinsic_csv:
        intrinsic_csv = Path(args.intrinsic_csv)
        if not intrinsic_csv.is_absolute():
            intrinsic_csv = repo_root / intrinsic_csv
        intrinsic_csv = intrinsic_csv.resolve()
    else:
        intrinsic_run = latest_run(
            runs, INTRINSIC_SUFFIX, "intrinsic_properties_candidate17.csv"
        )
        intrinsic_csv = intrinsic_run / "intrinsic_properties_candidate17.csv"

    indicators_csv = discovery_run / "discovered_indicators.csv"
    taxonomy_json = discovery_run / "domain_taxonomy.json"
    for path in (indicators_csv, taxonomy_json, intrinsic_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    return discovery_run, indicators_csv, taxonomy_json, intrinsic_csv




def validate_taxonomy_against_contract(
    taxonomy: dict[str, Any], config: dict[str, Any], repo_root: Path
) -> dict[str, Any] | None:
    rel = clean(config.get("taxonomy_contract"))
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"taxonomy contract not found: {path}")
    contract = load_json(path)
    frozen = {d["domain_id"]: list(d["candidates"]) for d in contract["domains"]}
    current = {d["domain_id"]: list(d["candidates"]) for d in taxonomy["domains"]}
    if current != frozen:
        raise ValueError(
            "Indicator run taxonomy does not match the frozen stage-2 taxonomy contract. "
            "Do not rank a stale discovery run; rerun dlys_domain_indicator_discovery_v2. "
            f"current={current}; frozen={frozen}"
        )
    frozen_names = contract.get("candidate_names_zh", {})
    if frozen_names and taxonomy.get("candidate_names_zh", {}) != frozen_names:
        raise ValueError("Indicator run candidate_names_zh do not match the frozen taxonomy contract")
    return contract

def normalize_text(text: str) -> str:
    text = clean(text).lower()
    text = text.replace("α", "alpha").replace("β", "beta")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compile_registry(registry: dict[str, Any]) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for metric in registry["metrics"]:
        aliases = sorted(
            {normalize_text(x) for x in metric.get("aliases", []) if normalize_text(x)},
            key=len,
            reverse=True,
        )
        row = dict(metric)
        row["_aliases_normalized"] = aliases
        compiled.append(row)
    return compiled


def map_indicator_to_registry(indicator: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    if not bool(indicator.get("quantifiable_single_residue", False)):
        return {"status": "unresolved", "reason": "indicator_marked_not_quantifiable_single_residue"}

    compiled = compile_registry(registry)
    fields = [
        normalize_text(indicator.get("measurement_target", "")),
        normalize_text(indicator.get("indicator_name", "")),
    ]
    fields = [x for x in fields if x]

    for rule in registry.get("mapping_policy", {}).get(
        "do_not_substitute_specific_measurements", []
    ):
        terms = [normalize_text(x) for x in rule.get("terms", []) if normalize_text(x)]
        if any(term and term in field for field in fields for term in terms):
            return {
                "status": "unresolved",
                "reason": rule.get("reason", "specific_measurement_not_registered"),
            }

    exact_matches: list[dict[str, Any]] = []
    substring_matches: list[tuple[int, dict[str, Any], str]] = []

    for metric in compiled:
        aliases = metric["_aliases_normalized"]
        if any(field in aliases for field in fields):
            exact_matches.append(metric)
            continue
        best_alias = ""
        for field in fields:
            for alias in aliases:
                if len(alias) >= 5 and re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", field):
                    if len(alias) > len(best_alias):
                        best_alias = alias
        if best_alias:
            substring_matches.append((len(best_alias), metric, best_alias))

    unique_exact = {m["registry_key"]: m for m in exact_matches}
    if len(unique_exact) == 1:
        metric = next(iter(unique_exact.values()))
        return {"status": "resolved", "match_type": "exact_alias", **_public_metric(metric)}
    if len(unique_exact) > 1:
        return {
            "status": "unresolved",
            "reason": "ambiguous_exact_alias_matches",
            "matches": sorted(unique_exact),
        }

    if substring_matches:
        matched = {x[1]["registry_key"]: x[1] for x in substring_matches}
        if len(matched) == 1:
            metric = next(iter(matched.values()))
            return {"status": "resolved", "match_type": "safe_alias_substring", **_public_metric(metric)}
        return {
            "status": "unresolved",
            "reason": "ambiguous_substring_alias_matches",
            "matches": sorted(matched),
        }

    return {"status": "unresolved", "reason": "no_safe_registry_alias_match"}


def _public_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "registry_key": metric["registry_key"],
        "family": metric["family"],
        "tool": metric["tool"],
        "tool_id": metric["tool_id"],
        "tool_label": metric["label"],
    }


def _bool_num(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"1", "true", "yes", "y", "t"} else 0.0
    return 1.0 if bool(value) else 0.0


def _load_aaindex_values(accession: str, candidates: list[str]) -> dict[str, float]:
    from aaindex import aaindex1

    values = aaindex1.values(accession)
    return {aa: float(values[aa]) for aa in candidates}


def resolve_metric_values(
    mapping: dict[str, Any], candidates: list[str], intrinsic: pd.DataFrame
) -> dict[str, float]:
    if mapping.get("status") != "resolved":
        raise ValueError("cannot resolve values for unresolved mapping")
    idx = intrinsic.set_index("aa", drop=False)
    missing = [aa for aa in candidates if aa not in idx.index]
    if missing:
        raise ValueError(f"Intrinsic table missing candidates: {missing}")

    tool = mapping["tool"]
    tool_id = mapping["tool_id"]
    if tool == "AAindex":
        return _load_aaindex_values(tool_id, candidates)

    if tool == "peptides.py":
        if tool_id == "nominal_sidechain_charge_pH7_4":
            return {
                aa: float(idx.loc[aa, "nominal_sidechain_charge_pH7_4"])
                for aa in candidates
            }
        if tool_id == "absolute_nominal_sidechain_charge_pH7_4":
            return {
                aa: abs(float(idx.loc[aa, "nominal_sidechain_charge_pH7_4"]))
                for aa in candidates
            }

    if tool == "RDKit":
        if tool_id == "sidechain_HBD_plus_HBA":
            return {
                aa: float(idx.loc[aa, "sidechain_HBD_reference"])
                + float(idx.loc[aa, "sidechain_HBA_reference"])
                for aa in candidates
            }
        if tool_id not in idx.columns:
            raise ValueError(f"Intrinsic RDKit-derived column missing: {tool_id}")
        series = idx[tool_id]
        return {
            aa: _bool_num(series.loc[aa]) if series.dtype == bool or isinstance(series.loc[aa], (bool, np.bool_))
            else float(series.loc[aa])
            for aa in candidates
        }

    raise ValueError(f"No deterministic resolver for tool={tool!r}, tool_id={tool_id!r}")


def transform_for_ranking(
    direction: str,
    raw_values: np.ndarray,
    target_value: float | None,
) -> tuple[np.ndarray, int, str] | None:
    raw = np.asarray(raw_values, dtype=float)
    if np.isnan(raw).any():
        return None
    if direction == "higher":
        return raw, 1, "raw_benefit"
    if direction == "lower":
        return raw, -1, "raw_cost"
    if direction == "closer_to_zero":
        return np.abs(raw), -1, "absolute_deviation_from_zero_cost"
    if direction == "presence":
        return (raw > 0).astype(float), 1, "binary_presence_benefit"
    if direction == "absence":
        return (raw > 0).astype(float), -1, "binary_presence_cost"
    if direction == "target":
        if target_value is None or np.isnan(float(target_value)):
            return None
        return np.abs(raw - float(target_value)), -1, "absolute_deviation_from_target_cost"
    if direction == "context_dependent":
        return None
    raise ValueError(f"Unknown direction: {direction}")


def _criterion_rank_signature(criterion: dict[str, Any]) -> tuple[float, ...]:
    values = np.asarray(criterion["ranking_values"], dtype=float)
    if int(criterion["criterion_type"]) == 1:
        ranks = rankdata(-values, method="average")
    else:
        ranks = rankdata(values, method="average")
    return tuple(float(x) for x in ranks)


def _apply_domain_rank_redundancy_filter(
    domain_id: str,
    criteria: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    domains = set(config.get("ranking_policy", {}).get("drop_exact_rank_redundant_criteria_domains", []))
    if domain_id not in domains or len(criteria) < 2:
        return criteria, audit_rows

    priority = config.get("ranking_policy", {}).get("metric_priority_by_domain", {}).get(domain_id, [])
    order = {key: i for i, key in enumerate(priority)}
    indexed = list(enumerate(criteria))
    indexed.sort(key=lambda item: (order.get(item[1].get("registry_key", ""), len(order) + item[0]), item[0]))

    kept: list[dict[str, Any]] = []
    signatures: dict[tuple[float, ...], dict[str, Any]] = {}
    dropped_ids: dict[str, str] = {}
    for _, criterion in indexed:
        sig = _criterion_rank_signature(criterion)
        if sig in signatures:
            kept_criterion = signatures[sig]
            dropped_ids[criterion["indicator_id"]] = kept_criterion["registry_key"]
            continue
        signatures[sig] = criterion
        kept.append(criterion)

    if dropped_ids:
        for row in audit_rows:
            iid = row.get("indicator_id", "")
            if iid in dropped_ids:
                row["used_in_topsis"] = False
                row["exclusion_reason"] = (
                    "exact_rank_redundant_with:" + dropped_ids[iid]
                    + ";retained_as_corroborative_not_independent_vote"
                )
    return kept, audit_rows


def prepare_domain_problem(
    domain: dict[str, Any],
    indicator_df: pd.DataFrame,
    intrinsic: pd.DataFrame,
    registry: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    did = domain["domain_id"]
    candidates = list(domain["candidates"])
    ddf = indicator_df[indicator_df["domain_id"].astype(str) == did].copy()
    if ddf.empty:
        raise ValueError(f"No discovered indicators for {did}")

    used_registry_keys: set[str] = set()
    criteria: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for _, row in ddf.iterrows():
        indicator = row.to_dict()
        role = clean(indicator.get("role"))
        mapping = map_indicator_to_registry(indicator, registry)
        base_audit = {
            "domain_id": did,
            "domain_name_zh": domain["name_zh"],
            "indicator_id": clean(indicator.get("indicator_id")),
            "indicator_name": clean(indicator.get("indicator_name")),
            "measurement_target": clean(indicator.get("measurement_target")),
            "role": role,
            "preferred_direction": clean(indicator.get("preferred_direction")),
            "mapping_status": mapping.get("status"),
            "mapping_reason": mapping.get("reason", ""),
            "registry_key": mapping.get("registry_key", ""),
            "tool": mapping.get("tool", ""),
            "tool_id": mapping.get("tool_id", ""),
            "used_in_topsis": False,
            "exclusion_reason": "",
        }

        if role != "primary_rank":
            base_audit["exclusion_reason"] = f"role_{role}_not_primary_rank"
            audit_rows.append(base_audit)
            continue
        if mapping.get("status") != "resolved":
            base_audit["exclusion_reason"] = mapping.get("reason", "unresolved")
            audit_rows.append(base_audit)
            continue

        if config["ranking_policy"].get("enforce_domain_metric_direction_policy", False):
            domain_policy = config.get("domain_metric_direction_policy", {}).get(did, {})
            if mapping["registry_key"] in domain_policy:
                allowed_directions = list(domain_policy[mapping["registry_key"]])
                proposed_direction = clean(indicator.get("preferred_direction"))
                if not allowed_directions:
                    base_audit["exclusion_reason"] = "metric_validation_only_by_preregistered_domain_policy"
                    audit_rows.append(base_audit)
                    continue
                if proposed_direction not in allowed_directions:
                    base_audit["exclusion_reason"] = (
                        "direction_conflicts_with_preregistered_domain_policy:"
                        f"proposed={proposed_direction};allowed={','.join(allowed_directions)}"
                    )
                    audit_rows.append(base_audit)
                    continue

        if mapping["registry_key"] in used_registry_keys:
            base_audit["exclusion_reason"] = "duplicate_registry_key"
            audit_rows.append(base_audit)
            continue

        try:
            values_map = resolve_metric_values(mapping, candidates, intrinsic)
        except Exception as exc:
            base_audit["exclusion_reason"] = f"numeric_resolution_failed:{type(exc).__name__}:{exc}"
            audit_rows.append(base_audit)
            continue

        raw_values = np.asarray([values_map[aa] for aa in candidates], dtype=float)
        target_value = indicator.get("target_value")
        if target_value is None:
            target_value = None
        elif isinstance(target_value, str) and clean(target_value) == "":
            target_value = None
        elif not isinstance(target_value, str) and pd.isna(target_value):
            target_value = None
        transformed = transform_for_ranking(
            clean(indicator.get("preferred_direction")), raw_values, target_value
        )
        if transformed is None:
            base_audit["exclusion_reason"] = "direction_not_operationalizable_for_ranking"
            audit_rows.append(base_audit)
            continue
        ranking_values, criterion_type, transform_label = transformed
        if config["ranking_policy"].get("drop_constant_criteria", True) and np.ptp(ranking_values) <= 1e-12:
            base_audit["exclusion_reason"] = "constant_within_domain"
            audit_rows.append(base_audit)
            continue

        used_registry_keys.add(mapping["registry_key"])
        base_audit["used_in_topsis"] = True
        audit_rows.append(base_audit)
        criteria.append(
            {
                "indicator_id": clean(indicator.get("indicator_id")),
                "indicator_name": clean(indicator.get("indicator_name")),
                "measurement_target": clean(indicator.get("measurement_target")),
                "preferred_direction": clean(indicator.get("preferred_direction")),
                "target_value": target_value,
                "why_relevant_to_dlys_system": clean(indicator.get("why_relevant_to_dlys_system")),
                **mapping,
                "raw_values": raw_values,
                "ranking_values": ranking_values,
                "criterion_type": int(criterion_type),
                "transform": transform_label,
            }
        )

    criteria, audit_rows = _apply_domain_rank_redundancy_filter(
        did, criteria, audit_rows, config
    )

    return {
        "domain_id": did,
        "domain_name_zh": domain["name_zh"],
        "candidates": candidates,
        "criteria": criteria,
        "criteria_audit": audit_rows,
    }


def run_topsis(
    matrix: np.ndarray,
    types: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from pymcdm.methods import TOPSIS

    n_criteria = int(matrix.shape[1])
    if weights is None:
        weights = np.full(n_criteria, 1.0 / n_criteria, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_criteria,):
            raise ValueError(f"weights must have shape ({n_criteria},), got {weights.shape}")
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("TOPSIS weights must be finite and strictly positive")
        weights = weights / weights.sum()
    # PyMCDM warns when Pareto-dominant/dominated alternatives exist. We report
    # these relationships explicitly in pareto_dominance.csv, so the library
    # warning is optionally silenced to keep the CLI readable without hiding it.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Alternatives with indices .* are (?:dominant|dominated).*",
            category=UserWarning,
            module=r"pymcdm\.validators",
        )
        preferences = np.asarray(TOPSIS()(matrix, weights, types), dtype=float)
    # TOPSIS preference is higher-better; use SciPy for transparent rank conversion.
    ranks = np.asarray(rankdata(-preferences, method="min"), dtype=float)
    return preferences, ranks


def borda_rank_sum(matrix: np.ndarray, types: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_alt, n_criteria = matrix.shape
    criterion_ranks = np.zeros((n_alt, n_criteria), dtype=float)
    for j in range(n_criteria):
        values = matrix[:, j]
        criterion_ranks[:, j] = rankdata(-values if types[j] == 1 else values, method="average")
    rank_sum = criterion_ranks.sum(axis=1)
    borda_ranks = rankdata(rank_sum, method="min")
    return rank_sum, borda_ranks, criterion_ranks


def _winner_for_sensitivity(matrix: np.ndarray, types: np.ndarray, candidates: list[str]) -> tuple[str, np.ndarray]:
    if matrix.shape[1] == 1:
        values = matrix[:, 0]
        ranks = rankdata(-values if types[0] == 1 else values, method="min")
        winner = candidates[int(np.argmin(ranks))]
        return winner, ranks
    preferences, ranks = run_topsis(matrix, types)
    return candidates[int(np.argmax(preferences))], ranks


def weight_perturbation_sensitivity(
    matrix: np.ndarray,
    types: np.ndarray,
    candidates: list[str],
    criteria: list[dict[str, Any]],
    baseline_winner: str,
    fraction: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """Deterministic one-at-a-time perturbation around equal TOPSIS weights.

    This is a sensitivity diagnostic, not weight tuning. Each criterion weight is
    multiplied by (1-fraction) and (1+fraction) in turn, then the full vector is
    renormalized before rerunning TOPSIS.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError("weight_perturbation_fraction must be between 0 and 1")
    n = len(criteria)
    base = np.full(n, 1.0 / n, dtype=float)
    rows: list[dict[str, Any]] = []
    stable = 0
    total = 0
    for j, criterion in enumerate(criteria):
        for label, factor in (("down", 1.0 - fraction), ("up", 1.0 + fraction)):
            weights = base.copy()
            weights[j] *= factor
            weights /= weights.sum()
            prefs, ranks = run_topsis(matrix, types, weights=weights)
            winner = candidates[int(np.argmax(prefs))]
            same = winner == baseline_winner
            stable += int(same)
            total += 1
            rows.append(
                {
                    "perturbed_indicator_id": criterion["indicator_id"],
                    "perturbed_indicator_name": criterion["indicator_name"],
                    "perturbation": label,
                    "fraction": float(fraction),
                    "weights": "|".join(f"{x:.8f}" for x in weights),
                    "baseline_winner": baseline_winner,
                    "new_winner": winner,
                    "winner_changed": not same,
                    "baseline_winner_rank_after_perturbation": float(
                        ranks[candidates.index(baseline_winner)]
                    ),
                }
            )
    return rows, stable, total


def pareto_dominance_rows(
    matrix: np.ndarray, types: np.ndarray, candidates: list[str]
) -> list[dict[str, Any]]:
    """Return explicit pairwise Pareto dominance relationships.

    This is diagnostic only; no candidate is removed from TOPSIS.
    """
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i == j:
                continue
            left_vals = matrix[i]
            right_vals = matrix[j]
            ge = []
            gt = []
            for k, ctype in enumerate(types):
                if int(ctype) == 1:
                    ge.append(left_vals[k] >= right_vals[k] - 1e-12)
                    gt.append(left_vals[k] > right_vals[k] + 1e-12)
                else:
                    ge.append(left_vals[k] <= right_vals[k] + 1e-12)
                    gt.append(left_vals[k] < right_vals[k] - 1e-12)
            if all(ge) and any(gt):
                rows.append({"dominant_candidate": left, "dominated_candidate": right})
    return rows


def _margin_to_second(values: np.ndarray, higher_is_better: bool) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size < 2:
        return float("nan")
    return float(arr[-1] - arr[-2]) if higher_is_better else float(arr[1] - arr[0])


def _diagnostic_flags(
    *, winner_agreement: bool, loo_stable: int, loo_total: int,
    weight_stable: int, weight_total: int, dominance_rows: list[dict[str, Any]]
) -> list[str]:
    flags: list[str] = []
    if not winner_agreement:
        flags.append("borda_disagrees_with_primary_topsis")
    if loo_total and loo_stable < loo_total:
        flags.append("criterion_ablation_changes_winner")
    if weight_total and weight_stable < weight_total:
        flags.append("local_weight_perturbation_changes_winner")
    if dominance_rows:
        flags.append("pareto_dominance_present")
    if not flags:
        flags.append("no_instability_flag_in_registered_diagnostics")
    return flags


def _single_criterion_fallback_config(
    problem: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    policy = config.get("ranking_policy", {}).get("single_criterion_fallback", {})
    if not bool(policy.get("enabled", False)):
        return None
    return policy.get("domains", {}).get(problem["domain_id"])


def _rank_single_criterion(
    problem: dict[str, Any], fallback: dict[str, Any]
) -> dict[str, Any]:
    criterion = problem["criteria"][0]
    allowed = set(fallback.get("allowed_registry_keys", []))
    if criterion.get("registry_key") not in allowed:
        return {
            "domain_id": problem["domain_id"],
            "domain_name_zh": problem["domain_name_zh"],
            "status": "insufficient_multicriteria_criteria",
            "reason": (
                "single_criterion_fallback_not_authorized_for_registry_key="
                f"{criterion.get('registry_key','')}"
            ),
            "criteria": problem["criteria"],
            "candidates": problem["candidates"],
        }

    values = np.asarray(criterion["ranking_values"], dtype=float)
    ctype = int(criterion["criterion_type"])
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return {
            "domain_id": problem["domain_id"],
            "domain_name_zh": problem["domain_name_zh"],
            "status": "insufficient_multicriteria_criteria",
            "reason": "single_criterion_fallback_axis_is_constant",
            "criteria": problem["criteria"],
            "candidates": problem["candidates"],
        }
    if ctype == 1:
        suitability = (values - lo) / (hi - lo)
    else:
        suitability = (hi - values) / (hi - lo)
    primary_ranks = rankdata(-suitability, method="min")
    candidates = problem["candidates"]
    winner = candidates[int(np.argmax(suitability))]
    margin = _margin_to_second(suitability, higher_is_better=True)
    ranking_rows = [
        {
            "candidate": aa,
            "primary_score": float(suitability[i]),
            "primary_rank": float(primary_ranks[i]),
        }
        for i, aa in enumerate(candidates)
    ]
    return {
        "domain_id": problem["domain_id"],
        "domain_name_zh": problem["domain_name_zh"],
        "status": "completed_single_criterion",
        "primary_method": fallback.get("method_label", "preregistered_single_axis_rank"),
        "primary_winner": winner,
        "primary_margin_to_second": margin,
        "single_criterion_reason": fallback.get("reason", ""),
        "criteria": problem["criteria"],
        "candidates": candidates,
        "criterion_ranks": primary_ranks.reshape(-1, 1),
        "ranking_rows": ranking_rows,
        "pareto_dominance_rows": [],
        "diagnostic_flags": [
            "single_criterion_fallback_used",
            "no_independent_second_axis_after_qualification_or_redundancy_audit",
        ],
        "loo_rows": [],
        "loo_stability_count": 0,
        "loo_total": 0,
        "weight_perturbation_rows": [],
        "weight_perturbation_stability_count": 0,
        "weight_perturbation_total": 0,
    }


def rank_domain(problem: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    criteria = problem["criteria"]
    candidates = problem["candidates"]
    minimum = int(config["ranking_policy"]["minimum_primary_criteria"])
    if len(criteria) < minimum:
        fallback = _single_criterion_fallback_config(problem, config)
        if len(criteria) == 1 and fallback is not None:
            return _rank_single_criterion(problem, fallback)
        return {
            "domain_id": problem["domain_id"],
            "domain_name_zh": problem["domain_name_zh"],
            "status": "insufficient_multicriteria_criteria",
            "reason": f"usable_primary_rank_criteria={len(criteria)} < {minimum}",
            "criteria": criteria,
            "candidates": candidates,
        }

    matrix = np.column_stack([x["ranking_values"] for x in criteria]).astype(float)
    types = np.asarray([x["criterion_type"] for x in criteria], dtype=int)
    preferences, topsis_ranks = run_topsis(matrix, types)
    rank_sum, borda_ranks, criterion_ranks = borda_rank_sum(matrix, types)

    topsis_winner = candidates[int(np.argmax(preferences))]
    borda_winner = candidates[int(np.argmin(rank_sum))]
    topsis_margin_to_second = _margin_to_second(preferences, higher_is_better=True)
    borda_rank_sum_margin_to_second = _margin_to_second(rank_sum, higher_is_better=False)
    dominance_rows = pareto_dominance_rows(matrix, types, candidates)

    ranking_rows: list[dict[str, Any]] = []
    for i, aa in enumerate(candidates):
        ranking_rows.append(
            {
                "candidate": aa,
                "topsis_preference": float(preferences[i]),
                "topsis_rank": float(topsis_ranks[i]),
                "borda_rank_sum": float(rank_sum[i]),
                "borda_rank": float(borda_ranks[i]),
                "primary_score": float(preferences[i]),
                "primary_rank": float(topsis_ranks[i]),
            }
        )

    loo_rows: list[dict[str, Any]] = []
    stable = 0
    for j, criterion in enumerate(criteria):
        keep = [k for k in range(len(criteria)) if k != j]
        reduced = matrix[:, keep]
        reduced_types = types[keep]
        new_winner, reduced_ranks = _winner_for_sensitivity(reduced, reduced_types, candidates)
        baseline_rank = float(reduced_ranks[candidates.index(topsis_winner)])
        same = new_winner == topsis_winner
        stable += int(same)
        loo_rows.append(
            {
                "domain_id": problem["domain_id"],
                "omitted_indicator_id": criterion["indicator_id"],
                "omitted_indicator_name": criterion["indicator_name"],
                "baseline_winner": topsis_winner,
                "new_winner": new_winner,
                "winner_changed": not same,
                "baseline_winner_rank_after_omission": baseline_rank,
            }
        )

    perturbation_rows, perturbation_stable, perturbation_total = weight_perturbation_sensitivity(
        matrix,
        types,
        candidates,
        criteria,
        topsis_winner,
        float(config["ranking_policy"].get("weight_perturbation_fraction", 0.20)),
    )
    for row in perturbation_rows:
        row["domain_id"] = problem["domain_id"]

    diagnostic_flags = _diagnostic_flags(
        winner_agreement=topsis_winner == borda_winner,
        loo_stable=stable,
        loo_total=len(loo_rows),
        weight_stable=perturbation_stable,
        weight_total=perturbation_total,
        dominance_rows=dominance_rows,
    )

    return {
        "domain_id": problem["domain_id"],
        "domain_name_zh": problem["domain_name_zh"],
        "status": "completed",
        "primary_method": "TOPSIS",
        "primary_winner": topsis_winner,
        "primary_margin_to_second": topsis_margin_to_second,
        "criteria": criteria,
        "candidates": candidates,
        "matrix": matrix,
        "types": types,
        "criterion_ranks": criterion_ranks,
        "ranking_rows": ranking_rows,
        "winner_topsis": topsis_winner,
        "winner_borda": borda_winner,
        "winner_agreement": topsis_winner == borda_winner,
        "topsis_margin_to_second": topsis_margin_to_second,
        "borda_rank_sum_margin_to_second": borda_rank_sum_margin_to_second,
        "pareto_dominance_rows": dominance_rows,
        "diagnostic_flags": diagnostic_flags,
        "loo_rows": loo_rows,
        "loo_stability_count": stable,
        "loo_total": len(loo_rows),
        "weight_perturbation_rows": perturbation_rows,
        "weight_perturbation_stability_count": perturbation_stable,
        "weight_perturbation_total": perturbation_total,
    }


def build_wide_decision_matrix(problems: list[dict[str, Any]], names: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for problem in problems:
        for i, aa in enumerate(problem["candidates"]):
            row: dict[str, Any] = {
                "domain_id": problem["domain_id"],
                "domain_name_zh": problem["domain_name_zh"],
                "candidate": aa,
                "candidate_name_zh": names.get(aa, aa),
            }
            for criterion in problem["criteria"]:
                key = f"{criterion['indicator_id']}__{criterion['registry_key']}"
                row[key] = float(criterion["raw_values"][i])
            rows.append(row)
    return pd.DataFrame(rows)


def load_posthoc_targets(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not payload.get("not_for_selection", False):
        raise ValueError("post-hoc reference target file must declare not_for_selection=true")
    targets = {str(k): str(v) for k, v in payload.get("targets", {}).items()}
    if set(targets) != {"D1", "D2", "D3"}:
        raise ValueError("post-hoc reference targets must contain D1, D2, D3")
    return targets


def write_outputs(
    out: Path,
    problems: list[dict[str, Any]],
    results: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    names: dict[str, str],
    posthoc_targets: dict[str, str],
    config: dict[str, Any],
) -> None:
    criteria_used_rows: list[dict[str, Any]] = []
    criteria_unresolved_rows: list[dict[str, Any]] = []
    matrix_long_rows: list[dict[str, Any]] = []
    topsis_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    borda_rows: list[dict[str, Any]] = []
    criterion_rank_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    weight_perturbation_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    result_by_domain = {r["domain_id"]: r for r in results}

    for problem in problems:
        result = result_by_domain[problem["domain_id"]]
        primary_method = result.get("primary_method", "")

        for audit in problem["criteria_audit"]:
            if audit["used_in_topsis"]:
                criterion = next(
                    c for c in problem["criteria"] if c["indicator_id"] == audit["indicator_id"]
                )
                criteria_used_rows.append(
                    {
                        **audit,
                        "used_in_primary_decision": True,
                        "primary_method": primary_method,
                        "family": criterion["family"],
                        "tool_label": criterion["tool_label"],
                        "transform": criterion["transform"],
                        "criterion_type": criterion["criterion_type"],
                        "equal_weight": 1.0 / len(problem["criteria"]),
                        "why_relevant_to_dlys_system": criterion["why_relevant_to_dlys_system"],
                    }
                )
            else:
                criteria_unresolved_rows.append({**audit, "used_in_primary_decision": False})

        for criterion in problem["criteria"]:
            for aa, raw, operational in zip(
                problem["candidates"], criterion["raw_values"], criterion["ranking_values"]
            ):
                matrix_long_rows.append(
                    {
                        "domain_id": problem["domain_id"],
                        "domain_name_zh": problem["domain_name_zh"],
                        "candidate": aa,
                        "candidate_name_zh": names.get(aa, aa),
                        "indicator_id": criterion["indicator_id"],
                        "indicator_name": criterion["indicator_name"],
                        "registry_key": criterion["registry_key"],
                        "raw_value": float(raw),
                        "operational_value": float(operational),
                        "preferred_direction": criterion["preferred_direction"],
                        "criterion_type": criterion["criterion_type"],
                    }
                )

        if not str(result["status"]).startswith("completed"):
            agreement_rows.append(
                {
                    "domain_id": problem["domain_id"],
                    "domain_name_zh": problem["domain_name_zh"],
                    "status": result["status"],
                    "reason": result.get("reason", ""),
                    "primary_method": "",
                    "primary_winner": "",
                    "topsis_winner": "",
                    "borda_winner": "",
                    "winner_agreement": "",
                    "criterion_ablation_stability": "",
                    "weight_perturbation_stability": "",
                }
            )
            target = posthoc_targets[problem["domain_id"]]
            selected_rows.append(
                {
                    "domain_id": problem["domain_id"],
                    "domain_name_zh": problem["domain_name_zh"],
                    "status": result["status"],
                    "reason": result.get("reason", ""),
                    "primary_method": "",
                    "selected_candidate": "",
                    "selected_candidate_name_zh": "",
                    "topsis_borda_agree": "",
                    "loo_winner_stability_count": "",
                    "loo_total": "",
                    "weight_perturbation_stability_count": "",
                    "weight_perturbation_total": "",
                    "diagnostic_flags": "",
                    "posthoc_reference_candidate": target,
                    "posthoc_reference_name_zh": names.get(target, target),
                    "posthoc_reference_match": "",
                    "posthoc_reference_used_for_selection": False,
                }
            )
            continue

        is_single = result["status"] == "completed_single_criterion"
        primary_winner = result["primary_winner"]
        if is_single:
            for row in result["ranking_rows"]:
                single_rows.append(
                    {
                        "domain_id": problem["domain_id"],
                        "domain_name_zh": problem["domain_name_zh"],
                        "candidate": row["candidate"],
                        "candidate_name_zh": names.get(row["candidate"], row["candidate"]),
                        "primary_score": row["primary_score"],
                        "primary_rank": row["primary_rank"],
                        "criterion_registry_key": result["criteria"][0]["registry_key"],
                        "criterion_name": result["criteria"][0]["indicator_name"],
                    }
                )
        else:
            for row in result["ranking_rows"]:
                topsis_rows.append(
                    {
                        "domain_id": problem["domain_id"],
                        "domain_name_zh": problem["domain_name_zh"],
                        "candidate": row["candidate"],
                        "candidate_name_zh": names.get(row["candidate"], row["candidate"]),
                        "topsis_preference": row["topsis_preference"],
                        "topsis_rank": row["topsis_rank"],
                    }
                )
                borda_rows.append(
                    {
                        "domain_id": problem["domain_id"],
                        "domain_name_zh": problem["domain_name_zh"],
                        "candidate": row["candidate"],
                        "candidate_name_zh": names.get(row["candidate"], row["candidate"]),
                        "borda_rank_sum": row["borda_rank_sum"],
                        "borda_rank": row["borda_rank"],
                    }
                )

        for j, criterion in enumerate(result["criteria"]):
            for i, aa in enumerate(result["candidates"]):
                criterion_rank_rows.append(
                    {
                        "domain_id": problem["domain_id"],
                        "candidate": aa,
                        "candidate_name_zh": names.get(aa, aa),
                        "indicator_id": criterion["indicator_id"],
                        "indicator_name": criterion["indicator_name"],
                        "criterion_rank": float(result["criterion_ranks"][i, j]),
                    }
                )

        if is_single:
            agreement_rows.append(
                {
                    "domain_id": problem["domain_id"],
                    "domain_name_zh": problem["domain_name_zh"],
                    "status": result["status"],
                    "reason": result.get("single_criterion_reason", ""),
                    "primary_method": primary_method,
                    "primary_winner": primary_winner,
                    "primary_winner_name_zh": names.get(primary_winner, primary_winner),
                    "topsis_winner": "",
                    "borda_winner": "",
                    "winner_agreement": "not_applicable_single_criterion",
                    "criterion_ablation_stability": "not_applicable_single_criterion",
                    "weight_perturbation_stability": "not_applicable_single_criterion",
                    "primary_margin_to_second": result["primary_margin_to_second"],
                    "diagnostic_flags": "|".join(result["diagnostic_flags"]),
                }
            )
            diagnostic_rows.append(
                {
                    "domain_id": problem["domain_id"],
                    "domain_name_zh": problem["domain_name_zh"],
                    "primary_method": primary_method,
                    "primary_winner": primary_winner,
                    "primary_winner_name_zh": names.get(primary_winner, primary_winner),
                    "borda_diagnostic_winner": "",
                    "borda_agrees": "not_applicable_single_criterion",
                    "primary_margin_to_second": result["primary_margin_to_second"],
                    "criterion_ablation_stability": "not_applicable_single_criterion",
                    "local_weight_perturbation_stability": "not_applicable_single_criterion",
                    "diagnostic_flags": "|".join(result["diagnostic_flags"]),
                    "method_reason": result.get("single_criterion_reason", ""),
                }
            )
        else:
            agreement_rows.append(
                {
                    "domain_id": problem["domain_id"],
                    "domain_name_zh": problem["domain_name_zh"],
                    "status": "completed",
                    "reason": "",
                    "primary_method": "TOPSIS",
                    "primary_winner": primary_winner,
                    "primary_winner_name_zh": names.get(primary_winner, primary_winner),
                    "topsis_winner": result["winner_topsis"],
                    "topsis_winner_name_zh": names.get(result["winner_topsis"], result["winner_topsis"]),
                    "borda_winner": result["winner_borda"],
                    "borda_winner_name_zh": names.get(result["winner_borda"], result["winner_borda"]),
                    "winner_agreement": result["winner_agreement"],
                    "criterion_ablation_stability": f"{result['loo_stability_count']}/{result['loo_total']}",
                    "weight_perturbation_stability": f"{result['weight_perturbation_stability_count']}/{result['weight_perturbation_total']}",
                    "topsis_margin_to_second": result["topsis_margin_to_second"],
                    "borda_rank_sum_margin_to_second": result["borda_rank_sum_margin_to_second"],
                    "diagnostic_flags": "|".join(result["diagnostic_flags"]),
                }
            )
            diagnostic_rows.append(
                {
                    "domain_id": problem["domain_id"],
                    "domain_name_zh": problem["domain_name_zh"],
                    "primary_method": "TOPSIS",
                    "primary_winner": primary_winner,
                    "primary_winner_name_zh": names.get(primary_winner, primary_winner),
                    "borda_diagnostic_winner": result["winner_borda"],
                    "borda_agrees": result["winner_agreement"],
                    "topsis_margin_to_second": result["topsis_margin_to_second"],
                    "borda_rank_sum_margin_to_second": result["borda_rank_sum_margin_to_second"],
                    "criterion_ablation_stability": f"{result['loo_stability_count']}/{result['loo_total']}",
                    "local_weight_perturbation_stability": f"{result['weight_perturbation_stability_count']}/{result['weight_perturbation_total']}",
                    "diagnostic_flags": "|".join(result["diagnostic_flags"]),
                }
            )
            for drow in result["pareto_dominance_rows"]:
                pareto_rows.append(
                    {
                        "domain_id": problem["domain_id"],
                        "domain_name_zh": problem["domain_name_zh"],
                        **drow,
                        "dominant_candidate_name_zh": names.get(drow["dominant_candidate"], drow["dominant_candidate"]),
                        "dominated_candidate_name_zh": names.get(drow["dominated_candidate"], drow["dominated_candidate"]),
                    }
                )
            loo_rows.extend(result["loo_rows"])
            weight_perturbation_rows.extend(result["weight_perturbation_rows"])

        target = posthoc_targets[problem["domain_id"]]
        selected_rows.append(
            {
                "domain_id": problem["domain_id"],
                "domain_name_zh": problem["domain_name_zh"],
                "status": "completed",
                "reason": result.get("single_criterion_reason", "") if is_single else "",
                "primary_method": primary_method,
                "selected_candidate": primary_winner,
                "selected_candidate_name_zh": names.get(primary_winner, primary_winner),
                "topsis_borda_agree": "not_applicable_single_criterion" if is_single else result["winner_agreement"],
                "loo_winner_stability_count": "" if is_single else result["loo_stability_count"],
                "loo_total": "" if is_single else result["loo_total"],
                "weight_perturbation_stability_count": "" if is_single else result["weight_perturbation_stability_count"],
                "weight_perturbation_total": "" if is_single else result["weight_perturbation_total"],
                "primary_margin_to_second": result["primary_margin_to_second"],
                "diagnostic_flags": "|".join(result["diagnostic_flags"]),
                "posthoc_reference_candidate": target,
                "posthoc_reference_name_zh": names.get(target, target),
                "posthoc_reference_match": primary_winner == target,
                "posthoc_reference_used_for_selection": False,
            }
        )

    pd.DataFrame(criteria_used_rows).to_csv(out / "criteria_used.csv", index=False)
    pd.DataFrame(criteria_unresolved_rows).to_csv(out / "criteria_unresolved.csv", index=False)
    pd.DataFrame(matrix_long_rows).to_csv(out / "decision_matrix_long.csv", index=False)
    build_wide_decision_matrix(problems, names).to_csv(out / "decision_matrix_wide.csv", index=False)
    pd.DataFrame(topsis_rows).to_csv(out / "topsis_rankings.csv", index=False)
    pd.DataFrame(single_rows).to_csv(out / "single_criterion_rankings.csv", index=False)
    pd.DataFrame(borda_rows).to_csv(out / "borda_rankings.csv", index=False)
    pd.DataFrame(criterion_rank_rows).to_csv(out / "criterion_individual_ranks.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(out / "ranking_agreement.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(out / "leave_one_criterion_out.csv", index=False)
    pd.DataFrame(weight_perturbation_rows).to_csv(out / "weight_perturbation_sensitivity.csv", index=False)
    pd.DataFrame(pareto_rows).to_csv(out / "pareto_dominance.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(out / "decision_diagnostics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "selected_representatives.csv", index=False)

    summary = [
        "# D-Lys 第二嵌段自适应决策排序 V3.1",
        "",
        "> 第一阶段已固定D-赖氨酸。域内若保留≥2个合格且非冗余指标，使用等权TOPSIS；若预注册允许且资格化/冗余审计后仅剩一个独立决策轴，则使用单轴确定性排序，而不是人为制造第二指标。已知项目/实验参考代表仅在全部排名完成后做post-hoc对照。",
        "",
        "## 结果",
        "",
    ]
    for row in selected_rows:
        if row["status"] != "completed":
            summary.append(f"- **{row['domain_name_zh']}**：{row['status']}（{row['reason']}）；未生成代表候选。")
            continue
        if row["primary_method"] == "TOPSIS":
            summary.append(
                f"- **{row['domain_name_zh']} → {row['selected_candidate_name_zh']}**；方法=TOPSIS；"
                f"TOPSIS/Borda一致={row['topsis_borda_agree']}；"
                f"criterion-ablation={row['loo_winner_stability_count']}/{row['loo_total']}；"
                f"权重±20%={row['weight_perturbation_stability_count']}/{row['weight_perturbation_total']}；"
                f"诊断={row.get('diagnostic_flags','')}；post-hoc参考一致={row['posthoc_reference_match']}。"
            )
        else:
            summary.append(
                f"- **{row['domain_name_zh']} → {row['selected_candidate_name_zh']}**；方法={row['primary_method']}；"
                f"仅保留一个独立合格决策轴，因此不伪造TOPSIS/Borda/权重扰动；"
                f"诊断={row.get('diagnostic_flags','')}；post-hoc参考一致={row['posthoc_reference_match']}。"
            )
    summary += ["", "## 方法边界", ""] + [f"- {x}" for x in config["claims_boundary"]]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="D-Lys stage-2 multicriteria ranking V3")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY.relative_to(ROOT)))
    parser.add_argument("--indicator-run", default="latest")
    parser.add_argument("--intrinsic-csv")
    parser.add_argument("--posthoc-reference-config", default=str(DEFAULT_REFERENCE.relative_to(ROOT)))
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    reference_path = Path(args.posthoc_reference_config)
    if not reference_path.is_absolute():
        reference_path = repo_root / reference_path

    config = load_json(config_path)
    registry = load_json(registry_path)
    versions = require_versions(config)
    discovery_run, indicators_csv, taxonomy_json, intrinsic_csv = load_inputs(repo_root, args)

    indicators = pd.read_csv(indicators_csv)
    taxonomy = load_json(taxonomy_json)
    taxonomy_contract = validate_taxonomy_against_contract(taxonomy, config, repo_root)
    taxonomy_contract_path = Path(config.get("taxonomy_contract", ""))
    if taxonomy_contract_path and not taxonomy_contract_path.is_absolute():
        taxonomy_contract_path = repo_root / taxonomy_contract_path
    taxonomy_contract_path = taxonomy_contract_path.resolve() if str(taxonomy_contract_path) else None
    intrinsic = pd.read_csv(intrinsic_csv)
    names = taxonomy["candidate_names_zh"]

    problems = [
        prepare_domain_problem(domain, indicators, intrinsic, registry, config)
        for domain in taxonomy["domains"]
    ]
    results = [rank_domain(problem, config) for problem in problems]

    # IMPORTANT: reference targets are loaded only after all rankings are computed.
    # They are used solely for post-hoc comparison and cannot affect selection.
    posthoc_targets = load_posthoc_targets(reference_path)

    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = repo_root / out
        out = out.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = repo_root / "runs" / f"{stamp}{RUN_SUFFIX}"
    out.mkdir(parents=True, exist_ok=False)

    write_outputs(out, problems, results, taxonomy, names, posthoc_targets, config)

    manifest = {
        "schema_version": config["schema_version"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage1_dlys_already_selected": True,
        "all_17_candidates_ranked_within_preregistered_domains": True,
        "indicator_run": str(discovery_run),
        "indicator_sha256": sha256_file(indicators_csv),
        "taxonomy_sha256": sha256_file(taxonomy_json),
        "intrinsic_csv": str(intrinsic_csv),
        "intrinsic_sha256": sha256_file(intrinsic_csv),
        "metric_registry_sha256": sha256_file(registry_path),
        "ranking_config_sha256": sha256_file(config_path),
        "taxonomy_contract": str(taxonomy_contract_path) if taxonomy_contract_path else None,
        "taxonomy_contract_sha256": sha256_file(taxonomy_contract_path) if taxonomy_contract_path else None,
        "versions": versions,
        "mature_tools": {
            "literature_retrieval_upstream": ["Europe PMC", "OpenAlex"],
            "semantic_reranking_upstream": ["ncbi/MedCPT-Cross-Encoder"],
            "physicochemical_indices": ["AAindex"],
            "structure_descriptors": ["RDKit"],
            "charge_descriptors": ["peptides.py"],
            "multicriteria_ranking": ["PyMCDM TOPSIS"],
            "rank_aggregation": ["SciPy rankdata"],
        },
        "llm_tool_mapping": False,
        "llm_candidate_ranking": False,
        "llm_numeric_weights": False,
        "equal_topsis_weights": True,
        "hard_gates_exclude_candidates": False,
        "posthoc_reference_config": str(reference_path),
        "posthoc_reference_targets_loaded_after_ranking": True,
        "posthoc_reference_targets_used_for_selection": False,
        "short_peptide_ap_used_for_final_selection": False,
        "cpp_used_for_final_selection": False,
        "camsol_used_for_final_selection": False,
        "molecular_dynamics_used_for_final_selection": False,
        "domains": [
            {
                "domain_id": result["domain_id"],
                "status": result["status"],
                "winner_topsis": result.get("winner_topsis"),
                "winner_borda": result.get("winner_borda"),
                "winner_agreement": result.get("winner_agreement"),
                "topsis_margin_to_second": result.get("topsis_margin_to_second"),
                "borda_rank_sum_margin_to_second": result.get("borda_rank_sum_margin_to_second"),
                "diagnostic_flags": result.get("diagnostic_flags"),
                "loo_stability_count": result.get("loo_stability_count"),
                "loo_total": result.get("loo_total"),
                "weight_perturbation_stability_count": result.get(
                    "weight_perturbation_stability_count"
                ),
                "weight_perturbation_total": result.get("weight_perturbation_total"),
                "posthoc_reference_target": posthoc_targets[result["domain_id"]],
                "posthoc_reference_match": (
                    result.get("winner_topsis") == posthoc_targets[result["domain_id"]]
                    if result["status"] == "completed"
                    else None
                ),
            }
            for result in results
        ],
        "claims_boundary": config["claims_boundary"],
    }
    write_json(out / "manifest.json", manifest)

    print("=" * 84)
    print("D-Lys stage-2 multicriteria ranking V3")
    print("=" * 84)
    print(f"indicator run : {discovery_run}")
    print(f"intrinsic     : {intrinsic_csv}")
    print(f"run           : {out}")
    for result in results:
        if result["status"] == "completed":
            target = posthoc_targets[result["domain_id"]]
            print(
                f"{result['domain_id']} {result['domain_name_zh']}: "
                f"TOPSIS={names.get(result['winner_topsis'], result['winner_topsis'])}; "
                f"Borda={names.get(result['winner_borda'], result['winner_borda'])}; "
                f"agree={result['winner_agreement']}; "
                f"criterion_ablation={result['loo_stability_count']}/{result['loo_total']}; "
                f"weight_perturbation={result['weight_perturbation_stability_count']}/"
                f"{result['weight_perturbation_total']}; "
                f"posthoc_reference_match={result['winner_topsis'] == target}"
            )
        else:
            print(f"{result['domain_id']} {result['domain_name_zh']}: {result['status']} ({result.get('reason','')})")


if __name__ == "__main__":
    main()
