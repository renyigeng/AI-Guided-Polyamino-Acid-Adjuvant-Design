from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from src import dlys_multicriteria_ranking_v2 as mod


ROOT = Path(__file__).resolve().parents[1]


def load_registry():
    return json.loads((ROOT / "configs" / "dlys_metric_registry_v2.json").read_text(encoding="utf-8"))


def load_config():
    return json.loads((ROOT / "configs" / "dlys_multicriteria_ranking_v2.json").read_text(encoding="utf-8"))


def indicator(name: str, target: str, role="primary_rank", direction="higher"):
    return {
        "indicator_id": name,
        "indicator_name": name,
        "measurement_target": target,
        "role": role,
        "preferred_direction": direction,
        "target_value": np.nan,
        "quantifiable_single_residue": True,
        "why_relevant_to_dlys_system": "test",
    }


def test_registry_maps_standard_concepts_without_llm():
    reg = load_registry()
    mapped = mod.map_indicator_to_registry(indicator("I1", "hydrophobicity"), reg)
    assert mapped["status"] == "resolved"
    assert mapped["registry_key"] == "hydropathy"
    assert mapped["tool"] == "AAindex"


def test_registry_rejects_ad_hoc_custom_score():
    mapped = mod.map_indicator_to_registry(indicator("I1", "pi stacking score"), load_registry())
    assert mapped["status"] == "unresolved"
    assert mapped["reason"] == "no_safe_registry_alias_match"


def test_registry_rejects_composite_multi_property_indicator():
    mapped = mod.map_indicator_to_registry(
        indicator("I1", "hydrophobicity and polarity"), load_registry()
    )
    assert mapped["status"] == "unresolved"
    assert "ambiguous" in mapped["reason"]


def test_transform_directions_are_auditable():
    raw = np.array([-1.0, 0.0, 1.0])
    values, typ, label = mod.transform_for_ranking("closer_to_zero", raw, None)
    assert np.allclose(values, [1, 0, 1])
    assert typ == -1
    assert "zero" in label
    values, typ, _ = mod.transform_for_ranking("target", np.array([1.0, 2.0]), 1.5)
    assert np.allclose(values, [0.5, 0.5])
    assert typ == -1


def test_all_candidates_stay_in_problem_no_hard_gate_exclusion(monkeypatch):
    domain = {"domain_id": "D1", "name_zh": "D1", "candidates": ["F", "W", "Y"]}
    indicators = pd.DataFrame(
        [
            indicator("I1", "hydrophobicity", role="hard_gate", direction="higher"),
            indicator("I2", "side chain aromatic ring count", role="primary_rank", direction="higher"),
            indicator("I3", "side chain volume", role="primary_rank", direction="higher"),
        ]
    )
    indicators["domain_id"] = "D1"
    intrinsic = pd.DataFrame(
        {
            "aa": ["F", "W", "Y"],
            "sidechain_aromatic_ring_count": [1, 2, 1],
        }
    )

    def fake_aaindex(accession, candidates):
        assert accession == "KRIW790103"
        return {"F": 115.5, "W": 145.5, "Y": 117.3}

    monkeypatch.setattr(mod, "_load_aaindex_values", fake_aaindex)
    problem = mod.prepare_domain_problem(domain, indicators, intrinsic, load_registry(), load_config())
    assert problem["candidates"] == ["F", "W", "Y"]
    assert len(problem["candidates"]) == 3
    assert all(row["exclusion_reason"] != "hard_gate_failed" for row in problem["criteria_audit"])


def test_duplicate_registry_key_gets_one_vote(monkeypatch):
    domain = {"domain_id": "D1", "name_zh": "D1", "candidates": ["F", "W", "Y"]}
    rows = [
        indicator("I1", "hydrophobicity", direction="higher"),
        indicator("I2", "hydropathy", direction="higher"),
        indicator("I3", "side chain aromatic ring count", direction="higher"),
    ]
    indicators = pd.DataFrame(rows)
    indicators["domain_id"] = "D1"
    intrinsic = pd.DataFrame(
        {"aa": ["F", "W", "Y"], "sidechain_aromatic_ring_count": [1, 2, 1]}
    )
    monkeypatch.setattr(
        mod,
        "_load_aaindex_values",
        lambda accession, candidates: {"F": 2.8, "W": -0.9, "Y": -1.3},
    )
    problem = mod.prepare_domain_problem(domain, indicators, intrinsic, load_registry(), load_config())
    keys = [x["registry_key"] for x in problem["criteria"]]
    assert keys.count("hydropathy") == 1
    assert "aromatic_ring_count" not in keys
    assert any(x["exclusion_reason"] == "duplicate_registry_key" for x in problem["criteria_audit"])
    assert any(
        x["registry_key"] == "aromatic_ring_count"
        and x["exclusion_reason"] == "metric_validation_only_by_preregistered_domain_policy"
        for x in problem["criteria_audit"]
    )


def test_less_than_two_criteria_refuses_fake_multicriteria_result():
    problem = {
        "domain_id": "D1",
        "domain_name_zh": "D1",
        "candidates": ["F", "W", "Y"],
        "criteria": [
            {
                "ranking_values": np.array([2.8, -0.9, -1.3]),
                "criterion_type": 1,
            }
        ],
    }
    result = mod.rank_domain(problem, load_config())
    assert result["status"] == "insufficient_multicriteria_criteria"


def test_borda_rank_sum_is_independent_of_topsis():
    matrix = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
    types = np.array([1, 1])
    rank_sum, ranks, criterion_ranks = mod.borda_rank_sum(matrix, types)
    assert criterion_ranks.shape == (3, 2)
    assert np.allclose(rank_sum, [4.0, 4.0, 4.0])
    assert np.all(ranks == 1)


def test_pymcdm_wrapper_call_shape_with_stub(monkeypatch):
    # Test integration shape even when pymcdm is unavailable in the CI container.
    methods = types.ModuleType("pymcdm.methods")
    helpers = types.ModuleType("pymcdm.helpers")

    class FakeTOPSIS:
        def __call__(self, matrix, weights, criteria_types):
            return np.asarray(matrix) @ np.asarray(weights)

    methods.TOPSIS = FakeTOPSIS
    helpers.rrankdata = lambda prefs: rankdata(-np.asarray(prefs), method="min")
    monkeypatch.setitem(sys.modules, "pymcdm", types.ModuleType("pymcdm"))
    monkeypatch.setitem(sys.modules, "pymcdm.methods", methods)
    monkeypatch.setitem(sys.modules, "pymcdm.helpers", helpers)

    matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    prefs, ranks = mod.run_topsis(matrix, np.array([1, 1]))
    assert prefs[1] > prefs[0]
    assert ranks[1] == 1


def test_posthoc_reference_targets_are_explicitly_not_for_selection():
    path = ROOT / "configs" / "dlys_posthoc_reference_targets_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["not_for_selection"] is True
    targets = mod.load_posthoc_targets(path)
    assert targets == {"D1": "F", "D2": "L", "D3": "Q"}


def test_reference_targets_loaded_after_ranking_in_main_source():
    source = (ROOT / "src" / "dlys_multicriteria_ranking_v2.py").read_text(encoding="utf-8")
    rank_pos = source.index("results = [rank_domain")
    ref_pos = source.index("posthoc_targets = load_posthoc_targets")
    assert ref_pos > rank_pos


def test_registry_maps_van_der_waals_volume_to_sidechain_volume():
    mapped = mod.map_indicator_to_registry(
        indicator("I1", "side-chain van der Waals volume"), load_registry()
    )
    assert mapped["status"] == "resolved"
    assert mapped["registry_key"] == "sidechain_volume"
    assert mapped["tool_id"] == "KRIW790103"


def test_registry_maps_explicit_hbd_plus_hba_count():
    mapped = mod.map_indicator_to_registry(
        indicator("I1", "number of hydrogen-bond donors and acceptors in the side chain"),
        load_registry(),
    )
    assert mapped["status"] == "resolved"
    assert mapped["registry_key"] == "hbond_site_count"
    assert mapped["tool_id"] == "sidechain_HBD_plus_HBA"


def test_registry_does_not_substitute_rotatable_bonds_with_aaindex_flexibility():
    mapped = mod.map_indicator_to_registry(
        indicator("I1", "side-chain conformational flexibility / rotatable bond count"),
        load_registry(),
    )
    assert mapped["status"] == "unresolved"
    assert mapped["reason"] == "specific_rotatable_bond_metric_not_registered"


def test_registry_does_not_substitute_logp_with_kyte_doolittle_hydropathy():
    mapped = mod.map_indicator_to_registry(
        indicator("I1", "side-chain hydrophobicity measured by octanol-water partition coefficient logP"),
        load_registry(),
    )
    assert mapped["status"] == "unresolved"
    assert mapped["reason"] == "specific_partition_coefficient_metric_not_registered"


def test_registry_maps_formal_charge_at_ph74_to_peptides_charge():
    mapped = mod.map_indicator_to_registry(
        indicator("I3", "net formal charge of the amino acid side chain at pH 7.4", direction="closer_to_zero"),
        load_registry(),
    )
    assert mapped["status"] == "resolved"
    assert mapped["registry_key"] == "net_sidechain_charge_pH7_4"
    assert mapped["tool"] == "peptides.py"
    assert mapped["tool_id"] == "nominal_sidechain_charge_pH7_4"


def test_registry_maps_hydration_free_energy_to_registered_aaindex_scale():
    mapped = mod.map_indicator_to_registry(
        indicator("I2", "side-chain hydration free energy", direction="lower"),
        load_registry(),
    )
    assert mapped["status"] == "resolved"
    assert mapped["registry_key"] == "aqueous_solution_free_energy"
    assert mapped["tool"] == "AAindex"
    assert mapped["tool_id"] == "CHAM820102"


def test_d2_beta_branching_presence_is_rejected_by_preregistered_direction_policy():
    domain = {"domain_id": "D2", "name_zh": "D2", "candidates": ["I", "L", "V"]}
    indicators = pd.DataFrame(
        [indicator("I1", "beta branching", direction="presence")]
    )
    indicators["domain_id"] = "D2"
    intrinsic = pd.DataFrame(
        {"aa": ["I", "L", "V"], "beta_branched": [True, False, True]}
    )
    problem = mod.prepare_domain_problem(
        domain, indicators, intrinsic, load_registry(), load_config()
    )
    assert problem["criteria"] == []
    assert "direction_conflicts_with_preregistered_domain_policy" in problem["criteria_audit"][0]["exclusion_reason"]


def test_d2_beta_branching_absence_is_allowed_by_preregistered_direction_policy():
    domain = {"domain_id": "D2", "name_zh": "D2", "candidates": ["I", "L", "V"]}
    indicators = pd.DataFrame(
        [indicator("I1", "beta branching", direction="absence")]
    )
    indicators["domain_id"] = "D2"
    intrinsic = pd.DataFrame(
        {"aa": ["I", "L", "V"], "beta_branched": [True, False, True]}
    )
    problem = mod.prepare_domain_problem(
        domain, indicators, intrinsic, load_registry(), load_config()
    )
    assert len(problem["criteria"]) == 1
    criterion = problem["criteria"][0]
    assert criterion["registry_key"] == "beta_branching"
    assert criterion["preferred_direction"] == "absence"
    assert criterion["criterion_type"] == -1


def test_weight_perturbation_is_deterministic_and_not_weight_tuning(monkeypatch):
    methods = types.ModuleType("pymcdm.methods")

    class FakeTOPSIS:
        def __call__(self, matrix, weights, criteria_types):
            matrix = np.asarray(matrix, dtype=float)
            weights = np.asarray(weights, dtype=float)
            signed = matrix * np.asarray(criteria_types, dtype=float)
            return signed @ weights

    methods.TOPSIS = FakeTOPSIS
    monkeypatch.setitem(sys.modules, "pymcdm", types.ModuleType("pymcdm"))
    monkeypatch.setitem(sys.modules, "pymcdm.methods", methods)

    matrix = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    criteria = [
        {"indicator_id": "I1", "indicator_name": "one"},
        {"indicator_id": "I2", "indicator_name": "two"},
    ]
    rows, stable, total = mod.weight_perturbation_sensitivity(
        matrix, np.array([1, 1]), ["A", "B", "C"], criteria, "A", 0.20
    )
    assert total == 4
    assert stable == 4
    assert len(rows) == 4
    assert {row["perturbation"] for row in rows} == {"up", "down"}
    for row in rows:
        weights = np.array([float(x) for x in row["weights"].split("|")])
        assert np.isclose(weights.sum(), 1.0)


def test_pareto_dominance_is_reported_without_deleting_candidates():
    matrix = np.array([[3.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    rows = mod.pareto_dominance_rows(matrix, np.array([1, 1]), ["A", "B", "C"])
    assert {("A", "B"), ("A", "C")} <= {
        (r["dominant_candidate"], r["dominated_candidate"]) for r in rows
    }


def test_taxonomy_contract_rejects_stale_discovery_taxonomy(tmp_path):
    cfg = load_config()
    taxonomy = {
        "domains": [
            {"domain_id": "D1", "candidates": ["F", "W", "Y"]},
            {"domain_id": "D2", "candidates": ["A", "G", "I", "L", "M", "P", "V"]},
            {"domain_id": "D3", "candidates": ["C", "D", "E", "N", "Q", "S", "T"]},
        ],
        "candidate_names_zh": json.loads(
            (ROOT / "configs" / "dlys_stage2_taxonomy_v3.json").read_text(encoding="utf-8")
        )["candidate_names_zh"],
    }
    with pytest.raises(ValueError, match="stale discovery run"):
        mod.validate_taxonomy_against_contract(taxonomy, cfg, ROOT)


def test_d1_exact_rank_redundant_second_axis_is_not_double_counted(monkeypatch):
    domain = {"domain_id": "D1", "name_zh": "D1", "candidates": ["F", "W", "Y"]}
    rows = [
        indicator("I1", "hydrophobicity", direction="higher"),
        indicator("I2", "side-chain hydration free energy", direction="lower"),
    ]
    indicators = pd.DataFrame(rows)
    indicators["domain_id"] = "D1"
    intrinsic = pd.DataFrame({"aa": ["F", "W", "Y"]})

    def fake_aaindex(accession, candidates):
        if accession == "KYTJ820101":
            return {"F": 2.8, "W": -0.9, "Y": -1.3}
        if accession == "CHAM820102":
            # lower-is-better gives the same F > W > Y ordinal decision as hydropathy
            return {"F": 1.06, "W": 1.60, "Y": 4.91}
        raise AssertionError(accession)

    monkeypatch.setattr(mod, "_load_aaindex_values", fake_aaindex)
    problem = mod.prepare_domain_problem(domain, indicators, intrinsic, load_registry(), load_config())
    assert [c["registry_key"] for c in problem["criteria"]] == ["hydropathy"]
    dropped = [x for x in problem["criteria_audit"] if "exact_rank_redundant_with" in x["exclusion_reason"]]
    assert len(dropped) == 1
    assert dropped[0]["registry_key"] == "aqueous_solution_free_energy"


def test_d1_authorized_single_axis_fallback_selects_phe_without_fake_topsis():
    problem = {
        "domain_id": "D1",
        "domain_name_zh": "D1",
        "candidates": ["F", "W", "Y"],
        "criteria": [
            {
                "indicator_id": "I1",
                "indicator_name": "hydropathy",
                "registry_key": "hydropathy",
                "ranking_values": np.array([2.8, -0.9, -1.3]),
                "criterion_type": 1,
            }
        ],
    }
    result = mod.rank_domain(problem, load_config())
    assert result["status"] == "completed_single_criterion"
    assert result["primary_method"] == "preregistered_single_axis_rank"
    assert result["primary_winner"] == "F"
    assert "winner_topsis" not in result
    assert result["weight_perturbation_total"] == 0
