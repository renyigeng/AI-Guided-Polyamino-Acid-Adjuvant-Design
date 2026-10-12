import sys
import types

import numpy as np
import pandas as pd

from src.dlys_multicriteria_ranking_v1 import (
    _borda,
    _run_topsis,
    operationalize_gate,
    prepare_domain_problem,
    rank_domain,
    transform_for_ranking,
)


def intrinsic_df():
    return pd.DataFrame([
        {
            "aa": "F", "nominal_sidechain_charge_pH7_4": 0.0,
            "sidechain_aromatic_ring_count": 1, "sidechain_HBD_reference": 0,
            "sidechain_HBA_reference": 0, "has_sidechain_amide": False,
            "beta_branched": False,
        },
        {
            "aa": "W", "nominal_sidechain_charge_pH7_4": 0.0,
            "sidechain_aromatic_ring_count": 2, "sidechain_HBD_reference": 1,
            "sidechain_HBA_reference": 0, "has_sidechain_amide": False,
            "beta_branched": False,
        },
        {
            "aa": "Y", "nominal_sidechain_charge_pH7_4": -0.002,
            "sidechain_aromatic_ring_count": 1, "sidechain_HBD_reference": 1,
            "sidechain_HBA_reference": 1, "has_sidechain_amide": False,
            "beta_branched": False,
        },
    ])


def config():
    return {
        "ranking_policy": {
            "minimum_numeric_primary_criteria": 2,
            "near_neutral_abs_max": 0.25,
            "drop_constant_criteria": True,
            "deduplicate_same_registry_key": True,
        }
    }


def test_gate_operationalization_is_conservative():
    assert operationalize_gate("presence", 1.0, 0.25) is True
    assert operationalize_gate("absence", 0.0, 0.25) is True
    assert operationalize_gate("near_neutral", -0.1, 0.25) is True
    assert operationalize_gate("near_neutral", -1.0, 0.25) is False
    assert operationalize_gate("higher", 3.0, 0.25) is None


def test_transform_direction():
    raw = np.array([2.0, -1.0, 0.0])
    v, t, label = transform_for_ranking("higher", raw)
    assert t == 1 and np.allclose(v, raw)
    v, t, label = transform_for_ranking("lower", raw)
    assert t == -1 and np.allclose(v, raw)
    v, t, label = transform_for_ranking("near_neutral", raw)
    assert t == 1 and np.allclose(v, [-2.0, -1.0, 0.0])
    assert transform_for_ranking("context_dependent", raw) is None


def test_borda_respects_benefit_and_cost():
    matrix = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
    types = np.array([1, -1])
    points, ranks, criterion_ranks = _borda(matrix, types)
    assert int(np.argmax(points)) == 0
    assert ranks[0] == 1
    assert criterion_ranks[0, 0] == 1
    assert criterion_ranks[0, 1] == 1


def install_fake_pymcdm(monkeypatch):
    pkg = types.ModuleType("pymcdm")
    methods = types.ModuleType("pymcdm.methods")
    helpers = types.ModuleType("pymcdm.helpers")

    class TOPSIS:
        def __call__(self, matrix, weights, types):
            # Test double only: deterministic weighted signed sum, sufficient to verify integration shape.
            return np.asarray(matrix) @ (np.asarray(weights) * np.asarray(types))

    def rrankdata(values):
        from scipy.stats import rankdata
        return rankdata(-np.asarray(values), method="min")

    methods.TOPSIS = TOPSIS
    helpers.rrankdata = rrankdata
    monkeypatch.setitem(sys.modules, "pymcdm", pkg)
    monkeypatch.setitem(sys.modules, "pymcdm.methods", methods)
    monkeypatch.setitem(sys.modules, "pymcdm.helpers", helpers)


def test_run_topsis_uses_pymcdm_api(monkeypatch):
    install_fake_pymcdm(monkeypatch)
    matrix = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
    pref, ranks = _run_topsis(matrix, np.array([1, -1]))
    assert len(pref) == 3
    assert ranks[0] == 1


def test_rank_domain_requires_at_least_two_primary_criteria():
    problem = {
        "domain_id": "D1", "domain_name_zh": "芳香域", "eligible_candidates": ["F", "W", "Y"],
        "criteria": [{"indicator_id": "I1", "indicator_name": "x", "registry_key": "x",
                      "tool": "x", "tool_id": "x", "tool_label": "x", "preferred_direction": "higher",
                      "physical_rationale": "x", "criterion_type": 1, "transform": "raw",
                      "raw_values": np.array([3.,2.,1.]), "values": np.array([3.,2.,1.])}],
    }
    out = rank_domain(problem, config())
    assert out["status"] == "insufficient_multicriteria_criteria"


def test_prepare_problem_deduplicates_same_registry_and_drops_constant(monkeypatch):
    # Patch numerical resolver so this test is independent from external aaindex installation.
    import src.dlys_multicriteria_ranking_v1 as m

    values = {
        "hydropathy": {"F": 2.8, "W": -0.9, "Y": -1.3},
        "polarity": {"F": 5.2, "W": 5.4, "Y": 6.2},
        "constant": {"F": 1.0, "W": 1.0, "Y": 1.0},
    }

    def fake_resolve(key, tool, tool_id, candidates, intrinsic):
        return {aa: values[key][aa] for aa in candidates}

    monkeypatch.setattr(m, "resolve_registry_values", fake_resolve)
    plan = pd.DataFrame([
        {"domain_id":"D1","indicator_id":"I1","indicator_name":"疏水性","role":"primary_rank","preferred_direction":"higher",
         "registry_key":"hydropathy","tool":"AAindex","tool_id":"KYTJ820101","tool_label":"KD","physical_rationale":"cohesion"},
        {"domain_id":"D1","indicator_id":"I2","indicator_name":"重复疏水性","role":"tie_break","preferred_direction":"higher",
         "registry_key":"hydropathy","tool":"AAindex","tool_id":"KYTJ820101","tool_label":"KD","physical_rationale":"dup"},
        {"domain_id":"D1","indicator_id":"I3","indicator_name":"极性","role":"primary_rank","preferred_direction":"lower",
         "registry_key":"polarity","tool":"AAindex","tool_id":"GRAR740102","tool_label":"pol","physical_rationale":"hydration"},
        {"domain_id":"D1","indicator_id":"I4","indicator_name":"常数","role":"primary_rank","preferred_direction":"higher",
         "registry_key":"constant","tool":"x","tool_id":"x","tool_label":"x","physical_rationale":"constant"},
    ])
    domain = {"domain_id":"D1","name_zh":"芳香域","candidates":["F","W","Y"]}
    problem = prepare_domain_problem(domain, plan, intrinsic_df(), config())
    assert [c["registry_key"] for c in problem["criteria"]] == ["hydropathy", "polarity"]


def test_rank_domain_outputs_topsis_borda_and_loo(monkeypatch):
    install_fake_pymcdm(monkeypatch)
    problem = {
        "domain_id": "D1", "domain_name_zh": "芳香域", "eligible_candidates": ["F", "W", "Y"],
        "criteria": [
            {"indicator_id":"I1","indicator_name":"疏水性","registry_key":"hydro","tool":"x","tool_id":"x","tool_label":"x",
             "preferred_direction":"higher","physical_rationale":"x","criterion_type":1,"transform":"raw",
             "raw_values":np.array([3.,2.,1.]),"values":np.array([3.,2.,1.])},
            {"indicator_id":"I2","indicator_name":"极性","registry_key":"polar","tool":"x","tool_id":"x","tool_label":"x",
             "preferred_direction":"lower","physical_rationale":"x","criterion_type":-1,"transform":"raw",
             "raw_values":np.array([1.,2.,3.]),"values":np.array([1.,2.,3.])},
        ],
    }
    out = rank_domain(problem, config())
    assert out["status"] == "completed"
    assert out["winner_topsis"] == "F"
    assert out["winner_borda"] == "F"
    assert out["winner_agreement"] is True
    assert out["loo_total"] == 2
    assert out["loo_winner_stability_count"] == 2
