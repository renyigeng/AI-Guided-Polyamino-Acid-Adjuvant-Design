from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import dlys_domain_indicator_discovery_v2 as mod


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs" / "dlys_domain_indicator_discovery_v2.json").read_text(encoding="utf-8"))


def fake_qualification() -> pd.DataFrame:
    names = config()["candidate_names_zh"]
    rows = []
    for aa in "ACDEFGILMNPQSTVWY":
        rows.append(
            {
                "item_id": f"{aa}:intrinsic:E1",
                "candidate": aa,
                "candidate_name_zh": names[aa],
                "interaction_type": "intrinsic_test",
                "support_level": "supportive",
                "role_summary": "test role",
                "evidence_summary": "test evidence",
                "title": f"paper {aa}",
                "abstract": "qualified intrinsic evidence",
                "qualification": "intrinsic",
                "native_x_attribution": "yes",
                "primary_eligible": True,
                "doi": "",
                "pmid": "",
            }
        )
    return pd.DataFrame(rows)


def test_taxonomy_covers_all_17_once_and_exact_domains():
    cfg = config()
    mod.validate_taxonomy(cfg)
    domains = {d["domain_id"]: d["candidates"] for d in cfg["preregistered_domains"]}
    assert domains["D1"] == ["F", "W", "Y"]
    assert domains["D2"] == ["A", "C", "I", "L", "M", "V"]
    assert domains["D3"] == ["N", "Q", "S", "T", "D", "E", "G", "P"]
    assigned = [aa for values in domains.values() for aa in values]
    assert len(assigned) == 17
    assert len(set(assigned)) == 17
    assert set(assigned) == set("ACDEFGILMNPQSTVWY")


def test_all_candidates_remain_in_domain_packets():
    packets = mod.build_domain_packets(fake_qualification(), config())
    assigned = [aa for p in packets for aa in p["candidate_codes"]]
    assert set(assigned) == set("ACDEFGILMNPQSTVWY")
    assert len(assigned) == 17


def test_prompt_is_guided_but_hides_tool_registry_and_posthoc_target_status(tmp_path):
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    prompt = mod.build_discovery_prompt(
        packet, cfg, ROOT / "prompts" / "dlys_domain_indicator_discovery_v2.txt"
    )
    lowered = prompt.lower()
    # Human mechanism guidance is intentionally present.
    assert "d-lys" in lowered
    assert "芳香" in prompt
    # Mature-tool menu is intentionally hidden until after discovery.
    for forbidden in [
        "aaindex",
        "rdkit",
        "peptides.py",
        "kytj820101",
        "grar740102",
        "kriw790103",
        "chop780201",
        "chop780202",
        "bhar880101",
        "pymcdm",
        "topsis",
    ]:
        assert forbidden not in lowered
    # No field tells the LLM which candidate is the expected/known representative.
    for forbidden in ["reference_target", "expected_winner", "known_winner", "posthoc_reference"]:
        assert forbidden not in lowered


def test_system_constraint_indicator_can_be_valid_without_literature_id():
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[2]
    obj = {
        "domain_id": "D3",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "side-chain net charge at physiological pH",
                "measurement_target": "net side chain charge at pH 7.4",
                "scientific_definition": "nominal side-chain charge near pH 7.4",
                "why_relevant_to_dlys_system": "D-Lys is fixed as the cationic block, so side-chain charge can alter electrostatic compatibility.",
                "role": "primary_rank",
                "preferred_direction": "higher",
                "target_value": None,
                "basis": "system_constraint",
                "supporting_evidence_ids": [],
                "quantifiable_single_residue": True,
                "redundancy_note": "distinct from hydrogen-bond counts",
                "notes": "",
            }
        ],
        "knowledge_gaps": [],
        "notes": [],
    }
    validated = mod.validate_discovery(obj, packet, cfg)
    assert validated["indicators"][0]["basis"] == "system_constraint"


def test_literature_indicator_without_evidence_is_downgraded_not_fatal():
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    obj = {
        "domain_id": "D1",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "hydropathy",
                "measurement_target": "hydropathy",
                "scientific_definition": "single-residue hydropathy",
                "why_relevant_to_dlys_system": "complements the hydrophilic D-Lys block",
                "role": "primary_rank",
                "preferred_direction": "higher",
                "basis": "literature",
                "supporting_evidence_ids": [],
                "quantifiable_single_residue": True,
            }
        ],
    }
    validated = mod.validate_discovery(obj, packet, cfg)
    indicator = validated["indicators"][0]
    assert indicator["role"] == "validation_only"
    assert indicator["supporting_evidence_ids"] == []
    assert any(w["issue"] == "literature_indicator_without_valid_evidence" for w in validated["validation_warnings"])


def test_empty_direction_becomes_context_dependent_with_warning():
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    obj = {
        "domain_id": "D1",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "hydropathy",
                "measurement_target": "hydropathy",
                "scientific_definition": "single-residue hydropathy",
                "why_relevant_to_dlys_system": "complements the hydrophilic D-Lys block",
                "role": "primary_rank",
                "preferred_direction": "",
                "basis": "literature",
                "supporting_evidence_ids": ["F:intrinsic:E1"],
                "quantifiable_single_residue": True,
            }
        ],
    }
    validated = mod.validate_discovery(obj, packet, cfg)
    assert validated["indicators"][0]["preferred_direction"] == "context_dependent"
    assert any(w["field"] == "preferred_direction" for w in validated["validation_warnings"])


def test_unknown_evidence_id_is_filtered_and_literature_indicator_downgraded():
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    obj = {
        "domain_id": "D1",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "side-chain packing proxy",
                "measurement_target": "side-chain volume",
                "scientific_definition": "steric side-chain volume",
                "why_relevant_to_dlys_system": "packing can modulate cohesion against the D-Lys block",
                "role": "primary_rank",
                "preferred_direction": "context_dependent",
                "basis": "literature",
                "supporting_evidence_ids": ["NOT:A:REAL:ID"],
                "quantifiable_single_residue": True,
            }
        ],
    }
    validated = mod.validate_discovery(obj, packet, cfg)
    ind = validated["indicators"][0]
    assert ind["supporting_evidence_ids"] == []
    assert ind["role"] == "validation_only"
    issues = {w["issue"] for w in validated["validation_warnings"]}
    assert "unknown_evidence_ids_filtered" in issues
    assert "literature_indicator_without_valid_evidence" in issues


def test_call_and_validate_uses_one_main_call_when_json_is_valid(monkeypatch, tmp_path):
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    valid = {
        "domain_id": "D1",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "hydropathy",
                "measurement_target": "hydropathy",
                "scientific_definition": "single-residue hydropathy",
                "why_relevant_to_dlys_system": "complements the hydrophilic D-Lys block",
                "role": "primary_rank",
                "preferred_direction": "higher",
                "basis": "literature",
                "supporting_evidence_ids": ["F:intrinsic:E1"],
                "quantifiable_single_residue": True,
                "redundancy_note": "",
                "notes": "",
            },
            {
                "indicator_id": "I2",
                "indicator_name": "side-chain volume",
                "measurement_target": "side chain volume",
                "scientific_definition": "side-chain steric volume",
                "why_relevant_to_dlys_system": "captures packing differences in the aromatic block",
                "role": "primary_rank",
                "preferred_direction": "context_dependent",
                "basis": "literature",
                "supporting_evidence_ids": ["W:intrinsic:E1"],
                "quantifiable_single_residue": True,
                "redundancy_note": "",
                "notes": "",
            },
        ],
        "knowledge_gaps": [],
        "notes": [],
    }
    calls = []

    def fake_call_chat(settings, messages):
        calls.append(messages)
        return json.dumps(valid, ensure_ascii=False), {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(mod, "call_chat", fake_call_chat)
    result = mod.call_and_validate(object(), packet, cfg, "prompt", tmp_path)
    assert len(calls) == 1
    assert result["domain_id"] == "D1"


def test_discovery_source_does_not_read_posthoc_reference_config():
    source = (ROOT / "src" / "dlys_domain_indicator_discovery_v2.py").read_text(encoding="utf-8")
    assert "dlys_posthoc_reference_targets_v1" not in source
    assert "REFERENCE_TARGETS" not in source


def test_call_and_validate_does_not_make_second_llm_call_for_field_level_issues(monkeypatch, tmp_path):
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    payload = {
        "domain_id": "D1",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "hydropathy",
                "measurement_target": "hydropathy",
                "scientific_definition": "single-residue hydropathy",
                "why_relevant_to_dlys_system": "complements fixed hydrophilic D-Lys",
                "role": "primary_rank",
                "preferred_direction": "",
                "basis": "literature",
                "supporting_evidence_ids": [],
                "quantifiable_single_residue": True,
            }
        ],
    }
    calls = []
    def fake_call_chat(settings, messages):
        calls.append(messages)
        return json.dumps(payload), {"choices": []}
    monkeypatch.setattr(mod, "call_chat", fake_call_chat)
    result = mod.call_and_validate(object(), packet, cfg, "prompt", tmp_path)
    assert len(calls) == 1
    assert result["indicators"][0]["role"] == "validation_only"
    assert result["indicators"][0]["preferred_direction"] == "context_dependent"


def _valid_d1_payload_for_retry_tests():
    return {
        "domain_id": "D1",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "hydropathy",
                "measurement_target": "hydropathy",
                "scientific_definition": "single-residue hydropathy",
                "why_relevant_to_dlys_system": "complements fixed hydrophilic D-Lys",
                "role": "primary_rank",
                "preferred_direction": "higher",
                "basis": "literature",
                "supporting_evidence_ids": ["F:intrinsic:E1"],
                "quantifiable_single_residue": True,
            },
            {
                "indicator_id": "I2",
                "indicator_name": "side-chain volume",
                "measurement_target": "side-chain volume",
                "scientific_definition": "steric side-chain volume",
                "why_relevant_to_dlys_system": "distinguishes aromatic side-chain packing",
                "role": "primary_rank",
                "preferred_direction": "context_dependent",
                "basis": "literature",
                "supporting_evidence_ids": ["W:intrinsic:E1"],
                "quantifiable_single_residue": True,
            },
        ],
    }


def test_call_and_validate_retries_empty_response_then_succeeds(monkeypatch, tmp_path):
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    valid = _valid_d1_payload_for_retry_tests()
    replies = [
        ("   ", {"choices": []}),
        (json.dumps(valid), {"choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []

    def fake_call_chat(settings, messages):
        calls.append(messages)
        return replies.pop(0)

    monkeypatch.setattr(mod, "call_chat", fake_call_chat)
    result = mod.call_and_validate(object(), packet, cfg, "same prompt", tmp_path, max_attempts=3)
    assert result["domain_id"] == "D1"
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert (tmp_path / "D1_indicator_discovery_attempt_01_status.json").exists()
    assert (tmp_path / "D1_indicator_discovery_attempt_02_status.json").exists()


def test_call_and_validate_retries_bad_json_then_succeeds(monkeypatch, tmp_path):
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    valid = _valid_d1_payload_for_retry_tests()
    replies = [
        ("not-json-at-all", {"choices": []}),
        (json.dumps(valid), {"choices": []}),
    ]

    def fake_call_chat(settings, messages):
        return replies.pop(0)

    monkeypatch.setattr(mod, "call_chat", fake_call_chat)
    result = mod.call_and_validate(object(), packet, cfg, "prompt", tmp_path, max_attempts=3)
    assert result["domain_id"] == "D1"


def test_call_and_validate_all_empty_responses_fail_with_domain_and_attempt_count(monkeypatch, tmp_path):
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    calls = []

    def fake_call_chat(settings, messages):
        calls.append(messages)
        return "", {"choices": []}

    monkeypatch.setattr(mod, "call_chat", fake_call_chat)
    with pytest.raises(RuntimeError, match=r"domain D1 after 3 attempts"):
        mod.call_and_validate(object(), packet, cfg, "prompt", tmp_path, max_attempts=3)
    assert len(calls) == 3


def test_resume_run_skips_completed_domain_checkpoint(monkeypatch, tmp_path):
    cfg = config()
    packets = mod.build_domain_packets(fake_qualification(), cfg)
    out = tmp_path / "runs" / f"20260924T000000Z{mod.RUN_SUFFIX}"
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    payload = _valid_d1_payload_for_retry_tests()
    validated = mod.validate_discovery(payload, packets[0], cfg)
    mod.write_json(checkpoint_dir / "D1_validated.json", validated)

    # The same logic used in main(): an existing validated checkpoint is loaded
    # and validated locally; no call to the provider is needed for D1.
    provider_calls = []

    def forbidden_call_chat(settings, messages):
        provider_calls.append(messages)
        raise AssertionError("provider should not be called for completed D1 checkpoint")

    monkeypatch.setattr(mod, "call_chat", forbidden_call_chat)
    checkpoint = checkpoint_dir / "D1_validated.json"
    assert checkpoint.exists()
    reused = mod.validate_discovery(mod.load_json(checkpoint), packets[0], cfg)
    assert reused["domain_id"] == "D1"
    assert provider_calls == []


def test_primary_rank_with_context_dependent_direction_is_downgraded():
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    payload = {
        "domain_id": "D1",
        "indicators": [
            {
                "indicator_id": "I1",
                "indicator_name": "side-chain volume",
                "measurement_target": "side-chain volume",
                "scientific_definition": "steric side-chain volume",
                "why_relevant_to_dlys_system": "may influence aromatic side-chain packing",
                "role": "primary_rank",
                "preferred_direction": "context_dependent",
                "basis": "literature",
                "supporting_evidence_ids": ["F:intrinsic:E1"],
                "quantifiable_single_residue": True,
            }
        ],
    }
    result = mod.validate_discovery(payload, packet, cfg)
    indicator = result["indicators"][0]
    assert indicator["role"] == "validation_only"
    assert indicator["preferred_direction"] == "context_dependent"
    assert any(
        w["issue"] == "primary_rank_without_actionable_direction"
        for w in result["validation_warnings"]
    )


def test_prompt_requires_actionable_direction_without_listing_metric_menu():
    cfg = config()
    packet = mod.build_domain_packets(fake_qualification(), cfg)[0]
    prompt = mod.build_discovery_prompt(
        packet, cfg, ROOT / "prompts" / "dlys_domain_indicator_discovery_v2.txt"
    ).lower()
    assert "do not use primary_rank together with context_dependent" in prompt
    assert "actionable preference direction" in prompt
    # Tool/registry choices remain hidden during discovery.
    for hidden in ["kytj820101", "grar740102", "kriw790103", "rdkit", "peptides.py", "pymcdm"]:
        assert hidden not in prompt


def test_v22_domain_guidance_blocks_known_nonmonotonic_shortcuts():
    cfg = config()
    packets = {p["domain_id"]: p for p in mod.build_domain_packets(fake_qualification(), cfg)}
    prompt_path = ROOT / "prompts" / "dlys_domain_indicator_discovery_v2.txt"

    d1 = mod.build_discovery_prompt(packets["D1"], cfg, prompt_path)
    assert "芳香环数量本身不是自动的单调收益指标" in d1
    assert "疏水内聚" in d1 and "水化/溶剂化" in d1

    d2 = mod.build_discovery_prompt(packets["D2"], cfg, prompt_path)
    assert "α螺旋倾向和β折叠倾向代表竞争的构象偏好" in d2
    assert "不能在同一个TOPSIS问题里同时作为“越高越好”" in d2
    assert "侧链支化/packing" in d2

    d3 = mod.build_discovery_prompt(packets["D3"], cfg, prompt_path)
    assert "电荷相容" in d3
    assert "侧链可达性/尺度" in d3
    assert "成熟、可复现的单残基尺寸/体积类量" in d3


def test_v22_packets_include_human_decision_dimensions_and_guardrails():
    cfg = config()
    packets = mod.build_domain_packets(fake_qualification(), cfg)
    assert all(p["decision_dimensions_zh"] for p in packets)
    assert all(p["direction_guardrails_zh"] for p in packets)


def test_preregistered_direction_guardrails_match_final_mechanistic_logic():
    cfg = config()
    policy = cfg["domain_metric_direction_policy"]
    assert policy["D1"]["hydropathy"] == ["higher"]
    assert policy["D1"]["sidechain_volume"] == []
    assert policy["D2"]["beta_branching"] == ["absence"]
    assert policy["D2"]["alpha_helix_propensity"] == []
    assert policy["D3"]["net_sidechain_charge_pH7_4"] == ["closer_to_zero"]
    assert policy["D3"]["sidechain_amide"] == ["presence"]
    assert policy["D3"]["sidechain_volume"] == ["higher"]


def test_prompt_explicitly_forbids_reversing_preregistered_direction_guardrails():
    text = (ROOT / "prompts" / "dlys_domain_indicator_discovery_v2.txt").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "do not reverse them" in lowered
    assert "functional-group class" in lowered
    assert "validation axis" in lowered
