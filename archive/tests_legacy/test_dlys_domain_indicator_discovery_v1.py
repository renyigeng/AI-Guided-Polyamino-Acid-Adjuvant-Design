import json
from pathlib import Path

import pandas as pd
import pytest

from src.dlys_domain_indicator_discovery_v1 import (
    build_discovery_prompt, build_domain_packets, build_mapping_prompt,
    validate_discovery, validate_mapping, validate_taxonomy,
)


def cfg():
    return json.loads(Path("configs/dlys_domain_indicator_discovery_v1.json").read_text(encoding="utf-8"))


def qual_df():
    rows = []
    c = cfg()
    for aa in c["candidate_names_zh"]:
        rows.append({
            "item_id": f"{aa}:x:E01", "candidate": aa, "candidate_name_zh": c["candidate_names_zh"][aa],
            "interaction_type": "hydrophobic", "support_level": "explicit", "role_summary": "native effect",
            "evidence_summary": "evidence", "title": "title", "abstract": "abstract", "qualification": "intrinsic",
            "native_x_attribution": "yes", "primary_eligible": True,
        })
    return pd.DataFrame(rows)


def test_taxonomy_covers_all_17_exactly_once():
    validate_taxonomy(cfg())


def test_discovery_prompt_does_not_expose_tool_registry_or_old_proxy_tools():
    c = cfg(); p = build_domain_packets(qual_df(), c)[0]
    text = build_discovery_prompt(p, c)
    assert "KYTJ820101" not in text
    assert "AAindex" not in text
    assert "RDKit" not in text
    assert "peptides.py" not in text
    assert "CamSol" not in text
    assert "CellPPD" not in text
    assert "XGBoost" not in text


def test_prompt_preregisters_domain_but_does_not_rank_candidates():
    c = cfg(); p = build_domain_packets(qual_df(), c)[0]
    text = build_discovery_prompt(p, c)
    assert "classification is fixed" in text
    assert "不推荐任何候选" in text
    assert "candidate_names_zh" in text


def test_packets_use_primary_eligible_only():
    c = cfg(); df = qual_df()
    df.loc[df["candidate"] == "F", "primary_eligible"] = False
    packets = build_domain_packets(df, c)
    d1 = next(p for p in packets if p["domain_id"] == "D1")
    assert all(x["candidate"] != "F" for x in d1["evidence"])


def test_validate_discovery_requires_real_evidence_ids():
    c = cfg(); p = build_domain_packets(qual_df(), c)[0]
    good_id = p["evidence"][0]["item_id"]
    obj = {"domain_id":"D1","indicators":[{
        "indicator_id":"I1","indicator_name":"hydrophobicity","role":"primary_rank",
        "preferred_direction":"higher","physical_rationale":"cohesion","evidence_item_ids":[good_id],"scope_note":"single-AA"
    }]}
    out = validate_discovery(obj, p)
    assert out["indicators"][0]["indicator_name"] == "hydrophobicity"
    bad = json.loads(json.dumps(obj)); bad["indicators"][0]["evidence_item_ids"] = ["UNKNOWN"]
    with pytest.raises(ValueError): validate_discovery(bad, p)


def test_mapping_occurs_only_after_discovery_and_cannot_add_indicator():
    c = cfg(); p = build_domain_packets(qual_df(), c)[0]
    disc = validate_discovery({"domain_id":"D1","indicators":[{
        "indicator_id":"I1","indicator_name":"hydrophobicity","role":"primary_rank",
        "preferred_direction":"higher","physical_rationale":"cohesion","evidence_item_ids":[p["evidence"][0]["item_id"]],"scope_note":"single-AA"
    }]}, p)
    prompt = build_mapping_prompt(disc, c)
    assert "KYTJ820101" in prompt
    mapped = validate_mapping({"domain_id":"D1","mappings":[{"indicator_id":"I1","registry_key":"hydropathy","mapping_reason":"match"}]}, disc, c)
    assert mapped["mappings"][0]["registry_key"] == "hydropathy"
    with pytest.raises(ValueError):
        validate_mapping({"domain_id":"D1","mappings":[
            {"indicator_id":"I1","registry_key":"hydropathy","mapping_reason":"match"},
            {"indicator_id":"I2","registry_key":"polarity","mapping_reason":"extra"}
        ]}, disc, c)


def test_scope_exclusions_are_explicit_and_not_llm_decisions():
    c = cfg()
    excluded = {x["candidate"] for x in c["scope_exclusions"]}
    assert excluded == {"C","D","E","G","P"}


def test_call_json_retries_once_on_empty_or_malformed_content(monkeypatch):
    import src.dlys_domain_indicator_discovery_v1 as mod

    calls = []
    responses = [
        ("", {"usage": {"total_tokens": 1}}),
        ('{"domain_id":"D1","indicators":[]}', {"usage": {"total_tokens": 2}}),
    ]

    def fake_call_chat(settings, messages):
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(mod, "call_chat", fake_call_chat)
    obj, raw, content = mod.call_json(object(), "system", "prompt", semantic_retries=1)
    assert obj["domain_id"] == "D1"
    assert len(calls) == 2
    assert "previous response could not be parsed" in calls[1][-1]["content"].lower()
    assert raw["usage"]["total_tokens"] == 2


def test_recover_validated_checkpoint_from_prior_raw_response(tmp_path):
    import src.dlys_domain_indicator_discovery_v1 as mod

    c = cfg()
    packet = build_domain_packets(qual_df(), c)[0]
    eid = packet["evidence"][0]["item_id"]
    raw_path = tmp_path / "raw.txt"
    ckpt = tmp_path / "checkpoint.json"
    raw_path.write_text(json.dumps({
        "domain_id": "D1",
        "indicators": [{
            "indicator_id": "I1",
            "indicator_name": "hydrophobicity",
            "role": "primary_rank",
            "preferred_direction": "higher",
            "physical_rationale": "cohesion",
            "evidence_item_ids": [eid],
            "scope_note": "single-AA",
        }],
    }), encoding="utf-8")

    recovered = mod._load_or_recover_json(
        ckpt,
        raw_path,
        lambda obj: validate_discovery(obj, packet),
    )
    assert recovered is not None
    assert ckpt.exists()
    assert recovered["indicators"][0]["indicator_id"] == "I1"


def test_validate_discovery_normalizes_common_enum_aliases():
    c = cfg(); p = build_domain_packets(qual_df(), c)[0]
    eid = p["evidence"][0]["item_id"]
    out = validate_discovery({"domain_id":"D1","indicators":[{
        "indicator_id":"I1","indicator_name":"hydrophobicity","role":"primary_indicator",
        "preferred_direction":"maximize","physical_rationale":"cohesion",
        "evidence_item_ids":[eid],"scope_note":"single-AA"
    }]}, p)
    assert out["indicators"][0]["role"] == "primary_rank"
    assert out["indicators"][0]["preferred_direction"] == "higher"


def test_call_validated_discovery_repairs_unknown_enum_once(monkeypatch):
    import src.dlys_domain_indicator_discovery_v1 as mod
    c = cfg(); packet = build_domain_packets(qual_df(), c)[0]
    eid = packet["evidence"][0]["item_id"]
    bad = {"domain_id":"D1","indicators":[{
        "indicator_id":"I1","indicator_name":"hydrophobicity","role":"important",
        "preferred_direction":"higher","physical_rationale":"cohesion",
        "evidence_item_ids":[eid],"scope_note":"single-AA"
    }]}
    good = {"domain_id":"D1","indicators":[{
        "indicator_id":"I1","indicator_name":"hydrophobicity","role":"primary_rank",
        "preferred_direction":"higher","physical_rationale":"cohesion",
        "evidence_item_ids":[eid],"scope_note":"single-AA"
    }]}
    calls=[]
    responses=[(bad,{"usage":{}},json.dumps(bad)),(good,{"usage":{}},json.dumps(good))]
    def fake_call_json(settings, system, prompt, **kwargs):
        calls.append((system,prompt)); return responses.pop(0)
    monkeypatch.setattr(mod,"call_json",fake_call_json)
    out, _, _ = mod.call_validated_discovery(object(),"system","prompt",packet)
    assert out["indicators"][0]["role"] == "primary_rank"
    assert len(calls) == 2
    assert "Change ONLY role" in calls[1][1]


def test_call_validated_discovery_does_not_repair_non_enum_validation_errors(monkeypatch):
    import src.dlys_domain_indicator_discovery_v1 as mod
    c = cfg(); packet = build_domain_packets(qual_df(), c)[0]
    bad = {"domain_id":"D1","indicators":[{
        "indicator_id":"I1","indicator_name":"x","role":"primary_rank",
        "preferred_direction":"higher","physical_rationale":"x",
        "evidence_item_ids":["UNKNOWN"],"scope_note":"x"
    }]}
    calls=[]
    def fake_call_json(settings, system, prompt, **kwargs):
        calls.append(1); return bad,{"usage":{}},json.dumps(bad)
    monkeypatch.setattr(mod,"call_json",fake_call_json)
    with pytest.raises(ValueError, match="evidence ids invalid"):
        mod.call_validated_discovery(object(),"system","prompt",packet)
    assert len(calls) == 1
