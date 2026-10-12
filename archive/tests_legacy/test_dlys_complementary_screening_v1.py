from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dlys_complementary_screening_v1 import evaluate, selected_summary


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "dlys_complementary_screening_v1.json").read_text(encoding="utf-8"))


def _intrinsic() -> pd.DataFrame:
    rows = [
        # aromatic
        dict(aa="F", mechanism_domain="aromatic_hydrophobic", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=1, sidechain_HBD_reference=0, sidechain_HBA_reference=0, beta_branched=False, has_sidechain_amide=False),
        dict(aa="W", mechanism_domain="aromatic_hydrophobic", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=2, sidechain_HBD_reference=1, sidechain_HBA_reference=0, beta_branched=False, has_sidechain_amide=False),
        dict(aa="Y", mechanism_domain="aromatic_hydrophobic", nominal_sidechain_charge_pH7_4=-0.002, sidechain_aromatic_ring_count=1, sidechain_HBD_reference=1, sidechain_HBA_reference=1, beta_branched=False, has_sidechain_amide=False),
        # aliphatic
        dict(aa="A", mechanism_domain="nonaromatic_hydrophobic", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=0, sidechain_HBA_reference=0, beta_branched=False, has_sidechain_amide=False),
        dict(aa="I", mechanism_domain="nonaromatic_hydrophobic", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=0, sidechain_HBA_reference=0, beta_branched=True, has_sidechain_amide=False),
        dict(aa="L", mechanism_domain="nonaromatic_hydrophobic", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=0, sidechain_HBA_reference=0, beta_branched=False, has_sidechain_amide=False),
        dict(aa="M", mechanism_domain="nonaromatic_hydrophobic", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=0, sidechain_HBA_reference=1, beta_branched=False, has_sidechain_amide=False),
        dict(aa="V", mechanism_domain="nonaromatic_hydrophobic", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=0, sidechain_HBA_reference=0, beta_branched=True, has_sidechain_amide=False),
        dict(aa="C", mechanism_domain="nonaromatic_hydrophobic", nominal_sidechain_charge_pH7_4=-0.14, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=1, sidechain_HBA_reference=1, beta_branched=False, has_sidechain_amide=False),
        # polar
        dict(aa="N", mechanism_domain="polar_interaction_conformational", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=1, sidechain_HBA_reference=1, beta_branched=False, has_sidechain_amide=True),
        dict(aa="Q", mechanism_domain="polar_interaction_conformational", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=1, sidechain_HBA_reference=1, beta_branched=False, has_sidechain_amide=True),
        dict(aa="P", mechanism_domain="polar_interaction_conformational", nominal_sidechain_charge_pH7_4=0.0, sidechain_aromatic_ring_count=0, sidechain_HBD_reference=0, sidechain_HBA_reference=0, beta_branched=False, has_sidechain_amide=False),
    ]
    return pd.DataFrame(rows)


def _aaindex() -> pd.DataFrame:
    vals = {
        "F": (2.8, 1.13, 1.38, 5.2, 115.5, 0.314),
        "W": (-0.9, 1.08, 1.37, 5.4, 145.5, 0.305),
        "Y": (-1.3, 0.69, 1.47, 6.2, 117.3, 0.420),
        "A": (1.8, 1.42, 0.83, 8.1, 27.5, 0.357),
        "I": (4.5, 1.08, 1.60, 5.2, 93.5, 0.462),
        "L": (3.8, 1.21, 1.30, 4.9, 93.5, 0.365),
        "M": (1.9, 1.45, 1.05, 5.7, 94.1, 0.295),
        "V": (4.2, 1.06, 1.70, 5.9, 71.5, 0.386),
        "C": (2.5, 0.70, 1.19, 5.5, 44.6, 0.346),
        "N": (-3.5, 0.67, 0.89, 11.6, 58.7, 0.463),
        "Q": (-3.5, 1.11, 1.10, 10.5, 80.7, 0.493),
        "P": (-1.6, 0.57, 0.55, 8.0, 41.9, 0.509),
    }
    rows = []
    for aa, (hyd, alpha, beta, pol, vol, flex) in vals.items():
        rows.append({
            "aa": aa,
            "aaindex_hydropathy": hyd,
            "aaindex_alpha_helix": alpha,
            "aaindex_beta_sheet": beta,
            "aaindex_polarity": pol,
            "aaindex_sidechain_volume": vol,
            "aaindex_flexibility": flex,
            "aaindex_helix_minus_beta": alpha - beta,
        })
    return pd.DataFrame(rows)


def test_f_l_q_selected_without_ap() -> None:
    rank = evaluate(_intrinsic(), _aaindex(), CONFIG, ap_ranking=None)
    selected = dict(zip(rank.loc[rank.selected_representative, "selector_id"], rank.loc[rank.selected_representative, "aa"]))
    assert selected == {
        "aromatic_hydrophobic_complement": "F",
        "aliphatic_hydrophobic_conformation": "L",
        "neutral_amide_multivalent": "Q",
    }
    assert not rank["short_peptide_ap_used_for_selection"].any()


def test_tryptophan_can_have_higher_ap_without_changing_f_selection() -> None:
    ap = pd.DataFrame({
        "residue": ["F", "W", "Y"],
        "ai_assembly_prior_mean_shap": [0.068, 0.078, 0.065],
        "ai_rank_in_domain": [2, 1, 3],
    })
    rank = evaluate(_intrinsic(), _aaindex(), CONFIG, ap_ranking=ap)
    arom = rank[rank.selector_id.eq("aromatic_hydrophobic_complement")].sort_values("rank_within_selector")
    assert arom.iloc[0]["aa"] == "F"
    assert arom.loc[arom.aa.eq("W"), "short_peptide_ai_rank_diagnostic"].iloc[0] == 1
    assert arom.loc[arom.aa.eq("F"), "short_peptide_ai_rank_diagnostic"].iloc[0] == 2


def test_q_beats_n_on_frozen_neutral_amide_numeric_rules() -> None:
    rank = evaluate(_intrinsic(), _aaindex(), CONFIG, ap_ranking=None)
    polar = rank[rank.selector_id.eq("neutral_amide_multivalent")].sort_values("rank_within_selector")
    assert list(polar.head(2)["aa"]) == ["Q", "N"]
    assert polar.iloc[0]["aaindex_sidechain_volume"] > polar.iloc[1]["aaindex_sidechain_volume"]
    assert polar.iloc[0]["aaindex_flexibility"] > polar.iloc[1]["aaindex_flexibility"]


def test_i_and_v_are_blocked_by_beta_branch_gate() -> None:
    rank = evaluate(_intrinsic(), _aaindex(), CONFIG, ap_ranking=None)
    alip = rank[rank.selector_id.eq("aliphatic_hydrophobic_conformation")]
    assert not bool(alip.loc[alip.aa.eq("I"), "passes_all_mechanism_gates"].iloc[0])
    assert not bool(alip.loc[alip.aa.eq("V"), "passes_all_mechanism_gates"].iloc[0])
    assert bool(alip.loc[alip.aa.eq("L"), "passes_all_mechanism_gates"].iloc[0])


def test_selected_summary_declares_ap_nonselection() -> None:
    rank = evaluate(_intrinsic(), _aaindex(), CONFIG, ap_ranking=None)
    summary = selected_summary(rank, CONFIG)
    assert set(summary["aa"]) == {"F", "L", "Q"}
    assert not summary["short_peptide_ap_used_for_selection"].any()
