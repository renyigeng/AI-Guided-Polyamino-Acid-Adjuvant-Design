from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.intrinsic_material_properties_v1 import (
    build_rdkit_sidechain_properties,
    candidate_domain_map,
    sidechain_atom_indices,
    spearman_redundancy,
    validate_intrinsic_table,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "intrinsic_material_properties_v1.json").read_text(encoding="utf-8"))


def test_candidate17_is_exact_and_nonoverlapping():
    mapping = candidate_domain_map(CONFIG)
    assert set(mapping) == set("FWYACILMVNQSTDEGP")
    assert len(mapping) == 17


def test_rdkit_sidechain_features_for_anchor_residues():
    pytest.importorskip("rdkit")
    table = build_rdkit_sidechain_properties(["F", "L", "Q", "A", "I", "V", "T", "P", "G"])
    table = table.set_index("aa")

    assert table.loc["F", "sidechain_aromatic_ring_count"] == 1
    assert table.loc["F", "sidechain_HBD_reference"] == 0
    assert table.loc["F", "sidechain_HBA_reference"] == 0

    assert table.loc["Q", "has_sidechain_amide"]
    assert table.loc["Q", "sidechain_HBD_reference"] == 1
    assert table.loc["Q", "sidechain_HBA_reference"] == 1

    assert not table.loc["L", "beta_branched"]
    assert table.loc["I", "beta_branched"]
    assert table.loc["V", "beta_branched"]
    assert table.loc["T", "beta_branched"]
    assert table.loc["I", "first_sidechain_branch_distance_from_CA"] == 1
    assert table.loc["V", "first_sidechain_branch_distance_from_CA"] == 1
    assert table.loc["T", "first_sidechain_branch_distance_from_CA"] == 1
    assert table.loc["L", "has_sidechain_branchpoint"]
    assert table.loc["L", "first_sidechain_branch_distance_from_CA"] == 2
    assert not table.loc["A", "has_sidechain_branchpoint"]
    assert pd.isna(table.loc["A", "first_sidechain_branch_distance_from_CA"])
    assert table.loc["P", "backbone_sidechain_ring"]
    assert table.loc["G", "sidechain_heavy_atom_count"] == 0


def test_rdkit_backbone_atom_names_are_excluded():
    Chem = pytest.importorskip("rdkit.Chem")
    mol = Chem.MolFromFASTA("Q", sanitize=True, flavor=0)
    sidechain = sidechain_atom_indices(mol)
    names = {mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip() for i in sidechain}
    assert names == {"CB", "CG", "CD", "OE1", "NE2"}


def test_redundancy_flags_use_absolute_spearman_threshold():
    frame = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 6, 8],
        "c": [4, 3, 2, 1],
        "d": [1, 4, 2, 3],
    })
    corr, flags = spearman_redundancy(frame, ["a", "b", "c", "d"], 0.9)
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    flagged = {frozenset((r.feature_a, r.feature_b)) for r in flags.itertuples()}
    assert frozenset(("a", "b")) in flagged
    assert frozenset(("a", "c")) in flagged


def test_peptides_properties_if_dependency_present():
    peptides = pytest.importorskip("peptides")
    from src.intrinsic_material_properties_v1 import build_peptide_properties

    frame = build_peptide_properties(["F", "L", "Q", "D", "E", "P"], CONFIG).set_index("aa")
    assert frame.loc["L", "kyte_doolittle"] > frame.loc["Q", "kyte_doolittle"]
    assert frame.loc["D", "nominal_sidechain_charge_pH7_4"] < -0.8
    assert frame.loc["E", "nominal_sidechain_charge_pH7_4"] < -0.8
    assert abs(frame.loc["P", "nominal_sidechain_charge_pH7_4"]) < 1e-9
    assert all(f"KF{i}" in frame.columns for i in range(1, 11))


def test_full_candidate17_validation_allows_structural_na_branch_distance():
    pytest.importorskip("rdkit")
    residues = list("FWYACILMVNQSTDEGP")
    table = build_rdkit_sidechain_properties(residues)
    table.insert(1, "mechanism_domain", table["aa"].map(candidate_domain_map(CONFIG)))

    # Add the required peptide-derived numeric columns with harmless test values;
    # this test targets the structural-nullability contract only.
    for i in range(1, 11):
        table[f"KF{i}"] = 0.0
    table["kyte_doolittle"] = 0.0
    table["nominal_sidechain_charge_pH7_4"] = 0.0

    validate_intrinsic_table(table, CONFIG)
    by_aa = table.set_index("aa")
    assert pd.isna(by_aa.loc["F", "first_sidechain_branch_distance_from_CA"])
    assert by_aa.loc["L", "first_sidechain_branch_distance_from_CA"] == 2
    assert by_aa.loc["T", "first_sidechain_branch_distance_from_CA"] == 1
