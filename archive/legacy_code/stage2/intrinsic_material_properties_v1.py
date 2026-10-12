from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "intrinsic_material_properties_v1.json"
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_domain_map(config: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for domain, residues in config["candidate17"].items():
        for residue in residues:
            residue = str(residue)
            if residue in mapping:
                raise ValueError(f"Residue {residue} appears in more than one mechanism domain.")
            mapping[residue] = str(domain)
    expected = set("FWYACILMVNQSTDEGP")
    if set(mapping) != expected:
        raise ValueError(
            "candidate17 mechanism domains do not match the frozen 17-residue set: "
            f"missing={sorted(expected - set(mapping))}, extra={sorted(set(mapping) - expected)}"
        )
    return mapping


def _require_peptides():
    try:
        import peptides
    except ImportError as exc:  # pragma: no cover - exercised in user environment
        raise RuntimeError(
            "peptides.py is required. Install the frozen version with: "
            "python -m pip install peptides==0.5.0"
        ) from exc
    return peptides


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ImportError as exc:  # pragma: no cover - exercised in user environment
        raise RuntimeError(
            "RDKit is required. Install the frozen version with: "
            "python -m pip install rdkit==2026.3.6"
        ) from exc
    return Chem, Lipinski


def build_peptide_properties(residues: list[str], config: dict) -> pd.DataFrame:
    peptides = _require_peptides()
    scale = str(config["peptides"]["hydropathy_scale"])
    pka_scale = str(config["peptides"]["pka_scale"])
    ph = float(config["peptides"]["charge_pH"])
    flank = str(config["peptides"]["internal_charge_flank"])
    if not flank or set(flank) != {"G"}:
        raise ValueError("internal_charge_flank must contain glycine residues only.")

    hydropathy_table = peptides.tables.HYDROPHOBICITY[scale]
    reference = peptides.Peptide(flank + "G" + flank).charge(pH=ph, pKscale=pka_scale)

    rows: list[dict[str, object]] = []
    for aa in residues:
        peptide = peptides.Peptide(aa)
        kidera = list(peptide.kidera_factors())
        internal_sequence = flank + aa + flank
        internal_charge = (
            peptides.Peptide(internal_sequence).charge(pH=ph, pKscale=pka_scale) - reference
        )
        row: dict[str, object] = {
            "aa": aa,
            "kyte_doolittle": float(hydropathy_table[aa]),
            "nominal_sidechain_charge_pH7_4": float(internal_charge),
        }
        for index, value in enumerate(kidera, start=1):
            row[f"KF{index}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def sidechain_atom_indices(mol) -> set[int]:
    indices: set[int] = set()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise ValueError("RDKit FASTA molecule is missing PDB residue atom names.")
        if info.GetName().strip() not in BACKBONE_ATOM_NAMES:
            indices.add(atom.GetIdx())
    return indices


def _count_sidechain_lipinski_sites(mol, sidechain: set[int], query) -> int:
    count = 0
    for match in mol.GetSubstructMatches(query):
        atoms = set(int(i) for i in match)
        if atoms and atoms.issubset(sidechain):
            count += 1
    return count


def _has_sidechain_smarts(mol, sidechain: set[int], smarts: str) -> bool:
    Chem, _ = _require_rdkit()
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Invalid SMARTS pattern: {smarts}")
    for match in mol.GetSubstructMatches(query):
        atoms = set(int(i) for i in match)
        if atoms and atoms.issubset(sidechain):
            return True
    return False


def _branch_metrics(mol, sidechain: set[int]) -> tuple[bool, bool, int | None]:
    ca = None
    cb = None
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetName().strip() if info else ""
        if name == "CA":
            ca = atom.GetIdx()
        elif name == "CB":
            cb = atom.GetIdx()
    if ca is None or cb is None:
        return False, False, None

    def is_sp3_carbon_branchpoint(idx: int) -> bool:
        # A structural branch point must be tetrahedral carbon with at least
        # three heavy-atom neighbours. Counting only carbon neighbours would
        # incorrectly miss threonine, whose beta carbon is attached to both
        # a methyl group and a hydroxyl oxygen. Requiring SP3 excludes carbonyl
        # carbons in amides/carboxylates from being misclassified as branches.
        from rdkit import Chem

        atom = mol.GetAtomWithIdx(idx)
        if idx not in sidechain or atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
            return False
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            return False
        heavy_neighbors = sum(n.GetAtomicNum() > 1 for n in atom.GetNeighbors())
        return heavy_neighbors >= 3

    beta_branched = is_sp3_carbon_branchpoint(cb)
    candidates: list[tuple[int, int]] = []
    for idx in sidechain:
        if is_sp3_carbon_branchpoint(idx):
            from rdkit import Chem
            distance = len(Chem.GetShortestPath(mol, ca, idx)) - 1
            candidates.append((distance, idx))
    if not candidates:
        return beta_branched, False, None
    return beta_branched, True, min(candidates)[0]


def build_rdkit_sidechain_properties(residues: list[str]) -> pd.DataFrame:
    Chem, Lipinski = _require_rdkit()
    rows: list[dict[str, object]] = []
    for aa in residues:
        mol = Chem.MolFromFASTA(aa, sanitize=True, flavor=0)
        if mol is None:
            raise ValueError(f"RDKit MolFromFASTA failed for residue {aa}.")
        sidechain = sidechain_atom_indices(mol)
        atom_rings = mol.GetRingInfo().AtomRings()
        aromatic_rings = 0
        sidechain_rings = 0
        backbone_sidechain_ring = False
        for ring in atom_rings:
            ring_set = set(int(i) for i in ring)
            if ring_set.issubset(sidechain):
                sidechain_rings += 1
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
                    aromatic_rings += 1
            elif ring_set & sidechain:
                backbone_sidechain_ring = True

        beta_branched, has_branchpoint, first_branch_distance = _branch_metrics(mol, sidechain)
        rows.append(
            {
                "aa": aa,
                "canonical_smiles_rdkit": Chem.MolToSmiles(mol, canonical=True),
                "sidechain_heavy_atom_count": int(len(sidechain)),
                "sidechain_heteroatom_count": int(
                    sum(mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6) for i in sidechain)
                ),
                "sidechain_aromatic_atom_count": int(
                    sum(mol.GetAtomWithIdx(i).GetIsAromatic() for i in sidechain)
                ),
                "sidechain_aromatic_ring_count": int(aromatic_rings),
                "sidechain_ring_count": int(sidechain_rings),
                "backbone_sidechain_ring": bool(backbone_sidechain_ring),
                "sidechain_HBD_reference": int(
                    _count_sidechain_lipinski_sites(mol, sidechain, Lipinski.HDonorSmarts)
                ),
                "sidechain_HBA_reference": int(
                    _count_sidechain_lipinski_sites(mol, sidechain, Lipinski.HAcceptorSmarts)
                ),
                "beta_branched": bool(beta_branched),
                "has_sidechain_branchpoint": bool(has_branchpoint),
                "first_sidechain_branch_distance_from_CA": first_branch_distance,
                "has_sidechain_amide": _has_sidechain_smarts(mol, sidechain, "[NX3][CX3](=[OX1])"),
                "has_sidechain_hydroxyl": _has_sidechain_smarts(mol, sidechain, "[#6;X4,c][OX2H]"),
                "has_sidechain_carboxylic_acid": _has_sidechain_smarts(
                    mol, sidechain, "[CX3](=[OX1])[OX2H1]"
                ),
                "has_sidechain_thiol": _has_sidechain_smarts(mol, sidechain, "[SX2H]"),
                "has_sidechain_thioether": _has_sidechain_smarts(
                    mol, sidechain, "[#6][SX2][#6]"
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_intrinsic_table(frame: pd.DataFrame, config: dict) -> None:
    domains = candidate_domain_map(config)
    if frame["aa"].duplicated().any():
        raise ValueError("Intrinsic table contains duplicate amino-acid rows.")
    if set(frame["aa"]) != set(domains):
        raise ValueError("Intrinsic table does not contain exactly the frozen 17 candidates.")
    # Branch distance is structurally undefined for an unbranched side chain.
    # Keep it nullable rather than encoding an arbitrary sentinel such as 0,
    # which would create a false quantitative meaning in downstream analyses.
    nullable_numeric = {"first_sidechain_branch_distance_from_CA"}
    numeric = frame.select_dtypes(include=[np.number])
    required_numeric = numeric.drop(
        columns=[c for c in nullable_numeric if c in numeric.columns],
        errors="ignore",
    )
    if required_numeric.isna().any().any():
        bad = required_numeric.columns[required_numeric.isna().any()].tolist()
        raise ValueError(f"Intrinsic numeric table contains missing values: {bad}")

    if "first_sidechain_branch_distance_from_CA" in frame.columns:
        distance = frame["first_sidechain_branch_distance_from_CA"]
        has_branch = frame["has_sidechain_branchpoint"].astype(bool)
        inconsistent = (has_branch & distance.isna()) | (~has_branch & distance.notna())
        if inconsistent.any():
            bad_residues = frame.loc[inconsistent, "aa"].tolist()
            raise ValueError(
                "Branch-distance nullability is inconsistent with "
                f"has_sidechain_branchpoint for residues: {bad_residues}"
            )
        if ((distance.dropna() < 1) | (distance.dropna() % 1 != 0)).any():
            raise ValueError("Branch distance must be a positive integer when defined.")

    if "beta_branched" in frame.columns:
        beta = frame["beta_branched"].astype(bool)
        distance = frame["first_sidechain_branch_distance_from_CA"]
        inconsistent_beta = beta & (distance != 1)
        if inconsistent_beta.fillna(False).any():
            bad_residues = frame.loc[inconsistent_beta.fillna(False), "aa"].tolist()
            raise ValueError(
                "beta_branched residues must have their first branch point "
                f"one bond from CA: {bad_residues}"
            )


def spearman_redundancy(frame: pd.DataFrame, columns: list[str], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame[columns].astype(float)
    corr = data.rank(method="average").corr(method="pearson")
    rows: list[dict[str, object]] = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            rho = float(corr.loc[left, right])
            if abs(rho) >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "spearman_rho": rho})
    pairs = pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman_rho"])
    if not pairs.empty:
        pairs = pairs.reindex(pairs["spearman_rho"].abs().sort_values(ascending=False).index)
    return corr, pairs


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intrinsic material-property table for the frozen 17 X residues.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    domains = candidate_domain_map(config)
    residues = [aa for domain in config["candidate17"].values() for aa in domain]

    peptide = build_peptide_properties(residues, config)
    structural = build_rdkit_sidechain_properties(residues)
    frame = peptide.merge(structural, on="aa", validate="one_to_one")
    frame.insert(1, "mechanism_domain", frame["aa"].map(domains))
    validate_intrinsic_table(frame, config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "runs" / f"{timestamp}__intrinsic-material-properties-v1"
    output_dir.mkdir(parents=True, exist_ok=False)

    frame.to_csv(output_dir / "intrinsic_properties_candidate17.csv", index=False)

    redundancy_columns = list(config["redundancy"]["continuous_features"])
    corr, pairs = spearman_redundancy(
        frame,
        redundancy_columns,
        float(config["redundancy"]["flag_abs_spearman_at_or_above"]),
    )
    corr.to_csv(output_dir / "continuous_feature_spearman.csv")
    pairs.to_csv(output_dir / "continuous_feature_redundancy_flags.csv", index=False)

    manifest = {
        "module": config["module"],
        "version": config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": int(len(frame)),
        "candidate17": config["candidate17"],
        "sources": config["sources"],
        "peptides_settings": config["peptides"],
        "redundancy": config["redundancy"],
        "method_freeze": config["method_freeze"],
        "packages": {
            "peptides": package_version("peptides"),
            "rdkit": package_version("rdkit"),
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
        },
        "config_sha256": sha256_file(args.config),
        "outputs": {
            "intrinsic_table": "intrinsic_properties_candidate17.csv",
            "spearman_matrix": "continuous_feature_spearman.csv",
            "redundancy_flags": "continuous_feature_redundancy_flags.csv",
        },
    }
    write_json(output_dir / "analysis_manifest.json", manifest)

    print("=" * 84)
    print("Intrinsic material properties v1")
    print("=" * 84)
    print(f"candidates       : {len(frame)}")
    print("continuous space : Kidera10 + Kyte-Doolittle + nominal side-chain charge")
    print("structure        : RDKit MolFromFASTA(L-AA) -> side-chain-specific features")
    print("ranking          : NONE (descriptive / redundancy audit only)")
    print(f"run              : {output_dir}")
    print(f"redundancy flags : {len(pairs)} pairs with |Spearman rho| >= {config['redundancy']['flag_abs_spearman_at_or_above']}")


if __name__ == "__main__":
    main()
