import pandas as pd

from src.dlys_hydropathy_scale_audit_v1 import audit_hydropathy_tables


def _base():
    aas = "ACDEFGHIKLMNPQRSTVWY"
    return {aa: float(i) for i, aa in enumerate(aas)}


def test_scale_audit_orients_sign_and_counts_winners():
    kd = _base()
    kd.update({"F": 9.0, "W": 3.0, "Y": 2.0})
    same = {aa: 2.0 * value + 1.0 for aa, value in kd.items()}
    flipped = {aa: -value for aa, value in kd.items()}
    alt = {aa: value for aa, value in kd.items()}
    alt.update({"F": 1.0, "W": 10.0, "Y": 0.0})

    detail, summary = audit_hydropathy_tables(
        {"KyteDoolittle": kd, "Same": same, "Flipped": flipped, "Alt": alt}
    )
    assert set(detail["scale"]) == {"KyteDoolittle", "Same", "Flipped", "Alt"}
    assert detail.loc[detail["scale"] == "Flipped", "orientation_sign"].item() == -1
    assert detail.loc[detail["scale"] == "KyteDoolittle", "winner"].item() == "F"
    assert summary.iloc[0]["n_scales"] == 4
    assert 0 <= summary.iloc[0]["F_unique_winner_fraction"] <= 1


def test_scale_audit_returns_dataframes():
    kd = _base()
    kd.update({"F": 4.0, "W": 2.0, "Y": 1.0})
    detail, summary = audit_hydropathy_tables({"KyteDoolittle": kd})
    assert isinstance(detail, pd.DataFrame)
    assert isinstance(summary, pd.DataFrame)
    assert detail.iloc[0]["winner"] == "F"
