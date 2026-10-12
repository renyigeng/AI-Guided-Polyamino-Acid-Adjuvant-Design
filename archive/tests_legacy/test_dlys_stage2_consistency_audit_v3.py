from pathlib import Path

from src.dlys_stage2_consistency_audit_v3 import audit_repo

ROOT = Path(__file__).resolve().parents[1]


def test_stage2_repository_contract_is_consistent():
    report = audit_repo(ROOT)
    assert report["status"] == "passed", report["errors"]
    assert all(row["ok"] for row in report["checks"])
    assert report["frozen_domains"] == {
        "D1": ["F", "W", "Y"],
        "D2": ["A", "C", "I", "L", "M", "V"],
        "D3": ["N", "Q", "S", "T", "D", "E", "G", "P"],
    }
