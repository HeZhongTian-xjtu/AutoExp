from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8")) or {}


def test_registered_dataset_contract() -> None:
    manifest = _manifest()
    contract = manifest["dataset_contract"]
    data_root = ROOT / "data"
    if not (data_root / "train.csv").is_file():
        pytest.skip("dataset is staged only in a Trial workspace")
    for filename in contract.get("required_files", ["train.csv"]):
        assert (data_root / filename).is_file(), f"missing required dataset file: {filename}"

    with (data_root / "train.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert set(contract.get("required_columns", [])) <= set(header)


def test_runtime_contract_is_complete() -> None:
    manifest = _manifest()
    config_path = ROOT / "configs" / "experiment.yaml"
    if not config_path.is_file():
        pytest.skip("runtime config is generated only in a Trial workspace")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    assert (ROOT / manifest["entrypoint"]).is_file()
    assert (ROOT / manifest["evaluator_entrypoint"]).is_file()
    assert (ROOT / "smoke.py").is_file()
    assert set(manifest.get("parameter_policy", {})) <= set(config)
    json.dumps(config)
