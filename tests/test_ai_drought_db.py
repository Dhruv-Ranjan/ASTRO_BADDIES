import json
from pathlib import Path

from ai_drought_db import build_location_rows, load_region_payloads


ROOT = Path(__file__).resolve().parents[1]


def test_load_region_payloads_reads_real_results_files():
    data = load_region_payloads(ROOT)
    assert isinstance(data, list)
    assert len(data) >= 5
    assert all("region" in item for item in data)
    assert all("drought_index" in item for item in data)


def test_build_location_rows_produces_db_ready_records():
    payloads = load_region_payloads(ROOT)
    rows = build_location_rows(payloads)
    assert rows
    assert rows[0]["drought_index"] >= 0
    assert rows[0]["drought_index"] <= 100
    assert rows[0]["source"] == "ai_model"
