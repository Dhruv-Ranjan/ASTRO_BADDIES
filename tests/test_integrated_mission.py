from AURORA.integrated_mission import (
    MissionResources,
    plan_mission,
    update_belief_after_observation,
)


def test_plan_selects_high_value_feasible_region():
    rows = [
        {"region": "uncertain drought", "country": "India", "drought_index": 90, "confidence": 0.1, "lat": 12, "lon": 77},
        {"region": "confident drought", "country": "India", "drought_index": 90, "confidence": 0.95, "lat": 13, "lon": 78},
    ]
    plan = plan_mission(rows)
    assert plan["selection"]["region"] == "uncertain drought"
    assert plan["selection"]["information_gain"] > 0
    assert plan["resources_after"]["battery_wh"] < plan["resources_before"]["battery_wh"]


def test_plan_reports_resource_rejection():
    rows = [{"region": "low power", "drought_index": 80, "confidence": 0.1, "lat": 0, "lon": 0}]
    plan = plan_mission(rows, MissionResources(battery_wh=11.1))
    assert plan["selection"] is None
    assert plan["candidates"][0]["rejection_reason"] == "battery reserve"


def test_observation_updates_belief():
    result = update_belief_after_observation(
        {"drought_index": 70, "confidence": 0.2},
        observed_drought_index=40,
    )
    assert result["posterior_drought_index"] < 70
    assert 0.0 < result["posterior_uncertainty"] < 1.0
