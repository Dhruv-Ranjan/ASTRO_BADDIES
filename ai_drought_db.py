import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "ai_drought.db"
RESULT_FILES = (
    "multi_region_sequence_drought_results.json",
    "end_to_end_known_regions_results.json",
    "end_to_end_more_regions_results.json",
)

REGION_COORDS = {
    "western ghats": {"lat": 12.9716, "lon": 77.5946},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "cape town": {"lat": -33.9249, "lon": 18.4241},
    "nairobi": {"lat": -1.2864, "lon": 36.8172},
    "sacramento": {"lat": 38.5816, "lon": -121.4944},
    "argentina": {"lat": -38.4161, "lon": -63.6167},
    "congo basin": {"lat": -2.5, "lon": 23.5},
    "india": {"lat": 20.5937, "lon": 78.9629},
    "sahel": {"lat": 15.0, "lon": 20.0},
    "ethiopia": {"lat": 9.145, "lon": 40.4897},
    "somalia": {"lat": 5.1521, "lon": 46.1996},
    "south africa": {"lat": -30.5595, "lon": 22.9375},
    "kenya": {"lat": -0.0236, "lon": 37.9062},
    "california": {"lat": 36.7783, "lon": -119.4179},
    "canada": {"lat": 56.1304, "lon": -106.3468},
    "australia": {"lat": -25.2744, "lon": 133.7751},
    "mexico": {"lat": 23.6345, "lon": -102.5528},
    "spain": {"lat": 40.4637, "lon": -3.7492},
    "morocco": {"lat": 31.7917, "lon": -7.0926},
    "turkey": {"lat": 38.9637, "lon": 35.2433},
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if number != number:
            return default
        return number
    except (TypeError, ValueError):
        return default


def normalize_region_name(name: Any) -> str:
    if name is None:
        return "Unknown region"
    text = str(name).strip()
    return text or "Unknown region"


def infer_region_coordinates(region_name: str) -> Dict[str, float]:
    key = normalize_region_name(region_name).lower()
    for candidate, coords in REGION_COORDS.items():
        if candidate in key or key in candidate:
            return coords
    return {"lat": 0.0, "lon": 0.0}


def result_coordinates(region_data: Dict[str, Any]) -> Dict[str, float] | None:
    samples = region_data.get("patches") or region_data.get("selected_candidates") or []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        lat = safe_float(sample.get("lat"), float("nan"))
        lon = safe_float(sample.get("lon"), float("nan"))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return {"lat": lat, "lon": lon}
    return None


def load_region_payloads(root: Path = ROOT) -> List[Dict[str, Any]]:
    best_rows: Dict[str, Dict[str, Any]] = {}
    primary_region_names = set()
    for filename in RESULT_FILES:
        path = root / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            continue
        for region_name, region_data in results.items():
            if not isinstance(region_data, dict):
                continue
            region_key = normalize_region_name(region_name)
            if filename == RESULT_FILES[0]:
                primary_region_names.add(region_key)
            elif region_key in primary_region_names:
                continue
            if region_data.get("status") != "ok":
                continue
            median_probability = safe_float(region_data.get("median_combined_probability"))
            mean_probability = safe_float(region_data.get("mean_combined_probability"))
            model_probability = safe_float(region_data.get("median_model_probability"))
            physics_probability = safe_float(region_data.get("median_physics_probability"))
            confidence = safe_float(region_data.get("mean_confidence"))
            ndvi = safe_float(region_data.get("mean_ndvi"))
            ndmi = safe_float(region_data.get("mean_ndmi"))
            ndwi = safe_float(region_data.get("mean_ndwi"))
            samples = region_data.get("patches") or region_data.get("selected_candidates") or []
            sample = next((item for item in samples if isinstance(item, dict)), {})
            combined_probability = median_probability or mean_probability
            candidate = {
                "region": region_key,
                "coordinates": result_coordinates(region_data),
                "status": region_data.get("status", "unknown"),
                "expected": region_data.get("expected", "unknown"),
                "combined_probability": combined_probability,
                "classifier_score": model_probability,
                "physics_score": physics_probability,
                "model_probability": model_probability,
                "physics_probability": physics_probability,
                "confidence": confidence,
                "drought_index": round((combined_probability) * 100.0, 2),
                "ndvi": ndvi,
                "ndmi": ndmi,
                "ndwi": ndwi,
                "forecast_horizon_days": int(safe_float(sample.get("forecast_horizon_days", payload.get("forecast_horizon_days")))),
                "forecast_target_date": sample.get("forecast_target_date"),
                "scenes_used": int(safe_float(region_data.get("scenes_used_per_patch", len(sample.get("scene_dates", []))))),
                "scoring_mode": sample.get("scoring_mode", "vegetation_mask"),
                "forecast_method": sample.get("forecast_method", payload.get("forecast_method", "")),
                "model_horizon_note": payload.get("model_horizon_note", ""),
                "source": "ai_model",
                "country": region_data.get("country", "global"),
            }
            current = best_rows.get(region_key)
            if current is None:
                best_rows[region_key] = candidate
    rows = sorted(best_rows.values(), key=lambda item: item["drought_index"], reverse=True)
    return rows


def build_location_rows(payloads: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in payloads:
        region = item.get("region", "Unknown region")
        coords = item.get("coordinates") or infer_region_coordinates(region)
        raw_index = float(item.get("drought_index", (item.get("combined_probability", 0.0) * 100.0)))
        drought_index = max(0.0, min(100.0, raw_index))
        rows.append(
            {
                "region": normalize_region_name(region),
                "country": item.get("country", "Global"),
                "lat": coords["lat"],
                "lon": coords["lon"],
                "drought_index": drought_index,
                "classifier_score": float(item.get("classifier_score", item.get("model_probability", 0.0))),
                "physics_score": float(item.get("physics_score", item.get("physics_probability", 0.0))),
                "confidence": float(item.get("confidence", 0.0)),
                "ndvi": float(item.get("ndvi", 0.0)),
                "ndmi": float(item.get("ndmi", 0.0)),
                "ndwi": float(item.get("ndwi", 0.0)),
                "forecast_horizon_days": int(item.get("forecast_horizon_days", 0)),
                "forecast_target_date": item.get("forecast_target_date"),
                "scenes_used": int(item.get("scenes_used", 0)),
                "scoring_mode": item.get("scoring_mode", "vegetation_mask"),
                "forecast_method": item.get("forecast_method", ""),
                "model_horizon_note": item.get("model_horizon_note", ""),
                "expected": item.get("expected", "unknown"),
                "status": item.get("status", "unknown"),
                "source": item.get("source", "ai_model"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS drought_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT NOT NULL,
            country TEXT NOT NULL,
            lat REAL,
            lon REAL,
            drought_index REAL,
            confidence REAL,
            ndvi REAL,
            ndmi REAL,
            ndwi REAL,
            expected TEXT,
            status TEXT,
            source TEXT,
            updated_at TEXT,
            forecast_horizon_days INTEGER NOT NULL DEFAULT 0,
            classifier_score REAL NOT NULL DEFAULT 0,
            physics_score REAL NOT NULL DEFAULT 0,
            forecast_target_date TEXT,
            scenes_used INTEGER NOT NULL DEFAULT 0,
            scoring_mode TEXT,
            forecast_method TEXT,
            model_horizon_note TEXT
        )
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(drought_locations)")}
    migrations = {
        "forecast_horizon_days": "INTEGER NOT NULL DEFAULT 0",
        "classifier_score": "REAL NOT NULL DEFAULT 0",
        "physics_score": "REAL NOT NULL DEFAULT 0",
        "forecast_target_date": "TEXT",
        "scenes_used": "INTEGER NOT NULL DEFAULT 0",
        "scoring_mode": "TEXT",
        "forecast_method": "TEXT",
        "model_horizon_note": "TEXT",
    }
    for column, definition in migrations.items():
        if column not in columns:
            connection.execute(f"ALTER TABLE drought_locations ADD COLUMN {column} {definition}")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_drought_region ON drought_locations(region)"
    )
    connection.commit()


def sync_database(root: Path = ROOT, database_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    payloads = load_region_payloads(root)
    rows = build_location_rows(payloads)

    snapshot_path = root / "ai_drought_snapshot.json"
    if rows:
        snapshot_path.write_text(json.dumps({"locations": rows}, indent=2), encoding="utf-8")
    elif snapshot_path.exists():
        snapshot_path.unlink()

    connection = sqlite3.connect(database_path)
    ensure_schema(connection)
    connection.execute("DELETE FROM drought_locations")
    connection.executemany(
        """
        INSERT INTO drought_locations (
            region, country, lat, lon, drought_index, confidence, ndvi, ndmi, ndwi,
            expected, status, source, updated_at, forecast_horizon_days,
            classifier_score, physics_score, forecast_target_date, scenes_used,
            scoring_mode, forecast_method, model_horizon_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["region"],
                row["country"],
                row["lat"],
                row["lon"],
                row["drought_index"],
                row["confidence"],
                row["ndvi"],
                row["ndmi"],
                row["ndwi"],
                row["expected"],
                row["status"],
                row["source"],
                row["updated_at"],
                row["forecast_horizon_days"],
                row["classifier_score"],
                row["physics_score"],
                row["forecast_target_date"],
                row["scenes_used"],
                row["scoring_mode"],
                row["forecast_method"],
                row["model_horizon_note"],
            )
            for row in rows
        ],
    )
    connection.commit()
    connection.close()
    return rows


def fetch_region_rows(database_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
         SELECT region, country, lat, lon, drought_index, confidence, ndvi, ndmi, ndwi,
             expected, status, source, updated_at, forecast_horizon_days,
               classifier_score, physics_score, forecast_target_date, scenes_used,
               scoring_mode, forecast_method, model_horizon_note
        FROM drought_locations ORDER BY drought_index DESC
        """
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    sync_database()
    print(f"AI drought database synced: {DB_PATH}")
