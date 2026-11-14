#!/usr/bin/env python3
"""
AURORA Sentinel-2 external-data ingestion and validation pipeline.

Downloads/streams Sentinel-2 L2A assets from Microsoft Planetary Computer,
computes ROI statistics and spectral indices, and writes a CSV suitable for
AURORA external-validation experiments.

Install:
  python -m pip install pystac-client planetary-computer rasterio numpy pandas shapely pyproj

Example:
  python aurora_sentinel2_planetary_computer.py \
    --bbox 77.50 12.90 77.70 13.10 \
    --start 2024-01-01 --end 2024-12-31 \
    --cloud-max 40 --output sentinel2_bengaluru.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pystac_client
import planetary_computer
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Sentinel-2 L2A surface reflectance assets.
BANDS = {
    "B02": "blue",       # 490 nm, 10 m
    "B03": "green",      # 560 nm, 10 m
    "B04": "red",        # 665 nm, 10 m
    "B08": "nir",        # 842 nm, 10 m
    "B11": "swir1",      # 1610 nm, 20 m
    "B12": "swir2",      # 2190 nm, 20 m
    "SCL": "scl",        # Scene classification, 20 m
}

# SCL classes to exclude from valid reflectance statistics.
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9/10 clouds,
# 11 snow/ice. Class 2 dark features and 7 unclassified are retained.
INVALID_SCL = {0, 1, 3, 8, 9, 10, 11}


@dataclass
class Config:
    bbox: Tuple[float, float, float, float]
    start: str
    end: str
    cloud_max: float
    output: Path
    max_items: int
    sleep_seconds: float
    save_metadata: Optional[Path]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Sentinel-2 Planetary Computer ingestion for AURORA")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                   help="AOI bounding box in WGS84 degrees")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--cloud-max", type=float, default=40.0,
                   help="Maximum scene cloud cover percentage (default: 40)")
    p.add_argument("--output", type=Path, default=Path("sentinel2_aurora_validation.csv"))
    p.add_argument("--max-items", type=int, default=100,
                   help="Maximum STAC scenes to process")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Seconds to wait between scenes")
    p.add_argument("--save-metadata", type=Path, default=None,
                   help="Optional JSON metadata output path")
    a = p.parse_args()
    if not (-180 <= a.bbox[0] < a.bbox[2] <= 180 and -90 <= a.bbox[1] < a.bbox[3] <= 90):
        p.error("Invalid WGS84 bbox")
    datetime.fromisoformat(a.start)
    datetime.fromisoformat(a.end)
    if a.start > a.end:
        p.error("--start must not be after --end")
    return Config(tuple(a.bbox), a.start, a.end, a.cloud_max, a.output,
                  a.max_items, a.sleep, a.save_metadata)


def open_catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def robust_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ("mean", "median", "std", "min", "max", "p05", "p25", "p75", "p95", "valid_pixels")}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_p05": float(np.percentile(x, 5)),
        f"{prefix}_p25": float(np.percentile(x, 25)),
        f"{prefix}_p75": float(np.percentile(x, 75)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_valid_pixels": int(x.size),
    }


def read_asset(item, key: str, geometry_wgs84) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    href = item.assets[key].href
    with rasterio.open(href) as src:
        # Sentinel-2 assets are normally in a UTM CRS, whereas the CLI bbox is
        # explicitly WGS84. rasterio.mask expects geometries in the raster's
        # CRS; passing longitude/latitude coordinates directly selects the
        # wrong area (or raises "Input shapes do not overlap raster").
        geometry = mapping(geometry_wgs84)
        if src.crs and src.crs.to_epsg() != 4326:
            geometry = transform_geom("EPSG:4326", src.crs, geometry)
        arr, transform = mask(src, [geometry], crop=True, filled=False)
        # Preserve the crop/nodata mask as NaN so invalid pixels cannot leak
        # into reflectance and index statistics during resampling.
        data = arr[0].astype(np.float32).filled(np.nan)
        return data, transform, src.crs


def resample_to_reference(data: np.ndarray, src_transform, src_crs, ref_shape, ref_transform, ref_crs) -> np.ndarray:
    """This pipeline uses nearest-neighbour resizing for ROI-level statistics.

    For rigorous pixel-level products, replace this with rasterio.warp.reproject
    using the reference grid. ROI means are generally stable, but this choice is
    recorded in the output metadata.
    """
    if data.shape == ref_shape:
        return data
    from rasterio.warp import reproject, Resampling
    out = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=data,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=np.nan,
    )
    return out


def safe_index(num, den, eps=1e-6):
    return np.where(np.abs(den) > eps, num / den, np.nan)


def process_item(item, geometry_wgs84) -> Dict[str, object]:
    # Read 10 m reference band first.
    arrays = {}
    transforms = {}
    crs = None
    for key in ("B02", "B03", "B04", "B08"):
        arr, tr, c = read_asset(item, key, geometry_wgs84)
        arrays[key] = arr / 10000.0
        transforms[key] = tr
        crs = c

    ref_shape = arrays["B04"].shape
    ref_transform = transforms["B04"]

    for key in ("B11", "B12", "SCL"):
        arr, tr, c = read_asset(item, key, geometry_wgs84)
        arrays[key] = resample_to_reference(arr, tr, c, ref_shape, ref_transform, crs)
        transforms[key] = ref_transform
        if key != "SCL":
            arrays[key] = arrays[key] / 10000.0

    scl = arrays["SCL"]
    # Use a sentinel for masked/no-data SCL pixels before converting to an
    # integer class. Directly casting NaN emits warnings and is platform
    # dependent.
    scl_classes = np.where(np.isfinite(scl), np.rint(scl), -1).astype(np.int16)
    valid = np.isfinite(arrays["B04"])
    valid &= np.isfinite(arrays["B08"])
    valid &= np.isfinite(scl)
    valid &= ~np.isin(scl_classes, list(INVALID_SCL))

    # Reflectance sanity mask.
    for key in ("B02", "B03", "B04", "B08", "B11", "B12"):
        valid &= np.isfinite(arrays[key]) & (arrays[key] >= 0) & (arrays[key] <= 1.5)

    b = {k: np.where(valid, v, np.nan) for k, v in arrays.items()}
    blue, green, red, nir = b["B02"], b["B03"], b["B04"], b["B08"]
    swir1, swir2 = b["B11"], b["B12"]

    ndvi = safe_index(nir - red, nir + red)
    ndmi = safe_index(nir - swir1, nir + swir1)
    ndwi = safe_index(green - nir, green + nir)
    savi = 1.5 * safe_index(nir - red, nir + red + 0.5)
    evi = 2.5 * safe_index(nir - red, nir + 6 * red - 7.5 * blue + 1)
    nbr = safe_index(nir - swir2, nir + swir2)

    row: Dict[str, object] = {
        "item_id": item.id,
        "collection": COLLECTION,
        "datetime_utc": item.datetime.astimezone(timezone.utc).isoformat() if item.datetime else None,
        "platform": item.properties.get("platform"),
        "constellation": item.properties.get("constellation"),
        "mgrs_tile": item.properties.get("s2:mgrs_tile"),
        "scene_cloud_cover_pct": item.properties.get("eo:cloud_cover"),
        "aoi_bbox_min_lon": geometry_wgs84.bounds[0],
        "aoi_bbox_min_lat": geometry_wgs84.bounds[1],
        "aoi_bbox_max_lon": geometry_wgs84.bounds[2],
        "aoi_bbox_max_lat": geometry_wgs84.bounds[3],
        "valid_fraction": float(np.mean(valid)),
        "scl_cloud_shadow_fraction": float(np.mean(np.isin(scl_classes, [3]))),
        "scl_cloud_fraction": float(np.mean(np.isin(scl_classes, [8, 9, 10]))),
    }
    for name, arr in [("blue", blue), ("green", green), ("red", red), ("nir", nir),
                      ("swir1", swir1), ("swir2", swir2), ("ndvi", ndvi),
                      ("ndmi", ndmi), ("ndwi", ndwi), ("savi", savi),
                      ("evi", evi), ("nbr", nbr)]:
        row.update(robust_stats(arr, name))
    return row


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = parse_args()
    geometry = box(*cfg.bbox)
    try:
        catalog = open_catalog()
    except Exception as exc:
        print(
            "Could not connect to Microsoft Planetary Computer. Check your "
            f"internet/proxy access and try again. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Searching Microsoft Planetary Computer STAC...")
    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(cfg.bbox),
        datetime=f"{cfg.start}/{cfg.end}",
        query={"eo:cloud_cover": {"lt": cfg.cloud_max}},
        max_items=cfg.max_items,
    )
    items = list(search.items())
    items.sort(key=lambda x: x.datetime or datetime.min.replace(tzinfo=timezone.utc))
    print(f"Found {len(items)} Sentinel-2 L2A scenes.")
    if not items:
        print("No scenes found. Relax --cloud-max, expand the date range, or check the bbox.", file=sys.stderr)
        return 2

    rows: List[Dict[str, object]] = []
    failures = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.id}")
        try:
            rows.append(process_item(item, geometry))
        except Exception as exc:
            failures.append({"item_id": item.id, "error": repr(exc)})
            print(f"  FAILED: {exc}", file=sys.stderr)
        if cfg.sleep_seconds:
            time.sleep(cfg.sleep_seconds)

    if not rows:
        print("Every scene failed; no CSV written.", file=sys.stderr)
        return 3

    df = pd.DataFrame(rows).sort_values("datetime_utc").reset_index(drop=True)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output, index=False, float_format="%.10g")

    metadata = {
        "pipeline": "AURORA Sentinel-2 external-data ingestion",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Microsoft Planetary Computer",
        "stac_url": STAC_URL,
        "collection": COLLECTION,
        "bbox_wgs84": list(cfg.bbox),
        "date_range": [cfg.start, cfg.end],
        "cloud_max_pct": cfg.cloud_max,
        "scenes_found": len(items),
        "scenes_processed": len(rows),
        "scenes_failed": failures,
        "bands": BANDS,
        "reflectance_scale": 0.0001,
        "invalid_scl_classes": sorted(INVALID_SCL),
        "resampling": "nearest-neighbour to B04 10-m ROI grid for 20-m assets",
        "csv_path": str(cfg.output.resolve()),
        "csv_sha256": sha256_file(cfg.output),
        "columns": list(df.columns),
        "external_validation_status": "DATA_INGESTED_NOT_YET_MODEL_VALIDATED",
        "note": "This CSV contains independent Sentinel-2 observations and ROI spectral statistics. It does not by itself prove AURORA model skill; compare predictions against these observations in a separate validation stage.",
    }
    metadata_path = cfg.save_metadata or cfg.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(df)} rows to: {cfg.output.resolve()}")
    print(f"Wrote metadata to: {metadata_path.resolve()}")
    print(f"CSV SHA-256: {metadata['csv_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
