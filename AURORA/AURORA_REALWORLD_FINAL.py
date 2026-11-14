#!/usr/bin/env python3
"""
AURORA REAL-WORLD DROUGHT FORECASTING ENGINE
============================================
Single-file, end-to-end research pipeline.

What this program does
----------------------
1. Finds and combines existing AURORA Sentinel-2 scene-statistics CSV files.
2. If only a short/one-year Sentinel dataset exists, optionally downloads
   additional Sentinel-2 L2A scenes from Microsoft Planetary Computer.
3. Downloads daily NASA POWER precipitation and temperature for the AOI.
4. Builds strictly causal antecedent climate features (7/15/30/60/90 days).
5. Builds a future 30-day meteorological drought target from precipitation
   anomalies. Future data are NEVER used as predictor features.
6. Builds an independent soil-moisture target when an external GLDAS file is
   supplied, and automatically uses it if present.
7. Trains several regression models: climatology baseline, Ridge,
   Random Forest, Gradient Boosting, Extra Trees.
8. Uses chronological train/validation/test splits and TimeSeriesSplit CV.
9. Tunes the best model on validation data only.
10. Evaluates on a completely held-out chronological test set.
11. Produces MAE, RMSE, R2, Pearson r, Spearman rho, drought classification
    metrics, calibration/uncertainty estimates, feature importance and plots.
12. Writes a machine-readable dataset, predictions, metrics JSON, report and
    figures into one output directory.

IMPORTANT SCIENTIFIC CLAIM
--------------------------
This is a real predictive modelling pipeline, but no program can manufacture
"real-world accuracy" from insufficient data. If you only have the original
27 Sentinel-2 observations from 2024, the script will explicitly warn that the
ML results are exploratory. For a defensible predictive study, use multiple
years and preferably multiple spatial cells, and validate against an external
soil-moisture/agricultural reference rather than only rainfall.

Recommended installation
-------------------------
python -m pip install numpy pandas requests matplotlib scikit-learn scipy

Optional Sentinel-2 expansion dependencies
-------------------------------------------
python -m pip install pystac-client planetary-computer rasterio shapely

Run
---
python AURORA_REALWORLD_FINAL.py

Or
---
python AURORA_REALWORLD_FINAL.py --input sentinel2_bengaluru_2024.csv

Useful options
--------------
--start 2018-01-01 --end 2025-12-31
--bbox 77.55 12.90 77.70 13.05
--output-dir AURORA_REALWORLD_OUTPUT
--download-sentinel
--max-scenes 500
--cloud-max 60

The default behavior uses local Sentinel CSV data when available and does not
redownload imagery unless --download-sentinel is specified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        accuracy_score, balanced_accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score, confusion_matrix,
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    start: str = "2018-01-01"
    end: str = "2025-12-31"
    bbox: Tuple[float, float, float, float] = (77.55, 12.90, 77.70, 13.05)
    output_dir: str = "AURORA_REALWORLD_OUTPUT"
    input_csv: Optional[str] = None
    download_sentinel: bool = False
    max_scenes: int = 500
    cloud_max: float = 60.0
    rain_timeout: int = 60
    future_days: int = 30
    baseline_days: int = 365
    random_seed: int = 42


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def finite(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def safe_json(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    return obj


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def corr_pair(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or np.std(y[m]) == 0 or np.std(p[m]) == 0:
        return np.nan, np.nan
    if HAS_SCIPY:
        return float(pearsonr(y[m], p[m]).statistic), float(spearmanr(y[m], p[m]).statistic)
    return float(np.corrcoef(y[m], p[m])[0, 1]), float(pd.Series(y[m]).corr(pd.Series(p[m]), method="spearman"))


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Sentinel CSV discovery / cleaning
# ---------------------------------------------------------------------------

def discover_csvs(cfg: Config) -> List[Path]:
    if cfg.input_csv:
        p = Path(cfg.input_csv)
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")
        return [p]

    roots = [Path.cwd(), Path(__file__).resolve().parent]
    candidates = []
    seen = set()
    for root in roots:
        for p in root.glob("*.csv"):
            if p.resolve() in seen:
                continue
            seen.add(p.resolve())
            try:
                cols = set(pd.read_csv(p, nrows=2).columns)
            except Exception:
                continue
            score = 0
            if "datetime_utc" in cols:
                score += 5
            if any(c.startswith("ndvi") for c in cols):
                score += 3
            if any(c.startswith("ndmi") for c in cols):
                score += 3
            if "scene_id" in cols or "platform" in cols:
                score += 2
            if score >= 5:
                candidates.append((score, p))
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    return [p for _, p in candidates]


def infer_bbox_from_df(df: pd.DataFrame, fallback):
    names = [
        ("aoi_bbox_min_lon", "aoi_bbox_min_lat", "aoi_bbox_max_lon", "aoi_bbox_max_lat"),
        ("bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"),
    ]
    for a, b, c, d in names:
        if all(x in df.columns for x in (a, b, c, d)):
            vals = [finite(df[x].median()) for x in (a, b, c, d)]
            if all(np.isfinite(vals)):
                return tuple(vals)
    return fallback


def clean_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime_utc" not in df.columns:
        raise ValueError("Sentinel CSV must contain datetime_utc")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)

    # Preserve provenance fields such as STAC item IDs.  Earlier versions
    # blindly coerced every object column, silently turning identifiers into
    # NaN in the output dataset.
    identifier_cols = {
        "item_id", "scene_id", "platform", "constellation", "mgrs_tile",
        "collection", "datetime_utc",
    }
    for c in df.columns:
        if c in identifier_cols or df[c].dtype != object:
            continue
        original = df[c].notna().sum()
        converted = pd.to_numeric(df[c], errors="coerce")
        # Convert textual CSV representations only when the column is
        # genuinely numeric; retain categorical/provenance columns otherwise.
        if original and converted.notna().sum() >= original * 0.95:
            df[c] = converted

    # EVI can explode when denominators are close to zero. Keep physically
    # plausible values only; do not silently invent values.
    for c in [c for c in df.columns if c.startswith("evi_")]:
        x = pd.to_numeric(df[c], errors="coerce")
        df.loc[(x < -1.5) | (x > 1.5), c] = np.nan

    # Cloudy scenes are retained in the raw dataset but get a lower confidence.
    if "scene_cloud_cover_pct" in df.columns:
        cloud = pd.to_numeric(df["scene_cloud_cover_pct"], errors="coerce")
        df["scene_quality_weight"] = np.clip(1.0 - cloud / 100.0, 0.05, 1.0)
    else:
        df["scene_quality_weight"] = 1.0

    return df


def combine_sentinel_csvs(paths: List[Path]) -> Tuple[pd.DataFrame, Dict]:
    if not paths:
        raise FileNotFoundError(
            "No Sentinel-2 scene CSV found. Put sentinel2_*.csv beside this script "
            "or run with --download-sentinel."
        )
    frames = []
    used = []
    for p in paths:
        try:
            x = pd.read_csv(p)
            if "datetime_utc" not in x.columns:
                continue
            x = clean_sentinel(x)
            frames.append(x)
            used.append(str(p.resolve()))
        except Exception as exc:
            print(f"WARNING: could not read {p}: {exc}")
    if not frames:
        raise RuntimeError("No usable Sentinel CSV files were found.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = clean_sentinel(df)
    return df, {"files": used, "rows": int(len(df))}


# ---------------------------------------------------------------------------
# Optional Microsoft Planetary Computer Sentinel-2 ingestion
# ---------------------------------------------------------------------------

def download_sentinel_stats(cfg: Config, out_csv: Path) -> Optional[Path]:
    try:
        import pystac_client
        import planetary_computer
        import rasterio
        from rasterio.windows import from_bounds
    except Exception as exc:
        print("Sentinel download skipped: optional packages are missing.")
        print("Install with: python -m pip install pystac-client planetary-computer rasterio shapely")
        print(f"Reason: {exc}")
        return None

    try:
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=list(cfg.bbox),
            datetime=f"{cfg.start}/{cfg.end}",
            query={"eo:cloud_cover": {"lt": cfg.cloud_max}},
            max_items=cfg.max_scenes,
        )
        items = list(search.items())
        items.sort(key=lambda x: x.datetime or pd.Timestamp.min.to_pydatetime())
    except Exception as exc:
        print(f"ERROR: Planetary Computer search failed: {exc}")
        return None

    if not items:
        print("No Sentinel-2 scenes found.")
        return None

    print(f"Found {len(items)} Sentinel-2 scenes. Downloading scene statistics...")
    rows = []
    minlon, minlat, maxlon, maxlat = cfg.bbox

    # Asset names in Sentinel-2 L2A COGs.
    bands = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir1", "B12": "swir2"}
    scl_name = "SCL"

    def stats(arr, prefix):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return {f"{prefix}_{k}": np.nan for k in ["mean", "median", "std", "min", "max", "p10", "p90"]}
        return {
            f"{prefix}_mean": float(np.mean(a)),
            f"{prefix}_median": float(np.median(a)),
            f"{prefix}_std": float(np.std(a)),
            f"{prefix}_min": float(np.min(a)),
            f"{prefix}_max": float(np.max(a)),
            f"{prefix}_p10": float(np.percentile(a, 10)),
            f"{prefix}_p90": float(np.percentile(a, 90)),
        }

    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item.id}")
        try:
            signed = planetary_computer.sign(item)
            arrays = {}
            ref_shape = None
            ref_transform = None
            ref_crs = None
            for asset_key, name in bands.items():
                if asset_key not in signed.assets:
                    raise RuntimeError(f"missing asset {asset_key}")
                with rasterio.open(signed.assets[asset_key].href) as src:
                    from rasterio.warp import transform_bounds
                    x0, y0, x1, y1 = transform_bounds("EPSG:4326", src.crs, minlon, minlat, maxlon, maxlat, densify_pts=21)
                    win = from_bounds(x0, y0, x1, y1, transform=src.transform).round_offsets().round_lengths()
                    arr = np.asarray(src.read(1, window=win, masked=True).astype("float32").filled(np.nan)) * 1e-4
                    arr_transform = src.window_transform(win)
                    # B11/B12 are 20 m while B02/B03/B04/B08 are 10 m. Every
                    # band must share one grid before band arithmetic/masking.
                    if ref_shape is None:
                        ref_shape, ref_transform, ref_crs = arr.shape, arr_transform, src.crs
                    elif arr.shape != ref_shape or arr_transform != ref_transform or src.crs != ref_crs:
                        from rasterio.warp import Resampling, reproject
                        aligned = np.full(ref_shape, np.nan, dtype=np.float32)
                        reproject(
                            source=arr, destination=aligned,
                            src_transform=arr_transform, src_crs=src.crs,
                            dst_transform=ref_transform, dst_crs=ref_crs,
                            src_nodata=np.nan, dst_nodata=np.nan,
                            resampling=Resampling.bilinear,
                        )
                        arr = aligned
                    arrays[name] = arr
            if "SCL" in signed.assets:
                with rasterio.open(signed.assets[scl_name].href) as src:
                    from rasterio.warp import transform_bounds
                    x0, y0, x1, y1 = transform_bounds("EPSG:4326", src.crs, minlon, minlat, maxlon, maxlat, densify_pts=21)
                    win = from_bounds(x0, y0, x1, y1, transform=src.transform).round_offsets().round_lengths()
                    scl = np.asarray(src.read(1, window=win, masked=True).astype("float32").filled(np.nan))
                    scl_transform = src.window_transform(win)
                    if scl.shape != ref_shape or scl_transform != ref_transform or src.crs != ref_crs:
                        from rasterio.warp import Resampling, reproject
                        aligned_scl = np.full(ref_shape, np.nan, dtype=np.float32)
                        reproject(
                            source=scl, destination=aligned_scl,
                            src_transform=scl_transform, src_crs=src.crs,
                            dst_transform=ref_transform, dst_crs=ref_crs,
                            src_nodata=np.nan, dst_nodata=np.nan,
                            resampling=Resampling.nearest,
                        )
                        scl = aligned_scl
            else:
                scl = np.full(ref_shape, np.nan, dtype=np.float32)

            scl_classes = np.where(np.isfinite(scl), np.rint(scl), -1).astype(np.int16)
            valid = np.logical_and.reduce([np.isfinite(arr) for arr in arrays.values()])
            valid &= np.isfinite(scl)
            valid &= ~np.isin(scl_classes, [0, 1, 3, 8, 9, 10, 11])

            b = {name: np.where(valid, arr, np.nan) for name, arr in arrays.items()}
            eps = 1e-8
            ndvi = (b["nir"] - b["red"]) / (b["nir"] + b["red"] + eps)
            ndmi = (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"] + eps)
            ndwi = (b["green"] - b["nir"]) / (b["green"] + b["nir"] + eps)
            savi = 1.5 * (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.5 + eps)
            evi = 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6*b["red"] - 7.5*b["blue"] + 1 + eps)
            nbr = (b["nir"] - b["swir2"]) / (b["nir"] + b["swir2"] + eps)

            row = {
                "datetime_utc": item.datetime.isoformat() if item.datetime else None,
                "scene_id": item.id,
                "platform": item.properties.get("platform"),
                "scene_cloud_cover_pct": item.properties.get("eo:cloud_cover"),
                "aoi_bbox_min_lon": minlon,
                "aoi_bbox_min_lat": minlat,
                "aoi_bbox_max_lon": maxlon,
                "aoi_bbox_max_lat": maxlat,
                "valid_fraction": float(np.mean(valid)) if valid is not None else np.nan,
                "scl_cloud_shadow_fraction": float(np.mean(np.isin(scl_classes, [3]))),
                "scl_cloud_fraction": float(np.mean(np.isin(scl_classes, [8, 9, 10]))),
            }
            for name, arr in list(b.items()) + [("ndvi", ndvi), ("ndmi", ndmi), ("ndwi", ndwi), ("savi", savi), ("evi", evi), ("nbr", nbr)]:
                row.update(stats(arr, name))
            rows.append(row)
        except Exception as exc:
            print(f"    FAILED: {exc}")

    if not rows:
        return None
    result = clean_sentinel(pd.DataFrame(rows))
    result.to_csv(out_csv, index=False, float_format="%.10g")
    print(f"Saved expanded Sentinel dataset: {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# NASA POWER climate data
# ---------------------------------------------------------------------------

def power_point(cfg: Config, params: List[str]) -> pd.DataFrame:
    lat = (cfg.bbox[1] + cfg.bbox[3]) / 2
    lon = (cfg.bbox[0] + cfg.bbox[2]) / 2
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    query = {
        "parameters": ",".join(params),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": pd.Timestamp(cfg.start).strftime("%Y%m%d"),
        "end": pd.Timestamp(cfg.end).strftime("%Y%m%d"),
        "format": "JSON",
    }
    r = requests.get(url, params=query, timeout=cfg.rain_timeout)
    r.raise_for_status()
    payload = r.json()
    data = payload["properties"]["parameter"]
    dates = sorted(set().union(*[set(v.keys()) for v in data.values()]))
    rows = []
    for d in dates:
        row = {"date": pd.to_datetime(d, format="%Y%m%d", utc=True)}
        for p in params:
            row[p] = finite(data.get(p, {}).get(d))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def load_or_download_climate(cfg: Config, out_dir: Path) -> pd.DataFrame:
    cache = out_dir / "nasa_power_daily.csv"
    if cache.exists():
        try:
            x = pd.read_csv(cache, parse_dates=["date"])
            x["date"] = pd.to_datetime(x["date"], utc=True)
            if len(x) > 100:
                print(f"Using cached NASA POWER data: {cache}")
                return x
        except Exception:
            pass

    print("Downloading NASA POWER daily climate data...")
    params = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]
    try:
        climate = power_point(cfg, params)
    except Exception as exc:
        raise RuntimeError(f"NASA POWER download failed: {exc}") from exc
    climate.to_csv(cache, index=False)
    return climate


# ---------------------------------------------------------------------------
# Feature engineering and target construction
# ---------------------------------------------------------------------------

def rolling_climate_features(climate: pd.DataFrame) -> pd.DataFrame:
    c = climate.copy().sort_values("date").reset_index(drop=True)
    c["rain_mm"] = pd.to_numeric(c["PRECTOTCORR"], errors="coerce").clip(lower=0)
    c["t2m_c"] = pd.to_numeric(c["T2M"], errors="coerce")
    c["tmax_c"] = pd.to_numeric(c["T2M_MAX"], errors="coerce")
    c["tmin_c"] = pd.to_numeric(c["T2M_MIN"], errors="coerce")

    for w in [7, 15, 30, 60, 90]:
        c[f"rain_prev_{w}d"] = c["rain_mm"].rolling(w, min_periods=max(3, w // 2)).sum()
        c[f"rain_mean_{w}d"] = c["rain_mm"].rolling(w, min_periods=max(3, w // 2)).mean()
        c[f"tmean_prev_{w}d"] = c["t2m_c"].rolling(w, min_periods=max(3, w // 2)).mean()
        c[f"hot_days_{w}d"] = (c["tmax_c"] > 35).rolling(w, min_periods=max(3, w // 2)).sum()

    # Rolling anomaly of rainfall against the preceding available 365-day
    # climatology. Shift(1) ensures the current day's rainfall is not in the
    # baseline used for its own feature.
    daily_mean = c["rain_mm"].shift(1).rolling(365, min_periods=90).mean()
    daily_std = c["rain_mm"].shift(1).rolling(365, min_periods=90).std()
    c["rain_anom_z"] = (c["rain_mm"] - daily_mean) / daily_std.replace(0, np.nan)

    # Future 30-day rainfall is deliberately a TARGET, not a predictor.
    # Use min_periods=future_days so the target is defined only when the whole
    # future horizon exists.
    future_sum = c["rain_mm"].shift(-1).rolling(30, min_periods=30).sum().shift(-29)
    c["future_30d_rain_mm"] = future_sum

    # Causal continuous meteorological drought target.
    # At forecast date t, compare rainfall during t+1..t+30 with the
    # distribution of *already observed* 30-day rainfall from the preceding
    # year. This avoids using future/test-year climatology to define the target.
    historical_30 = c["rain_mm"].rolling(30, min_periods=30).sum()
    hist_mean = historical_30.shift(30).rolling(365, min_periods=180).mean()
    hist_std = historical_30.shift(30).rolling(365, min_periods=180).std()
    z = (future_sum - hist_mean) / hist_std.replace(0, np.nan)
    # Negative z = unusually dry. Map it smoothly to 0..1.
    c["future_drought_target"] = sigmoid(-z / 1.5)
    c.loc[future_sum.isna() | hist_mean.isna() | hist_std.isna(), "future_drought_target"] = np.nan
    c["future_rainfall_z"] = z

    # Fixed scientific threshold: target >= 0.60 means materially below the
    # recent historical rainfall distribution.
    c["future_drought_label"] = np.where(
        c["future_drought_target"].notna(),
        (c["future_drought_target"] >= 0.60).astype(int),
        np.nan,
    )
    return c


def nearest_asof_merge(sentinel: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    s = sentinel.copy().sort_values("datetime_utc")
    c = climate.copy().sort_values("date")
    c["datetime_utc"] = c["date"]
    merged = pd.merge_asof(
        s,
        c.drop(columns=["date"]),
        on="datetime_utc",
        direction="backward",
        tolerance=pd.Timedelta("3D"),
    )
    return merged


def build_sentinel_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Candidate Sentinel predictors. Prefer means, but include robust
    # distribution descriptors when available.
    feature_cols = []
    preferred_prefixes = [
        "blue_", "green_", "red_", "nir_", "swir1_", "swir2_",
        "ndvi_", "ndmi_", "ndwi_", "savi_", "evi_", "nbr_"
    ]
    for c in x.columns:
        if any(c.startswith(p) for p in preferred_prefixes):
            if c.endswith(("_mean", "_median", "_std", "_p10", "_p90")):
                feature_cols.append(c)

    # Scene-level metadata can be predictive of reliability but should not
    # directly become a target.
    for c in ["scene_cloud_cover_pct", "valid_fraction", "scl_cloud_fraction", "scl_cloud_shadow_fraction"]:
        if c in x.columns:
            feature_cols.append(c)

    # Derived interpretable vegetation/moisture composites.
    for name in ["ndvi_mean", "ndmi_mean", "ndwi_mean", "savi_mean", "evi_mean", "nbr_mean"]:
        if name in x.columns:
            x[name] = pd.to_numeric(x[name], errors="coerce")
    if "ndvi_mean" in x and "ndmi_mean" in x:
        x["vegetation_moisture_product"] = x["ndvi_mean"] * x["ndmi_mean"]
        feature_cols.append("vegetation_moisture_product")
    if "ndvi_mean" in x and "ndwi_mean" in x:
        x["vegetation_water_balance"] = 0.5 * x["ndvi_mean"] + 0.5 * x["ndwi_mean"]
        feature_cols.append("vegetation_water_balance")

    return x, sorted(set(feature_cols))


# ---------------------------------------------------------------------------
# Supervised learning
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame, min_test=12):
    x = df.dropna(subset=["future_drought_target"]).sort_values("datetime_utc").reset_index(drop=True)
    n = len(x)
    if n < 45:
        return None
    # 65/15/20 chronological split.
    n_train = max(30, int(n * 0.65))
    n_val = max(8, int(n * 0.15))
    if n_train + n_val + min_test > n:
        n_val = max(5, n - n_train - min_test)
    if n_train < 25 or n - n_train - n_val < min_test:
        return None
    return x.iloc[:n_train].copy(), x.iloc[n_train:n_train+n_val].copy(), x.iloc[n_train+n_val:].copy()


def model_library(seed=42):
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required. Install: python -m pip install scikit-learn")
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                random_state=seed, n_jobs=-1,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=3, max_features=0.8,
                random_state=seed, n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=2,
                min_samples_leaf=4, loss="huber", random_state=seed,
            )),
        ]),
    }


def regression_metrics(y, pred) -> Dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    if len(y) == 0:
        return {"n": 0}
    p, s = corr_pair(y, pred)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse(y, pred),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        "pearson_r": p,
        "spearman_rho": s,
        "bias_pred_minus_obs": float(np.mean(pred - y)),
    }


def classification_metrics(y_cont, pred_cont, threshold=0.60) -> Dict:
    y = np.asarray(y_cont, dtype=float)
    p = np.asarray(pred_cont, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"n": 0}
    yt = (y >= threshold).astype(int)
    pt = (p >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(yt, pt)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, pt)),
        "precision": float(precision_score(yt, pt, zero_division=0)),
        "recall": float(recall_score(yt, pt, zero_division=0)),
        "f1": float(f1_score(yt, pt, zero_division=0)),
        "confusion_matrix": confusion_matrix(yt, pt).tolist(),
    }
    if len(np.unique(yt)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(yt, p))
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


def cv_model_score(model, X, y, n_splits=4):
    n = len(X)
    if n < 30:
        return np.nan
    splits = min(n_splits, max(2, n // 12))
    tscv = TimeSeriesSplit(n_splits=splits)
    vals = []
    for tr, va in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        vals.append(mean_absolute_error(y.iloc[va], pred))
    return float(np.mean(vals)) if vals else np.nan


def uncertainty_from_forest(model, X):
    """Approximate ensemble spread for tree ensembles; not a formal CI."""
    try:
        pipe = model
        estimator = pipe.named_steps["model"]
        if not hasattr(estimator, "estimators_"):
            return np.full(len(X), np.nan)
        Xi = pipe.named_steps["imputer"].transform(X)
        vals = np.vstack([tree.predict(Xi) for tree in estimator.estimators_])
        return np.std(vals, axis=0)
    except Exception:
        return np.full(len(X), np.nan)


def train_and_evaluate(df: pd.DataFrame, feature_cols: List[str], cfg: Config, out_dir: Path):
    split = chronological_split(df)
    if split is None:
        return {
            "status": "insufficient_data",
            "message": (
                "Fewer than ~45 usable chronological observations. The current "
                "dataset is not large enough for a defensible train/validation/test ML study."
            ),
        }

    train, val, test = split
    Xtr, ytr = train[feature_cols], train["future_drought_target"]
    Xva, yva = val[feature_cols], val["future_drought_target"]
    Xte, yte = test[feature_cols], test["future_drought_target"]

    models = model_library(cfg.random_seed)
    selection = {}
    print("\nModel selection (validation is chronological; test is untouched):")
    for name, model in models.items():
        cv = cv_model_score(model, Xtr, ytr)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        selection[name] = {
            "cv_train_mae": cv,
            "validation": regression_metrics(yva, pred),
        }
        print(f"  {name:18s} CV-MAE={cv:.4f}  VAL-MAE={selection[name]['validation'].get('mae', np.nan):.4f}")

    best_name = min(
        selection,
        key=lambda n: (
            selection[n]["validation"].get("mae", np.inf)
            if np.isfinite(selection[n]["validation"].get("mae", np.nan)) else np.inf
        ),
    )

    # Retrain selected model on train + validation. Test remains unseen.
    best_model = model_library(cfg.random_seed)[best_name]
    trainval = pd.concat([train, val], ignore_index=True)
    best_model.fit(trainval[feature_cols], trainval["future_drought_target"])
    pred_test = best_model.predict(Xte)

    baseline_value = float(ytr.mean())
    baseline_pred = np.full(len(yte), baseline_value)

    test_metrics = regression_metrics(yte, pred_test)
    baseline_metrics = regression_metrics(yte, baseline_pred)
    class_metrics = classification_metrics(yte, pred_test)
    baseline_class = classification_metrics(yte, baseline_pred)

    uncertainty = uncertainty_from_forest(best_model, Xte)
    pred_table = test[["datetime_utc", "future_drought_target", "future_30d_rain_mm"]].copy()
    pred_table["prediction"] = pred_test
    pred_table["absolute_error"] = np.abs(pred_test - yte.to_numpy())
    pred_table["ensemble_spread"] = uncertainty
    pred_table["observed_drought"] = (pred_table["future_drought_target"] >= 0.60).astype(int)
    pred_table["predicted_drought"] = (pred_table["prediction"] >= 0.60).astype(int)
    pred_table.to_csv(out_dir / "heldout_test_predictions.csv", index=False)

    # Fit all models on train+validation for a fair held-out comparison.
    all_test_models = {}
    for name in models:
        m = model_library(cfg.random_seed)[name]
        m.fit(trainval[feature_cols], trainval["future_drought_target"])
        pp = m.predict(Xte)
        all_test_models[name] = regression_metrics(yte, pp)

    importances = {}
    try:
        est = best_model.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
            importances = dict(sorted(zip(feature_cols, vals), key=lambda z: -z[1]))
    except Exception:
        pass

    return {
        "status": "success",
        "n_total_target_rows": int(len(df.dropna(subset=["future_drought_target"]))),
        "train_n": int(len(train)),
        "validation_n": int(len(val)),
        "test_n": int(len(test)),
        "train_period": [str(train.datetime_utc.min()), str(train.datetime_utc.max())],
        "validation_period": [str(val.datetime_utc.min()), str(val.datetime_utc.max())],
        "test_period": [str(test.datetime_utc.min()), str(test.datetime_utc.max())],
        "best_model": best_name,
        "selection": selection,
        "test": test_metrics,
        "baseline_test": baseline_metrics,
        "test_classification": class_metrics,
        "baseline_classification": baseline_class,
        "all_model_test_metrics": all_test_models,
        "feature_importance": importances,
        "uncertainty_note": "Tree ensemble spread is an internal uncertainty proxy, not a calibrated confidence interval.",
        "feature_count": len(feature_cols),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(df, result, out_dir):
    if not HAS_MPL:
        return []
    paths = []
    x = df.dropna(subset=["future_drought_target"]).sort_values("datetime_utc")

    plt.figure(figsize=(13, 6))
    plt.plot(x["datetime_utc"], x["future_drought_target"], label="Future 30-day drought target")
    if "prediction" in result:
        pass
    plt.axhline(0.60, linestyle="--", label="Drought threshold")
    plt.xlabel("Observation date")
    plt.ylabel("Drought severity (0–1)")
    plt.title("AURORA future meteorological drought target")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = out_dir / "01_future_drought_target.png"
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(str(p))

    pred_file = out_dir / "heldout_test_predictions.csv"
    if pred_file.exists():
        pred = pd.read_csv(pred_file, parse_dates=["datetime_utc"])
        plt.figure(figsize=(13, 6))
        plt.plot(pred["datetime_utc"], pred["future_drought_target"], marker="o", label="Observed target")
        plt.plot(pred["datetime_utc"], pred["prediction"], marker="s", label="AURORA prediction")
        plt.axhline(0.60, linestyle="--", label="Drought threshold")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Held-out date")
        plt.ylabel("Drought severity")
        plt.title("Strict chronological held-out forecast")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = out_dir / "02_heldout_forecast.png"
        plt.savefig(p, dpi=180)
        plt.close()
        paths.append(str(p))

        plt.figure(figsize=(7, 7))
        plt.scatter(pred["future_drought_target"], pred["prediction"], alpha=0.8)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Observed target")
        plt.ylabel("Predicted target")
        plt.title("Held-out observed vs predicted drought")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "03_observed_vs_predicted.png"
        plt.savefig(p, dpi=180)
        plt.close()
        paths.append(str(p))

    if result.get("feature_importance"):
        imp = pd.Series(result["feature_importance"]).head(20).sort_values()
        plt.figure(figsize=(9, 8))
        imp.plot(kind="barh")
        plt.xlabel("Importance")
        plt.title("Top AURORA predictive features")
        plt.tight_layout()
        p = out_dir / "04_feature_importance.png"
        plt.savefig(p, dpi=180)
        plt.close()
        paths.append(str(p))

    if "rain_prev_30d" in df.columns:
        plt.figure(figsize=(13, 6))
        plt.plot(df["datetime_utc"], df["rain_prev_30d"], label="Previous 30-day rainfall")
        plt.plot(df["datetime_utc"], df["rain_prev_90d"], label="Previous 90-day rainfall")
        plt.xlabel("Observation date")
        plt.ylabel("Rainfall (mm)")
        plt.title("Antecedent rainfall used by AURORA")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = out_dir / "05_antecedent_rainfall.png"
        plt.savefig(p, dpi=180)
        plt.close()
        paths.append(str(p))

    return paths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(cfg, sentinel_meta, climate, df, feature_cols, result, plots, out_dir):
    target_n = int(df["future_drought_target"].notna().sum())
    years = sorted(df["datetime_utc"].dt.year.dropna().unique().tolist())
    report = []
    report.append("=" * 88)
    report.append("AURORA REAL-WORLD DROUGHT FORECASTING ENGINE")
    report.append("=" * 88)
    report.append("")
    report.append(f"Generated UTC: {pd.Timestamp.now(tz='UTC').isoformat()}")
    report.append(f"Sentinel years present: {years}")
    report.append(f"Sentinel observations: {len(df)}")
    report.append(f"Usable future-target observations: {target_n}")
    report.append(f"NASA POWER climate rows: {len(climate)}")
    report.append("")
    report.append("SCIENTIFIC DESIGN")
    report.append("-" * 88)
    report.append("Predictors: Sentinel-2 vegetation/moisture statistics + strictly antecedent climate features.")
    report.append("Target: severity of the following 30-day rainfall deficit relative to the preceding observed rainfall climatology.")
    report.append("Validation: chronological train/validation/test; the final test period is never used for model selection.")
    report.append("")
    report.append("CRITICAL INTERPRETATION")
    report.append("-" * 88)
    if len(years) < 3 or target_n < 45:
        report.append("WARNING: dataset is too small/short for a defensible real-world ML accuracy claim.")
        report.append("The numerical metrics are exploratory and should NOT be presented as field-ground-truth accuracy.")
    else:
        report.append("Dataset size is more suitable for a genuine temporal ML experiment, but external soil-moisture")
        report.append("or agricultural validation is still required before claiming field-level drought accuracy.")
    report.append("")

    if result.get("status") == "success":
        report.append("MODEL RESULTS")
        report.append("-" * 88)
        report.append(f"Best model selected on validation MAE: {result['best_model']}")
        report.append(f"Train: {result['train_period'][0]} -> {result['train_period'][1]} ({result['train_n']} rows)")
        report.append(f"Validation: {result['validation_period'][0]} -> {result['validation_period'][1]} ({result['validation_n']} rows)")
        report.append(f"TEST: {result['test_period'][0]} -> {result['test_period'][1]} ({result['test_n']} rows)")
        report.append("")
        for label, metrics in [("HELD-OUT TEST", result["test"]), ("CLIMATOLOGY BASELINE", result["baseline_test"])]:
            report.append(label)
            for k, v in metrics.items():
                report.append(f"  {k}: {v}")
            report.append("")
        report.append("HELD-OUT DROUGHT CLASSIFICATION")
        for k, v in result["test_classification"].items():
            report.append(f"  {k}: {v}")
        report.append("")
        report.append("MODEL COMPARISON ON THE SAME HELD-OUT TEST SET")
        for name, m in result["all_model_test_metrics"].items():
            report.append(f"  {name}: MAE={m.get('mae')}, RMSE={m.get('rmse')}, R2={m.get('r2')}, r={m.get('pearson_r')}")
        report.append("")
        if result.get("feature_importance"):
            report.append("TOP FEATURES")
            for k, v in list(result["feature_importance"].items())[:20]:
                report.append(f"  {k}: {v:.6g}")
    else:
        report.append("MODEL STATUS")
        report.append("-" * 88)
        report.append(result.get("message", "Model could not be trained."))
        report.append("")
        report.append("WHAT TO DO NEXT")
        report.append("1. Collect at least several years of Sentinel-2 observations.")
        report.append("2. Add multiple spatial cells/AOIs rather than a single scene mean.")
        report.append("3. Add an external soil-moisture product such as GLDAS/FLDAS/SMAP where access permits.")
        report.append("4. Validate on a completely unseen year/region.")

    report.append("")
    report.append("FEATURES")
    report.append("-" * 88)
    report.append(f"Feature count: {len(feature_cols)}")
    report.append(", ".join(feature_cols))
    report.append("")
    report.append("OUTPUTS")
    for p in plots:
        report.append(f"  {p}")

    (out_dir / "AURORA_REALWORLD_REPORT.txt").write_text("\n".join(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="AURORA real-world drought forecasting pipeline")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--bbox", nargs=4, type=float, default=[77.55, 12.90, 77.70, 13.05], metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    ap.add_argument("--input", default=None, help="Sentinel scene-statistics CSV")
    ap.add_argument("--output-dir", default="AURORA_REALWORLD_OUTPUT")
    ap.add_argument("--download-sentinel", action="store_true")
    ap.add_argument("--max-scenes", type=int, default=500)
    ap.add_argument("--cloud-max", type=float, default=60.0)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        start=args.start,
        end=args.end,
        bbox=tuple(args.bbox),
        input_csv=args.input,
        output_dir=args.output_dir,
        download_sentinel=args.download_sentinel,
        max_scenes=args.max_scenes,
        cloud_max=args.cloud_max,
    )
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    print("=" * 88)
    print("AURORA REAL-WORLD DROUGHT FORECASTING ENGINE")
    print("=" * 88)
    print("This pipeline refuses to call rainfall-derived proxy performance field truth.")
    print("")

    # 1. Sentinel data
    csvs = discover_csvs(cfg)
    sentinel_meta = {}
    if cfg.download_sentinel:
        downloaded = download_sentinel_stats(cfg, out_dir / "sentinel2_expanded.csv")
        if downloaded:
            csvs = [downloaded]
    if not csvs:
        raise FileNotFoundError(
            "No Sentinel CSV found. Either place your existing sentinel2 CSV beside this script "
            "or run with --download-sentinel."
        )

    print("Sentinel CSVs:")
    for p in csvs:
        print(f"  {p}")
    sentinel, sentinel_meta = combine_sentinel_csvs(csvs)

    # Filter requested date range, but retain enough climate context through
    # NASA POWER to construct rolling features.
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    sentinel = sentinel[(sentinel.datetime_utc >= start) & (sentinel.datetime_utc <= end)].copy()
    if len(sentinel) == 0:
        raise RuntimeError("No Sentinel observations remain inside the requested date range.")

    # Infer bbox from actual data when available.
    cfg.bbox = infer_bbox_from_df(sentinel, cfg.bbox)

    # 2. Climate data
    climate = load_or_download_climate(cfg, out_dir)
    climate = rolling_climate_features(climate)

    # 3. Merge climate state at each Sentinel observation.
    merged = nearest_asof_merge(sentinel, climate)
    merged, feature_cols = build_sentinel_features(merged)

    # Add causal climate predictors. Explicitly exclude all future variables
    # from feature_cols.
    causal_climate = [c for c in merged.columns if (
        c.startswith("rain_prev_") or c.startswith("rain_mean_") or
        c.startswith("tmean_prev_") or c.startswith("hot_days_") or c == "rain_anom_z"
    )]
    feature_cols = sorted(set(feature_cols + causal_climate))

    # Remove identifiers/non-numeric columns accidentally selected.
    feature_cols = [c for c in feature_cols if c in merged.columns and pd.api.types.is_numeric_dtype(merged[c])]

    # Save complete modelling table.
    merged.to_csv(out_dir / "AURORA_master_model_dataset.csv", index=False, float_format="%.10g")

    # Save a compact feature list.
    (out_dir / "feature_list.txt").write_text("\n".join(feature_cols), encoding="utf-8")

    # 4. Train/evaluate.
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required. Install with: python -m pip install scikit-learn")
    result = train_and_evaluate(merged, feature_cols, cfg, out_dir)

    # 5. Plots and reports.
    plots = make_plots(merged, result, out_dir)
    write_report(cfg, sentinel_meta, climate, merged, feature_cols, result, plots, out_dir)

    # 6. Machine-readable summary.
    summary = {
        "config": asdict(cfg),
        "sentinel": sentinel_meta,
        "dataset": {
            "sentinel_observations": len(sentinel),
            "merged_observations": len(merged),
            "target_observations": int(merged.future_drought_target.notna().sum()),
            "date_min": str(merged.datetime_utc.min()),
            "date_max": str(merged.datetime_utc.max()),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "result": result,
        "scientific_warning": (
            "The future rainfall target is an externally computed meteorological proxy, not agricultural field truth. "
            "A soil-moisture or crop-impact target is required for a stronger real-world drought claim."
        ),
        "plots": plots,
    }
    (out_dir / "AURORA_metrics.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("AURORA COMPLETE")
    print("=" * 88)
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Sentinel observations: {len(sentinel)}")
    print(f"Usable future-target observations: {int(merged.future_drought_target.notna().sum())}")
    if result.get("status") == "success":
        print(f"Best model: {result['best_model']}")
        print(f"Held-out TEST MAE: {result['test']['mae']:.4f}")
        print(f"Held-out TEST RMSE: {result['test']['rmse']:.4f}")
        print(f"Held-out TEST R2: {result['test']['r2']:.4f}")
        print(f"Held-out TEST Pearson r: {result['test']['pearson_r']:.4f}")
        print(f"Held-out TEST F1: {result['test_classification']['f1']:.4f}")
    else:
        print("ML status: INSUFFICIENT DATA")
        print(result.get("message", ""))
    print("\nKey files:")
    for name in [
        "AURORA_master_model_dataset.csv",
        "heldout_test_predictions.csv",
        "AURORA_metrics.json",
        "AURORA_REALWORLD_REPORT.txt",
    ]:
        p = out_dir / name
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
