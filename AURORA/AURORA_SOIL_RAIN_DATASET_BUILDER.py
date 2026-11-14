r"""
AURORA SOIL + RAIN DATASET BUILDER AND FORECASTING PIPELINE
===========================================================

Purpose
-------
Build a new, reproducible AURORA dataset combining:

1. Existing Sentinel-2 scene-statistics CSV.
2. Historical rainfall and meteorological variables.
3. Independent ERA5-Land soil-moisture reanalysis variables.
4. Causal lagged/rolling predictors.
5. Future 30-day rainfall-deficit and soil-moisture drought targets.
6. Chronological machine-learning evaluation.

This version intentionally uses normal, explicit imports. It does not use
optional-import try/except blocks, so VS Code/Pylance will not underline
imports merely because an optional package is absent.

Data provider
-------------
Open-Meteo Historical Weather API, using ERA5-Land where available.
The API provides historical hourly precipitation, temperature, ET0, VPD,
and soil moisture at several depth ranges. Soil moisture is returned in
m^3/m^3. Rainfall is returned in mm.

API documentation:
https://open-meteo.com/en/docs/historical-weather-api

Scientific qualification
------------------------
ERA5-Land is a reanalysis/land-surface model reference, not a field sensor.
It is independent of Sentinel-2 imagery, but it is spatially coarse relative
to individual farms and should be described as an independent reanalysis
reference rather than absolute ground truth.

Windows PowerShell usage
------------------------
Install dependencies once:

    py -m pip install numpy pandas requests scipy scikit-learn matplotlib

Run the complete pipeline:

    py .\\AURORA_SOIL_RAIN_DATASET_BUILDER.py `
      --sentinel-file ".\AURORA_REALWORLD_OUTPUT\sentinel2_expanded.csv" `
      --output-dir ".\AURORA_SOIL_RAIN_OUTPUT" `
      --lat 12.9716 `
      --lon 77.5946 `
      --download-weather `
      --overwrite

PowerShell uses the backtick (`) for line continuation. You can also put
all arguments on one line to avoid continuation problems.

If your Sentinel file has a different path:

    py .\AURORA_SOIL_RAIN_DATASET_BUILDER.py --sentinel-file "G:\AURORA\AURORA_REALWORLD_OUTPUT\sentinel2_expanded.csv" --output-dir "G:\AURORA\AURORA_SOIL_RAIN_OUTPUT" --lat 12.9716 --lon 77.5946 --download-weather --overwrite

Outputs
-------
AURORA_SOIL_RAIN_OUTPUT/
    sentinel_input_normalized.csv
    era5_land_hourly_raw.csv
    era5_land_daily_reference.csv
    AURORA_soil_rain_master_dataset.csv
    AURORA_soil_rain_usable_dataset.csv
    future_rain_soil_targets.csv
    heldout_test_predictions_soil.csv
    AURORA_soil_rain_metrics.json
    AURORA_SOIL_RAIN_REPORT.txt
    feature_list.txt
    target_timeseries.png
    prediction_scatter.png
    absolute_error.png
r"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


API_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_LAT = 12.9716
DEFAULT_LON = 77.5946
DEFAULT_OUTPUT = "AURORA_SOIL_RAIN_OUTPUT"
HORIZON_DAYS = 30

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]

TIMESTAMP_CANDIDATES = ["datetime_utc", "timestamp", "datetime", "date", "time", "acquisition_time"]


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "AURORA_soil_rain_pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build an AURORA Sentinel + rainfall + ERA5-Land soil-moisture dataset.",
    )
    parser.add_argument("--sentinel-file", type=str, default=None, help="Existing Sentinel scene-statistics CSV.")
    parser.add_argument("--sentinel-dir", type=str, default=None, help="Directory searched for Sentinel CSV files.")
    parser.add_argument("--weather-file", type=str, default=None, help="Previously downloaded Open-Meteo hourly CSV.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD start date.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional YYYY-MM-DD end date.")
    parser.add_argument("--download-weather", action="store_true", help="Download historical ERA5-Land data.")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_timestamp_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    for col in df.columns:
        text = str(col).lower()
        if "time" in text or "date" in text:
            return col
    raise ValueError("No timestamp column found. Expected datetime_utc, timestamp, datetime, date, or time.")


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    result = pd.to_datetime(series, errors="coerce", utc=True)
    if result.isna().all():
        result = pd.to_datetime(series.astype(str), errors="coerce", format="mixed", utc=True)
    return result


def discover_sentinel_file(args: argparse.Namespace, output_dir: Path) -> Path:
    candidates: List[Path] = []
    if args.sentinel_file:
        candidates.append(Path(args.sentinel_file))
    if args.sentinel_dir:
        candidates.extend(sorted(Path(args.sentinel_dir).glob("*.csv")))
    candidates.extend([
        Path("AURORA_REALWORLD_OUTPUT/sentinel2_expanded.csv"),
        Path("AURORA_REALWORLD_OUTPUT/AURORA_master_model_dataset.csv"),
        Path("sentinel2_expanded.csv"),
        Path("sentinel2_bengaluru_2024.csv"),
    ])
    seen = set()
    for path in candidates:
        path = path.expanduser()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.exists() and path.is_file():
            logging.info("Using Sentinel CSV: %s", path)
            return path
    raise FileNotFoundError(
        "Could not find a Sentinel CSV. Pass --sentinel-file \"G:\\AURORA\\...\\sentinel2_expanded.csv\"."
    )


def load_sentinel(path: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Sentinel CSV is empty: {path}")
    timestamp_col = find_timestamp_column(df)
    df = df.copy()
    df["datetime_utc"] = normalize_timestamp_series(df[timestamp_col])
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    df = df.drop_duplicates(subset=["datetime_utc"], keep="last").reset_index(drop=True)
    numeric_cols = []
    for col in df.columns:
        if col == "datetime_utc":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(0.20 * len(df))):
            df[col] = converted
            numeric_cols.append(col)
    df = df[["datetime_utc"] + numeric_cols]
    df.to_csv(output_dir / "sentinel_input_normalized.csv", index=False)
    logging.info("Sentinel observations: %d", len(df))
    logging.info("Numeric Sentinel columns retained: %d", len(numeric_cols))
    return df


def determine_date_range(sentinel: pd.DataFrame, args: argparse.Namespace) -> Tuple[str, str]:
    start = sentinel["datetime_utc"].min().normalize() - pd.Timedelta(days=120)
    end = sentinel["datetime_utc"].max().normalize() + pd.Timedelta(days=args.horizon_days + 2)
    if args.start_date:
        start = pd.Timestamp(args.start_date, tz="UTC").normalize()
    if args.end_date:
        end = pd.Timestamp(args.end_date, tz="UTC").normalize()
    if end <= start:
        raise ValueError("End date must be after start date.")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def download_open_meteo(start_date: str, end_date: str, lat: float, lon: float, output_dir: Path, overwrite: bool) -> pd.DataFrame:
    raw_path = output_dir / "era5_land_hourly_raw.csv"
    if raw_path.exists() and not overwrite:
        logging.info("Using cached weather file: %s", raw_path)
        return pd.read_csv(raw_path, parse_dates=["datetime_utc"])

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "models": "era5_land",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "kmh",
    }
    logging.info("Downloading ERA5-Land historical data from %s to %s", start_date, end_date)
    logging.info("This may take a little time; the response contains hourly data.")
    response = requests.get(API_URL, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo returned an error: {payload.get('reason', payload)}")
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError("Open-Meteo response did not contain hourly data.")

    hourly = payload["hourly"]
    frame = pd.DataFrame(hourly)
    frame = frame.rename(columns={"time": "datetime_utc"})
    frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True)
    frame["latitude_requested"] = lat
    frame["longitude_requested"] = lon
    frame["latitude_grid"] = payload.get("latitude")
    frame["longitude_grid"] = payload.get("longitude")
    frame["elevation_m"] = payload.get("elevation")
    frame["weather_model"] = payload.get("generationtime_ms", np.nan)
    frame.to_csv(raw_path, index=False)
    logging.info("Saved hourly weather/soil reference: %s", raw_path)
    return frame


def load_weather_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime_utc" not in df.columns:
        col = find_timestamp_column(df)
        df = df.rename(columns={col: "datetime_utc"})
    df["datetime_utc"] = normalize_timestamp_series(df["datetime_utc"])
    return df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")


def weighted_root_zone(surface: pd.Series, mid: pd.Series, deep: pd.Series) -> pd.Series:
    # Approximate depth-weighted average across 0-7, 7-28, and 28-100 cm.
    # The weights correspond to layer thicknesses 7, 21, and 72 cm.
    return (7.0 * surface + 21.0 * mid + 72.0 * deep) / 100.0


def aggregate_weather_daily(hourly: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    df = hourly.copy()
    df["datetime_utc"] = normalize_timestamp_series(df["datetime_utc"])
    df = df.dropna(subset=["datetime_utc"]).set_index("datetime_utc").sort_index()

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    daily = df[numeric_cols].resample("1D").agg("mean")

    if "precipitation" in df.columns:
        daily["rain_mm"] = df["precipitation"].resample("1D").sum(min_count=1)
    elif "rain" in df.columns:
        daily["rain_mm"] = df["rain"].resample("1D").sum(min_count=1)
    else:
        raise ValueError("Weather file contains neither precipitation nor rain.")

    if "rain" in df.columns:
        daily["liquid_rain_mm"] = df["rain"].resample("1D").sum(min_count=1)
    else:
        daily["liquid_rain_mm"] = daily["rain_mm"]

    rename = {
        "temperature_2m": "temperature_2m_c",
        "relative_humidity_2m": "relative_humidity_pct",
        "et0_fao_evapotranspiration": "et0_mm",
        "vapour_pressure_deficit": "vpd_kpa",
        "soil_moisture_0_to_7cm": "sm_0_7_m3m3",
        "soil_moisture_7_to_28cm": "sm_7_28_m3m3",
        "soil_moisture_28_to_100cm": "sm_28_100_m3m3",
        "soil_moisture_100_to_255cm": "sm_100_255_m3m3",
    }
    daily = daily.rename(columns=rename).reset_index()

    required = ["sm_0_7_m3m3", "sm_7_28_m3m3", "sm_28_100_m3m3"]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise ValueError(f"Missing soil-moisture columns from weather reference: {missing}")

    daily["sm_surface_m3m3"] = daily["sm_0_7_m3m3"]
    daily["sm_root_zone_m3m3"] = weighted_root_zone(
        daily["sm_0_7_m3m3"], daily["sm_7_28_m3m3"], daily["sm_28_100_m3m3"]
    )
    profile_parts = [c for c in ["sm_0_7_m3m3", "sm_7_28_m3m3", "sm_28_100_m3m3", "sm_100_255_m3m3"] if c in daily.columns]
    daily["sm_profile_mean_m3m3"] = daily[profile_parts].mean(axis=1)

    daily = daily.replace([np.inf, -np.inf], np.nan).sort_values("datetime_utc")
    daily.to_csv(output_dir / "era5_land_daily_reference.csv", index=False)
    logging.info("Daily independent reference rows: %d", len(daily))
    return daily


def add_causal_daily_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy().sort_values("datetime_utc").reset_index(drop=True)
    out = out.set_index("datetime_utc")

    base_columns = [
        "rain_mm", "liquid_rain_mm", "temperature_2m_c", "relative_humidity_pct",
        "et0_mm", "vpd_kpa", "sm_surface_m3m3", "sm_root_zone_m3m3", "sm_profile_mean_m3m3",
    ]
    base_columns = [c for c in base_columns if c in out.columns]

    for col in base_columns:
        for lag in [1, 3, 7, 14, 30, 60, 90]:
            out[f"{col}_lag_{lag}d"] = out[col].shift(lag)
        for window in [7, 15, 30, 60, 90]:
            shifted = out[col].shift(1)
            out[f"{col}_past_{window}d_mean"] = shifted.rolling(window, min_periods=max(3, window // 3)).mean()
            out[f"{col}_past_{window}d_std"] = shifted.rolling(window, min_periods=max(3, window // 3)).std()
            out[f"{col}_past_{window}d_sum"] = shifted.rolling(window, min_periods=max(3, window // 3)).sum()
        out[f"{col}_change_7d"] = out[col].shift(1) - out[col].shift(8)
        out[f"{col}_change_30d"] = out[col].shift(1) - out[col].shift(31)

    # Calendar variables are known at prediction time and do not leak future values.
    out["day_of_year_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12.0)

    out = out.reset_index()
    return out


def build_future_targets(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = daily.copy().sort_values("datetime_utc").set_index("datetime_utc")
    target = pd.DataFrame(index=df.index)

    # Future window begins after the observation day, preventing same-day leakage.
    for col, prefix in [
        ("rain_mm", "rain"),
        ("sm_surface_m3m3", "surface_soil"),
        ("sm_root_zone_m3m3", "root_zone_soil"),
        ("sm_profile_mean_m3m3", "profile_soil"),
    ]:
        if col not in df.columns:
            continue
        future_values = df[col].shift(-1).rolling(horizon, min_periods=horizon).mean().shift(-(horizon - 1))
        past_values = df[col].shift(1).rolling(30, min_periods=20).mean()
        past_std = df[col].shift(1).rolling(30, min_periods=20).std()
        target[f"future_30d_{prefix}_mean"] = future_values
        target[f"past_30d_{prefix}_mean"] = past_values
        target[f"past_30d_{prefix}_std"] = past_std
        target[f"future_30d_{prefix}_change"] = future_values - past_values
        target[f"future_30d_{prefix}_z"] = (future_values - past_values) / past_std.replace(0, np.nan)

    # Rainfall deficit: lower future rainfall than recent rainfall = higher drought.
    if "future_30d_rain_mean" in target.columns:
        rain_z = target["future_30d_rain_z"]
        target["future_rain_drought_severity"] = 1.0 / (1.0 + np.exp(np.clip(rain_z, -30, 30)))

    # Soil moisture drought: lower future moisture than recent moisture = higher drought.
    for prefix in ["surface_soil", "root_zone_soil", "profile_soil"]:
        zcol = f"future_30d_{prefix}_z"
        if zcol in target.columns:
            target[f"future_{prefix}_drought_severity"] = 1.0 / (1.0 + np.exp(np.clip(target[zcol], -30, 30)))

    return target.reset_index()


def merge_sentinel_with_daily(sentinel: pd.DataFrame, daily_features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    left = sentinel.sort_values("datetime_utc").copy()
    right = daily_features.sort_values("datetime_utc").copy()
    merged = pd.merge_asof(left, right, on="datetime_utc", direction="backward", tolerance=pd.Timedelta(days=2))

    # Daily target rows are anchored at midnight UTC, while Sentinel scene timestamps
    # are typically a few hours after midnight. A same-day scene should still match the
    # daily target for that date, so use a 1-day tolerance instead of requiring an exact
    # timestamp match.
    merged = pd.merge_asof(
        merged.sort_values("datetime_utc"),
        targets.sort_values("datetime_utc"),
        on="datetime_utc",
        direction="backward",
        tolerance=pd.Timedelta(days=1),
    )
    return merged.sort_values("datetime_utc").reset_index(drop=True)


def identify_predictors(df: pd.DataFrame, target_columns: List[str]) -> List[str]:
    excluded_exact = {
        "datetime_utc", "future_30d_rain_mean", "future_30d_surface_soil_mean",
        "future_30d_root_zone_soil_mean", "future_30d_profile_soil_mean",
    }
    predictors = []
    for col in df.columns:
        if col in excluded_exact or col in target_columns:
            continue
        if col.startswith("future_") or col.startswith("past_30d_"):
            continue
        if col in {"weather_model", "latitude_requested", "longitude_requested", "latitude_grid", "longitude_grid", "elevation_m"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().sum() >= 10 and df[col].nunique(dropna=True) > 1:
                predictors.append(col)
    return predictors


def safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    try:
        if method == "pearson":
            return float(pearsonr(a[mask], b[mask]).statistic)
        return float(spearmanr(a[mask], b[mask]).statistic)
    except Exception:
        return float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "pearson_r": safe_corr(y_true, y_pred, "pearson"),
        "spearman_r": safe_corr(y_true, y_pred, "spearman"),
        "bias_pred_minus_observed": float(np.mean(y_pred - y_true)),
        "observed_mean": float(np.mean(y_true)),
        "predicted_mean": float(np.mean(y_pred)),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, object]:
    observed = (y_true >= threshold).astype(int)
    predicted = (y_pred >= threshold).astype(int)
    result: Dict[str, object] = {
        "threshold": float(threshold),
        "positive_rate_observed": float(observed.mean()),
        "positive_rate_predicted": float(predicted.mean()),
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "precision": float(precision_score(observed, predicted, zero_division=0)),
        "recall": float(recall_score(observed, predicted, zero_division=0)),
        "f1": float(f1_score(observed, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(observed, predicted, labels=[0, 1]).tolist(),
    }
    if len(np.unique(observed)) == 2:
        result["roc_auc"] = float(roc_auc_score(observed, y_pred))
    else:
        result["roc_auc"] = float("nan")
    return result


def model_candidates() -> Dict[str, object]:
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, loss="huber", random_state=42)),
        ]),
    }


def chronological_splits(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 30:
        raise ValueError(f"Only {n} usable rows are available; at least 30 are needed.")
    train_end = max(1, int(n * 0.60))
    val_end = max(train_end + 1, int(n * 0.80))
    if val_end >= n:
        val_end = n - 1
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    return train_idx, val_idx, test_idx


def train_and_evaluate(dataset: pd.DataFrame, predictors: List[str], target_col: str, output_dir: Path) -> Dict[str, object]:
    data = dataset.dropna(subset=[target_col]).sort_values("datetime_utc").reset_index(drop=True)
    if len(data) < 30:
        raise ValueError(f"Not enough target rows for {target_col}: {len(data)}")

    X = data[predictors].replace([np.inf, -np.inf], np.nan)
    y = data[target_col].astype(float).to_numpy()
    train_idx, val_idx, test_idx = chronological_splits(len(data))

    models = model_candidates()
    selection = {}
    for name, model in models.items():
        model.fit(X.iloc[train_idx], y[train_idx])
        val_pred = model.predict(X.iloc[val_idx])
        selection[name] = {
            "validation_mae": float(mean_absolute_error(y[val_idx], val_pred)),
            "validation_rmse": float(math.sqrt(mean_squared_error(y[val_idx], val_pred))),
        }
        logging.info("%-18s validation MAE = %.5f", name, selection[name]["validation_mae"])

    best_name = min(selection, key=lambda name: selection[name]["validation_mae"])
    best_model = models[best_name]
    fit_idx = np.concatenate([train_idx, val_idx])
    best_model.fit(X.iloc[fit_idx], y[fit_idx])
    test_pred = best_model.predict(X.iloc[test_idx])

    threshold = float(np.quantile(y[fit_idx], 0.75))
    metrics = regression_metrics(y[test_idx], test_pred)
    metrics.update({f"binary_{k}": v for k, v in binary_metrics(y[test_idx], test_pred, threshold).items()})
    metrics["target_column"] = target_col
    metrics["best_model"] = best_name
    metrics["predictor_count"] = len(predictors)
    metrics["train_rows"] = int(len(train_idx))
    metrics["validation_rows"] = int(len(val_idx))
    metrics["test_rows"] = int(len(test_idx))
    metrics["model_selection"] = selection

    predictions = data.iloc[test_idx][["datetime_utc", target_col]].copy()
    predictions = predictions.rename(columns={target_col: "observed_target"})
    predictions["prediction"] = test_pred
    predictions["absolute_error"] = np.abs(predictions["observed_target"] - predictions["prediction"])
    predictions["observed_drought"] = (predictions["observed_target"] >= threshold).astype(int)
    predictions["predicted_drought"] = (predictions["prediction"] >= threshold).astype(int)
    predictions.to_csv(output_dir / f"heldout_test_predictions_{target_col}.csv", index=False)

    return {"metrics": metrics, "predictions": predictions, "best_model": best_model}


def make_plots(predictions: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    if predictions.empty:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(predictions["datetime_utc"], predictions["observed_target"], marker="o", label="Observed")
    plt.plot(predictions["datetime_utc"], predictions["prediction"], marker="x", label="Predicted")
    plt.title(f"Held-out predictions: {target_col}")
    plt.xlabel("Date")
    plt.ylabel("Drought severity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"target_timeseries_{target_col}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(predictions["observed_target"], predictions["prediction"], alpha=0.75)
    lo = min(predictions["observed_target"].min(), predictions["prediction"].min())
    hi = max(predictions["observed_target"].max(), predictions["prediction"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(f"Observed vs predicted: {target_col}")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"prediction_scatter_{target_col}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(predictions["datetime_utc"].astype(str), predictions["absolute_error"])
    plt.xticks(rotation=70, ha="right")
    plt.title(f"Held-out absolute error: {target_col}")
    plt.ylabel("Absolute error")
    plt.tight_layout()
    plt.savefig(output_dir / f"absolute_error_{target_col}.png", dpi=180)
    plt.close()


def write_report(output_dir: Path, sentinel: pd.DataFrame, daily: pd.DataFrame, master: pd.DataFrame, metrics: Dict[str, object], predictors: List[str], args: argparse.Namespace) -> None:
    report = []
    report.append("AURORA SOIL + RAIN DATASET REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("Study configuration")
    report.append(f"Latitude: {args.lat}")
    report.append(f"Longitude: {args.lon}")
    report.append(f"Target horizon: {args.horizon_days} days")
    report.append("Reference: Open-Meteo historical ERA5-Land reanalysis")
    report.append("")
    report.append("Dataset sizes")
    report.append(f"Sentinel rows: {len(sentinel)}")
    report.append(f"Daily reference rows: {len(daily)}")
    report.append(f"Merged master rows: {len(master)}")
    report.append(f"Predictor count: {len(predictors)}")
    report.append("")
    report.append("Scientific interpretation")
    report.append("The soil-moisture target is an independent ERA5-Land reanalysis reference.")
    report.append("It is not equivalent to field-sensor ground truth and may be spatially coarse.")
    report.append("All future target variables are excluded from the predictor matrix.")
    report.append("The evaluation uses chronological train/validation/test partitions.")
    report.append("")
    report.append("Model results")
    for target, result in metrics.items():
        report.append("")
        report.append(f"Target: {target}")
        for key, value in result["metrics"].items():
            report.append(f"{key}: {value}")
    report.append("")
    report.append("Limitations")
    report.append("1. ERA5-Land is a model-based reanalysis, not direct local sampling.")
    report.append("2. Sentinel scene averages may hide local spatial variability.")
    report.append("3. Temporal observations can be autocorrelated; row count is not event count.")
    report.append("4. Irrigation, land cover, soil texture, runoff, and groundwater are not fully observed.")
    report.append("5. Results should be reported as predictive skill against an independent reanalysis reference.")
    (output_dir / "AURORA_SOIL_RAIN_REPORT.txt").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "feature_list.txt").write_text("\n".join(predictors), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    setup_logging(output_dir)
    logging.info("=" * 80)
    logging.info("AURORA SOIL + RAIN DATASET BUILDER")
    logging.info("=" * 80)

    sentinel_path = discover_sentinel_file(args, output_dir)
    sentinel = load_sentinel(sentinel_path, output_dir)
    start_date, end_date = determine_date_range(sentinel, args)

    weather_file = Path(args.weather_file).expanduser() if args.weather_file else output_dir / "era5_land_hourly_raw.csv"
    if weather_file.exists() and not args.download_weather:
        logging.info("Loading existing weather file: %s", weather_file)
        hourly = load_weather_file(weather_file)
    else:
        hourly = download_open_meteo(start_date, end_date, args.lat, args.lon, output_dir, args.overwrite)

    daily_reference = aggregate_weather_daily(hourly, output_dir)
    daily_features = add_causal_daily_features(daily_reference, args.horizon_days)
    targets = build_future_targets(daily_reference, args.horizon_days)
    targets.to_csv(output_dir / "future_rain_soil_targets.csv", index=False)

    master = merge_sentinel_with_daily(sentinel, daily_features, targets)
    master.to_csv(output_dir / "AURORA_soil_rain_master_dataset.csv", index=False)

    target_columns = [
        "future_rain_drought_severity",
        "future_surface_soil_drought_severity",
        "future_root_zone_soil_drought_severity",
        "future_profile_soil_drought_severity",
    ]
    available_targets = [c for c in target_columns if c in master.columns and master[c].notna().sum() >= 30]
    if not available_targets:
        raise RuntimeError("No usable future targets were created. Check the date range and downloaded data.")

    predictors = identify_predictors(master, target_columns)
    if not predictors:
        raise RuntimeError("No numeric predictors were found after leakage filtering.")

    usable_mask = master[available_targets].notna().any(axis=1)
    usable = master.loc[usable_mask].copy()
    usable.to_csv(output_dir / "AURORA_soil_rain_usable_dataset.csv", index=False)
    logging.info("Usable merged rows: %d", len(usable))
    logging.info("Predictors: %d", len(predictors))

    all_results: Dict[str, object] = {}
    for target in available_targets:
        logging.info("")
        logging.info("Training target: %s", target)
        result = train_and_evaluate(usable, predictors, target, output_dir)
        all_results[target] = {"metrics": result["metrics"]}
        make_plots(result["predictions"], target, output_dir)
        logging.info("Best model: %s", result["metrics"]["best_model"])
        logging.info("Test MAE: %.5f", result["metrics"]["mae"])
        logging.info("Test RMSE: %.5f", result["metrics"]["rmse"])
        logging.info("Test R2: %.5f", result["metrics"]["r2"])
        logging.info("Test Pearson r: %.5f", result["metrics"]["pearson_r"])
        logging.info("Test F1: %.5f", result["metrics"]["binary_f1"])

    metrics_payload = {
        "configuration": vars(args),
        "sentinel_file": str(sentinel_path),
        "date_range": {"start": start_date, "end": end_date},
        "sentinel_rows": len(sentinel),
        "daily_reference_rows": len(daily_reference),
        "master_rows": len(master),
        "usable_rows": len(usable),
        "predictors": predictors,
        "results": all_results,
    }
    (output_dir / "AURORA_soil_rain_metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")
    write_report(output_dir, sentinel, daily_reference, master, {k: {"metrics": v["metrics"]} for k, v in all_results.items()}, predictors, args)

    logging.info("")
    logging.info("=" * 80)
    logging.info("AURORA COMPLETE")
    logging.info("=" * 80)
    logging.info("Output directory: %s", output_dir)
    logging.info("Sentinel rows: %d", len(sentinel))
    logging.info("Usable rows: %d", len(usable))
    logging.info("Targets evaluated: %s", ", ".join(available_targets))
    logging.info("Key dataset: %s", output_dir / "AURORA_soil_rain_master_dataset.csv")
    logging.info("Key metrics: %s", output_dir / "AURORA_soil_rain_metrics.json")


if __name__ == "__main__":
    main()
