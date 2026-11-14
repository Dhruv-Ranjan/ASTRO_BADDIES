"""
AURORA V4 — SOIL-MOISTURE-ONLY FORECASTING PIPELINE
===================================================

Purpose
-------
Train and evaluate AURORA models using the usable Sentinel-2 + soil-moisture
dataset. Rainfall columns are ignored because the supplied dataset contains
rainfall fields that are entirely missing.

The script:
1. Loads the CSV.
2. Removes all-empty and constant columns.
3. Prevents future-target leakage.
4. Uses chronological train/validation/test splits.
5. Trains Ridge, Random Forest, Extra Trees, and Gradient Boosting models.
6. Predicts future 30-day soil moisture:
      - surface soil moisture
      - root-zone soil moisture
      - profile-average soil moisture
7. Saves predictions, metrics, plots, feature importance, and a report.

PowerShell example
------------------
py .\AURORA_SOILMOISTURE_FINAL.py `
  --input-file "G:\AURORA\AURORA_soil_rain_usable_dataset.csv" `
  --output-dir "G:\AURORA\AURORA_SOILMOISTURE_FINAL_OUTPUT"

One-line PowerShell example
---------------------------
py .\AURORA_SOILMOISTURE_FINAL.py --input-file "G:\AURORA\AURORA_soil_rain_usable_dataset.csv" --output-dir "G:\AURORA\AURORA_SOILMOISTURE_FINAL_OUTPUT"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("AURORA")


TARGETS = {
    "surface": "future_30d_surface_soil_mean",
    "root_zone": "future_30d_root_zone_soil_mean",
    "profile": "future_30d_profile_soil_mean",
}

TARGET_SEVERITY_COLUMNS = {
    "future_surface_soil_drought_severity",
    "future_root_zone_soil_drought_severity",
    "future_profile_soil_drought_severity",
}

RAIN_COLUMNS = {
    c for c in [
        "rain_mm",
        "liquid_rain_mm",
        "future_30d_rain_mean",
        "future_30d_rain_change",
        "future_30d_rain_z",
        "future_rain_drought_severity",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AURORA soil-moisture-only forecasting pipeline."
    )
    parser.add_argument("--input-file", required=True, help="Input CSV file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
        help="Chronological training fraction. Default: 0.60",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Chronological validation fraction. Default: 0.20",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
    )


def safe_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return None


def load_dataset(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    df = pd.read_csv(input_file)

    if "datetime_utc" not in df.columns:
        raise ValueError("The CSV must contain a 'datetime_utc' column.")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.dropna(subset=["datetime_utc"]).copy()
    df = df.sort_values("datetime_utc").drop_duplicates(
        subset=["datetime_utc"], keep="last"
    )
    df = df.reset_index(drop=True)

    LOGGER.info("Loaded %d rows and %d columns.", df.shape[0], df.shape[1])
    return df


def remove_invalid_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    removed_all_missing = []
    removed_constant = []

    for column in list(df.columns):
        if column == "datetime_utc":
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")

        if numeric.notna().sum() == 0:
            removed_all_missing.append(column)
            df = df.drop(columns=[column])
            continue

        if numeric.nunique(dropna=True) <= 1:
            removed_constant.append(column)
            df = df.drop(columns=[column])

    LOGGER.info(
        "Removed %d all-missing columns and %d constant columns.",
        len(removed_all_missing),
        len(removed_constant),
    )

    return df, removed_all_missing, removed_constant


def build_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    y = pd.to_numeric(df[target_column], errors="coerce")

    # Any column with "future" is excluded to prevent target leakage.
    # Rainfall columns are excluded because rainfall is missing in this dataset.
    excluded = {
        "datetime_utc",
        target_column,
        *TARGET_SEVERITY_COLUMNS,
        *RAIN_COLUMNS,
    }

    excluded.update(
        column
        for column in df.columns
        if column.startswith("future_")
    )

    feature_columns = []
    for column in df.columns:
        if column in excluded:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() > 0:
            feature_columns.append(column)

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    # Remove columns that became empty after numeric conversion.
    valid_columns = [
        column for column in X.columns
        if X[column].notna().sum() > 0 and X[column].nunique(dropna=True) > 1
    ]
    X = X[valid_columns]

    valid_rows = y.notna()
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    LOGGER.info(
        "Target %-12s | rows=%d | features=%d",
        target_column,
        len(y),
        X.shape[1],
    )

    return X, y, list(X.columns)


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_fraction: float,
    validation_fraction: float,
):
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")

    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train fraction + validation fraction must be below 1.")

    n = len(X)
    train_end = max(1, int(n * train_fraction))
    validation_end = max(train_end + 1, int(n * (train_fraction + validation_fraction)))

    if validation_end >= n:
        validation_end = n - 1

    X_train = X.iloc[:train_end].copy()
    X_val = X.iloc[train_end:validation_end].copy()
    X_test = X.iloc[validation_end:].copy()

    y_train = y.iloc[:train_end].copy()
    y_val = y.iloc[train_end:validation_end].copy()
    y_test = y.iloc[validation_end:].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_models(random_state: int) -> dict[str, object]:
    return {
        "Ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=3,
                        max_features=0.70,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=3,
                        max_features=0.70,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.035,
                        max_depth=2,
                        min_samples_leaf=5,
                        loss="huber",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    else:
        pearson_r, pearson_p = np.nan, np.nan

    return {
        "MAE": safe_float(mae),
        "RMSE": safe_float(rmse),
        "R2": safe_float(r2),
        "Pearson_r": safe_float(pearson_r),
        "Pearson_p": safe_float(pearson_p),
        "n": int(len(y_true)),
    }


def extract_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    estimator = model

    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def save_prediction_plot(
    output_path: Path,
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Observed", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Soil moisture (m³/m³)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_scatter_plot(
    output_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)
    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
    plt.xlabel("Observed soil moisture")
    plt.ylabel("Predicted soil moisture")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_target(
    df: pd.DataFrame,
    target_name: str,
    target_column: str,
    output_dir: Path,
    train_fraction: float,
    validation_fraction: float,
    random_state: int,
):
    X, y, feature_names = build_features(df, target_column)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = chronological_split(
        X,
        y,
        train_fraction,
        validation_fraction,
    )

    LOGGER.info(
        "%s split: train=%d, validation=%d, test=%d",
        target_name,
        len(X_train),
        len(X_val),
        len(X_test),
    )

    models = make_models(random_state)
    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        LOGGER.info("Training %s for %s...", model_name, target_name)

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = calculate_metrics(y_val, val_pred)
        test_metrics = calculate_metrics(y_test, test_pred)

        rows.append(
            {
                "target": target_name,
                "target_column": target_column,
                "model": model_name,
                "split": "validation",
                **val_metrics,
            }
        )
        rows.append(
            {
                "target": target_name,
                "target_column": target_column,
                "model": model_name,
                "split": "test",
                **test_metrics,
            }
        )

        prediction_frame = pd.DataFrame(
            {
                "datetime_utc": df["datetime_utc"].iloc[
                    len(df) - len(y_test):
                ].reset_index(drop=True),
                "observed": y_test.reset_index(drop=True),
                "predicted": test_pred,
                "residual": y_test.reset_index(drop=True).to_numpy() - test_pred,
            }
        )

        prediction_frame.to_csv(
            output_dir / f"{target_name}_{model_name}_test_predictions.csv",
            index=False,
        )

        importance = extract_feature_importance(model, feature_names)
        importance.to_csv(
            output_dir / f"{target_name}_{model_name}_feature_importance.csv",
            index=False,
        )

        save_prediction_plot(
            output_dir / f"{target_name}_{model_name}_timeseries.png",
            prediction_frame["datetime_utc"],
            prediction_frame["observed"].to_numpy(),
            prediction_frame["predicted"].to_numpy(),
            f"AURORA: {target_name} soil moisture — {model_name}",
        )

        save_scatter_plot(
            output_dir / f"{target_name}_{model_name}_scatter.png",
            prediction_frame["observed"].to_numpy(),
            prediction_frame["predicted"].to_numpy(),
            f"AURORA: {target_name} — {model_name}",
        )

        fitted_models[model_name] = {
            "model": model,
            "test_metrics": test_metrics,
        }

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        output_dir / f"{target_name}_metrics.csv",
        index=False,
    )

    test_metrics_df = metrics_df[metrics_df["split"] == "test"].copy()
    test_metrics_df = test_metrics_df.sort_values("RMSE", ascending=True)

    best_model_name = test_metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]["model"]

    best_pred = best_model.predict(X_test)
    best_predictions = pd.DataFrame(
        {
            "datetime_utc": df["datetime_utc"].iloc[
                len(df) - len(y_test):
            ].reset_index(drop=True),
            "observed": y_test.reset_index(drop=True),
            "predicted": best_pred,
            "residual": y_test.reset_index(drop=True).to_numpy() - best_pred,
        }
    )
    best_predictions.to_csv(
        output_dir / f"{target_name}_BEST_{best_model_name}_predictions.csv",
        index=False,
    )

    return {
        "target": target_name,
        "target_column": target_column,
        "n_rows": len(y),
        "n_features": len(feature_names),
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
        "test_rows": len(y_test),
        "best_model": best_model_name,
        "best_test_metrics": fitted_models[best_model_name]["test_metrics"],
        "all_metrics": rows,
        "feature_names": feature_names,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    input_file = Path(args.input_file)

    LOGGER.info("Starting AURORA soil-moisture-only pipeline.")
    LOGGER.info("Input: %s", input_file)
    LOGGER.info("Output: %s", output_dir)

    df = load_dataset(input_file)

    original_columns = list(df.columns)
    df, removed_all_missing, removed_constant = remove_invalid_columns(df)

    # Save a clean copy for traceability.
    df.to_csv(output_dir / "cleaned_input_dataset.csv", index=False)

    dataset_summary = {
        "input_file": str(input_file),
        "rows": int(len(df)),
        "columns_after_cleaning": int(len(df.columns)),
        "original_columns": int(len(original_columns)),
        "removed_all_missing_columns": removed_all_missing,
        "removed_constant_columns": removed_constant,
        "rainfall_columns_ignored": sorted(RAIN_COLUMNS),
        "targets": TARGETS,
        "leakage_rule": "All future_* columns are excluded from model features.",
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as file:
        json.dump(dataset_summary, file, indent=2)

    all_results = []

    for target_name, target_column in TARGETS.items():
        result = run_target(
            df=df,
            target_name=target_name,
            target_column=target_column,
            output_dir=output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            random_state=args.random_state,
        )
        all_results.append(result)

    summary_rows = []
    for result in all_results:
        metrics = result["best_test_metrics"]
        summary_rows.append(
            {
                "target": result["target"],
                "target_column": result["target_column"],
                "best_model": result["best_model"],
                "n_rows": result["n_rows"],
                "n_features": result["n_features"],
                "train_rows": result["train_rows"],
                "validation_rows": result["validation_rows"],
                "test_rows": result["test_rows"],
                **metrics,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "BEST_MODEL_SUMMARY.csv", index=False)

    report_lines = [
        "AURORA SOIL-MOISTURE-ONLY MODEL REPORT",
        "=" * 48,
        "",
        f"Input file: {input_file}",
        f"Rows after cleaning: {len(df)}",
        f"Columns after cleaning: {len(df.columns)}",
        "",
        "Important limitation:",
        "The supplied dataset has missing rainfall fields. Rainfall columns were",
        "therefore excluded. This experiment evaluates Sentinel + soil-moisture",
        "forecasting only.",
        "",
        "Leakage prevention:",
        "All columns beginning with 'future_' were excluded from model inputs.",
        "",
        "Chronological split:",
        f"Training fraction: {args.train_fraction:.2f}",
        f"Validation fraction: {args.validation_fraction:.2f}",
        f"Test fraction: {1.0 - args.train_fraction - args.validation_fraction:.2f}",
        "",
        "Best test results:",
        "",
    ]

    for row in summary_rows:
        report_lines.extend(
            [
                f"Target: {row['target']}",
                f"Best model: {row['best_model']}",
                f"MAE: {row['MAE']}",
                f"RMSE: {row['RMSE']}",
                f"R2: {row['R2']}",
                f"Pearson r: {row['Pearson_r']}",
                f"Test rows: {row['test_rows']}",
                "",
            ]
        )

    with open(output_dir / "AURORA_SOILMOISTURE_REPORT.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    LOGGER.info("Finished successfully.")
    LOGGER.info("Best-model summary:")
    LOGGER.info("\n%s", summary_df.to_string(index=False))
    LOGGER.info("Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
'''G:\AURORA\AURORA_SOIL_RAIN_OUTPUT\AURORA_soil_rain_usable_dataset.csv
py .\AURORA_SOILMOISTURE_FINAL.py --input-file "G:\AURORA\AURORA_SOIL_RAIN_OUTPUT\AURORA_soil_rain_usable_dataset.csv" --output-dir "G:\AURORA\AURORA_SOILMOISTURE_FINAL_OUTPUT"
'''