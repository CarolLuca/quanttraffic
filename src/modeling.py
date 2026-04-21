
import json
import os
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

from dateutil import tz
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, mean_squared_error, mean_absolute_error, average_precision_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from data_ingestion import POI_COLUMNS
from gpu_utils import use_gpu_acceleration
def _temporal_split_mask(frame, date_column, test_fraction=0.2):
    unique_dates = pd.Series(pd.to_datetime(frame[date_column].dropna().unique())).sort_values()
    cutoff_index = max(int(len(unique_dates) * (1 - test_fraction)) - 1, 0)
    cutoff = unique_dates.iloc[cutoff_index]
    train_mask = pd.to_datetime(frame[date_column]) <= cutoff
    test_mask = ~train_mask
    return train_mask, test_mask


def _coerce_model_inputs(frame, numeric_features, categorical_features, bool_features):
    formatted = frame.copy()
    for column in numeric_features:
        formatted[column] = pd.to_numeric(formatted[column], errors="coerce")
    for column in categorical_features:
        formatted[column] = formatted[column].astype(object)
        formatted.loc[formatted[column].isna(), column] = np.nan
    for column in bool_features:
        formatted[column] = formatted[column].fillna(False).astype(int)
    return formatted


def _sample_balanced_training_frame(frame: pd.DataFrame, sample_size: int, time_column: str) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame.sort_values(time_column).copy()
    severe = frame.loc[frame["is_severe"]].copy()
    mild = frame.loc[~frame["is_severe"]].copy()
    severe_sample = severe.sample(min(len(severe), sample_size // 2), random_state=42)
    mild_sample = mild.sample(min(len(mild), sample_size - len(severe_sample)), random_state=42)
    return pd.concat([severe_sample, mild_sample], ignore_index=True).sort_values(time_column).copy()


def _sample_natural_holdout_frame(frame: pd.DataFrame, sample_size: int, time_column: str) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame.sort_values(time_column).copy()

    prevalence = frame["is_severe"].mean()
    severe_target = int(round(sample_size * prevalence))
    severe_frame = frame.loc[frame["is_severe"]].copy()
    mild_frame = frame.loc[~frame["is_severe"]].copy()
    severe_target = min(len(severe_frame), severe_target)
    mild_target = min(len(mild_frame), sample_size - severe_target)
    if severe_target + mild_target < sample_size:
        remainder = sample_size - (severe_target + mild_target)
        extra_mild = min(len(mild_frame) - mild_target, remainder)
        mild_target += max(extra_mild, 0)
        remainder -= max(extra_mild, 0)
        if remainder > 0:
            severe_target += min(len(severe_frame) - severe_target, remainder)

    sampled = pd.concat(
        [
            severe_frame.sample(severe_target, random_state=42) if severe_target else severe_frame.head(0),
            mild_frame.sample(mild_target, random_state=42) if mild_target else mild_frame.head(0),
        ],
        ignore_index=True,
    )
    return sampled.sort_values(time_column).copy()


def train_severity_models(enriched, sample_size=250000, test_sample_size=150000, enable_gpu: bool = True):
    model_df = enriched.copy()
    model_df = model_df.loc[model_df["Severity"].notna() & model_df["Start_Time"].notna()].sort_values("Start_Time").copy()
    train_mask, test_mask = _temporal_split_mask(model_df, "Start_Time")
    train_df = _sample_balanced_training_frame(model_df.loc[train_mask].copy(), sample_size=sample_size, time_column="Start_Time")
    test_df = _sample_natural_holdout_frame(
        model_df.loc[test_mask].copy(),
        sample_size=test_sample_size,
        time_column="Start_Time",
    )
    model_df = pd.concat([train_df, test_df], ignore_index=True)

    target = model_df["is_severe"].astype(int)
    feature_frame = model_df[
        [
            "Distance(mi)",
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "Start_Lat",
            "Start_Lng",
            "duration_minutes",
            "weather_report_lag_min",
            "day_length_hours",
            "moon_illumination_pct",
            "hour",
            "month",
            "description_word_count",
            "weekday",
            "weather_bucket",
            "road_scale",
            "temperature_bucket",
            "visibility_bucket",
            "wind_bucket",
            "precip_bucket",
            "Sunrise_Sunset",
            "Civil_Twilight",
            "Timezone",
            "mentions_lane",
            "mentions_blocked",
            "mentions_overturn",
            "mentions_vehicle",
        ]
        + [col for col in POI_COLUMNS if col in model_df.columns]
    ].copy()

    numeric_features = [
        column
        for column in feature_frame.columns
        if column
        not in {
            "weekday",
            "weather_bucket",
            "road_scale",
            "temperature_bucket",
            "visibility_bucket",
            "wind_bucket",
            "precip_bucket",
            "Sunrise_Sunset",
            "Civil_Twilight",
            "Timezone",
        }
        and feature_frame[column].dtype != bool
    ]
    categorical_features = [
        "weekday",
        "weather_bucket",
        "road_scale",
        "temperature_bucket",
        "visibility_bucket",
        "wind_bucket",
        "precip_bucket",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Timezone",
    ]
    bool_features = [column for column in feature_frame.columns if feature_frame[column].dtype == bool]

    split_train_mask = model_df["Start_Time"] <= train_df["Start_Time"].max()
    split_test_mask = ~split_train_mask
    x_train = feature_frame.loc[split_train_mask]
    x_test = feature_frame.loc[split_test_mask]
    y_train = target.loc[split_train_mask]
    y_test = target.loc[split_test_mask]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("boolean", "passthrough", bool_features),
        ],
        remainder="drop",
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        ),
    }

    if use_gpu_acceleration(enable_gpu):
        try:
            import xgboost as xgb  # type: ignore

            positive = max(int(y_train.sum()), 1)
            negative = max(int((1 - y_train).sum()), 1)
            candidate_models["XGBoost (GPU)"] = xgb.XGBClassifier(
                n_estimators=450,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=float(negative / positive),
                tree_method="hist",
                device="cuda",
                random_state=42,
            )
        except Exception:
            pass

    results = []
    metric_rows = []
    x_train = _coerce_model_inputs(x_train, numeric_features, categorical_features, bool_features)
    x_test = _coerce_model_inputs(x_test, numeric_features, categorical_features, bool_features)
    for name, estimator in candidate_models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "pr_auc": float(average_precision_score(y_test, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
        }
        metric_rows.append(
            {
                "model": name,
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "test_prevalence": float(y_test.mean()),
                **metrics,
            }
        )
        results.append(
            {
                "name": name,
                "metrics": metrics,
                "model": pipeline,
                "test_prevalence": float(y_test.mean()),
            }
        )
    return pd.DataFrame(metric_rows).sort_values("pr_auc", ascending=False), results


def train_severity_ablation(enriched, sample_size=180000):
    model_df = enriched.copy()
    model_df = model_df.loc[model_df["Severity"].notna() & model_df["Start_Time"].notna()].copy()
    if len(model_df) > sample_size:
        model_df = model_df.sample(sample_size, random_state=42).sort_values("Start_Time")

    y = model_df["is_severe"].astype(int)
    feature_groups = {
        "Kaggle Core": [
            "Distance(mi)",
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "weekday",
            "hour",
            "month",
            "Sunrise_Sunset",
            "description_word_count",
            "mentions_lane",
            "mentions_blocked",
            "mentions_vehicle",
        ],
        "Core + Calendar": [
            "Distance(mi)",
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "weekday",
            "hour",
            "month",
            "Sunrise_Sunset",
            "moon_illumination_pct",
            "day_length_hours",
            "is_holiday_window",
            "is_dst_transition_day",
            "description_word_count",
            "mentions_lane",
            "mentions_blocked",
        ],
        "Core + Context": [
            "Distance(mi)",
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "weekday",
            "hour",
            "month",
            "Sunrise_Sunset",
            "road_scale",
            "weather_bucket",
            "intersection_context",
            "metro_case_study",
            "description_word_count",
            "mentions_lane",
            "mentions_vehicle",
        ]
        + [column for column in POI_COLUMNS if column in model_df.columns],
        "Full Structured": [
            "Distance(mi)",
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "weekday",
            "hour",
            "month",
            "Sunrise_Sunset",
            "moon_illumination_pct",
            "day_length_hours",
            "is_holiday_window",
            "is_dst_transition_day",
            "road_scale",
            "weather_bucket",
            "intersection_context",
            "metro_case_study",
            "description_word_count",
            "mentions_lane",
            "mentions_blocked",
            "mentions_overturn",
            "mentions_vehicle",
        ]
        + [column for column in POI_COLUMNS if column in model_df.columns],
    }

    train_mask, test_mask = _temporal_split_mask(model_df, "Start_Time")
    records = []
    for label, columns in feature_groups.items():
        available = [column for column in columns if column in model_df.columns]
        x_train = model_df.loc[train_mask, available].copy()
        x_test = model_df.loc[test_mask, available].copy()

        numeric_features = [col for col in available if pd.api.types.is_numeric_dtype(model_df[col]) and model_df[col].dtype != bool]
        bool_features = [col for col in available if model_df[col].dtype == bool]
        categorical_features = [col for col in available if col not in numeric_features + bool_features]
        x_train = _coerce_model_inputs(x_train, numeric_features, categorical_features, bool_features)
        x_test = _coerce_model_inputs(x_test, numeric_features, categorical_features, bool_features)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                    ),
                    numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
                ("boolean", "passthrough", bool_features),
            ]
        )
        model = Pipeline(
            [("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=400, class_weight="balanced"))]
        )
        model.fit(x_train, y.loc[train_mask])
        probability = model.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        records.append(
            {
                "feature_bundle": label,
                "roc_auc": float(roc_auc_score(y.loc[test_mask], probability)),
                "pr_auc": float(average_precision_score(y.loc[test_mask], probability)),
                "f1": float(f1_score(y.loc[test_mask], prediction, zero_division=0)),
            }
        )
    return pd.DataFrame(records).sort_values("pr_auc", ascending=False)


def _build_lagged_features(
    panel: pd.DataFrame,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    frequency: str = "daily",
) -> pd.DataFrame:
    lagged = panel.copy().sort_values("local_date").reset_index(drop=True)
    for lag in lags:
        lagged[f"lag_{lag}"] = lagged["accident_count"].shift(lag)
    for window in rolling_windows:
        lagged[f"rolling_mean_{window}"] = lagged["accident_count"].shift(1).rolling(window).mean()
        lagged[f"rolling_std_{window}"] = lagged["accident_count"].shift(1).rolling(window).std()
    lagged["month"] = lagged["local_date"].dt.month
    lagged["year"] = lagged["local_date"].dt.year
    if frequency == "daily":
        lagged["weekday"] = lagged["local_date"].dt.day_name()
        lagged["day_of_year"] = lagged["local_date"].dt.dayofyear
    else:
        lagged["month_name"] = lagged["local_date"].dt.month_name().str.slice(stop=3)
    lagged = lagged.dropna().copy()
    return lagged


def train_count_forecasters(panel, frequency: str = "daily", baseline_lag: int | None = None, enable_gpu: bool = True):
    scoped = panel.sort_values("local_date").copy()
    if frequency == "monthly":
        lags = (1, 2, 3, 6, 12)
        rolling_windows = (3, 6, 12)
        baseline_lag = baseline_lag or 12
        baseline_name = "Seasonal Naive (lag 12)"
    else:
        lags = (1, 7, 14, 28)
        rolling_windows = (7, 14, 28)
        baseline_lag = baseline_lag or 7
        baseline_name = "Seasonal Naive (lag 7)"

    features = _build_lagged_features(scoped, lags=lags, rolling_windows=rolling_windows, frequency=frequency)
    if features.empty:
        raise ValueError(f"Not enough {frequency} history to build lagged forecasting features.")

    train_mask, test_mask = _temporal_split_mask(features, "local_date")
    x_train = features.drop(columns=["accident_count", "local_date"])
    x_test = x_train.loc[test_mask].copy()
    x_train = x_train.loc[train_mask].copy()
    y_train = features.loc[train_mask, "accident_count"]
    y_test = features.loc[test_mask, "accident_count"]

    categorical_candidates = ["weekday", "month_name"]
    categorical_features = [column for column in categorical_candidates if column in x_train.columns]
    numeric_features = [column for column in x_train.columns if column not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
                ),
                categorical_features,
            ),
        ]
    )

    models = {
        baseline_name: None,
        "Poisson Regressor": Pipeline(
            [("preprocessor", preprocessor), ("model", PoissonRegressor(alpha=0.2, max_iter=1000))]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ]
        ),
    }

    if use_gpu_acceleration(enable_gpu):
        try:
            import xgboost as xgb  # type: ignore

            models["XGBoost Regressor (GPU)"] = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        xgb.XGBRegressor(
                            n_estimators=500,
                            max_depth=8,
                            learning_rate=0.05,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            objective="count:poisson",
                            eval_metric="poisson-nloglik",
                            tree_method="hist",
                            device="cuda",
                            random_state=42,
                        ),
                    ),
                ]
            )
        except Exception:
            pass

    results = []
    metric_rows = []
    base_forecast = features.loc[test_mask, ["local_date", "accident_count"]].copy()
    base_forecast["prediction"] = features.loc[test_mask, f"lag_{baseline_lag}"]
    baseline_metrics = regression_metrics(base_forecast["accident_count"], base_forecast["prediction"])
    metric_rows.append({"model": baseline_name, **baseline_metrics})
    results.append(
        {
            "name": baseline_name,
            "metrics": baseline_metrics,
            "forecast": base_forecast,
            "model": None,
        }
    )

    for name, model in models.items():
        if model is None:
            continue
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        forecast = features.loc[test_mask, ["local_date", "accident_count"]].copy()
        forecast["prediction"] = prediction
        metrics = regression_metrics(y_test, prediction)
        metric_rows.append({"model": name, **metrics})
        results.append({"name": name, "metrics": metrics, "forecast": forecast, "model": model})
    return pd.DataFrame(metric_rows).sort_values("rmse"), results


def build_risk_day_targets(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values("local_date")
    if "low_coverage_day" in df.columns:
        df = df.loc[~df["low_coverage_day"]].copy()
    high_volume_threshold = df["accident_count"].quantile(0.9)
    severe_weather_threshold = df["precip_share"].quantile(0.75)
    night_threshold = df["night_share"].quantile(0.75)
    intersection_threshold = df["intersection_share"].quantile(0.75)

    df["high_volume_day"] = df["accident_count"] >= high_volume_threshold
    df["severe_weather_day"] = (
        (df["precip_share"] >= severe_weather_threshold)
        & (df["low_visibility_share"] >= df["low_visibility_share"].quantile(0.7))
        & (df["accident_count"] >= df["accident_count"].median())
    )
    df["night_intersection_day"] = (
        (df["night_share"] >= night_threshold)
        & (df["intersection_share"] >= intersection_threshold)
        & (df["accident_count"] >= df["accident_count"].quantile(0.75))
    )
    df["holiday_window_day"] = df["holiday_share"] > 0
    return df


def train_risk_day_models(panel: pd.DataFrame, target_column: str) -> pd.DataFrame:
    labeled = build_risk_day_targets(panel)
    features = _build_lagged_features(
        labeled[
            [
                "local_date",
                "accident_count",
                "severe_share",
                "median_duration_min",
                "night_share",
                "intersection_share",
                "precip_share",
                "low_visibility_share",
                "holiday_share",
                "half_moon_share",
                "full_moon_share",
                "mean_day_length_hours",
            ]
        ].copy()
        ,
        lags=(1, 7, 14, 28),
        rolling_windows=(7, 14, 28),
        frequency="daily",
    )
    aligned = labeled.merge(features[["local_date"]], on="local_date", how="inner").merge(
        features, on="local_date", how="inner", suffixes=("", "_lagged")
    )

    y = aligned[target_column].astype(int)
    risk_target_columns = ["high_volume_day", "severe_weather_day", "night_intersection_day", "holiday_window_day"]
    drop_columns = [column for column in ["local_date", "accident_count", *risk_target_columns] if column in aligned.columns]
    x = aligned.drop(columns=drop_columns)
    x = x.select_dtypes(include=[np.number, "bool"]).copy()
    train_mask, test_mask = _temporal_split_mask(aligned, "local_date")

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "target": target_column,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "balanced_accuracy": np.nan,
                    "f1": np.nan,
                    "note": "Not enough class variation for a stable split",
                }
            ]
        )

    numeric_features = list(x.columns)
    model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                            ),
                            numeric_features,
                        )
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )
    model.fit(x.loc[train_mask], y_train)
    probability = model.predict_proba(x.loc[test_mask])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return pd.DataFrame(
        [
            {
                "target": target_column,
                "roc_auc": float(roc_auc_score(y_test, probability)),
                "pr_auc": float(average_precision_score(y_test, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                "f1": float(f1_score(y_test, prediction, zero_division=0)),
                "note": "OK",
            }
        ]
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
    smape = np.mean(2 * np.abs(y_pred - y_true) / denominator)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": float(smape),
    }


