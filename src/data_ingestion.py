
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

TIME_COLUMNS = ["Start_Time", "End_Time", "Weather_Timestamp"]
NUMERIC_COLUMNS = [
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Distance(mi)"
]
POI_COLUMNS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop"
]

def discover_dataset_path(data_dir="data"):
    data_path = Path(data_dir)
    candidates = sorted(data_path.glob("**/US_Accidents*.csv"))
    if not candidates:
        candidates = sorted(data_path.glob("**/US_Accidents*.parquet"))
    if not candidates:
        candidates = sorted(data_path.glob("**/*.parquet"))
    if not candidates:
        candidates = sorted(data_path.glob("**/*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No dataset found under data/. Place the US Accidents CSV or Parquet file under data/."
        )
    return candidates[0]


def _normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized_map = {_normalize_column_name(col): col for col in df.columns}
    time_columns = globals().get("TIME_COLUMNS", ["Start_Time", "End_Time", "Weather_Timestamp"])
    numeric_columns = globals().get(
        "NUMERIC_COLUMNS",
        [
            "Temperature(F)",
            "Wind_Chill(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "Distance(mi)",
        ],
    )
    poi_columns = globals().get(
        "POI_COLUMNS",
        [
            "Amenity",
            "Bump",
            "Crossing",
            "Give_Way",
            "Junction",
            "No_Exit",
            "Railway",
            "Roundabout",
            "Station",
            "Stop",
            "Traffic_Calming",
            "Traffic_Signal",
            "Turning_Loop",
        ],
    )
    canonical_aliases = {
        "id": "ID",
        "severity": "Severity",
        "start_time": "Start_Time",
        "end_time": "End_Time",
        "start_lat": "Start_Lat",
        "start_lng": "Start_Lng",
        "end_lat": "End_Lat",
        "end_lng": "End_Lng",
        "distance_mi": "Distance(mi)",
        "description": "Description",
        "street": "Street",
        "city": "City",
        "county": "County",
        "state": "State",
        "timezone": "Timezone",
        "weather_timestamp": "Weather_Timestamp",
        "temperaturef": "Temperature(F)",
        "wind_chillf": "Wind_Chill(F)",
        "humidity%": "Humidity(%)",
        "pressurein": "Pressure(in)",
        "visibilitymi": "Visibility(mi)",
        "wind_speedmph": "Wind_Speed(mph)",
        "precipitationin": "Precipitation(in)",
        "weather_condition": "Weather_Condition",
        "sunrise_sunset": "Sunrise_Sunset",
        "civil_twilight": "Civil_Twilight",
        "nautical_twilight": "Nautical_Twilight",
        "astronomical_twilight": "Astronomical_Twilight",
        "source": "Source",
    }
    rename_map = {}
    for normalized_name, canonical_name in canonical_aliases.items():
        if normalized_name in normalized_map and canonical_name not in df.columns:
            rename_map[normalized_map[normalized_name]] = canonical_name
    df = df.rename(columns=rename_map).copy()

    for column in time_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in poi_columns:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)

    if "Severity" in df.columns:
        df["Severity"] = df["Severity"].astype("Int64")

    return df


def load_accidents(csv_path=None, data_dir="data", nrows=None):
    csv_file = Path(csv_path) if csv_path else discover_dataset_path(data_dir)
    suffix = csv_file.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(csv_file, low_memory=False, nrows=nrows)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(csv_file)
        if nrows is not None:
            df = df.head(nrows).copy()
    else:
        raise ValueError("Unsupported dataset format: {}".format(csv_file))
    return harmonize_schema(df)


def build_accidents_clean(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean = clean.loc[clean["Start_Time"].notna() & clean["Start_Lat"].notna() & clean["Start_Lng"].notna()].copy()
    if "End_Time" in clean.columns:
        duration = (clean["End_Time"] - clean["Start_Time"]).dt.total_seconds().div(60.0)
        clean["duration_minutes"] = duration.clip(lower=0)
    else:
        clean["duration_minutes"] = np.nan

    clean["start_date"] = clean["Start_Time"].dt.date
    clean["year"] = clean["Start_Time"].dt.year
    clean["month"] = clean["Start_Time"].dt.month
    clean["month_name"] = clean["Start_Time"].dt.month_name()
    clean["quarter"] = clean["Start_Time"].dt.quarter
    clean["weekday"] = clean["Start_Time"].dt.day_name()
    clean["weekday_index"] = clean["Start_Time"].dt.dayofweek
    clean["is_weekend"] = clean["weekday_index"] >= 5
    clean["hour"] = clean["Start_Time"].dt.hour
    clean["week"] = clean["Start_Time"].dt.isocalendar().week.astype(int)
    clean["season"] = clean["month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    clean["rush_period"] = pd.cut(
        clean["hour"],
        bins=[-1, 5, 9, 15, 19, 24],
        labels=["Late Night", "Morning Rush", "Midday", "Evening Rush", "Night"],
    ).astype("string")
    clean["is_night"] = (
        clean.get("Sunrise_Sunset", pd.Series(index=clean.index, dtype="object")).fillna("").eq("Night")
        | clean.get("Civil_Twilight", pd.Series(index=clean.index, dtype="object")).fillna("").eq("Night")
    )
    clean["is_severe"] = clean["Severity"].fillna(0).astype(int) >= 3
    clean["description_text"] = (
        clean.get("Description", pd.Series(index=clean.index, dtype="object")).fillna("").astype(str)
    )
    desc = clean["description_text"].str.lower()
    clean["description_word_count"] = desc.str.split().str.len().fillna(0)
    clean["mentions_lane"] = desc.str.contains("lane", regex=False)
    clean["mentions_blocked"] = desc.str.contains("blocked", regex=False)
    clean["mentions_overturn"] = desc.str.contains("overturn", regex=False)
    clean["mentions_vehicle"] = desc.str.contains("vehicle", regex=False)
    clean["intersection_context"] = (
        clean.get("Junction", pd.Series(index=clean.index, data=False)).fillna(False)
        | clean.get("Crossing", pd.Series(index=clean.index, data=False)).fillna(False)
        | clean.get("Traffic_Signal", pd.Series(index=clean.index, data=False)).fillna(False)
        | clean.get("Stop", pd.Series(index=clean.index, data=False)).fillna(False)
    )

    if "Weather_Timestamp" in clean.columns:
        weather_lag = (clean["Start_Time"] - clean["Weather_Timestamp"]).dt.total_seconds().div(60.0)
        clean["weather_report_lag_min"] = weather_lag.clip(lower=0)
    else:
        clean["weather_report_lag_min"] = np.nan

    return clean


