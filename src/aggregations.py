
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

def build_panel_state_city(
    enriched: pd.DataFrame,
    top_cities: int = 40,
) -> pd.DataFrame:
    base = enriched.copy()
    base["local_date"] = pd.to_datetime(base["start_date"])
    top_city_names = (
        base["City"].fillna("Unknown").value_counts().head(top_cities).index.to_list()
        if "City" in base.columns
        else []
    )

    def _aggregate(frame, group_cols, scope_type):
        grouped = (
            frame.groupby(group_cols, dropna=False)
            .agg(
                accident_count=("ID", "size"),
                severe_share=("is_severe", "mean"),
                median_duration_min=("duration_minutes", "median"),
                night_share=("is_night", "mean"),
                intersection_share=("intersection_context", "mean"),
                precip_share=("precip_bucket", lambda s: s.isin(["Light", "Heavy"]).mean()),
                low_visibility_share=("visibility_bucket", lambda s: s.isin(["Very Low", "Low"]).mean()),
                holiday_share=("is_holiday_window", "mean"),
                half_moon_share=("is_half_moon_window", "mean"),
                full_moon_share=("is_full_moon_window", "mean"),
                mean_day_length_hours=("day_length_hours", "mean"),
            )
            .reset_index()
        )
        grouped["scope_type"] = scope_type
        return grouped

    panels = []
    national = _aggregate(base.assign(scope_name="United States"), ["local_date", "scope_name"], "national")
    panels.append(national)

    if "State" in base.columns:
        state_panel = _aggregate(
            base.assign(scope_name=base["State"].fillna("Unknown")),
            ["local_date", "scope_name"],
            "state",
        )
        panels.append(state_panel)

    metro_frame = base.loc[base["metro_case_study"] != "Other"].assign(scope_name=base["metro_case_study"])
    if not metro_frame.empty:
        metro_panel = _aggregate(metro_frame, ["local_date", "scope_name"], "metro")
        panels.append(metro_panel)

    if "City" in base.columns and top_city_names:
        city_frame = base.loc[base["City"].isin(top_city_names)].assign(scope_name=base["City"].fillna("Unknown"))
        city_panel = _aggregate(city_frame, ["local_date", "scope_name"], "city")
        panels.append(city_panel)

    panel = pd.concat(panels, ignore_index=True)
    panel["weekday"] = panel["local_date"].dt.day_name()
    panel["month"] = panel["local_date"].dt.month
    panel["year"] = panel["local_date"].dt.year
    panel["day_of_year"] = panel["local_date"].dt.dayofyear
    panel["is_weekend"] = panel["weekday"].isin(["Saturday", "Sunday"])
    return panel


def build_hotspot_panel(enriched: pd.DataFrame, precision: int = 2) -> pd.DataFrame:
    hotspot = enriched.copy()
    hotspot["lat_bin"] = hotspot["Start_Lat"].round(precision)
    hotspot["lng_bin"] = hotspot["Start_Lng"].round(precision)
    return (
        hotspot.groupby(["State", "metro_case_study", "lat_bin", "lng_bin"], dropna=False)
        .agg(
            accident_count=("ID", "size"),
            severe_share=("is_severe", "mean"),
            median_duration_min=("duration_minutes", "median"),
            night_share=("is_night", "mean"),
            intersection_share=("intersection_context", "mean"),
        )
        .reset_index()
        .sort_values("accident_count", ascending=False)
    )


def build_daily_context_panel(enriched: pd.DataFrame) -> pd.DataFrame:
    daily = (
        enriched.groupby("start_date", dropna=False)
        .agg(
            accident_count=("ID", "size"),
            severe_share=("is_severe", "mean"),
            mean_duration_min=("duration_minutes", "median"),
            mean_visibility=("Visibility(mi)", "median"),
            night_share=("is_night", "mean"),
            intersection_share=("intersection_context", "mean"),
            precipitation_share=("precip_bucket", lambda s: s.isin(["Light", "Heavy"]).mean()),
            holiday_window=("is_holiday_window", "max"),
            long_weekend=("is_long_weekend", "max"),
            dst_transition=("is_dst_transition_day", "max"),
            half_moon=("is_half_moon_window", "max"),
            full_moon=("is_full_moon_window", "max"),
            moon_illumination_pct=("moon_illumination_pct", "mean"),
            day_length_hours=("day_length_hours", "mean"),
        )
        .reset_index()
        .rename(columns={"start_date": "local_date"})
    )
    daily["local_date"] = pd.to_datetime(daily["local_date"])
    daily["weekday"] = daily["local_date"].dt.day_name()
    daily["month"] = daily["local_date"].dt.month
    daily["year"] = daily["local_date"].dt.year
    daily["rolling_28d_median"] = daily["accident_count"].rolling(28, min_periods=14).median()
    coverage_floor = daily["rolling_28d_median"] * 0.10
    daily["low_coverage_day"] = (
        daily["rolling_28d_median"].notna()
        & daily["accident_count"].lt(coverage_floor.fillna(-np.inf))
    )
    return daily


