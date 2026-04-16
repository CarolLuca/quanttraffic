
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
from pandas.tseries.holiday import USFederalHolidayCalendar

WEATHER_BUCKETS = {
    "Rain/Wet": ["rain", "drizzle", "showers", "storm", "thunderstorm"],
    "Snow/Ice": ["snow", "ice", "sleet", "freezing", "hail"],
    "Fog/Reduced Vis": ["fog", "haze", "mist", "smoke", "dust"],
    "Clear/Cloudy": ["clear", "cloudy", "fair", "overcast", "scattered"],
}

ROAD_KEYWORDS = {
    "Interstate": [" i-", "interstate", " interstate "],
    "Highway": ["hwy", "highway", "us-", "state route", "us hwy"],
    "Local Road": ["st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "dr", "drive", "ln", "lane"],
    "Ramp": ["ramp", "exit"]
}

from data_ingestion import *
from config import METRO_CASE_STUDIES
def _weather_bucket(condition):
    if pd.isna(condition):
        return "Unknown"
    text = str(condition).lower()
    for label, keywords in WEATHER_BUCKETS.items():
        if any(keyword in text for keyword in keywords):
            return label
    return "Other"


def _road_type(street_value):
    if pd.isna(street_value):
        return "Unknown"
    text = f" {str(street_value).lower()} "
    for label, keywords in ROAD_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return label
    return "Unknown"


def _moon_features(timestamp_series):
    ts = pd.to_datetime(timestamp_series, errors="coerce").dt.tz_localize(None)
    reference = pd.Timestamp("2001-01-01 00:00:00")
    days = (ts - reference).dt.total_seconds().div(86400.0)
    lunation = (0.20439731 + days / 29.53058867) % 1
    illumination = ((1 - np.cos(2 * np.pi * lunation)) / 2) * 100
    phase_names = pd.cut(
        lunation,
        bins=[-0.001, 0.03, 0.22, 0.28, 0.47, 0.53, 0.72, 0.78, 1.0],
        labels=[
            "New Moon",
            "Waxing Crescent",
            "First Quarter",
            "Waxing Gibbous",
            "Full Moon",
            "Waning Gibbous",
            "Last Quarter",
            "Waning Crescent",
        ],
    ).astype("string")
    return phase_names, illumination


def _approx_day_length_hours(date_series: pd.Series, lat_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series, errors="coerce")
    latitudes = pd.to_numeric(lat_series, errors="coerce").clip(-66.4, 66.4)
    day_of_year = dates.dt.dayofyear.fillna(172).astype(float)
    declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (day_of_year - 81) / 365.0)
    lat_radians = np.deg2rad(latitudes)
    cos_omega = -np.tan(lat_radians) * np.tan(declination)
    omega = np.arccos(np.clip(cos_omega, -1, 1))
    hours = 24 * omega / np.pi
    return pd.Series(hours, index=date_series.index)


def _federal_holidays(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    calendar = USFederalHolidayCalendar()
    return calendar.holidays(start=start_date.normalize(), end=end_date.normalize())


def _is_dst_transition(local_date, timezone_name):
    if pd.isna(local_date) or pd.isna(timezone_name):
        return False
    try:
        timezone = tz.gettz(str(timezone_name))
    except Exception:
        return False
    if timezone is None:
        return False

    noon = datetime.combine(local_date, time(hour=12))
    current_offset = noon.replace(tzinfo=timezone).utcoffset()
    prev_offset = (noon - timedelta(days=1)).replace(tzinfo=timezone).utcoffset()
    next_offset = (noon + timedelta(days=1)).replace(tzinfo=timezone).utcoffset()
    return current_offset != prev_offset or current_offset != next_offset


def _assign_metro(row: pd.Series) -> str:
    state = row.get("State")
    county = row.get("County")
    if pd.isna(state) or pd.isna(county):
        return "Other"
    county_clean = str(county).replace(" County", "").strip()
    for metro_name, spec in METRO_CASE_STUDIES.items():
        if state == spec["state"] and county_clean in spec["counties"]:
            return metro_name
    return "Other"


def build_accidents_enriched(clean: pd.DataFrame) -> pd.DataFrame:
    enriched = clean.copy()
    moon_phase, illumination = _moon_features(enriched["Start_Time"])
    enriched["moon_phase"] = moon_phase
    enriched["moon_illumination_pct"] = illumination
    enriched["day_length_hours"] = _approx_day_length_hours(enriched["Start_Time"], enriched["Start_Lat"])
    enriched["is_half_moon_window"] = enriched["moon_phase"].isin(["First Quarter", "Last Quarter"])
    enriched["is_full_moon_window"] = enriched["moon_phase"].eq("Full Moon")

    enriched["weather_bucket"] = enriched.get("Weather_Condition", pd.Series(index=enriched.index)).map(_weather_bucket)
    enriched["road_type"] = enriched.get("Street", pd.Series(index=enriched.index)).map(_road_type)
    enriched["road_scale"] = np.where(
        enriched["road_type"].isin(["Interstate", "Highway", "Ramp"]),
        "High-speed corridor",
        np.where(enriched["road_type"].eq("Local Road"), "Local network", "Unknown"),
    )

    if enriched["Start_Time"].notna().any():
        holidays = set(_federal_holidays(enriched["Start_Time"].min(), enriched["Start_Time"].max()).date)
    else:
        holidays = set()

    enriched["is_federal_holiday"] = enriched["start_date"].isin(holidays)
    enriched["days_to_nearest_holiday"] = enriched["start_date"].map(
        lambda current: _distance_to_nearest_holiday(current, holidays)
    )
    enriched["is_holiday_window"] = enriched["days_to_nearest_holiday"].le(1)
    enriched["is_long_weekend"] = (
        enriched["is_federal_holiday"]
        | ((enriched["weekday_index"] == 4) & enriched["days_to_nearest_holiday"].le(3))
        | ((enriched["weekday_index"] == 0) & enriched["days_to_nearest_holiday"].le(3))
    )
    enriched["is_dst_transition_day"] = [
        _is_dst_transition(local_date, timezone_name)
        for local_date, timezone_name in zip(enriched["start_date"], enriched.get("Timezone", pd.Series(index=enriched.index)))
    ]

    enriched["temperature_bucket"] = pd.cut(
        enriched.get("Temperature(F)", pd.Series(index=enriched.index)),
        bins=[-100, 32, 50, 68, 86, 130],
        labels=["Freezing", "Cold", "Mild", "Warm", "Hot"],
    ).astype("string")
    enriched["visibility_bucket"] = pd.cut(
        enriched.get("Visibility(mi)", pd.Series(index=enriched.index)),
        bins=[-0.1, 1, 3, 6, 100],
        labels=["Very Low", "Low", "Moderate", "High"],
    ).astype("string")
    enriched["wind_bucket"] = pd.cut(
        enriched.get("Wind_Speed(mph)", pd.Series(index=enriched.index)),
        bins=[-0.1, 5, 15, 30, 200],
        labels=["Calm", "Breezy", "Windy", "Severe Wind"],
    ).astype("string")
    enriched["precip_bucket"] = pd.cut(
        enriched.get("Precipitation(in)", pd.Series(index=enriched.index)).fillna(0),
        bins=[-0.001, 0, 0.05, 0.2, 10],
        labels=["None", "Trace", "Light", "Heavy"],
    ).astype("string")
    enriched["metro_case_study"] = enriched.apply(_assign_metro, axis=1)
    return enriched


def _distance_to_nearest_holiday(current_date, holidays):
    if pd.isna(current_date) or not holidays:
        return np.nan
    return min(abs((current_date - holiday).days) for holiday in holidays)


