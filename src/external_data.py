
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

STATE_ABBREVIATIONS = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}

FRED_SERIES = {
    "vmt_millions": {"series_id": "TRFVOLUSM227NFWA"},
    "wti_usd_per_barrel": {"series_id": "DCOILWTICO"},
    "gas_usd_per_gallon": {"series_id": "GASREGW"},
    "new_vehicle_cpi": {"series_id": "CUUR0000SETA01"},
    "used_vehicle_cpi": {"series_id": "CUSR0000SETA02"},
    "new_auto_loan_rate_pct": {"series_id": "RIFLPBCIANM60NM"}
}

def load_state_population_reference(cache_path="data/external/state_population_reference.csv"):
    cache = Path(cache_path)
    if cache.exists():
        reference = pd.read_csv(cache)
        required = {"state_name", "State", "population"}
        if required.issubset(reference.columns):
            return reference

    api_url = "https://api.census.gov/data/2023/acs/acs1?get=NAME,B01003_001E&for=state:*"
    response = requests.get(api_url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    reference = pd.DataFrame(payload[1:], columns=payload[0])
    reference = reference.rename(columns={"NAME": "state_name", "B01003_001E": "population"})
    reference["population"] = pd.to_numeric(reference["population"], errors="coerce")
    reference["State"] = reference["state_name"].map(STATE_ABBREVIATIONS)
    reference = reference.loc[reference["State"].notna(), ["state_name", "State", "population"]].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(cache, index=False)
    return reference


def load_fred_series(series_id: str, cache_dir="data/external") -> pd.DataFrame:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"fred_{series_id}.csv"
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        text = response.text
        cache_path.write_text(text, encoding="utf-8")

    frame = pd.read_csv(StringIO(text))
    value_column = [column for column in frame.columns if column != "observation_date"][0]
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=["observation_date"]).copy()
    frame = frame.rename(columns={value_column: series_id})
    return frame[["observation_date", series_id]].drop_duplicates(subset=["observation_date"])


def _monthly_series_mean(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    monthly = frame.copy()
    monthly["month"] = monthly["observation_date"].dt.to_period("M").dt.to_timestamp()
    return monthly.groupby("month", dropna=False)[value_column].mean().reset_index()


def build_national_macro_auto_context(enriched: pd.DataFrame, cache_dir="data/external") -> pd.DataFrame:
    monthly_accidents = (
        enriched.groupby(enriched["Start_Time"].dt.to_period("M"))
        .size()
        .rename("accident_count")
        .reset_index()
    )
    monthly_accidents["month"] = monthly_accidents["Start_Time"].dt.to_timestamp()
    monthly_accidents = monthly_accidents[["month", "accident_count"]]

    context = monthly_accidents.copy()
    for column_name, spec in FRED_SERIES.items():
        series_id = spec["series_id"]
        series_frame = load_fred_series(series_id, cache_dir=cache_dir)
        monthly_series = _monthly_series_mean(series_frame, series_id).rename(columns={series_id: column_name})
        context = context.merge(monthly_series, on="month", how="left")

    return context.sort_values("month").reset_index(drop=True)


def add_state_population_rates(panel, reference=None):
    augmented = panel.copy()
    population_reference = reference if reference is not None else load_state_population_reference()
    state_panel = augmented.loc[augmented["scope_type"] == "state"].merge(
        population_reference[["State", "population"]],
        left_on="scope_name",
        right_on="State",
        how="left",
    )
    state_panel["accidents_per_100k"] = (state_panel["accident_count"] / state_panel["population"]) * 100_000

    augmented = augmented.merge(
        state_panel[["local_date", "scope_name", "accidents_per_100k"]],
        on=["local_date", "scope_name"],
        how="left",
    )
    return augmented


