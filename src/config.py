
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

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

METRO_CASE_STUDIES = {
    "Los Angeles": {
        "state": "CA",
        "counties": {"Los Angeles", "Orange", "San Bernardino", "Riverside", "Ventura"},
        "wikimedia_query": "Los Angeles skyline",
    },
    "Houston": {
        "state": "TX",
        "counties": {"Harris", "Fort Bend", "Montgomery", "Brazoria", "Galveston"},
        "wikimedia_query": "Houston skyline",
    },
    "Miami": {
        "state": "FL",
        "counties": {"Miami-Dade", "Broward", "Palm Beach"},
        "wikimedia_query": "Miami skyline",
    },
    "Chicago": {
        "state": "IL",
        "counties": {"Cook", "DuPage", "Lake", "Kane", "McHenry", "Will", "Kendall"},
        "wikimedia_query": "Chicago skyline",
    },
    "Seattle": {
        "state": "WA",
        "counties": {"King", "Pierce", "Snohomish", "Kitsap"},
        "wikimedia_query": "Seattle skyline",
    },
}

STATE_ABBREVIATIONS = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

ROAD_KEYWORDS = {
    "Interstate": (" i-", " interstate", " interstate ", " ih "),
    "Highway": (" hwy", " highway", " us-", " sr-", " rte", " route "),
    "Local Road": (" street", " st ", " avenue", " ave", " boulevard", " blvd", " road", " rd "),
    "Ramp": (" ramp", " exit", " entrance"),
}

WEATHER_BUCKETS = {
    "Snow / Ice": ("snow", "sleet", "wintry", "ice", "icy", "blizzard"),
    "Rain / Storm": ("rain", "drizzle", "storm", "thunder", "shower", "hail"),
    "Fog / Smoke": ("fog", "smoke", "mist", "haze"),
    "Wind": ("wind", "gust"),
    "Clear / Fair": ("fair", "clear", "sunny"),
    "Clouds": ("cloud", "overcast"),
}

FRED_SERIES = {
    "wti_usd_per_barrel": {"series_id": "DCOILWTICO"},
    "gas_usd_per_gallon": {"series_id": "GASREGW"},
    "new_vehicle_cpi": {"series_id": "CUSR0000SETA01"},
    "used_vehicle_cpi": {"series_id": "CUSR0000SETA02"},
    "new_auto_loan_rate_pct": {"series_id": "TERMCBAUTO48NS"},
    "vmt_millions": {"series_id": "TRFVOLUSM227NFWA"},
}

def ensure_project_dirs(base_dir="."):
    base = Path(base_dir)
    directories = {
        "data": base / "data",
        "external": base / "data" / "external",
        "outputs": base / "outputs",
        "figures": base / "outputs" / "figures",
        "tables": base / "outputs" / "tables",
        "text": base / "outputs" / "text",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def fetch_wikimedia_image_url(query, timeout=10):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 1,
        "prop": "pageimages",
        "piprop": "original",
    }
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for payload in pages.values():
            return payload.get("original", {}).get("source")
    except Exception:
        return None
    return None


def build_pipeline_diagram(output_path):
    output = Path(output_path)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    x_positions = [0.03, 0.25, 0.47, 0.69, 0.88]
    labels = [
        "Raw Kaggle\nUS_Accidents",
        "Cleaning +\nSchema Audit",
        "Enrichment\nCalendar / Moon / Context",
        "Insight Layers\nEDA / Hotspots / NLP",
        "Models +\nStorytelling",
    ]
    colors = ["#173753", "#2f6690", "#3a7ca5", "#81c3d7", "#f4d35e"]
    for x, label, color in zip(x_positions, labels, colors):
        patch = FancyBboxPatch(
            (x, 0.3),
            0.16,
            0.35,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            fc=color,
            ec="#0b132b",
            lw=1.4,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + 0.08, 0.475, label, ha="center", va="center", fontsize=12, color="white", transform=ax.transAxes)
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        arrow = FancyArrowPatch(
            (left + 0.16, 0.475),
            (right, 0.475),
            arrowstyle="->",
            mutation_scale=18,
            lw=2,
            color="#0b132b",
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def export_dataframe(df, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    elif output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    else:
        raise ValueError(f"Unsupported export format for {output}.")
    return output


# ---------------------------------------------------------------------------
# GenAI integration (google-genai SDK)
# ---------------------------------------------------------------------------

