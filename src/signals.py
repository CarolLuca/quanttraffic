
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

def compute_special_signal_table(daily_context: pd.DataFrame) -> pd.DataFrame:
    stable = _compute_signal_stability_table(daily_context)
    order_cols = [
        "signal",
        "state",
        "days_in_group",
        "mean_daily_accidents",
        "lift_vs_baseline_pct",
        "full_years_considered",
        "yearly_sign_stability",
        "robustness_label",
        "passes_discussion_filter",
    ]
    existing_cols = [column for column in order_cols if column in stable.columns]
    return stable[existing_cols].copy()


def _filtered_daily_context(daily_context: pd.DataFrame) -> pd.DataFrame:
    filtered = daily_context.copy()
    if "low_coverage_day" in filtered.columns:
        filtered = filtered.loc[~filtered["low_coverage_day"]].copy()
    return filtered


def _full_years_for_stability(daily_context: pd.DataFrame) -> list[int]:
    years = sorted(pd.Series(daily_context["local_date"]).dt.year.dropna().unique().tolist())
    if len(years) <= 2:
        return years
    return years[1:-1]


def _signal_stability_record(
    frame: pd.DataFrame,
    signal: str,
    state_label: str,
    baseline: float,
    full_years: list[int],
    baseline_daily_context: pd.DataFrame,
) -> dict:
    mean_value = frame["accident_count"].mean()
    lift = ((mean_value - baseline) / baseline) * 100 if baseline else np.nan
    overall_sign = np.sign(lift) if not pd.isna(lift) else 0.0
    matched_years = 0
    considered_years = 0

    for year in full_years:
        year_frame = frame.loc[frame["year"] == year]
        if year_frame.empty:
            continue
        year_baseline_frame = baseline_daily_context.loc[baseline_daily_context["year"] == year]
        if year_baseline_frame.empty:
            continue
        year_baseline = year_baseline_frame["accident_count"].mean()
        if year_baseline == 0:
            continue
        year_lift = ((year_frame["accident_count"].mean() - year_baseline) / year_baseline) * 100
        year_sign = np.sign(year_lift)
        considered_years += 1
        if overall_sign == 0 or year_sign == overall_sign:
            matched_years += 1

    yearly_sign_stability = matched_years / considered_years if considered_years else np.nan
    passes_discussion_filter = bool(frame.shape[0] >= 25 and (yearly_sign_stability >= 0.60 if not pd.isna(yearly_sign_stability) else False))
    robustness_label = "discussion_ready" if passes_discussion_filter else "cautionary"
    return {
        "signal": signal,
        "state": state_label,
        "days_in_group": int(frame.shape[0]),
        "mean_daily_accidents": float(mean_value),
        "lift_vs_baseline_pct": float(lift),
        "full_years_considered": int(considered_years),
        "yearly_sign_stability": float(yearly_sign_stability) if not pd.isna(yearly_sign_stability) else np.nan,
        "robustness_label": robustness_label,
        "passes_discussion_filter": passes_discussion_filter,
    }


def _compute_signal_stability_table(daily_context: pd.DataFrame) -> pd.DataFrame:
    filtered = _filtered_daily_context(daily_context)
    baseline = filtered["accident_count"].mean()
    full_years = _full_years_for_stability(filtered)

    records = []
    signal_columns = [
        "holiday_window",
        "long_weekend",
        "dst_transition",
        "half_moon",
        "full_moon",
    ]
    for signal in signal_columns:
        for signal_state, group in filtered.groupby(signal, dropna=False):
            if pd.isna(signal_state):
                continue
            records.append(
                _signal_stability_record(
                    group,
                    signal,
                    str(signal_state),
                    baseline,
                    full_years,
                    filtered,
                )
            )

    moon_band_frame = filtered.assign(
        moon_band=pd.cut(
            filtered["moon_illumination_pct"],
            bins=[-0.1, 10, 35, 65, 90, 100.1],
            labels=["Very Dark", "Low", "Half-lit", "Bright", "Very Bright"],
        )
    )
    for band, group in moon_band_frame.groupby("moon_band", dropna=False):
        if pd.isna(band):
            continue
        records.append(
            _signal_stability_record(
                group,
                "moon_illumination_band",
                str(band),
                baseline,
                full_years,
                filtered,
            )
        )

    return pd.DataFrame(records).sort_values(
        ["passes_discussion_filter", "lift_vs_baseline_pct", "days_in_group"],
        ascending=[False, False, False],
    )


