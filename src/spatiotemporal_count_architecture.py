from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

try:
    import h3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h3 = None

from modeling import train_count_forecasters


@dataclass
class CountArchitectureConfig:
    h3_resolution: int = 7
    fallback_grid_degrees: float = 0.25
    analysis_window_days: int = 180
    max_cells: int = 140
    min_cell_events: int = 100
    text_sample_limit: int = 50000
    random_state: int = 42
    temporal_splits: int = 3
    anomaly_contamination: float = 0.01
    neighbor_k: int = 2


@dataclass
class CountArchitectureResult:
    grid: pd.DataFrame
    prediction_frame: pd.DataFrame
    metrics: pd.DataFrame
    branch_metrics: pd.DataFrame
    daily_comparison: pd.DataFrame
    existing_daily_forecast: Optional[pd.DataFrame]
    feature_catalog: pd.DataFrame
    notes: List[str]


def _cyclical_encode(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * series.astype(float) / period
    return np.sin(angle), np.cos(angle)


def _safe_mode(series: pd.Series, default=None):
    clean = series.dropna()
    if clean.empty:
        return default
    mode = clean.mode()
    if mode.empty:
        return clean.iloc[0]
    return mode.iloc[0]


def _spatial_cell_key(row: pd.Series, config: CountArchitectureConfig) -> str:
    lat = row.get("Start_Lat")
    lng = row.get("Start_Lng")
    if pd.isna(lat) or pd.isna(lng):
        return "unknown"
    if h3 is not None:
        try:
            return h3.latlng_to_cell(float(lat), float(lng), config.h3_resolution)
        except Exception:
            pass
    lat_bin = int(np.floor((float(lat) + 90.0) / config.fallback_grid_degrees))
    lng_bin = int(np.floor((float(lng) + 180.0) / config.fallback_grid_degrees))
    return f"grid_{lat_bin}_{lng_bin}"


def _adverse_weather_score(frame: pd.DataFrame) -> pd.Series:
    visibility = pd.to_numeric(frame.get("Visibility(mi)"), errors="coerce")
    precipitation = pd.to_numeric(frame.get("Precipitation(in)"), errors="coerce")
    wind = pd.to_numeric(frame.get("Wind_Speed(mph)"), errors="coerce")
    temperature = pd.to_numeric(frame.get("Temperature(F)"), errors="coerce")
    weather_condition = frame.get("Weather_Condition", pd.Series(index=frame.index, dtype="object")).fillna("")

    vis_score = (1 - visibility.fillna(10).clip(lower=0, upper=10) / 10).clip(0, 1)
    precip_score = precipitation.fillna(0).clip(lower=0, upper=2) / 2
    wind_score = wind.fillna(0).clip(lower=0, upper=60) / 60
    ice_risk = np.exp(-0.5 * ((temperature.fillna(50) - 32) / 5) ** 2)
    fog_flag = weather_condition.astype(str).str.contains("fog|mist|haze|smoke", case=False, regex=True).astype(float)

    score = (3.0 * vis_score + 2.5 * precip_score + 1.5 * wind_score + 2.0 * ice_risk + 2.0 * fog_flag) / 11.0
    return score.clip(0, 1)


def _heat_index(temp_f: pd.Series, humidity: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_f, errors="coerce")
    rh = pd.to_numeric(humidity, errors="coerce")
    hi = temp.copy()
    mask = temp.ge(80) & rh.notna()
    hi.loc[mask] = (
        -42.379
        + 2.04901523 * temp.loc[mask]
        + 10.14333127 * rh.loc[mask]
        - 0.22475541 * temp.loc[mask] * rh.loc[mask]
        - 0.00683783 * temp.loc[mask] ** 2
        - 0.05481717 * rh.loc[mask] ** 2
        + 0.00122874 * temp.loc[mask] ** 2 * rh.loc[mask]
        + 0.00085282 * temp.loc[mask] * rh.loc[mask] ** 2
        - 0.00000199 * temp.loc[mask] ** 2 * rh.loc[mask] ** 2
    )
    return hi


def _wind_chill(temp_f: pd.Series, wind_mph: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_f, errors="coerce")
    wind = pd.to_numeric(wind_mph, errors="coerce")
    wc = temp.copy()
    mask = temp.le(50) & wind.ge(3)
    wc.loc[mask] = 35.74 + 0.6215 * temp.loc[mask] - 35.75 * (wind.loc[mask] ** 0.16) + 0.4275 * temp.loc[mask] * (
        wind.loc[mask] ** 0.16
    )
    return wc


def _days_to_holiday(timestamp: pd.Series) -> pd.Series:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    dates = pd.to_datetime(timestamp, errors="coerce")
    if dates.isna().all():
        return pd.Series(index=timestamp.index, dtype=float)
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=dates.min().normalize(), end=dates.max().normalize()).date
    holiday_set = set(holidays)

    def _distance(current_date):
        if pd.isna(current_date) or not holiday_set:
            return np.nan
        return min(abs((current_date - holiday).days) for holiday in holiday_set)

    return dates.dt.date.map(_distance)


def _build_text_topics(frame: pd.DataFrame, config: CountArchitectureConfig) -> pd.DataFrame:
    if "Description" not in frame.columns:
        return pd.DataFrame(index=frame.index)

    texts = frame["Description"].fillna("").astype(str)
    if len(frame) > config.text_sample_limit:
        sample = texts.sample(config.text_sample_limit, random_state=config.random_state)
    else:
        sample = texts

    if sample.str.strip().eq("").all():
        return pd.DataFrame(index=frame.index)

    vectorizer = TfidfVectorizer(max_features=256, stop_words="english", ngram_range=(1, 2), min_df=3)
    try:
        vectorizer.fit(sample)
        matrix = vectorizer.transform(texts)
        if matrix.shape[1] <= 2:
            return pd.DataFrame(index=frame.index)
        n_components = min(6, matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
        topics = svd.fit_transform(matrix)
        return pd.DataFrame(
            topics,
            index=frame.index,
            columns=[f"text_topic_{i}" for i in range(topics.shape[1])],
        )
    except Exception:
        return pd.DataFrame(index=frame.index)


def _prepare_row_level_frame(raw: pd.DataFrame, config: CountArchitectureConfig) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame.loc[frame["Start_Time"].notna() & frame["Start_Lat"].notna() & frame["Start_Lng"].notna()].copy()
    frame = frame.loc[frame["Start_Lat"].between(24, 50) & frame["Start_Lng"].between(-125, -66)].copy()
    frame["Start_Time"] = pd.to_datetime(frame["Start_Time"], errors="coerce")
    frame = frame.loc[frame["Start_Time"].notna()].copy()

    start = frame["Start_Time"].min().floor("D")
    cutoff = start + pd.Timedelta(days=config.analysis_window_days)
    windowed = frame.loc[frame["Start_Time"] >= cutoff - pd.Timedelta(days=config.analysis_window_days)].copy()
    if windowed.empty:
        windowed = frame.copy()

    windowed["hour_bucket"] = windowed["Start_Time"].dt.floor("h")
    windowed["hour"] = windowed["Start_Time"].dt.hour
    windowed["dow"] = windowed["Start_Time"].dt.dayofweek
    windowed["month"] = windowed["Start_Time"].dt.month
    windowed["day"] = windowed["Start_Time"].dt.day
    windowed["week"] = windowed["Start_Time"].dt.isocalendar().week.astype(int)
    windowed["is_weekend"] = (windowed["dow"] >= 5).astype(int)
    windowed["is_night"] = windowed.get("Sunrise_Sunset", pd.Series(index=windowed.index, dtype="object")).fillna("").eq("Night")

    windowed["sin_hour"], windowed["cos_hour"] = _cyclical_encode(windowed["hour"], 24)
    windowed["sin_dow"], windowed["cos_dow"] = _cyclical_encode(windowed["dow"], 7)
    windowed["sin_month"], windowed["cos_month"] = _cyclical_encode(windowed["month"], 12)

    windowed["commute_score"] = np.maximum(
        np.maximum(0, 1 - np.abs(windowed["hour"] - 8) / 2),
        np.maximum(0, 1 - np.abs(windowed["hour"] - 17) / 2),
    )
    windowed["days_to_holiday"] = _days_to_holiday(windowed["Start_Time"])
    windowed["is_holiday"] = windowed["days_to_holiday"].fillna(99).eq(0).astype(int)
    windowed["is_holiday_window"] = windowed["days_to_holiday"].fillna(99).le(1).astype(int)
    windowed["payday_score"] = np.exp(-0.5 * ((windowed["day"] - 1) / 2) ** 2) + np.exp(-0.5 * ((windowed["day"] - 15) / 2) ** 2)
    windowed["adverse_weather_score"] = _adverse_weather_score(windowed)
    windowed["heat_index"] = _heat_index(windowed.get("Temperature(F)"), windowed.get("Humidity(%)"))
    windowed["wind_chill"] = _wind_chill(windowed.get("Temperature(F)"), windowed.get("Wind_Speed(mph)"))
    windowed["hour_bucket"] = pd.to_datetime(windowed["hour_bucket"], errors="coerce")
    windowed["time_idx"] = ((windowed["hour_bucket"] - windowed["hour_bucket"].min()).dt.total_seconds() / 3600).astype(int)
    windowed["spatial_cell"] = windowed.apply(lambda row: _spatial_cell_key(row, config), axis=1)

    for column in [
        "Weather_Condition",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Timezone",
        "State",
        "City",
        "County",
        "weather_bucket",
        "road_scale",
        "temperature_bucket",
        "visibility_bucket",
        "wind_bucket",
        "precip_bucket",
    ]:
        if column not in windowed.columns:
            windowed[column] = np.nan

    description = windowed.get("Description", pd.Series(index=windowed.index, dtype="object")).fillna("").astype(str).str.lower()
    windowed["description_word_count"] = description.str.split().str.len().fillna(0)
    windowed["mentions_lane"] = description.str.contains("lane", regex=False)
    windowed["mentions_blocked"] = description.str.contains("blocked", regex=False)
    windowed["mentions_overturn"] = description.str.contains("overturn", regex=False)
    windowed["mentions_vehicle"] = description.str.contains("vehicle", regex=False)

    if "intersection_context" not in windowed.columns:
        windowed["intersection_context"] = False
    windowed["intersection_context"] = windowed["intersection_context"].fillna(False).astype(int)

    text_topics = _build_text_topics(windowed, config)
    if not text_topics.empty:
        windowed = pd.concat([windowed, text_topics], axis=1)

    return windowed.sort_values(["time_idx", "spatial_cell"]).copy()


def _build_weather_regime(frame: pd.DataFrame, config: CountArchitectureConfig) -> pd.Series:
    candidates = [
        col
        for col in ["Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "adverse_weather_score"]
        if col in frame.columns
    ]
    if not candidates:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)

    wframe = frame[candidates].copy()
    wframe = wframe.apply(pd.to_numeric, errors="coerce")
    filled = wframe.fillna(wframe.median())
    if len(filled) < 20:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)

    n_clusters = min(6, max(2, len(candidates) + 1))
    try:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10)),
            ]
        )
        return pd.Series(pipeline.fit_predict(filled), index=frame.index, dtype=int)
    except Exception:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)


def _cell_level_risk_surface(frame: pd.DataFrame) -> pd.Series:
    cell_stats = frame.groupby("spatial_cell", dropna=False).agg(
        lat=("Start_Lat", "mean"),
        lng=("Start_Lng", "mean"),
        count=("spatial_cell", "size"),
    )
    if len(cell_stats) < 5:
        return frame.groupby("spatial_cell")["count"].transform("mean")

    model = KNeighborsRegressor(n_neighbors=min(5, len(cell_stats)), weights="distance")
    model.fit(cell_stats[["lat", "lng"]].fillna(cell_stats[["lat", "lng"]].median()), cell_stats["count"].fillna(0))
    preds = model.predict(cell_stats[["lat", "lng"]].fillna(cell_stats[["lat", "lng"]].median()))
    lookup = pd.Series(preds, index=cell_stats.index)
    return frame["spatial_cell"].map(lookup).fillna(lookup.mean())


def _build_neighbor_map(cells: Sequence[str], config: CountArchitectureConfig) -> Dict[str, List[str]]:
    cells = list(dict.fromkeys(cells))
    if h3 is not None:
        neighbor_map: Dict[str, List[str]] = {}
        available = set(cells)
        for cell in cells:
            try:
                ring = set(h3.grid_disk(cell, config.neighbor_k))
                ring.discard(cell)
                neighbor_map[cell] = [nbr for nbr in ring if nbr in available]
            except Exception:
                neighbor_map[cell] = []
        return neighbor_map

    parsed = {}
    for cell in cells:
        if not str(cell).startswith("grid_"):
            continue
        try:
            _, lat_bin, lng_bin = str(cell).split("_")
            parsed[cell] = (int(lat_bin), int(lng_bin))
        except Exception:
            continue

    neighbor_map: Dict[str, List[str]] = {}
    for cell, (lat_bin, lng_bin) in parsed.items():
        neighbors = []
        for other, (other_lat, other_lng) in parsed.items():
            if other == cell:
                continue
            if abs(other_lat - lat_bin) <= config.neighbor_k and abs(other_lng - lng_bin) <= config.neighbor_k:
                neighbors.append(other)
        neighbor_map[cell] = neighbors
    for cell in cells:
        neighbor_map.setdefault(cell, [])
    return neighbor_map


def _add_neighbor_features(grid: pd.DataFrame, config: CountArchitectureConfig) -> pd.DataFrame:
    cells = grid["spatial_cell"].dropna().unique().tolist()
    if len(cells) < 2:
        grid["neighbor_mean_count"] = 0.0
        grid["neighbor_max_count"] = 0.0
        return grid

    neighbor_map = _build_neighbor_map(cells, config)
    pivot = grid.pivot_table(index="time_idx", columns="spatial_cell", values="count", fill_value=0)
    mean_frames = []
    max_frames = []
    for cell in cells:
        neighbors = neighbor_map.get(cell, [])
        if not neighbors:
            continue
        neighbor_values = pivot[neighbors]
        mean_frames.append(neighbor_values.mean(axis=1).rename(cell))
        max_frames.append(neighbor_values.max(axis=1).rename(cell))

    if mean_frames:
        mean_df = pd.concat(mean_frames, axis=1).stack().reset_index()
        mean_df.columns = ["time_idx", "spatial_cell", "neighbor_mean_count"]
        grid = grid.merge(mean_df, on=["time_idx", "spatial_cell"], how="left")
    else:
        grid["neighbor_mean_count"] = np.nan

    if max_frames:
        max_df = pd.concat(max_frames, axis=1).stack().reset_index()
        max_df.columns = ["time_idx", "spatial_cell", "neighbor_max_count"]
        grid = grid.merge(max_df, on=["time_idx", "spatial_cell"], how="left")
    else:
        grid["neighbor_max_count"] = np.nan

    return grid


def _build_cell_hour_grid(frame: pd.DataFrame, config: CountArchitectureConfig) -> pd.DataFrame:
    working = frame.copy()
    working["weather_regime"] = _build_weather_regime(working, config)
    working["spatial_risk_surface"] = _cell_level_risk_surface(working)

    cell_counts = working["spatial_cell"].value_counts()
    active_cells = cell_counts[cell_counts >= config.min_cell_events].head(config.max_cells).index.tolist()
    working = working.loc[working["spatial_cell"].isin(active_cells)].copy()
    if working.empty:
        raise ValueError("No active spatial cells available after filtering.")

    working["state_mode"] = working.groupby("spatial_cell")["State"].transform(lambda s: _safe_mode(s, "Unknown"))
    working["city_mode"] = working.groupby("spatial_cell")["City"].transform(lambda s: _safe_mode(s, "Unknown"))
    working["cell_lat_mean"] = working.groupby("spatial_cell")["Start_Lat"].transform("mean")
    working["cell_lng_mean"] = working.groupby("spatial_cell")["Start_Lng"].transform("mean")
    working["cell_baseline_raw"] = working.groupby("spatial_cell")["spatial_cell"].transform("size")

    aggregated = (
        working.groupby(["spatial_cell", "time_idx", "hour_bucket"], dropna=False)
        .agg(
            count=("ID", "size"),
            mean_severity=("Severity", "mean"),
            mean_duration=("duration_minutes", "mean"),
            mean_temp=("Temperature(F)", "mean"),
            mean_visibility=("Visibility(mi)", "mean"),
            mean_wind=("Wind_Speed(mph)", "mean"),
            mean_precip=("Precipitation(in)", "mean"),
            mean_adverse=("adverse_weather_score", "mean"),
            mean_heat_index=("heat_index", "mean"),
            mean_wind_chill=("wind_chill", "mean"),
            commute_score=("commute_score", "mean"),
            is_holiday=("is_holiday", "max"),
            is_holiday_window=("is_holiday_window", "max"),
            is_weekend=("is_weekend", "max"),
            weather_regime=("weather_regime", _safe_mode),
            spatial_risk_surface=("spatial_risk_surface", "mean"),
            state_mode=("state_mode", _safe_mode),
            city_mode=("city_mode", _safe_mode),
            cell_lat_mean=("cell_lat_mean", "mean"),
            cell_lng_mean=("cell_lng_mean", "mean"),
            cell_baseline_raw=("cell_baseline_raw", "first"),
            intersection_context=("intersection_context", "mean"),
            adverse_weather_score=("adverse_weather_score", "mean"),
            description_word_count=("description_word_count", "mean"),
            mentions_lane=("mentions_lane", "mean"),
            mentions_blocked=("mentions_blocked", "mean"),
            mentions_overturn=("mentions_overturn", "mean"),
            mentions_vehicle=("mentions_vehicle", "mean"),
        )
        .reset_index()
    )

    topic_columns = [column for column in working.columns if column.startswith("text_topic_")]
    if topic_columns:
        topic_agg = working.groupby(["spatial_cell", "time_idx", "hour_bucket"], dropna=False)[topic_columns].mean().reset_index()
        aggregated = aggregated.merge(topic_agg, on=["spatial_cell", "time_idx", "hour_bucket"], how="left")

    full_index = pd.MultiIndex.from_product(
        [active_cells, range(int(working["time_idx"].min()), int(working["time_idx"].max()) + 1)],
        names=["spatial_cell", "time_idx"],
    )
    full_grid = pd.DataFrame(index=full_index).reset_index()
    time_lookup = working[["time_idx", "hour_bucket"]].drop_duplicates().sort_values("time_idx")
    full_grid = full_grid.merge(time_lookup, on="time_idx", how="left")
    grid = full_grid.merge(aggregated, on=["spatial_cell", "time_idx", "hour_bucket"], how="left")
    grid["count"] = grid["count"].fillna(0).astype(int)
    grid["cell_baseline_raw"] = grid["cell_baseline_raw"].fillna(grid.groupby("spatial_cell")["count"].transform("sum").fillna(0))

    cell_static = (
        working.groupby("spatial_cell", dropna=False)
        .agg(
            state_mode=("state_mode", _safe_mode),
            city_mode=("city_mode", _safe_mode),
            cell_lat_mean=("cell_lat_mean", "mean"),
            cell_lng_mean=("cell_lng_mean", "mean"),
            cell_baseline_raw=("cell_baseline_raw", "first"),
            spatial_risk_surface=("spatial_risk_surface", "mean"),
            intersection_context=("intersection_context", "mean"),
            adverse_weather_score=("adverse_weather_score", "mean"),
        )
        .reset_index()
    )
    grid = grid.merge(cell_static, on="spatial_cell", suffixes=("", "_static"), how="left")
    for column in [
        "state_mode_static",
        "city_mode_static",
        "cell_lat_mean_static",
        "cell_lng_mean_static",
        "cell_baseline_raw_static",
        "spatial_risk_surface_static",
        "intersection_context_static",
        "adverse_weather_score_static",
    ]:
        if column in grid.columns:
            base = column.replace("_static", "")
            grid[base] = grid[base].fillna(grid[column])
            grid = grid.drop(columns=[column])

    grid["state_code"] = pd.factorize(grid["state_mode"].fillna("Unknown"))[0]
    grid["city_code"] = pd.factorize(grid["city_mode"].fillna("Unknown"))[0]
    grid["cell_code"] = pd.factorize(grid["spatial_cell"])[0]
    grid["state_cell_count"] = grid.groupby(["state_code", "time_idx"])["count"].transform("sum")
    grid["cell_share_of_state"] = grid["count"] / grid["state_cell_count"].replace(0, np.nan)
    grid["cell_share_of_state"] = grid["cell_share_of_state"].fillna(0.0)

    grid["cell_baseline"] = grid.groupby("spatial_cell")["count"].transform("mean")
    train_cut = int(grid["time_idx"].quantile(0.7))
    train_mask = grid["time_idx"] <= train_cut
    cell_train_mean = grid.loc[train_mask].groupby("spatial_cell")["count"].mean()
    global_mean = float(grid.loc[train_mask, "count"].mean()) if train_mask.any() else float(grid["count"].mean())
    grid["cell_baseline"] = grid["spatial_cell"].map(cell_train_mean).fillna(global_mean)

    grid = grid.sort_values(["spatial_cell", "time_idx"]).reset_index(drop=True)
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        grid[f"lag_{lag}h"] = grid.groupby("spatial_cell")["count"].shift(lag)
    roll_source = grid.groupby("spatial_cell")["count"]
    grid["roll_24h_mean"] = roll_source.transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
    grid["roll_24h_std"] = roll_source.transform(lambda x: x.shift(1).rolling(24, min_periods=1).std())
    grid["roll_7d_mean"] = roll_source.transform(lambda x: x.shift(1).rolling(168, min_periods=1).mean())
    grid["roll_7d_std"] = roll_source.transform(lambda x: x.shift(1).rolling(168, min_periods=1).std())
    grid["roll_30d_mean"] = roll_source.transform(lambda x: x.shift(1).rolling(720, min_periods=1).mean())
    grid["ema_03"] = roll_source.transform(lambda x: x.shift(1).ewm(alpha=0.3).mean())
    grid = _add_neighbor_features(grid, config)

    grid["residual"] = grid["count"] - grid["roll_7d_mean"].fillna(grid["cell_baseline"])
    grid["rush_x_weather"] = grid["commute_score"].fillna(0) * grid["adverse_weather_score"].fillna(0)
    night_flag = grid["is_night"] if "is_night" in grid.columns else pd.Series(0, index=grid.index)
    grid["night_x_lowvis"] = night_flag.fillna(0).astype(float) * (1 - pd.to_numeric(grid["mean_visibility"], errors="coerce").fillna(10).clip(upper=10) / 10)
    grid["weekend_x_holiday"] = grid["is_weekend"].fillna(0).astype(float) * grid["is_holiday"].fillna(0).astype(float)
    grid["centrality_x_precip"] = grid["intersection_context"].fillna(0).astype(float) * grid["mean_precip"].fillna(0).astype(float)
    grid["rate_x_weather_delta"] = grid["cell_baseline"].fillna(0) * grid["adverse_weather_score"].fillna(0)
    grid["payday_x_dom"] = grid.get("payday_score", pd.Series(0, index=grid.index))
    grid["target_adj"] = grid["count"].clip(lower=0).astype(float)
    return grid


def _branch_feature_sets(grid: pd.DataFrame) -> Dict[str, List[str]]:
    topic_columns = [column for column in grid.columns if column.startswith("text_topic_")]
    return {
        "temporal": [
            "sin_hour",
            "cos_hour",
            "sin_dow",
            "cos_dow",
            "sin_month",
            "cos_month",
            "commute_score",
            "is_holiday",
            "is_holiday_window",
            "is_weekend",
            "payday_score",
            "adverse_weather_score",
            "mean_temp",
            "mean_precip",
            "mean_wind",
            "mean_visibility",
            "rush_x_weather",
            "night_x_lowvis",
        ]
        + [f"lag_{lag}h" for lag in [1, 2, 3, 6, 12, 24, 48, 168]]
        + ["roll_24h_mean", "roll_24h_std", "roll_7d_mean", "roll_7d_std", "roll_30d_mean", "ema_03"],
        "spatial": [
            "cell_baseline",
            "spatial_risk_surface",
            "neighbor_mean_count",
            "neighbor_max_count",
            "state_code",
            "city_code",
            "cell_code",
            "cell_lat_mean",
            "cell_lng_mean",
            "intersection_context",
            "centrality_x_precip",
            "cell_share_of_state",
        ],
        "tabular": [
            "sin_hour",
            "cos_hour",
            "sin_dow",
            "cos_dow",
            "sin_month",
            "cos_month",
            "commute_score",
            "is_holiday",
            "is_holiday_window",
            "is_weekend",
            "payday_score",
            "adverse_weather_score",
            "mean_severity",
            "mean_duration",
            "mean_temp",
            "mean_visibility",
            "mean_wind",
            "mean_precip",
            "mean_heat_index",
            "mean_wind_chill",
            "weather_regime",
            "cell_baseline",
            "spatial_risk_surface",
            "neighbor_mean_count",
            "neighbor_max_count",
            "intersection_context",
            "rush_x_weather",
            "night_x_lowvis",
            "weekend_x_holiday",
            "centrality_x_precip",
            "rate_x_weather_delta",
            "description_word_count",
            "mentions_lane",
            "mentions_blocked",
            "mentions_overturn",
            "mentions_vehicle",
            "state_code",
            "city_code",
            "cell_code",
            "cell_lat_mean",
            "cell_lng_mean",
            "cell_share_of_state",
        ] + topic_columns,
    }


def _clean_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    available = [column for column in features if column in frame.columns]
    cleaned = frame[available].copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.replace([np.inf, -np.inf], np.nan)


def _fit_branch_model(name: str, grid: pd.DataFrame, feature_names: Sequence[str], train_mask: pd.Series, random_state: int):
    x_train = _clean_features(grid.loc[train_mask], feature_names).fillna(0)
    y_train = grid.loc[train_mask, "target_adj"].astype(float)

    if name == "spatial":
        model = RandomForestRegressor(n_estimators=250, min_samples_leaf=4, n_jobs=-1, random_state=random_state)
    else:
        model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_depth=8 if name == "temporal" else 10,
            max_iter=250 if name == "temporal" else 300,
            min_samples_leaf=20,
            random_state=random_state,
        )

    model.fit(x_train, y_train)
    return model


def _predict_branch(model, grid: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    x = _clean_features(grid, feature_names).fillna(0)
    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0, None)


def _daily_metrics(frame: pd.DataFrame, actual_col: str = "actual", pred_col: str = "pred") -> Dict[str, float]:
    actual = np.asarray(frame[actual_col], dtype=float)
    pred = np.asarray(frame[pred_col], dtype=float)
    denom = np.maximum(np.abs(actual) + np.abs(pred), 1e-6)
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "smape": float(np.mean(2 * np.abs(pred - actual) / denom)),
    }


def _reconcile_by_state(predictions: pd.DataFrame, calibration_frame: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    state_scale = calibration_frame.groupby("state_mode")[["count", "stack_pred"]].sum()
    state_scale["scale"] = state_scale["count"] / state_scale["stack_pred"].replace(0, np.nan)
    state_scale["scale"] = state_scale["scale"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 2.0)
    overall_scale = calibration_frame["count"].sum() / max(calibration_frame["stack_pred"].sum(), 1e-6)
    overall_scale = float(np.clip(overall_scale, 0.5, 2.0))
    output["state_scale"] = output["state_mode"].map(state_scale["scale"]).fillna(overall_scale)
    output["reconciled_pred"] = np.clip(output["stack_pred"] * output["state_scale"], 0, None)
    return output


def train_spatiotemporal_count_ensemble(
    enriched: pd.DataFrame,
    daily_context: Optional[pd.DataFrame] = None,
    config: Optional[CountArchitectureConfig] = None,
) -> CountArchitectureResult:
    config = config or CountArchitectureConfig()
    row_level = _prepare_row_level_frame(enriched, config)
    grid = _build_cell_hour_grid(row_level, config)
    grid = grid.sort_values(["time_idx", "spatial_cell"]).reset_index(drop=True)

    unique_times = np.array(sorted(grid["time_idx"].dropna().unique()))
    if len(unique_times) < 20:
        raise ValueError("Not enough hourly history for the spatiotemporal ensemble.")

    train_end = unique_times[int(len(unique_times) * 0.6)]
    meta_end = unique_times[int(len(unique_times) * 0.8)]
    train_mask = grid["time_idx"] <= train_end
    meta_mask = (grid["time_idx"] > train_end) & (grid["time_idx"] <= meta_end)
    test_mask = grid["time_idx"] > meta_end

    grid["anomaly_score"] = np.nan
    grid["is_anomaly"] = 0
    anomaly_features = ["count", "roll_7d_mean", "roll_7d_std", "mean_adverse", "neighbor_mean_count", "cell_baseline"]
    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=config.anomaly_contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    anomaly_train = grid.loc[train_mask, anomaly_features].fillna(0)
    if len(anomaly_train) >= 30:
        anomaly_model.fit(anomaly_train)
        grid.loc[:, "anomaly_score"] = anomaly_model.score_samples(grid[anomaly_features].fillna(0))
        threshold = getattr(anomaly_model, "offset_", np.percentile(grid.loc[train_mask, "anomaly_score"].dropna(), 5))
        grid["is_anomaly"] = (grid["anomaly_score"] < threshold).astype(int)
        grid["target_adj"] = np.where(
            grid["is_anomaly"].eq(1),
            grid["roll_7d_mean"].fillna(grid["cell_baseline"]),
            grid["count"],
        ).astype(float)
    else:
        grid["target_adj"] = grid["count"].astype(float)

    feature_sets = _branch_feature_sets(grid)

    branch_oof = {name: np.full(len(grid), np.nan) for name in feature_sets}
    time_splitter = TimeSeriesSplit(n_splits=config.temporal_splits)
    for train_time_idx, val_time_idx in time_splitter.split(unique_times):
        train_times = unique_times[train_time_idx]
        val_times = unique_times[val_time_idx]
        fold_train_mask = grid["time_idx"].isin(train_times)
        fold_val_mask = grid["time_idx"].isin(val_times)
        for branch_name, features in feature_sets.items():
            model = _fit_branch_model(branch_name, grid, features, fold_train_mask, config.random_state)
            branch_oof[branch_name][fold_val_mask.to_numpy()] = _predict_branch(model, grid.loc[fold_val_mask], features)

    oof_rows = []
    for branch_name, preds in branch_oof.items():
        valid = ~np.isnan(preds)
        if valid.sum() == 0:
            continue
        metrics = _daily_metrics(pd.DataFrame({"actual": grid.loc[valid, "target_adj"], "pred": preds[valid]}))
        oof_rows.append({"branch": branch_name, **metrics})
    branch_metrics = pd.DataFrame(oof_rows).sort_values("rmse") if oof_rows else pd.DataFrame(columns=["branch", "mae", "rmse", "smape"])

    x_meta = pd.DataFrame({
        "temporal_pred": branch_oof["temporal"],
        "spatial_pred": branch_oof["spatial"],
        "tabular_pred": branch_oof["tabular"],
        "cell_baseline": grid["cell_baseline"],
        "roll_7d_mean": grid["roll_7d_mean"],
    })
    meta_valid = x_meta.notna().all(axis=1)
    meta_model = Ridge(alpha=5.0, positive=True)
    meta_model.fit(x_meta.loc[meta_valid], grid.loc[meta_valid, "target_adj"])

    final_branch_models = {
        name: _fit_branch_model(name, grid, features, train_mask | meta_mask, config.random_state)
        for name, features in feature_sets.items()
    }

    prediction_frame = grid.loc[test_mask, ["spatial_cell", "time_idx", "hour_bucket", "state_mode", "count", "target_adj"]].copy()
    prediction_frame["temporal_pred"] = _predict_branch(final_branch_models["temporal"], grid.loc[test_mask], feature_sets["temporal"])
    prediction_frame["spatial_pred"] = _predict_branch(final_branch_models["spatial"], grid.loc[test_mask], feature_sets["spatial"])
    prediction_frame["tabular_pred"] = _predict_branch(final_branch_models["tabular"], grid.loc[test_mask], feature_sets["tabular"])
    meta_input = prediction_frame[["temporal_pred", "spatial_pred", "tabular_pred"]].copy()
    meta_input["cell_baseline"] = grid.loc[test_mask, "cell_baseline"].to_numpy()
    meta_input["roll_7d_mean"] = grid.loc[test_mask, "roll_7d_mean"].fillna(0).to_numpy()
    prediction_frame["stack_pred"] = np.clip(meta_model.predict(meta_input.fillna(0)), 0, None)

    calibration_frame = grid.loc[meta_mask, ["count", "state_mode"]].copy()
    calibration_frame["stack_pred"] = np.clip(
        meta_model.predict(
            x_meta.loc[meta_mask, ["temporal_pred", "spatial_pred", "tabular_pred", "cell_baseline", "roll_7d_mean"]].fillna(0)
        ),
        0,
        None,
    )
    reconciled = _reconcile_by_state(prediction_frame, calibration_frame)
    prediction_frame["reconciled_pred"] = reconciled["reconciled_pred"].to_numpy()

    metrics = pd.DataFrame(
        [
            {"model": "stacked_ensemble", **_daily_metrics(prediction_frame.rename(columns={"target_adj": "actual", "reconciled_pred": "pred"}), actual_col="actual", pred_col="pred")},
            {"model": "stacked_raw", **_daily_metrics(prediction_frame.rename(columns={"target_adj": "actual", "stack_pred": "pred"}), actual_col="actual", pred_col="pred")},
            {"model": "temporal_branch", **_daily_metrics(prediction_frame.rename(columns={"target_adj": "actual", "temporal_pred": "pred"}), actual_col="actual", pred_col="pred")},
            {"model": "spatial_branch", **_daily_metrics(prediction_frame.rename(columns={"target_adj": "actual", "spatial_pred": "pred"}), actual_col="actual", pred_col="pred")},
            {"model": "tabular_branch", **_daily_metrics(prediction_frame.rename(columns={"target_adj": "actual", "tabular_pred": "pred"}), actual_col="actual", pred_col="pred")},
        ]
    ).sort_values("rmse")

    daily = prediction_frame.assign(local_date=pd.to_datetime(prediction_frame["hour_bucket"]).dt.floor("D")).groupby("local_date", dropna=False).agg(
        actual=("target_adj", "sum"),
        pred=("reconciled_pred", "sum"),
    ).reset_index()
    daily_comparison = pd.DataFrame([{"model": "spatiotemporal_daily_aggregate", **_daily_metrics(daily)}])

    existing_daily_forecast = None
    if daily_context is not None and not daily_context.empty:
        existing_metrics, existing_results = train_count_forecasters(daily_context)
        best_existing = existing_metrics.sort_values("rmse").iloc[0].to_dict()
        existing_daily_forecast = existing_metrics.copy()
        comparison_rows = [
            {"model": "new_spatiotemporal_daily", **daily_comparison.iloc[0].to_dict()},
            {"model": best_existing["model"], **{key: best_existing.get(key) for key in ["mae", "rmse", "smape"]}},
        ]
        daily_comparison = pd.DataFrame(comparison_rows).sort_values("rmse")

    feature_catalog = pd.DataFrame(
        [
            {"family": "temporal", "features": len(feature_sets["temporal"])},
            {"family": "spatial", "features": len(feature_sets["spatial"])},
            {"family": "tabular", "features": len(feature_sets["tabular"])},
            {"family": "text_topics", "features": len([c for c in grid.columns if c.startswith("text_topic_")])},
            {"family": "neighbors", "features": 2},
            {"family": "anomaly_filter", "features": len(anomaly_features)},
            {"family": "existing_daily_forecaster", "features": int(existing_daily_forecast is not None)},
        ]
    )

    notes = [
        f"Active cells retained: {grid['spatial_cell'].nunique()}",
        f"Analysis window days: {config.analysis_window_days}",
        f"H3 available: {h3 is not None}",
        "The requested TFT/ST-GNN/LightGBM stack is represented here with current-repo equivalents that fit the installed dependency set.",
        "If h3 is installed later, the same code will switch from grid bins to H3 cells automatically.",
    ]

    return CountArchitectureResult(
        grid=grid,
        prediction_frame=prediction_frame,
        metrics=metrics,
        branch_metrics=branch_metrics,
        daily_comparison=daily_comparison,
        existing_daily_forecast=existing_daily_forecast,
        feature_catalog=feature_catalog,
        notes=notes,
    )


def summarize_count_architecture(result: CountArchitectureResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"section": "branch_metrics", "rows": len(result.branch_metrics)},
            {"section": "metrics", "rows": len(result.metrics)},
            {"section": "daily_comparison", "rows": len(result.daily_comparison)},
            {"section": "prediction_frame", "rows": len(result.prediction_frame)},
        ]
    )
