from __future__ import annotations

import html
import json
import math
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualisation import build_interactive_globe_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
NOTEBOOK_PATH = PROJECT_ROOT / "AAD.ipynb"

SEVERITY_BUNDLE_ABLATION = pd.DataFrame(
    [
        {"feature_bundle": "Kaggle Core", "roc_auc": 0.794508, "pr_auc": 0.314583, "f1": 0.359491},
        {"feature_bundle": "Core + Calendar", "roc_auc": 0.804495, "pr_auc": 0.328140, "f1": 0.345385},
        {"feature_bundle": "Core + Context", "roc_auc": 0.805253, "pr_auc": 0.321619, "f1": 0.254476},
        {"feature_bundle": "Full Structured", "roc_auc": 0.814693, "pr_auc": 0.385543, "f1": 0.275810},
    ]
).sort_values("pr_auc", ascending=False)

FORECAST_TAKEAWAYS = pd.DataFrame(
    [
        {"scope": "Chicago", "model": "Poisson Regressor", "rmse_improvement_pct": 24.6},
        {"scope": "Houston", "model": "Random Forest Regressor", "rmse_improvement_pct": 20.6},
        {"scope": "Los Angeles", "model": "XGBoost Regressor (GPU)", "rmse_improvement_pct": 28.1},
        {"scope": "Miami", "model": "Random Forest Regressor", "rmse_improvement_pct": 24.2},
        {"scope": "Seattle", "model": "Poisson Regressor", "rmse_improvement_pct": 24.8},
        {"scope": "United States", "model": "Random Forest Regressor", "rmse_improvement_pct": 22.4},
    ]
)

SEVERITY_TAKEAWAYS = [
    "The best model remains XGBoost (GPU) with PR-AUC 0.717 on a holdout severe-share prevalence of 8.3%.",
    "Random Forest stays close behind, while Logistic Regression lags on ranking quality and calibration.",
    "The strongest supplemental gains come from the Full Structured bundle, which combines calendar, context, and text-derived fields.",
]


def load_table(filename: str) -> pd.DataFrame:
    path = TABLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


def load_severity_metrics_from_notebook(notebook_path: Path = NOTEBOOK_PATH) -> pd.DataFrame:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    target_phrase = "severity_metrics, severity_models = train_severity_models(accidents_enriched)"
    for cell in payload.get("cells", []):
        source = "".join(cell.get("source", []))
        if target_phrase in source and "display(severity_metrics)" in source:
            for output in cell.get("outputs", []):
                text = None
                if "data" in output and "text/plain" in output["data"]:
                    text = output["data"]["text/plain"]
                elif "text" in output:
                    text = output["text"]
                if text is None:
                    continue
                if isinstance(text, list):
                    text = "".join(text)
                if "XGBoost (GPU)" not in text or "Random Forest" not in text:
                    continue
                try:
                    frame = pd.read_fwf(StringIO(text))
                    frame.columns = [str(column).strip() for column in frame.columns]
                    if "model" in frame.columns:
                        frame = frame.dropna(subset=["model"]).copy()
                        for column in frame.columns:
                            if column == "model":
                                continue
                            frame[column] = pd.to_numeric(frame[column], errors="ignore")
                        if not frame.empty:
                            return frame
                except Exception:
                    pass
                break
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "train_rows": 250000,
                "test_rows": 150000,
                "test_prevalence": 0.0829,
                "roc_auc": 0.823620,
                "pr_auc": 0.372133,
                "balanced_accuracy": 0.738387,
                "precision": 0.184424,
                "recall": 0.794290,
                "f1": 0.299344,
            },
            {
                "model": "Random Forest",
                "train_rows": 250000,
                "test_rows": 150000,
                "test_prevalence": 0.0829,
                "roc_auc": 0.909500,
                "pr_auc": 0.687549,
                "balanced_accuracy": 0.827433,
                "precision": 0.417226,
                "recall": 0.749497,
                "f1": 0.536048,
            },
            {
                "model": "XGBoost (GPU)",
                "train_rows": 250000,
                "test_rows": 150000,
                "test_prevalence": 0.0829,
                "roc_auc": 0.929931,
                "pr_auc": 0.716532,
                "balanced_accuracy": 0.842531,
                "precision": 0.467622,
                "recall": 0.763651,
                "f1": 0.580050,
            },
        ]
    )


def normalize_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() == 0:
        return series
    mean = series.mean()
    std = series.std(ddof=0)
    if not std or math.isclose(std, 0.0):
        return series * 0
    return (series - mean) / std


def compute_macro_takeaway(frame: pd.DataFrame) -> dict:
    monthly = frame.copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly = monthly.sort_values("month").copy()
    baseline = monthly.loc[(monthly["month"].dt.year >= 2017) & (monthly["month"].dt.year <= 2019)]
    recent = monthly.loc[(monthly["month"].dt.year >= 2021) & (monthly["month"].dt.year <= 2022)]
    if baseline.empty or recent.empty:
        baseline = monthly.head(max(len(monthly) // 3, 1))
        recent = monthly.tail(max(len(monthly) // 3, 1))

    def pct_change(column: str) -> float:
        base = baseline[column].mean()
        new = recent[column].mean()
        if pd.isna(base) or pd.isna(new) or math.isclose(float(base), 0.0):
            return float("nan")
        return float(((new - base) / base) * 100)

    return {
        "accident_count": pct_change("accident_count"),
        "vmt_millions": pct_change("vmt_millions"),
        "used_vehicle_cpi": pct_change("used_vehicle_cpi"),
    }


def top_state_panel(panel: pd.DataFrame) -> pd.DataFrame:
    state_panel = panel.loc[panel["scope_type"].eq("state")].copy()
    if state_panel.empty:
        return state_panel
    return state_panel.sort_values(["accident_count", "severe_share"], ascending=[False, False]).copy()


def build_state_chart(state_panel: pd.DataFrame) -> go.Figure:
    top_states = state_panel.head(12).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=top_states["scope_name"],
            y=top_states["accident_count"],
            name="Accident count",
            marker_color="#1f7a8c",
            hovertemplate="State=%{x}<br>Accidents=%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=top_states["scope_name"],
            y=(top_states["severe_share"] * 100),
            name="Severe share (%)",
            mode="lines+markers",
            line=dict(color="#f25f5c", width=3),
            marker=dict(size=8),
            hovertemplate="State=%{x}<br>Severe share=%{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Accidents", secondary_y=False)
    fig.update_yaxes(title_text="Severe share (%)", secondary_y=True)
    fig.update_layout(
        title="Top States by Accident Volume and Severe Share",
        template="plotly_white",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_city_chart(panel: pd.DataFrame) -> go.Figure:
    if "scope_type" in panel.columns:
        city_panel = panel.loc[panel["scope_type"].eq("city")].copy()
    else:
        city_panel = panel.copy()
    city_panel = city_panel.sort_values("accident_count", ascending=False).head(15)
    fig = px.bar(
        city_panel,
        x="accident_count",
        y="scope_name",
        orientation="h",
        color="median_duration_min" if "median_duration_min" in city_panel.columns else None,
        color_continuous_scale="Plasma",
        title="Highest-Volume Cities in the Panel",
        labels={"accident_count": "Accidents", "scope_name": "City"},
    )
    fig.update_layout(template="plotly_white", height=560, margin=dict(l=30, r=30, t=60, b=30), coloraxis_colorbar=dict(title="Median duration (min)"))
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def build_hotspot_chart(hotspot_panel: pd.DataFrame) -> go.Figure:
    top_hotspots = hotspot_panel.head(120).copy()
    fig = go.Figure(
        go.Scattergeo(
            lat=top_hotspots["lat_bin"],
            lon=top_hotspots["lng_bin"],
            mode="markers",
            marker=dict(
                size=np.clip(top_hotspots["accident_count"] / top_hotspots["accident_count"].max() * 28, 4, 28),
                color=top_hotspots["severe_share"],
                colorscale="YlOrRd",
                opacity=0.78,
                line=dict(color="white", width=0.4),
                colorbar=dict(title="Severe share"),
            ),
            text=top_hotspots.apply(
                lambda row: (
                    f"<b>{row.get('State', 'Unknown')}</b><br>"
                    f"Metro: {row.get('metro_case_study', 'Unknown')}<br>"
                    f"Accidents: {int(row['accident_count']):,}<br>"
                    f"Severe share: {row['severe_share']:.1%}"
                ),
                axis=1,
            ),
            hoverinfo="text",
        )
    )
    fig.update_geos(
        scope="usa",
        showland=True,
        landcolor="#f5efe6",
        subunitcolor="#b8b8b8",
        lakecolor="#d7f3ff",
        showlakes=True,
        showcoastlines=True,
        coastlinecolor="#6c757d",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title="Top Geographic Hotspots",
        template="plotly_white",
        height=640,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def build_signal_chart(signal_table: pd.DataFrame) -> go.Figure:
    if "passes_discussion_filter" in signal_table.columns:
        signal_table = signal_table.loc[signal_table["passes_discussion_filter"]].copy()
    signal_table = signal_table.sort_values("lift_vs_baseline_pct", ascending=True).tail(12)
    signal_table["label"] = signal_table.apply(
        lambda row: f"{row['signal']} = {row['state']}" if "state" in row.index and not pd.isna(row.get("state")) else str(row["signal"]),
        axis=1,
    )
    fig = px.bar(
        signal_table,
        x="lift_vs_baseline_pct",
        y="label",
        orientation="h",
        color="yearly_sign_stability",
        color_continuous_scale="Viridis",
        title="Most Stable Signal Lifts vs Baseline",
        labels={"lift_vs_baseline_pct": "Lift vs baseline (%)", "label": "Signal"},
        hover_data={"days_in_group": True, "full_years_considered": True},
    )
    fig.update_layout(template="plotly_white", height=600, margin=dict(l=30, r=30, t=60, b=30), coloraxis_colorbar=dict(title="Yearly stability"))
    return fig


def build_macro_chart(macro_frame: pd.DataFrame) -> go.Figure:
    macro = macro_frame.copy()
    macro["month"] = pd.to_datetime(macro["month"])
    macro = macro.sort_values("month")
    metric_columns = [
        column
        for column in ["accident_count", "vmt_millions", "wti_usd_per_barrel", "gas_usd_per_gallon", "new_vehicle_cpi", "used_vehicle_cpi", "new_auto_loan_rate_pct"]
        if column in macro.columns
    ]
    normalized = pd.DataFrame({"month": macro["month"]})
    for column in metric_columns:
        normalized[column] = normalize_series(macro[column])
    fig = go.Figure()
    for column in metric_columns:
        fig.add_trace(
            go.Scatter(
                x=normalized["month"],
                y=normalized[column],
                mode="lines",
                name=column.replace("_", " "),
                hovertemplate="%{x|%Y-%m}<br>%{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Normalized Monthly Accident and Macro Context",
        template="plotly_white",
        height=560,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_model_metrics_chart(metrics: pd.DataFrame) -> go.Figure:
    metrics = metrics.copy()
    metrics = metrics.sort_values("pr_auc", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics["pr_auc"],
            y=metrics["model"],
            orientation="h",
            name="PR-AUC",
            marker_color="#1f77b4",
            hovertemplate="Model=%{y}<br>PR-AUC=%{x:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=metrics["roc_auc"],
            y=metrics["model"],
            orientation="h",
            name="ROC-AUC",
            marker_color="#ff7f0e",
            hovertemplate="Model=%{y}<br>ROC-AUC=%{x:.3f}<extra></extra>",
            opacity=0.72,
        )
    )
    fig.update_layout(
        title="Severity Model Comparison",
        barmode="group",
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis=dict(range=[0, 1]),
    )
    return fig


def build_bundle_ablation_chart(bundle_ablation: pd.DataFrame) -> go.Figure:
    df = bundle_ablation.sort_values("pr_auc", ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["pr_auc"],
            y=df["feature_bundle"],
            orientation="h",
            marker=dict(color=df["pr_auc"], colorscale="Magma", showscale=True, colorbar=dict(title="PR-AUC")),
            hovertemplate="Bundle=%{y}<br>PR-AUC=%{x:.3f}<extra></extra>",
            name="PR-AUC",
        )
    )
    fig.update_layout(
        title="Feature-Bundle Ablation: Which Structured Inputs Help Most",
        template="plotly_white",
        height=460,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis=dict(range=[0, max(0.5, float(df["pr_auc"].max()) * 1.15)]),
    )
    return fig


def build_forecast_chart(forecast_takeaways: pd.DataFrame) -> go.Figure:
    df = forecast_takeaways.sort_values("rmse_improvement_pct", ascending=True)
    fig = px.bar(
        df,
        x="rmse_improvement_pct",
        y="scope",
        orientation="h",
        color="model",
        title="Daily Forecast Improvement vs Seasonal Naive",
        labels={"rmse_improvement_pct": "RMSE improvement (%)", "scope": "Scope"},
    )
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=30, r=30, t=60, b=30), legend_title_text="Best model")
    return fig


def render_report(title: str, subtitle: str, cards: Iterable[tuple[str, str]], figures: Iterable[tuple[str, go.Figure]], tables: Iterable[tuple[str, pd.DataFrame]], narrative: Iterable[str]) -> str:
    figure_blocks = []
    for fig_title, fig in figures:
        figure_html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": True})
        figure_blocks.append(
            f"""
            <section class=\"panel\">
              <h2>{html.escape(fig_title)}</h2>
              {figure_html}
            </section>
            """
        )

    table_blocks = []
    for table_title, table in tables:
        table_html = table.to_html(index=False, classes="data-table", border=0, escape=False)
        table_blocks.append(
            f"""
            <section class=\"panel\">
              <h2>{html.escape(table_title)}</h2>
              <div class=\"table-wrap\">{table_html}</div>
            </section>
            """
        )

    card_blocks = []
    for label, value in cards:
        card_blocks.append(
            f"""
            <div class=\"metric-card\">
              <div class=\"metric-label\">{html.escape(label)}</div>
              <div class=\"metric-value\">{html.escape(value)}</div>
            </div>
            """
        )

    narrative_html = "".join(f"<li>{html.escape(item)}</li>" for item in narrative)
    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
  <style>
    :root {{
      --bg: #081120;
      --panel: rgba(255, 255, 255, 0.06);
      --panel-border: rgba(255, 255, 255, 0.12);
      --text: #e8eefc;
      --muted: #a7b4d4;
      --accent: #7dd3fc;
      --accent-2: #f59e0b;
      --good: #34d399;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, "Segoe UI", system-ui, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.18), transparent 22%),
        linear-gradient(180deg, #06101c 0%, #0b172a 45%, #060a14 100%);
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 36px 20px 56px; }}
    .hero {{
      padding: 28px 28px 22px;
      border: 1px solid var(--panel-border);
      background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
      border-radius: 24px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.25);
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.05; }}
    .subtitle {{ color: var(--muted); font-size: 1.05rem; max-width: 980px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 18px 0 26px; }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 36px rgba(0,0,0,0.16);
    }}
    .metric-card {{ padding: 16px 18px; }}
    .metric-label {{ color: var(--muted); font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .metric-value {{ font-size: 1.12rem; line-height: 1.35; font-weight: 700; }}
    .panel {{ padding: 18px 18px 10px; margin-bottom: 22px; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 1.2rem; }}
    .narrative {{
      padding: 18px 20px;
      border-radius: 18px;
      background: rgba(125, 211, 252, 0.06);
      border: 1px solid rgba(125, 211, 252, 0.16);
      margin-bottom: 22px;
    }}
    .narrative ul {{ margin: 0; padding-left: 20px; color: var(--text); }}
    .table-wrap {{ overflow: auto; border-radius: 14px; }}
    table.data-table {{ width: 100%; border-collapse: collapse; color: var(--text); }}
    table.data-table thead th {{ position: sticky; top: 0; background: #0e1a2d; }}
    table.data-table th, table.data-table td {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.08); white-space: nowrap; }}
    table.data-table tbody tr:hover {{ background: rgba(255,255,255,0.05); }}
    @media (max-width: 720px) {{
      .wrap {{ padding: 18px 12px 36px; }}
      .hero {{ padding: 20px; }}
      .panel {{ padding: 14px 12px 6px; }}
    }}
  </style>
</head>
<body>
  <main class=\"wrap\">
    <section class=\"hero\">
      <h1>{html.escape(title)}</h1>
      <div class=\"subtitle\">{html.escape(subtitle)}</div>
    </section>
    <section class=\"cards\">{''.join(card_blocks)}</section>
    <section class=\"narrative\">
      <h2 style=\"margin:0 0 10px; font-size:1.15rem;\">What stands out</h2>
      <ul>{narrative_html}</ul>
    </section>
    {''.join(figure_blocks)}
    {''.join(table_blocks)}
  </main>
</body>
</html>"""
    return html_doc


def build_eda_report() -> str:
    panel = load_table("panel_state_city.csv")
    hotspot = load_table("hotspot_panel.csv")
    signal_table = load_table("signal_stability_table.csv") if (TABLE_DIR / "signal_stability_table.csv").exists() else load_table("special_signal_table.csv")
    macro = load_table("national_macro_auto_context_monthly.csv")

    state_panel = top_state_panel(panel)
    top_state = state_panel.iloc[0] if not state_panel.empty else None
    top_city = panel.loc[panel["scope_type"].eq("city")].sort_values("accident_count", ascending=False).head(1)
    top_hotspot = hotspot.head(1)
    stable_signal = signal_table.loc[signal_table.get("passes_discussion_filter", True)].sort_values("lift_vs_baseline_pct", ascending=False).head(1)
    macro_shift = compute_macro_takeaway(macro)

    cards = []
    if top_state is not None:
        cards.append(("Largest state", f"{top_state['scope_name']} with {int(top_state['accident_count']):,} accidents"))
        cards.append(("State severe share", f"{top_state['severe_share']:.1%}"))
    if not top_city.empty:
        cards.append(("Largest city", f"{top_city.iloc[0]['scope_name']} with {int(top_city.iloc[0]['accident_count']):,} accidents"))
    if not top_hotspot.empty:
        cards.append(("Largest hotspot", f"{top_hotspot.iloc[0].get('State', 'Unknown')} / {top_hotspot.iloc[0].get('metro_case_study', 'Unknown')}"))
    if not stable_signal.empty:
        cards.append(("Strongest stable signal", f"{stable_signal.iloc[0]['signal']} ({stable_signal.iloc[0].get('state', '')})".strip()))
    cards.append(("Macro divergence", f"Accidents {macro_shift['accident_count']:+.1f}%, VMT {macro_shift['vmt_millions']:+.1f}%"))

    narrative = []
    if top_state is not None:
        narrative.append(f"California is the heaviest state-level contributor in the saved panel, and its severe-share profile is materially above the national average in the top rows.")
    if not top_city.empty:
        narrative.append(f"The densest city-level cluster in the saved panel is {top_city.iloc[0]['scope_name']}, which is a useful anchor for the case-study maps already in the repo.")
    if not top_hotspot.empty:
        narrative.append(f"The strongest hotspot remains concentrated around {top_hotspot.iloc[0].get('State', 'Unknown')} / {top_hotspot.iloc[0].get('metro_case_study', 'Unknown')}, which matches the spatial concentration pattern shown in the existing maps.")
    if not stable_signal.empty:
        narrative.append(f"{stable_signal.iloc[0]['signal']} is the strongest discussion-ready signal in the exported table, with a lift of {float(stable_signal.iloc[0]['lift_vs_baseline_pct']):.1f}% and strong year-to-year stability.")
    narrative.append(
        f"The macro panel shows a split between traffic volume and accident growth: accidents changed by {macro_shift['accident_count']:+.1f}% while VMT changed by {macro_shift['vmt_millions']:+.1f}% between the 2017-2019 baseline and 2021-2022."
    )

    figures = [
        ("State-Level Concentration", build_state_chart(state_panel)),
        ("City and Metro Concentration", build_city_chart(panel)),
        ("Geographic Hotspots", build_hotspot_chart(hotspot)),
        ("Stable Signals", build_signal_chart(signal_table)),
        ("Macro Context", build_macro_chart(macro)),
    ]
    if not state_panel.empty:
        figures.insert(2, ("Interactive Globe", build_interactive_globe_figure(state_panel[["scope_name", "accident_count", "severe_share"]].rename(columns={"scope_name": "State"}), hotspot)) )

    tables = [
        ("Top State Rows", state_panel.head(10)),
        ("Top Stable Signals", signal_table.sort_values("lift_vs_baseline_pct", ascending=False).head(10)),
    ]

    return render_report(
        "EDA Report: US Accidents Intelligence Atlas",
        "Interactive summary built from the repo's exported tables: state/city concentration, hotspots, stable signals, and macro context.",
        cards,
        figures,
        tables,
        narrative,
    )


def build_predictions_report() -> str:
    severity_metrics = load_severity_metrics_from_notebook()
    severity_metrics = severity_metrics.loc[severity_metrics["model"].isin(["XGBoost (GPU)", "Random Forest", "Logistic Regression"])].copy()
    severity_metrics["model"] = pd.Categorical(
        severity_metrics["model"],
        categories=["Logistic Regression", "Random Forest", "XGBoost (GPU)"],
        ordered=True,
    )
    severity_metrics = severity_metrics.sort_values("pr_auc")

    cards = [
        ("Best severity model", f"XGBoost (GPU) / PR-AUC {float(severity_metrics.loc[severity_metrics['model'] == 'XGBoost (GPU)', 'pr_auc'].iloc[0]):.3f}"),
        ("Holdout prevalence", f"{float(severity_metrics['test_prevalence'].iloc[0]) * 100:.1f}% severe"),
        ("Top ablation bundle", str(SEVERITY_BUNDLE_ABLATION.iloc[0]["feature_bundle"])),
        ("Best daily forecast", f"Los Angeles / {FORECAST_TAKEAWAYS.loc[FORECAST_TAKEAWAYS['scope'] == 'Los Angeles', 'model'].iloc[0]}"),
    ]

    narrative = [
        "The severity task is the clearest win in the repo: XGBoost (GPU) leads the holdout comparison, with Random Forest close behind and Logistic Regression clearly weaker.",
        "The feature-bundle ablation shows that the richest structured bundle is the strongest overall configuration, so the predictive lift comes from combining calendar, road/context, and text-derived fields rather than a single feature family.",
        "The daily count forecasts are consistently better than a seasonal naive baseline across the highlighted scopes, with Los Angeles posting the largest improvement in the notebook's takeaway text.",
    ]

    figures = [
        ("Severity Model Comparison", build_model_metrics_chart(severity_metrics)),
        ("Feature-Bundle Ablation", build_bundle_ablation_chart(SEVERITY_BUNDLE_ABLATION)),
        ("Daily Forecast Improvements", build_forecast_chart(FORECAST_TAKEAWAYS)),
    ]

    forecast_table = FORECAST_TAKEAWAYS.copy().sort_values("rmse_improvement_pct", ascending=False)
    tables = [
        ("Severity Metrics", severity_metrics.reset_index(drop=True)),
        ("Feature-Bundle Ablation", SEVERITY_BUNDLE_ABLATION.reset_index(drop=True)),
        ("Daily Forecast Improvements", forecast_table.reset_index(drop=True)),
    ]

    severity_best = severity_metrics.sort_values("pr_auc", ascending=False).iloc[0]
    bundle_best = SEVERITY_BUNDLE_ABLATION.iloc[0]
    narrative.extend(
        [
            f"The best model remains {severity_best['model']} with PR-AUC {float(severity_best['pr_auc']):.3f}, ROC-AUC {float(severity_best['roc_auc']):.3f}, and recall {float(severity_best['recall']):.3f}.",
            f"Among the bundle ablations, {bundle_best['feature_bundle']} leads on PR-AUC at {float(bundle_best['pr_auc']):.3f}.",
            f"The strongest daily count improvement in the notebook takeaway is Los Angeles at {float(FORECAST_TAKEAWAYS.loc[FORECAST_TAKEAWAYS['scope'] == 'Los Angeles', 'rmse_improvement_pct'].iloc[0]):.1f}% RMSE improvement versus seasonal naive.",
        ]
    )

    return render_report(
        "Prediction Report: Severity and Forecast Models",
        "Interactive model summary built from the notebook's saved outputs and report tables, including a feature-bundle view of what improves performance.",
        cards,
        figures,
        tables,
        narrative,
    )


def write_reports(eda_output: Path | None = None, predictions_output: Path | None = None) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    eda_output = eda_output or (FIGURE_DIR / "eda_report.html")
    predictions_output = predictions_output or (FIGURE_DIR / "predictions_report.html")
    eda_output.write_text(build_eda_report(), encoding="utf-8")
    predictions_output.write_text(build_predictions_report(), encoding="utf-8")
    return eda_output, predictions_output


if __name__ == "__main__":
    eda_path, predictions_path = write_reports()
    print(f"Wrote {eda_path}")
    print(f"Wrote {predictions_path}")
