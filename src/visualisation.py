
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

def build_interactive_globe_figure(state_summary, hotspot_panel=None):
    """Build a Plotly orthographic globe showing US accident intensity.

    Accepts state_summary with columns: State, accident_count, severe_share.
    """
    import plotly.graph_objects as go

    # US state centroids for plotting
    state_centroids = {
        "AL": (32.8, -86.8), "AK": (64.0, -153.0), "AZ": (34.3, -111.7),
        "AR": (34.8, -92.2), "CA": (36.8, -119.4), "CO": (39.1, -105.4),
        "CT": (41.6, -72.7), "DE": (39.0, -75.5), "FL": (27.8, -81.7),
        "GA": (33.2, -83.4), "HI": (19.7, -155.5), "ID": (44.1, -114.7),
        "IL": (40.0, -89.2), "IN": (39.8, -86.2), "IA": (42.0, -93.5),
        "KS": (38.5, -98.3), "KY": (37.8, -84.3), "LA": (31.1, -91.9),
        "ME": (45.3, -69.4), "MD": (39.1, -76.8), "MA": (42.4, -71.4),
        "MI": (44.3, -84.5), "MN": (46.3, -94.3), "MS": (32.7, -89.7),
        "MO": (38.5, -92.3), "MT": (46.9, -110.4), "NE": (41.5, -99.8),
        "NV": (38.8, -116.4), "NH": (43.5, -71.5), "NJ": (40.1, -74.5),
        "NM": (34.4, -106.0), "NY": (42.9, -75.5), "NC": (35.6, -79.8),
        "ND": (47.5, -100.5), "OH": (40.4, -82.7), "OK": (35.6, -97.5),
        "OR": (44.0, -120.5), "PA": (41.2, -77.2), "RI": (41.7, -71.5),
        "SC": (34.0, -81.0), "SD": (44.3, -100.3), "TN": (35.9, -86.4),
        "TX": (31.5, -99.3), "UT": (39.3, -111.7), "VT": (44.1, -72.6),
        "VA": (37.5, -78.9), "WA": (47.4, -120.7), "WV": (38.5, -80.5),
        "WI": (44.5, -89.8), "WY": (43.1, -107.6), "DC": (38.9, -77.0),
    }

    plot_df = state_summary.copy()
    plot_df["lat"] = plot_df["State"].map(lambda s: state_centroids.get(s, (0, 0))[0])
    plot_df["lon"] = plot_df["State"].map(lambda s: state_centroids.get(s, (0, 0))[1])
    plot_df = plot_df.loc[plot_df["lat"] != 0].copy()

    max_count = plot_df["accident_count"].max()
    plot_df["marker_size"] = (plot_df["accident_count"] / max_count * 50).clip(lower=5)

    fig = go.Figure()

    # Accident intensity bubbles
    fig.add_trace(
        go.Scattergeo(
            lat=plot_df["lat"],
            lon=plot_df["lon"],
            text=plot_df.apply(
                lambda r: (
                    f"<b>{r['State']}</b><br>"
                    f"Accidents: {int(r['accident_count']):,}<br>"
                    f"Severe: {r['severe_share']:.1%}<br>"
                    f"Duration: {r.get('median_duration_min', 0):.0f} min"
                ),
                axis=1,
            ),
            hoverinfo="text",
            marker=dict(
                size=plot_df["marker_size"],
                color=plot_df["severe_share"],
                colorscale="YlOrRd",
                colorbar=dict(title="Severe Share", x=1.05),
                opacity=0.8,
                line=dict(width=0.5, color="white"),
                sizemode="diameter",
            ),
            name="US States",
        )
    )

    # Optional: add hotspot-level points if provided
    if hotspot_panel is not None and not hotspot_panel.empty:
        top_hotspots = hotspot_panel.head(200).copy()
        max_hs = top_hotspots["accident_count"].max()
        top_hotspots["hs_size"] = (top_hotspots["accident_count"] / max_hs * 12).clip(lower=2)
        fig.add_trace(
            go.Scattergeo(
                lat=top_hotspots["lat_bin"],
                lon=top_hotspots["lng_bin"],
                text=top_hotspots.apply(
                    lambda r: f"Hotspot ({r['lat_bin']:.2f}, {r['lng_bin']:.2f})<br>Accidents: {int(r['accident_count']):,}",
                    axis=1,
                ),
                hoverinfo="text",
                marker=dict(
                    size=top_hotspots["hs_size"],
                    color="#ff4444",
                    opacity=0.4,
                    line=dict(width=0),
                ),
                name="Hotspot Grid Cells",
            )
        )

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=-96, lat=38, roll=0),
        showcoastlines=True,
        coastlinecolor="#444",
        showland=True,
        landcolor="#1a1a2e",
        showocean=True,
        oceancolor="#0f3460",
        showcountries=True,
        countrycolor="#555",
        showlakes=True,
        lakecolor="#0f3460",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=dict(
            text="🌍 US Traffic Accident Intensity Globe",
            font=dict(size=20),
        ),
        geo=dict(
            scope="world",
            showframe=False,
        ),
        paper_bgcolor="#0a0a23",
        plot_bgcolor="#0a0a23",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=60, b=0),
        height=700,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig

