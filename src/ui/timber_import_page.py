"""Streamlit section: upload a timber import file, validate it, forecast with the saved models.

Used by src/ui/timber_data_app.py ("Import a pile"); ``render_import_section`` can be
called from any Streamlit page. The upload format is the skeleton workbook
(data/templates/timber_import_template.xlsx, docs/timber_import_guide.md).
"""

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from src.data.timber_import import (
    PATHWAYS,
    SKELETON_XLSX,
    read_import,
    to_model_tables,
    validate_import,
)
from src.models.timber_forecast import forecast_pile, load_model_metadata
from src.visualization.charts import material_forecast_chart

MODELS = {"GBM": "gbm", "MLP": "mlp"}
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def _scenario_labels(model_tables: dict) -> dict[str, str]:
    """'A1FI|MRI-CGCM2.3.2' -> 'Most Likely' (the forecast's scenario ids -> chart labels)."""
    climate = model_tables["climate"].merge(model_tables["scenarios"], on="scenario_id")
    pairs = climate[["emissions_pathway", "climate_model", "scenario_label"]].drop_duplicates()
    return {
        f"{row.emissions_pathway}|{row.climate_model}": row.scenario_label
        for row in pairs.itertuples(index=False)
    }


def _chart(forecast: pd.DataFrame, labels: dict[str, str], title: str, asset: pd.Series):
    observed = forecast.loc[forecast["method"] == "observed"]
    projected = forecast.loc[forecast["method"] != "observed"]
    if observed.empty:  # no inspections: start the line at installation with zero loss
        history = pd.DataFrame({"year": [int(asset["year_installed"])], "value": [0.0]})
    else:
        history = observed.groupby("year", as_index=False)["value"].mean()
    projection = (
        projected.assign(condition=projected["scenario"].map(labels))
        .groupby(["year", "condition"], as_index=False)["value"]
        .mean()
    )
    figure = material_forecast_chart(title, history, projection)
    radius = float(asset["initial_diameter_mm"]) / 2
    figure.add_hline(
        y=radius,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text=f"Full radius ({radius:.0f} mm)",
        annotation_position="top left",
    )
    return figure


def render_import_section() -> None:
    st.markdown(
        "Upload a timber pile file in the template format. It is validated, then forecast "
        "to 2070 under the 3 climate scenarios with the saved GBM and MLP models (no "
        "retraining)."
    )
    if SKELETON_XLSX.exists():
        st.download_button(
            "Download the import template (.xlsx)",
            SKELETON_XLSX.read_bytes(),
            file_name=SKELETON_XLSX.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:",
        )
    upload = st.file_uploader("Timber import file", type=["xlsx"])
    if upload is None:
        st.info("The template's README sheet lists every column, its unit and allowed values.")
        return

    try:
        tables = read_import(upload)
    except Exception as exc:  # unreadable/corrupt workbook: report, don't crash the page
        st.error(f"Could not read the file: {exc}")
        return
    result = validate_import(tables)
    issues = result.issues_frame()
    if result.errors:
        st.error(f"{len(result.errors)} error(s) - fix these and upload again.")
    elif not issues.empty:
        st.warning(f"Valid, with {len(issues)} warning(s).")
    else:
        st.success("Valid - no issues found.")
    if not issues.empty:
        st.dataframe(issues, width="stretch", hide_index=True)
    if not result.ok:
        return

    model_tables = to_model_tables(result)
    assets = model_tables["assets"].set_index("asset_id", drop=False)
    left, middle, right = st.columns([2, 1, 1], gap="small")
    asset_id = left.selectbox("Pile", list(assets.index))
    pathway = middle.selectbox("Emissions pathway", list(PATHWAYS))
    anchor = right.toggle(
        "Anchor to latest inspection",
        value=True,
        help="Shift the forecast so it starts from the latest measured value. Needs at "
        "least one inspection; removes the models' ~28 mm bias.",
    )

    labels = _scenario_labels(model_tables)
    one_pile = {
        **model_tables,
        "assets": model_tables["assets"].loc[lambda f: f["asset_id"] == asset_id],
    }
    if one_pile["observations"] is not None:
        one_pile["observations"] = one_pile["observations"].loc[lambda f: f["asset_id"] == asset_id]
        if one_pile["observations"].empty:
            one_pile["observations"] = None

    outputs = {}
    for name, kind in MODELS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs[name] = forecast_pile(
                    **one_pile, model_kind=kind, pathway=pathway, anchor=anchor
                )
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"{name}: {exc}")
            return

    columns = st.columns(2, gap="small")
    for container, (name, forecast) in zip(columns, outputs.items(), strict=True):
        version = load_model_metadata(MODELS[name]).get("model_version", "unknown version")
        with container:
            st.plotly_chart(
                _chart(
                    forecast,
                    labels,
                    f"Timber - {name} ({asset_id}, {pathway})",
                    assets.loc[asset_id],
                ),
                width="stretch",
                config=CHART_CONFIG,
            )
            st.caption(f"Model: {version}")

    combined = pd.concat(outputs.values(), ignore_index=True).drop_duplicates()
    st.download_button(
        "Download forecasts (CSV, dashboard format)",
        combined.to_csv(index=False).encode("utf-8"),
        file_name=f"timber_forecast_{asset_id}_{pathway}.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    if one_pile["observations"] is None:
        st.warning(
            "No inspections for this pile: the forecast is the model alone, which runs "
            "~28 mm high (see docs/timber_model_updated.md §7.2)."
        )
