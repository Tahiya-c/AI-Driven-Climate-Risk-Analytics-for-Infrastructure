"""Main Streamlit dashboard matching the supplied low-fidelity layout."""

from collections.abc import Mapping
from math import isfinite
from typing import Any

import pandas as pd
import streamlit as st

from src.config import ROOT_DIR
from src.db import dashboard_data
from src.db import import_pipeline as pipeline
from src.db import repository as repo
from src.visualization.charts import (
    chart_frames,
    comparison_data_from_payload,
    material_forecast_chart,
)

PAGES = ["Concrete", "Timber", "Import piles"]
SAMPLE_CONCRETE_CSV = ROOT_DIR / "data" / "PileG_import.csv"
Y_AXIS_TITLES = {
    "Timber": "Material loss (mm)",
    "Concrete": "Concrete deterioration (carbonation depth, mm)",
}


# ---------------------------------------------------------
# Approved result files that users are allowed to download
# ---------------------------------------------------------

RESULT_FILES = {
    "Concrete Model Benchmark": ROOT_DIR / "results_concrete_benchmark.csv",
    "Concrete Benchmark Details": ROOT_DIR / "results_concrete_benchmark_detail.csv",
    "RNN Carbonation Results": ROOT_DIR / "results_rnn_carbonation.csv",
    "RNN Trajectory Results": ROOT_DIR / "results_rnn_trajectory.csv",
    "Benchmark by Asset": ROOT_DIR / "results_benchmark_asset.csv",
    "Benchmark by Scenario": ROOT_DIR / "results_benchmark_scenario.csv",
}


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        html, body {overflow-x: hidden;}
        *, *::before, *::after {box-sizing: border-box;}

        .block-container {
            padding: 1.5rem 1rem 2rem;
            width: auto;
            max-width: 1450px;
            margin: 0 auto;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #D8E0E8;
        }

        /* Compact sidebar: Streamlit's default spacing made it scroll. */
        [data-testid="stSidebarContent"] {
            overflow-x: hidden;
        }

        [data-testid="stSidebarHeader"] {
            height: 2.25rem;
            min-height: 2.25rem;
            padding-top: 0.5rem;
            padding-bottom: 0;
            margin-bottom: 0;
        }

        [data-testid="stSidebarUserContent"] {
            padding-top: 0;
            padding-bottom: 1rem;
        }

        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }

        [data-testid="stSidebarUserContent"] h1 {
            font-size: 1.35rem;
            padding: 0;
        }

        [data-testid="stSidebarUserContent"] hr {
            margin: 0.35rem 0;
        }

        [data-testid="stSidebarUserContent"] [data-testid="stCaptionContainer"] {
            margin-bottom: 0;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.9rem;
        }

        div[data-testid="stHorizontalBlock"] {
            width: 100%;
            min-width: 0;
        }

        div[data-testid="stHorizontalBlock"] > div {
            min-width: 0 !important;
        }

        div[data-testid="stPlotlyChart"] {
            width: 100%;
            max-width: 100%;
            min-width: 0;
            overflow: hidden;
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.5rem;
        }

        div[data-testid="stPlotlyChart"] > div {
            width: 100% !important;
            min-width: 0 !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-container,
        div[data-testid="stPlotlyChart"] .modebar {
            position: absolute !important;
            top: 0.4rem !important;
            right: auto !important;
            left: 7rem !important;
        }

        .modebar-btn[data-title="Autoscale"],
        .modebar-btn[data-title="Reset axes"] {
            display: none !important;
        }

        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
            }

            div[data-testid="stHorizontalBlock"] > div {
                flex: 1 1 100% !important;
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DATABASE READS (cached; cleared after every write)
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def _pile_options(material: str) -> pd.DataFrame:
    return dashboard_data.pile_options(material)


@st.cache_data(ttl=300, show_spinner="Loading forecasts...")
def _pile_model_data(material: str, import_id: str, asset_id: str):
    return dashboard_data.pile_model_data(material, import_id, asset_id)


@st.cache_data(ttl=300, show_spinner=False)
def _pile_summary(material: str, import_id: str, asset_id: str):
    return dashboard_data.pile_summary(material, import_id, asset_id)


@st.cache_data(ttl=300, show_spinner=False)
def _model_scores(material: str) -> pd.DataFrame:
    return repo.model_scores(material)


def _database_error(exc: Exception) -> None:
    st.error(f"Could not reach the database: {exc}")


# =========================================================
# CHARTS
# =========================================================

def _shared_ranges(data: Mapping[str, Mapping[str, pd.DataFrame]]):
    """One x and y range for every model, so the two charts compare directly."""
    values, years = [], []
    for model_data in data.values():
        for frame in (model_data["historical"], model_data["projection"]):
            values += frame["value"].tolist()
            years += frame["year"].tolist()
    pad = (max(values) - min(values)) * 0.05 or 0.1
    return [min(years), max(years)], [min(values) - pad, max(values) + pad]


def _model_figure(material, model_name, model_data, flags, ranges=None):
    figure = material_forecast_chart(
        f"{material} - {model_name}",
        model_data["historical"],
        model_data["projection"],
        flags,
    )
    figure.update_layout(yaxis_title=Y_AXIS_TITLES[material])
    if ranges:
        figure.update_xaxes(range=ranges[0])
        figure.update_yaxes(range=ranges[1])
    return figure


@st.dialog("Expanded chart", width="large")
def _show_expanded_chart(material, model_name, model_data, flags, ranges) -> None:
    """Show a material chart in a larger modal dialog."""
    figure = _model_figure(material, model_name, model_data, flags, ranges)
    figure.update_layout(height=680)
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def _render_model_charts(
    material: str,
    data: Mapping[str, Mapping[str, pd.DataFrame]],
    flags: list[dict[str, Any]] | None,
    key: str,
) -> None:
    """The two models' charts side by side, on shared axes."""
    ranges = _shared_ranges(data)
    columns = st.columns(len(data), gap="small")
    for container, (model_name, model_data) in zip(columns, data.items()):
        with container:
            st.plotly_chart(
                _model_figure(material, model_name, model_data, flags, ranges),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
                key=f"chart_{key}_{model_name}",
            )
            if st.button(
                "Open larger view",
                key=f"expand_{key}_{model_name.lower()}",
                icon=":material/open_in_full:",
                use_container_width=True,
            ):
                _show_expanded_chart(material, model_name, model_data, flags, ranges)


# =========================================================
# PILE SELECTION AND FLAGS (flags are saved in Supabase)
# =========================================================

@st.dialog("Add flag")
def _add_flag_dialog(material: str, pile: Mapping[str, Any]) -> None:
    """Collect a user-defined threshold for the selected material and pile."""
    metric = "Material loss" if material == "Timber" else "Carbonation depth"
    st.caption(f"{material} · {pile['label']}")
    with st.form(f"flag_form_{material}"):
        purpose = st.selectbox("Flag purpose", ["Maintenance", "Replacement"])
        value = st.number_input(f"{metric} (mm)", min_value=0.0, value=None, step=1.0)
        submitted = st.form_submit_button("Add flag", type="primary")
    if submitted:
        if value is None or not isfinite(value):
            st.error("Enter a finite, non-negative value in mm.")
            return
        repo.add_flag(
            material.lower(), pile["asset_id"], pile["import_id"], purpose, value
        )
        st.rerun()


def _pile_controls(material: str) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]]]:
    """Pile dropdown (reference piles and every upload) and its saved flags."""
    try:
        piles = _pile_options(material.lower())
    except Exception as exc:  # noqa: BLE001 - any database failure is shown, not raised
        _database_error(exc)
        return None, []
    if piles.empty:
        st.info(
            f"No {material.lower()} piles in the database yet. Use **Import piles** to "
            "add some, or run `python -m src.db.seed_reference`."
        )
        return None, []

    labels = piles["label"].tolist()
    keys = (piles["import_id"] + "|" + piles["asset_id"]).tolist()
    state_key = f"{material.lower()}_pile"
    pending = st.session_state.pop(f"{state_key}_pending", None)
    if pending in keys:  # set by "Open on the ... page" after an import
        st.session_state[state_key] = labels[keys.index(pending)]
    if st.session_state.get(state_key) not in labels:
        st.session_state.pop(state_key, None)
    choice = st.selectbox("Pile", labels, key=state_key)
    pile = piles.iloc[labels.index(choice)].to_dict()

    with st.sidebar:
        st.markdown("**Maintenance and replacement flags**")
        st.caption("Threshold lines on both charts, saved for everyone.")
        if st.button("Add flag", key=f"add_flag_{material}", icon=":material/flag:"):
            _add_flag_dialog(material, pile)
        flags = repo.list_flags(material.lower(), pile["asset_id"], pile["import_id"])
        for flag in flags:
            label, remove = st.columns([3, 2])
            label.write(f"{flag['purpose']}: {flag['value']:g} mm")
            if remove.button("Remove", key=f"remove_flag_{flag['id']}"):
                repo.delete_flag(flag["id"])
                st.rerun()

        st.divider()
        st.markdown("**Remove pile**")
        if pile["is_reference"]:
            st.caption("Reference piles can't be removed.")
        elif st.button(
            "Remove this pile",
            key=f"remove_pile_{material}",
            icon=":material/delete:",
        ):
            _remove_pile_dialog(material, pile)
    return pile, flags


@st.dialog("Remove pile")
def _remove_pile_dialog(material: str, pile: Mapping[str, Any]) -> None:
    """Confirm, then delete one imported pile and everything saved for it."""
    st.write(f"**{pile['label']}**")
    st.caption(
        "This deletes the pile's inspections, both models' forecasts, its results and "
        "its flags, for everyone. If it is the last pile in its upload, the upload and "
        "its stored file are deleted too. This can't be undone."
    )
    confirm = st.checkbox("Yes, remove this pile", key=f"remove_pile_confirm_{material}")
    if st.button("Remove pile", type="primary", disabled=not confirm):
        try:
            repo.delete_pile(material.lower(), pile["import_id"], pile["asset_id"])
        except Exception as exc:  # noqa: BLE001
            _database_error(exc)
            return
        st.cache_data.clear()
        st.session_state.pop(f"{material.lower()}_pile", None)
        st.session_state.pop("import_state", None)
        st.rerun()


def _render_pile_details(material: str, pile: Mapping[str, Any]) -> None:
    """Per-pile results saved with the forecast, and the models' published scores."""
    try:
        results, outcomes = _pile_summary(
            material.lower(), pile["import_id"], pile["asset_id"]
        )
        scores = _model_scores(material.lower())
    except Exception as exc:  # noqa: BLE001
        _database_error(exc)
        return
    with st.expander("Pile results"):
        if not outcomes.empty:
            columns = ["model", "condition", "value_2050", "final_year", "final_value"]
            if material == "Timber":
                columns.append("threshold_year")
            st.dataframe(
                outcomes[columns].rename(
                    columns={
                        "condition": "scenario",
                        "value_2050": "2050 (mm)",
                        "final_value": "final (mm)",
                        "threshold_year": "reaches replacement diameter",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        if not results.empty:
            columns = ["model", "n_inspections", "first_inspection", "last_inspection"]
            if material == "Timber":
                columns += ["anchor_offset_mm", "large_offset"]
            st.dataframe(results[columns], hide_index=True, use_container_width=True)
    with st.expander("Model scores"):
        if scores.empty:
            st.caption("No scores recorded.")
        else:
            st.dataframe(
                scores.drop(columns=["model_version"]),
                hide_index=True,
                use_container_width=True,
            )


def _go_to(page: str, material: str | None = None, pile_key: str | None = None) -> None:
    """Button callback: switch page (and optionally select a pile there)."""
    st.session_state["page"] = page
    if material and pile_key:
        st.session_state[f"{material}_pile_pending"] = pile_key


def render_dashboard(
    comparison_payload: Mapping[str, Any] | None = None,
) -> None:
    """Render the main dashboard."""

    st.set_page_config(
        page_title="Port Pile Deterioration",
        page_icon="🏗️",
        layout="wide",
    )

    _apply_styles()

    st.session_state.setdefault("page", "Timber")

    with st.sidebar:
        st.title("Port Pile Deterioration")
        st.caption("Port Kembla · timber and concrete piles")

        page = st.radio(
            "Page",
            PAGES,
            key="page",
            label_visibility="collapsed",
        )

        st.button(
            "Import piles",
            type="primary",
            icon=":material/upload_file:",
            use_container_width=True,
            on_click=_go_to,
            args=("Import piles",),
            disabled=page == "Import piles",
        )

        st.divider()

    if page == "Timber":
        _render_timber_page(comparison_payload)
    elif page == "Concrete":
        _render_concrete_page()
    else:
        _render_import_page()


# =========================================================
# EXPORT HELPER FUNCTIONS
# =========================================================

def _build_timber_export_data(
    data: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
) -> pd.DataFrame:
    """
    Create a clean CSV containing the currently displayed
    timber results from the GBM and MLP models.
    """

    export_frames: list[pd.DataFrame] = []

    for model_name, model_data in data.items():

        # -----------------------------
        # Historical data
        # -----------------------------
        historical = model_data["historical"].copy()

        historical["Scenario"] = "Historical"
        historical["Model"] = model_name
        historical["Data Type"] = "Historical"

        historical = historical.rename(
            columns={
                "year": "Year",
                "value": "Material Loss",
            }
        )

        # -----------------------------
        # Forecast data
        # -----------------------------
        projection = model_data["projection"].copy()

        projection = projection.rename(
            columns={
                "year": "Year",
                "condition": "Scenario",
                "value": "Material Loss",
            }
        )

        projection["Model"] = model_name
        projection["Data Type"] = "AI Forecast"

        export_frames.append(historical)
        export_frames.append(projection)

    export_data = pd.concat(
        export_frames,
        ignore_index=True,
    )

    export_data = export_data[
        [
            "Year",
            "Scenario",
            "Material Loss",
            "Model",
            "Data Type",
        ]
    ]

    export_data = export_data.sort_values(
        [
            "Model",
            "Year",
            "Scenario",
        ],
        ignore_index=True,
    )

    return export_data


def _build_concrete_export_data(
    data: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
    pile: str,
) -> pd.DataFrame:
    """
    Create a clean CSV containing the currently displayed
    concrete results for the selected pile.
    """

    export_frames: list[pd.DataFrame] = []

    for model_name, model_data in data.items():

        historical = model_data["historical"].copy()

        historical["Scenario"] = "Historical"
        historical["Model"] = model_name
        historical["Data Type"] = "Historical"

        historical = historical.rename(
            columns={
                "year": "Year",
                "value": "Carbonation Depth (mm)",
            }
        )

        projection = model_data["projection"].copy()

        projection = projection.rename(
            columns={
                "year": "Year",
                "condition": "Scenario",
                "value": "Carbonation Depth (mm)",
            }
        )

        projection["Model"] = model_name
        projection["Data Type"] = "AI Forecast"

        export_frames.append(historical)
        export_frames.append(projection)

    export_data = pd.concat(
        export_frames,
        ignore_index=True,
    )

    export_data.insert(
        0,
        "Pile",
        pile,
    )

    export_data = export_data[
        [
            "Pile",
            "Year",
            "Scenario",
            "Carbonation Depth (mm)",
            "Model",
            "Data Type",
        ]
    ]

    export_data = export_data.sort_values(
        [
            "Model",
            "Year",
            "Scenario",
        ],
        ignore_index=True,
    )

    return export_data


def _render_result_file_export() -> None:
    """
    Allow the user to download approved result datasets.
    """

    st.subheader("Download result dataset")

    st.caption(
        "Choose an available analysis result to export as a CSV file."
    )

    selected_result = st.selectbox(
        "Result dataset",
        list(RESULT_FILES.keys()),
        key="result_dataset_selector",
    )

    selected_path = RESULT_FILES[selected_result]

    if not selected_path.exists():
        st.warning(
            f"{selected_result} is not currently available."
        )
        return

    try:
        result_df = pd.read_csv(selected_path)

    except (
        pd.errors.ParserError,
        OSError,
    ) as exc:
        st.error(
            f"Could not load this result file: {exc}"
        )
        return

    with st.expander("Preview selected dataset"):
        st.dataframe(
            result_df.head(20),
            use_container_width=True,
        )

        st.caption(
            f"{len(result_df):,} rows × "
            f"{len(result_df.columns):,} columns"
        )

    csv_data = (
        result_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label=f"Download {selected_result}",
        data=csv_data,
        file_name=selected_path.name,
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
        key="download_result_dataset",
    )


# =========================================================
# TIMBER PAGE
# =========================================================

def _render_timber_page(
    comparison_payload: Mapping[str, Any] | None = None,
) -> None:
    """Render GBM and MLP Timber forecasts side by side."""

    st.title("Timber outlook")

    st.caption(
        "Compare material deterioration projections "
        "from the GBM and MLP models."
    )

    if comparison_payload is not None:
        try:
            data = comparison_data_from_payload(comparison_payload)["Timber"]
        except ValueError as exc:
            st.error(f"Could not load comparison data: {exc}")
            return
        _render_model_charts("Timber", data, [], "timber_payload")
        return

    pile, flags = _pile_controls("Timber")
    if pile is None:
        return

    try:
        data = _pile_model_data("timber", pile["import_id"], pile["asset_id"])
    except Exception as exc:  # noqa: BLE001
        _database_error(exc)
        return

    _render_model_charts("Timber", data, flags, "timber")
    _render_pile_details("Timber", pile)

    # ==================================================
    # TIMBER EXPORT
    # ==================================================

    st.divider()

    st.header("Export results")

    export_data = _build_timber_export_data(data)
    export_data.insert(0, "Pile", pile["asset_id"])

    st.download_button(
        label=f"Export {pile['asset_id']} results",
        data=export_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{pile['asset_id']}_timber_forecast.csv",
        mime="text/csv",
        icon=":material/download:",
        key="download_current_timber_pile",
    )

    st.divider()

    with st.expander("Backend response contract"):
        st.code(
            '''{
  "concrete": {
    "historical": [
      {
        "year": 2000,
        "value": 86.0
      }
    ],
    "projection": [
      {
        "year": 2027,
        "condition": "Most Likely",
        "value": 70.0
      }
    ]
  },
  "timber": {
    "historical": [],
    "projection": []
  }
}''',
            language="json",
        )


# =========================================================
# CONCRETE PAGE
# =========================================================

def _render_concrete_page() -> None:
    """Render RNN and Ridge concrete forecasts side by side."""

    st.title("Concrete outlook")

    st.caption(
        "Compare concrete deterioration projections "
        "from the RNN and Ridge models."
    )

    pile, flags = _pile_controls("Concrete")
    if pile is None:
        return

    try:
        data = _pile_model_data("concrete", pile["import_id"], pile["asset_id"])
    except Exception as exc:  # noqa: BLE001
        _database_error(exc)
        return

    _render_model_charts("Concrete", data, flags, "concrete")
    _render_pile_details("Concrete", pile)

    # ==================================================
    # CONCRETE EXPORT
    # ==================================================

    st.divider()

    st.header("Export results")

    st.caption(
        "Download the results currently being viewed, "
        "or choose another available result dataset."
    )

    current_export_col, dataset_export_col = st.columns(
        2,
        gap="large",
    )

    # --------------------------------------------------
    # Current pile export
    # --------------------------------------------------

    with current_export_col:

        st.subheader("Current pile")

        st.caption(
            "Download the historical data and AI forecasts "
            f"currently displayed for {pile['asset_id']}. "
            "The CSV contains both RNN and Ridge results."
        )

        export_data = _build_concrete_export_data(
            data,
            pile["asset_id"],
        )

        current_csv = (
            export_data
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=f"Export {pile['asset_id']} results",
            data=current_csv,
            file_name=(
                f"{pile['asset_id']}_concrete_forecast.csv"
            ),
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
            key="download_current_pile",
        )

        with st.expander(
            "Preview current export"
        ):

            st.dataframe(
                export_data.head(20),
                use_container_width=True,
            )

            st.caption(
                f"{len(export_data):,} rows × "
                f"{len(export_data.columns):,} columns"
            )

    # --------------------------------------------------
    # Other result datasets
    # --------------------------------------------------

    with dataset_export_col:
        _render_result_file_export()


# =========================================================
# IMPORT PAGE
# =========================================================

def _render_import_page() -> None:
    """Choose a material, see what to supply, upload, validate, forecast, save."""

    st.title("Import piles")

    st.caption(
        "Upload Port Kembla pile data, check it, and forecast it with both models. "
        "There is no login: every upload is saved and visible to everyone."
    )

    material = st.radio(
        "Material",
        ["Timber", "Concrete"],
        horizontal=True,
        key="import_material",
    )
    key = material.lower()

    # --------------------------------------------------
    # Step 1: what the engineer must supply
    # --------------------------------------------------

    st.subheader("1. Data to supply")

    if material == "Timber":
        st.caption(
            "Four CSV files, one per sheet below, named timber_assets.csv, "
            "site_climate.csv, climate_changes.csv and timber_observations.csv (matching "
            "the Timber_kembla_v6.3 workbook). timber_observations (measured remaining "
            "diameter) is optional but recommended: the forecast is anchored to it. Enter "
            "the six pile coefficients as numbers (see the coefficient table below)."
        )
    else:
        st.caption(
            "One CSV with the columns asset_id, zone, wc_ratio, fc_mpa, year, "
            "carb_depth_mm: one row per inspection, the pile's properties repeated on "
            "each row. Inspections must reach at least 2006: the models read the years "
            "just before the forecast starts. Climate is filled in from the reference "
            "Port Kembla data through 2070."
        )

    st.dataframe(
        pipeline.required_features(key),
        hide_index=True,
        use_container_width=True,
    )

    template_col, sample_col = st.columns(2)

    template_name, template_data, template_mime = pipeline.template_download(key)
    template_col.download_button(
        "Download CSV template" + (" (zip of 4 CSVs)" if material == "Timber" else ""),
        data=template_data,
        file_name=template_name,
        mime=template_mime,
        icon=":material/download:",
        use_container_width=True,
        key=f"download_template_{key}",
    )

    if material == "Concrete" and SAMPLE_CONCRETE_CSV.exists():
        sample_col.download_button(
            "Download sample",
            data=SAMPLE_CONCRETE_CSV.read_bytes(),
            file_name=SAMPLE_CONCRETE_CSV.name,
            mime="text/csv",
            icon=":material/science:",
            use_container_width=True,
            key="download_concrete_sample",
        )

    if material == "Timber":
        from src.data.timber_import import coefficient_table

        with st.expander("Coefficient values (AS 5604)"):
            st.dataframe(
                coefficient_table(),
                hide_index=True,
                use_container_width=True,
            )

    # --------------------------------------------------
    # Step 2: upload and validate
    # --------------------------------------------------

    st.subheader("2. Upload and validate")

    files = st.file_uploader(
        "Import file" + ("" if material == "Timber" else " (CSV)"),
        type=["csv"],
        accept_multiple_files=True,
        key=f"upload_{key}",
        help=(
            "Timber: the 4 CSVs, named <sheet>.csv. Concrete: one CSV with one row per "
            "inspection (or one CSV per sheet, named <sheet>.csv)."
        ),
    )

    uploaded_by = st.text_input(
        "Your name (optional)",
        key="uploaded_by",
        max_chars=80,
    )

    signature = (key, tuple((f.name, f.size) for f in files or []))
    state = st.session_state.get("import_state")
    if state and state["signature"] != signature:
        state = None
        st.session_state.pop("import_state", None)

    if st.button(
        "Validate",
        type="primary",
        disabled=not files,
        icon=":material/fact_check:",
    ):
        contents = [(f.name, f.getvalue()) for f in files]
        try:
            tables, notes = pipeline.read_upload(key, contents)
        except Exception as exc:  # noqa: BLE001 - unreadable file
            st.error(f"Could not read the file: {exc}")
            return
        state = {
            "signature": signature,
            "contents": contents,
            "tables": tables,
            "notes": notes,
            "result": pipeline.validate(key, tables),
        }
        st.session_state["import_state"] = state

    if not state:
        _render_manage_imports(key)
        return

    result = state["result"]
    _render_validation(result, state["notes"], key)

    if not result.ok:
        st.error(
            f"{len(result.errors)} error(s) to fix before forecasting. "
            "Correct the file and upload it again."
        )
        _render_manage_imports(key)
        return

    # --------------------------------------------------
    # Step 3: predict with both models and save
    # --------------------------------------------------

    st.subheader("3. Predict and save")

    if state.get("import_id") is None:
        models = " and ".join(label for label, _ in pipeline.MODELS[key])
        st.caption(
            f"Forecasts every pile with the {models} models (A1FI climate, three "
            "scenarios) and saves the file, piles, inspections and forecasts to the database."
        )

        st.markdown("**Pile names**")
        st.caption(
            "Each pile is named from its asset_id in the file. Change a name here to "
            "save the pile under a different one, e.g. PileG → Berth 102 Pile 7."
        )
        file_names = sorted(result.tables[f"{key}_assets"]["asset_id"].astype(str))
        edited = st.data_editor(
            pd.DataFrame({"In file": file_names, "Pile name": file_names}),
            hide_index=True,
            disabled=["In file"],
            use_container_width=True,
            key=f"pile_names_{key}_{hash(signature)}",
        )
        names = dict(zip(edited["In file"], edited["Pile name"]))

        if st.button(
            "Predict and save",
            type="primary",
            icon=":material/insights:",
        ):
            problems = pipeline.check_pile_names(names)
            if problems:
                for problem in problems:
                    st.error(problem)
                _render_manage_imports(key)
                return
            result = pipeline.rename_piles(result, names)
            with st.spinner("Forecasting..."):
                try:
                    forecasts = pipeline.forecast(key, result)
                except (ValueError, FileNotFoundError) as exc:
                    st.error(f"Could not forecast: {exc}")
                    return
            with st.spinner("Saving to the database..."):
                file_name, content = _stored_file(key, state["contents"])
                try:
                    import_id = pipeline.save(
                        key,
                        file_name,
                        content,
                        uploaded_by.strip() or None,
                        result,
                        forecasts,
                        state["tables"],
                    )
                except Exception as exc:  # noqa: BLE001
                    _database_error(exc)
                    return
            st.cache_data.clear()
            state.update(forecasts=forecasts, import_id=import_id)

    if state.get("forecasts"):
        _render_import_results(material, state)

    _render_manage_imports(key)


def _stored_file(material: str, contents: list[tuple[str, bytes]]) -> tuple[str, bytes]:
    """The upload as one file for storage: the .xlsx itself, or the CSVs zipped."""
    import re
    import zipfile
    from io import BytesIO

    def safe(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    xlsx = [item for item in contents if item[0].lower().endswith(".xlsx")]
    if xlsx:
        return safe(xlsx[0][0]), xlsx[0][1]
    if len(contents) == 1:
        return safe(contents[0][0]), contents[0][1]
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents:
            archive.writestr(safe(name), data)
    return f"{material}_import_csvs.zip", buffer.getvalue()


def _render_validation(result, notes: list[str], material: str) -> None:
    """Issue counts, every issue, and a preview of the checked sheets."""
    asset_sheet = f"{material}_assets"
    n_piles = len(result.tables.get(asset_sheet, []))
    n_errors = len(result.errors)
    n_warnings = len(result.issues) - n_errors

    piles_col, errors_col, warnings_col = st.columns(3)
    piles_col.metric("Piles", n_piles)
    errors_col.metric("Errors", n_errors)
    warnings_col.metric("Warnings", n_warnings)

    for note in notes:
        st.warning(note)

    if result.ok:
        st.success("The file is valid." + (" Check the warnings below." if n_warnings else ""))

    issues = result.issues_frame()
    if not issues.empty:
        st.dataframe(issues, hide_index=True, use_container_width=True)

    with st.expander("Preview the checked data"):
        sheets = [sheet for sheet, frame in result.tables.items() if not frame.empty]
        if sheets:
            for tab, sheet in zip(st.tabs(sheets), sheets):
                with tab:
                    st.dataframe(
                        result.tables[sheet].head(200),
                        hide_index=True,
                        use_container_width=True,
                    )


def _render_import_results(material: str, state: Mapping[str, Any]) -> None:
    """Both models' charts for one imported pile, straight from the forecast."""
    forecasts = state["forecasts"]
    key = material.lower()
    asset_ids = sorted(forecasts[0].points["asset_id"].unique())

    st.success(f"Saved {len(asset_ids)} pile(s). They now appear on the {material} page.")

    asset_id = st.selectbox("Imported pile", asset_ids, key=f"import_result_{key}")

    data = {
        f.label: chart_frames(f.points.loc[f.points["asset_id"] == asset_id])
        for f in forecasts
    }
    _render_model_charts(material, data, [], f"import_{key}_{asset_id}")

    outcomes = pd.concat(
        [f.outcomes.assign(model=f.label) for f in forecasts], ignore_index=True
    )
    outcomes = outcomes.loc[outcomes["asset_id"] == asset_id]
    columns = ["model", "scenario", "value_2050", "final_year", "final_value"]
    if material == "Timber":
        columns.append("threshold_year")
        for f in forecasts:
            offset = (f.anchor or {}).get("offsets_mm", {}).get(asset_id)
            if offset is not None:
                st.caption(
                    f"{f.label}: forecast shifted {offset:+.1f} mm to continue from the "
                    "pile's inspections."
                )
            large = (f.anchor or {}).get("large_offsets", [])
            if any(item["series"] == asset_id for item in large):
                st.warning(
                    f"{f.label}: the shift is over 15 mm, so the model is far from this "
                    "pile's measurements. Treat its forecast with caution."
                )
    st.dataframe(outcomes[columns], hide_index=True, use_container_width=True)

    st.button(
        f"Open {asset_id} on the {material} page",
        icon=":material/arrow_forward:",
        on_click=_go_to,
        args=(material, key, f"{state['import_id']}|{asset_id}"),
    )


def _render_manage_imports(material: str) -> None:
    """List earlier uploads, and delete one (with everything saved under it)."""
    with st.expander("Uploaded files"):
        try:
            imports = repo.list_imports(material)
        except Exception as exc:  # noqa: BLE001
            _database_error(exc)
            return
        if not imports.empty:
            imports = imports.loc[imports["uploaded_by"] != repo.REFERENCE_UPLOADER]
        if imports.empty:
            st.caption("No uploads yet.")
            return
        st.dataframe(
            imports.drop(columns=["import_id", "material"]),
            hide_index=True,
            use_container_width=True,
        )
        labels = {
            row["import_id"]: (
                f"{row['file_name']} · "
                f"{pd.Timestamp(row['created_at']).strftime('%d %b %Y %H:%M')}"
            )
            for row in imports.to_dict("records")
        }
        chosen = st.selectbox(
            "Delete an upload",
            list(labels),
            format_func=labels.get,
            key=f"delete_choice_{material}",
        )
        confirm = st.checkbox(
            "Yes, delete this upload, its piles, forecasts and flags for everyone",
            key=f"delete_confirm_{material}",
        )
        if st.button("Delete upload", disabled=not confirm, key=f"delete_{material}"):
            try:
                repo.delete_import(chosen)
            except Exception as exc:  # noqa: BLE001
                _database_error(exc)
                return
            st.cache_data.clear()
            st.session_state.pop("import_state", None)
            st.rerun()
