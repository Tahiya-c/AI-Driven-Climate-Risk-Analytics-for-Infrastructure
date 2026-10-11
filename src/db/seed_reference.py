"""Load the reference piles and the saved models into Supabase.

    python -m src.db.seed_reference            # both materials
    python -m src.db.seed_reference --material concrete

Run again after retraining a model: it replaces the reference imports (an import row
with ``uploaded_by = 'Reference data'`` per material) and registers the artifacts'
model versions. User uploads are never touched.

    timber   : PileB, trained-on-PileA GBM and MLP forecasts
               (data/dashboard/dashboard_data_timber_pileB_from_pileA_{gbm,mlp}.csv,
               written by the saved timber models)
    concrete : PileA-G, forecast here by the saved RNN and Ridge (src/models/concrete_saved.py)
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import ROOT_DIR
from src.db import repository as repo
from src.db.client import get_client
from src.db.import_pipeline import (
    CONCRETE_ASSET_COLUMNS,
    PATHWAY,
    TIMBER_ASSET_COLUMNS,
    _outcomes,
    _pile_results,
)
from src.models.timber_coefficients import (
    K_CONTACT,
    K_KNOT,
    K_PROTECT,
    K_SHELTER,
    lookup_hw_sw,
)

DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"
CONCRETE_CSV = ROOT_DIR / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"
TIMBER_ASSETS_CSV = ROOT_DIR / "data" / "templates" / "port_kembla" / "timber_assets.csv"


def register_model(material: str, kind: str, meta: dict) -> str:
    """Upsert a model version from its artifact metadata and make it the active one."""
    model_version = meta["model_version"]
    client = get_client()
    client.table("model_versions").update({"is_active": False}).eq("material", material).eq(
        "model_kind", kind).neq("model_version", model_version).execute()
    params = meta.get("params") or meta.get("model_params") or {}
    client.table("model_versions").upsert(
        {
            "model_version": model_version,
            "material": material,
            "model_kind": kind,
            "saved_at": meta.get("saved_at_utc"),
            "trained_on": str(meta.get("trained_on", "")) or None,
            "library_versions": meta.get("library_versions", {}),
            "params": json.loads(json.dumps(params, default=str)),
            "metadata": json.loads(json.dumps(meta, default=str)),
            "is_active": True,
        },
        on_conflict="model_version",
    ).execute()
    return model_version


# ------------------------------------------------------------------------ timber
def _timber_reference_asset(asset_id: str) -> pd.DataFrame:
    """The template pile, with its AS 5604 codes turned into the k_ numbers the
    import uses. Undefined codes (PileB's MTN3/CNT3/KNT4) take the workbook's
    approximate-match value, i.e. the last row of each table."""
    row = pd.read_csv(TIMBER_ASSETS_CSV).set_index("asset_id").loc[asset_id]

    def approx(code, table):
        return table.get(code, table[next(reversed(table))])

    return pd.DataFrame(
        [
            {
                "asset_id": asset_id,
                "description": row["description"],
                "timber_type": row["timber_type"],
                "year_installed": int(row["year_installed"]),
                "zone": row["zone"],
                "initial_diameter_mm": float(row["initial_diameter_mm"]),
                "sapwood_layer_mm": float(row["sapwood_layer_mm"]),
                "replacement_diameter_mm": float(row["replacement_diameter_mm"]),
                "k_heartwood": lookup_hw_sw(row["heartwood_class"], row["zone"]),
                "k_sapwood": lookup_hw_sw(row["sapwood_class"], row["zone"]),
                "k_shelter": approx(row["exposure"], K_SHELTER),
                "k_protect": approx(row["maintenance"], K_PROTECT),
                "k_contact": approx(row["contact"], K_CONTACT),
                "k_knot": approx(row["knot"], K_KNOT),
            }
        ]
    )


def seed_timber() -> str:
    from src.models.timber_forecast import load_model_metadata

    frames = {
        kind: pd.read_csv(DASHBOARD_DIR / f"dashboard_data_timber_pileB_from_pileA_{kind}.csv")
        for kind in ("gbm", "mlp")
    }
    asset_ids = sorted(set(frames["gbm"]["asset_id"]) & set(frames["mlp"]["asset_id"]))
    assets = pd.concat([_timber_reference_asset(a) for a in asset_ids], ignore_index=True)

    observed = frames["gbm"].loc[frames["gbm"]["method"] == "observed"]
    observed = observed.groupby(["asset_id", "year"], as_index=False)["value"].mean()
    diameters = assets.set_index("asset_id")["initial_diameter_mm"]
    observations = pd.DataFrame(
        {
            "asset_id": observed["asset_id"],
            "inspection_year": observed["year"],
            "remaining_diameter_mm": (
                observed["asset_id"].map(diameters) - 2 * observed["value"]
            ).round(4),
            "notes": "simulated: Timber_kembla_v6.3 workbook",
        }
    )

    removed = repo.delete_reference_imports("timber")
    import_id = repo.create_import(
        "timber", "dashboard_data_timber_pileB_from_pileA_{gbm,mlp}.csv",
        repo.REFERENCE_UPLOADER, "validated", 0, 0,
    )
    repo.save_assets("timber", import_id, assets, TIMBER_ASSET_COLUMNS)
    repo.save_observations("timber", import_id, observations)
    for kind, points in frames.items():
        meta = load_model_metadata(kind)
        version = register_model("timber", kind, meta)
        repo.save_run(
            import_id, version, kind, PATHWAY, False, None, points,
            _pile_results(observations, "inspection_year", asset_ids, None),
            _outcomes(points, assets.set_index("asset_id")),
        )
    repo.set_import_status(import_id, "forecast")
    print(f"timber: {len(asset_ids)} pile(s) seeded (replaced {removed} old reference import)")
    return import_id


# ------------------------------------------------------------------------ concrete
def seed_concrete() -> str:
    from src.models import concrete_saved

    reference = pd.read_csv(CONCRETE_CSV)
    assets = (
        reference.groupby("asset_code", as_index=False)[["zone", "wc_ratio", "fc_mpa"]].first()
        .rename(columns={"asset_code": "asset_id"})
    )
    # One measured history per pile: the Most Likely series up to the cutoff.
    history = reference.loc[
        (reference["emissions_pathway"] == PATHWAY)
        & (reference["climate_model"] == "MRI-CGCM2.3.2")
        & (reference["year"] <= 2026)
    ]
    observations = history.rename(
        columns={"asset_code": "asset_id", "carb_depth_xc_mm": "carb_depth_mm"}
    )[["asset_id", "year", "carb_depth_mm"]]

    forecasts = {
        kind: concrete_saved.forecast_reference(kind) for kind in concrete_saved.MODEL_KINDS
    }

    removed = repo.delete_reference_imports("concrete")
    import_id = repo.create_import(
        "concrete", CONCRETE_CSV.name, repo.REFERENCE_UPLOADER, "validated", 0, 0,
    )
    repo.save_assets("concrete", import_id, assets, CONCRETE_ASSET_COLUMNS)
    repo.save_observations("concrete", import_id, observations)
    for kind, points in forecasts.items():
        meta = concrete_saved.load_metadata(kind)
        version = register_model("concrete", kind, meta)
        repo.save_run(
            import_id, version, kind, PATHWAY, False, None, points,
            _pile_results(observations, "year", assets["asset_id"], None),
            _outcomes(points, None),
        )
    repo.set_import_status(import_id, "forecast")
    print(f"concrete: {len(assets)} pile(s) seeded (replaced {removed} old reference import)")
    return import_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--material", choices=("timber", "concrete", "both"), default="both")
    args = parser.parse_args()
    if args.material in ("timber", "both"):
        seed_timber()
    if args.material in ("concrete", "both"):
        seed_concrete()


if __name__ == "__main__":
    main()
