from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

cwd = Path.cwd()
project_root = cwd if (cwd / "data").exists() else cwd.parents[1]

csv_path = (
    project_root
    / "data"
    / "processed"
    / "concrete_kembla_v6_consolidated.csv"
)

df = pd.read_csv(csv_path)

target = "carb_depth_increment_mm"


# ---------------------------------------------------------
# TARGET DISTRIBUTION BY PILE
# ---------------------------------------------------------

stats = (
    df.groupby("asset_code")[target]
    .agg(
        Count="count",
        Mean="mean",
        Std="std",
        Min="min",
        Max="max",
    )
    .reset_index()
)

stats["Range"] = stats["Max"] - stats["Min"]

print("\n")
print("=" * 90)
print("TARGET DISTRIBUTION BY PILE")
print("=" * 90)

print(stats.to_string(index=False))


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

artifacts = project_root / "artifacts"
artifacts.mkdir(exist_ok=True)

stats.to_csv(
    artifacts / "pile_target_statistics.csv",
    index=False,
)

print(
    "\nSaved to artifacts/pile_target_statistics.csv"
)

# ---------------------------------------------------------
# STRUCTURAL CHARACTERISTICS BY PILE
# ---------------------------------------------------------

characteristic_columns = [
    "asset_code",
    "year_built",
    "zone",
    "dist_from_coast_km",
    "exposure_class",
    "carbonation_class",
    "chloride_class",
    "cover_h_mm",
    "member_diameter_mm",
    "fc_mpa",
    "wc_ratio",
    "cement_content_kgm3",
]


pile_characteristics = (
    df[characteristic_columns]
    .drop_duplicates(
        subset=["asset_code"]
    )
    .sort_values(
        "asset_code"
    )
)


print("\n")
print("=" * 120)
print("STRUCTURAL CHARACTERISTICS BY PILE")
print("=" * 120)

print(
    pile_characteristics.to_string(
        index=False
    )
)


# Save results

pile_characteristics.to_csv(
    artifacts
    / "pile_characteristics.csv",
    index=False,
)


print(
    "\nSaved to artifacts/pile_characteristics.csv"
)