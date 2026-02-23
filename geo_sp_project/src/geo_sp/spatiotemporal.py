"""
src/geo_sp/spatiotemporal.py
Spatial mapping + spatiotemporal feature construction (Step 2).
"""
from __future__ import annotations
from datetime import datetime
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ── Build GeoDataFrame ───────────────────────────────────────────────────────

def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geom = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def add_buffers(gdf: gpd.GeoDataFrame,
                sizes: dict = None) -> gpd.GeoDataFrame:
    """Add buffer columns (in projected CRS, stored as WGS84 polygons)."""
    if sizes is None:
        sizes = {"buffer_500m": 500, "buffer_1km": 1000, "buffer_5km": 5000}
    proj = gdf.to_crs("EPSG:28356")
    for name, dist in sizes.items():
        proj[name] = proj.geometry.buffer(dist)
    return proj.to_crs("EPSG:4326")


# ── Temporal features ────────────────────────────────────────────────────────

def _season(month: int) -> str:
    return {12: "summer", 1: "summer", 2: "summer",
            3: "autumn", 4: "autumn", 5: "autumn",
            6: "winter", 7: "winter", 8: "winter"}.get(month, "spring")


def _is_holiday(date) -> int:
    return 1 if date.month in [1, 4, 7, 9, 10, 12] else 0


def add_temporal_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["year"]         = gdf["date"].dt.year
    gdf["month"]        = gdf["date"].dt.month
    gdf["day"]          = gdf["date"].dt.day
    gdf["day_of_year"]  = gdf["date"].dt.dayofyear
    gdf["day_of_week"]  = gdf["date"].dt.dayofweek
    gdf["week_of_year"] = gdf["date"].dt.isocalendar().week.astype(int)
    gdf["is_weekend"]   = gdf["day_of_week"].isin([5, 6]).astype(int)
    gdf["season"]       = gdf["month"].map(_season)
    gdf["is_holiday"]   = gdf["date"].apply(_is_holiday)
    return gdf


# ── Derived weather features ─────────────────────────────────────────────────

def _classify_rain(v) -> str:
    if pd.isna(v) or v < 1:   return "none"
    if v < 10:                 return "light"
    if v < 25:                 return "moderate"
    if v < 50:                 return "heavy"
    return "very_heavy"


def _dry_days(df_site: pd.DataFrame) -> pd.DataFrame:
    df_site = df_site.sort_values("date").copy()
    last_rain, counts = None, []
    for _, row in df_site.iterrows():
        p = row.get("precipitation_mm", 0) or 0
        if p < 1:
            counts.append(999 if last_rain is None else (row["date"] - last_rain).days)
        else:
            last_rain = row["date"]
            counts.append(0)
    df_site["days_since_rain"] = counts
    return df_site


def add_derived_weather(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["rain_24h_category"] = gdf["rain_24h_mm"].apply(_classify_rain)

    parts = []
    for sid in gdf["site_id"].unique():
        mask = gdf["site_id"] == sid
        parts.append(_dry_days(gdf[mask].copy()))
    gdf = pd.concat(parts, ignore_index=True)

    monthly_mean = gdf.groupby(["site_id", "month"])["temp_24h_avg_C"].transform("mean")
    gdf["temp_anomaly_C"]    = gdf["temp_24h_avg_C"] - monthly_mean
    gdf["feels_like_temp_C"] = gdf["temp_24h_avg_C"] - 0.5 * gdf.get("wind_24h_avg_ms", 0)
    return gdf


# ── Spatial context features ─────────────────────────────────────────────────

SYDNEY_CBD = Point(151.2093, -33.8688)


def add_spatial_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    unique  = gdf.groupby("site_id").first().reset_index()

    def _nn(sid):
        pt    = unique.loc[unique["site_id"] == sid, "geometry"].values[0]
        others = unique.loc[unique["site_id"] != sid, "geometry"]
        return others.distance(pt).min() if len(others) else np.nan

    nn_map = {sid: _nn(sid) for sid in unique["site_id"]}
    gdf["nearest_site_dist_deg"] = gdf["site_id"].map(nn_map)
    gdf["dist_from_sydney_deg"]  = gdf.geometry.distance(SYDNEY_CBD)
    return gdf


# ── Lagged features ──────────────────────────────────────────────────────────

LAG_FEATURES = ["rain_24h_mm", "temp_24h_avg_C", "wind_24h_avg_ms", "humidity_24h_avg_pct"]
LAG_DAYS     = [1, 2, 3, 7]


def add_lagged_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.sort_values(["site_id", "date"]).copy()
    for feat in LAG_FEATURES:
        for lag in LAG_DAYS:
            gdf[f"{feat}_lag{lag}d"] = gdf.groupby("site_id")[feat].shift(lag)
    return gdf


# ── Save & document ──────────────────────────────────────────────────────────

def save_dataset(gdf: gpd.GeoDataFrame, buffer_cols: list,
                 csv_path: str, parquet_path: str, gpkg_path: str) -> None:
    df_csv = gdf.copy()
    df_csv["geometry_wkt"] = df_csv.geometry.apply(lambda x: x.wkt)
    for col in buffer_cols:
        if col in df_csv.columns:
            df_csv[f"{col}_wkt"] = df_csv[col].apply(
                lambda x: x.wkt if hasattr(x, "wkt") else None
            )
    drop = ["geometry"] + buffer_cols
    df_csv.drop(columns=[c for c in drop if c in df_csv.columns]).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    try:
        gdf.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path}")
    except Exception as e:
        print(f"  Parquet skipped: {e}")

    try:
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"  Saved: {gpkg_path}")
    except Exception as e:
        print(f"  GeoPackage skipped: {e}")


def make_feature_docs(gdf: gpd.GeoDataFrame,
                      exclude_cols: list) -> pd.DataFrame:
    rows = []
    for col in gdf.columns:
        if col in exclude_cols:
            continue
        dtype   = str(gdf[col].dtype)
        null_pct = gdf[col].isnull().mean() * 100
        if gdf[col].dtype in ["float64", "int64"]:
            rng = f"{gdf[col].min():.2f} – {gdf[col].max():.2f} (mean: {gdf[col].mean():.2f})"
        else:
            rng = f"{gdf[col].nunique()} unique values"
        rows.append(dict(feature=col, dtype=dtype, missing_pct=f"{null_pct:.2f}%", range=rng))
    return pd.DataFrame(rows)


def save_metadata(gdf: gpd.GeoDataFrame, buffer_sizes: dict, path: str) -> None:
    meta = dict(
        creation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_records=len(gdf),
        total_features=len(gdf.columns),
        sites_count=int(gdf["site_id"].nunique()),
        date_range=dict(start=str(gdf["date"].min()), end=str(gdf["date"].max())),
        crs=str(gdf.crs),
        buffer_sizes_meters=buffer_sizes,
    )
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {path}")


def validate(gdf: gpd.GeoDataFrame, df_sites: pd.DataFrame) -> dict:
    return dict(
        all_sites_have_data=gdf["site_id"].nunique() == len(df_sites),
        no_duplicates=len(gdf) == len(gdf.drop_duplicates(["site_id", "date"])),
        geometries_valid=gdf.geometry.is_valid.all(),
        coords_in_nsw=(
            (gdf["lat"] > -38) & (gdf["lat"] < -28) &
            (gdf["lon"] > 150) & (gdf["lon"] < 154)
        ).all(),
    )
