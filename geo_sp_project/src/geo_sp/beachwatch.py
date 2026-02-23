"""
src/geo_sp/beachwatch.py
Beachwatch site registry builder + geographic feature extraction (Phase 1).
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
import pandas as pd
import geopandas as gpd
import requests
import geemap
import ee
from shapely.geometry import Point


# ── Fetch sites ──────────────────────────────────────────────────────────────

def fetch_sites(api_url: str) -> tuple[pd.DataFrame, dict]:
    """Return (df_registry, raw_geojson) from Beachwatch public API."""
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    gj = r.json()

    rows = []
    for feat in gj["features"]:
        props  = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        rows.append(dict(
            site_id=props["id"],
            site_name=props["siteName"],
            lon=coords[0], lat=coords[1],
            region=props.get("region", "Unknown"),
            pollution_forecast=props.get("pollutionForecast"),
            latest_result=props.get("latestResult"),
            latest_rating=props.get("latestResultRating"),
            snapshot_date=datetime.now().strftime("%Y-%m-%d"),
        ))
    return pd.DataFrame(rows), gj


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geom = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def build_buffers(gdf: gpd.GeoDataFrame, radius_m: int = 500) -> gpd.GeoDataFrame:
    """Return GeoDataFrame with buffer polygon geometry."""
    proj  = gdf.to_crs("EPSG:28356")
    proj["buffer_geom"] = proj.geometry.buffer(radius_m)
    out   = proj.set_geometry("buffer_geom").to_crs("EPSG:4326")
    return out


def save_registry(df: pd.DataFrame, csv_path: str,
                  parquet_path: Optional[str] = None) -> None:
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    if parquet_path:
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"  Saved: {parquet_path}")
        except Exception as e:
            print(f"  Parquet skipped: {e}")


def save_buffer_csv(gdf_buffers: gpd.GeoDataFrame, path: str) -> None:
    df = pd.DataFrame(dict(
        unit_id="bwsite_" + gdf_buffers["site_id"].astype(str),
        site_id=gdf_buffers["site_id"],
        site_name=gdf_buffers["site_name"],
        buffer_geom_wkt=gdf_buffers.geometry.apply(lambda x: x.wkt),
    ))
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ── GEE feature extraction ───────────────────────────────────────────────────

def load_gee_layers() -> dict:
    """Return dict of GEE images for geographic feature extraction."""
    dem   = ee.Image("AU/GA/DEM_1SEC/v10/DEM-H").select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    layers = dict(dem=dem, slope=slope, has_imperv=False, has_ndvi=False)

    try:
        imperv = (
            ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
            .filterDate("2020-01-01", "2020-12-31").mosaic().select("built_surface")
        )
        layers.update(imperv=imperv, has_imperv=True)
        print("  ✓ Impervious surface")
    except Exception as e:
        print(f"  ⚠ Impervious surface unavailable: {e}")

    try:
        ndvi = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate("2023-01-01", "2023-12-31")
            .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
            .mean()
        )
        layers.update(ndvi=ndvi, has_ndvi=True)
        print("  ✓ NDVI")
    except Exception as e:
        print(f"  ⚠ NDVI unavailable: {e}")

    return layers


def make_feature_extractor(layers: dict):
    """Return a GEE map function that adds geo features to each site point."""
    dem, slope = layers["dem"], layers["slope"]
    has_imperv = layers.get("has_imperv", False)
    has_ndvi   = layers.get("has_ndvi", False)

    def _extract(feature):
        pt   = feature.geometry()
        b500 = pt.buffer(500)
        b1k  = pt.buffer(1000)
        elev_d  = dem.reduceRegion(ee.Reducer.mean(), pt,    30, maxPixels=1e9)
        slope_d = slope.reduceRegion(ee.Reducer.mean(), b500, 30, maxPixels=1e9)
        result  = {"elevation_m": elev_d.get("elevation"),
                   "slope_500m_deg": slope_d.get("slope")}
        if has_imperv:
            imp_d = layers["imperv"].reduceRegion(ee.Reducer.mean(), b500, 100, maxPixels=1e9)
            result["imperv_500m_pct"] = imp_d.get("built_surface")
        if has_ndvi:
            ndvi_d = layers["ndvi"].reduceRegion(ee.Reducer.mean(), b1k, 10, maxPixels=1e9)
            result["ndvi_1km"] = ndvi_d.get("NDVI")
        return feature.set(result)

    return _extract


def export_registry_to_drive(fc: ee.FeatureCollection,
                              layers: dict,
                              description: str = "beachwatch_site_registry_with_geo") -> None:
    extractor = make_feature_extractor(layers)
    fc_out    = fc.map(extractor)
    fields    = ["id", "siteName", "region", "pollutionForecast",
                 "latestResult", "latestResultRating", "elevation_m", "slope_500m_deg"]
    if layers.get("has_imperv"):
        fields.append("imperv_500m_pct")
    if layers.get("has_ndvi"):
        fields.append("ndvi_1km")

    task = ee.batch.Export.table.toDrive(
        collection=fc_out, description=description,
        fileFormat="CSV", selectors=fields,
    )
    task.start()
    print(f"  Export task started: {description}")
    print("  → https://code.earthengine.google.com/tasks")


def qa_check(df: pd.DataFrame) -> dict:
    return {
        "site_id_unique":   df["site_id"].is_unique,
        "no_null_coords":   df[["lon", "lat"]].notna().all().all(),
        "coords_in_nsw":    (
            (df["lat"] > -38) & (df["lat"] < -28) &
            (df["lon"] > 150) & (df["lon"] < 154)
        ).all(),
        "total_sites": len(df),
    }


def build_site_map(fc: ee.FeatureCollection, dem: ee.Image,
                   center: list, zoom: int) -> geemap.Map:
    m = geemap.Map(center=center, zoom=zoom)
    m.addLayer(fc, {"color": "red"}, "Beachwatch Sites")
    m.addLayer(dem, dict(min=0, max=100, palette=["blue","cyan","green","yellow","red"]),
               "Elevation", False)
    return m
