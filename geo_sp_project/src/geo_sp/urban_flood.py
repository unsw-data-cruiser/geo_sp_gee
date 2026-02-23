"""
src/geo_sp/urban_flood.py
Urban flood risk analysis — multiple rainfall scenarios (Phase 2-3).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import folium
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import ee


# ── Drainage pits ────────────────────────────────────────────────────────────

DRAINAGE_PITS_URL = (
    "https://mapservices.randwick.nsw.gov.au/arcgis/rest/services/"
    "extDrainage/DrainageBeachPollutionWebsite/MapServer/0/query"
)


def fetch_drainage_pits(region: ee.Geometry) -> Optional[gpd.GeoDataFrame]:
    bounds = region.bounds().getInfo()
    coords = bounds["coordinates"][0]
    bbox = dict(
        xmin=min(c[0] for c in coords), ymin=min(c[1] for c in coords),
        xmax=max(c[0] for c in coords), ymax=max(c[1] for c in coords),
    )
    params = dict(
        where="1=1", geometryType="esriGeometryEnvelope",
        geometry=f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}",
        inSR="4326", spatialRel="esriSpatialRelIntersects",
        outFields="*", returnGeometry="true", outSR="4326", f="geojson",
    )
    try:
        r = requests.get(DRAINAGE_PITS_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if data.get("features"):
            gdf = gpd.GeoDataFrame.from_features(data["features"])
            gdf.crs = "EPSG:4326"
            return gdf
    except Exception as e:
        print(f"  Drainage pits fetch failed: {e}")
    return None


# ── Terrain ──────────────────────────────────────────────────────────────────

def compute_depression_depth(elevation: ee.Image, region: ee.Geometry) -> ee.Image:
    focal_min = elevation.reduceNeighborhood(
        reducer=ee.Reducer.min(), kernel=ee.Kernel.square(radius=1, units="pixels")
    )
    depth = focal_min.subtract(elevation)
    sig   = depth.gt(0.1)
    stats = depth.updateMask(sig).reduceRegion(
        reducer=ee.Reducer.count().combine(ee.Reducer.mean(), "", True)
                                  .combine(ee.Reducer.max(), "", True),
        geometry=region, scale=30, maxPixels=1e9
    ).getInfo()
    if (stats.get("elevation_count") or 0) == 0:
        mean_elev = elevation.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=1e9
        ).getInfo().get("elevation", 50)
        depth = ee.Image(mean_elev).subtract(elevation).max(0)
        print(f"  Using mean elevation {mean_elev:.1f}m as depression reference")
    return depth


def compute_flow_accumulation(elevation: ee.Image, region: ee.Geometry) -> ee.Image:
    slope   = ee.Terrain.slope(elevation)
    inv_s   = slope.add(0.1).pow(-1)
    flow    = inv_s.reduceNeighborhood(
        reducer=ee.Reducer.sum(), kernel=ee.Kernel.circle(radius=3, units="pixels")
    )
    stats   = flow.reduceRegion(
        reducer=ee.Reducer.minMax(), geometry=region, scale=30, maxPixels=1e9
    ).getInfo()
    mn, mx  = stats.get("slope_min", 0), stats.get("slope_max", 1)
    if mx and mn is not None and mx > mn:
        return flow.unitScale(mn, mx)
    return inv_s.divide(10)


# ── Road network ─────────────────────────────────────────────────────────────

def fetch_roads_osmnx(region: ee.Geometry) -> gpd.GeoDataFrame:
    import osmnx as ox
    try:
        G = ox.graph_from_place("Randwick, New South Wales, Australia", network_type="drive")
    except Exception:
        b = region.bounds().getInfo()["coordinates"][0]
        n, s = max(c[1] for c in b), min(c[1] for c in b)
        e, w = max(c[0] for c in b), min(c[0] for c in b)
        G = ox.graph_from_bbox(n, s, e, w, network_type="drive")

    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf = gdf.to_crs("EPSG:4326").reset_index()
    gdf["road_id"] = gdf.index
    return gdf


# ── Terrain sampling ─────────────────────────────────────────────────────────

def sample_terrain_at_roads(gdf_roads: gpd.GeoDataFrame,
                             elevation: ee.Image,
                             flow_acc: ee.Image,
                             depression: ee.Image,
                             sample_size: int = 300) -> gpd.GeoDataFrame:
    pts, attrs = [], []
    for _, road in gdf_roads.head(sample_size).iterrows():
        for frac in [0.0, 0.5, 1.0]:
            try:
                pt = road.geometry.interpolate(frac, normalized=True)
                pts.append(pt)
                attrs.append(dict(road_id=road["road_id"],
                                  highway=road.get("highway", "unknown"),
                                  name=road.get("name", "unnamed")))
            except Exception:
                continue

    gdf_pts = gpd.GeoDataFrame(attrs, geometry=pts, crs="EPSG:4326")
    terrain  = elevation.addBands(flow_acc).addBands(depression).rename(
        ["elevation", "flow_acc", "depression"]
    )
    features = [ee.Feature(ee.Geometry.Point([p.x, p.y]), {"id": i})
                for i, p in enumerate(gdf_pts.geometry)]
    sampled  = terrain.sampleRegions(
        collection=ee.FeatureCollection(features), scale=30, geometries=False
    )
    try:
        rows = [f["properties"] for f in sampled.getInfo()["features"]]
        df_t = pd.DataFrame(rows).set_index("id")
        gdf_pts = gdf_pts.join(df_t, how="left")
    except Exception as e:
        print(f"  Terrain sampling error: {e} — using defaults")
        gdf_pts["elevation"], gdf_pts["flow_acc"], gdf_pts["depression"] = 50.0, 0.5, 0.0

    for col, default in [("elevation", gdf_pts["elevation"].mean()),
                          ("flow_acc", 0.5), ("depression", 0.0)]:
        gdf_pts[col] = gdf_pts[col].fillna(default)
    return gdf_pts


# ── Drainage distance ────────────────────────────────────────────────────────

def add_drain_distance(gdf_pts: gpd.GeoDataFrame,
                       gdf_drains: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if gdf_drains is not None and len(gdf_pts) > 0:
        tree  = cKDTree(np.array([[p.x, p.y] for p in gdf_drains.geometry]))
        dists, _ = tree.query(np.array([[p.x, p.y] for p in gdf_pts.geometry]), k=1)
        gdf_pts["dist_to_drain_m"] = dists * 111_000
    else:
        gdf_pts["dist_to_drain_m"] = 100.0
    return gdf_pts


# ── Risk scoring ─────────────────────────────────────────────────────────────

def calculate_flood_risk(df: pd.DataFrame, rainfall_mm_hr: float) -> pd.DataFrame:
    if df.empty:
        return df
    r = df.copy()

    def _norm_inv(col):
        mn, mx = r[col].min(), r[col].max()
        return 1 - (r[col] - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=r.index)

    r["elev_risk"]  = _norm_inv("elevation") if "elevation" in r.columns else 0.5
    r["flow_risk"]  = r.get("flow_acc", pd.Series(0.5, index=r.index)).fillna(0.5)
    dep_mx = r["depression"].max() if "depression" in r.columns else 0
    r["dep_risk"]   = (r["depression"].fillna(0) / dep_mx) if dep_mx > 0 else 0.0
    r["drain_risk"] = np.clip(r.get("dist_to_drain_m", 100) / 200, 0, 1)

    mult = min(rainfall_mm_hr / 50, 2.0)
    r["base_risk"] = (
        r["elev_risk"] * 0.20 + r["flow_risk"] * 0.30
        + r["dep_risk"] * 0.20 + r["drain_risk"] * 0.30
    )
    r["flood_risk"] = np.clip(r["base_risk"] * mult, 0, 1)
    r["risk_cat"]   = pd.cut(r["flood_risk"], bins=[0, 0.3, 0.6, 1.0],
                              labels=["Low", "Medium", "High"])
    return r


def run_all_scenarios(gdf_pts: gpd.GeoDataFrame,
                      scenarios: dict) -> dict[str, pd.DataFrame]:
    results = {}
    for name, mm in scenarios.items():
        df = calculate_flood_risk(gdf_pts, mm)
        results[name] = df
        print(f"  {name}: mean={df['flood_risk'].mean():.3f}, "
              f"high%={(df['flood_risk'] > 0.6).mean()*100:.1f}%")
    return results


def aggregate_to_roads(results: dict, gdf_roads: gpd.GeoDataFrame) -> dict:
    agg = {}
    for name, df in results.items():
        road_risk = df.groupby("road_id").agg(
            flood_risk=("flood_risk", "mean"),
            elevation=("elevation", "mean"),
            dist_to_drain_m=("dist_to_drain_m", "mean"),
        ).reset_index()
        merged = gdf_roads.merge(road_risk, on="road_id", how="left")
        merged["flood_risk"] = merged["flood_risk"].fillna(0)
        merged["risk_cat"]   = pd.cut(merged["flood_risk"], bins=[0, 0.3, 0.6, 1.0],
                                       labels=["Low", "Medium", "High"])
        agg[name] = merged
    return agg


def comparison_table(agg: dict, scenarios: dict) -> pd.DataFrame:
    rows = []
    for name, gdf in agg.items():
        rows.append(dict(
            Scenario=name,
            Rainfall_mm_hr=scenarios[name],
            Mean_Risk=gdf["flood_risk"].mean(),
            High_Risk_Pct=(gdf["flood_risk"] > 0.6).mean() * 100,
            Medium_Risk_Pct=((gdf["flood_risk"] >= 0.3) & (gdf["flood_risk"] <= 0.6)).mean() * 100,
            Low_Risk_Pct=(gdf["flood_risk"] < 0.3).mean() * 100,
        ))
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_scenarios(df_cmp: pd.DataFrame, save_path: str = "flood_risk_analysis.png"):
    if df_cmp.empty:
        print("  No data — skipping plots")
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = [s.split("(")[0].strip() for s in df_cmp["Scenario"]]
    xp     = np.arange(len(df_cmp))

    ax = axes[0, 0]
    ax.plot(df_cmp["Rainfall_mm_hr"], df_cmp["Mean_Risk"], "o-", color="steelblue")
    ax.set(xlabel="Rainfall (mm/hr)", ylabel="Mean Flood Risk", title="Risk vs Rainfall")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df_cmp["Rainfall_mm_hr"], df_cmp["High_Risk_Pct"], "s-", color="crimson")
    ax.set(xlabel="Rainfall (mm/hr)", ylabel="High-Risk Roads (%)", title="High-Risk %")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.bar(xp, df_cmp["Low_Risk_Pct"],    label="Low",    color="green",  alpha=0.7)
    ax.bar(xp, df_cmp["Medium_Risk_Pct"], bottom=df_cmp["Low_Risk_Pct"],
           label="Medium", color="orange", alpha=0.7)
    ax.bar(xp, df_cmp["High_Risk_Pct"],
           bottom=df_cmp["Low_Risk_Pct"] + df_cmp["Medium_Risk_Pct"],
           label="High", color="red", alpha=0.7)
    ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set(ylabel="Roads (%)", title="Risk Distribution"); ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    base = df_cmp.iloc[0]["Mean_Risk"] or 1e-9
    ax.bar(xp, df_cmp["Mean_Risk"] / base, color="darkblue", alpha=0.7)
    ax.axhline(1, color="red", ls="--")
    ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set(ylabel="Risk Multiplier", title="Relative to Light Rain"); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ── Folium map ───────────────────────────────────────────────────────────────

def build_flood_map(roads_gdf: gpd.GeoDataFrame,
                    pits_gdf: Optional[gpd.GeoDataFrame],
                    scenario_name: str, rainfall: int,
                    center: list) -> folium.Map:
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    def _color(risk):
        if pd.isna(risk): return "gray"
        return "green" if risk < 0.3 else ("orange" if risk < 0.6 else "red")

    for _, road in roads_gdf.head(200).iterrows():
        if pd.notna(road.get("flood_risk")):
            try:
                folium.PolyLine(
                    [(c[1], c[0]) for c in road.geometry.coords],
                    color=_color(road["flood_risk"]), weight=3, opacity=0.7,
                    popup=f"Risk: {road['flood_risk']:.3f} | {road.get('name','?')}",
                ).add_to(m)
            except Exception:
                continue

    if pits_gdf is not None:
        for _, pit in pits_gdf.head(100).iterrows():
            folium.CircleMarker(
                [pit.geometry.y, pit.geometry.x], radius=2,
                color="blue", fill=True, fillOpacity=0.6,
            ).add_to(m)

    legend = f"""
    <div style="position:fixed;top:10px;right:10px;width:200px;background:white;
                border:2px solid grey;z-index:9999;padding:10px;font-size:12px">
    <b>{scenario_name}</b><br><b>{rainfall} mm/hr</b><hr>
    <i style="background:green;width:20px;height:10px;display:inline-block"></i> Low<br>
    <i style="background:orange;width:20px;height:10px;display:inline-block"></i> Medium<br>
    <i style="background:red;width:20px;height:10px;display:inline-block"></i> High<br>
    <i style="background:blue;border-radius:50%;width:10px;height:10px;display:inline-block"></i> Drain Pits
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m
