"""
src/geo_sp/coastal_inundation.py
Multi-scenario coastal flood inundation modelling (Phase 2-1).
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import requests
import folium
import ee


# ── Inundation modelling ────────────────────────────────────────────────────

def compute_inundation(elevation: ee.Image, region: ee.Geometry,
                       scenarios: dict, lga_area_km2: float) -> dict:
    """
    For every scenario compute ocean-connected inundation mask + statistics.
    Returns dict keyed by scenario name.
    """
    expanded = region.buffer(5000)
    results  = {}

    for name, params in scenarios.items():
        wl = params["water_level"]
        print(f"  Processing {name} ({wl}m)... ", end="", flush=True)

        low_ext   = elevation.clip(expanded).lt(wl)
        ocean_pix = elevation.clip(expanded).lt(0)
        cost      = low_ext.Not().multiply(1e10)
        cost_dist = cost.cumulativeCost(
            source=ocean_pix.selfMask(), maxDistance=50000, geodeticDistance=False
        )
        connected      = cost_dist.lt(10000)
        inundation_mask = low_ext.And(connected).clip(region)

        raw = inundation_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=30, maxPixels=1e9
        ).getInfo()
        area_km2 = (raw.get("elevation", 0) or 0) / 1e6
        pct      = (area_km2 / lga_area_km2) * 100 if area_km2 > 0 else 0

        print(f"{area_km2:.3f} km² ({pct:.2f}%)")
        results[name] = dict(
            water_level=wl,
            inundation_mask=inundation_mask,
            area_km2=area_km2,
            area_pct=pct,
        )
    return results


# ── OSM asset queries ────────────────────────────────────────────────────────

def query_osm_roads(bbox: List[float]) -> pd.DataFrame:
    """Fetch road network from Overpass API."""
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"]
      ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out geom;
    """
    try:
        r = requests.post("http://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=120)
        r.raise_for_status()
        rows = []
        for el in r.json().get("elements", []):
            if el["type"] == "way" and "geometry" in el:
                rows.append(dict(
                    osm_id=el["id"],
                    highway_type=el["tags"].get("highway", "unknown"),
                    name=el["tags"].get("name", "Unnamed"),
                    geometry=[(n["lon"], n["lat"]) for n in el["geometry"]],
                ))
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  OSM roads failed: {e}")
        return pd.DataFrame()


def query_osm_buildings(bbox: List[float]) -> pd.DataFrame:
    """Fetch building centroids from Overpass API."""
    query = f"""
    [out:json][timeout:60];
    (
      way["building"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
      relation["building"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out center;
    """
    try:
        r = requests.post("http://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=120)
        r.raise_for_status()
        rows = []
        for el in r.json().get("elements", []):
            if "center" in el:
                rows.append(dict(
                    osm_id=el["id"],
                    building_type=el["tags"].get("building", "yes"),
                    name=el["tags"].get("name", "Unnamed"),
                    lon=el["center"]["lon"],
                    lat=el["center"]["lat"],
                ))
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  OSM buildings failed: {e}")
        return pd.DataFrame()


# ── Impact analysis ──────────────────────────────────────────────────────────

def asset_impact(inundation_results: dict, df_buildings: pd.DataFrame,
                 df_roads: pd.DataFrame, scenarios: dict) -> pd.DataFrame:
    """Count affected buildings & road segments for each scenario."""
    rows = []
    for name, res in inundation_results.items():
        mask = res["inundation_mask"]
        affected_b = 0
        if not df_buildings.empty:
            pts = [ee.Feature(ee.Geometry.Point([r["lon"], r["lat"]]))
                   for _, r in df_buildings.iterrows()]
            sampled = mask.reduceRegions(
                collection=ee.FeatureCollection(pts),
                reducer=ee.Reducer.first(), scale=30
            )
            affected_b = sampled.filter(ee.Filter.eq("first", 1)).size().getInfo()

        affected_r = 0
        if not df_roads.empty:
            pts = []
            for _, road in df_roads.iterrows():
                coords = road["geometry"]
                if coords:
                    mid = coords[len(coords) // 2]
                    pts.append(ee.Feature(ee.Geometry.Point([mid[0], mid[1]])))
            if pts:
                sampled = mask.reduceRegions(
                    collection=ee.FeatureCollection(pts),
                    reducer=ee.Reducer.first(), scale=30
                )
                affected_r = sampled.filter(ee.Filter.eq("first", 1)).size().getInfo()

        rows.append(dict(
            Scenario=name,
            Water_Level_m=res["water_level"],
            Return_Period=scenarios[name]["return_period"],
            Inundated_Area_km2=res["area_km2"],
            Inundated_Area_pct=res["area_pct"],
            Affected_Buildings=affected_b,
            Affected_Road_Segments=affected_r,
        ))
    return pd.DataFrame(rows)


# ── Folium visualisation ─────────────────────────────────────────────────────

SCENARIO_COLORS = {
    "Low": "#3182bd", "Medium": "#e6550d",
    "High": "#fd8d3c", "Extreme": "#d62728",
}


def build_inundation_map(inundation_results: dict, randwick: ee.Geometry,
                         df_buildings: pd.DataFrame, center: list) -> folium.Map:
    """Return a Folium map with inundation scenario layers."""
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # LGA boundary
    try:
        folium.GeoJson(
            randwick.getInfo(),
            name="LGA Boundary",
            style_function=lambda x: {"fillColor": "transparent", "color": "white", "weight": 3},
        ).add_to(m)
    except Exception as e:
        print(f"  Warning: boundary layer failed: {e}")

    # Inundation layers
    for name, res in inundation_results.items():
        try:
            map_id = res["inundation_mask"].selfMask().getMapId(
                {"palette": [SCENARIO_COLORS[name]], "opacity": 0.6}
            )
            folium.TileLayer(
                tiles=map_id["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name=f"Inundation {name} ({res['water_level']}m)",
                overlay=True, control=True,
                show=(name == "Medium"),
            ).add_to(m)
        except Exception as e:
            print(f"  Warning: {name} layer failed: {e}")

    # Buildings sample
    if not df_buildings.empty:
        grp = folium.FeatureGroup(name="Buildings (sample)", show=False)
        for _, b in df_buildings.sample(min(100, len(df_buildings))).iterrows():
            folium.CircleMarker(
                location=[b["lat"], b["lon"]], radius=3,
                color="yellow", fill=True, fillOpacity=0.6,
                popup=f"{b['building_type']}: {b['name']}",
            ).add_to(grp)
        grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── GEE export ───────────────────────────────────────────────────────────────

def export_inundation_to_drive(inundation_results: dict,
                                region: ee.Geometry,
                                drive_folder: str = "Randwick_Coastal_Inundation") -> None:
    """Start GEE export tasks for each scenario GeoTIFF."""
    for name, res in inundation_results.items():
        img = res["inundation_mask"].unmask(0).toByte()
        wl  = res["water_level"]
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"Inundation_{name}_{wl}m",
            folder=drive_folder,
            fileNamePrefix=f"inundation_{name}_{wl}m",
            region=region, scale=30, crs="EPSG:4326", maxPixels=1e9,
        )
        task.start()
        print(f"  Export started: {name} ({wl}m)")
    print(f"\n  → Check status: https://code.earthengine.google.com/tasks")
