"""
src/geo_sp/boundaries.py
Fetch LGA / suburb boundaries from NSW ArcGIS services.
"""
from __future__ import annotations
from typing import Optional
import requests
import ee


def _fetch_geometry(layer_url: str, where: str, label: str) -> Optional[ee.Geometry]:
    """Generic ArcGIS FeatureServer → EE Geometry fetcher."""
    url = layer_url.rstrip("/") + "/query"
    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }
    try:
        print(f"Fetching {label}... ", end="", flush=True)
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        gj = r.json()

        if not gj.get("features"):
            print("not found")
            return None

        geom = gj["features"][0]["geometry"]
        t = geom["type"]
        if t == "Polygon":
            result = ee.Geometry.Polygon(geom["coordinates"])
        elif t == "MultiPolygon":
            result = ee.Geometry.MultiPolygon(geom["coordinates"])
        else:
            print(f"unsupported type: {t}")
            return None

        print("✓")
        return result
    except Exception as e:
        print(f"ERROR: {str(e)[:60]}")
        return None


def get_lga(name: str, lga_layer: str) -> Optional[ee.Geometry]:
    """Return EE geometry for an LGA by name."""
    return _fetch_geometry(
        lga_layer,
        f"UPPER(lganame) = UPPER('{name}')",
        f"LGA/{name}",
    )


def get_suburb(name: str, suburb_layer: str) -> Optional[ee.Geometry]:
    """Return EE geometry for a suburb by name."""
    return _fetch_geometry(
        suburb_layer,
        f"UPPER(suburbname) = UPPER('{name}')",
        f"Suburb/{name}",
    )


def get_lga_info(geom: ee.Geometry) -> dict:
    """Return area (km²) and centroid for a geometry."""
    area_km2 = geom.area().divide(1e6).getInfo()
    centroid  = geom.centroid().coordinates().getInfo()
    return {"area_km2": area_km2, "centroid": centroid}
