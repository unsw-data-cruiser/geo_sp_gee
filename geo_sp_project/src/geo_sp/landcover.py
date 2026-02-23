"""
src/geo_sp/landcover.py
ESA WorldCover land-cover analysis over a region.
"""
from __future__ import annotations
import ee
import geemap


def load_worldcover() -> ee.Image:
    """Return ESA WorldCover v200 classification image."""
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")


def landcover_stats(wc: ee.Image, region: ee.Geometry, scale: int = 50) -> dict:
    """Return km² and % for built / water / tree / grass."""
    area_km2 = region.area().divide(1e6).getInfo()
    classes  = {"built": 50, "water": 80, "tree": 10, "grass": 30}
    results  = {"total_area_km2": area_km2}

    for name, code in classes.items():
        mask = wc.eq(code)
        raw  = mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=scale, maxPixels=1e9
        ).getInfo()
        km2  = raw.get("Map", 0) / 1e6
        results[f"{name}_km2"]  = km2
        results[f"{name}_pct"]  = (km2 / area_km2) * 100
    return results


def build_landcover_map(wc: ee.Image, region: ee.Geometry,
                        suburbs: dict, center: list, zoom: int) -> geemap.Map:
    """Return a geemap Map with land-cover layers."""
    m = geemap.Map(center=center, zoom=zoom)
    m.add_basemap("SATELLITE")

    fc = ee.FeatureCollection([ee.Feature(region)])
    m.addLayer(fc, {"color": "white", "width": 3}, "LGA Boundary")

    for name, geom in suburbs.items():
        sub_fc = ee.FeatureCollection([ee.Feature(geom)])
        m.addLayer(sub_fc, {"color": "yellow", "width": 1}, f"Suburb: {name}", False)

    m.addLayer(wc.eq(50).clip(region).selfMask(), {"palette": ["red"],       "opacity": 0.6}, "Built-up")
    m.addLayer(wc.eq(80).clip(region).selfMask(), {"palette": ["blue"],      "opacity": 0.6}, "Water")
    m.addLayer(wc.eq(10).clip(region).selfMask(), {"palette": ["darkgreen"], "opacity": 0.6}, "Trees",  False)
    m.addLayer(wc.eq(30).clip(region).selfMask(), {"palette": ["lightgreen"],"opacity": 0.6}, "Grass",  False)

    lc_viz = dict(min=10, max=95,
                  palette=["006400","FFBB22","FFFF4C","F096FF","FA0000",
                           "B4B4B4","F0F0F0","0064C8","0096A0","00CF75"])
    m.addLayer(wc.clip(region), lc_viz, "Full Land Cover", False)
    return m
