"""
src/geo_sp/terrain.py
DEM / terrain analysis using SRTM + EE Terrain helpers.
"""
from __future__ import annotations
import ee
import geemap


def load_terrain_layers() -> dict:
    """Return dict of EE Images: elevation, slope, aspect, hillshade."""
    srtm      = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select("elevation")
    slope     = ee.Terrain.slope(elevation)
    aspect    = ee.Terrain.aspect(elevation)
    hillshade = ee.Terrain.hillshade(elevation)
    return dict(elevation=elevation, slope=slope, aspect=aspect, hillshade=hillshade)


def elevation_stats(elevation: ee.Image, region: ee.Geometry, scale: int = 30) -> dict:
    """Compute min/max/mean/stdDev elevation over region."""
    reducer = (
        ee.Reducer.min()
        .combine(ee.Reducer.max(),    "", True)
        .combine(ee.Reducer.mean(),   "", True)
        .combine(ee.Reducer.stdDev(), "", True)
    )
    return elevation.reduceRegion(
        reducer=reducer, geometry=region, scale=scale, maxPixels=1e9
    ).getInfo()


def slope_stats(slope: ee.Image, region: ee.Geometry, scale: int = 30) -> dict:
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.max(), "", True)
        .combine(ee.Reducer.percentile([25, 50, 75]), "", True)
    )
    return slope.reduceRegion(
        reducer=reducer, geometry=region, scale=scale, maxPixels=1e9
    ).getInfo()


def slope_class_pcts(slope: ee.Image, region: ee.Geometry, scale: int = 30) -> dict:
    """Return percentage of flat/gentle/moderate/steep terrain."""
    def _area_pct(mask):
        area = mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=scale, maxPixels=1e9
        ).getInfo()
        total = region.area().getInfo()
        return (area.get("slope", 0) / total) * 100

    return {
        "flat_lte5":       _area_pct(slope.lte(5)),
        "gentle_5_15":     _area_pct(slope.gt(5).And(slope.lte(15))),
        "moderate_15_25":  _area_pct(slope.gt(15).And(slope.lte(25))),
        "steep_gt25":      _area_pct(slope.gt(25)),
    }


def build_terrain_map(layers: dict, region: ee.Geometry,
                      center: list, zoom: int) -> geemap.Map:
    """Return a geemap Map with terrain layers added."""
    m = geemap.Map(center=center, zoom=zoom)
    m.add_basemap("SATELLITE")

    elev_viz = dict(min=0, max=150,
                    palette=["0000FF","00FFFF","00FF00","FFFF00","FFA500","FF0000","8B0000"])
    slope_viz = dict(min=0, max=30, palette=["white","yellow","orange","red","darkred"])
    hillshade_viz = dict(min=0, max=255, palette=["000000","FFFFFF"])
    aspect_viz  = dict(min=0, max=360,
                       palette=["red","yellow","green","cyan","blue","magenta","orange","pink","red"])

    fc = ee.FeatureCollection([ee.Feature(region)])
    m.addLayer(fc, {"color": "white", "width": 3}, "LGA Boundary")
    m.addLayer(layers["elevation"].clip(region), elev_viz, "Elevation (colour)")
    m.addLayer(layers["slope"].clip(region),     slope_viz, "Slope", False)
    m.addLayer(layers["hillshade"].clip(region), hillshade_viz, "Hillshade", False)
    m.addLayer(layers["aspect"].clip(region),    aspect_viz, "Aspect", False)

    elev_shaded = layers["elevation"].blend(layers["hillshade"].multiply(0.3))
    m.addLayer(elev_shaded.clip(region), elev_viz, "Elevation+Hillshade", False)
    m.addLayer(layers["slope"].gt(20).selfMask().clip(region),
               {"palette": ["red"]}, "Steep Slope Warning (>20°)", False)
    return m
