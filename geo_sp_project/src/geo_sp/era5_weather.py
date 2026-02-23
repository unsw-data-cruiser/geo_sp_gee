"""
src/geo_sp/era5_weather.py
ERA5-Land near-realtime weather extraction via Google Earth Engine.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd
import ee
import geemap


def get_era5_collection(region: ee.Geometry, days_back: int = 7) -> ee.ImageCollection:
    """Return ERA5-Land hourly collection clipped to region+5km for last N days."""
    end   = datetime.now()
    start = end - timedelta(days=days_back)
    buffered = region.buffer(5000)
    return (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .filterBounds(buffered)
    )


def add_derived_bands(image: ee.Image) -> ee.Image:
    """Add derived meteorological variables to each ERA5 image."""
    temp_c     = image.select("temperature_2m").subtract(273.15).rename("temp_celsius")
    dew_c      = image.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_celsius")
    u          = image.select("u_component_of_wind_10m")
    v          = image.select("v_component_of_wind_10m")
    wind_spd   = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed_10m")
    wind_dir   = u.atan2(v).multiply(180).divide(3.14159).add(180).rename("wind_direction")
    precip_mm  = image.select("total_precipitation").multiply(1000).rename("precipitation_mm")
    e_s = temp_c.multiply(17.27).divide(temp_c.add(237.3)).exp().multiply(6.112)
    e   = dew_c.multiply(17.27).divide(dew_c.add(237.3)).exp().multiply(6.112)
    rh  = e.divide(e_s).multiply(100).rename("relative_humidity")
    pres = image.select("surface_pressure").divide(100).rename("pressure_hpa")
    solar = image.select("surface_solar_radiation_downwards").divide(3600).rename("solar_radiation_wm2")
    return image.addBands([temp_c, dew_c, wind_spd, wind_dir, precip_mm, rh, pres, solar])


def extract_timeseries(collection: ee.ImageCollection,
                       region: ee.Geometry, scale: int = 5000) -> pd.DataFrame:
    """Extract mean hourly values over region; return DataFrame."""
    def _reduce(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region,
            scale=scale, maxPixels=1e9, bestEffort=True
        )
        return ee.Feature(None, stats).set({
            "system:time_start": img.get("system:time_start"),
            "datetime": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd HH:mm:ss"),
        })

    print("Extracting ERA5 time series... ", end="", flush=True)
    features = collection.map(_reduce).getInfo()["features"]
    print("✓")

    rows = [f["properties"] for f in features]
    df = pd.DataFrame(rows)
    if "system:time_start" in df.columns:
        df["timestamp"] = pd.to_datetime(df["system:time_start"], unit="ms")
    return df


def weather_summary(df: pd.DataFrame) -> None:
    """Print a quick statistical summary of the extracted weather DataFrame."""
    cols = {
        "Temperature (°C)":    "temp_celsius",
        "Precipitation (mm)":  "precipitation_mm",
        "Wind speed (m/s)":    "wind_speed_10m",
        "Humidity (%)":        "relative_humidity",
        "Pressure (hPa)":      "pressure_hpa",
        "Solar rad (W/m²)":    "solar_radiation_wm2",
    }
    print("\n" + "="*60)
    print("ERA5 Weather Summary")
    print("="*60)
    for label, col in cols.items():
        if col in df.columns:
            s = df[col]
            print(f"\n{label}:")
            print(f"  min={s.min():.1f}  max={s.max():.1f}  mean={s.mean():.1f}")


def build_weather_map(collection: ee.ImageCollection,
                      region: ee.Geometry, center: list, zoom: int) -> geemap.Map:
    """Return geemap Map showing latest ERA5 temperature layer."""
    m = geemap.Map(center=center, zoom=zoom)
    m.add_basemap("SATELLITE")
    fc = ee.FeatureCollection([ee.Feature(region)])
    m.addLayer(fc, {"color": "yellow", "width": 3}, "Boundary")

    latest = collection.sort("system:time_start", False).first()
    m.addLayer(latest.select("temp_celsius"),
               dict(min=10, max=35, palette=["blue","cyan","green","yellow","orange","red"]),
               "Temperature (°C)")
    m.addLayer(latest.select("wind_speed_10m"),
               dict(min=0, max=15, palette=["white","lightblue","blue","darkblue","purple"]),
               "Wind Speed (m/s)", False)
    m.addLayer(latest.select("precipitation_mm"),
               dict(min=0, max=5,  palette=["white","lightblue","blue","darkblue","navy"]),
               "Precipitation (mm)", False)
    return m
